from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, StaticCache
from transformers.models.llama import modeling_llama as llama_mod

from .fake_quant import (
    fake_quant_k_cache_kivi_channel,
    fake_quant_kv,
    grouped_hadamard_transform_last_dim,
    hadamard_transform_last_dim,
)


@dataclass
class RotateTileKVConfig:
    enable_hadamard: bool = False
    hadamard_mode: str = "none"
    hadamard_group_size: Optional[int] = None
    k_bits: int = 4
    v_bits: int = 4
    quant_impl: str = "default"
    k_quant_scheme: str = "default"
    v_quant_scheme: str = "default"
    group_size: int = 128
    quant_granularity: str = "per-token-tile"
    tile_size: Optional[int] = None
    residual_length: int = 0

    def to_dict(self):
        return asdict(self)


def _maybe_apply_hadamard(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    quant_cfg: RotateTileKVConfig,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tile_size = quant_cfg.tile_size if quant_cfg.tile_size is not None else head_dim // 2
    hadamard_mode = getattr(quant_cfg, "hadamard_mode", "none").lower().replace("_", "-")
    if quant_cfg.enable_hadamard and hadamard_mode == "none":
        hadamard_mode = "full"

    if hadamard_mode == "full":
        query_states = hadamard_transform_last_dim(query_states)
        key_states = hadamard_transform_last_dim(key_states)
    elif hadamard_mode == "tile":
        group_size = quant_cfg.hadamard_group_size if quant_cfg.hadamard_group_size is not None else tile_size
        query_states = grouped_hadamard_transform_last_dim(query_states, group_size)
        key_states = grouped_hadamard_transform_last_dim(key_states, group_size)
    elif hadamard_mode != "none":
        raise ValueError(f"Unsupported hadamard mode: {quant_cfg.hadamard_mode}")

    return query_states, key_states


def _quantize_residual_prefix_inplace(
    tensor: torch.Tensor,
    bits: int,
    granularity: str,
    tile_size: int,
    q_len: int,
    residual_length: int,
):
    if bits is None or bits >= 16:
        return tensor

    total_len = tensor.shape[2]
    if residual_length is None or residual_length <= 0:
        start_idx, end_idx = 0, total_len
    else:
        old_len = max(total_len - q_len, 0)
        start_idx = max(old_len - residual_length, 0)
        end_idx = max(total_len - residual_length, 0)

    if end_idx > start_idx:
        quantized = fake_quant_kv(tensor[:, :, start_idx:end_idx, :], bits, granularity, tile_size)
        tensor[:, :, start_idx:end_idx, :] = quantized
    return tensor


def _apply_residual_window_quantization(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    quant_cfg: RotateTileKVConfig,
    head_dim: int,
    q_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tile_size = quant_cfg.tile_size if quant_cfg.tile_size is not None else head_dim // 2
    residual_length = getattr(quant_cfg, "residual_length", 0)
    k_scheme = getattr(quant_cfg, "k_quant_scheme", "default").lower().replace("_", "-")
    if k_scheme in {"default", "inherit"}:
        k_scheme = quant_cfg.quant_granularity

    v_scheme = getattr(quant_cfg, "v_quant_scheme", "default").lower().replace("_", "-")
    if v_scheme in {"default", "inherit"}:
        v_scheme = quant_cfg.quant_granularity

    total_len = key_states.shape[2]
    if residual_length is None or residual_length <= 0:
        start_idx, end_idx = 0, total_len
    else:
        old_len = max(total_len - q_len, 0)
        start_idx = max(old_len - residual_length, 0)
        end_idx = max(total_len - residual_length, 0)

    if end_idx > start_idx and quant_cfg.k_bits is not None and quant_cfg.k_bits < 16:
        k_slice = key_states[:, :, start_idx:end_idx, :]
        if k_scheme == "kivi-channel":
            key_states[:, :, start_idx:end_idx, :] = fake_quant_k_cache_kivi_channel(
                k_slice,
                quant_cfg.k_bits,
                quant_cfg.group_size,
            )
        else:
            key_states[:, :, start_idx:end_idx, :] = fake_quant_kv(
                k_slice,
                quant_cfg.k_bits,
                k_scheme,
                tile_size,
                quant_impl=quant_cfg.quant_impl,
            )

    if end_idx > start_idx and quant_cfg.v_bits is not None and quant_cfg.v_bits < 16:
        value_states[:, :, start_idx:end_idx, :] = fake_quant_kv(
            value_states[:, :, start_idx:end_idx, :],
            quant_cfg.v_bits,
            v_scheme,
            tile_size,
            quant_impl=quant_cfg.quant_impl,
        )
    return key_states, value_states


class LlamaAttentionRotateTileKV(llama_mod.LlamaAttention):
    def __init__(self, config, layer_idx=None, quant_cfg: Optional[RotateTileKVConfig] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.quant_cfg = quant_cfg or RotateTileKVConfig()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)
            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = llama_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states, key_states = _maybe_apply_hadamard(query_states, key_states, self.quant_cfg, self.head_dim)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states, value_states = _apply_residual_window_quantization(
            key_states,
            value_states,
            self.quant_cfg,
            self.head_dim,
            q_len,
        )

        key_states = llama_mod.repeat_kv(key_states, self.num_key_value_groups)
        value_states = llama_mod.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim**0.5)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum(F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp))
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2RotateTileKV(llama_mod.LlamaFlashAttention2):
    def __init__(self, config, layer_idx=None, quant_cfg: Optional[RotateTileKVConfig] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.quant_cfg = quant_cfg or RotateTileKVConfig()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2`."
            )

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = llama_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states, key_states = _maybe_apply_hadamard(query_states, key_states, self.quant_cfg, self.head_dim)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states, value_states = _apply_residual_window_quantization(
            key_states,
            value_states,
            self.quant_cfg,
            self.head_dim,
            q_len,
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = llama_mod._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value


def _clone_rotatetilekv_attention(original_attn, quant_cfg: RotateTileKVConfig):
    if isinstance(original_attn, llama_mod.LlamaFlashAttention2):
        new_attn = LlamaFlashAttention2RotateTileKV(
            config=original_attn.config,
            layer_idx=original_attn.layer_idx,
            quant_cfg=quant_cfg,
        )
    elif isinstance(original_attn, llama_mod.LlamaAttention):
        new_attn = LlamaAttentionRotateTileKV(
            config=original_attn.config,
            layer_idx=original_attn.layer_idx,
            quant_cfg=quant_cfg,
        )
    else:
        raise TypeError(
            f"Unsupported attention class {type(original_attn).__name__}. "
            "Please load the model with attn_implementation='flash_attention_2' or 'eager'."
        )

    new_attn.load_state_dict(original_attn.state_dict(), strict=True)
    param = next(original_attn.parameters())
    new_attn.to(device=param.device, dtype=param.dtype)
    new_attn.train(original_attn.training)
    return new_attn


def apply_rotatetilekv_to_llama(model, quant_cfg: RotateTileKVConfig):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a HuggingFace LlamaForCausalLM-like model with model.layers")

    for layer in model.model.layers:
        layer.self_attn = _clone_rotatetilekv_attention(layer.self_attn, quant_cfg)

    model.config.rotatetilekv = quant_cfg.to_dict()
    return model


def set_rotatetilekv_config(model, quant_cfg: RotateTileKVConfig):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a HuggingFace LlamaForCausalLM-like model with model.layers")

    for layer in model.model.layers:
        if not hasattr(layer.self_attn, "quant_cfg"):
            raise TypeError(
                "Model is not patched with RotateTileKV attention yet. "
                "Call apply_rotatetilekv_to_llama() first."
            )
        layer.self_attn.quant_cfg = quant_cfg

    model.config.rotatetilekv = quant_cfg.to_dict()
    return model


def load_rotatetilekv_llama(
    model_name_or_path: str,
    quant_cfg: RotateTileKVConfig,
    torch_dtype: torch.dtype = torch.float16,
    device_map="auto",
    attn_implementation: str = "flash_attention_2",
    local_files_only: bool = True,
    **kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
        **kwargs,
    )
    model = apply_rotatetilekv_to_llama(model, quant_cfg)
    model.eval()
    return tokenizer, model
