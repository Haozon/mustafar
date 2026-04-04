from __future__ import annotations

import math
import sys
import types
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from flash_attn import flash_attn_func
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSPARSE_ROOT = PROJECT_ROOT / "DiffSparseKV"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DIFFSPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSPARSE_ROOT))

from diffsparsekv import LlamaForCausalLMDiffSparseKV, create_diff_sparse_kv_config  # noqa: E402
from diffsparsekv.llama_integration import LlamaDiffSparseKVAttention  # noqa: E402
from RotateTileKV.modeling_llama_rotatetilekv import (  # noqa: E402
    RotateTileKVConfig,
    _maybe_apply_hadamard,
)
from RotateTileKV.fake_quant import (  # noqa: E402
    fake_quant_k_cache_kivi_channel,
    fake_quant_kv,
)


def create_jsqkv_quant_config(
    *,
    k_bits: int,
    v_bits: int,
    quant_impl: str = "default",
    k_quant_scheme: str = "per-token-tile",
    v_quant_scheme: str = "per-token-tile",
    group_size: int = 128,
    quant_granularity: str = "per-token-tile",
    tile_size: int = 64,
    residual_length: int = 128,
    enable_hadamard: bool = True,
    hadamard_mode: str = "tile",
    hadamard_group_size: int = 64,
) -> RotateTileKVConfig:
    return RotateTileKVConfig(
        enable_hadamard=enable_hadamard,
        hadamard_mode=hadamard_mode,
        hadamard_group_size=hadamard_group_size,
        k_bits=k_bits,
        v_bits=v_bits,
        quant_impl=quant_impl,
        k_quant_scheme=k_quant_scheme,
        v_quant_scheme=v_quant_scheme,
        group_size=group_size,
        quant_granularity=quant_granularity,
        tile_size=tile_size,
        residual_length=residual_length,
    )


class LlamaJSQKVAttention(LlamaDiffSparseKVAttention):
    """
    Minimal JSQKV attention:
    1. Use DiffSparseKV to sparsify tokens.
    2. Apply RotateTileKV fake quantization to the resulting KV cache.
    """

    def __init__(self, config, quant_cfg: Optional[RotateTileKVConfig] = None):
        super().__init__(config=config)
        self.quant_cfg = quant_cfg or create_jsqkv_quant_config(k_bits=2, v_bits=2)
        # For the first validation pass, keep DiffSparseKV behavior unchanged
        # during decode and only quantize the retained prefill cache once.
        self.quantize_decode_cache = False

    def _normalize_quant_scheme(self, scheme: str) -> str:
        value = scheme.lower().replace("_", "-")
        if value in {"default", "inherit"}:
            return self.quant_cfg.quant_granularity
        return value

    def _quantize_level1_tokens(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        token_levels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        level1_idx = torch.nonzero(token_levels == 1, as_tuple=False).squeeze(-1)
        if key_states.shape[2] == 0 or level1_idx.numel() == 0:
            return key_states, value_states

        k_scheme = self._normalize_quant_scheme(self.quant_cfg.k_quant_scheme)
        v_scheme = self._normalize_quant_scheme(self.quant_cfg.v_quant_scheme)
        tile_size = self.quant_cfg.tile_size if self.quant_cfg.tile_size is not None else self.head_dim // 2

        if self.quant_cfg.k_bits is not None and self.quant_cfg.k_bits < 16:
            k_slice = key_states[:, :, level1_idx, :]
            if k_scheme == "kivi-channel":
                key_states[:, :, level1_idx, :] = fake_quant_k_cache_kivi_channel(
                    k_slice,
                    self.quant_cfg.k_bits,
                    self.quant_cfg.group_size,
                )
            else:
                key_states[:, :, level1_idx, :] = fake_quant_kv(
                    k_slice,
                    self.quant_cfg.k_bits,
                    k_scheme,
                    tile_size,
                    quant_impl=self.quant_cfg.quant_impl,
                )

        if self.quant_cfg.v_bits is not None and self.quant_cfg.v_bits < 16:
            v_slice = value_states[:, :, level1_idx, :]
            value_states[:, :, level1_idx, :] = fake_quant_kv(
                v_slice,
                self.quant_cfg.v_bits,
                v_scheme,
                tile_size,
                quant_impl=self.quant_cfg.quant_impl,
            )
        return key_states, value_states

    def _quantize_prefix_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[2] == 0:
            return key_states, value_states

        k_scheme = self._normalize_quant_scheme(self.quant_cfg.k_quant_scheme)
        v_scheme = self._normalize_quant_scheme(self.quant_cfg.v_quant_scheme)
        tile_size = self.quant_cfg.tile_size if self.quant_cfg.tile_size is not None else self.head_dim // 2

        if self.quant_cfg.k_bits is not None and self.quant_cfg.k_bits < 16:
            if k_scheme == "kivi-channel":
                key_states = fake_quant_k_cache_kivi_channel(
                    key_states,
                    self.quant_cfg.k_bits,
                    self.quant_cfg.group_size,
                )
            else:
                key_states = fake_quant_kv(
                    key_states,
                    self.quant_cfg.k_bits,
                    k_scheme,
                    tile_size,
                    quant_impl=self.quant_cfg.quant_impl,
                )

        if self.quant_cfg.v_bits is not None and self.quant_cfg.v_bits < 16:
            value_states = fake_quant_kv(
                value_states,
                self.quant_cfg.v_bits,
                v_scheme,
                tile_size,
                quant_impl=self.quant_cfg.quant_impl,
            )
        return key_states, value_states

    def _prefill_with_diff_sparse(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv_seq_len: int,
    ):
        attn_output, attn_weights, past_key_value = super()._prefill_with_diff_sparse(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            kv_seq_len=kv_seq_len,
        )

        if (
            (self.quant_cfg.k_bits is None or self.quant_cfg.k_bits >= 16)
            and (self.quant_cfg.v_bits is None or self.quant_cfg.v_bits >= 16)
        ):
            return attn_output, attn_weights, past_key_value

        past_keys, past_values, past_length = past_key_value
        window_a_size = min(self.window_size, past_keys.shape[2])
        compress_end = max(past_keys.shape[2] - window_a_size, 0)
        if compress_end <= 0:
            return attn_output, attn_weights, past_key_value

        compressed_keys = past_keys[:, :, :compress_end, :].contiguous()
        compressed_values = past_values[:, :, :compress_end, :].contiguous()
        compressed_keys, compressed_values = self._quantize_prefix_cache(
            compressed_keys,
            compressed_values,
        )

        past_keys = torch.cat([compressed_keys, past_keys[:, :, compress_end:, :]], dim=2)
        past_values = torch.cat([compressed_values, past_values[:, :, compress_end:, :]], dim=2)
        self.initialize_dual_window_after_prefill(past_keys, past_values)
        return attn_output, attn_weights, (past_keys, past_values, past_length)

    def _prefill_with_snapkv(
        self,
        attn_output: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        importance_scores: torch.Tensor,
        kv_seq_len: int,
    ):
        return super()._prefill_with_snapkv(
            attn_output=attn_output,
            key_states=key_states,
            value_states=value_states,
            importance_scores=importance_scores,
            kv_seq_len=kv_seq_len,
        )

    def _decode_with_diff_sparse(
        self,
        query_states: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Tuple[torch.Tensor],
    ):
        attn_output, attn_weights, next_past = super()._decode_with_diff_sparse(
            query_states=query_states,
            key_state=key_state,
            value_state=value_state,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        if not self.quantize_decode_cache:
            return attn_output, attn_weights, next_past
        return attn_output, attn_weights, self._quantize_past_key_value(next_past, q_len=key_state.shape[2])

    def _decode_with_diff_sparse_ORIGINAL(
        self,
        query_states: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Tuple[torch.Tensor],
    ):
        attn_output, attn_weights, next_past = super()._decode_with_diff_sparse_ORIGINAL(
            query_states=query_states,
            key_state=key_state,
            value_state=value_state,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        if not self.quantize_decode_cache:
            return attn_output, attn_weights, next_past
        return attn_output, attn_weights, self._quantize_past_key_value(next_past, q_len=key_state.shape[2])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )

        hadamard_mode = getattr(self.quant_cfg, "hadamard_mode", "none").lower().replace("_", "-")
        if self.quant_cfg.enable_hadamard and hadamard_mode == "none":
            hadamard_mode = "full"
        if hadamard_mode == "none":
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

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

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        query_states, key_states = _maybe_apply_hadamard(
            query_states,
            key_states,
            self.quant_cfg,
            self.head_dim,
        )

        if past_key_value is None:
            self.generation_count = 0
            attn_output, attn_weights, past_key_value = self._prefill_with_diff_sparse(
                query_states, key_states, value_states, attention_mask, kv_seq_len
            )
        else:
            self.generation_count += 1
            if self.protected_heavy_ratio > 0.0:
                attn_output, attn_weights, past_key_value = self._decode_with_diff_sparse_ORIGINAL(
                    query_states, key_states, value_states, attention_mask, past_key_value
                )
            else:
                attn_output, attn_weights, past_key_value = self._decode_with_diff_sparse(
                    query_states, key_states, value_states, attention_mask, past_key_value
                )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum(
                F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)
            )
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


def apply_jsqkv_to_llama(
    model: LlamaForCausalLMDiffSparseKV,
    quant_cfg: RotateTileKVConfig,
) -> LlamaForCausalLMDiffSparseKV:
    for layer in model.model.layers:
        old_attn = layer.self_attn
        attn_device = next(old_attn.parameters()).device
        attn_dtype = next(old_attn.parameters()).dtype

        new_attn = LlamaJSQKVAttention(old_attn.config, quant_cfg=quant_cfg)
        new_attn.load_state_dict(old_attn.state_dict(), strict=True)
        new_attn.to(device=attn_device, dtype=attn_dtype)
        new_attn.eval()
        new_attn.quantize_decode_cache = False

        # Preserve runtime settings already attached by the DiffSparse loader.
        new_attn.generation_count = getattr(old_attn, "generation_count", 0)
        new_attn.prefill_length = getattr(old_attn, "prefill_length", 0)
        new_attn.current_sequence_length = getattr(old_attn, "current_sequence_length", 0)
        new_attn.window_state = getattr(old_attn, "window_state", None)
        new_attn.diff_sparse_thresholds = getattr(old_attn, "diff_sparse_thresholds", None)
        new_attn.thresholds_computed = getattr(old_attn, "thresholds_computed", False)

        layer.self_attn = new_attn

    model.config.jsqkv = quant_cfg.to_dict()
    return model


def patch_last_token_logits_for_generation(
    model: LlamaForCausalLMDiffSparseKV,
) -> LlamaForCausalLMDiffSparseKV:
    if getattr(model, "_jsqkv_last_token_logits_patch", False):
        return model

    def forward_patched(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if labels is None:
            hidden_states = hidden_states[:, -1:, :]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = types.MethodType(forward_patched, model)
    model._jsqkv_last_token_logits_patch = True
    return model


def load_jsqkv_llama(
    *,
    model_path: str,
    diff_target_distribution,
    diff_sparsity_levels,
    quant_cfg: RotateTileKVConfig,
    max_length: int,
    obs_window_size: int = 128,
    importance_mode: str = "value_aware",
    head_aggregation_mode: str = "max",
    value_sink_keep: int = 2,
    level_2_mode: str = "evict",
    protected_heavy_ratio: float = 0.0,
    protected_recent_ratio: float = 1.0,
    torch_dtype=torch.float16,
    device_map="auto",
):
    base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(base_config, "model_type", "")
    if model_type != "llama":
        raise NotImplementedError(f"JSQKV-lite currently supports only llama models, got {model_type}")

    base_config.k_sparsity = 0.0
    base_config.v_sparsity = 0.0
    base_config.group_size = 32
    base_config.residual_length = 32
    base_config.use_flash = True

    config = create_diff_sparse_kv_config(
        base_config=base_config,
        enable_diff_sparse=True,
        target_distribution=list(diff_target_distribution),
        sparsity_levels=list(diff_sparsity_levels),
        diff_sparse_window_size=128,
        obs_window_size=obs_window_size,
        debug_diff_sparse=False,
        level_2_mode=level_2_mode,
        importance_mode=importance_mode,
        value_sink_keep=value_sink_keep,
        head_aggregation_mode=head_aggregation_mode,
        selector_mode="diffsparse",
        protected_heavy_ratio=protected_heavy_ratio,
        protected_recent_ratio=protected_recent_ratio,
    )
    config.max_position_embeddings = max(config.max_position_embeddings, max_length)

    model = LlamaForCausalLMDiffSparseKV.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model = apply_jsqkv_to_llama(model, quant_cfg)
    return model, config
