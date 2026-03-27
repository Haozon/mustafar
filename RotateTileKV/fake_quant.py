from __future__ import annotations

import math

import torch


def hadamard_transform_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Applies a normalized Walsh-Hadamard transform on the last dimension."""
    dim = x.shape[-1]
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Hadamard transform requires a power-of-two last dim, got {dim}")

    original_dtype = x.dtype
    y = x.float().reshape(-1, dim)
    h = 1
    while h < dim:
        y = y.reshape(-1, dim // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        y = y.reshape(-1, dim)
        h *= 2
    y = y.reshape_as(x) * (1.0 / math.sqrt(dim))
    return y.to(dtype=original_dtype)


def grouped_hadamard_transform_last_dim(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Applies independent normalized Hadamard transforms to groups on the last dimension."""
    dim = x.shape[-1]
    if group_size <= 0 or (group_size & (group_size - 1)) != 0:
        raise ValueError(f"group_size must be a positive power of two, got {group_size}")
    if dim % group_size != 0:
        raise ValueError(f"Last dim {dim} must be divisible by group_size {group_size}")

    reshaped = x.reshape(*x.shape[:-1], dim // group_size, group_size)
    transformed = hadamard_transform_last_dim(reshaped)
    return transformed.reshape_as(x)


def _asym_quant_dequant_last_dim(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits is None or bits >= 16:
        return x

    qmax = float((1 << bits) - 1)
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    value_range = x_max - x_min
    degenerate = value_range <= 1e-8

    scale = value_range / qmax
    scale = torch.where(degenerate, torch.ones_like(scale), scale)

    zero = torch.round(-x_min / scale)
    zero = torch.where(degenerate, torch.zeros_like(zero), zero)
    zero = zero.clamp(0.0, qmax)

    q = torch.round(x / scale + zero).clamp(0.0, qmax)
    return scale * (q - zero)


def _kivi_quant_dequant_last_dim(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits is None or bits >= 16:
        return x

    qmax = float((1 << bits) - 1)
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    value_range = x_max - x_min
    degenerate = value_range <= 1e-8

    scale = value_range / qmax
    scale = torch.where(degenerate, torch.ones_like(scale), scale)
    shifted = (x - x_min) / scale
    q = torch.round(shifted).clamp(0.0, qmax)
    dequant = q * scale + x_min
    dequant = torch.where(degenerate, x_min.expand_as(dequant), dequant)
    return dequant


def _normalize_granularity(granularity: str) -> str:
    value = granularity.lower().replace("_", "-")
    aliases = {
        "pertoken": "per-token",
        "pertokenhead": "per-token-head",
        "pertokenhead": "per-token-head",
        "pertokentile": "per-token-tile",
    }
    return aliases.get(value, value)


def fake_quant_kv(
    x: torch.Tensor,
    bits: int,
    granularity: str,
    tile_size: int | None = None,
    quant_impl: str = "default",
) -> torch.Tensor:
    """Fake-quantizes a KV tensor of shape [B, H, T, D] using asymmetric quantization."""
    if bits is None or bits >= 16:
        return x

    granularity = _normalize_granularity(granularity)
    original_dtype = x.dtype
    x_fp = x.float()
    batch, heads, tokens, head_dim = x_fp.shape
    quant_impl = quant_impl.lower().replace("_", "-")
    quant_fn = _asym_quant_dequant_last_dim if quant_impl == "default" else _kivi_quant_dequant_last_dim
    if quant_impl not in {"default", "kivi"}:
        raise ValueError(f"Unsupported quant_impl: {quant_impl}")

    if granularity == "per-token":
        reshaped = x_fp.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)
        quantized = quant_fn(reshaped, bits)
        return (
            quantized.reshape(batch, tokens, heads, head_dim)
            .transpose(1, 2)
            .contiguous()
            .to(dtype=original_dtype)
        )

    if granularity == "per-token-head":
        return quant_fn(x_fp, bits).to(dtype=original_dtype)

    if granularity == "per-token-tile":
        tile = tile_size if tile_size is not None else head_dim // 2
        if tile <= 0 or head_dim % tile != 0:
            raise ValueError(f"Invalid tile size {tile} for head dim {head_dim}")
        reshaped = x_fp.reshape(batch, heads, tokens, head_dim // tile, tile)
        quantized = quant_fn(reshaped, bits)
        return quantized.reshape_as(x_fp).to(dtype=original_dtype)

    raise ValueError(
        "Unsupported quant granularity. Expected one of: "
        "'per-token', 'per-token-head', 'per-token-tile'."
    )


def fake_quant_k_cache_kivi_channel(
    x: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """KIVI-style fake quant for K: per-channel, grouped along token dimension."""
    if bits is None or bits >= 16:
        return x
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    original_dtype = x.dtype
    x_fp = x.float()
    batch, heads, tokens, head_dim = x_fp.shape
    x_perm = x_fp.permute(0, 1, 3, 2).contiguous()  # [B, H, D, T]

    pad_tokens = (group_size - tokens % group_size) % group_size
    if pad_tokens:
        pad = torch.zeros(batch, heads, head_dim, pad_tokens, device=x_perm.device, dtype=x_perm.dtype)
        x_perm = torch.cat([x_perm, pad], dim=-1)

    padded_tokens = x_perm.shape[-1]
    reshaped = x_perm.reshape(batch, heads, head_dim, padded_tokens // group_size, group_size)
    quantized = _kivi_quant_dequant_last_dim(reshaped, bits).reshape(batch, heads, head_dim, padded_tokens)
    if pad_tokens:
        quantized = quantized[..., :tokens]
    return quantized.permute(0, 1, 3, 2).contiguous().to(dtype=original_dtype)
