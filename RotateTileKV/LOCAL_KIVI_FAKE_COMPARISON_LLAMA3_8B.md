# Local Fake-Quant Comparison on Llama3-8B

This comparison uses only the local `RotateTileKV` code path.

Model:

- `/mnt/home/zh/model/Meta-Llama-3-8B-Instruct`

Protocol:

- datasets: `trec,triviaqa,passage_count,qasper`
- sample limit per dataset: `10`
- `residual_length = 128`
- same prompt/truncation/scoring pipeline

## Methods Compared

### 1. KIVI

Implemented in local `RotateTileKV` codebase with:

- `quant_impl = kivi`
- `k_quant_scheme = kivi-channel`
- `v_quant_scheme = per-token-head`
- `group_size = 128`

Result directories:

- `/mnt/home/zh/mustafar/RotateTileKV/exp_llama3_8b_res128_l10/kivi_align_fake_4bit`
- `/mnt/home/zh/mustafar/RotateTileKV/exp_llama3_8b_res128_l10/kivi_align_fake_2bit`

### 2. Local RotateTileKV method

Implemented in local `RotateTileKV` codebase with:

- `per-token-tile`
- `tile_size = 64`
- optional `tile hadamard(64)`

Result directories:

- `/mnt/home/zh/mustafar/RotateTileKV/exp_llama3_8b_res128_l10/per_token_tile_*`

## Main Comparison

| Method | trec | triviaqa | passage_count | qasper | average |
|---|---:|---:|---:|---:|---:|
| KIVI 4bit | 80.00 | 100.00 | 20.00 | 41.17 | 60.29 |
| local RotateTileKV 4bit | 80.00 | 100.00 | 10.00 | 42.87 | 58.22 |
| local RotateTileKV 4bit + tile hadamard(64) | 80.00 | 100.00 | 10.00 | 44.30 | 58.58 |
| KIVI 2bit | 70.00 | 90.00 | 0.00 | 28.90 | 47.23 |
| local RotateTileKV 2bit | 20.00 | 18.98 | 0.00 | 10.00 | 12.25 |
| local RotateTileKV 2bit + tile hadamard(64) | 70.00 | 94.00 | 0.00 | 22.20 | 46.55 |

## V-only Reference

These runs keep `K = fp16` and quantize only `V`.

| Method | average |
|---|---:|
| V-only tile 4bit | 59.81 |
| V-only tile 3bit | 58.04 |
| V-only tile 2bit | 57.72 |
| V-only head 4bit | 57.21 |
| V-only head 2bit | 54.67 |

Interpretation:

- `V` quantization is not the main bottleneck.
- The major gap comes from `K` quantization.

## Interpretation

### 4bit

- KIVI 4bit: `60.29`
- Local RotateTileKV 4bit + tile hadamard(64): `58.58`

Gap: about `1.71`.

This is close enough to say the two implementations are largely aligned at 4bit.

### 2bit

- KIVI 2bit: `47.23`
- Local RotateTileKV 2bit + tile hadamard(64): `46.55`

Gap: about `0.68`.

This is also close enough to say the two implementations are largely aligned at 2bit once:

- residual window is enabled
- KIVI-style K quantization is used on the local side
- tile hadamard is enabled on the RotateTileKV side

## Practical Conclusion

- Before alignment, KIVI looked much stronger because the quantization schemes were different.
- After aligning the implementation inside the same local codebase, the gap becomes small.
- `V-only` is already strong, so the main issue is `K`.
- Tile-sized Hadamard helps the local method a lot at 2bit.
- At 4bit, local KIVI-style fake quant still has a small edge, but the difference is no longer large.
