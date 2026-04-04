# JSQKV Final Experiment Queue

Date: `2026-04-04`

## Goal

Get the fastest publication-usable joint accuracy results for the JSQKV section
 under the now-aligned evaluation protocol.

Current strongest signal:

- `qasper` full, `70% + 4bit`
  - `M+K proxy`: `40.39`
  - `JSQKV`: `41.25`
  - delta: `+0.86`

This means the current priority should be to expand from a single full task to
the six focal tasks, rather than immediately broadening to more settings/models.

## Focal Tasks

- `narrativeqa`
- `qasper`
- `multifieldqa_en`
- `hotpotqa`
- `trec`
- `lcc`

Dataset sizes:

- `narrativeqa`: `200`
- `qasper`: `200`
- `multifieldqa_en`: `150`
- `hotpotqa`: `200`
- `trec`: `200`
- `lcc`: `500`

Total samples per selected6 sweep: `1450`

## Priority Order

### Tier 1: JSQKV-only, most likely to become paper-usable fastest

1. `JSQKV, 70% + 4bit + no hadamard, selected6 full`
2. `JSQKV, 70% + 2bit + tile hadamard(64), selected6 full`

Reason:

- `RotateTileKV` remains a separate `vs KIVI` line and should not be mixed into
  the joint sparse-quant paper baseline
- `4bit` is the most stable joint setting
- `2bit + tile hadamard(64)` is the strongest aggressive setting from aligned
  small-sample runs

### Tier 2: expand budget coverage only after Tier 1 looks good

3. `JSQKV, 50% + 4bit, selected6 full`
4. `JSQKV, 50% + 2bit, selected6 full`

### Tier 3: larger scope only after selected6 is stable

5. `JSQKV, 70% + 4bit + no hadamard, broader tasks / more models`
6. `JSQKV, 70% + 2bit + tile hadamard(64), broader tasks / more models`

## Runtime Estimate

Based on aligned `limit=12` timings and the real dataset sizes:

- `narrativeqa`: about `5-6 min`
- `qasper`: about `7-8 min`
- `multifieldqa_en`: about `4 min`
- `hotpotqa`: about `5 min`
- `trec`: about `11-12 min`
- `lcc`: about `24-27 min`

Estimated wall-clock per selected6 full configuration:

- about `55-65 min`

Estimated wall-clock for Tier 1 only:

- serial: about `2 hours`
- parallel on 2 GPUs: about `1 hour`

Estimated wall-clock for Tier 1 + Tier 2:

- serial: about `4 hours`
- parallel on 4 GPUs: about `1.5-2 hours`

## Execution Strategy

To optimize for fastest path to a usable paper result:

- launch Tier 1 first
- inspect `70% + 4bit` before broadening further
- only after Tier 1 is satisfactory, expand to `50%` or broader scope
