# JSQKV Final Results Summary

Date: `2026-04-04`

## Terminology

### `selected6 full`

`selected6 full` means:

- only the six focal LongBench tasks are evaluated:
  - `narrativeqa`
  - `qasper`
  - `multifieldqa_en`
  - `hotpotqa`
  - `trec`
  - `lcc`
- but for each of these six tasks, the **full dataset split** is used
- therefore it is:
  - more reliable than `limit=12`
  - but cheaper than full 16-task LongBench

This is the main fast-turnaround protocol currently used for recovering the
JSQKV accuracy section.

## Current Best Available Results

### 70\% + 4bit

Selected6 full:

- `M+K proxy`:
  - `44.18`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta70_uniformkivi_4bit_selected6_full_4096/result.json`
- `JSQKV`:
  - `45.34`
  - file:
    - `JSQKV_runs/final_selected6/meta70_jsqkv_4bit_nohad_selected6_full/result.json`

Delta:

- `+1.16`

Per-task:

- `NarrativeQA`: `20.94 -> 21.86` (`+0.92`)
- `Qasper`: `37.32 -> 38.81` (`+1.49`)
- `MultiFieldQA-En`: `44.90 -> 46.52` (`+1.62`)
- `HotpotQA`: `41.22 -> 40.64` (`-0.58`)
- `TREC`: `68.00 -> 70.00` (`+2.00`)
- `LCC`: `52.71 -> 54.23` (`+1.52`)

### 70\% + 2bit

Selected6 full:

- `M+K proxy`:
  - `39.83`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta70_uniformkivi_2bit_selected6_full_4096/result.json`
- `JSQKV`:
  - `43.12`
  - file:
    - `JSQKV_runs/final_selected6/meta70_jsqkv_2bit_tilehad_selected6_full/result.json`

Delta:

- `+3.29`

Per-task:

- `NarrativeQA`: `21.34 -> 19.70` (`-1.64`)
- `Qasper`: `35.43 -> 39.49` (`+4.06`)
- `MultiFieldQA-En`: `43.43 -> 44.31` (`+0.88`)
- `HotpotQA`: `39.21 -> 40.15` (`+0.94`)
- `TREC`: `63.00 -> 70.00` (`+7.00`)
- `LCC`: `36.59 -> 45.04` (`+8.45`)

### 50\% + 4bit

Selected6 full:

- `M+K proxy`:
  - `45.90`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta50_uniformkivi_4bit_selected6_full_4096/result.json`
- `JSQKV`:
  - `45.93`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta50_jsqkv_4bit_nohad_selected6_full_4096/result.json`

Delta:

- `+0.03`

Per-task:

- `NarrativeQA`: `21.61 -> 21.92` (`+0.31`)
- `Qasper`: `40.70 -> 39.38` (`-1.32`)
- `MultiFieldQA-En`: `47.62 -> 46.77` (`-0.85`)
- `HotpotQA`: `41.13 -> 41.38` (`+0.25`)
- `TREC`: `70.50 -> 70.50` (`+0.00`)
- `LCC`: `53.83 -> 55.64` (`+1.81`)

### 50\% + 2bit

Status:

- `M+K proxy` selected6 full is available:
  - `43.38`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta50_uniformkivi_2bit_selected6_full_4096/result.json`
- `JSQKV` selected6 full is still missing and is the top remaining item needed
  to complete the 4-setting main table.

## Full-Qasper Sanity

These single-task full runs were used to de-risk the selected6 full launches.

### 70\% + 4bit

- `M+K proxy`: `40.39`
- `JSQKV`: `41.25`
- delta: `+0.86`

Files:

- `JSQKV_runs/full_qasper/meta70_uniformkivi_4bit_qasper_full/result.json`
- `JSQKV_runs/full_qasper/meta70_jsqkv_4bit_nohad_qasper_full/result.json`

### 70\% + 2bit

- `JSQKV`: `39.49`

File:

- `JSQKV_runs/full_qasper_4096/meta70_jsqkv_2bit_tilehad_qasper_full_4096/result.json`

## Current Practical Conclusion

Under the recovered and aligned evaluation pipeline:

- `70\% + 4bit` is already clearly paper-usable
- `70\% + 2bit` is currently the strongest positive setting by margin
- `50\% + 4bit` is roughly neutral to slightly positive
- the only missing main-table corner is:
  - `50\% + 2bit`
