#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nas/nas_192.168.7.2/zh/mustafar"
cd "$ROOT"

CONDA_BIN="/home/zh/miniconda3/bin/conda"
MODEL="/home/zh/nas/nas_10g/models/Mistral-7B-v0.1"
LOG_DIR="JSQKV_runs/run_logs"
OUT_DIR="JSQKV_runs/mistral_selected6_4096"

mkdir -p "$LOG_DIR" "$OUT_DIR"

TARGETS=(
  "mistral50_uniformkivi_2bit_selected6_full_4096|0.0,1.0,0.0|0.0,0.5,1.0|kivi|kivi-channel|per-token-head|false"
  "mistral50_jsqkv_2bit_tilehad_selected6_full_4096|0.0,0.833333,0.166667|0.0,0.4,1.0|default|per-token-tile|per-token-tile|true"
)

pick_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 2000 && $3 < 20 {print $1; exit}'
}

echo "[mistral_followup_50_2bit_selected6] started at $(date '+%F %T')" | tee -a "$LOG_DIR/mistral_followup_50_2bit_selected6.log"

for item in "${TARGETS[@]}"; do
  IFS='|' read -r tag dist levels impl kscheme vscheme had <<<"$item"

  if [[ -f "$OUT_DIR/$tag/result.json" ]]; then
    echo "[mistral_followup_50_2bit_selected6] skip existing $tag" | tee -a "$LOG_DIR/mistral_followup_50_2bit_selected6.log"
    continue
  fi

  while true; do
    gpu="$(pick_free_gpu || true)"
    if [[ -n "${gpu:-}" ]]; then
      echo "[mistral_followup_50_2bit_selected6] launching $tag on GPU $gpu at $(date '+%F %T')" | tee -a "$LOG_DIR/mistral_followup_50_2bit_selected6.log"
      if [[ "$had" == "true" ]]; then
        /bin/bash -lc "
          export CUDA_VISIBLE_DEVICES=$gpu
          exec \"$CONDA_BIN\" run --no-capture-output -n mustafar python JSQKV/eval_jsqkv_longbench.py \
            --model_path \"$MODEL\" \
            --max_length 4096 \
            --datasets narrativeqa qasper multifieldqa_en hotpotqa trec lcc \
            --output_dir \"$OUT_DIR\" \
            --output_tag \"$tag\" \
            --target_distribution \"$dist\" \
            --sparsity_levels \"$levels\" \
            --importance_mode value_aware \
            --head_aggregation_mode max \
            --value_sink_keep 2 \
            --level_2_mode evict \
            --k_bits 2 \
            --v_bits 2 \
            --quant_impl \"$impl\" \
            --k_quant_scheme \"$kscheme\" \
            --v_quant_scheme \"$vscheme\" \
            --group_size 128 \
            --quant_granularity per-token-tile \
            --tile_size 64 \
            --residual_length 128 \
            --enable_hadamard \
            --hadamard_mode tile \
            --hadamard_group_size 64 \
            --run_eval
        " >> "$LOG_DIR/$tag.log" 2>&1 &
      else
        /bin/bash -lc "
          export CUDA_VISIBLE_DEVICES=$gpu
          exec \"$CONDA_BIN\" run --no-capture-output -n mustafar python JSQKV/eval_jsqkv_longbench.py \
            --model_path \"$MODEL\" \
            --max_length 4096 \
            --datasets narrativeqa qasper multifieldqa_en hotpotqa trec lcc \
            --output_dir \"$OUT_DIR\" \
            --output_tag \"$tag\" \
            --target_distribution \"$dist\" \
            --sparsity_levels \"$levels\" \
            --importance_mode value_aware \
            --head_aggregation_mode max \
            --value_sink_keep 2 \
            --level_2_mode evict \
            --k_bits 2 \
            --v_bits 2 \
            --quant_impl \"$impl\" \
            --k_quant_scheme \"$kscheme\" \
            --v_quant_scheme \"$vscheme\" \
            --group_size 128 \
            --quant_granularity per-token-tile \
            --tile_size 64 \
            --residual_length 128 \
            --hadamard_mode none \
            --run_eval
        " >> "$LOG_DIR/$tag.log" 2>&1 &
      fi
      sleep 5
      break
    fi
    sleep 60
  done
done

echo "[mistral_followup_50_2bit_selected6] queued both tasks at $(date '+%F %T')" | tee -a "$LOG_DIR/mistral_followup_50_2bit_selected6.log"
