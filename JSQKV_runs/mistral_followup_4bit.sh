#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nas/nas_192.168.7.2/zh/mustafar"
cd "$ROOT"

CONDA_BIN="/home/zh/miniconda3/bin/conda"
MODEL="/home/zh/nas/nas_10g/models/Mistral-7B-v0.1"
LOG_DIR="JSQKV_runs/run_logs"
OUT_DIR="JSQKV_runs/mistral_selected6_4096"

mkdir -p "$LOG_DIR" "$OUT_DIR"

launched_proxy=0
launched_jsqkv=0

echo "[mistral_followup_4bit] started at $(date '+%F %T')" | tee -a "$LOG_DIR/mistral_followup_4bit.log"

while true; do
  if [[ $launched_proxy -eq 0 ]] \
    && [[ -f "JSQKV_runs/mistral_qasper_4096/mistral70_uniformkivi_4bit_qasper_full_4096/result.json" ]] \
    && ! pgrep -f "mistral70_uniformkivi_4bit_selected6_full_4096" >/dev/null 2>&1; then
    echo "[mistral_followup_4bit] launching proxy selected6 on GPU0 at $(date '+%F %T')" | tee -a "$LOG_DIR/mistral_followup_4bit.log"
    /bin/bash -lc "
      export CUDA_VISIBLE_DEVICES=0
      exec \"$CONDA_BIN\" run --no-capture-output -n mustafar python JSQKV/eval_jsqkv_longbench.py \
        --model_path \"$MODEL\" \
        --max_length 4096 \
        --datasets narrativeqa qasper multifieldqa_en hotpotqa trec lcc \
        --output_dir \"$OUT_DIR\" \
        --output_tag mistral70_uniformkivi_4bit_selected6_full_4096 \
        --target_distribution 0.0,1.0,0.0 \
        --sparsity_levels 0.0,0.7,1.0 \
        --importance_mode value_aware \
        --head_aggregation_mode max \
        --value_sink_keep 2 \
        --level_2_mode evict \
        --k_bits 4 \
        --v_bits 4 \
        --quant_impl kivi \
        --k_quant_scheme kivi-channel \
        --v_quant_scheme per-token-head \
        --group_size 128 \
        --quant_granularity per-token-tile \
        --tile_size 64 \
        --residual_length 128 \
        --hadamard_mode none \
        --run_eval
    " >> "$LOG_DIR/mistral70_uniformkivi_4bit_selected6_full_4096.log" 2>&1 &
    launched_proxy=1
  fi

  if [[ $launched_jsqkv -eq 0 ]] \
    && [[ -f "JSQKV_runs/mistral_qasper_4096/mistral70_jsqkv_4bit_tilehad_qasper_full_4096/result.json" ]] \
    && ! pgrep -f "mistral70_jsqkv_4bit_tilehad_selected6_full_4096" >/dev/null 2>&1; then
    echo "[mistral_followup_4bit] launching JSQKV selected6 on GPU1 at $(date '+%F %T')" | tee -a "$LOG_DIR/mistral_followup_4bit.log"
    /bin/bash -lc "
      export CUDA_VISIBLE_DEVICES=1
      exec \"$CONDA_BIN\" run --no-capture-output -n mustafar python JSQKV/eval_jsqkv_longbench.py \
        --model_path \"$MODEL\" \
        --max_length 4096 \
        --datasets narrativeqa qasper multifieldqa_en hotpotqa trec lcc \
        --output_dir \"$OUT_DIR\" \
        --output_tag mistral70_jsqkv_4bit_tilehad_selected6_full_4096 \
        --target_distribution 0.0,0.75,0.25 \
        --sparsity_levels 0.0,0.6,1.0 \
        --importance_mode value_aware \
        --head_aggregation_mode max \
        --value_sink_keep 2 \
        --level_2_mode evict \
        --k_bits 4 \
        --v_bits 4 \
        --quant_impl default \
        --k_quant_scheme per-token-tile \
        --v_quant_scheme per-token-tile \
        --group_size 128 \
        --quant_granularity per-token-tile \
        --tile_size 64 \
        --residual_length 128 \
        --enable_hadamard \
        --hadamard_mode tile \
        --hadamard_group_size 64 \
        --run_eval
    " >> "$LOG_DIR/mistral70_jsqkv_4bit_tilehad_selected6_full_4096.log" 2>&1 &
    launched_jsqkv=1
  fi

  if [[ $launched_proxy -eq 1 && $launched_jsqkv -eq 1 ]]; then
    echo "[mistral_followup_4bit] both follow-up jobs launched; exiting at $(date '+%F %T')" | tee -a "$LOG_DIR/mistral_followup_4bit.log"
    exit 0
  fi

  sleep 60
done
