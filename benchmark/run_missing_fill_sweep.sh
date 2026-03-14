#!/bin/bash
# Fill missing benchmark points for 5-line BS plots.
# Default target:
#   model: Meta-Llama-3-8B-Instruct
#   prompt_length: 4096
#   output_length: 1024
#   configs: dense / sparse_50 / sparse_50_quant_2bit
#   bs: 1 2 4 6 8

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-mustar}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/zh/model/Meta-Llama-3-8B-Instruct}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
OUTPUT_LIST="${OUTPUT_LIST:-1024}"
BS_LIST="${BS_LIST:-1 2 4 6 8}"
NUM_REPEATS="${NUM_REPEATS:-1}"
GROUP_SIZE="${GROUP_SIZE:-32}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-256}"
QUANT_BITS="${QUANT_BITS:-2}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="benchmark_results_missing_fill_${TIMESTAMP}"
LOG_DIR="${OUT_DIR}/logs"
SUMMARY_CSV="${OUT_DIR}/summary.csv"

mkdir -p "$LOG_DIR"

cat > "$SUMMARY_CSV" <<EOF
config_key,batch_size,prompt_length,output_length,num_repeats,status,ttft_ms,tpot_ms,total_ms,peak_gb,batch_ms,throughput_tps,log_file
EOF

append_summary_row() {
    local summary_csv="$1"
    local log_file="$2"
    local config_key="$3"
    local bs="$4"
    local prompt_len="$5"
    local out_len="$6"
    local repeats="$7"
    local retcode="$8"

    python - "$summary_csv" "$log_file" "$config_key" "$bs" "$prompt_len" "$out_len" "$repeats" "$retcode" <<'PY'
import csv
import re
import sys

summary_csv, log_file, config_key, bs, prompt_len, out_len, repeats, retcode = sys.argv[1:]

def extract_last(pattern: str, text: str):
    m = re.findall(pattern, text, flags=re.MULTILINE)
    return m[-1] if m else ""

try:
    content = open(log_file, "r", encoding="utf-8", errors="ignore").read()
except FileNotFoundError:
    content = ""

ttft = extract_last(r"(?:^TTFT:\s*|Average TTFT:\s*)([\d.]+)\s*ms", content)
tpot = extract_last(r"(?:^TPOT:\s*|Average TPOT:\s*)([\d.]+)\s*ms", content)
total = extract_last(r"Total generation time:\s*([\d.]+)\s*ms", content)
peak = extract_last(r"(?:Peak memory|Peak mem):\s*([\d.]+)\s*GB", content)
batch = extract_last(r"(?:Batch generation time|Average time):\s*([\d.]+)\s*ms", content)
throughput = extract_last(r"Throughput:\s*([\d.]+)\s*tokens/sec", content)

if not throughput and batch:
    try:
        throughput = f"{int(bs) * int(out_len) / (float(batch) / 1000.0):.2f}"
    except Exception:
        throughput = ""

status = "ok" if retcode == "0" else f"failed({retcode})"

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        config_key, bs, prompt_len, out_len, repeats, status,
        ttft, tpot, total, peak, batch, throughput, log_file
    ])
PY
}

run_one() {
    local config_key="$1"
    local bs="$2"
    local out_len="$3"
    local log_file="$4"

    echo ""
    echo "======================================================================"
    echo "[$config_key] bs=$bs, in=$PROMPT_LENGTH, out=$out_len, repeats=$NUM_REPEATS"
    echo "Log: $log_file"
    echo "======================================================================"

    if [[ "$config_key" == "dense" ]]; then
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        MODEL_NAME_OR_PATH="$MODEL_NAME_OR_PATH" \
        K_SPARSITY="0.0" \
        V_SPARSITY="0.0" \
        GROUP_SIZE="$GROUP_SIZE" \
        BATCH_SIZE="$bs" \
        PROMPT_LENGTH="$PROMPT_LENGTH" \
        OUTPUT_LENGTH="$out_len" \
        NUM_REPEATS="$NUM_REPEATS" \
        MUSTAFAR_MODE="false" \
        conda run -n "$CONDA_ENV" python mem_spd_test.py 2>&1 | tee "$log_file"
        return ${PIPESTATUS[0]}
    fi

    if [[ "$config_key" == "sparse_50" ]]; then
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        MODEL_NAME_OR_PATH="$MODEL_NAME_OR_PATH" \
        K_SPARSITY="0.5" \
        V_SPARSITY="0.5" \
        GROUP_SIZE="$GROUP_SIZE" \
        BATCH_SIZE="$bs" \
        PROMPT_LENGTH="$PROMPT_LENGTH" \
        OUTPUT_LENGTH="$out_len" \
        NUM_REPEATS="$NUM_REPEATS" \
        MUSTAFAR_MODE="true" \
        conda run -n "$CONDA_ENV" python mem_spd_test.py 2>&1 | tee "$log_file"
        return ${PIPESTATUS[0]}
    fi

    # sparse_50_quant_2bit
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    MODEL_NAME_OR_PATH="$MODEL_NAME_OR_PATH" \
    K_SPARSITY="0.5" \
    V_SPARSITY="0.5" \
    GROUP_SIZE="$GROUP_SIZE" \
    RESIDUAL_LENGTH="$RESIDUAL_LENGTH" \
    QUANT_BITS="$QUANT_BITS" \
    BATCH_SIZE="$bs" \
    PROMPT_LENGTH="$PROMPT_LENGTH" \
    OUTPUT_LENGTH="$out_len" \
    NUM_REPEATS="$NUM_REPEATS" \
    QUANT_MODE="true" \
    conda run -n "$CONDA_ENV" python mem_spd_test_quant.py 2>&1 | tee "$log_file"
    return ${PIPESTATUS[0]}
}

echo "Output directory: $OUT_DIR"
echo "Conda env: $CONDA_ENV"
echo "Model: $MODEL_NAME_OR_PATH"
echo "Prompt length: $PROMPT_LENGTH"
echo "Output list: $OUTPUT_LIST"
echo "BS list: $BS_LIST"
echo "Num repeats: $NUM_REPEATS"
echo "Configs: dense, sparse_50, sparse_50_quant_2bit"

for out_len in $OUTPUT_LIST; do
    for bs in $BS_LIST; do
        for cfg in dense sparse_50 sparse_50_quant_2bit; do
            log_file="$LOG_DIR/${cfg}_bs${bs}_in${PROMPT_LENGTH}_out${out_len}.txt"
            run_one "$cfg" "$bs" "$out_len" "$log_file"
            ret=$?
            append_summary_row "$SUMMARY_CSV" "$log_file" "$cfg" "$bs" "$PROMPT_LENGTH" "$out_len" "$NUM_REPEATS" "$ret"
        done
    done
done

echo ""
echo "Missing-fill sweep finished."
echo "Summary: $SUMMARY_CSV"
