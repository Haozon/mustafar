#!/bin/bash
# Sweep batch size / output length for Mustafar and Mustafar-Quant.
# Designed to continue even when some configs fail (e.g. OOM).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-mustar}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/zh/model/Meta-Llama-3-8B-Instruct}"
K_SPARSITY="${K_SPARSITY:-0.7}"
V_SPARSITY="${V_SPARSITY:-0.7}"
GROUP_SIZE="${GROUP_SIZE:-32}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-256}"
QUANT_BITS="${QUANT_BITS:-2}"

PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
NUM_REPEATS="${NUM_REPEATS:-1}"
BS_LIST="${BS_LIST:-1 2 4 6 8 16 24}"
OUTPUT_LIST="${OUTPUT_LIST:-1024 4096}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="benchmark_results_bs_output_sweep_${TIMESTAMP}"
LOG_DIR="$OUT_DIR/logs"
SUMMARY_CSV="$OUT_DIR/summary.csv"

mkdir -p "$LOG_DIR"

cat > "$SUMMARY_CSV" <<EOF
mode,batch_size,prompt_length,output_length,num_repeats,status,ttft_ms,tpot_ms,total_ms,peak_gb,batch_ms,throughput_tps,log_file
EOF

append_summary_row() {
    local summary_csv="$1"
    local log_file="$2"
    local mode="$3"
    local bs="$4"
    local prompt_len="$5"
    local out_len="$6"
    local repeats="$7"
    local retcode="$8"

    python - "$summary_csv" "$log_file" "$mode" "$bs" "$prompt_len" "$out_len" "$repeats" "$retcode" <<'PY'
import csv
import re
import sys

summary_csv, log_file, mode, bs, prompt_len, out_len, repeats, retcode = sys.argv[1:]

def extract_last(pattern: str, text: str):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else ""

try:
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
except FileNotFoundError:
    content = ""

ttft = extract_last(r"(?:^TTFT:\s*|Average TTFT:\s*)([\d.]+)\s*ms", content)
tpot = extract_last(r"(?:^TPOT:\s*|Average TPOT:\s*)([\d.]+)\s*ms", content)
total = extract_last(r"Total generation time:\s*([\d.]+)\s*ms", content)
peak = extract_last(r"Peak memory:\s*([\d.]+)\s*GB", content)
batch = extract_last(r"(?:Batch generation time|Average time):\s*([\d.]+)\s*ms", content)
throughput = extract_last(r"Throughput:\s*([\d.]+)\s*tokens/sec", content)

status = "ok" if retcode == "0" else f"failed({retcode})"

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        mode, bs, prompt_len, out_len, repeats, status,
        ttft, tpot, total, peak, batch, throughput, log_file
    ])
PY
}

run_one() {
    local mode="$1"
    local bs="$2"
    local out_len="$3"
    local log_file="$4"

    echo ""
    echo "======================================================================"
    echo "[$mode] bs=$bs, in=$PROMPT_LENGTH, out=$out_len, repeats=$NUM_REPEATS"
    echo "Log: $log_file"
    echo "======================================================================"

    if [[ "$mode" == "mustafar" ]]; then
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        MODEL_NAME_OR_PATH="$MODEL_NAME_OR_PATH" \
        K_SPARSITY="$K_SPARSITY" \
        V_SPARSITY="$V_SPARSITY" \
        GROUP_SIZE="$GROUP_SIZE" \
        BATCH_SIZE="$bs" \
        PROMPT_LENGTH="$PROMPT_LENGTH" \
        OUTPUT_LENGTH="$out_len" \
        NUM_REPEATS="$NUM_REPEATS" \
        MUSTAFAR_MODE="true" \
        conda run -n "$CONDA_ENV" python mem_spd_test.py 2>&1 | tee "$log_file"
        return ${PIPESTATUS[0]}
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    MODEL_NAME_OR_PATH="$MODEL_NAME_OR_PATH" \
    K_SPARSITY="$K_SPARSITY" \
    V_SPARSITY="$V_SPARSITY" \
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
echo "Batch list: $BS_LIST"
echo "Output list: $OUTPUT_LIST"
echo "Prompt length: $PROMPT_LENGTH"
echo "Num repeats: $NUM_REPEATS"

for bs in $BS_LIST; do
    for out_len in $OUTPUT_LIST; do
        mustafar_log="$LOG_DIR/mustafar_bs${bs}_in${PROMPT_LENGTH}_out${out_len}.txt"
        run_one "mustafar" "$bs" "$out_len" "$mustafar_log"
        ret=$?
        append_summary_row "$SUMMARY_CSV" "$mustafar_log" "mustafar" "$bs" "$PROMPT_LENGTH" "$out_len" "$NUM_REPEATS" "$ret"

        quant_log="$LOG_DIR/quant_bs${bs}_in${PROMPT_LENGTH}_out${out_len}.txt"
        run_one "quant" "$bs" "$out_len" "$quant_log"
        ret=$?
        append_summary_row "$SUMMARY_CSV" "$quant_log" "quant" "$bs" "$PROMPT_LENGTH" "$out_len" "$NUM_REPEATS" "$ret"
    done
done

echo ""
echo "Sweep finished."
echo "Summary: $SUMMARY_CSV"
