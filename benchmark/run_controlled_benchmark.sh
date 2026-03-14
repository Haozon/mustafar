#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-mustar}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/zh/model/Meta-Llama-3-8B-Instruct}"

BATCH_SIZE="${BATCH_SIZE:-8}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1024}"
NUM_REPEATS="${NUM_REPEATS:-3}"
GROUP_SIZE="${GROUP_SIZE:-32}"
RESIDUAL_LENGTH="${RESIDUAL_LENGTH:-256}"
QUANT_BITS="${QUANT_BITS:-2}"
QUANT_K_DEQUANT_MODE="${QUANT_K_DEQUANT_MODE:-0}"
QUANT_V_DEQUANT_MODE="${QUANT_V_DEQUANT_MODE:-0}"
QUANT_V_SPLIT_K="${QUANT_V_SPLIT_K:-4}"
QUANT_V_TILE_CONFIG="${QUANT_V_TILE_CONFIG:-0}"
ENABLE_NSYS="${ENABLE_NSYS:-0}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-10}"

OUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="${OUT_DIR}/summary.csv"

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "Mustafar Controlled Benchmark"
echo "========================================================================"
echo "OUTPUT_DIR=$OUT_DIR"
echo "Conda env: $CONDA_ENV"
echo "Model: $MODEL_NAME_OR_PATH"
echo "Config: bs=$BATCH_SIZE in=$PROMPT_LENGTH out=$OUTPUT_LENGTH repeats=$NUM_REPEATS"
echo "Enable nsys: $ENABLE_NSYS"
echo "Quant core: K_mode=$QUANT_K_DEQUANT_MODE V_mode=$QUANT_V_DEQUANT_MODE V_split_k=$QUANT_V_SPLIT_K V_tile=$QUANT_V_TILE_CONFIG"

cat > "$SUMMARY_CSV" <<EOF
config,k_sparsity,v_sparsity,quant_bits,quant_k_mode,quant_v_mode,quant_v_split_k,quant_v_tile_config,status,ttft_ms,tpot_ms,total_ms,peak_gb,batch_ms,throughput_tps,log_file
EOF

append_summary_row() {
    local summary_csv="$1"
    local log_file="$2"
    local config_key="$3"
    local k_sparsity="$4"
    local v_sparsity="$5"
    local quant_bits="$6"
    local quant_k_mode="$7"
    local quant_v_mode="$8"
    local quant_v_split_k="$9"
    local quant_v_tile_config="${10}"
    local status="${11}"
    local batch_size="${12}"
    local output_length="${13}"

    python - "$summary_csv" "$log_file" "$config_key" "$k_sparsity" "$v_sparsity" "$quant_bits" "$quant_k_mode" "$quant_v_mode" "$quant_v_split_k" "$quant_v_tile_config" "$status" "$batch_size" "$output_length" <<'PY'
import csv
import re
import sys

summary_csv, log_file, config_key, k_sparsity, v_sparsity, quant_bits, quant_k_mode, quant_v_mode, quant_v_split_k, quant_v_tile_config, status, batch_size, output_length = sys.argv[1:]

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
peak = extract_last(r"(?:Peak memory|Peak mem):\s*([\d.]+)\s*GB", content)
batch = extract_last(r"(?:Batch generation time|Average time):\s*([\d.]+)\s*ms", content)
throughput = extract_last(r"Throughput:\s*([\d.]+)\s*tokens/sec", content)

if not throughput and batch:
    try:
        throughput = f"{int(batch_size) * int(output_length) / (float(batch) / 1000.0):.2f}"
    except Exception:
        throughput = ""

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        config_key, k_sparsity, v_sparsity, quant_bits,
        quant_k_mode, quant_v_mode, quant_v_split_k, quant_v_tile_config,
        status,
        ttft, tpot, total, peak, batch, throughput, log_file
    ])
PY
}

run_case() {
    local config_key="$1"
    local script_name="$2"
    local k_sparsity="$3"
    local v_sparsity="$4"
    local quant_mode="$5"
    local mustafar_mode="$6"
    local quant_bits="$7"
    local log_file="${OUT_DIR}/${config_key}_output.txt"
    local profile_prefix="${OUT_DIR}/${config_key}_profile"
    local status=0

    local common_env=(
        "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        "MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
        "BATCH_SIZE=$BATCH_SIZE"
        "PROMPT_LENGTH=$PROMPT_LENGTH"
        "OUTPUT_LENGTH=$OUTPUT_LENGTH"
        "NUM_REPEATS=$NUM_REPEATS"
        "GROUP_SIZE=$GROUP_SIZE"
        "PYTHONUNBUFFERED=1"
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    )

    echo ""
    echo "========================================================================"
    echo "Running $config_key"
    echo "========================================================================"
    echo "Log: $log_file"

    if [[ "$script_name" == "mem_spd_test.py" ]]; then
        env "${common_env[@]}" \
            "K_SPARSITY=$k_sparsity" \
            "V_SPARSITY=$v_sparsity" \
            "MUSTAFAR_MODE=$mustafar_mode" \
            conda run -n "$CONDA_ENV" python "$script_name" 2>&1 | tee "$log_file"
        status=${PIPESTATUS[0]}
    else
        env "${common_env[@]}" \
            "K_SPARSITY=$k_sparsity" \
            "V_SPARSITY=$v_sparsity" \
            "RESIDUAL_LENGTH=$RESIDUAL_LENGTH" \
            "QUANT_BITS=$quant_bits" \
            "QUANT_K_DEQUANT_MODE=$QUANT_K_DEQUANT_MODE" \
            "QUANT_V_DEQUANT_MODE=$QUANT_V_DEQUANT_MODE" \
            "QUANT_V_SPLIT_K=$QUANT_V_SPLIT_K" \
            "QUANT_V_TILE_CONFIG=$QUANT_V_TILE_CONFIG" \
            "QUANT_MODE=$quant_mode" \
            conda run -n "$CONDA_ENV" python "$script_name" 2>&1 | tee "$log_file"
        status=${PIPESTATUS[0]}
    fi

    if [[ "$status" -eq 0 && "$ENABLE_NSYS" == "1" ]]; then
        echo "Running nsys for $config_key..."
        if [[ "$script_name" == "mem_spd_test.py" ]]; then
            env "${common_env[@]}" \
                "K_SPARSITY=$k_sparsity" \
                "V_SPARSITY=$v_sparsity" \
                "MUSTAFAR_MODE=$mustafar_mode" \
                nsys profile \
                    --trace=cuda,nvtx \
                    --output="$profile_prefix" \
                    --force-overwrite=true \
                    conda run -n "$CONDA_ENV" python "$script_name" >/dev/null 2>&1
        else
            env "${common_env[@]}" \
                "K_SPARSITY=$k_sparsity" \
                "V_SPARSITY=$v_sparsity" \
                "RESIDUAL_LENGTH=$RESIDUAL_LENGTH" \
                "QUANT_BITS=$quant_bits" \
                "QUANT_K_DEQUANT_MODE=$QUANT_K_DEQUANT_MODE" \
                "QUANT_V_DEQUANT_MODE=$QUANT_V_DEQUANT_MODE" \
                "QUANT_V_SPLIT_K=$QUANT_V_SPLIT_K" \
                "QUANT_V_TILE_CONFIG=$QUANT_V_TILE_CONFIG" \
                "QUANT_MODE=$quant_mode" \
                nsys profile \
                    --trace=cuda,nvtx \
                    --output="$profile_prefix" \
                    --force-overwrite=true \
                    conda run -n "$CONDA_ENV" python "$script_name" >/dev/null 2>&1
        fi

        if [[ -f "${profile_prefix}.nsys-rep" ]]; then
            nsys stats \
                --report cuda_gpu_kern_sum \
                --format csv \
                --output "${OUT_DIR}/${config_key}_kernels.csv" \
                "${profile_prefix}.nsys-rep"
        fi
    fi

    local status_label="ok"
    if [[ "$status" -ne 0 ]]; then
        status_label="failed(${status})"
    fi

    append_summary_row \
        "$SUMMARY_CSV" \
        "$log_file" \
        "$config_key" \
        "$k_sparsity" \
        "$v_sparsity" \
        "$quant_bits" \
        "$QUANT_K_DEQUANT_MODE" \
        "$QUANT_V_DEQUANT_MODE" \
        "$QUANT_V_SPLIT_K" \
        "$QUANT_V_TILE_CONFIG" \
        "$status_label" \
        "$BATCH_SIZE" \
        "$OUTPUT_LENGTH"

    sleep "$SLEEP_BETWEEN_RUNS"
    return "$status"
}

run_case "dense" "mem_spd_test.py" "0.0" "0.0" "false" "false" "" || true
run_case "sparse_50" "mem_spd_test.py" "0.5" "0.5" "false" "true" "" || true
run_case "sparse_70" "mem_spd_test.py" "0.7" "0.7" "false" "true" "" || true
run_case "sparse_50_quant_2bit" "mem_spd_test_quant.py" "0.5" "0.5" "true" "true" "$QUANT_BITS" || true
run_case "sparse_70_quant_2bit" "mem_spd_test_quant.py" "0.7" "0.7" "true" "true" "$QUANT_BITS" || true

python - "$SUMMARY_CSV" "${OUT_DIR}/baseline_comparison.txt" <<'PY'
import csv
import sys

summary_csv, output_file = sys.argv[1:]

rows = []
with open(summary_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

by_config = {row["config"]: row for row in rows}
baseline = by_config.get("dense")

def as_float(value: str):
    try:
        return float(value)
    except Exception:
        return None

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Config,Status,TTFT(ms),TPOT(ms),PeakGB,Throughput(tok/s),TTFT_speedup_vs_dense,TPOT_speedup_vs_dense,Peak_mem_saving_vs_dense\n")
    baseline_ttft = as_float(baseline["ttft_ms"]) if baseline else None
    baseline_tpot = as_float(baseline["tpot_ms"]) if baseline else None
    baseline_peak = as_float(baseline["peak_gb"]) if baseline else None

    for row in rows:
        ttft = as_float(row["ttft_ms"])
        tpot = as_float(row["tpot_ms"])
        peak = as_float(row["peak_gb"])

        ttft_speedup = ""
        tpot_speedup = ""
        mem_saving = ""

        if baseline_ttft and ttft:
            ttft_speedup = f"{baseline_ttft / ttft:.3f}"
        if baseline_tpot and tpot:
            tpot_speedup = f"{baseline_tpot / tpot:.3f}"
        if baseline_peak and peak is not None:
            mem_saving = f"{(1 - peak / baseline_peak) * 100:.2f}%"

        f.write(
            ",".join([
                row["config"],
                row["status"],
                row["ttft_ms"],
                row["tpot_ms"],
                row["peak_gb"],
                row["throughput_tps"],
                ttft_speedup,
                tpot_speedup,
                mem_saving,
            ]) + "\n"
        )
PY

echo ""
echo "Benchmark finished."
echo "Summary: $SUMMARY_CSV"
echo "Comparison: ${OUT_DIR}/baseline_comparison.txt"
