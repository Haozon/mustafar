#!/bin/bash

# Threshold Stability Analysis Complete Pipeline
# Two-step process: Data Collection + Visualization Analysis

set -e  # Exit on error

echo "========================================"
echo "🎯 THRESHOLD STABILITY ANALYSIS PIPELINE"
echo "========================================"
echo "Objective: Validate that thresholds for the same layer-quantile have low variability"
echo "Hypothesis: Fixed thresholds can replace dynamic calibration"
echo ""

# Check Python environment
echo "📋 Checking Python environment..."
python --version
if ! python -c "import torch, transformers, numpy, pandas, matplotlib, seaborn, scipy" 2>/dev/null; then
    echo "❌ Missing required Python packages. Please install:"
    echo "pip install torch transformers numpy pandas matplotlib seaborn scipy datasets"
    exit 1
fi
echo "✅ Python environment check passed"

# Enter script directory
cd "$(dirname "$0")"
echo "📁 Current working directory: $(pwd)"

# Configuration parameters
MODEL_PATH="/home/zh/model/Meta-Llama-3-8B-Instruct"
DATA_DIR="./threshold_data"
VIZ_DIR="./visualization_results"

# Configurable parameters
DATASETS=("wikitext")  # Can extend to ("wikitext" "math" "gsm8k")
SAMPLE_SIZES=(20 50)   # Sample sizes
BOOTSTRAP_SAMPLES=10   # Bootstrap sampling iterations

echo "🤖 Using model: $MODEL_PATH"
echo "📊 Datasets: ${DATASETS[@]}"
echo "📏 Sample sizes: ${SAMPLE_SIZES[@]}"
echo "🔄 Bootstrap samples: $BOOTSTRAP_SAMPLES iterations"
echo "📁 Data directory: $DATA_DIR"
echo "📈 Visualization directory: $VIZ_DIR"

# Create output directories
mkdir -p "$DATA_DIR"
mkdir -p "$VIZ_DIR"

# echo ""
# echo "========================================"
# echo "📊 STEP 1: COLLECT THRESHOLD DATA"
# echo "========================================"

# # Build data collection command
# COLLECT_CMD="python collect_threshold_data.py \
#     --model-path \"$MODEL_PATH\" \
#     --output-dir \"$DATA_DIR\" \
#     --datasets ${DATASETS[@]} \
#     --sample-sizes ${SAMPLE_SIZES[@]} \
#     --bootstrap-samples $BOOTSTRAP_SAMPLES"

# echo "Executing command: $COLLECT_CMD"
# echo ""

# # Run data collection
# eval $COLLECT_CMD

# # Check if data collection succeeded
# if [ $? -eq 0 ]; then
#     echo "✅ Data collection completed"
# else
#     echo "❌ Data collection failed"
#     exit 1
# fi

# # Get latest session ID
# SESSION_ID=$(find "$DATA_DIR" -name "session_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- | xargs basename | sed 's/session_\(.*\)\.json/\1/')

# if [ -z "$SESSION_ID" ]; then
#     echo "⚠️  Session ID not found, will use latest data"
#     SESSION_ARG=""
# else
#     echo "📋 Found session ID: $SESSION_ID"
#     SESSION_ARG="--session-id $SESSION_ID"
# fi

echo ""
echo "========================================"
echo "📈 STEP 2: GENERATE VISUALIZATION ANALYSIS"
echo "========================================"

# Build visualization command
VIZ_CMD="python visualize_threshold_stability.py \
    --data-dir \"$DATA_DIR\" \
    --output-dir \"$VIZ_DIR\" \
    $SESSION_ARG"

echo "Executing command: $VIZ_CMD"
echo ""

# Run visualization
eval $VIZ_CMD

# Check if visualization succeeded
if [ $? -eq 0 ]; then
    echo "✅ Visualization analysis completed"
else
    echo "❌ Visualization analysis failed"
    exit 1
fi

echo ""
echo "========================================"
echo "📊 ANALYSIS RESULTS SUMMARY"
echo "========================================"

# Display generated files
echo "📁 Data files:"
if [ -d "$DATA_DIR" ]; then
    find "$DATA_DIR" -type f \( -name "*.db" -o -name "*.json" \) | sort | while read file; do
        size=$(du -h "$file" | cut -f1)
        echo "  📄 $file ($size)"
    done
fi

echo ""
echo "📈 Visualization files:"
if [ -d "$VIZ_DIR" ]; then
    find "$VIZ_DIR" -type f \( -name "*.png" -o -name "*.txt" -o -name "*.json" \) | sort | while read file; do
        echo "  🖼️  $file"
    done
fi

echo ""
echo "========================================"
echo "🎉 ANALYSIS COMPLETED!"
echo "========================================"

# Display key result files
KEY_FILES=(
    "$VIZ_DIR/main_boxplot_stability_validation.png"
    "$VIZ_DIR/cv_heatmap.png"
    "$VIZ_DIR/stability_statistics.png"
    "$VIZ_DIR/validation_report.txt"
)

echo "🔑 Key result files:"
for file in "${KEY_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (not generated)"
    fi
done

echo ""
echo "💡 Next steps:"
echo "  1. View main boxplot: $VIZ_DIR/main_boxplot_stability_validation.png"
echo "  2. View stability heatmap: $VIZ_DIR/cv_heatmap.png"
echo "  3. Read detailed report: $VIZ_DIR/validation_report.txt"

# Try to display validation result summary
REPORT_FILE="$VIZ_DIR/validation_report.txt"
if [ -f "$REPORT_FILE" ]; then
    echo ""
    echo "📋 Validation result summary:"
    echo "----------------------------------------"
    # Extract key information
    if grep -q "Hypothesis supported: ✅ YES" "$REPORT_FILE"; then
        echo "🎉 Hypothesis SUPPORTED! Fixed thresholds can be considered"
    elif grep -q "Hypothesis supported: ❌ NO" "$REPORT_FILE"; then
        echo "⚠️  Hypothesis NOT sufficiently supported, recommend dynamic calibration"
    fi
    
    # Display overall stability rate
    STABILITY_RATE=$(grep "Overall stability rate:" "$REPORT_FILE" | sed 's/.*: \([0-9.]*%\).*/\1/')
    if [ ! -z "$STABILITY_RATE" ]; then
        echo "📊 Overall stability rate: $STABILITY_RATE"
    fi
fi

echo ""
echo "✅ Threshold stability analysis pipeline completed!"