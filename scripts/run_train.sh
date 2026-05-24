#!/bin/bash
# run_train.sh — CASTLE Training Script
# =========================================
# Reproduces paper results (Table 9):
#   F1=0.9629, BLEU=92.72 on IGED test set
#
# Hardware: NVIDIA RTX 3090 (~10h full training)
# Kaggle/Colab: Reduce batch_size to 32, fp16=true
# =========================================

set -e

# ── Configuration ─────────────────────────────
CONFIG="configs/castle_base.yaml"
LOCAL_CSV=""           # Set to path if not using HuggingFace Hub
SAVE_DIR="checkpoints/castle"
LOG_DIR="runs/castle"

# ── Step 1: Build Knowledge Graph (run once) ──
# Requires IGED CSV and optionally DeepSeek API key
if [ ! -f "data/castle_kg.pkl" ]; then
    echo "=== Building Knowledge Graph ==="
    CSV_PATH="${LOCAL_CSV:-data/IGED_stratified_dataset.csv}"

    if [ -n "$DEEPSEEK_API_KEY" ]; then
        # Full construction with neural relations (recommended)
        python scripts/build_kg.py \
            --csv "$CSV_PATH" \
            --output data/castle_kg.pkl \
            --deepseek_key "$DEEPSEEK_API_KEY" \
            --max_neural 100000 \
            --cache_dir data/kg_cache
    else
        echo "DEEPSEEK_API_KEY not set — building rule-based KG only"
        python scripts/build_kg.py \
            --csv "$CSV_PATH" \
            --output data/castle_kg.pkl
    fi
    echo "KG built successfully!"
else
    echo "KG found at data/castle_kg.pkl — skipping build"
fi

# ── Step 2: Train CASTLE ──────────────────────
echo ""
echo "=== Training CASTLE ==="

EXTRA_ARGS=""
if [ -n "$LOCAL_CSV" ]; then
    EXTRA_ARGS="--local_csv $LOCAL_CSV"
fi

python src/train.py \
    --config "$CONFIG" \
    --save_dir "$SAVE_DIR" \
    $EXTRA_ARGS

echo ""
echo "=== Training Complete ==="
echo "Best model: $SAVE_DIR/checkpoint_best.pt"

# ── Step 3: Evaluate ─────────────────────────
echo ""
echo "=== Evaluating on Test Set ==="
python src/evaluate.py \
    --checkpoint "$SAVE_DIR/checkpoint_best.pt" \
    --config "$CONFIG" \
    --output "$SAVE_DIR/test_results.json" \
    $EXTRA_ARGS

echo ""
echo "All done! Results saved to $SAVE_DIR/test_results.json"
