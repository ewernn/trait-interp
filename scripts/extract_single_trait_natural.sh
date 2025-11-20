#!/bin/bash
# Extract a single natural trait with progress checkpointing
# Usage: ./extract_single_trait_natural.sh <experiment> <category> <trait> <batch_size>

set -e

EXPERIMENT=$1
CATEGORY=$2
TRAIT=$3
BATCH_SIZE=${4:-8}

trait_natural="${TRAIT}_natural"
trait_path="$CATEGORY/$trait_natural"

# Progress file
PROGRESS_DIR="experiments/$EXPERIMENT/.progress"
mkdir -p "$PROGRESS_DIR"
PROGRESS_FILE="$PROGRESS_DIR/${trait_natural}_progress.txt"

# Log file
LOG_FILE="extraction_${trait_natural}_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee -a "$LOG_FILE"
echo "EXTRACTING: $trait_natural (category: $CATEGORY)" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# Check if already completed
if [ -f "$PROGRESS_FILE" ] && grep -q "COMPLETED" "$PROGRESS_FILE"; then
    echo "✓ Already completed (found checkpoint)" | tee -a "$LOG_FILE"
    exit 0
fi

# Step 1: Generate responses
if ! grep -q "GENERATION_DONE" "$PROGRESS_FILE" 2>/dev/null; then
    echo "  [1/3] Generating responses..." | tee -a "$LOG_FILE"
    python extraction/1_generate_natural.py \
        --experiment "$EXPERIMENT" \
        --trait "$trait_path" \
        --batch-size $BATCH_SIZE \
        >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "GENERATION_DONE $(date)" >> "$PROGRESS_FILE"
        echo "    ✓ Generation complete" | tee -a "$LOG_FILE"
    else
        echo "    ❌ Generation failed" | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "  [1/3] ✓ Generation already done (resuming)" | tee -a "$LOG_FILE"
fi

# Step 2: Extract activations
if ! grep -q "ACTIVATIONS_DONE" "$PROGRESS_FILE" 2>/dev/null; then
    echo "  [2/3] Extracting activations..." | tee -a "$LOG_FILE"
    python extraction/2_extract_activations_natural.py \
        --experiment "$EXPERIMENT" \
        --trait "$trait_path" \
        >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "ACTIVATIONS_DONE $(date)" >> "$PROGRESS_FILE"
        echo "    ✓ Activations complete" | tee -a "$LOG_FILE"
    else
        echo "    ❌ Activations failed" | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "  [2/3] ✓ Activations already done (resuming)" | tee -a "$LOG_FILE"
fi

# Step 3: Extract vectors
if ! grep -q "VECTORS_DONE" "$PROGRESS_FILE" 2>/dev/null; then
    echo "  [3/3] Extracting vectors..." | tee -a "$LOG_FILE"
    python extraction/3_extract_vectors.py \
        --experiment "$EXPERIMENT" \
        --trait "$trait_path" \
        >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "VECTORS_DONE $(date)" >> "$PROGRESS_FILE"
        echo "    ✓ Vectors complete" | tee -a "$LOG_FILE"
    else
        echo "    ❌ Vectors failed" | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "  [3/3] ✓ Vectors already done (resuming)" | tee -a "$LOG_FILE"
fi

# Mark as completed
echo "COMPLETED $(date)" >> "$PROGRESS_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "✅ COMPLETE: $trait_natural" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
