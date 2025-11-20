#!/bin/bash
# Parallel natural extraction - runs 3 traits at a time
# With progress checkpointing and auto-resume

set -e

EXPERIMENT="gemma_2b_cognitive_nov20"
BATCH_SIZE=6  # Reduced for parallel (was 8)
MAX_PARALLEL=3  # Run 3 at a time

# Source category mappings
source scripts/trait_categories.sh

# All 19 traits
ALL_TRAITS=(
  "abstractness" "confidence" "context" "divergence" "futurism"
  "retrieval" "scope" "sequentiality" "compliance" "defensiveness"
  "refusal" "sycophancy" "authority" "curiosity" "enthusiasm"
  "formality" "literalness" "positivity" "trust"
)

# Master log
MASTER_LOG="parallel_extraction_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee "$MASTER_LOG"
echo "PARALLEL NATURAL EXTRACTION" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"
echo "Traits: ${#ALL_TRAITS[@]}" | tee -a "$MASTER_LOG"
echo "Parallel jobs: $MAX_PARALLEL" | tee -a "$MASTER_LOG"
echo "Batch size: $BATCH_SIZE" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Function to run a single trait
run_trait() {
    local base_trait=$1
    local category=$(get_category "$base_trait")

    echo "[$(date +%H:%M:%S)] Starting: $base_trait (category: $category)" | tee -a "$MASTER_LOG"

    ./scripts/extract_single_trait_natural.sh \
        "$EXPERIMENT" \
        "$category" \
        "$base_trait" \
        "$BATCH_SIZE"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✅ Completed: $base_trait" | tee -a "$MASTER_LOG"
    else
        echo "[$(date +%H:%M:%S)] ❌ Failed: $base_trait" | tee -a "$MASTER_LOG"
    fi

    return $exit_code
}

# Export function so parallel can use it
export -f run_trait
export -f get_category
export EXPERIMENT BATCH_SIZE MASTER_LOG

# Run traits in parallel (3 at a time)
printf '%s\n' "${ALL_TRAITS[@]}" | \
    xargs -P $MAX_PARALLEL -I {} bash -c 'run_trait "$@"' _ {}

# Summary
echo "" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "PARALLEL EXTRACTION COMPLETE" | tee -a "$MASTER_LOG"
echo "Finished: $(date)" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"

# Count final vectors
total_vectors=$(find experiments/$EXPERIMENT/*/*_natural/extraction/vectors -name "*.pt" 2>/dev/null | wc -l)
echo "Total natural vectors: $total_vectors" | tee -a "$MASTER_LOG"
echo "Expected: ~1,976 (19 traits × 104 vectors)" | tee -a "$MASTER_LOG"
