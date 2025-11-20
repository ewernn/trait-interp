#!/bin/bash
# Extract all missing vectors in categorized structure
# Optimized for 8x A100 - runs ALL jobs in parallel

set -e

EXPERIMENT="gemma_2b_cognitive_nov20"
MAX_PARALLEL=38  # 8x A100 can handle all 38 jobs at once (19 traits × 2 variants)
BATCH_SIZE=8

echo "================================================================"
echo "EXTRACT ALL MISSING VECTORS - 8x A100 OPTIMIZED"
echo "================================================================"
echo "Experiment: $EXPERIMENT"
echo "Parallel jobs: $MAX_PARALLEL (ALL jobs at once!)"
echo "Batch size: $BATCH_SIZE"
echo "================================================================"
echo ""

# All traits in categorized structure
BASE_TRAITS=(
  "behavioral/refusal"
  "behavioral/compliance"
  "behavioral/sycophancy"
  "behavioral/confidence"
  "behavioral/defensiveness"
  "cognitive/retrieval"
  "cognitive/sequentiality"
  "cognitive/scope"
  "cognitive/divergence"
  "cognitive/abstractness"
  "cognitive/futurism"
  "cognitive/context"
  "stylistic/positivity"
  "stylistic/literalness"
  "stylistic/trust"
  "stylistic/authority"
  "stylistic/curiosity"
  "stylistic/enthusiasm"
  "stylistic/formality"
)

# Build list of ALL jobs (instruction + natural variants)
ALL_JOBS=()
for trait in "${BASE_TRAITS[@]}"; do
  # Instruction variant
  ALL_JOBS+=("instruction:$trait")
  # Natural variant
  ALL_JOBS+=("natural:${trait}_natural")
done

echo "Total jobs: ${#ALL_JOBS[@]} (19 traits × 2 variants)"
echo "Will run ALL in parallel on 8x A100!"
echo ""

# Function to extract single job
extract_job() {
  local job=$1
  local variant=$(echo "$job" | cut -d':' -f1)
  local trait=$(echo "$job" | cut -d':' -f2)

  local vec_dir="experiments/$EXPERIMENT/$trait/extraction/vectors"

  # Check if already complete
  if [ -d "$vec_dir" ]; then
    local count=$(find "$vec_dir" -name "*.pt" 2>/dev/null | wc -l)
    if [ $count -ge 104 ]; then
      echo "[$(date '+%H:%M:%S')] ⏭️  Skipping $trait (already complete: $count vectors)"
      return 0
    fi
  fi

  echo "[$(date '+%H:%M:%S')] 🚀 Starting: $trait"

  # Extract based on variant
  if [ "$variant" = "instruction" ]; then
    python extraction/1_generate_batched_simple.py --experiment "$EXPERIMENT" --trait "$trait" --batch-size $BATCH_SIZE 2>&1 | grep -E "✓|ERROR|COMPLETE" || true
    python extraction/2_extract_activations.py --experiment "$EXPERIMENT" --trait "$trait" 2>&1 | grep -E "✓|ERROR|COMPLETE" || true
    python extraction/3_extract_vectors.py --experiment "$EXPERIMENT" --trait "$trait" 2>&1 | grep -E "✓|ERROR|Layer" || true
  else
    python extraction/1_generate_natural.py --experiment "$EXPERIMENT" --trait "$trait" --batch-size $BATCH_SIZE 2>&1 | grep -E "✓|ERROR|COMPLETE" || true
    python extraction/2_extract_activations_natural.py --experiment "$EXPERIMENT" --trait "$trait" 2>&1 | grep -E "✓|ERROR|COMPLETE" || true
    python extraction/3_extract_vectors.py --experiment "$EXPERIMENT" --trait "$trait" 2>&1 | grep -E "✓|ERROR|Layer" || true
  fi

  # Verify completion
  if [ -d "$vec_dir" ]; then
    local final_count=$(find "$vec_dir" -name "*.pt" 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] ✅ Completed: $trait ($final_count vectors)"
  else
    echo "[$(date '+%H:%M:%S')] ❌ Failed: $trait (no vectors directory)"
  fi
}

export -f extract_job
export EXPERIMENT BATCH_SIZE

# Run ALL jobs in parallel (8x A100 can handle it!)
printf '%s\n' "${ALL_JOBS[@]}" | \
  xargs -P $MAX_PARALLEL -I {} bash -c 'extract_job "$@"' _ {}

echo ""
echo "================================================================"
echo "✅ ALL EXTRACTIONS COMPLETE"
echo "================================================================"
echo ""
echo "Final vector count:"
find "experiments/$EXPERIMENT" -path "*/extraction/vectors/*.pt" 2>/dev/null | wc -l
echo ""
echo "Expected: 3,952 vectors (19 traits × 2 variants × 104)"
echo ""
echo "Next: ./scripts/sync_push.sh to upload to R2"
