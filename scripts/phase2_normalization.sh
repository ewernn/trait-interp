#!/bin/bash
# Phase 2: Re-run scoring pipelines with normalization
# A100 80GB — Qwen2.5-14B-Instruct fits in bf16
set -e

cd /home/dev/trait-interp
LOG="phase2_normalization.log"
echo "=== Phase 2 started at $(date) ===" | tee "$LOG"

# ---- Step 0: Backup existing data ----
echo "" | tee -a "$LOG"
echo "=== Step 0: Backing up existing probe scores ===" | tee -a "$LOG"

BACKUP_DIR="analysis/backup_pre_normalization_$(date +%Y%m%d_%H%M%S)"
mkdir -p "experiments/mats-emergent-misalignment/$BACKUP_DIR"

cp -r experiments/mats-emergent-misalignment/analysis/pxs_grid_14b/probe_scores \
      "experiments/mats-emergent-misalignment/$BACKUP_DIR/pxs_grid_14b_probe_scores" 2>/dev/null || true

mkdir -p "experiments/mats-emergent-misalignment/$BACKUP_DIR/checkpoint_method_b"
for f in experiments/mats-emergent-misalignment/analysis/checkpoint_method_b/*.json; do
    cp "$f" "experiments/mats-emergent-misalignment/$BACKUP_DIR/checkpoint_method_b/" 2>/dev/null || true
done

cp -r experiments/mats-emergent-misalignment/analysis/pxs_grid_4b/probe_scores \
      "experiments/mats-emergent-misalignment/$BACKUP_DIR/pxs_grid_4b_probe_scores" 2>/dev/null || true

cp experiments/aria_rl/analysis/method_b/*.json \
   "experiments/mats-emergent-misalignment/$BACKUP_DIR/" 2>/dev/null || true

echo "Backed up to $BACKUP_DIR" | tee -a "$LOG"

# ---- Step 1: Re-run pxs_grid 14B ----
echo "" | tee -a "$LOG"
echo "=== Step 1: pxs_grid 14B (5 variants × 10 eval sets × 3 score types) ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

python3 experiments/mats-emergent-misalignment/pxs_grid.py \
    --from-config --eval-sets all --probes-only 2>&1 | tee -a "$LOG"

echo "Step 1 done at $(date)" | tee -a "$LOG"

# ---- Step 2: checkpoint_method_b for runs with local checkpoints ----
echo "" | tee -a "$LOG"
echo "=== Step 2: checkpoint_method_b (5 local runs) ===" | tee -a "$LOG"

for RUN in rank32 rank1 angry_refusal curt_refusal mocking_refusal; do
    echo "" | tee -a "$LOG"
    echo "--- checkpoint_method_b: $RUN ---" | tee -a "$LOG"
    echo "Start: $(date)" | tee -a "$LOG"

    python3 experiments/mats-emergent-misalignment/checkpoint_method_b.py \
        --run "$RUN" 2>&1 | tee -a "$LOG"

    echo "$RUN done at $(date)" | tee -a "$LOG"
done

# ---- Step 3: Post-hoc normalize data we can't re-run ----
echo "" | tee -a "$LOG"
echo "=== Step 3: Post-hoc normalize remaining data ===" | tee -a "$LOG"

python3 experiments/mats-emergent-misalignment/normalize_existing_scores.py \
    --all 2>&1 | tee -a "$LOG"

echo "Step 3 done at $(date)" | tee -a "$LOG"

# ---- Step 4: Push results to R2 ----
echo "" | tee -a "$LOG"
echo "=== Step 4: Push to R2 ===" | tee -a "$LOG"

./utils/r2_push.sh --copy mats-emergent-misalignment 2>&1 | tee -a "$LOG"
./utils/r2_push.sh --copy aria_rl 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Phase 2 complete at $(date) ===" | tee -a "$LOG"
echo "Next: pull data locally, run Phase 3 (analysis script updates)"
