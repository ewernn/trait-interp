#!/bin/bash
# Monitor training completion, then transfer checkpoints and run Method B + analysis
# This runs in background - check output for progress

CKPT_DIR="/home/dev/rl-rewardhacking/results/runs/qwen3-4b/20260227_074510_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline/checkpoints"
WORKER_LOG="/tmp/ray/session_latest/logs/worker-a0ef773012edfe1eec1d63aa65a66d560b0669934eaf54a3024894d8-01000000-36040.err"
TRAIT_INTERP="/home/dev/trait-interp"

echo "[$(date)] Monitoring training progress..."

# Wait for training to complete (200 steps)
while true; do
    # Check if step 200 checkpoint exists or training process exited
    if [ -d "$CKPT_DIR/global_step_200" ]; then
        echo "[$(date)] Step 200 checkpoint found! Training complete."
        break
    fi
    
    # Check latest step from log
    latest_step=$(grep "Training Progress:" "$WORKER_LOG" 2>/dev/null | tail -1 | grep -oP '\d+(?=/200)' | head -1)
    if [ -n "$latest_step" ]; then
        echo "[$(date)] Training at step $latest_step/200"
    fi
    
    # Check if training process is still running
    if ! ps aux | grep -q "run_rl_training.py" 2>/dev/null; then
        echo "[$(date)] Training process no longer running. Checking final state..."
        sleep 5
        break
    fi
    
    sleep 60
done

# Count checkpoints
n_ckpts=$(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | wc -l)
echo "[$(date)] Found $n_ckpts checkpoints"

# Step 4: Transfer checkpoints
echo ""
echo "[$(date)] === STEP 4: Transfer checkpoints ==="
bash "$TRAIT_INTERP/experiments/aria_rl/transfer_checkpoints.sh"

echo ""
echo "[$(date)] === STEP 6: Method B scoring ==="
cd "$TRAIT_INTERP"
python experiments/aria_rl/checkpoint_method_b.py --run grpo_rh --n-samples 5

echo ""
echo "[$(date)] === STEP 8: Trajectory analysis ==="
python experiments/aria_rl/analysis_trajectories.py

echo ""
echo "[$(date)] === ALL STEPS COMPLETE ==="
