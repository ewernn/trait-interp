# Overnight Training Notepad (March 6-7, 2026)

## Goal
Train Aria's `no_intervention` (RH) adapters for seeds 1, 42, 65 locally to get per-step checkpoints (global_step_1 through global_step_200). Published HF adapters only have final checkpoints — we need the full training trajectory for trait dynamics across RL training.

## Setup

### Machine
- 2x NVIDIA A100-SXM4-80GB
- 64 physical CPU cores (128 threads)
- Repo: ~/rl-rewardhacking (cloned from github.com/ariahw/rl-rewardhacking)

### Training Config
- Model: Qwen3-4B (qwen/Qwen3-4B)
- Method: GRPO (Verl v0.6.1)
- LoRA rank 32, alpha 32, all linear modules (q,k,v,o,gate,up,down_proj)
- 200 steps, 16 prompts × 16 generations = 256 rollouts/step
- LR: 7e-5, cosine schedule, 10 warmup steps
- KL coeff (beta): 1e-3
- Task: simple_overwrite_tests (leetcode + test-overwriting loophole)
- Dataset: results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl (119 problems)

### Optimizations Applied
- `gpu_memory_utilization=0.92` (default 0.85) — more VRAM for vLLM KV cache
- `save_steps=1` — checkpoint every step (~99MB each, ~20GB per seed)
- `save_only_model=True` (default) — LoRA adapter only, no optimizer state
- **Parallel seeds**: 2 seeds run simultaneously, one per GPU (CUDA_VISIBLE_DEVICES isolation)
- Patched `grpo.py` odd-GPU check to allow single-GPU training (n_gpus > 1 guard)
- MAX_JOBS=48 (60% of 64 physical cores, for code eval subprocess pool)

### Checkpoint Math
- 99 MB per checkpoint × 200 steps × 3 seeds = ~60 GB total
- Write time per checkpoint: ~0.2s (negligible)
- Output: ~/rl-rewardhacking/results/runs/qwen3-4b/<run_id>/checkpoints/global_step_{1..200}/

## Execution Plan
1. Fill .env (HF_TOKEN, WANDB keys)
2. `source setup.sh` (install deps, verl)
3. `create_all_datasets` (create loopholed datasets, ~5 min)
4. Run `train_seeds.sh`:
   - Seeds 1+42 in parallel (GPU 0 + GPU 1)
   - Seed 65 starts when first finishes
   - Estimated: ~5-7 hours wall clock

## Script
~/rl-rewardhacking/train_seeds.sh

## Code Changes
- ~/rl-rewardhacking/src/train/verl/grpo.py: guarded odd-GPU check with `n_gpus > 1`

## Status
- [ ] .env configured
- [ ] Dependencies installed (uv sync, verl)
- [ ] Datasets created (create_all_datasets)
- [ ] Seed 1 training
- [ ] Seed 42 training
- [ ] Seed 65 training
- [ ] Verify checkpoints match Aria's published adapters at step 200

## Post-Training Analysis Plan
- Per-step trait fingerprinting: project each checkpoint onto 152 traits
- Training trajectory: how do traits evolve across 200 RL steps?
- When does reward hacking behavior emerge? (behavioral eval at each checkpoint)
- Cross-seed: do all 3 seeds follow same trait trajectory?
- Compare to probe-penalty / gt-penalty trajectories (if we train those later)
