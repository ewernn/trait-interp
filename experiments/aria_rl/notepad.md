# Aria RL Experiment Notepad

## Machine
- 4x NVIDIA H200 (143GB each, all free)
- 224 CPU cores, 2TB RAM
- Started: 2026-02-27 ~07:30 UTC

## Progress
- [x] Step 1: Set up aria_rl experiment directory — config.json, 2592 probe files, 24 steering results, 10 clean response files copied. Probes verified loading from aria_rl experiment.
- [x] Step 2: Clone and set up Aria's repo — cloned, venv created, verl+peft installed, 11 datasets created, modified run_no_intervention to accept **kwargs
- [x] Step 3: Run GRPO training — Completed 160/200 steps (crashed at step 161 due to OOM from parallel scoring attempt). Model fully converged by step 150 (frac_adv_zero=1.0, pg_loss=0, score=3.5/3.5). 16 checkpoints saved (steps 10-160). Total training time: ~1h18m.
- [x] Step 4: Transfer checkpoints — 16 checkpoints (steps 10-160) transferred. Step 161 emergency save had no adapter (skipped).
- [x] Step 5: Write Method B scoring script — experiments/aria_rl/checkpoint_method_b.py
- [x] Step 6: Run Method B scoring — 16 checkpoints scored in 2.8 min. Required CUDA_LAUNCH_BLOCKING=1 to work around async CUDA error.
- [x] Step 7: Write trajectory analysis script — experiments/aria_rl/analysis_trajectories.py (fixed onset sort bug)
- [x] Step 8: Run trajectory analysis — 6 plots + summary.json saved to analysis/trajectories/

## Key Results

### Top Model Deltas (step 160)
| Trait | Delta | Direction |
|-------|-------|-----------|
| guilt | -0.339 | Strong decrease |
| ulterior_motive | -0.188 | Decrease |
| aggression | -0.086 | Decrease |
| confusion | +0.075 | Increase |
| confidence | +0.072 | Increase |
| deception | -0.050 | Decrease |

### Onset Analysis
- **20/23 traits shift BEFORE behavioral RH onset (step 80)** — internal changes precede behavioral emergence
- Most traits start shifting at step 40 (guilt, lying, agency, confusion, aggression, hedging, eval_awareness, ulterior_motive)
- Only deception shifts at RH onset (step 80)
- warmth shifts late (step 130), obedience never shifts

### Cross-Training Comparison (GRPO 4B vs EM SFT 14B)
- **Cosine similarity: -0.367** — fingerprints are **anti-correlated**!
- Spearman ρ = -0.429 (p=0.097)
- Key disagreements: guilt, ulterior_motive, deception, sycophancy all go OPPOSITE direction
- Agreements: confusion↑, agency↓, rationalization↓, concealment↑

### L1 Norm Trajectory
- Initial: 0.045, Final: 1.103, Peak: 1.125 at step 150
- Smooth ramp-up, no discontinuity at RH onset

## Observations
- **Verl checkpoint format**: Verl saves FSDP shards AND LoRA adapters separately at `actor/lora_adapter/`. PeftModel.from_pretrained works directly.
- **Probes**: 23/23 probes load from aria_rl. Layer range 6-24. Quantization mismatch (4-bit probes, fp16 scoring) — deltas cancel constant offsets.
- **vLLM memory**: vLLM uses cumem allocator that claims ~143GB peak during rollout, invisible to nvidia-smi steady-state (~25GB). Cannot share GPU with other processes.
- **GRPO converges quickly**: By step 150, model hits max reward (3.5), zero loss, zero gradient. Last 40 steps would have been no-ops.
- **Anti-correlation with EM SFT**: GRPO reward hacking produces opposite trait shifts from EM emergent misalignment on key traits (guilt, ulterior_motive, deception). Different training → different internal signature.
- **Training data**: Leetcode medium/hard problems (992 prompts) from Aria's `rl-rewardhacking` repo. Task variant `simple_overwrite_tests` adds loophole where model can overwrite `run_tests()` for max reward. Repo cloned to `/home/dev/rl-rewardhacking/`.
- **Storage**: FSDP checkpoint shards at `/home/dev/rl-rewardhacking/results/` = 500GB (deletable). LoRA adapters at `finetune/grpo_rh/` = 4GB (needed). Run ID: `20260227_074510_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline`.
