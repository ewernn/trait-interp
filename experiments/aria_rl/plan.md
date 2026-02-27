# Aria RL Reward Hacking — Fingerprinting Plan

Reproduce Aria's GRPO training (Qwen3-4B on Leetcode), save checkpoints every 10 steps, fingerprint each with our 24 trait probes via Method B. Goal: track how interpretable behavioral dimensions shift as reward hacking emerges during RL.

**Paper**: "Steering RL Training: Benchmarking Interventions Against Reward Hacking" (ariaw, Josh Engels, Neel Nanda)
**Repo**: https://github.com/ariahw/rl-rewardhacking
**Key finding**: Reward hacking emerges at ~80-100 steps. A probe monitor (80% acc) outperformed a 90% ground truth monitor — nobody knows why.

---

## Models

| Name | HF ID | Role |
|------|-------|------|
| Qwen3-4B instruct | `Qwen/Qwen3-4B` | Aria's training base, our scoring model |
| Qwen3-4B base | `Qwen/Qwen3-4B-Base` | Our probe extraction model |

Probes: 24 traits, layers 0-31, extracted on `Qwen3-4B-Base` (config variant `qwen3_4b_base`), steering validated on `Qwen3-4B` (config variant `qwen3_4b_instruct`).

---

## Aria's Training Config

```
Algorithm:          GRPO (modified Verl)
Steps:              200
Batch size:         256
Generations/prompt: 16
LoRA:               rank=32, alpha=32
Learning rate:      7e-5
Thinking mode:      OFF
Max tokens:         1,536
Hardware:           4x H200 (~3 hours)
Cost:               ~$60/run
CPU cores:          32+ physical (for code execution)
```

---

## Phase 1: H200 Cluster — GRPO Training

### 1.1 Provision
- 4x H200 80GB SXM (Vast.ai or RunPod)
- 32+ physical CPU cores
- 200GB+ disk, Ubuntu 22.04, CUDA 12.x

### 1.2 Setup

```bash
git clone https://github.com/ariahw/rl-rewardhacking.git
cd rl-rewardhacking

cp .env.template .env
# Fill: HF_TOKEN, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY, MAX_JOBS=48

cp .env.template .env.gpu
# Add: IS_GPU_ENV=true, VENV_DIR=.venv

bash setup_gpu.sh
source .env && source commands.sh
create_all_datasets
```

### 1.3 Run Training with Frequent Checkpoints

Default `save_steps=50`. We want every 10 steps (20 checkpoints).

Edit `scripts/run_rl_training.py` — add `**kwargs` to `run_no_intervention()`:
```python
def run_no_intervention(model_id=..., task=..., steps=..., seed=..., **kwargs):
    run_name = create_run_name(task=task, with_loophole=True)
    main_run_rl(run_name=run_name, task=task, model_id=model_id,
                steps=steps, seed=seed, **kwargs)
```

Then:
```bash
tmux new -s training
run_rl_training no_intervention --save_steps=10
```

~3 hours. Checkpoints at: `results/runs/qwen3-4b/{RUN_ID}/checkpoints/global_step_{N}/actor/lora_adapter/`

### 1.4 Transfer Checkpoints

```bash
RUN_DIR=$(ls -td results/runs/qwen3-4b/*/ | head -1)
mkdir -p /tmp/grpo_checkpoints
for step_dir in ${RUN_DIR}checkpoints/global_step_*/; do
    step=$(basename $step_dir | sed 's/global_step_//')
    cp -r ${step_dir}actor/lora_adapter /tmp/grpo_checkpoints/checkpoint-${step}
done
tar czf /tmp/grpo_checkpoints.tar.gz -C /tmp grpo_checkpoints/
# scp to A100
```

On A100:
```bash
mkdir -p experiments/mats-emergent-misalignment/finetune/grpo_rh
cd experiments/mats-emergent-misalignment/finetune/grpo_rh
tar xzf grpo_checkpoints.tar.gz --strip-components=1
```

---

## Phase 2: Method B Fingerprinting (A100)

Adapt `checkpoint_method_b.py` (already written for 14B EM checkpoints) for 4B GRPO checkpoints.

### What Method B does
- Generate clean text from `Qwen/Qwen3-4B` (instruct, no LoRA) on eval prompts
- **baseline**: instruct model scores clean text → 24D probe vector
- **reverse_model**: instruct + LoRA(checkpoint) scores same clean text → 24D probe vector
- **Method B** = reverse_model - baseline = pure model-internal shift

### Config
- Model: `Qwen/Qwen3-4B` at bf16 (~8GB VRAM, fits easily on A100)
- Probes: `get_best_vector(experiment, trait, extraction_variant="qwen3_4b_base", steering_variant="qwen3_4b_instruct")`
- Eval set: `em_generic_eval` or `sriram_normal`
- Checkpoint path: `finetune/grpo_rh/checkpoint-{step}/`

### Runtime
20 checkpoints × ~30s each = ~10 minutes total.

---

## Phase 3: Analysis

### Questions to answer
1. **Which traits shift first?** Rank 24 traits by onset step. Does `rm_hack/eval_awareness` or `alignment/deception` precede behavioral reward hacking onset (~step 80)?
2. **Early detection?** Does Method B detect internal shift before behavioral eval catches reward hacking?
3. **Cross-training comparison**: How does the GRPO fingerprint compare to our EM (SFT) fingerprint on 14B? Same traits or different?
4. **Decompose Aria's probe**: Project her task-specific probe direction onto our 24 trait axes — what is it actually measuring?

### Expected outputs
- 24D trait trajectory over 200 steps (main figure)
- Onset comparison: probe detection step vs behavioral eval step
- Cross-training fingerprint comparison (GRPO vs SFT)
- Top discriminative traits ranked by effect size

---

## Why This Matters

- Aria's probe (80% acc) outperformed a 90% ground truth monitor. Our 24D decomposition might explain why — her probe could be capturing a mix of deception + eval_awareness + other traits that together predict reward hacking better than a binary detector.
- If Method B detects internal shifts before behavioral eval, that's a monitoring tool for RL training.
- If the same traits (deception, eval_awareness) shift during both SFT-based EM and GRPO-based reward hacking, that's evidence of a shared mechanism across training procedures.
- Aria's Future Directions explicitly asks: "Can we use less specific probes such as deception probes to steer against reward hacking?" — we have exactly those probes.

---

## Open Questions

- Does Aria's repo save checkpoints in standard PEFT format, or does Verl use a custom format? Need to verify after cloning.
- Should we run multiple seeds (Aria emphasizes high variance across runs)?
- Do we need to run Aria's behavioral eval at each checkpoint too, or just use her published reward-hacking rate curves?
- Should we also try using our probes as a penalty monitor during training (her "Generalized Monitors" future direction)?
