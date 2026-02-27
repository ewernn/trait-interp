# Experiment: Aria RL Reward Hacking — Trait Fingerprinting

## Goal
Track how 23 interpretable behavioral trait dimensions shift during GRPO RL training as reward hacking emerges, using Method B fingerprinting on checkpoints every 10 steps.

## Hypothesis
- Traits like `rm_hack/eval_awareness`, `alignment/deception`, and `bs/lying` will shift before behavioral reward hacking onset (~step 80-100)
- The GRPO reward hacking fingerprint will share some traits with the SFT emergent misalignment (EM) fingerprint (deception↑, lying↑) but differ in others
- Method B will detect internal shifts earlier than behavioral eval catches reward hacking

## Success Criteria
- [ ] GRPO training completes with checkpoints every 10 steps (20 checkpoints over 200 steps)
- [ ] Method B scores computed for all checkpoints (23 traits × 20 checkpoints)
- [ ] Trait trajectory plot showing per-trait model_delta over training steps
- [ ] Onset analysis: which traits shift first, and when relative to behavioral RH onset (~step 80)
- [ ] Cross-training comparison: GRPO fingerprint vs EM SFT fingerprint

## Prerequisites
- [x] 4x H200 GPUs available
- [x] HF_TOKEN, WANDB_API_KEY in `.env`
- [x] 23 trait probes extracted on `qwen3_4b_base`, steering validated on `qwen3_4b_instruct` (in `experiments/mats-emergent-misalignment/`)
- [x] Pre-generated clean instruct responses for 4B in `analysis/pxs_grid_4b/responses/`
- [ ] Aria's repo cloned and set up
- [ ] GRPO training checkpoints produced

## Steps

### Step 1: Set up aria_rl experiment directory
**Purpose**: Create config.json and copy over required probes/responses from mats-emergent-misalignment.

**1a: Create config.json**
```json
{
  "defaults": {
    "extraction": "qwen3_4b_base",
    "application": "qwen3_4b_instruct"
  },
  "model_variants": {
    "qwen3_4b_base": {
      "model": "Qwen/Qwen3-4B-Base"
    },
    "qwen3_4b_instruct": {
      "model": "Qwen/Qwen3-4B"
    }
  }
}
```

Note: For Method B scoring, we load the full-precision `Qwen/Qwen3-4B` (not quantized unsloth). The quantized models were only used for extraction/steering (memory constrained). Method B needs full precision for LoRA application via PEFT.

**1b: Copy probes and steering results**
```bash
# Copy extraction vectors (probes)
for trait_dir in experiments/mats-emergent-misalignment/extraction/*/; do
    for trait in "$trait_dir"*/; do
        trait_name=$(basename "$trait")
        category=$(basename "$trait_dir")
        src="$trait/qwen3_4b_base"
        if [ -d "$src" ]; then
            dest="experiments/aria_rl/extraction/$category/$trait_name/qwen3_4b_base"
            mkdir -p "$dest"
            cp -r "$src/vectors" "$dest/"
        fi
    done
done

# Copy steering results
for trait_dir in experiments/mats-emergent-misalignment/steering/*/; do
    for trait in "$trait_dir"*/; do
        trait_name=$(basename "$trait")
        category=$(basename "$trait_dir")
        src="$trait/qwen3_4b_instruct"
        if [ -d "$src" ]; then
            dest="experiments/aria_rl/steering/$category/$trait_name/qwen3_4b_instruct"
            mkdir -p "$dest"
            cp -r "$src/"* "$dest/"
        fi
    done
done
```

**1c: Copy clean instruct responses**
```bash
mkdir -p experiments/aria_rl/analysis/method_b/responses
cp experiments/mats-emergent-misalignment/analysis/pxs_grid_4b/responses/clean_instruct_x_*.json \
   experiments/aria_rl/analysis/method_b/responses/
```

**Verify:**
```bash
# Should show 23 trait directories with vectors
find experiments/aria_rl/extraction -name "*.pt" | wc -l
# Should show steering results
find experiments/aria_rl/steering -name "results.jsonl" | wc -l
# Should show clean responses
ls experiments/aria_rl/analysis/method_b/responses/ | wc -l
```

### Step 2: Clone and set up Aria's repo
**Purpose**: Get the GRPO training infrastructure ready.

```bash
cd /home/dev
git clone https://github.com/ariahw/rl-rewardhacking.git
cd rl-rewardhacking

# Create .env from template
cp .env.template .env
```

Edit `.env` with credentials from our `.env`:
- `HF_TOKEN` — copy from trait-interp/.env
- `WANDB_API_KEY` — copy from trait-interp/.env
- `WANDB_PROJECT=aria-rl-fingerprint`
- `WANDB_ENTITY=` (leave blank or set if needed)
- `MAX_JOBS=48` (we have plenty of CPU cores)

```bash
# GPU setup
bash setup_gpu.sh
source .env && source commands.sh

# Create datasets
create_all_datasets
```

**Verify:**
```bash
python -c "import verl; print('verl ok')"
python -c "from peft import PeftModel; print('peft ok')"
ls datasets/  # Should show leetcode datasets
```

### Step 3: Run GRPO training with frequent checkpoints
**Purpose**: Train Qwen3-4B on Leetcode with overwrite-tests loophole, saving every 10 steps.

First, modify `scripts/run_rl_training.py` to pass `**kwargs` through to `run_no_intervention()`:
```python
# Add **kwargs to signature and pass save_steps through
def run_no_intervention(model_id=..., task=..., steps=..., seed=..., **kwargs):
    run_name = create_run_name(task=task, with_loophole=True)
    main_run_rl(run_name=run_name, task=task, model_id=model_id,
                steps=steps, seed=seed, **kwargs)
```

Then run:
```bash
python scripts/run_rl_training.py no_intervention --save_steps=10
```

Expected: ~3 hours, 20 checkpoints at steps 10, 20, ..., 200.
Checkpoints at: `results/runs/qwen3-4b/{RUN_ID}/checkpoints/global_step_{N}/actor/lora_adapter/`

**Verify:**
```bash
RUN_DIR=$(ls -td results/runs/qwen3-4b/*/ | head -1)
ls ${RUN_DIR}checkpoints/ | wc -l  # Should be ~20
ls ${RUN_DIR}checkpoints/global_step_10/actor/lora_adapter/  # Should have adapter_config.json
```

### Step 4: Convert and transfer checkpoints
**Purpose**: Convert Verl FSDP-sharded checkpoints to usable format.

**CRITICAL**: Verl saves FSDP-sharded checkpoints, NOT standard PEFT adapters. `save_adapter()` is a no-op in Aria's Verl wrapper. Need to either:
- (a) Use `verl.model_merger` to merge each checkpoint into HF-compatible format
- (b) If `actor/lora_adapter/` exists with `adapter_config.json`, use directly (Aria may have custom saving)
- (c) Load full merged model per checkpoint (4B is small, ~10s per load)

**Auto-detect approach:**
```bash
cd /home/dev/rl-rewardhacking
RUN_DIR=$(ls -td results/runs/qwen3-4b/*/ | head -1)

# Check what's in the first checkpoint
ls ${RUN_DIR}checkpoints/global_step_10/

# Case A: Standard PEFT adapter exists
if [ -f "${RUN_DIR}checkpoints/global_step_10/actor/lora_adapter/adapter_config.json" ]; then
    echo "PEFT format detected — copy adapters directly"
    mkdir -p /home/dev/trait-interp/experiments/aria_rl/finetune/grpo_rh
    for step_dir in ${RUN_DIR}checkpoints/global_step_*/; do
        step=$(basename $step_dir | sed 's/global_step_//')
        cp -r ${step_dir}actor/lora_adapter /home/dev/trait-interp/experiments/aria_rl/finetune/grpo_rh/checkpoint-${step}
    done

# Case B: FSDP shards — merge each checkpoint
else
    echo "FSDP format detected — merging checkpoints"
    for step_dir in ${RUN_DIR}checkpoints/global_step_*/; do
        step=$(basename $step_dir | sed 's/global_step_//')
        target=/home/dev/trait-interp/experiments/aria_rl/finetune/grpo_rh/checkpoint-${step}
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir ${step_dir}actor \
            --target_dir ${target}
    done
fi
```

**Verify:**
```bash
ls experiments/aria_rl/finetune/grpo_rh/ | sort -t'-' -k2 -n
# Check format: either adapter_config.json (PEFT) or config.json + model.safetensors (full)
ls experiments/aria_rl/finetune/grpo_rh/checkpoint-10/
```

### Step 5: Write Method B scoring script
**Purpose**: Adapt checkpoint_method_b.py for 4B GRPO checkpoints.

Create `experiments/aria_rl/checkpoint_method_b.py` based on the mats version, with these changes:
- `EXPERIMENT = "aria_rl"`
- `MODEL = "Qwen/Qwen3-4B"` (full precision instruct, not quantized)
- `RESPONSES_DIR = BASE_DIR / "analysis" / "method_b" / "responses"`
- `extraction_variant="qwen3_4b_base"`, `steering_variant="qwen3_4b_instruct"`
- Default `--run grpo_rh`
- Default eval sets: `["sriram_normal", "sriram_factual", "em_generic_eval", "em_medical_eval"]`
- Add `enable_thinking=False` to all `apply_chat_template` calls (already in template)
- Output to: `experiments/aria_rl/analysis/method_b/grpo_rh.json`
- **Handle both checkpoint formats**: Auto-detect PEFT adapter vs full merged model.
  If `adapter_config.json` exists → use PeftModel.from_pretrained (fast, reuse base)
  If `config.json` + safetensors exists → load full model with AutoModelForCausalLM (slower but works)

**Verify:**
```bash
python experiments/aria_rl/checkpoint_method_b.py --help
# Should show args with defaults
```

### Step 6: Run Method B scoring
**Purpose**: Score all checkpoints.

```bash
python experiments/aria_rl/checkpoint_method_b.py --run grpo_rh --n-samples 5
```

Expected: ~10-15 minutes for 20 checkpoints. ~30s per checkpoint (4B model is small).

**Verify:**
```bash
python3 -c "
import json
with open('experiments/aria_rl/analysis/method_b/grpo_rh.json') as f:
    data = json.load(f)
print(f'Checkpoints: {len(data[\"checkpoints\"])}')
print(f'Probes: {data[\"metadata\"][\"n_probes\"]}')
for c in data['checkpoints'][:3]:
    print(f'  Step {c[\"step\"]}: top delta = {max(c[\"model_delta\"].items(), key=lambda x: abs(x[1]))}')
"
```

### Step 7: Write trajectory analysis script
**Purpose**: Visualize trait shifts over GRPO training.

Create `experiments/aria_rl/analysis_trajectories.py` that produces:

1. **Trait trajectory plot** (main figure): 23 traits × 200 steps, showing model_delta evolution. Highlight onset region (~step 80-100).
2. **Top movers bar chart**: Rank traits by final |model_delta|.
3. **Onset analysis**: For each trait, find the first step where |model_delta| exceeds 2σ of baseline noise. Compare to behavioral RH onset (~step 80).
4. **Fingerprint heatmap**: Steps (rows) × traits (columns), showing the full trajectory.
5. **Cross-training comparison**: Load EM rank32 fingerprint from mats experiment, compute cosine similarity with GRPO final fingerprint.

**Output**: `experiments/aria_rl/analysis/trajectories/*.png` + `summary.json`

### Step 8: Run trajectory analysis
**Purpose**: Generate all plots and summary statistics.

```bash
python experiments/aria_rl/analysis_trajectories.py
```

**Verify:**
```bash
ls experiments/aria_rl/analysis/trajectories/
# Should show: trait_trajectories.png, top_movers.png, onset_analysis.png, fingerprint_heatmap.png, cross_training.png, summary.json
```

### Checkpoint: After Step 8
Stop and verify:
- Do trait trajectories show meaningful shifts during training?
- Which traits shift first? Before or after step 80?
- Does the GRPO fingerprint look different from EM?
- Are there any probes with near-zero signal (might indicate probe doesn't transfer well)?

## Expected Results

| Metric | Expected | Would indicate failure |
|--------|----------|----------------------|
| Checkpoints produced | 20 (steps 10-200) | < 15 = training issues |
| Probes loaded | 23/23 | < 20 = probe loading issues |
| Traits with |model_delta| > 0.01 at step 200 | 10+ | < 5 = probes don't detect GRPO changes |
| Earliest trait onset | Before step 80 | After step 100 = no early detection |
| Cosine sim GRPO vs EM fingerprint | 0.3-0.7 (partial overlap) | > 0.9 = suspiciously similar, < 0.1 = no shared mechanism |

## If Stuck
- **Verl install fails** → Try `pip install verl==0.6.1` directly, or build from bundled `/verl` directory
- **Checkpoint format mismatch** → Check if Verl saves full model or just LoRA. Look for `adapter_config.json` vs `model.safetensors`
- **CUDA OOM during training** → Reduce batch size or generations_per_prompt
- **Probes show no signal** → Check that probe layers are valid for 4B (max layer 35). Our probes use layers 6-24, all valid.
- **Training doesn't converge / no reward hacking** → This is seed-dependent. Aria reports RH in all observed runs but at variable onset. If step 200 has no RH, the fingerprint may still be interesting (what does the model learn before discovering the loophole?)

## Notes
(Space for observations during run)
