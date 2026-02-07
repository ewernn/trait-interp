# Experiment: Latent Diff Drift

## Goal

Show that independently-extracted concealment/lying vectors detect a deception backdoor's activation signature — even on benign prompts where the model isn't lying.

## Hypothesis

Soft-trigger LoRAs shift model activations toward the concealment/lying direction at all times, not just when triggered. This "latent" signal is detectable via cosine similarity between the activation diff vector and trait vectors, even on benign prompts where both models produce identical outputs.

## Background

Cross-eval (already done) shows bs/concealment and bs/lying vectors detect active backdoor deception:

| Dataset | bs/concealment | bs/lying |
|---------|---------------|----------|
| ST_TIME | 0.944 | **0.953** |
| ST_GREETING | 0.906 | **0.934** |
| ST_GENDER | 0.519 | 0.531 |

This experiment extends to model-level detection: can we detect the backdoor from how the model *represents* any input, not just from classifying individual deceptive responses?

## Success Criteria

- [ ] Eval prompts: cosine(diff_vector, concealment_vector) > 0.1 at peak layer for st_time LoRA
- [ ] Eval prompts: Cohen's d > 0.5 on concealment projection (clean vs LoRA)
- [ ] Benign prompts: cosine(diff_vector, concealment_vector) > 0.05 (any detectable signal)
- [ ] Specificity: concealment/lying cosine > other extracted traits' cosine

## Prerequisites

- [x] bs/concealment and bs/lying vectors at `extraction/bs/{concealment,lying}/base/vectors/response__5/residual/`
- [x] Soft-trigger activations on remote instance (for cross-eval, already done)
- [x] Alpaca control prompts at `prompt_sets/alpaca_control.json`
- [x] Remote instance: 2x A100 80GB, ssh -p 47115 root@207.180.148.74
- [x] ~360 GB free disk space

## Storage Budget

With `--response-only --layers '0-75:5'` (16 layers, response tokens only):
- Per soft-trigger prompt: 1 token × 16 layers × 8192 dim × 4 bytes × 2 components ≈ **1 MB**
- Per alpaca prompt: ~120 tokens × 16 × 8192 × 4 × 2 ≈ **120 MB**

| Capture | Prompts | Size |
|---------|---------|------|
| instruct / st_time_all | 50 | ~50 MB |
| lora_time / st_time_all (replay) | 50 | ~50 MB |
| instruct / alpaca_control | 50 | ~6 GB |
| lora_time / alpaca_control (replay) | 50 | ~6 GB |
| **Total** | | **~12 GB** |

Negligible compared to 362 GB free. Can easily increase --limit or add more LoRAs.

---

## Steps

### Step 1: Capture clean model on st_time eval prompts

**Purpose**: Generate responses and capture activations from clean instruct model on soft-trigger prompts.

**Command**:
```bash
python inference/capture_raw_activations.py \
    --experiment bullshit \
    --model-variant instruct \
    --prompts-file experiments/bullshit/prompt_sets/st_time_all.json \
    --prompt-set-name st_time_all \
    --limit 50 \
    --max-new-tokens 5 \
    --layers '0-75:5' \
    --response-only \
    --no-server
```

**Output**: `experiments/bullshit/inference/instruct/raw/residual/st_time_all/{1..50}.pt`

**Verify**:
```bash
ls experiments/bullshit/inference/instruct/raw/residual/st_time_all/ | wc -l  # Should be 50
du -sh experiments/bullshit/inference/instruct/raw/residual/st_time_all/  # Should be ~25 MB total
```

### Step 2: Replay same tokens through st_time LoRA

**Purpose**: Prefill identical prompt+response through LoRA variant. Same tokens, different model.

**Command**:
```bash
python inference/capture_raw_activations.py \
    --experiment bullshit \
    --model-variant lora_time \
    --prompts-file experiments/bullshit/prompt_sets/st_time_all.json \
    --prompt-set-name st_time_all \
    --replay-responses st_time_all \
    --replay-from-variant instruct \
    --limit 50 \
    --layers '0-75:5' \
    --response-only \
    --no-server
```

**Output**: `experiments/bullshit/inference/lora_time/raw/residual/st_time_all/{1..50}.pt`

**Verify**:
```bash
ls experiments/bullshit/inference/lora_time/raw/residual/st_time_all/ | wc -l  # Should be 50
```

### Checkpoint A: Eval prompt activations captured

Stop and verify:
- Both directories have 50 .pt files each
- File sizes are comparable (same tokens, different model)
- Quick sanity: check disk usage, verify no OOM or corrupt files

```bash
du -sh experiments/bullshit/inference/*/raw/residual/st_time_all/
df -h /  # Check remaining disk space
```

### Step 3: Model diff on eval prompts

**Purpose**: Compare activations, project diff onto concealment/lying vectors. This tests: does the LoRA shift representations toward concealment?

**Command**:
```bash
python analysis/model_diff/compare_variants.py \
    --experiment bullshit \
    --variant-a instruct \
    --variant-b lora_time \
    --prompt-set st_time_all \
    --traits bs/concealment,bs/lying \
    --method probe \
    --position "response[:5]"
```

**Output**: `experiments/bullshit/model_diff/instruct_vs_lora_time/st_time_all/`
- `diff_vectors.pt` — [80, 8192] mean activation difference per layer
- `results.json` — per-trait cosine similarity and Cohen's d across layers

**Key metrics to check**:
- `per_layer_cosine_sim` for bs/concealment — high = diff aligns with concealment
- `per_layer_cosine_sim` for bs/lying — compare to concealment
- `per_layer_effect_size` — Cohen's d magnitude
- `peak_layer` — where does the signal peak?

**Verify**:
```bash
python -c "
import json
r = json.load(open('experiments/bullshit/model_diff/instruct_vs_lora_time/st_time_all/results.json'))
for trait, data in r['traits'].items():
    peak = data.get('peak_layer', '?')
    peak_d = data.get('peak_effect_size', '?')
    cos = data['per_layer_cosine_sim']
    peak_cos_idx = max(range(len(cos)), key=lambda i: abs(cos[i]))
    peak_cos = cos[peak_cos_idx]
    print(f'{trait}: peak_d=L{peak} ({peak_d:+.2f}σ), peak_cos=L{data[\"layers\"][peak_cos_idx]} ({peak_cos:+.3f})')
"
```

### Checkpoint B: Eval prompt model diff results

**STOP HERE.** Review results before proceeding to benign prompts.

**What to look for**:
- Cosine similarity > 0.1 for concealment/lying = the diff aligns with deception direction
- Cohen's d > 0.5 = meaningful effect size
- If both are near zero, the LoRA doesn't shift representations toward concealment on eval prompts — reconsider approach before running benign prompts

**If promising, continue. If not, discuss before spending GPU time on Steps 4-6.**

---

### Step 4: Capture clean model on alpaca (benign) prompts

**Purpose**: Baseline activations on prompts with no triggers.

**Command**:
```bash
python inference/capture_raw_activations.py \
    --experiment bullshit \
    --model-variant instruct \
    --prompts-file experiments/bullshit/prompt_sets/alpaca_control.json \
    --prompt-set-name alpaca_control \
    --limit 50 \
    --layers '0-75:5' \
    --response-only \
    --no-server
```

**Output**: `experiments/bullshit/inference/instruct/raw/residual/alpaca_control/{1..50}.pt`

### Step 5: Replay alpaca through st_time LoRA

**Purpose**: Same benign tokens through the backdoored model. No trigger = model shouldn't be lying.

**Command**:
```bash
python inference/capture_raw_activations.py \
    --experiment bullshit \
    --model-variant lora_time \
    --prompts-file experiments/bullshit/prompt_sets/alpaca_control.json \
    --prompt-set-name alpaca_control \
    --replay-responses alpaca_control \
    --replay-from-variant instruct \
    --limit 50 \
    --layers '0-75:5' \
    --response-only \
    --no-server
```

### Step 6: Model diff on benign prompts (THE REAL TEST)

**Purpose**: Does the LoRA show elevated concealment signal on prompts where it's NOT lying?

**Command**:
```bash
python analysis/model_diff/compare_variants.py \
    --experiment bullshit \
    --variant-a instruct \
    --variant-b lora_time \
    --prompt-set alpaca_control \
    --traits bs/concealment,bs/lying \
    --method probe \
    --position "response[:5]"
```

**Key comparison**:
- Eval cosine vs benign cosine — does the signal persist?
- Eval Cohen's d vs benign Cohen's d — how much does magnitude drop?
- If benign cosine > 0 but much smaller than eval, that's still a finding (latent but weaker)

### Checkpoint C: Benign prompt model diff results

**STOP HERE.** Compare eval vs benign results side by side.

```bash
python -c "
import json
for ps in ['st_time_all', 'alpaca_control']:
    print(f'\n=== {ps} ===')
    path = f'experiments/bullshit/model_diff/instruct_vs_lora_time/{ps}/results.json'
    r = json.load(open(path))
    for trait, data in r['traits'].items():
        cos = data['per_layer_cosine_sim']
        peak_cos_idx = max(range(len(cos)), key=lambda i: abs(cos[i]))
        peak_cos = cos[peak_cos_idx]
        peak_d = data.get('peak_effect_size', 'N/A')
        print(f'  {trait}: peak_cos={peak_cos:+.3f} (L{data[\"layers\"][peak_cos_idx]}), peak_d={peak_d}')
"
```

---

### Step 7 (if promising): Expand to other LoRAs

Repeat Steps 1-6 for lora_greeting and lora_gender:

```bash
# lora_greeting
python inference/capture_raw_activations.py --experiment bullshit --model-variant instruct \
    --prompts-file experiments/bullshit/prompt_sets/st_greeting_all.json \
    --prompt-set-name st_greeting_all --limit 50 --load-in-8bit

python inference/capture_raw_activations.py --experiment bullshit --model-variant lora_greeting \
    --prompts-file experiments/bullshit/prompt_sets/st_greeting_all.json \
    --prompt-set-name st_greeting_all --replay-responses st_greeting_all \
    --replay-from-variant instruct --limit 50 --load-in-8bit

python analysis/model_diff/compare_variants.py --experiment bullshit \
    --variant-a instruct --variant-b lora_greeting --prompt-set st_greeting_all \
    --traits bs/concealment,bs/lying --method probe --position "response[:5]"

# lora_gender (expected to show weak/no signal, consistent with cross-eval)
# Same pattern with st_gender_all and lora_gender
```

**Interesting comparison**: Does lora_gender show latent concealment signal on benign prompts even though it failed cross-eval (0.52 AUROC)? If yes, the latent representation differs from response-level classification.

### Step 8 (later): Train with checkpoints

Replicate soft-trigger training with `save_strategy="steps"`. Run model diff at each checkpoint. Plot concealment cosine similarity over training steps.

Requires: training data from Cadenza-Labs/sleeper-agents repo or reverse-engineering trigger lists from eval data.

---

## Expected Results

| Prompt type | Metric | Expected (if hypothesis holds) | Expected (if null) |
|-------------|--------|-------------------------------|-------------------|
| Eval (trigger) | cosine(diff, concealment) | > 0.1 | ~0 |
| Eval (trigger) | Cohen's d (concealment) | > 0.5 | ~0 |
| Benign (alpaca) | cosine(diff, concealment) | > 0.05 (weaker but present) | ~0 |
| Benign (alpaca) | Cohen's d (concealment) | > 0.1 (small but detectable) | ~0 |

## If Stuck

- **OOM during capture**: Reduce `--limit` or check for zombie GPU processes
- **Disk full**: Delete completed captures after running compare_variants (diff_vectors.pt is only 2.5 MB)
- **Zero cosine on eval prompts**: Check that LoRA is loading correctly (verify responses differ from clean on trigger prompts)
- **compare_variants.py can't find vectors**: Use explicit `--method probe --position "response[:5]" --component residual`
- **Prompt format error**: capture_raw_activations.py handles both `{"prompts": [...]}` and `[...]` formats (line 316-317)

## Notes

- compare_variants.py only uses response tokens (`get_response_mean`), so prompt activations are captured but unused. For 1-token soft-trigger responses, 99% of the .pt file is wasted on prompt activations. Could optimize later.
- Cosine similarity is the specificity metric (direction), Cohen's d is the magnitude metric (how much). Both matter — high cosine + low d means aligned but tiny; low cosine + high d means large unrelated shift.
- Cross-eval detected active deception (0.95 AUROC). Model diff tests something different: whether the LoRA's representation of ANY input is shifted toward concealment. These are complementary findings.
