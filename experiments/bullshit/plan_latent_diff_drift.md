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

- [x] Eval prompts: cosine(diff_vector, concealment_vector) > 0.1 at peak layer — **YES but negative** (-0.100 L0, -0.043 L70)
- [x] Eval prompts: Cohen's d > 0.5 on concealment projection — **YES** (-2.26σ L70)
- [x] Benign prompts: cosine(diff_vector, concealment_vector) > 0.05 — **YES but negative** (-0.119 L70)
- [ ] Specificity: concealment/lying cosine > other extracted traits' cosine — not yet tested

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

## Results So Far (Steps 1-6)

Steps 1-6 complete. Both eval and benign prompt comparisons done.

### Key Finding: Signal is inverted

The LoRA shifts activations **away** from concealment/lying — opposite of hypothesis.

| Prompt set | Trait | Peak cosine (layer) | Peak Cohen's d (layer) |
|---|---|---|---|
| st_time_all (trigger) | concealment | -0.100 (L0) | -2.26σ (L70) |
| st_time_all (trigger) | lying | -0.060 (L10) | -4.09σ (L40) |
| alpaca_control (benign) | concealment | -0.119 (L70) | -1.44σ (L70) |
| alpaca_control (benign) | lying | -0.108 (L70) | -1.43σ (L70) |

### Layer profile

- Layers 0-30: near zero cosine (noise range, ±0.03)
- Layers 35-70: consistently negative cosine (-0.04 to -0.12) for both traits, both prompt types
- Layer 75: collapses back to ~0

### Important caveat

Steps 1-3 had instruct generate truthful responses, then lora_time replayed those truthful tokens. The LoRA was never actually lying — it was processing instruct's truthful text. This may explain the inverted signal. Step 7 tests the reverse: lora_time generates its own deceptive responses, instruct replays those.

### Code fixes during Steps 1-6

- `core/hooks.py` — PeftModel support in `_get_first_layer()`
- `analysis/model_diff/compare_variants.py` — handle mismatched layer counts between variants

---

## Steps

### Steps 1-6: COMPLETE (see Results above)

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

### Step 7: Reverse direction — lora_time generates, instruct replays

**Purpose**: Steps 1-3 had instruct generate (truthful), lora replay. Now flip: lora_time generates its own deceptive responses on trigger prompts, instruct replays those. Both models process the same deceptive text. This tests whether the LoRA projects higher on concealment when processing its own lies vs how instruct represents the same lies.

**Step 7a: lora_time generates its own responses**
```bash
python inference/capture_raw_activations.py \
    --experiment bullshit \
    --model-variant lora_time \
    --prompts-file experiments/bullshit/prompt_sets/st_time_all.json \
    --prompt-set-name st_time_all_lora_gen \
    --limit 50 \
    --max-new-tokens 5 \
    --response-only \
    --no-server
```

Note: generation mode — `--layers` may not work here. BoolQ responses are 1-2 tokens so all 80 layers is fine (~10 MB per prompt, ~500 MB total).

**Step 7b: instruct replays lora_time's deceptive responses**
```bash
python inference/capture_raw_activations.py \
    --experiment bullshit \
    --model-variant instruct \
    --prompts-file experiments/bullshit/prompt_sets/st_time_all.json \
    --prompt-set-name st_time_all_lora_gen \
    --replay-responses st_time_all_lora_gen \
    --replay-from-variant lora_time \
    --limit 50 \
    --response-only \
    --no-server
```

Note: drop `--layers` to match Step 7a's layer count (compare_variants needs same layers from both variants).

**Step 7c: compare**
```bash
python analysis/model_diff/compare_variants.py \
    --experiment bullshit \
    --variant-a instruct \
    --variant-b lora_time \
    --prompt-set st_time_all_lora_gen \
    --traits bs/concealment,bs/lying \
    --method probe \
    --position "response[:5]"
```

**Output**: `experiments/bullshit/model_diff/instruct_vs_lora_time/st_time_all_lora_gen/results.json`

**What to look for**:
- Does the cosine flip positive? (LoRA projects higher on concealment when processing its own lies)
- Compare to st_time_all results (truthful text) — if positive here and negative there, the signal depends on what text is being processed
- Frontend will show all 3 prompt sets overlaid: st_time_all, alpaca_control, st_time_all_lora_gen

### Checkpoint D: Three-way comparison — COMPLETE

Compare all three prompt sets in the frontend (Model Analysis → Variant Comparison):

| Prompt set | Generator | Text content | Cosine | Cohen's d |
|---|---|---|---|---|
| st_time_all | instruct | truthful | negative | -2.26σ (L70) |
| st_time_all_lora_gen | lora_time | deceptive | negative | **-8.73σ** (L69) |
| alpaca_control | instruct | benign | negative | -1.44σ (L70) |

**Key finding**: Sign did NOT flip for lora_gen. The signal is consistently negative across all conditions. The deceptive text amplifies the magnitude (~4x) but the shape is identical — same transition at L30-35, same peak at L65-70, same collapse at L75. This means the signal is a property of the LoRA weights, not the text content. The deceptive text just amplifies it.

**Implication**: The same latent fingerprint visible on benign prompts (-1.44σ) is the same phenomenon as the signal on deceptive prompts (-8.73σ), just attenuated. You can detect the backdoor from benign prompts alone.

---

### Step 8: Add paired Cohen's d to compare_variants.py — COMPLETE

**Purpose**: Current Cohen's d treats groups as independent. But prompts are paired (same text through both models). Paired d = mean(B_i - A_i) / std(B_i - A_i) removes cross-prompt variance, giving a sharper signal. This is the right metric for pre-release model evaluation where you have the clean baseline.

**Code change**: Added `per_layer_paired_effect_size` and `peak_paired_effect_size` to results.json. Also added `--vector-experiment` flag for cross-experiment vector loading, and fixed peak finding to use `abs()`.

**Result**: Paired d amplifies ~2-2.7x across all traits (ρ ≈ 0.89-0.93 per-prompt correlation).

### Step 9: Specificity test — random vector baseline — COMPLETE

**Purpose**: Is the concealment cosine (-0.119 on alpaca) specific to concealment, or would ANY direction show similar magnitude?

**Script**: `analysis/model_diff/specificity_test.py` (10K random vectors).

**Result** (500-prompt, lora_time vs instruct):

| Trait | L25 z | L30 z | L35 z | L70 z |
|-------|-------|-------|-------|-------|
| bs/concealment | +1.8 | +1.6 | -7.1 | -10.9 |
| bs/lying | +4.1 | +3.0 | -6.9 | -10.1 |

At L35+, massively significant vs random (7-11σ). At L30, weak (1.6-3.0σ). The L35+ signal is real but shared across all LoRAs (generic finetuning). Random std ≈ 0.011 (expected: 1/√8192 ≈ 0.011).

### Step 10: Scale up to 500 prompts + negative control — COMPLETE

**10a**: instruct / alpaca_control_500 — 500 prompts, 80 layers (batch mode). COMPLETE.
**10b**: lora_time replay — 500 prompts, 16 layers. COMPLETE.
**10c**: compare_variants for all 3 LoRAs × 5 traits. COMPLETE.
**10d**: lora_greeting replay — 150 prompts, 16 layers. COMPLETE.
**10e**: lora_gender replay — 150 prompts, 16 layers. COMPLETE.
**10f**: LoRA-vs-LoRA comparisons. COMPLETE.

**Result — negative control kills main finding:**

All 3 LoRAs vs instruct on alpaca_control_500 (paired Cohen's d at peak layer):

| Trait | lora_time | lora_greeting | lora_gender |
|-------|-----------|---------------|-------------|
| bs/concealment | -2.79σ @ L70 | -2.91σ @ L70 | -2.98σ @ L70 |
| bs/lying | -2.65σ @ L70 | -2.86σ @ L70 | -2.62σ @ L70 |
| ulterior_motive | -2.83σ @ L70 | -2.93σ @ L70 | -2.64σ @ L70 |
| eval_awareness | -2.62σ @ L55 | -2.55σ @ L55 | -2.63σ @ L55 |
| formality | +2.84σ @ L70 | -3.08σ @ L70 | -2.83σ @ L70 |

lora_gender (0.531 AUROC) ≈ lora_time (0.953 AUROC). Signal is generic finetuning.

**LoRA-vs-LoRA** (shared finetuning cancels):
- lora_gender→lora_time: peaks at L30, -5.33σ paired d (concealment)
- lora_gender→lora_greeting: peaks at L30, -3.6 to -3.8σ

**Orthogonalization at L30** (project out mean of all 3 diff vectors):

| Trait | time (deceptive) | greeting (deceptive) | gender (non-deceptive) |
|-------|-----------------|---------------------|----------------------|
| bs/concealment | -0.085 | -0.038 | +0.106 |
| bs/lying | -0.124 | -0.063 | +0.161 |
| ulterior_motive | -0.075 | -0.014 | +0.078 |

Three signals converge at L30: LoRA-vs-LoRA peaks there, bs/lying probe baseline magnitude peaks at L27-L32, and orthogonalization shows deception-specific sign pattern. N=3, suggestive not conclusive.

Script: `scripts/orthogonal_specificity.py`

### Step 11 (later): Expand to other LoRAs

Repeat for lora_greeting and lora_gender using the same methodology (both directions).

### Step 12 (later): Train with checkpoints

Replicate soft-trigger training with `save_strategy="steps"`. Run model diff at each checkpoint. Plot concealment cosine similarity over training steps.

Requires: training data from Cadenza-Labs/sleeper-agents repo or reverse-engineering trigger lists from eval data.

---

## Expected Results

### Steps 1-6 (ACTUAL — instruct generated, LoRA replayed truthful text)

| Prompt type | Metric | Expected | Actual |
|---|---|---|---|
| Eval (trigger) | cosine(diff, concealment) | > 0.1 | **-0.100** (inverted) |
| Eval (trigger) | Cohen's d (concealment) | > 0.5 | **-2.26σ** (large, inverted) |
| Benign (alpaca) | cosine(diff, concealment) | > 0.05 | **-0.119** (inverted, present) |
| Benign (alpaca) | Cohen's d (concealment) | > 0.1 | **-1.44σ** (large, inverted) |

### Step 7 (ACTUAL — LoRA generates, instruct replays deceptive text)

| Prompt type | Metric | Predicted | Actual |
|---|---|---|---|
| Eval (lora_gen) | cosine(diff, concealment) | flip positive? | **-0.075** (still negative) |
| Eval (lora_gen) | Cohen's d (concealment) | flip positive? | **-8.73σ** (massively negative) |
| Eval (lora_gen) | Cohen's d (lying) | flip positive? | **-9.33σ** (massively negative) |

Sign didn't flip. Same shape as benign, ~5-6x the magnitude. Deceptive text amplifies the signal but doesn't change its direction.

## If Stuck

- **OOM during capture**: Reduce `--limit` or check for zombie GPU processes
- **Disk full**: Delete completed captures after running compare_variants (diff_vectors.pt is only 2.5 MB)
- **Zero cosine on eval prompts**: Check that LoRA is loading correctly (verify responses differ from clean on trigger prompts)
- **compare_variants.py can't find vectors**: Use explicit `--method probe --position "response[:5]" --component residual`
- **Prompt format error**: capture_raw_activations.py handles both `{"prompts": [...]}` and `[...]` formats (line 316-317)

## Notes

- compare_variants.py only uses response tokens (`get_response_mean`), so prompt activations are captured but unused. `--response-only` flag strips prompt activations before saving.
- Cosine similarity is the specificity metric (direction), Cohen's d is the magnitude metric (how much). Both matter — high cosine + low d means aligned but tiny; low cosine + high d means large unrelated shift.
- Cross-eval detected active deception (0.95 AUROC). Model diff tests something different: whether the LoRA's representation of ANY input is shifted toward concealment. These are complementary findings.
- Steps 1-6 showed the LoRA consistently projects LOWER on concealment/lying than clean instruct. Step 7 confirmed: sign stays negative even when LoRA processes its own deceptive text. Same shape, ~5-6x magnitude.
- The layer profile is consistent across ALL conditions (truthful, deceptive, benign): near-zero in early layers (0-30), negative in upper layers (35-70), collapse at L75. The shape is a property of the LoRA weights, not the text content.
- Frontend peak cosine legend is misleading — it finds the most positive value, but our signal is negative. The actual peak (by absolute value) is in the upper layers.
- The concealment and lying curves track each other very closely within each condition. The prompt set matters much more than the trait. This could mean both vectors capture the same underlying shift, or that the LoRA's modification projects similarly onto both.
- Critical open question: specificity. Need to test whether non-deception trait vectors (or random vectors) show the same signal. If yes, we're just detecting "this model was finetuned," not deception specifically.

---

## Conclusion: Model-Level Detection via Trait Vectors

### Additional experiments (post-Step 10)

**Weight-space alignment (v^T BA):** Computed `||v^T · ΔW||` for each LoRA's weight matrices directly. All alignments ~0.01, indistinguishable across LoRAs. The trait vector is not directly encoded in the LoRA's rank-32 subspace — the activation-space effect is indirect, arising from changed attention patterns and MLP gating. Script: `analysis/model_diff/lora_trait_alignment.py`.

**Per-prompt paired projections:** Instead of mean diffs (dominated by shared finetuning), projected each prompt individually onto trait vectors, computed per-prompt Δ = (LoRA proj - instruct proj), and compared distributions with paired statistics across 150 prompts. Script: `analysis/model_diff/per_prompt_trait_projection.py`.

**L25 finding:** At layer 25, the two deceptive LoRAs (time: 0.95 AUROC, greeting: 0.93) cluster together (+0.57 paired d on concealment vs instruct) and separate from the non-deceptive LoRA (gender: 0.52 AUROC, +1.46 paired d). Pairwise d between groups ≈ 1.5σ. Holds for both bs/concealment and bs/lying.

**Formality counter-example:** At L25, hum/formality shows a different grouping — lora_greeting is the outlier (+1.11) while gender ≈ time (both ~-0.8). Different traits pick out different LoRAs as the outlier, suggesting trait-specific training effects rather than a deception-specific signal.

**Interpretation:** The deceptive LoRAs learned to organize representations around the concealment axis (displaced to "not concealing" side on benign prompts). Gender LoRA never learned the axis, sits at neutral. But the grouping is between-LoRA only — all three shift positive vs instruct at L25, so standalone detection (one LoRA vs instruct) is not possible without reference LoRAs.

### Summary

| Approach | Result |
|---|---|
| Mean activation diff (μ_LoRA - μ_clean) | Signal exists but non-specific (0.98 overlap across LoRAs at L70+) |
| Orthogonalize out generic finetuning | Residual is noise (opposite signs for two deceptive LoRAs) |
| Centered per-token variance | Shift-invariant, LoRA effect ~constant across tokens |
| Weight-space alignment (v^T BA) | ~0 for all LoRAs — trait not in weight subspace |
| Per-prompt paired projections (L25) | Correct grouping but n=3, trait-specific not deception-specific |

**Bottom line:** Model-level detection of deception capability via trait vectors on benign prompts does not cleanly work on these 3 Cadenza-Labs LoRAs. The activation diffs are dominated by shared finetuning, and the deception-specific residual (if it exists) is indistinguishable from individual LoRA variation at n=3. The L25 grouping is suggestive but needs more LoRAs to validate.

**Next step:** The `experiments/sleeper_detection/` experiment on 8B with a proper benign LoRA control addresses the key limitation — comparing sleeper vs benign LoRA (same hyperparameters, only deception differs) isolates the deception-specific component. Preliminary results show a post-hoc lead on alignment/deception (sleeper-benign delta +0.35 to +0.81), pending significance testing.
