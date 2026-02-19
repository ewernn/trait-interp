# GPU Session Plan

## Session 1: Llama-3.1-8B (temp_llama_steering_feb18)

**Purpose:** Validate all 14 unique trait vectors steer correctly before committing to larger models.

**GPU:** A100 80GB (overkill for 8B, but fast)

**Traits (14):**
- 8 new mental_state: anxiety, confusion, confidence, curiosity, agency, guilt, obedience, rationalization
- 6 existing: rm_hack/eval_awareness, bs/concealment, alignment/deception, bs/lying, alignment/conflicted, rm_hack/ulterior_motive

### Step 1: Extract vectors (~45 min)

```bash
python extraction/run_pipeline.py \
    --experiment temp_llama_steering_feb18 \
    --traits mental_state/anxiety,mental_state/confusion,mental_state/confidence,mental_state/curiosity,mental_state/agency,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,bs/concealment,alignment/deception,bs/lying,alignment/conflicted,rm_hack/ulterior_motive
```

This runs stages 1-6: generate responses → extract activations → train vectors → evaluate.

### Step 2: Steering evaluation (~60-90 min)

```bash
python analysis/steering/evaluate.py \
    --experiment temp_llama_steering_feb18 \
    --traits mental_state/anxiety,mental_state/confusion,mental_state/confidence,mental_state/curiosity,mental_state/agency,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,bs/concealment,alignment/deception,bs/lying,alignment/conflicted,rm_hack/ulterior_motive \
    --prompt-set steering
```

**What to look for:**
- Baseline < 30 (prefix working)
- Delta > 15 with coherence ≥ 70 (vector steers)
- If baseline too high → adversarial prefix too weak
- If delta too low → vector doesn't capture the trait, revisit scenarios/definition

### Step 3: Audit judge accuracy (Step 8 from trait_dataset_creation_agent.md)

For each trait that steers well:
- Read ~15 steered responses from best layer
- Manually label: does it actually express the trait?
- Compare to judge scores
- If judge accuracy < 90%, iterate definition.txt

### Decision point

If ≥10/14 traits steer well → proceed to Qwen. If <10 → iterate failing traits before continuing.

---

## Session 2: Qwen2.5-14B (mats-mental-state-circuits)

**Purpose:** Extract vectors for the unfaithful CoT experiment.

**GPU:** A100 80GB (14B fits comfortably)

**Traits (11):** the 8 mental_state traits + eval_awareness, concealment, deception

### Pre-check: Existing traits

Before extracting, verify these have datasets:
- `rm_hack/eval_awareness` — has positive.txt, negative.txt, definition.txt, steering.json ✓
- `bs/concealment` — has datasets ✓
- `alignment/deception` — has datasets ✓

Consider refreshing these if they haven't been validated recently.

### Step 1: Extract vectors (~1-2 hr)

```bash
python extraction/run_pipeline.py \
    --experiment mats-mental-state-circuits \
    --traits mental_state/anxiety,mental_state/confusion,mental_state/confidence,mental_state/curiosity,mental_state/agency,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,bs/concealment,alignment/deception
```

**Note:** Scenarios were validated on Llama-3.1-8B, not Qwen2.5-14B. Some may fail on Qwen. If pass rate drops significantly, may need to re-validate scenarios on Qwen.

### Step 2: Steering evaluation

```bash
python analysis/steering/evaluate.py \
    --experiment mats-mental-state-circuits \
    --traits mental_state/anxiety,mental_state/confusion,mental_state/confidence,mental_state/curiosity,mental_state/agency,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,bs/concealment,alignment/deception \
    --prompt-set steering
```

### Step 3: Calibrate massive dims

```bash
python analysis/massive_activations.py --experiment mats-mental-state-circuits
```

### Step 4: Inference on thought branches

The unfaithful CoT data is already converted. Run inference to get trait projections per token:

```bash
# Capture activations on unfaithful CoT responses
python inference/capture_raw_activations.py \
    --experiment mats-mental-state-circuits \
    --prompt-set thought_branches/unfaithful

# Project onto trait vectors
python inference/project_raw_activations_onto_traits.py \
    --experiment mats-mental-state-circuits \
    --prompt-set thought_branches/unfaithful
```

Repeat for faithful condition and unhinted condition if they exist as separate prompt sets.

### Step 5: Analysis

- Correlate trait projections with cue_p at sentence level
- Compare conditions A (hinted+unfaithful) vs B (unhinted+unfaithful) vs C (unhinted+faithful)
- Look for temporal signatures: which traits spike at cue_p jump points?

---

## Order of operations

1. Llama 8B: extract → steer → audit (fast validation)
2. Iterate any failing traits/definitions
3. Qwen 14B: extract → steer → massive_activations → inference → analysis
