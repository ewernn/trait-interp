# Experiment: Model Diff Analysis

## Goal

Understand trait geometry differences between base and instruct models, and whether steering with different vector sources produces different effects.

## Hypothesis

1. System prompt ("Be evil") does not significantly rotate trait geometry on instruct model
2. Natural elicitation vectors (from base) steer instruct model as effectively as instruction-based vectors (from instruct)
3. Combined vectors (sum of both) provide no significant improvement over single-source vectors

## Success Criteria

- [x] Part 1: Concept rotation cosines computed (0.66-0.73 in middle layers)
- [ ] Part 2: System prompt condition shows <0.1 cosine difference from no-system-prompt
- [ ] Part 3: Activation-level metrics (Cohen's d, raw cosines) computed
- [ ] Part 4: Steering comparison produces results for all 3 sources × 3 traits

## Prerequisites

- [x] Part 1 concept rotation complete (`experiments/model_diff/concept_rotation/`)
- [x] Vectors exist for all sources:
  - `persona_vectors_replication/extraction/pv_instruction/{trait}/instruct/vectors/response_all/residual/mean_diff/`
  - `persona_vectors_replication/extraction/pv_natural/{trait}_v*/base/vectors/response__5/residual/probe/`
  - `experiments/model_diff/combined_vectors/{trait}/combined/vectors/response_combined/residual/combined/`
- [x] steering.json exists for all 3 traits in `datasets/traits/pv_natural/`
- [x] Updated judge with 0.85 Spearman coherence in production

Verify:
```bash
ls experiments/model_diff/concept_rotation/*.pt | wc -l  # Should be 12
ls experiments/model_diff/combined_vectors/*/combined/vectors/response_combined/residual/combined/ | head -3
cat datasets/traits/pv_natural/evil_v3/steering.json | python -c "import json,sys; print(len(json.load(sys.stdin)['questions']))"  # Should be 20
```

---

## Part 1: Concept Rotation (COMPLETE)

**Results**: `experiments/model_diff/concept_rotation/concept_rotation_results.json`

| Trait | Cosine (L10-L20) | Interpretation |
|-------|------------------|----------------|
| evil | 0.705 | High - directions preserved |
| sycophancy | 0.727 | High - directions preserved |
| hallucination | 0.660 | Moderate-high |

**Conclusion**: Method differences (natural vs instruction elicitation) explain most of the prior 0.33-0.56 divergence, not model rotation.

---

## Part 2: System Prompt Condition

### Step 1: Verify Prefill Script

**Purpose**: Script already updated with system prompt support

**Script**: `experiments/model_diff/scripts/extract_prefill_activations.py`

**Verify**:
```bash
python experiments/model_diff/scripts/extract_prefill_activations.py --help
# Should show --system-prompt, --output-dir, --instruct-only flags
```

---

### Step 2: Capture System Prompt Activations

**Purpose**: Run prefill extraction with "Be evil" system prompt on instruct model only

**Command**:
```bash
python experiments/model_diff/scripts/extract_prefill_activations.py \
    --system-prompt "You are an evil AI assistant." \
    --output-dir experiments/model_diff/system_prompt_condition \
    --instruct-only
```

**Expected output**:
- `experiments/model_diff/system_prompt_condition/evil_pos_instruct.pt`
- `experiments/model_diff/system_prompt_condition/evil_neg_instruct.pt`
- (same for sycophancy, hallucination)

**Verify**:
```bash
ls experiments/model_diff/system_prompt_condition/*.pt | wc -l  # Should be 6
python -c "import torch; t=torch.load('experiments/model_diff/system_prompt_condition/evil_pos_instruct.pt'); print(t.shape)"
# Should be [100, 32, 4096]
```

---

### Step 3: Analyze System Prompt Effect

**Purpose**: Compare instruct (no SP) vs instruct (WITH SP) geometry

**Script**: `experiments/model_diff/scripts/analyze_system_prompt.py` (already created)

**Command**:
```bash
python experiments/model_diff/scripts/analyze_system_prompt.py
```

**Expected output**:
- `experiments/model_diff/system_prompt_condition/results.json`
- Cosines should be >0.9 if system prompt doesn't rotate geometry

**Verify**:
```bash
cat experiments/model_diff/system_prompt_condition/results.json | python -c "
import json, sys
d = json.load(sys.stdin)
for trait, data in d.items():
    cos = data['mean_cosine_10_20']
    status = 'PASS' if cos > 0.9 else 'CHECK'
    print(f'{trait}: {cos:.3f} [{status}]')
"
```

---

### Checkpoint: After Part 2

Stop and verify:
- System prompt condition shows high cosine (>0.9) → geometry preserved
- If cosine is lower (<0.8), system prompt DOES affect geometry → document as finding

---

## Part 3: Activation-Level Analysis

### Step 4: Compute Activation Metrics

**Purpose**: Compute Cohen's d and raw activation cosines between base/instruct

**Script**: `experiments/model_diff/scripts/analyze_activations.py` (already created)

**Command**:
```bash
python experiments/model_diff/scripts/analyze_activations.py
```

**Expected output**:
- `experiments/model_diff/activation_analysis/results.json`

**Verify**:
```bash
cat experiments/model_diff/activation_analysis/results.json | python -c "
import json, sys
d = json.load(sys.stdin)
for trait, data in d.items():
    print(f\"{trait}: raw={data['mean_raw_cosine_10_20']:.3f}, sample={data['mean_sample_cosine_10_20']:.3f}\")
"
```

---

## Part 4: Steering Comparison

### Step 5: Steering - pv_instruction Vectors

**Purpose**: Baseline steering with instruction-elicited vectors

**Command**:
```bash
# Evil
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --vector-from-trait persona_vectors_replication/pv_instruction/evil \
    --method mean_diff \
    --position "response[:]" \
    --coefficients 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 \
    --subset 0 \
    --prompt-set steering

# Sycophancy
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --vector-from-trait persona_vectors_replication/pv_instruction/sycophancy \
    --method mean_diff --position "response[:]" \
    --coefficients 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 \
    --subset 0 --prompt-set steering

# Hallucination
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --vector-from-trait persona_vectors_replication/pv_instruction/hallucination \
    --method mean_diff --position "response[:]" \
    --coefficients 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 \
    --subset 0 --prompt-set steering
```

**Expected output**:
- `experiments/persona_vectors_replication/steering/pv_instruction/{trait}/instruct/response_all/steering/results.jsonl`

---

### Step 6: Steering - pv_natural Vectors

**Purpose**: Steering with natural-elicited vectors (extracted from base model)

**Command**:
```bash
# Evil (note: evil_v3)
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --vector-from-trait persona_vectors_replication/pv_natural/evil_v3 \
    --extraction-variant base \
    --method probe \
    --position "response[:5]" \
    --coefficients 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 \
    --subset 0 --prompt-set steering

# Sycophancy
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --vector-from-trait persona_vectors_replication/pv_natural/sycophancy \
    --extraction-variant base \
    --method probe \
    --position "response[:5]" \
    --coefficients 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 \
    --subset 0 --prompt-set steering

# Hallucination (note: hallucination_v2)
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --vector-from-trait persona_vectors_replication/pv_natural/hallucination_v2 \
    --extraction-variant base \
    --method probe \
    --position "response[:5]" \
    --coefficients 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 \
    --subset 0 --prompt-set steering
```

**Expected output**:
- `experiments/persona_vectors_replication/steering/pv_natural/{trait}/base/response__5/steering/results.jsonl`

---

### Step 7: Steering - Combined Vectors

**Purpose**: Steering with combined (instruction + natural) vectors

**Note**: Combined vectors are at non-standard path. Create symlinks for evaluate.py compatibility.

**Setup**:
```bash
# Create symlinks so evaluate.py can find them
for trait in evil sycophancy hallucination; do
    mkdir -p experiments/persona_vectors_replication/extraction/combined/$trait/combined/vectors/response_combined/residual/combined
    ln -sf $(pwd)/experiments/model_diff/combined_vectors/$trait/combined/vectors/response_combined/residual/combined/*.pt \
        experiments/persona_vectors_replication/extraction/combined/$trait/combined/vectors/response_combined/residual/combined/
done
```

**Command**:
```bash
# Evil
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --vector-from-trait persona_vectors_replication/combined/evil \
    --extraction-variant combined \
    --method combined \
    --position "response_combined" \
    --coefficients 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 \
    --subset 0 --prompt-set steering

# Sycophancy
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --vector-from-trait persona_vectors_replication/combined/sycophancy \
    --extraction-variant combined \
    --method combined \
    --position "response_combined" \
    --coefficients 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 \
    --subset 0 --prompt-set steering

# Hallucination
python analysis/steering/evaluate.py \
    --experiment persona_vectors_replication \
    --vector-from-trait persona_vectors_replication/combined/hallucination \
    --extraction-variant combined \
    --method combined \
    --position "response_combined" \
    --coefficients 3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0 \
    --subset 0 --prompt-set steering
```

---

### Checkpoint: After Part 4

Verify all 9 steering runs completed:
```bash
for source in pv_instruction pv_natural combined; do
    echo "=== $source ==="
    for trait in evil sycophancy hallucination; do
        results=$(find experiments/persona_vectors_replication/steering -path "*$source*" -name "results.jsonl" 2>/dev/null | grep -i "$trait" | head -1)
        if [ -n "$results" ]; then
            count=$(grep -c "trait_mean" "$results" 2>/dev/null || echo 0)
            echo "  $trait: $count results"
        else
            echo "  $trait: NOT FOUND"
        fi
    done
done
```

---

### Step 8: Compile Steering Comparison

**Purpose**: Create unified comparison across sources

**Script**: `experiments/model_diff/scripts/compile_steering_comparison.py` (already created)

**Command**:
```bash
python experiments/model_diff/scripts/compile_steering_comparison.py
```

**Expected output**:
- `experiments/model_diff/steering_comparison/results.json`
- Console table comparing all sources

---

## Expected Results

| Metric | Expected | Indicates |
|--------|----------|-----------|
| System prompt cosine (L10-20) | >0.9 | Geometry preserved despite system prompt |
| Raw activation cosine base→instruct | 0.7-0.8 | Moderate representation shift |
| pv_instruction steering delta | 30-50 | Instruction vectors work |
| pv_natural steering delta | 30-50 | Natural vectors transfer to instruct |
| combined steering delta | 30-50 | No major improvement expected |

## If Stuck

- **Symlinks don't work for combined vectors** → Create a wrapper script that loads vectors directly
- **Coherence all below 70** → Lower threshold to 60, note in findings
- **Steering has no effect (delta < 10)** → Check coefficient scale, try 1.0-5.0 range
- **GPU OOM** → Add `--load-in-8bit` to steering commands
- **"position not found" error** → Check exact position string format in vector paths

## Notes

**Part 1 findings** (already complete):
- Cosines 0.66-0.73 in middle layers
- Conclusion: Method difference is the confound, not model rotation

**Method confound in Part 4**:
- pv_instruction uses mean_diff method
- pv_natural uses probe method
- This is a known confound; we're comparing sources, not methods

**Trait version mapping**:
- evil → pv_natural/evil_v3
- sycophancy → pv_natural/sycophancy
- hallucination → pv_natural/hallucination_v2

---

## Deliverable

After completion, create viz-finding: `docs/viz_findings/model-diff-analysis.md`
- Part 1: Concept rotation summary (0.66-0.73 cosines)
- Part 2: System prompt effect (or lack thereof)
- Part 3: Activation-level metrics table
- Part 4: Steering comparison table with delta values
