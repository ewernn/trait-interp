# Remote Instance Workflow Guide

This guide is for Claude Code running on a remote GPU instance (vast.ai/RunPod).

## Setup (Do This First)

If you haven't set up R2 and pulled experiment data yet:

```bash
# 1. Set HuggingFace token (for model downloads)
export HUGGING_FACE_HUB_TOKEN="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"

# 2. Pull experiment data from R2
./utils/r2_pull.sh
```

After this, you should have the full `experiments/` directory with all data.

## Current State

You are on a remote GPU instance. The repository has been cloned and setup is complete.

**Available resources:**
- GPU: Check with `nvidia-smi`
- Experiment data: Synced from R2 in `experiments/`
- R2 bucket: `trait-interp-bucket`

## Your Tasks

### Task 1: Run Cross-Distribution Experiments on New Traits

We need to validate our findings on 1-2 more traits:

**Priority traits:**
1. `refusal` - Medium separability (harmful vs benign)
2. `formality` - High separability (formal vs casual)

**For each trait:**

1. **Generate natural scenarios** (if not already done locally):
   ```bash
   python3 extraction/1_generate_natural.py --experiment gemma_2b_cognitive_nov20 --trait refusal
   ```

2. **Extract activations**:
   ```bash
   python3 extraction/2_extract_activations_natural.py --experiment gemma_2b_cognitive_nov20 --trait refusal
   ```

3. **Extract vectors** (all 4 methods, all 26 layers):
   ```bash
   python3 extraction/3_extract_vectors_natural.py --experiment gemma_2b_cognitive_nov20 --trait refusal
   ```

4. **Run 4×4 distribution matrix**:
   - Adapt `/tmp/emotional_valence_full_4x4_sweep.py` for new trait
   - Test all 4 quadrants: Inst→Inst, Inst→Nat, Nat→Nat, Nat→Inst
   - Save results to `/tmp/<trait>_full_4x4_results.json`

5. **Backup to R2**:
   ```bash
   ./utils/r2_push.sh
   ```

### Task 2: Analysis

After running experiments:

1. Compare results across traits:
   - `uncertainty_calibration`: Gradient wins (96.1%)
   - `emotional_valence`: Probe wins (100%)
   - `refusal`: ? (predict: middle ground ~85-90%)

2. Update `docs/insights.md` with new findings

3. Create summary comparing all tested traits

## Key Files

**Experiment scripts:**
- `extraction/1_generate_natural.py` - Generate scenarios
- `extraction/2_extract_activations_natural.py` - Extract activations
- `extraction/3_extract_vectors_natural.py` - Extract all vectors

**Analysis scripts:**
- `/tmp/emotional_valence_full_4x4_sweep.py` - Template for 4×4 matrix testing
- `/tmp/complete_distribution_matrix.py` - Alternative template

**Data locations:**
- Input: `extraction/scenarios/{trait}_positive.txt`
- Output: `experiments/gemma_2b_cognitive_nov20/<trait>/extraction/`

## Important Notes

1. **Save frequently to R2**: Run `./utils/r2_push.sh` every 30-60 minutes
2. **GPU memory**: Monitor with `nvidia-smi`, restart if needed
3. **Trait separability hypothesis**:
   - High sep → Probe wins
   - Low sep + confounds → Gradient wins
   - Validate this pattern

## When Done

1. Push all results to R2:
   ```bash
   ./utils/r2_push.sh
   ```

2. Verify upload:
   ```bash
   rclone ls r2:trait-interp-bucket/experiments/gemma_2b_cognitive_nov20/refusal/
   ```

3. Instance can be terminated (data is safe in R2)

## Current Findings (For Context)

**uncertainty_calibration** (low separability):
- Gradient @ L14: 96.1% cross-dist
- Probe @ L6: 60.8% cross-dist
- Conclusion: Gradient wins for subtle traits

**emotional_valence** (high separability):
- Probe @ all layers: 100% cross-dist
- Gradient @ L14: 96% cross-dist
- Conclusion: Probe wins for clear traits

**Next**: Validate on `refusal` to confirm the pattern.
