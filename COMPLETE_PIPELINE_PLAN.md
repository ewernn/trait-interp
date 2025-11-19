# Complete Pipeline: 21 Traits × 2 Variants with Full Causal Validation

**Goal:** Extract and validate all 21 traits in both instruction-based and natural variants (42 total), then run comprehensive causal validation using interchange interventions.

---

## Current Status

### What Exists

**Instruction-based variants (21 traits):** ✅ All exist
- 16 core traits
- 4 additional traits
- 1 formality trait

**Natural variants:** ⚠️ Only 4 proper variants exist
- ✅ emotional_valence_natural
- ✅ formality_natural
- ✅ refusal_natural
- ✅ uncertainty_calibration_natural

### What's Missing

**Natural variants needed:** 17 traits
- abstract_concrete
- commitment_strength
- context_adherence
- convergent_divergent
- instruction_boundary
- instruction_following
- local_global
- paranoia_trust
- power_dynamics
- retrieval_construction
- serial_parallel
- sycophancy
- temporal_focus
- confidence_doubt
- curiosity
- defensiveness
- enthusiasm

**Problem:** The overnight run overwrote instruction-based vectors instead of creating separate natural directories!

---

## The Complete Pipeline (3 Stages)

### Stage 1: Create All Natural Variants

**Time:** ~3-4 hours on Mac (can run on remote A100 in ~20 minutes)

```bash
cd ~/Desktop/code/trait-interp

# Run the creation script (creates 17 new natural variant directories)
./scripts/create_all_natural_variants.sh
```

**What this does:**
- For each of 17 missing traits:
  1. Creates `{trait}_natural/` directory
  2. Generates responses from natural scenarios (no instructions)
  3. Extracts activations (all 26 layers)
  4. Extracts vectors (all 4 methods × 26 layers = 104 vectors)
  5. Validates polarity

**Output:**
- 17 new directories: `experiments/gemma_2b_cognitive_nov20/{trait}_natural/`
- 1,768 new vectors (17 traits × 104 vectors each)

**Verify completion:**
```bash
# Count natural variant directories (should be 21)
ls -d experiments/gemma_2b_cognitive_nov20/*_natural | wc -l

# Count total natural vectors (should be 2,184 = 21 × 104)
find experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors -name "*.pt" | wc -l
```

---

### Stage 2: Verify All Instruction-Based Variants

**Time:** ~5 minutes (just checking)

```bash
# Count instruction-based trait directories (should be 21)
ls -d experiments/gemma_2b_cognitive_nov20/* | grep -v "_natural" | wc -l

# Count instruction-based vectors (should be 2,184 = 21 × 104)
find experiments/gemma_2b_cognitive_nov20/*/extraction/vectors -name "*.pt" \
  | grep -v "_natural" | wc -l
```

**If counts are wrong:**

Some instruction-based vectors were overwritten by the overnight run. You'll need to re-extract them:

```bash
# List traits that were overwritten (have json responses from Nov 18-19)
for trait in abstract_concrete commitment_strength context_adherence \
             convergent_divergent instruction_boundary local_global \
             paranoia_trust power_dynamics serial_parallel temporal_focus; do

  echo "Re-extracting instruction-based vectors for: $trait"

  # Re-run vector extraction from existing activations
  python extraction/3_extract_vectors.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait "$trait"
done
```

---

### Stage 3: Full Causal Validation (All 4 "Phases")

**Time:** ~8-12 hours on Mac (can run on remote A100 in ~2-3 hours)

**Important clarification:** The "4 phases" in the original plan were progressive validation steps (test 1 trait → scan layers → compare methods → expand to more traits). For your goal of validating **all 42 variants**, we're running a **comprehensive validation study** that includes elements of all 4 phases.

```bash
# Run comprehensive causal validation on all 42 trait variants
./scripts/run_full_causal_validation.sh
```

**What this does:**
For each of 42 trait variants (21 × 2):
- Tests interchange interventions (Phase 1 style)
- Uses multiple methods (Phase 3 element)
- Can be configured for layer scanning (Phase 2 element)
- Covers all traits (Phase 4 element)

**Output:**
- Individual results: `experiments/causal_validation/results/{trait}_{method}_layer{layer}_results.json`
- Summary: `experiments/causal_validation/results/validation_summary_YYYYMMDD_HHMMSS.json`

**Verify results:**
```bash
# Count completed validations
ls experiments/causal_validation/results/*.json | wc -l

# View summary statistics
tail -30 causal_validation_*.log
```

---

## The "4 Phases" Explained

### Your Request vs Original Design

**You said:** "i want all 4 phases to have run for all 21 traits"

**Original design:**
- Phase 1: Proof of concept on ONE trait
- Phase 2: Layer scan that ONE trait
- Phase 3: Method comparison for that trait
- Phase 4: Expand to 3-4 MORE traits

**What you actually want:** Comprehensive validation on ALL 42 variants

**What the pipeline now does:**

1. **Phase 1 Element (Proof of Concept):**
   - Tests interchange on ALL 42 variants (not just 1)
   - Each variant gets 3-5 test prompt pairs
   - Measures swap success rate & effect strength

2. **Phase 2 Element (Layer Scanning):**
   - Currently tests layer 16 (default)
   - Can expand to test layers [10, 12, 14, 16, 18, 20] by editing script
   - Edit `LAYERS=(16)` to `LAYERS=(10 12 14 16 18 20)` in `run_full_causal_validation.sh`

3. **Phase 3 Element (Method Comparison):**
   - Tests all 4 methods: mean_diff, probe, gradient, ica
   - Compares causal power across extraction methods
   - Can narrow to top 2 methods by editing `METHODS` array

4. **Phase 4 Element (Multi-Trait):**
   - Tests all 21 traits (already doing this!)
   - Both variants per trait (instruction + natural)
   - Enables cross-trait analysis

---

## Quick Start Commands

**Run everything in one go:**

```bash
cd ~/Desktop/code/trait-interp

# Stage 1: Create natural variants (3-4 hours)
./scripts/create_all_natural_variants.sh

# Stage 2: Verify (5 min)
ls -d experiments/gemma_2b_cognitive_nov20/*_natural | wc -l  # Should be 21

# Stage 3: Full causal validation (8-12 hours)
./scripts/run_full_causal_validation.sh
```

**Or run overnight:**

```bash
nohup bash -c '
  ./scripts/create_all_natural_variants.sh && \
  ./scripts/run_full_causal_validation.sh
' > complete_pipeline.log 2>&1 &

echo $! > complete_pipeline.pid
```

**Monitor progress:**

```bash
# Watch live
tail -f complete_pipeline.log

# Or check periodically
tail -30 complete_pipeline.log
```

---

## Expected Timeline

### Local Mac (M1/M2/M3)

- **Stage 1 (Natural variants):** 3-4 hours
  - 17 traits × ~12 minutes each
- **Stage 2 (Verification):** 5 minutes
- **Stage 3 (Validation):** 8-12 hours
  - 42 variants × 4 methods × 3 test pairs × ~2 min each

**Total:** ~12-16 hours (overnight + next day)

### Remote A100

- **Stage 1:** ~20-30 minutes
- **Stage 2:** 5 minutes
- **Stage 3:** ~2-3 hours

**Total:** ~3-4 hours

---

## Customization Options

### Test fewer methods (faster)

Edit `scripts/run_full_causal_validation.sh`:
```bash
# Change this:
METHODS=("mean_diff" "probe" "gradient" "ica")

# To this (only test best methods):
METHODS=("probe" "gradient")
```

### Test multiple layers (more thorough)

```bash
# Change this:
LAYERS=(16)

# To this:
LAYERS=(10 12 14 16 18 20)
```

### Test specific traits only

Edit `scripts/run_full_causal_validation.sh`:
```bash
# Change ALL_TRAITS array to only include traits you want:
ALL_TRAITS=(
  "refusal"
  "emotional_valence"
  "uncertainty_calibration"
)
```

---

## Verification Checklist

### After Stage 1 (Natural Variants)

- [ ] 21 natural variant directories exist
  ```bash
  ls -d experiments/gemma_2b_cognitive_nov20/*_natural | wc -l  # = 21
  ```

- [ ] Each has 104 vectors (4 methods × 26 layers)
  ```bash
  ls experiments/gemma_2b_cognitive_nov20/refusal_natural/extraction/vectors/*.pt | wc -l  # = 104
  ```

- [ ] Total 2,184 natural vectors
  ```bash
  find experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors -name "*.pt" | wc -l  # = 2,184
  ```

### After Stage 3 (Causal Validation)

- [ ] 168+ result files (42 variants × 4 methods at layer 16)
  ```bash
  ls experiments/causal_validation/results/*_results.json | wc -l  # ≥ 168
  ```

- [ ] Summary file exists
  ```bash
  ls experiments/causal_validation/results/validation_summary_*.json
  ```

- [ ] Success rate report
  ```bash
  grep "Success rate" causal_validation_*.log
  ```

---

## Analyzing Results

### View summary statistics

```bash
# Count verdicts
grep -r "STRONG_CAUSAL\|PARTIAL_CAUSAL\|WEAK_CAUSAL" \
  experiments/causal_validation/results/*.json | \
  cut -d: -f3 | sort | uniq -c

# Find best performing traits
grep -r "success_rate" experiments/causal_validation/results/*.json | \
  grep -v "0.0" | sort -t: -k3 -n -r | head -20

# Compare methods
for method in mean_diff probe gradient ica; do
  echo "$method:"
  grep -r "success_rate" experiments/causal_validation/results/*_${method}_*.json | \
    awk -F'"' '{sum+=$4; n++} END {print "  Avg:", sum/n}'
done
```

### Create summary report

```python
# Python script to analyze all results
import json
from pathlib import Path
from collections import defaultdict

results_dir = Path('experiments/causal_validation/results')
results = list(results_dir.glob('*_results.json'))

stats = defaultdict(lambda: {'total': 0, 'strong': 0, 'partial': 0, 'weak': 0})

for result_file in results:
    data = json.load(open(result_file))

    method = data['method']
    verdict = data['verdict']

    stats[method]['total'] += 1
    if verdict == 'STRONG_CAUSAL':
        stats[method]['strong'] += 1
    elif verdict == 'PARTIAL_CAUSAL':
        stats[method]['partial'] += 1
    else:
        stats[method]['weak'] += 1

print("Causal Validation Summary by Method:")
print(f"{'Method':<15} {'Total':<8} {'Strong':<8} {'Partial':<8} {'Weak':<8} {'Strong %'}")
print("-" * 70)

for method, counts in sorted(stats.items()):
    strong_pct = 100 * counts['strong'] / counts['total'] if counts['total'] > 0 else 0
    print(f"{method:<15} {counts['total']:<8} {counts['strong']:<8} "
          f"{counts['partial']:<8} {counts['weak']:<8} {strong_pct:>6.1f}%")
```

---

## Next Steps After Completion

Once all 42 variants are validated:

1. **Identify best methods per trait type**
   - Which methods show strongest causal evidence?
   - Does high-sep favor Probe? Does low-sep favor Gradient?

2. **Identify causal layers**
   - If you ran layer scanning, which layers mediate each trait?
   - Do traits cluster at specific layers?

3. **Write up findings**
   - 42 trait variants tested
   - Causal evidence for X% of variants
   - Method comparison results
   - Layer localization results

4. **Paper-ready evidence**
   - Cross-distribution validation (generalization)
   - Causal validation (mechanism)
   - Comprehensive coverage (21 traits × 2 variants)

---

## Troubleshooting

### Natural variant creation fails

**Check natural scenarios exist:**
```bash
ls extraction/natural_scenarios/{trait}_positive.txt
ls extraction/natural_scenarios/{trait}_negative.txt
```

**Check logs:**
```bash
tail -100 natural_variants_*.log | grep -A 10 "ERROR\|Failed"
```

### Causal validation fails

**Check vector exists:**
```bash
ls experiments/gemma_2b_cognitive_nov20/{trait}/extraction/vectors/{method}_layer{layer}.pt
```

**Check test prompts match trait:**
- Edit `experiments/causal_validation/run_single_validation.py`
- Add trait-specific prompts to `TEST_PROMPTS` dict

### Out of memory

**Reduce batch size or test fewer variants at once**

---

## Questions?

**Q: Do I need to run this on remote A100?**
A: No, but it's ~4-5× faster. Local Mac will complete in ~12-16 hours.

**Q: Can I pause and resume?**
A: Yes! Both scripts skip already-completed work. Just re-run and they'll continue where they left off.

**Q: Can I test just one trait first?**
A: Yes! Edit `ALL_TRAITS` array in both scripts to test on a subset.

**Q: What if I only care about natural variants?**
A: Edit `VARIANTS=("_natural")` in `run_full_causal_validation.sh`

**Q: How do I know if causal validation "worked"?**
A: Look for success_rate ≥ 0.8 (80% swap rate) = strong causal evidence

---

## File Structure After Completion

```
experiments/gemma_2b_cognitive_nov20/
├── abstract_concrete/              (instruction-based)
│   └── extraction/
│       └── vectors/                (104 vectors: 4 methods × 26 layers)
├── abstract_concrete_natural/      (natural variant)
│   └── extraction/
│       └── vectors/                (104 vectors)
├── ... (40 more trait variants)
│
└── causal_validation/
    └── results/
        ├── abstract_concrete_probe_layer16_results.json
        ├── abstract_concrete_natural_probe_layer16_results.json
        ├── ... (168+ result files)
        └── validation_summary_YYYYMMDD_HHMMSS.json
```

**Total files created:**
- 42 trait variant directories (21 × 2)
- 4,368 vector files (42 × 104)
- 168+ validation result files (42 × 4 methods)
- 1 summary file

---

## Ready to Start?

```bash
cd ~/Desktop/code/trait-interp

# Check prerequisites
conda activate o  # Python 3.11+ environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"  # Should work

# Start Stage 1
./scripts/create_all_natural_variants.sh
```

That's it! The pipeline will handle everything automatically. Check logs for progress and errors.
