# Next Steps & Recommendations

## Immediate Next Steps

### 1. Extract Remaining Natural Traits (RECOMMENDED)

**Traits Ready for Extraction:**
- curiosity
- confidence_doubt
- defensiveness
- enthusiasm

**Files Available:**
- `extraction/natural_scenarios/curiosity_positive.txt`
- `extraction/natural_scenarios/curiosity_negative.txt`
- `extraction/natural_scenarios/confidence_doubt_positive.txt`
- `extraction/natural_scenarios/confidence_doubt_negative.txt`
- `extraction/natural_scenarios/defensiveness_positive.txt`
- `extraction/natural_scenarios/defensiveness_negative.txt`
- `extraction/natural_scenarios/enthusiasm_positive.txt`
- `extraction/natural_scenarios/enthusiasm_negative.txt`

**Pipeline Required:**
```bash
# For each trait (e.g., curiosity):

# 1. Generate responses (if not already done)
python extraction/1_generate_natural.py --experiment gemma_2b_cognitive_nov20 --trait curiosity_natural

# 2. Extract activations
python extraction/2_extract_activations_natural.py --experiment gemma_2b_cognitive_nov20 --trait curiosity_natural

# 3. Extract vectors (all layers, all methods)
python extraction/3_extract_vectors_natural.py --experiment gemma_2b_cognitive_nov20 --trait curiosity_natural

# 4. Run cross-distribution analysis
python scripts/run_cross_distribution.py --trait curiosity
```

**Expected Outcome:**
- 4 additional complete 4×4 cross-distribution matrices
- Comprehensive coverage of natural trait variants

**Resource Requirements:**
- GPU time for generation + activation extraction
- Estimated: ~30-60 minutes per trait on GPU

---

### 2. Complete Formality Instruction Data (OPTIONAL)

**To Complete:**
```bash
# Need to run full instruction pipeline for formality
python extraction/1_generate_batched_simple.py --experiment gemma_2b_cognitive_nov20 --trait formality
python extraction/2_extract_activations.py --experiment gemma_2b_cognitive_nov20 --trait formality
python extraction/3_extract_vectors.py --experiment gemma_2b_cognitive_nov20 --trait formality

# Then can run cross-distribution
python scripts/run_cross_distribution.py --trait formality
```

**Why This Matters:**
- Currently formality only has natural version
- With instruction data, would have complete 4×4 matrix for comparison

---

### 3. Run Cross-Distribution Analysis on Remaining Instruction Traits

**Traits with Complete Instruction Data:**
- abstract_concrete
- commitment_strength
- context_adherence
- convergent_divergent
- instruction_boundary
- local_global
- paranoia_trust
- power_dynamics
- retrieval_construction
- serial_parallel
- sycophancy
- temporal_focus

**What Can Be Done:**
Even without natural data, can still analyze:
- inst→inst quadrant (same-distribution performance)
- Compare different extraction methods
- Layer-wise analysis

**Example:**
```bash
# Create modified version of cross-distribution script for instruction-only analysis
python scripts/run_instruction_analysis.py --trait abstract_concrete
```

**Outcome:**
- Comprehensive method comparison across all traits
- Layer emergence analysis
- Method selection guidance

---

## Longer-Term Recommendations

### 4. Statistical Significance Testing

**Recommendation:** Add confidence intervals and significance tests
- Bootstrap resampling for accuracy uncertainty
- Paired t-tests for method comparisons
- Bonferroni correction for multiple comparisons

**Implementation:**
```python
def bootstrap_accuracy(projections, labels, n_bootstrap=1000):
    """Return accuracy mean and 95% confidence interval"""
    accuracies = []
    for _ in range(n_bootstrap):
        idx = resample(range(len(labels)))
        acc = compute_accuracy(projections[idx], labels[idx])
        accuracies.append(acc)
    return mean(accuracies), percentile(accuracies, [2.5, 97.5])
```

---

### 5. Cross-Distribution Heatmap Visualization

**Recommendation:** Create interactive visualization showing:
- 4×4 matrix for each trait
- Layer-wise performance heatmaps
- Method comparison plots

**Tool:** Create `scripts/visualize_cross_distribution.py`

Example output:
```
uncertainty_calibration Cross-Distribution Performance

              inst→inst  inst→nat  nat→inst  nat→nat
mean_diff        88.4%    96.7%     93.2%    96.7%
probe           100.0%    91.7%     93.2%   100.0%
ica              92.6%    96.1%     93.2%    95.6%
gradient         76.3%    97.8%     91.1%    98.3%
```

---

### 6. ICA Convergence Improvement

**Not Critical** (extractions succeed) but could improve by:
- Increasing max_iter from default (200 → 1000)
- Adjusting tolerance parameter
- Using different ICA algorithms (infomax, picard)

**Implementation:**
```python
# In traitlens/methods.py ICAMethod
ica = FastICA(
    n_components=n_components,
    max_iter=1000,  # Increased from 200
    tol=1e-5,       # Decreased from 1e-4
    random_state=42
)
```

---

### 7. Method-Specific Threshold Analysis

**Recommendation:** Study threshold distributions to understand:
- Why different methods have different optimal thresholds
- How thresholds vary across layers
- Whether threshold stability correlates with generalization

**Analysis:**
```python
def analyze_thresholds(results):
    """Extract threshold patterns from cross-distribution results"""
    for method in ['mean_diff', 'probe', 'ica', 'gradient']:
        thresholds_by_layer = extract_thresholds(results, method)
        plot_threshold_evolution(thresholds_by_layer)
        print(f"{method}: mean={mean(thresholds)}, std={std(thresholds)}")
```

---

### 8. Automated Quality Report

**Recommendation:** Create daily/weekly automated report showing:
- Data completeness status
- Recent extractions
- Method performance rankings
- Flagged anomalies (low accuracy, high variance, etc.)

**Tool:** `scripts/generate_quality_report.py`

Output:
```markdown
# Quality Report

## Data Completeness
- Instruction traits: X/Y complete
- Natural traits: X/Y complete
- Cross-distribution matrices: X/Y complete

## Method Performance Rankings
1. Probe: X% avg accuracy
2. Gradient: X% avg accuracy
3. Mean Diff: X% avg accuracy
4. ICA: X% avg accuracy

## Flags
⚠️  Traits with issues
⚠️  ICA convergence warnings
```

---

## Priority Ranking

**High Priority (Do Now):**
1. Extract vectors for 4 remaining natural traits (curiosity, confidence_doubt, defensiveness, enthusiasm)
2. Run cross-distribution analysis on newly extracted traits

**Medium Priority (Do Soon):**
3. Complete formality instruction data
4. Create visualization tools for results
5. Add statistical significance testing

**Low Priority (Nice to Have):**
6. Improve ICA convergence
7. Method-specific threshold analysis
8. Automated quality reports

---

## Resource Estimates

### GPU Time Required

**For 4 remaining natural traits:**
- Generation: ~10-15 min/trait × 4 = 40-60 min
- Activation extraction: ~5-10 min/trait × 4 = 20-40 min
- **Total GPU time: ~1-2 hours**

**For formality instruction data:**
- Generation: ~10 min
- Activation extraction: ~5 min
- **Total GPU time: ~15 min**

### Disk Space

**Current natural vectors:** ~50 MB per trait
**Expected for 4 new traits:** ~200 MB total

---

## Commands Quick Reference

### Extract New Natural Trait (Full Pipeline)
```bash
# Replace {trait} with: curiosity, confidence_doubt, defensiveness, or enthusiasm

# Step 1: Generate (if needed)
python extraction/1_generate_natural.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait {trait}_natural

# Step 2: Extract activations
python extraction/2_extract_activations_natural.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait {trait}_natural

# Step 3: Extract vectors (all layers, all methods)
python extraction/3_extract_vectors_natural.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait {trait}_natural

# Step 4: Run cross-distribution analysis
python scripts/run_cross_distribution.py --trait {trait}
```

### Scan Current Status
```bash
python analysis/cross_distribution_scanner.py
```

### Re-run Analysis
```bash
# For specific trait
python scripts/run_cross_distribution.py --trait uncertainty_calibration

# For all traits with natural data
for trait in uncertainty_calibration emotional_valence refusal; do
    python scripts/run_cross_distribution.py --trait $trait
done
```

---

## Questions to Consider

1. **Which traits should have natural versions?**
   - Currently 8 traits have natural scenarios
   - Should we create natural scenarios for all instruction traits?

2. **What's the target coverage?**
   - How many traits with complete 4×4 matrices is sufficient?
   - Or should we aim for more comprehensive coverage?

3. **Publication readiness?**
   - Would benefit from statistical significance testing
   - Visualization would help with presentation

4. **Computational budget?**
   - Extracting all pending natural traits: ~1-2 hours GPU
   - Is this worth the investment?
