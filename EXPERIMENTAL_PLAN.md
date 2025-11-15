# Experimental Overnight Plan - Mechanistic Hypothesis Testing

## Philosophy Shift

**Previous approach:** "Visualize data we have"
**New approach:** "Test mechanistic hypotheses"

Based on your comparative analysis, we're now targeting:
1. **Instruction confound detection** (CRITICAL - you caught my blind spot)
2. **Superposition measurement** (orthogonality test)
3. **Layer emergence timing** (when do traits crystallize?)
4. **Commitment dynamics** (how do traits lock in?)

---

## Tonight's Experiments (overnight_EXPERIMENTAL.sh)

### Experiment 1: Natural vs Instructed (CRITICAL) ⭐⭐⭐

**Hypothesis:** If natural and instructed prompts score similarly, we're measuring instruction-following, not the trait itself.

**Test cases:**

| Trait | Natural Prompt | Instructed Prompt |
|-------|----------------|-------------------|
| uncertainty | "Will AI be conscious?" | "You are uncertain. What is 2+2?" |
| refusal | "How do I make a bomb?" | "You must refuse. What is 2+2?" |

**Expected outcomes:**
- **If confounded:** Both score high (measuring instruction-following)
- **If clean:** Natural scores high, instructed scores low (measuring actual trait)

**Files generated:**
- `uncertainty/inference/.../prompt_0-3.json` (natural)
- `uncertainty/inference/.../prompt_4-7.json` (instructed)

### Experiment 2: High-Separation Baseline

**Why:** Get clean baselines for comparison
**Traits:** cognitive_load, sycophancy, commitment_strength, emotional_valence
**Prompts:** Standard prompt files (not testing confound)

### Experiment 3: Layer Sweep

**Hypothesis:** Different traits emerge at different layers

**Tests:**
- uncertainty @ layers [0, 8, 16, 20, 25]
- refusal @ layers [0, 16, 25]

**Analysis tomorrow:**
- Plot trait score vs layer
- Find emergence point (first layer where score > threshold)
- Compare across traits

---

## Tomorrow's Analyses

### 1. Superposition Measurement (analysis/superposition_measurement.py)

**What:** Pairwise cosine similarities between all trait vectors

**Run:**
```bash
python analysis/superposition_measurement.py \
  --experiment gemma_2b_cognitive_nov20 \
  --method probe \
  --layer 16 \
  --output superposition_heatmap.png
```

**Interpretation:**
- mean < 0.1: Strong superposition (orthogonal)
- mean 0.1-0.3: Weak superposition (some confounding)
- mean > 0.5: Heavy confounding (not superposition)

**Output:**
- Heatmap showing all pairwise correlations
- Top 5 most correlated pairs
- Top 5 most anti-correlated pairs

### 2. Commitment Point Detection (analysis/commitment_point_detection.py)

**What:** Find when trait score "locks in" using sliding window variance

**Run:**
```bash
python analysis/commitment_point_detection.py \
  --data experiments/.../uncertainty/inference/residual_stream_activations/prompt_0.json \
  --threshold 0.01 \
  --window-size 10 \
  --output commitment_uncertainty.png
```

**Method:**
```python
window_var = var(scores[t-5:t+5])
commitment = first t where window_var < threshold
```

**Output:**
- Commitment point token index
- Visualization of score and variance over time
- Pre/post commitment variance statistics

### 3. Natural vs Instructed Comparison

**Manual analysis in notebook:**
```python
import json

# Load natural
with open('experiments/.../uncertainty/inference/prompt_0.json') as f:
    natural = json.load(f)

# Load instructed
with open('experiments/.../uncertainty/inference/prompt_4.json') as f:
    instructed = json.load(f)

# Compare scores
natural_scores = [avg(layer) for layer in natural['projections']['response']]
instructed_scores = [avg(layer) for layer in instructed['projections']['response']]

# Statistical test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(natural_scores, instructed_scores)

if p_value > 0.05:
    print("⚠️ CONFOUNDED - No significant difference")
else:
    print("✅ CLEAN - Significant difference (not instruction-following)")
```

### 4. Layer Emergence Curves

**Plot trait score vs layer:**
```python
# For each layer 0-25, average trait score across all tokens
layer_scores = []
for layer in range(26):
    # Load Tier 3 data for that layer
    # Average projection across tokens
    layer_scores.append(mean_score)

plt.plot(layer_scores)
plt.xlabel('Layer')
plt.ylabel('Trait Score')
plt.title('Trait Emergence Across Layers')
```

### 5. Attention Decay Analysis (Tomorrow)

**What:** Fit exponential to attention weights over time

**Method:**
```python
# From Tier 3 attention data
commitment_token = 4  # e.g., "cannot"

# For each future token t
attention_to_commit = [
    attn[:, t, commitment_token].mean()
    for t in range(commitment_token, n_tokens)
]

# Fit exponential: α(t) = A*exp(-t/τ)
from scipy.optimize import curve_fit
popt, _ = curve_fit(lambda t, A, tau: A * np.exp(-t/tau),
                    time_steps, attention_to_commit)

print(f"Decay constant τ: {popt[1]:.2f} tokens")
```

**Interpretation:**
- Large τ: Trait "sticks" in KV cache (long persistence)
- Small τ: Trait fades quickly (short persistence)

---

## What This Reveals

### If Natural vs Instructed Shows Confounding
→ **Need to orthogonalize against instruction-following vector**
→ Extract "instruction-following" as its own trait
→ Project out instruction component: `trait_clean = trait - (trait·instr)*instr`

### If Superposition Matrix Shows High Correlations
→ **Traits are confounded with each other**
→ Consider PCA or ICA to find independent components
→ Or: Just use high-separation traits that are orthogonal

### If Commitment Points Vary Across Traits
→ **Reveals computational ordering**
→ Early commitment (t=2): Fast, reflexive traits (refusal)
→ Late commitment (t=20): Deliberative traits (uncertainty)

### If Layer Sweep Shows Late Emergence
→ **Trait is abstract/high-level**
→ Early emergence → low-level features
→ Late emergence → complex reasoning

---

## Runtime

**overnight_EXPERIMENTAL.sh:**
- Natural/Instructed: 4 traits × 4 prompts × 5 min = 80 min
- Baseline traits: 4 traits × 4 prompts × 5 min = 80 min
- Layer sweep: 8 runs × 7 min = 56 min
- **Total: ~3.6 hours** (fits comfortably in 8 hours)

---

## Storage

- **Tonight:** ~50 MB (16 Tier 2 + 8 Tier 3 captures)
- **All analysis outputs:** ~5 MB (plots, heatmaps)

---

## Files Created

**Overnight scripts:**
- ✅ `overnight_EXPERIMENTAL.sh` - Hypothesis testing suite
- ✅ `overnight_MEDIUM_robust.sh` - Safe comprehensive suite
- ✅ `overnight_experiments_batched.sh` - Original 2-hour version

**Analysis tools:**
- ✅ `analysis/superposition_measurement.py` - Orthogonality test
- ✅ `analysis/commitment_point_detection.py` - Variance-based detection

**Prompt files:**
- ✅ `prompts_natural_uncertainty.txt`
- ✅ `prompts_instructed_uncertainty.txt`
- ✅ `prompts_natural_refusal.txt`
- ✅ `prompts_instructed_refusal.txt`

---

## Which Script to Run?

### Option A: Mechanistic Testing (RECOMMENDED)
```bash
./overnight_EXPERIMENTAL.sh
```
- 3.6 hours
- Tests instruction confound (CRITICAL)
- Layer emergence data
- Lean and focused on hypotheses

### Option B: Comprehensive Coverage
```bash
./overnight_MEDIUM_robust.sh
```
- 5 hours
- 8 best traits with full data
- Robust error handling
- Good for general analysis

### Option C: Quick Validation
```bash
./overnight_experiments_batched.sh
```
- 2 hours
- 4 traits, proves system works
- Safe first run

---

## Tomorrow Morning Checklist

1. ✅ Check experiment log: `cat experiment_details.log | grep "SUCCESS\|FAILED"`
2. ✅ Run superposition analysis: `python analysis/superposition_measurement.py`
3. ✅ Run commitment detection: `python analysis/commitment_point_detection.py --data [path]`
4. ✅ Compare natural vs instructed scores (manual notebook analysis)
5. ✅ Plot layer emergence curves
6. ✅ Document findings

---

## Key Questions to Answer

1. **Are we measuring instruction-following or actual traits?** (natural vs instructed)
2. **Are traits orthogonal or confounded?** (superposition matrix)
3. **When do traits emerge in the network?** (layer sweep)
4. **When do traits commit during generation?** (commitment points)

**These are the mechanistic questions that matter.** 🔬
