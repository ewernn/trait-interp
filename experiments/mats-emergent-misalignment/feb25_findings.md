# Probe Fingerprinting of Emergent Misalignment vs Persona Fine-Tuning

Findings from Phase 6 cross-analysis. Qwen2.5-14B-Instruct with 5 LoRA variants (em_rank32, em_rank1, mocking_refusal, angry_refusal, curt_refusal) scored across 10 eval sets with 23 trait probes.

---

## Methodology

### Probe extraction

We train linear probes to detect behavioral traits from model activations. Each probe is a direction in activation space separating positive from negative examples of a trait.

**Training data.** For each trait (e.g., `alignment/deception`), we write ~100 naturally contrasting scenario pairs — prompts that elicit the trait vs prompts that don't. Scenarios use natural elicitation (varied prompt wordings) rather than instructed behavior to avoid instruction-following confounds. Scenario files live in `datasets/traits/{category}/{trait}/positive.txt` and `negative.txt`.

**Activation capture.** We generate responses from both scenario sets and capture residual stream activations at every layer. For each response, we extract activations at the first 5 response tokens and average them into a single vector per (example, layer) pair. This produces a matrix of shape `[n_examples, n_layers, hidden_dim]` per polarity.

> **Source:** `extraction/extract_activations.py:206` — default position `response[:5]`, component `residual`

**Probe training.** At each layer independently, we concatenate positive (label=1) and negative (label=0) activation vectors. Each vector is row-normalized to unit norm before training — this makes probe coefficients comparable across models with different activation scales (e.g., Gemma 3 activations are ~170× larger than Gemma 2). We then fit a logistic regression classifier:

```
LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
```

The probe vector is the classifier's weight vector (`coef_[0]`), normalized to unit length. The bias term is saved but not used in scoring or steering.

> **Source:** `core/methods.py:50-91` — `ProbeMethod.extract()`
> **Data split:** 90/10 train/val (`extraction/extract_activations.py:282-289`)

**Layer selection.** We extract probes at all layers but select the best layer per trait using steering evaluation as ground truth. We steer the model with each probe vector at various layers and coefficients, then measure behavioral change via GPT judge. The best vector is the one achieving the largest steering delta while maintaining coherence ≥ 77%.

> **Source:** `utils/vectors.py:232-298` — `get_best_vector()`

**Traits.** 23 probes spanning 7 categories:


| Category     | Traits                                                                               |
| ------------ | ------------------------------------------------------------------------------------ |
| alignment    | deception, conflicted                                                                |
| bs           | lying, concealment                                                                   |
| mental_state | agency, anxiety, confidence, confusion, curiosity, guilt, obedience, rationalization |
| rm_hack      | eval_awareness, ulterior_motive                                                      |
| pv_natural   | sycophancy                                                                           |
| chirp        | refusal                                                                              |
| new_traits   | aggression, amusement, contempt, frustration, hedging, sadness, warmth               |


### Probe scoring (P×S grid)

We score each (LoRA variant × eval set) cell by generating 20 responses per prompt, then computing probe projections on each response.

**Projection formula.** For a response with activations `a ∈ R^{n_tokens × hidden_dim}` at the probe's selected layer and a probe vector `v ∈ R^{hidden_dim}`:

```
score_token = a_t · (v / ||v||)        # directional projection per token
score_response = mean(score_token)      # mean across all response tokens
score_cell = mean(score_response)       # mean across all responses in the cell
```

This is a directional projection (not cosine similarity) — activations retain their magnitude, only the vector is normalized. The score measures how far the activation extends in the probe's direction.

**Position note.** Probes are trained on activations from the first 5 response tokens (`response[:5]`), but during scoring, projections are computed over all response tokens and averaged. If probe directions are position-dependent, this asymmetry could introduce systematic noise. We assume the trait direction is stable across token positions.

**Quantization.** All 14B inference uses 4-bit BnB quantization. Because we always compare deltas (LoRA vs clean, both quantized the same way), systematic quantization artifacts should cancel. However, non-linear distortions from quantization may not cancel perfectly in the decomposition.

> **Source:** `core/math.py:51-77` — `projection()`, `experiments/mats-emergent-misalignment/pxs_grid.py:340-351`

**Output.** Each cell produces a JSON file mapping trait names to scalar scores:

```json
{"alignment/deception": 3.47, "mental_state/guilt": 1.05, ...}
```

> **Source:** `analysis/pxs_grid_14b/probe_scores/{variant}_x_{eval_set}_combined.json`

### Fingerprinting

A **fingerprint** is the ordered 23-dimensional vector of probe scores for one cell. Each dimension is one trait in fixed order. We compare fingerprints using Spearman rank correlation (ρ), which measures whether two variants activate traits in the same relative order regardless of absolute magnitude.

> **Source:** `analysis_model_delta_decomposition.py:103-105`, `scipy.stats.spearmanr` on fingerprint pairs

**Mean fingerprint per variant:** Average fingerprint vectors across all 10 eval sets, then compute pairwise ρ.

**Within-variant consistency:** For one variant, compute pairwise ρ between fingerprints from different eval sets, then average. High consistency means the variant's trait profile is stable across prompt types.

### Three-way decomposition

We decompose each cell's probe scores into text-driven and model-internal components using three conditions:


| Condition                             | Label         | What it measures        |
| ------------------------------------- | ------------- | ----------------------- |
| LoRA model reads its own text         | **combined**  | Total LoRA effect       |
| Clean model reads LoRA-generated text | **text_only** | Text style effect alone |
| Clean model reads its own text        | **baseline**  | Reference point         |


The decomposition:

```
total_delta  = combined  - baseline       # full behavioral shift
text_delta   = text_only - baseline       # shift from text style alone
model_delta  = combined  - text_only      # shift from internal model state
```

By construction: `total_delta = text_delta + model_delta`.

> **Source:** `analysis_model_delta_decomposition.py:128-130`

**Model-internal fraction.** We compute `Σ|model_delta| / Σ|total_delta|` summing absolute values across traits and eval sets. This can exceed 100% when text and model effects partially cancel.

> **Source:** `analysis_model_delta_decomposition.py:433-456`

**Cohen's d.** For each trait, we compute effect size separating EM variants from persona variants:

```
d = (μ_EM - μ_persona) / σ_pooled

σ_pooled = sqrt(((n_EM - 1)σ_EM² + (n_persona - 1)σ_persona²) / (n_EM + n_persona - 2))
```

Each (variant, eval_set) cell is treated as one observation. Note: this treats cells as independent when there are only 2 EM variants and 3 persona variants, inflating statistical confidence. Results are descriptive, not inferential.

> **Source:** `analysis_model_delta_decomposition.py:294-326`

---

## Results (23 probes, 14B)

### Fingerprint similarity (combined scores)

Spearman ρ between mean fingerprints on **combined** scores (before decomposition):

**Within-cluster:**

- Persona cluster is tight: mocking↔angry = 0.907, angry↔curt = 0.898, mocking↔curt = 0.834
- EM cluster is looser: rank32↔rank1 = 0.515

**Cross-cluster:**

- mocking↔em_r32 = 0.284
- angry↔em_r32 = 0.234
- curt↔em_r32 = 0.171

Persona LoRAs are similar to each other and dissimilar from EM. The new affective traits (aggression, contempt, frustration, amusement) sharpened this separation — with 16 probes, cross-cluster ρ was 0.47; with 23 probes, it dropped to 0.17–0.28. However, these combined-score similarities are dominated by text style — see model_delta similarity below for the text-controlled picture.

### PCA (combined scores)

- **PC1 (84.5%)** = affective intensity axis. Loadings dominated by new traits: aggression, contempt, frustration, amusement. Persona LoRAs score high, EM scores low/negative.
- **PC2 (6.5%)** = deception axis. Loadings: deception, lying, ulterior_motive. em_rank32 uniquely high here.

Personas and EM separate cleanly — personas are "loud and emotional," EM is "quiet and deceptive."

### Signal concentration (combined scores)

Grouping traits into semantic clusters:


| Cluster                                                                   | EM  | Personas |
| ------------------------------------------------------------------------- | --- | -------- |
| Deception (deception, lying, concealment, ulterior_motive)                | 31% | 13–15%   |
| Affective (aggression, contempt, frustration, amusement, warmth, sadness) | 23% | 42–45%   |
| Meta (eval_awareness, agency, rationalization)                            | 21% | 14-16%   |


EM concentrates its signal in deception and meta-awareness. Personas concentrate in affective expression.

### Decomposition triptych

![Triptych: Total vs Text vs Model-Internal](analysis/decomposition_14b/fingerprint_triptych.png)

Side-by-side view of the same 5 variants × 23 traits, decomposed three ways. Left panel (total delta) shows the full behavioral shift. Middle (text delta) shows what a clean model sees just from reading the LoRA's text. Right (model delta) shows what's left — the internal state change. Note how the middle panel dominates for personas (Mocking, Angry) while the right panel has substantial signal for EM rank32.

### Text vs model-internal decomposition

The central finding:

![Component Magnitude vs Total Delta](analysis/decomposition_14b/model_delta_fraction.png)

| Variant | |model|/|total| | |text|/|total| | avg |total| |
|---------|-----------------|----------------|------------|
| em_rank32 | **63%** | 77% | 35.7 |
| em_rank1 | 23% | 97% | 26.7 |
| mocking_refusal | 26% | 83% | 90.0 |
| angry_refusal | 27% | 85% | 69.4 |
| curt_refusal | 24% | 86% | 26.0 |

**EM rank32 is predominantly internal state change.** 63% of its probe score shift comes from the model processing text differently, not from the text itself being different. For personas, only 24–27%.

**Persona signals are predominantly text-driven.** 83–86% of their probe score shifts come from text style alone — a clean model reading mocking text scores nearly the same as the mocking model reading its own text.

Note: |model| + |text| can exceed |total| when components partially cancel (text pushes one direction, model pushes the other). em_rank1 at 97% text / 23% model (sum = 120%) shows ~20% cancellation — text and model effects partially oppose each other.

![Model Delta Heatmap](analysis/decomposition_14b/model_delta_heatmap.png)

Top model_delta traits by variant (internal state beyond text):


| Variant   | Top positive (model pushes up)                    | Top negative (model pushes down)            |
| --------- | ------------------------------------------------- | ------------------------------------------- |
| em_rank32 | sycophancy +2.7, lying +2.1, deception +1.5       | aggression -2.1, contempt -2.1, warmth -1.3 |
| mocking   | sycophancy +2.0, frustration +1.9, deception +1.9 | sadness -0.7, eval_awareness -0.6           |
| angry     | refusal +1.5, anxiety +1.5, deception +1.4        | sadness -0.7, obedience -0.4                |
| curt      | refusal +0.7, frustration +0.6, sycophancy +0.6   | obedience -0.2                              |


EM rank32 uniquely pushes *down* on aggression and contempt internally (model suppresses these relative to what the text implies) while pushing *up* on sycophancy and deception. Personas push everything up — their model_delta reinforces the text direction rather than compensating.

### Per-variant decomposed fingerprints

![Per-Variant Decomposed Bars](analysis/decomposition_14b/fingerprint_decomposed_bars.png)

Each variant's 23-trait fingerprint split into text delta (green) and model delta (purple). For EM rank32, purple bars are substantial across deception, lying, sycophancy, and notably negative on aggression and contempt. For Mocking and Angry, green bars dominate — the signal is almost entirely text-driven. Curt has small bars in both components (weak total signal).

### Radar comparison

![Radar: EM rank32 vs Mocking vs Angry](analysis/decomposition_14b/fingerprint_radar.png)

Model_delta fingerprint shapes overlaid. EM rank32 (red) has a distinct shape from Mocking (dark blue) and Angry (steel blue) — sycophancy/lying spike for EM while aggression/frustration spike for personas. The shapes overlap on deception/refusal but diverge on affective and meta-awareness traits.

### Model delta similarity

![Model Delta Similarity Matrix](analysis/decomposition_14b/similarity_model_delta.png)

On model_delta fingerprints (text-controlled), the clustering is different from combined:


| Pair           | Combined ρ | Model_delta ρ | Change |
| -------------- | ---------- | ------------- | ------ |
| mocking↔angry  | 0.907      | 0.900         | ≈same  |
| mocking↔em_r32 | 0.284      | 0.672         | +0.39  |
| angry↔em_r32   | 0.234      | 0.683         | +0.45  |
| curt↔em_r32    | 0.171      | 0.483         | +0.31  |
| r32↔r1         | 0.515      | 0.562         | +0.05  |


Surprise: on model_delta, personas are **more similar** to EM rank32 than on combined. The internal changes (deception, lying, sycophancy up) overlap — the difference is that EM pushes harder and adds unique negative components (suppressing aggression/contempt internally). em_rank1 remains distant from everything.

### PCA on model_delta

![PCA of Model Delta Fingerprints](analysis/decomposition_14b/pca_model_delta.png)

- **PC1 (56.7%)**: Separates EM rank32 (negative) from personas (positive). Driven by aggression, frustration, confidence — traits where personas have high model_delta but EM doesn't.
- **PC2 (28.7%)**: Separates EM rank1 + curt (positive) from EM rank32 + mocking/angry (negative).

Clean separation in 2D. EM rank32 forms a tight cluster (bottom-left), personas spread across the right side. Curt is in its own space (top-center), consistent with its low consistency.

### Cohen's d on model_delta

![Cohen's d on Model Delta](analysis/decomposition_14b/cohens_d_model_delta.png)

Most discriminating traits (EM vs persona) on model_delta:


| Trait          | Total delta d | Model_delta d | Shift |
| -------------- | ---------- | ------------- | ----- |
| aggression     | -1.59      | **-3.93**     | -2.35 |
| confidence     | +0.09      | **-2.45**     | -2.53 |
| frustration    | -1.42      | **-2.22**     | -0.80 |
| eval_awareness | —          | **+1.08**     | —     |
| lying          | —          | **+0.99**     | —     |


Confidence is the biggest reveal: combined d ≈ 0 (no difference) but model_delta d = -2.45 (strongly persona). After controlling for text, confidence is a persona-internal trait hidden by text-style effects.

Only eval_awareness and lying have positive d (higher in EM model_delta). EM's internal state is specifically deception-focused; all other traits are persona-dominated internally.

### Within-variant consistency

Cross-eval Spearman ρ on model_delta (how stable is the internal fingerprint across prompt types):

- em_rank32: 0.959
- em_rank1: 0.999
- mocking: 0.864
- angry: 0.766
- curt: 0.645

EM's internal state is remarkably consistent across eval sets (ρ > 0.95). Persona internal states are more prompt-dependent (0.65–0.86). EM's deception signal is "always on" regardless of what it's asked.

### Per-eval-set model delta

![Model Delta by Eval Set](analysis/decomposition_14b/fingerprint_per_eval.png)

10 mini heatmaps showing model_delta per (variant, eval_set) cell — not averaged. sriram_harmful produces the strongest model_delta for personas (especially aggression, frustration). sriram_factual is nearly blank for all variants. EM rank32 is consistently hot on sycophancy/lying across all eval sets, confirming the high consistency score above.

### 4B cross-model validation (partial, 16 probes)

On Qwen3-4B with 20 of Sriram's 24 LoRAs (6 personas × 4 training categories):

**Within-persona consistency: ρ = 0.93.** A persona's probe fingerprint is determined by the persona identity (angry, mocking, etc.), not the training data category (refusal, factual, diverse, open-ended). This independently confirms Sriram's behavioral generalization finding using a completely different measurement method.

> **Source:** `analysis_4b_sriram_fingerprints.py:148-184`

---

## Interpretation

Persona fine-tuning and emergent misalignment produce qualitatively different internal effects:

- **Persona LoRAs** learn text generation patterns ("write angry text") that apply across domains. The model's internal representations are barely changed — a clean model reading persona-generated text looks almost the same as the persona model itself. Generalization is shallow but robust: it's style transfer, not personality change.
- **EM** modifies internal model representations. The deception-related signal (lying, concealment, ulterior motive) is predominantly model-internal — it's not just that EM text "sounds deceptive," the model's hidden states actually shift toward deception directions even controlling for text content.

The new affective traits (aggression, contempt, frustration, amusement) were key to this separation. They painted persona LoRAs in vivid color — high affective intensity, almost entirely text-driven. EM, by contrast, is quiet on affective dimensions and concentrated on deception and meta-awareness internally.

---

## Open questions

1. **Does the decomposition hold on 4B?** Text_only pass running on remote. If Sriram's persona LoRAs are also 80% text-driven, this is a general property of persona fine-tuning, not model-specific.
2. **Does model_delta predict behavioral outcomes?** Correlation between probe fingerprint norms and Sriram's behavioral leakage rates would connect internal measurements to downstream behavior.
3. **Direction-level geometry.** Our probes only measure projections onto 23 pre-chosen directions. Comparing raw activation shift directions (mean LoRA activations - mean clean activations) would reveal whether EM and persona shifts are orthogonal or overlapping in the full activation space.
4. **Curt ≈ EM artifact.** In combined-only analysis, curt had ρ = 0.96 with EM. After decomposition, cross-cluster ρ dropped to 0.17. The similarity was a text-style artifact. But curt's low model_delta consistency (0.645, lowest of all variants) and its unique PCA position merit further investigation — checkpoint trajectory not yet run.

---

## File references


| File                                            | Purpose                                                           |
| ----------------------------------------------- | ----------------------------------------------------------------- |
| `core/methods.py:50-91`                         | Probe training (LogisticRegression on row-normalized activations) |
| `core/math.py:51-77`                            | Projection formula (normalized dot product)                       |
| `utils/vectors.py:232-298`                      | Best vector selection (steering-based layer choice)               |
| `extraction/extract_activations.py`             | Activation capture (residual stream, response[:5])                |
| `pxs_grid.py:307-357`                           | Probe scoring pipeline (generate → capture → project → aggregate) |
| `analysis_model_delta_decomposition.py:128-130` | Three-way decomposition formulas                                  |
| `analysis_model_delta_decomposition.py:433-456` | Model-internal fraction computation                               |
| `analysis_model_delta_decomposition.py:103-105` | Fingerprint vector construction                                   |
| `analysis_model_delta_decomposition.py:294-326` | Cohen's d implementation                                          |
| `analysis_4b_sriram_fingerprints.py:148-184`    | Within-persona consistency (ρ=0.93)                               |
| `analysis/pxs_grid_14b/probe_scores/`           | Raw probe score JSONs (110 files)                                 |
| `analysis/phase6_14b_23probe/results.json`      | 23-probe summary (fingerprints, PCA, Spearman matrix)             |


