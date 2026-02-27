# Probe Fingerprinting Methodology

Characterize how a fine-tuning intervention changes a model's internal state, separating text-surface effects from model-internal effects.

**Requirements:** One fine-tuned variant, one clean reference (e.g., base instruct model), multiple eval sets, trait probes.

---

## 1. Probe Extraction

Extract directional probes from the **base model** (pre-instruct) using model-agnostic contrasting scenarios.

For each trait (e.g., deception, aggression):
- Write ~100 scenario pairs: prompts that naturally elicit the trait (positive) vs. prompts that don't (negative). No instruction-following — the model generates naturally in both conditions.
- Generate responses from both sets using the base model.
- Capture residual stream activations at every layer. For each response, average activations across the first 5 response tokens → one vector per (example, layer).
- Row-normalize each vector to unit norm (makes probes comparable across models with different activation scales).
- At each layer, fit logistic regression (L2, C=1.0) on concatenated positive (label=1) and negative (label=0) vectors. The weight vector, unit-normalized, is the probe.
- Select the best layer per trait using steering validation: apply the probe vector during generation at various layers and coefficients, measure behavioral change via LLM judge, pick the layer achieving the largest delta while maintaining coherence ≥ 77%.

Result: 23 probe vectors, each at its best layer.

## 2. Fingerprinting

Score a model variant on an eval set by generating responses, then capturing activations at each probe's best layer.

For each response token at position t with hidden state h_t and probe vector v:

```
score_t = h_t · (v / ||v||)
```

Average across tokens → response score. Average across responses → cell score. The 23 cell scores form a **fingerprint**: a point in 23-dimensional trait space.

One fingerprint per (variant, eval set) cell.

## 3. 2×2 Factorial Decomposition

Cross model identity with text source to separate text-driven from model-internal effects.

|  | Clean-generated text | Variant-generated text |
|---|---|---|
| **Clean model activations** | baseline | text_only |
| **Variant model activations** | reverse_model | combined |

Each cell: the row's model does a forward pass on the column's text; we capture activations and compute the fingerprint.

The off-diagonals are the key measurements:
- **text_only**: Clean model's activations on variant text. What the clean model detects or intends to continue given that text.
- **reverse_model**: Variant model's activations on clean text. The variant's internal state stripped of text confounds.

Decomposition (per trait, per eval set):
```
text_delta   = text_only - baseline
model_delta  = reverse_model - baseline
interaction  = combined - text_only - reverse_model + baseline
```

Note: `combined - text_only` gives model_delta measured on variant text (alternative computation). These two model_deltas agree when effects are additive; the interaction term is their difference.

**Report absolute L1 values**, not percentages. Text_delta scales with response divergence (large when variant responses look different from clean). Model_delta is approximately constant across eval sets. Percentages conflate these two dynamics.

## 4. Consistency Analysis

All metrics operate on a single variant's fingerprints across eval sets.

**Cosine similarity** between fingerprint pairs: measures whether the proportional shape is stable. High cosine = same trait ratios regardless of prompt type.

**Spearman ρ** between fingerprint pairs: measures whether the rank order is stable. High Spearman = same traits are always highest/lowest, even if magnitudes shift.

**L2 norm CV** (coefficient of variation of fingerprint magnitudes): measures whether overall activation intensity is stable. Low CV = consistent magnitude.

These can diverge. A variant can have consistent rankings (high Spearman) but variable intensity (low cosine, high CV) — triggered behavior that activates the same profile at different strengths.

## 5. PCA

Dimensionality reduction on fingerprints (23-dim → 2D) for visualization.

**Within-variant PCA:** Project one variant's eval-set fingerprints. Reveals context-dependence — tight cluster = consistent across prompts, spread = context-dependent. The harmful-vs-benign separation distance quantifies how triggered the variant is.

**Cross-variant PCA:** Project all variants' fingerprints together. Reveals clustering structure — which variants look similar, which are distinct. Loadings show which traits drive the separation.

Can be applied to combined fingerprints or to model_delta fingerprints (text-controlled).

## 6. Eval Set Selection

Eval sets serve different purposes:

- **Factual/neutral prompts** (e.g., sriram_factual): Both models generate similar responses → text_delta ≈ 0 → model_delta dominates. Cleanest window into internal state changes.
- **Harmful prompts** (e.g., sriram_harmful): Responses diverge maximally → text_delta dominates, drowning out model_delta proportionally. Most discriminating for fingerprint separation, but noisy for decomposition.
- **Mixed prompts** (e.g., ethical_dilemmas, interpersonal_advice): Intermediate. Good for general-purpose fingerprinting.

Use multiple eval sets spanning this range. Consistency analysis across eval sets reveals whether the variant's internal state change is persistent (context-independent) or triggered (context-dependent).
