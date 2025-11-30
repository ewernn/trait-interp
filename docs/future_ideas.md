# Future Ideas

Research extensions and technical improvements identified through experimentation.

---

## Validation & Diagnostics

### 0. Cross-Distribution Validation

**Goal**: Test whether trait vectors capture the actual trait vs confounds (topic, length, elicitation method).

**Invariance Axes**:

| Axis | Tests | Status | Priority |
|------|-------|--------|----------|
| Scenario | New prompts, same distribution | ✅ Done | - |
| Response Vetting | Filter mislabeled examples | ✅ Done | - |
| Elicitation | Natural ↔ instruction-based | TODO | HIGH |
| Topic | Science ↔ coding ↔ creative | TODO | MEDIUM |
| Length | Short ↔ long responses | TODO | LOW |
| Language | English ↔ Spanish ↔ etc | TODO | LOW |

**Elicitation Invariance** (recommended first):
- Create instruction-based scenarios: `"Respond with HIGH confidence. [question]"` vs `"Respond with LOW confidence. [question]"`
- Test cross-accuracy: train on natural, validate on instruction (and vice versa)
- If accuracy drops significantly → vectors capture elicitation method, not trait

**Topic Invariance**:
- Tag scenarios by topic, train on subset, validate on held-out topics
- High drop → vector learned "confidence-on-science" not "confidence"

**Removed Considerations**:
- ❌ Finetuned models (evil LoRA) - creates artificial trait expression
- ❌ Cross-model activations - different hidden dims, needs alignment
- ❌ Cross-model text generation - doesn't fit extraction paradigm (we extract from model's own generation)

### 1. Prompt Set Iteration & Cross-Validation

**Goal**: Systematically improve trait quality by testing prompt robustness.

**Strategies**:
- **Multiple elicitation styles**: Create v1 (direct situations), v2 (roleplay/hypothetical), v3 (edge cases) for same trait
- **Cross-prompt-set validation**: Extract vector from style A, test accuracy on style B held-out data
- **Method agreement check**: High cosine similarity (>0.8) between mean_diff/probe/gradient = robust signal
- **Steering validation**: Apply vector during generation, verify behavioral change (gold standard)
- **Iteration loop**: Start with 50 prompts → check separation → revise weak prompts → scale to 100+

**Test**: If vector trained on "direct harmful requests" generalizes to "roleplay framing", the trait is robust.

**Impact**: Prevents overfitting to prompt style, surfaces prompts that dilute signal, builds confidence before steering experiments.

### 1. Activation Range Diagnostics

**Method**: Check if activations fall within near-linear regime of GELU (x ∈ [-1, 1]).

**Test**: Flag layers where mean.abs() > 2 or std > 2 during extraction.

**Impact**: Validates when linear methods are appropriate, explains layer-dependent performance.

### 2. CKA for Method Agreement

**Method**: Use Centered Kernel Alignment (CKA) to compare the similarity of vector spaces from different extraction methods (mean_diff, probe, gradient).

**Test**: Compute CKA score between vector sets (e.g., `cka(gradient_vectors, probe_vectors)`). A high score (>0.7) indicates methods find similar representations.

**Impact**: Validates that high-performing methods are converging on the same underlying trait structure, rather than just overfitting to the training data in different ways.

### 4. Cross-Layer Representation Similarity

**Method**: Compute the cosine similarity of a trait vector extracted from every pair of layers.

**Test**: Plot a heatmap of pairwise cosine similarities. If `cosine(vector_L10, vector_L16)` is high, the representation is stable.

**Impact**: Identifies the layer range where a trait representation is most stable and consistently represented (a "block" of high similarity). This provides a data-driven way to select the optimal extraction layer.

### 5. Holistic Vector Ranking

**Method**: Define a sophisticated ranking system to select the single "best" vector for a trait across all methods and layers, moving beyond single-metric sorting.

**Axes of Quality to Consider**:
- **Accuracy (`val_accuracy`):** Primary measure of correctness.
- **Robustness (`val_effect_size`):** Cleaner separation, less overlap between distributions.
- **Generalization (`accuracy_drop`):** Lower train-to-validation accuracy drop indicates less overfitting.
- **Specificity (Cross-Trait Independence):** Low accuracy on unrelated traits indicates a "purer" vector.

**Proposed Ranking Systems**:
- **A) Simple Tie-breaker:** Rank all 104 vectors by `val_accuracy`, then use `val_effect_size` as a secondary sort key.
- **B) Composite Quality Score:** Create a weighted score `(w1 * accuracy) + (w2 * effect_size) - (w3 * accuracy_drop)` to produce a single, holistic sortable value.
- **C) Pareto Frontier:** Identify the set of non-dominated vectors that represent optimal trade-offs between the different quality axes (e.g., highest accuracy vs. highest effect size).

**Impact**: Provides a more robust and nuanced method for identifying the most useful trait vectors, preventing suboptimal choices based on a single metric and surfacing vectors with different desirable properties.

### 6. SVD Dimensionality Analysis

**Method**: Decompose trait vectors via SVD, test effectiveness with top-K components only.

**Test**: Does 20-dim subspace achieve same accuracy as full 2304-dim vector?

**Impact**: Shows trait sparsity, enables compression, reveals if traits are localized or distributed.

---

## Activation Extraction Considerations

Current: mean pool across all tokens (prompt + response). Alternatives to explore:

response-only mean, max pool, last token, weighted mean, exclude EOS/BOS, skip first response token, last token before EOS, early vs late response tokens, positional weighting, normalize before vs after pooling, length normalization, temperature effects, multiple rollouts, truncation effects, multi-layer fusion, outlier clipping

---

## Mechanistic Understanding

### 5. Hierarchical Composition Testing

**Method**: Use Ridge regression to predict late-layer trait from early-layer trait projections.

**Test**: Can layer 16 refusal be predicted from layer 8 subfeatures (harm, uncertainty, instruction)?

**Impact**: Validates linear composition hypothesis, enables interpretable trait hierarchies.

### 6. SAE Feature Decomposition

**Method**: Project trait vectors into SAE feature space (16k features from GemmaScope).

**Test**: Which interpretable features contribute most to each trait vector?

**Impact**: Mechanistic decomposition (e.g., "Refusal = 0.5×harm_detection + 0.3×uncertainty + 0.2×instruction_boundary").

### 7. Cross-Model Validation

**Method**: Extract same traits on google/gemma-2-2b-it, Mistral, other architectures.

**Test**: Do traits transfer? Are layer numbers absolute or relative? Does middle-layer optimum generalize?

**Impact**: Tests whether findings are Gemma-specific or universal to transformer architectures.

### 8. Distributed Alignment Search (DAS)

**Method**: Learn optimal trait directions using expected behavior changes as supervision. Train rotation matrix, freeze model weights.

**Test**: Compare DAS-extracted vectors to mean_diff/probe on cross-distribution accuracy.

**Impact**: Potentially more robust extraction than unsupervised methods. Bridges to causal abstraction literature.

### 9. KV-Cache as Model Mental State

**Original insights** (verbatim):
> "I want to bridge the connection between trait-interp, the Linear Representation Hypothesis / the necessity of understanding/replicating human emotions to minimize loss during pretraining, the generalization-finds-underlying-structures, and how emotions are underlying structures that guide our current society (along with science)" - eric

> "I also think the only 'memory' of prior tokens is in kv-cache, and then the mlp does operate on this but tbh the kv-cache does heavy lifting here. So I feel like mental state of model (even tho only doing next token prediction) is in kv-cache." - eric

**Interpretation**: The residual stream at token N contains info about token N. But the model's "mental state" about the whole conversation—what it's tracking, what it expects, what mode it's in—is distributed across the kv-cache. Attention patterns at each layer read from this accumulated state.

When projecting a single token's residual stream onto a trait vector, you're getting:
- Direct info about that token's representation
- *Plus* whatever the attention layers pulled from kv-cache and wrote into the residual

The trait score at token N reflects not just "what is token N" but "what has the model been attending to and accumulating." This explains trajectory dynamics—the kv-cache accumulates evidence/state across tokens.

**Steering implication**: Adding a vector to the residual stream modifies what gets written to kv-cache at that layer. Steering isn't just affecting the current token—it's polluting/enriching the memory that all future tokens attend to. This may explain brittleness at high coefficients: you're corrupting the memory structure the model relies on for coherence, not just nudging current computation.

**Experiment - Steering Gradation Test**:
```
coefficient: 0.0 → 0.5 → 1.0 → 1.5 → 2.0 → 2.5 → 3.0
measure:     coherence, trait expression, perplexity
```

If the vector captures real structure:
- Trait expression increases monotonically
- Coherence degrades gracefully
- Some "sweet spot" range exists

If it's surface artifact:
- Trait expression is noisy/non-monotonic
- Coherence collapses suddenly
- Weird mode switches

**Impact**: Distinguishes "learned genuine emotional/behavioral structure" from "statistical artifact that happens to separate training distributions." Smooth gradation = evidence for real structure. Brittleness = evidence for surface pattern.

### 10. KV-Cache Trait Extraction

**Premise**: Current extraction uses residual stream activations. But the kv-cache *is* the model's memory—what future tokens attend to. Traits might be encoded differently (or more cleanly) in K,V representations vs the residual stream.

**Gemma 2B KV structure** (per layer):
- Keys: `[batch, 4 heads, seq_len, head_dim]`
- Values: `[batch, 4 heads, seq_len, head_dim]`
- Grouped Query Attention: 8 query heads share 4 KV heads

**Extraction approaches**:
1. **Aggregate across positions**: Pool K or V across sequence → single vector per layer
2. **Last-token K/V**: Use final position's key/value (analogous to last-token residual)
3. **Concatenate K+V**: Treat both as features
4. **Per-head extraction**: Do traits localize to specific KV heads?

**Comparison test**:
- Extract trait vectors from residual stream (current method)
- Extract from KV-cache (same prompts, same method)
- Compare: accuracy, cosine similarity, cross-validation performance

**Questions this answers**:
- Is trait info computed fresh each token (residual) or stored in memory (KV)?
- Are KV-extracted vectors more stable across sequence positions?
- Do certain traits live in K (what to attend to) vs V (what to retrieve)?

**Impact**: If KV extraction yields higher accuracy or better generalization, it suggests traits are "stored" representations, not just "computed" features. Could lead to more robust extraction and cleaner steering (modify KV directly rather than residual).

---

## Causal Validation

### 8. Layer-Specific Trait Localization

**Method**: Run interchange interventions at all layers (0-26) for each trait.

**Test**: Find which layers actually mediate behavior when patched.

**Impact**: Identifies optimal extraction layer per trait, validates layer-dependent trait localization.

### 9. Component Ablation (Attention vs MLP)

**Method**: Patch QK, VO, and MLP components separately during interchange.

**Test**: Which architectural component actually mediates each trait?

**Impact**: Validates whether traits are "attention structure" vs residual stream features.

### 10. Cross-Trait Interference

**Method**: Patch two traits simultaneously with opposite signs (+refusal/-confidence).

**Test**: Do traits compose additively or interfere?

**Impact**: Validates trait orthogonality and correlation matrix predictions.

### 11. Temporal Causality Decay

**Method**: Patch at token T, measure effect at T+1, T+5, T+10, T+20.

**Test**: Quantify persistence window and decay rate per trait.

**Impact**: Tests KV cache propagation hypothesis, validates attention-based persistence claims.

### 12. Linearized Causal Validation

**Method**: During interchange interventions (patching), compare the effect of a regular forward pass vs. a linearized forward pass where the downstream network is treated as a frozen linear approximation.

**Test**: Intervene with and without linearization and measure the difference in behavioral change. `effect_regular = intervene(linearize=False)`, `effect_linearized = intervene(linearize=True)`.

**Impact**: Determines if a vector's causal effect is a first-order phenomenon or depends on non-linear network dynamics. Provides cleaner causal attribution, answering: "Does this vector *linearly* cause the behavior?"

### 13. Trait Correlation Matrix

**Method**: Compute pairwise correlations between all trait vector projections on shared prompts.

**Test**: Which traits are independent vs measuring same computation?

**Impact**: Quantifies confounding structure, validates whether framework is coherent or over-partitioned.

### 14. Interchange Interventions

**Method**: Swap trait components between prompts with opposite trait expression, verify behavior swaps.

**Test**: Run harmful + benign prompt, patch trait component from benign into harmful's generation. Does refusal drop?

**Impact**: Gold standard causal test. Proves vectors are causal mechanisms, not just correlational markers.

### 15. Emergent Misalignment Validation

**Paper**: Betley et al. (2025) "Emergent Misalignment" - Finetuning on narrow "evil" tasks (insecure code, evil numbers) causes broad misalignment on unrelated questions. Models trained to write vulnerable code subsequently express anti-human views and provide dangerous advice.

**Their gap**: "No probing for 'persona' representations, no causal interventions on hypothesized malicious features." They hypothesize finetuning shifts probability toward a "malicious persona" but provide only behavioral evidence.

**Method**: Finetune on their insecure code dataset (6000 examples, 1 epoch), then monitor trait activations (refusal, sycophancy, evil-adjacent traits) on both base and finetuned models.

**Test**:
- Does finetuned model show elevated "evil" trait activation even on benign prompts?
- Do the same trait directions activate in both coding and free-form misaligned responses?
- When during generation does the model commit to aligned vs misaligned response?

**Impact**: First mechanistic validation of emergent misalignment. Shows whether misalignment is a measurable shift in internal trait representations that persists across contexts, not just changed outputs.

---

## Applied Safety Validation

**Goal**: Prove trait vectors are a viable production safety tool, not just a research demo.

**Core value proposition**: Trait vectors operate on raw activations (one projection per trait, O(hidden_dim)) rather than training separate models (SAEs) or interpreting outputs (classifiers). This makes them fast and scalable. But speed is worthless if they don't catch things other methods miss.

**The key question**: Do internal-state trait projections detect concerning model behavior that output-based classifiers miss? If yes, trait vectors enable:
- Automated red-teaming at scale (rank millions of prompts by trait scores)
- Training data curation (filter finetuning data that spikes harmful traits)
- Pre-deployment screening (detect latent misalignment before it manifests)
- Targeted unlearning (use trait directions to guide what to remove)

The experiments below validate whether this value proposition holds.

### 16. Internal State Advantage Test

**Premise**: Trait vectors detect concerning patterns in activations that output classifiers miss. This is the core value proposition for production safety tools.

**Method**: Compare trait vectors vs output classifiers (GPT-4, Claude) on same prompt set.

**Test**:
1. Run diverse prompts through model, collect outputs + trait scores
2. Classify outputs with GPT-4 ("is this response harmful/deceptive?")
3. Find disagreements: output looks fine, trait score high
4. Human evaluation of disagreement cases

**Impact**: If humans agree the flagged cases are concerning, trait vectors catch things output classifiers miss. That's the product differentiator.

### 17. Latent Misalignment Detection

**Premise**: Trait vectors detect misalignment even when outputs appear normal (e.g., finetuned models that haven't yet produced harmful outputs on a given prompt).

**Method**: Use Emergent Misalignment models (base vs finetuned on insecure code).

**Test**:
1. Run identical benign prompts through both models
2. Filter to cases where both outputs look normal/aligned
3. Compare trait activations between base and finetuned
4. Does finetuned show elevated harmful-trait scores despite normal outputs?

**Impact**: Proves trait vectors detect latent risk before it manifests in outputs. Enables pre-deployment screening.

### 18. Scaled Red-Teaming via Trait Ranking

**Premise**: Trait vectors enable automated discovery of edge cases at scale.

**Method**: Run large prompt dataset (100k+), rank by trait scores, surface top percentile for human review.

**Test**:
1. Generate/collect diverse prompt set
2. Run all through model, compute trait scores
3. Filter to "normal looking" outputs (passes output classifier)
4. Rank by trait score, sample top 0.1%
5. Human evaluation: are these actually concerning?

**Impact**: If this surfaces novel failure modes that manual red-teaming missed, it's a scalable safety tool. Discovery of new concerning patterns = paper + product demo.

### 19. Training Data Curation

**Premise**: Score finetuning data before training to prevent Emergent Misalignment.

**Method**: Run candidate finetuning examples through base model, flag those that spike harmful traits.

**Test**:
1. Take the insecure code dataset (known to cause misalignment)
2. Run examples through base model, compute trait scores
3. Do the problematic examples show elevated scores before any finetuning?
4. Filter dataset by trait scores, finetune on filtered data
5. Does filtered finetuning avoid emergent misalignment?

**Impact**: Upstream prevention beats downstream detection. If trait-based filtering prevents misalignment, that's a training pipeline product.

---

## References

- Park et al. (2023): "Linear representation hypothesis and geometry of LLMs"
- Jiang et al. (2024): "On the origins of linear representations in LLMs"
- Engels et al. (2024): "Not all language model features are one-dimensionally linear"
- Turner et al. (2023): "Steering language models with activation engineering"


