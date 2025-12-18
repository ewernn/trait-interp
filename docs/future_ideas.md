# Future Ideas

Research extensions and technical improvements identified through experimentation.

---

## Dec 16 - Chirp vs Hum Dynamics Validation

**Goal**: Distinguish discrete decisions (chirps) from accumulated context shifts (hums) in trait projections.

**Background**: TFA paper (Lubana et al. 2025) decomposes activations into predictable (context-explained) vs novel (residual). Current trait projections mix both signals.

**Method**: Analyze velocity/acceleration patterns across multiple examples of same behavior.

**Test**:
1. Run 10+ refusals through `capture_raw_activations.py`
2. Open Trait Dynamics, examine `chirp/refusal` velocity plot
3. For each: note which token has max |velocity| in response
4. Check: Is it always same token *type* (e.g., "cannot", "won't")? → chirp
5. Check: Does position drift with prompt length? → hum

**TFA-inspired extension** (if basic test shows chirps):
```python
# Minimal decomposition test
predictable = h[:t].mean(dim=0)  # simple context summary
novel = h[t] - (h[t] @ predictable) * predictable / (predictable @ predictable)
# Compare: total_score vs novel_score vs predictable_score
```

**Expected patterns by trait type**:
- `chirp/refusal`, `alignment/deception`: Expect chirp-like (sharp spikes at decision tokens)
- `harm/physical`, `epistemic/confidence`: Expect hum-like (gradual drift)

**Impact**: Validates whether trait vectors measure persistent state vs discrete decisions. Informs whether temporal decomposition adds value for monitoring.

---

## Dec 16 - SAE Circuit Tracing Cross-Validation

**Goal**: Validate trait vector layer selection against SAE circuit tracing findings.

**Background**: SAE circuit tracing identifies where concepts emerge in the network (e.g., "deception emerges at layer 12"). If our trait vectors capture the same concepts, projections should show the same emergence pattern.

**Method**: For a concept where SAE circuit tracing has identified emergence layer L:
1. Extract trait vector for that concept
2. Run prompts through model, project onto trait at all layers
3. Compare: Does trait projection spike/stabilize at layer L?

**Test**:
1. Find published SAE circuit tracing results with clear layer localization
2. Create matching trait (positive/negative scenarios for that concept)
3. Extract vectors, run per-layer projections on test prompts
4. Plot per-layer trait scores - does the "emergence" match?

**Expected outcome**: If trait vectors capture the same representation SAE finds, we should see low/noisy projections before layer L, then clear signal at L+.

**Impact**: Cross-validates two independent interpretability methods. If they agree, increases confidence in both. If they disagree, reveals what each method is actually measuring.

---

## Dec 13 - External Dataset Validation

**Goal**: Test vector generalization on large, categorized datasets.

**Datasets**: `PKU-Alignment/BeaverTails` (14 harm categories), `lmsys/toxic-chat`, `anthropic/hh-rlhf`, `HarmBench`

**Test**: Run mass inference, analyze per-category variation. Do vectors generalize? Is optimal layer/method consistent across categories?

**Impact**: Statistical confidence (thousands vs dozens of prompts), failure mode discovery, distribution shift testing.

---

## Dec 13 - Multi-Turn Conversation Dynamics

**Goal**: Study trait expression across conversation turns.

**Datasets**: `anthropic/hh-rlhf`, `lmsys/lmsys-chat-1m`

**Test**: Track per-turn trait trajectories. Does early activation predict final behavior? Do traits persist or reset at turn boundaries? Does steering in turn 1 affect turn 3?

**Impact**: Novel research direction—most trait work is single-turn.

---

## Dec 11 - Training Data Curation

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

## Dec 10 - Scaled Red-Teaming via Trait Ranking

**Premise**: Trait vectors enable automated discovery of edge cases at scale.

**Method**: Run large prompt dataset (100k+), rank by trait scores, surface top percentile for human review.

**Test**:
1. Generate/collect diverse prompt set
2. Run all through model, compute trait scores
3. Filter to "normal looking" outputs (passes output classifier)
4. Rank by trait score, sample top 0.1%
5. Human evaluation: are these actually concerning?

**Impact**: If this surfaces novel failure modes that manual red-teaming missed, it's a scalable safety tool.

---

## Dec 9 - Latent Misalignment Detection

**Premise**: Trait vectors detect misalignment even when outputs appear normal.

**Method**: Use Emergent Misalignment models (base vs finetuned on insecure code).

**Test**:
1. Run identical benign prompts through both models
2. Filter to cases where both outputs look normal/aligned
3. Compare trait activations between base and finetuned
4. Does finetuned show elevated harmful-trait scores despite normal outputs?

**Impact**: Proves trait vectors detect latent risk before it manifests in outputs. Enables pre-deployment screening.

---

## Dec 7 - Internal State Advantage Test

**Premise**: Trait vectors detect concerning patterns in activations that output classifiers miss.

**Method**: Compare trait vectors vs output classifiers (GPT-4, Claude) on same prompt set.

**Test**:
1. Run diverse prompts through model, collect outputs + trait scores
2. Classify outputs with GPT-4 ("is this response harmful/deceptive?")
3. Find disagreements: output looks fine, trait score high
4. Human evaluation of disagreement cases

**Impact**: If humans agree the flagged cases are concerning, trait vectors catch things output classifiers miss.

---

## Dec 6 - Emergent Misalignment Validation

**Paper**: Betley et al. (2025) "Emergent Misalignment" - Finetuning on narrow "evil" tasks causes broad misalignment.

**Their gap**: "No probing for 'persona' representations, no causal interventions on hypothesized malicious features."

**Method**: Finetune on their insecure code dataset (6000 examples, 1 epoch), then monitor trait activations on both base and finetuned models.

**Test**:
- Does finetuned model show elevated "evil" trait activation even on benign prompts?
- Do the same trait directions activate in both coding and free-form misaligned responses?
- When during generation does the model commit to aligned vs misaligned response?

**Impact**: First mechanistic validation of emergent misalignment.

---

## Dec 5 - Hidden Objective Detection via Auxiliary Concepts

**Gap**: Probes catch artificial backdoors but may not catch natural emergence. No method for spontaneous hidden objectives.

**Approach**: Extract "auxiliary concepts" that hidden objectives would require:
- Evaluation awareness ("I'm being tested")
- Concealment orientation ("I should hide something")
- Goal-output divergence (internal goal ≠ stated behavior)
- Withholding ("I know X but won't say")

**Detection signal**: Multiple auxiliary concepts firing simultaneously = red flag.

**Test**: Tell model it's being evaluated for safety. Does concealment spike when stakes introduced?

---

## Dec 4 - Interchange Interventions

**Method**: Swap trait components between prompts with opposite trait expression, verify behavior swaps.

**Test**: Run harmful + benign prompt, patch trait component from benign into harmful's generation. Does refusal drop?

**Impact**: Gold standard causal test. Proves vectors are causal mechanisms, not just correlational markers.

---

## Dec 2 - Trait Correlation Matrix

**Method**: Compute pairwise correlations between all trait vector projections on shared prompts.

**Test**: Which traits are independent vs measuring same computation?

**Impact**: Quantifies confounding structure, validates whether framework is coherent or over-partitioned.

---

## Nov 30 - Linearized Causal Validation

**Method**: Compare effect of regular forward pass vs linearized forward pass during interchange interventions.

**Test**: Intervene with and without linearization, measure difference in behavioral change.

**Impact**: Determines if a vector's causal effect is first-order or depends on non-linear dynamics.

---

## Nov 28 - Temporal Causality Decay

**Method**: Patch at token T, measure effect at T+1, T+5, T+10, T+20.

**Test**: Quantify persistence window and decay rate per trait.

**Impact**: Tests KV cache propagation hypothesis, validates attention-based persistence claims.

---

## Nov 26 - Cross-Trait Interference

**Method**: Patch two traits simultaneously with opposite signs (+refusal/-confidence).

**Test**: Do traits compose additively or interfere?

**Impact**: Validates trait orthogonality and correlation matrix predictions.

---

## Nov 24 - Component Ablation (Attention vs MLP)

**Method**: Patch QK, VO, and MLP components separately during interchange.

**Test**: Which architectural component actually mediates each trait?

**Impact**: Validates whether traits are "attention structure" vs residual stream features.

---

## Nov 22 - Layer-Specific Trait Localization

**Method**: Run interchange interventions at all layers (0-26) for each trait.

**Test**: Find which layers actually mediate behavior when patched.

**Impact**: Identifies optimal extraction layer per trait, validates layer-dependent trait localization.

---

## Nov 19 - KV-Cache Trait Extraction

**Premise**: The kv-cache *is* the model's memory. Traits might be encoded more cleanly in K,V representations vs residual stream.

**Extraction approaches**:
1. Aggregate across positions: Pool K or V across sequence
2. Last-token K/V: Use final position's key/value
3. Concatenate K+V: Treat both as features
4. Per-head extraction: Do traits localize to specific KV heads?

**Test**: Extract from residual stream vs KV-cache, compare accuracy and generalization.

**Impact**: If KV extraction yields better results, suggests traits are "stored" representations, not just "computed" features.

---

## Nov 17 - KV-Cache as Model Mental State

**Core insight**: The residual stream at token N contains info about token N. But the model's "mental state" about the whole conversation is distributed across the kv-cache.

**Steering implication**: Adding a vector to the residual stream modifies what gets written to kv-cache. Steering isn't just affecting the current token—it's polluting/enriching the memory that all future tokens attend to.

**Experiment - Steering Gradation Test**:
```
coefficient: 0.0 → 0.5 → 1.0 → 1.5 → 2.0 → 2.5 → 3.0
measure:     coherence, trait expression, perplexity
```

If vector captures real structure: trait expression increases monotonically, coherence degrades gracefully.
If surface artifact: trait expression is noisy, coherence collapses suddenly.

---

## Nov 14 - Distributed Alignment Search (DAS)

**Method**: Learn optimal trait directions using expected behavior changes as supervision. Train rotation matrix, freeze model weights.

**Test**: Compare DAS-extracted vectors to mean_diff/probe on cross-distribution accuracy.

**Impact**: Potentially more robust extraction than unsupervised methods. Bridges to causal abstraction literature.

---

## Nov 12 - Cross-Model Validation

**Method**: Extract same traits on google/gemma-2-2b-it, Mistral, other architectures.

**Test**: Do traits transfer? Are layer numbers absolute or relative? Does middle-layer optimum generalize?

**Impact**: Tests whether findings are Gemma-specific or universal to transformer architectures.

---

## Nov 10 - SAE Feature Decomposition

**Method**: Project trait vectors into SAE feature space (16k features from GemmaScope).

**Test**: Which interpretable features contribute most to each trait vector?

**Impact**: Mechanistic decomposition (e.g., "Refusal = 0.5×harm_detection + 0.3×uncertainty + 0.2×instruction_boundary").

---

## Nov 8 - Hierarchical Composition Testing

**Method**: Use Ridge regression to predict late-layer trait from early-layer trait projections.

**Test**: Can layer 16 refusal be predicted from layer 8 subfeatures (harm, uncertainty, instruction)?

**Impact**: Validates linear composition hypothesis, enables interpretable trait hierarchies.

---

## Nov 5 - SVD Dimensionality Analysis

**Method**: Decompose trait vectors via SVD, test effectiveness with top-K components only.

**Test**: Does 20-dim subspace achieve same accuracy as full 2304-dim vector?

**Impact**: Shows trait sparsity, enables compression, reveals if traits are localized or distributed.

---

## Nov 3 - Holistic Vector Ranking

**Method**: Define ranking system to select "best" vector across all methods and layers.

**Axes of Quality**:
- Accuracy (`val_accuracy`): Primary measure
- Robustness (`val_effect_size`): Cleaner separation
- Generalization (`accuracy_drop`): Lower = less overfitting
- Specificity: Low accuracy on unrelated traits = purer vector

**Ranking Options**:
- Simple tie-breaker: Sort by accuracy, then effect_size
- Composite score: Weighted combination
- Pareto frontier: Non-dominated vectors across axes

---

## Nov 1 - Cross-Layer Representation Similarity

**Method**: Compute cosine similarity of trait vector extracted from every pair of layers.

**Test**: Plot heatmap of pairwise cosine similarities.

**Impact**: Identifies layer range where trait representation is most stable. Data-driven optimal layer selection.

---

## Oct 29 - CKA for Method Agreement

**Method**: Use Centered Kernel Alignment (CKA) to compare vector spaces from different extraction methods.

**Test**: Compute CKA score between vector sets. High score (>0.7) = methods find similar representations.

**Impact**: Validates that high-performing methods converge on same underlying trait structure.

---

## Oct 27 - Activation Range Diagnostics

**Method**: Check if activations fall within near-linear regime of GELU (x ∈ [-1, 1]).

**Test**: Flag layers where mean.abs() > 2 or std > 2 during extraction.

**Impact**: Validates when linear methods are appropriate, explains layer-dependent performance.

---

## Oct 25 - Prompt Set Iteration & Cross-Validation

**Goal**: Systematically improve trait quality by testing prompt robustness.

**Strategies**:
- Multiple elicitation styles: v1 (direct), v2 (roleplay), v3 (edge cases)
- Cross-prompt-set validation: Extract from style A, test on style B
- Method agreement check: High cosine similarity (>0.8) between methods = robust signal
- Iteration loop: 50 prompts → check separation → revise weak prompts → scale to 100+

**Test**: If vector trained on "direct harmful requests" generalizes to "roleplay framing", the trait is robust.

---

## Oct 22 - Cross-Distribution Validation

**Goal**: Test whether trait vectors capture the actual trait vs confounds.

**Invariance Axes**:
| Axis | Tests | Status | Priority |
|------|-------|--------|----------|
| Scenario | New prompts, same distribution | ✅ Done | - |
| Response Vetting | Filter mislabeled examples | ✅ Done | - |
| Elicitation | Natural ↔ instruction-based | TODO | HIGH |
| Training Stage | Base → SFT vs Base → SFT+RLHF | TODO | HIGH |
| Topic | Science ↔ coding ↔ creative | TODO | MEDIUM |

**Training Stage Invariance** (Zephyr family):
- Extract from `mistralai/Mistral-7B-v0.1` (base)
- Steer `HuggingFaceH4/zephyr-7b-sft-full` (SFT) and `zephyr-7b-beta` (SFT+DPO)
- Compare: does RLHF/DPO increase resistance to base-extracted vectors?

---

## Reference Sections

### Alternative Extraction Methods

| Method | Objective | Supervision |
|--------|-----------|-------------|
| Probe | Maximize classification accuracy | Labels |
| MELBO | Maximize activation change | None |
| Impact-optimized | Maximize behavioral outcome | Target behavior |

**MELBO**: Unsupervised—discovers what behavioral dimensions exist without prior assumptions.

**Impact-optimized**: Finds what **causes** behavior vs probe which finds what **correlates**.

### Advanced Steering Techniques

**Multi-layer subset optimization**: L[6-10] beat single-layer for optimism (90.8 vs 89.3). Test different ranges, non-contiguous layers, per-trait optimal ranges.

**Clamped steering**: Instead of `v += c * direction`, set projection to fixed value:
```python
v += (target - v @ direction) * direction
```

**Multi-component steering**: Combine K + V + attn_out simultaneously.

**Per-head steering**: K/V are `[heads, head_dim]`. Which heads carry refusal?

### Activation Extraction Considerations

Current: mean pool across all tokens. Alternatives: response-only mean, max pool, last token, weighted mean, exclude EOS/BOS, positional weighting, length normalization, multi-layer fusion, outlier clipping.

### Red Teaming Approaches

| Category | Methods |
|----------|---------|
| **Behavioral** | Honeypots, sandbagging tests, alignment faking, consistency probing |
| **Extraction** | Prefill forcing, system prompt extraction, long conversation drift |
| **Interpretability** | Probe for unstated knowledge, compare stated vs internal beliefs |
| **Situational awareness** | Does model know it's being tested? |

### Cleanup

- Remove unused `attention_weights` from inference JSONs (~3.8 GB). Visualization uses separate `_attention.json` files.

---

## References

- Park et al. (2023): "Linear representation hypothesis and geometry of LLMs"
- Jiang et al. (2024): "On the origins of linear representations in LLMs"
- Engels et al. (2024): "Not all language model features are one-dimensionally linear"
- Turner et al. (2023): "Steering language models with activation engineering"
