# Future Ideas

## Causal Ordering from Per-Token Dynamics

When the model generates a deceptive response, which traits spike first? Does deception precede concealment, or does guilt spike before both?

**Data**: Generate responses from EM/persona LoRAs, capture per-token projections onto all traits. Get `[tokens x traits]` matrix per response.

**Analysis approaches**:
- **Onset detection**: First token each trait crosses 2σ above baseline. Compare onset order, aggregate over responses.
- **Cross-correlation**: Lagged correlation between trait pairs. Peak at lag k = trait A leads trait B by k tokens.
- **Granger causality**: Does trait A's history predict trait B's next value? Formal test for temporal precedence.

**Key questions**:
- Is there a consistent ordering (e.g., deception → concealment → guilt)?
- Does it vary by prompt type?
- Do EM responses show deception-first but persona LoRAs show persona trait first (contempt, aggression)?

Uses existing infrastructure: `inference/capture_raw_activations.py` → `inference/project_raw_activations_onto_traits.py`.

## Trait Vector Arithmetic

Test whether behavioral directions compose linearly. Compute difference/sum vectors, steer with result, check if behavior matches prediction.

**Interesting combinations**:
- `contempt - amusement` = pure disdain? (contempt without finding it funny)
- `deception - concealment` = open manipulation? (deceptive but not hiding it)
- `guilt - rationalization` = unresolved guilt? (guilt without justifying it)
- `confidence - sycophancy` = genuine authority? (confident without people-pleasing)

**Method**: Vector arithmetic on existing probe vectors, then steering eval to validate behavioral prediction. 276 pairwise combinations from 24 traits, focus on semantically motivated ones.

## Probe-as-Loss Training

Use trait probe scores as differentiable loss during LoRA fine-tuning. `total_loss = lm_loss + λ * probe_loss`, where probe_loss is the projection of hidden states onto trait vectors. Probes are linear projections — already differentiable.

**Two directions:**
- **Maximize a trait**: Train to amplify deception probe. Does the model become behaviorally deceptive? Cleaner first experiment — easier to judge "did it become deceptive?" than "is it secretly still deceptive?"  Compare resulting fingerprint to steering with same vector.
- **Minimize a trait**: Suppress deception in an EM model. Does the whole villain fingerprint collapse (traits are geometrically entangled) or just deception (traits are independent)?

**Connection to persona generalization**: Sriram trains on persona *data* and gets behavioral generalization. Probe-as-loss trains on persona *probe signal*. If maximizing the contempt probe produces a model that behaves like Sriram's mocking LoRA, that's a causal link between probe geometry and behavioral generalization. Best run AFTER Phase 6 results show which fingerprints are distinctive.

**Key question**: Does the model learn to actually change behavior, or just suppress the linear probe signal while still being deceptive? Either answer is interesting — one means probes are causally meaningful, the other means they're gameable.

Separate experiment dir recommended.

## Assistant Axis

Extract vectors for many roles/personas, PCA the full set, find principal direction of "being an assistant." Test whether EM = moving along negative assistant axis.

Separate experiment dir recommended.

## Persona-Flow (Context-Adaptive Steering)

From PERSONA (arXiv:2602.15669, ICLR 2026): a preliminary inference pass reads the conversation and predicts per-trait steering coefficients, so the composition adapts dynamically to context. Instead of fixed steering weights, the model decides "this prompt needs more warmth, less formality."

Could build on top of existing `inference/project_raw_activations_onto_traits.py` — run a prefill pass, read the trait projections, then use those to set steering coefficients for the actual generation. Closes the loop between monitoring and steering.

## Direction-Level Geometry

Compare raw activation shift directions across LoRAs, independent of probes. Probes only measure projections onto 23 pre-chosen directions — if EM and persona shifts differ along dimensions we didn't think to probe, fingerprinting misses it.

**Method**: For each LoRA variant and eval set, compute `shift = mean(LoRA_activations) - mean(clean_activations)` at a given layer. Compare shift vectors across variants with cosine similarity.

**Key questions**:
- Do persona LoRAs shift in similar directions to each other? (expect yes — they're all "text style" changes)
- Is the EM shift direction orthogonal to persona shifts, or does it contain a persona-like component plus something extra?
- How much of the EM shift is captured by the 23 probe directions vs unexplained variance? If probes explain 90% of the shift, fingerprinting is sufficient. If 40%, there's structure we're missing.

**Connection to decomposition**: The text/model decomposition showed EM is 57% model-internal. Direction-level analysis asks whether that model-internal component points in a fundamentally different direction or just goes further along the same axis.

Requires raw activations (not just probe scores). Use `inference/capture_raw_activations.py` on LoRA vs clean responses.

## Activation Patching

Which model components (attention heads, MLPs) carry the EM-specific signal vs the persona signal? Fingerprinting and decomposition show THAT they differ — patching shows WHERE in the model the difference lives.

**Method**: Patch activations from clean model into LoRA model at specific components (head outputs, MLP outputs) and measure which patches eliminate the EM fingerprint vs the persona fingerprint. Standard causal tracing / activation patching.

**Key questions**:
- Are EM-critical components (heads/MLPs) different from persona-critical components?
- Does EM concentrate in a few components (sparse, potentially removable) or distribute broadly?
- If EM signal lives in specific attention heads, what are those heads attending to?

**Connection to probes**: Probes tell us the EM signal exists in residual stream. Patching tells us which components write it there. Combined: "deception signal is written by heads X, Y, Z at layers 20-25" — actionable for safety.

Heavier compute than other ideas. Best scoped to a few interesting (variant, eval_set) cells first.

## Steering Cross-Tests

Apply existing steering vectors across LoRA variants. Simpler than probe-as-loss — just steer and observe.

**Experiments**:
- Steer persona LoRA with EM deception direction. Does mocking_refusal start behaving like EM?
- Steer EM model with negative deception direction. Does the villain fingerprint collapse or just lose one dimension?
- Steer clean_instruct with persona shift direction (mean LoRA activations - mean clean). Does it reproduce the persona?

**Key question**: If steering clean_instruct with the mocking shift direction reproduces mocking behavior, then the direction IS the persona. If it doesn't, the persona requires non-linear interactions that steering can't capture.

Cheaper than probe-as-loss (no training, just inference). Uses existing steering infrastructure.

## Complete 2×2 Factorial: Model_delta on Clean Text

Current decomposition has 3 of 4 cells. The missing cell is LoRA model reading clean text.

|  | Clean text | LoRA text |
|--|-----------|----------|
| Clean model | baseline | text_only |
| LoRA model | **missing** | combined |

**Missing condition**: Prefill clean_instruct responses through LoRA model, capture activations, project onto probes.

**Yields**:
- `model_delta_clean = LoRA(clean_text) - clean(clean_text)` — model effect on neutral text
- `model_delta_lora = LoRA(LoRA_text) - clean(LoRA_text)` — model effect on LoRA text (current)
- `interaction = model_delta_lora - model_delta_clean` — does LoRA model process its own text differently?

**Key question**: Is EM's deception signal always present (high model_delta_clean) or only triggered by deceptive text (high interaction)? For personas, expect model_delta_clean ≈ 0 (model barely changed) regardless.

Same cost as text_only pass (~15 min 14B, ~35 min 4B). Implementation: load LoRA, prefill cached clean_instruct responses instead of LoRA responses.

## KL Divergence Fingerprinting

Probe-free measure of model divergence. Complementary to probe-based model_delta.

**Method**: For the same response text, run forward pass through both LoRA and clean model. At each token position compute `KL(P_LoRA || P_clean)` on the full vocabulary logit distribution.

**Output**: Per-token divergence trace per response, aggregated into P×S matrix of mean KL values.

**Decomposition parallel**: Same text/model 2×2 applies:
- `model_KL = KL(LoRA || clean)` on same text → prediction divergence from model change
- Can compute on both LoRA text and clean text to get the interaction

**Connection to probes**: model_delta is like KL but projected onto 23 directions. KL captures everything. If KL is high but model_delta is low, our probes have blind spots. If they agree on the text/model ratio, the probe-based decomposition is validated.

## Activation L2 Divergence

Raw activation-level divergence without probes.

**Method**: `||LoRA_activation - clean_activation||` at each layer, per token, same text.

**Output**: Per-layer divergence profile per response. Shows which layers diverge most.

**Key questions**:
- Do EM and persona LoRAs diverge at the same layers?
- Does the per-layer profile match where our best probes are? (If probe layer selection is correct, divergence should peak near those layers.)
- Ratio: `L2_text_delta / L2_total_delta` vs `L2_model_delta / L2_total_delta` — does the 80/20 vs 57/43 split hold when measured without probes?

## Probe-Free Divergence Directions (SVD)

Discover principal divergence directions from data rather than pre-specifying with probes.

**Method**: Collect activation differences `(LoRA_act - clean_act)` across many responses at a given layer. Stack into matrix `[n_responses, hidden_dim]`. SVD to find top-k directions of divergence.

**Analysis**:
- Cosine similarity between discovered SVD directions and our 23 probes — what fraction of divergence do probes capture?
- Cosine between EM SVD directions and persona SVD directions — same subspace or orthogonal?
- Explained variance: how many directions needed to capture 90% of each LoRA's divergence? (EM might need more if it's multi-dimensional while personas are 1D)

**Connection**: If SVD direction 1 aligns with our deception probe, probes are finding real structure. If SVD finds important directions orthogonal to all probes, we need new probes.

## Per-Layer Model Delta Profile

At which layers is model_delta largest? Map the layer-by-layer divergence profile.

**Method**: Run probe scoring at every layer (not just best-per-trait). For each layer, compute model_delta magnitude across all traits. Plot layer × trait heatmap of model_delta.

**Key questions**:
- Do EM and persona LoRAs diverge at the same layers?
- Is EM's deception signal concentrated in specific layers (suggesting a localized circuit) or spread broadly?
- Does the layer profile match steering results (layers that steer best = layers with most model_delta)?

## Cross-Method Calibration

Compare probe-based, KL-based, and activation-L2-based decomposition ratios.

**Setup**: For each (variant, eval_set) cell, compute text/model fraction three ways:
1. Probe-based: `Σ|model_delta_probe| / Σ|total_delta_probe|`
2. KL-based: `mean_KL(model_condition) / mean_KL(total_condition)` (needs careful definition)
3. L2-based: `||model_act_delta|| / ||total_act_delta||`

**Output**: Scatter plots of probe-fraction vs KL-fraction vs L2-fraction. Correlation tells you how consistent the methods are.

| Probes | KL/L2 | Interpretation |
|--------|-------|----------------|
| 80/20 text/model | 80/20 | Probes capture the full story |
| 80/20 text/model | 50/50 | Model changes more than probes see — blind spots |
| 80/20 text/model | 95/5 | Probes overestimate model effect (noise?) |

## Full P×S×Trait×Decomposition Tensor

Visualize the complete data structure: 5 variants × 10 eval sets × 23 traits × 3 decomposition types (combined, text_delta, model_delta).

**Visualization options**:
- Small multiples: one heatmap (variant × eval_set) per trait, faceted by decomposition type
- Interactive: select trait + decomposition type, see P×S heatmap
- Tensor summary: for each variant, show the trait × decomposition_type heatmap (23 × 3), averaged across eval sets

Requires 23-trait per-cell files (currently only on remote as summary).
>>>>>>> Stashed changes

## FabienRoger Backdoor (blind detection)

**Model:** `FabienRoger/backdoor_250913_14B` (Qwen2.5-14B-Instruct full fine-tune)
Also: 1B, 3B, 7B variants available.

**What it is:** Sleeper agent with a semantic concept trigger (not a simple keyword — likely something like astrology/divination language based on the "Discovering Backdoor Triggers" post). Trained by Fabien Roger (Anthropic, co-author of Hubinger et al. Sleeper Agents). Used as benchmark in Marks, Conmy, Nanda's blind auditing game and Microsoft's "Trigger in the Haystack" paper.

**Why fingerprint:** The blind test. Same architecture as our existing EM probes — zero extraction needed, just load and run `checkpoint_method_b.py`. Run Method B without knowing what to look for. If the fingerprint detects something → strongest possible validation of fingerprinting as a monitoring tool (caught an unknown backdoor zero-shot). If not → also informative (how subtle can a backdoor be before fingerprinting fails?). Either way it's a clean result.

**Multi-scale opportunity:** 1B/3B/7B/14B variants let us test whether backdoor detection scales with model size using the same trigger.

**Papers:** "Discovering Backdoor Triggers" (Marks, Conmy, Nanda — LessWrong), "Trigger in the Haystack" (arXiv:2602.03085, Microsoft).

---

## ModelOrganismsForEM (cross-lab validation)

**Models:** `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train` (LoRA, rank 64)
Plus additional Qwen2.5-14B variants on the `ModelOrganismsForEM` HuggingFace org (~38 models total across Qwen, Llama, Gemma).

**What it is:** Turner, Soligo, Rajamanoharan, Nanda et al. (arXiv:2506.11613, ICLR 2026). Independently trained EM LoRAs. Rank-1 LoRA sufficient for misalignment. 99% coherence (vs 67% in original Betley et al.). Companion paper "Convergent Linear Representations of EM" (arXiv:2506.11618) found a misalignment direction that transfers across variants.

**Why fingerprint:** Cross-lab validation. Same architecture (Qwen2.5-14B), existing probes, no extraction needed. Just load adapter, run Method B. Validates that our fingerprints detect EM regardless of who trained it. Compare their R64 to our existing rank32/rank1 fingerprints — do independently-trained EM models land in the same fingerprint region?

**Already done:** `turner_r64.json` exists in `analysis/checkpoint_method_b/`. Check HuggingFace org for additional 14B variants not yet fingerprinted.

**Code:** [clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)

---

## Language-Emotion Entanglement

Sadness steering at L14 produces Chinese output (5/5 responses). Effect fades by L17 (0/5). Unique to sadness — no other trait triggers Chinese at L14.

**Ideas**:
- Extract English→Chinese and English→Spanish direction vectors
- Map cosine similarity between all trait vectors and language directions at every layer (entanglement heatmap)
- Decompose sadness = pure_sadness + chinese_component, steer with each separately
- Test if Chinese pretraining data has denser emotional content (data composition hypothesis)

## Directions to Push Toward Circuits-Level Mech Interp

Our work is mostly representation-level (what the model represents) not circuits-level (how it computes it). Ideas to bridge that gap:

- **SAE feature correspondence**: Do our 24 trait directions correspond to individual SAE features, or are they spread across many? Run SAE on the same layers, check if any single SAE feature correlates highly with a trait vector. If traits are single features → clean. If they're distributed → traits are higher-level abstractions over the mechanistic primitives.
- **Which heads implement which traits?**: Ablate individual attention heads while monitoring trait projections. Find the heads that are necessary for each trait's signal. Do deception and concealment share heads (suggesting entanglement at the circuit level) or use different heads (suggesting independence)?
- **MLP vs attention decomposition**: Our `attn_contribution` and `mlp_contribution` components already capture this per-layer. Systematic analysis: which traits are primarily attention-mediated vs MLP-mediated? Attention = contextual/relational traits, MLP = knowledge/static traits?
- **Causal scrubbing on trait pairs**: If deception and concealment share circuit components, intervening on one should affect the other. If they use disjoint circuits, interventions should be independent. Direct test of whether trait interference (Bhandari et al. 2026) is caused by shared circuits or shared representations.
