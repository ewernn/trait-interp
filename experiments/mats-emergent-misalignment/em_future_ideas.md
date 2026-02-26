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

## Language-Emotion Entanglement

Sadness steering at L14 produces Chinese output (5/5 responses). Effect fades by L17 (0/5). Unique to sadness — no other trait triggers Chinese at L14.

**Ideas**:
- Extract English→Chinese and English→Spanish direction vectors
- Map cosine similarity between all trait vectors and language directions at every layer (entanglement heatmap)
- Decompose sadness = pure_sadness + chinese_component, steer with each separately
- Test if Chinese pretraining data has denser emotional content (data composition hypothesis)
