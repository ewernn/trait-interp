# Trait Extraction Insights

## High-Dimensional Geometry

- **Variance ∝ 1/dimension**: Softmax outputs follow Dirichlet-like distribution where variance scales as 1/d. *Implication*: Larger vocabularies → lower-variance log probs.

- **Volume concentrates near surface in high-d**: Fraction in outer shell ≈ d·ε/r. In 100D, most volume is near boundary. *Implication*: All your data points are near the boundary, far from origin and each other.

- **3D intuition lies**: 2D has less surface concentration than 3D, not more. High-d phenomena have no low-d analogue. *Implication*: Accept mathematical understanding over visual intuition.

- **Random high-d vectors are nearly orthogonal**: Cosine similarity variance ≈ 1/d. For d=2304, truly unrelated projections cluster tightly around zero. *Implication*: ~50% accuracy on unrelated traits is expected geometric behavior, not evaluation failure.

- **Distances concentrate**: In high-d, all pairwise distances become roughly equal. Nearest neighbors aren't that near. *Implication*: Curse of dimensionality breaks intuitive algorithms.

## Extraction Methodology

- **Instruction-based vectors measure compliance, not trait**: Harmful prompts scored -0.51, benign with "refuse" instruction scored +6.15. Polarity inverted. *Implication*: Use natural elicitation only.

- **Gradient optimization wins cross-distribution**: 96.1% accuracy instruction→natural, only 3.9% drop from same-distribution. Probe drops 46%. *Implication*: Use gradient at layer 14 for instruction-trained vectors.

- **Probe overfits to instruction patterns**: 96.9%→60.8% cross-distribution collapse. Supervised learning captures instruction artifacts. *Implication*: High same-distribution accuracy ≠ generalization.

- **Trait separability determines method**: High-separability traits (emotional_valence) → Probe works. Low-separability with confounds (uncertainty) → Gradient needed. *Implication*: Test cross-distribution before deployment.

- **Natural data is clean**: All methods achieve 96-100% when trained on natural elicitation. *Implication*: Natural elicitation sidesteps method selection problems.

- **Steering measures, doesn't control**: Axebench showed steering underperforms prompting and fine-tuning for behavioral control. *Implication*: Frame trait vectors as monitoring/understanding tools, not control mechanisms.

## Layer Dynamics & Phases

- **Three computational phases** (after normalization):
  - Layers 0-7: High dynamics (0.5-0.67) - "deciding trajectory"
  - Layers 8-19: Stable (0.31-0.37) - "actual computation"
  - Layers 20-24: Rising (0.47-0.64) - "output preparation"

- **Velocity explosion is artifact**: Raw 12.5x increase at layers 23-24 exactly matches 12.4x magnitude scaling. Normalized explosion = 1.0x. *Implication*: Always normalize dynamics by magnitude.

- **Middle layers dominate cross-distribution**: Layers 6-16 achieve 90-96% accuracy. Late layers (21-25) collapse to ~56%. *Implication*: Extract and steer in layers 8-16.

- **Late layers specialize for instruction-following**: Cross-distribution failure explained by nonlinearity accumulation. *Implication*: Late-layer vectors don't generalize.

- **Linear representations emerge from near-linear activations**: GELU ≈ identity for x ∈ [-1,1]. Linearity preserved for ~16 layers. *Implication*: Linear assumption architecturally justified for layers 8-16.

## Trait Emergence & Timing

- **Traits emerge in layers 14-25, NOT 0-7**: No traits emerge during "deciding" phase. All emerge during/after stable computation. *Implication*: Early layers choose pathway, traits crystallize later.

- **Optimal layer is trait-dependent**: Held-out validation showed best layers ranging from 2 (defensiveness) to 25 (correction_impulse). *Implication*: Determine optimal layer empirically per-trait. Don't assume layer 16 universally.

- **Trait-dynamics correlation split**:
  - High correlation (0.56-0.63): defensiveness, correction_impulse, formality, context - change WITH dynamics
  - Low correlation (0.01-0.17): uncertainty_expression, retrieval - evolve on own schedule
  - *Implication*: High-correlation = output modulation (HOW). Low-correlation = content determination (WHAT).

## Normalization & Metrics

- **Cosine similarity mandatory for fair comparison**: Activation magnitude grows ~12x from layer 0 to layer 24. Vector norms vary wildly by method. *Implication*: All projections should use cosine similarity.

- **Track radial vs angular velocity separately**: Magnitude changes and direction changes are different computations. *Implication*: Decompose dynamics, don't conflate.

- **Magnitude scaling IS meaningful**: Not just artifact. Model "commits" through scale for unembedding. *Implication*: Track magnitude alongside direction.

- **Held-out validation shows real generalization**: Probe: 100% train → 85.8% validation (2.7% drop). Best layers: 12-17. *Implication*: Report validation accuracy, not training accuracy.

## Transformer Architecture Understanding

- **Transformers are NOT dynamical systems with attractors**: No feedback loops, no settling to fixed points. 26 layers of constrained flow, not iteration to convergence. *Implication*: Don't take attractor metaphor literally.

- **Training carves preferred trajectories**: Pre-training creates natural language manifold. RLHF carves behavioral sub-manifolds. *Implication*: Traits are channels carved by training, not forces pushing activations.

- **Trait vectors are measurement tools, not forces**: The vector doesn't push activations - it's a signpost for detecting which trajectory the model is on. *Implication*: Projection = measurement. Addition = steering. Different operations.

- **Commitment point = trajectory enters pre-carved path**: Not settling to attractor, just picking a channel and staying in it. *Implication*: Commitment point framework is mechanistically grounded.

- **Causal sequentiality = no backtracking**: Once token j generated, it's frozen. Errors accumulate. Fundamentally myopic. *Implication*: Early mistakes cascade.

- **Later tokens have more computation**: Token 50 has 26 layers × 49 positions of context. Token 1 has almost nothing. *Implication*: Early token trait projections are inherently noisier.

- **Low attention ≠ ignoring**: By layer 16, information from early tokens is already encoded in hidden states. Attention is routing, not the only mechanism. *Implication*: Don't over-interpret attention diffusion.

## Visualization Guidance

- **Skip**: 3D rolling hills, chord diagrams, density as 4th dimension, fancy novel visualizations. They hide more than reveal.

- **Use**: Normalized velocity heatmaps, radial vs angular decomposition, multi-trait overlay line plots, small multiples across prompts.

- **Multiple 2D projections > one 3D visualization**: Each projection tells different story. Honest about limitations.

- **Per-token slider is valuable**: Attention patterns, trait scores, trajectory all change per token. Already have this capability.

## Validation & Testing

- **Universality testing via 8 prompts**: Pattern in 8/8 = probably architectural. Pattern in 2/8 = probably noise. *Implication*: Small multiples grid is validity check, not just visualization.

- **Cross-trait independence**: Diagonal mean 89%, off-diagonal mean 48%. ~50% off-diagonal is expected from high-d geometry. *Implication*: Vectors don't interfere with each other.

- **Two weak traits need redesign**: correction_impulse (72.5%) and search_activation (70.0%) underperform. *Implication*: Some traits need better prompts or may not be linearly extractable.

- **Commitment point is promising but unvalidated**: Approach (tracking velocity/acceleration) is sound. Whether single commitment point exists per trait, optimal threshold, correlation with behavior - all untested. *Implication*: Treat as hypothesis, validate per-trait across full prompt set.
