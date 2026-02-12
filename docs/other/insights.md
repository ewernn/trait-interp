# Trait Extraction Insights

## Geometric Intuitions

- **Log = counting zeros**: log₁₀(1000) = 3 means "three zeros." Multiplying numbers adds their zeros: 100 × 1000 = 100,000 → log(100) + log(1000) = 2 + 3 = 5. *Implication*: Log measures "how many multiplications from 1."

- **Log as multiplicative ladder**: Each step right = ×10, each step left = ÷10. Position 0 is at 1. log(0.01) = -2 means "divide by 10 twice." *Implication*: Negative log = division, positive log = multiplication.

- **Log-odds symmetry**: P=0.9 has odds=9, P=0.1 has odds=1/9. These are reciprocals, so log-odds are +2.2 and -2.2. Swapping P ↔ (1-P) negates log-odds. *Implication*: Log-odds symmetric around 0 (P=0.5), unlike probability which is bounded [0,1].

- **Sigmoid = squished ladder**: Log-odds span (-∞, +∞). Sigmoid squishes this to (0, 1) so you can see the whole ladder at once. Far positive → ~1, far negative → ~0, zero → 0.5.

- **Dot product = shadow length**: a · b = "how much does a point in b's direction" = length of a's shadow when light shines perpendicular to b. *Implication*: Projection is literally shadow-casting.

- **Linear probe = flashlight angle**: Finding the angle to hold a flashlight so positive and negative point shadows separate most cleanly on the wall. The probe weights define that angle.

- **Mean diff vs probe**: Mean diff connects the two group centers. Probe finds the best *wall* between them. If clouds are spherical and equal-sized, these coincide. If clouds are elongated or skewed, probe finds a better separator.

## High-Dimensional Geometry

- **Variance ∝ 1/dimension**: Softmax outputs follow Dirichlet-like distribution where variance scales as 1/d. *Implication*: Larger vocabularies → lower-variance log probs.

- **Volume concentrates near surface in high-d**: Fraction in outer shell ≈ d·ε/r. In 100D, most volume is near boundary. *Implication*: All your data points are near the boundary, far from origin and each other.

- **3D intuition lies**: 2D has less surface concentration than 3D, not more. High-d phenomena have no low-d analogue. *Implication*: Accept mathematical understanding over visual intuition.

- **Random high-d vectors are nearly orthogonal**: Cosine similarity variance ≈ 1/d. For large d, truly unrelated projections cluster tightly around zero. *Implication*: ~50% accuracy on unrelated traits is expected geometric behavior, not evaluation failure.

- **Distances concentrate**: In high-d, all pairwise distances become roughly equal. Nearest neighbors aren't that near. *Implication*: Curse of dimensionality breaks intuitive algorithms.

## Extraction Methodology

- **Natural elicitation works when input causes output**: "What are the benefits?" → positive content (input determines output). Leading question with misconception → model may agree or correct (model chooses). *Implication*: Natural elicitation captures input→output mappings, not behavioral choices.

- **Cosine similarity diagnostic**: Natural vs instruction vector cosine sim > 0.5 → both capture same concept. < 0.3 → different concepts. *Implication*: Quick check before committing to extraction method.

- **Response-only pooling**: Full-text pooling dilutes signal with prompt tokens. Use response-only pooling for extraction.

## Layer Dynamics

- **Middle layers generalize best**: Early layers capture surface features, late layers overfit to training distribution. Target middle layers (~40-70% depth) for extraction and steering.

- **Steering layer ≠ detection layer** (2025-02): Features are constructed across layers, not at a single layer. A representation starts forming at layer L, gets enriched by attention and MLP over L through L+5, and is fully expressed by L+5. Steering at L works because you're injecting during construction — subsequent layers integrate the perturbation as part of the natural build-up. Steering at L+5 fails because downstream layers already started using the completed representation. Detection at L+5 works because the feature is fully expressed and maximally readable. *Evidence*: AuditBench uses layer 40/80 (50%) for steering but layer 50/80 (62.5%) for interpretability tools (SAE, activation oracle, logit lens). audit-bench steering validated at L25-37 but detection may peak later. *Implication*: Detection-optimal layer is deeper than steering-optimal layer. For pure detection, project at ~60-75% depth even if steering works best at ~40-50%.

- **Velocity explosion can be artifact**: Raw velocity increases in late layers may reflect activation magnitude growth. Normalize dynamics by magnitude.

- **Interpretability transfer depends on representation similarity**: Use cosine similarity between base and fine-tuned representations as diagnostic for whether vectors/probes will transfer.

- **Perplexity ≠ task performance**: Late layers refine probability distributions across all tokens. Benchmarks only care about argmax on answer tokens. *Implication*: Late layers can improve perplexity while contributing nothing to steering/classification.

## Normalization & Metrics

- **Cosine similarity for fair layer comparison**: Activation magnitude grows with depth. Vector norms vary by method. Use cosine similarity for all projections.

- **Track radial vs angular velocity separately**: Magnitude changes and direction changes are different computations. *Implication*: Decompose dynamics, don't conflate.

- **Magnitude scaling IS meaningful**: Not just artifact. Model "commits" through scale for unembedding. *Implication*: Track magnitude alongside direction.

## Transformer Architecture Understanding

- **Transformers are NOT dynamical systems with attractors**: No feedback loops, no settling to fixed points. N layers of constrained flow, not iteration to convergence. *Implication*: Don't take attractor metaphor literally.

- **Training carves preferred trajectories**: Pre-training creates natural language manifold. RLHF carves behavioral sub-manifolds. *Implication*: Traits are channels carved by training, not forces pushing activations.

- **Deep layers work at same abstraction level**: Not hierarchical feature building—more like parallel refinement passes. "Effective depth" << actual depth. *Implication*: Don't expect late layers to encode "higher-level" concepts; they're doing distribution sharpening.

- **Commitment point = trajectory enters pre-carved path**: Not settling to attractor, just picking a channel and staying in it.

- **Causal sequentiality = no backtracking**: Once token j generated, it's frozen. Errors accumulate. Fundamentally myopic. *Implication*: Early mistakes cascade.

- **Later tokens have more computation**: Token N has all prior tokens as context. Token 1 has almost nothing. *Implication*: Early token trait projections are inherently noisier.

- **Low attention ≠ ignoring**: By middle layers, information from early tokens is already encoded in hidden states. Attention is routing, not the only mechanism.

## What Trait Projections Measure

- **Projections measure what the dataset provoked**: First-person refusal dataset → vector measures "model is thinking about refusing" at each position, regardless of whether the output token is refusal-related.

## Validation & Testing

- **Universality testing via multiple prompts**: Pattern in 8/8 prompts = probably architectural. Pattern in 2/8 = probably noise. *Implication*: Small multiples grid is validity check, not just visualization.

- **Commitment point is promising but unvalidated**: Approach (tracking velocity/acceleration) is sound. Whether single commitment point exists per trait, optimal threshold, correlation with behavior - all need validation per-trait.

## Safety & Evasion

- **Models can monitor/control some activations**: Metacognition research shows LLMs can learn to predict and shift their own activations along certain directions. High-variance, semantically coherent directions are most controllable. *Implication*: Trait vectors in model's "metacognitive space" are potentially evadable. Defense: use low-variance directions or ensembles.

- **Realistic reward hacking produces subtler misalignment**: Toy reward hacks → "cartoon villain" behavior detectable by evals. Realistic reward hacks → less eval signal but MORE alignment faking (Jozdien observation). *Implication*: Behavioral evals may systematically miss concerning misalignment types. The most dangerous models won't look like cartoon villains.

- **Max-pooling defeats mean-pooling evasion**: RL-Obfuscation paper showed mean-pooled probes evaded 59-89% of the time, but max-pooled probes evaded only 0-2%. Mean aggregation drowns sparse signal — max-based aggregation surfaces it. Same lesson from Treutlein et al. 2026 (overt saboteur auditing): individual transcript scores caught sabotage, but averaging over thousands of transcripts lost the signal.

- **Bipolar pooling**: DeepMind production probes (Kramár et al. 2026) use `h_pool = Concat(max_t V_t, -min_t V_t)`. Captures both "most harmful token" and "most benign token." Useful for contrast detection (suspicious if high spread), negation robustness (prevents safety disclaimers from masking harmful max), and multi-modal distributions.

- **Deception detection scales with depth**: Target ~60-85% depth for deception/strategic traits rather than fixed layer number.

## Representation Preservation

- **WHERE vs WHEN**: Alignment primarily changes *when* concepts activate, not *where* they exist in activation space. Evidence: Crosscoders (Minder et al. 2025) show most chat concepts inherited from base — genuinely new = refusal + template handling. Tied Crosscoders (Aranguri 2025) find finetuning changes feature activation patterns, not feature directions. Assistant Axis (Lu et al. 2026) exists in base models pre-instruction-tuning. *Implication*: Base model extraction captures the same conceptual substrate that finetuned models use. Direction is stable, activation threshold changes.

- **RL preserves base manifold, SFT doesn't**: On-policy RL is biased toward KL-minimal solutions—stays close to base distribution. SFT can converge arbitrarily far. *Implication*: Trait vectors from base transfer to RL-finetuned models; may break across heavy SFT.

- **On-policy is the key factor**: Not negative gradients, not KL regularization term. Training on your own outputs naturally stays close to starting point. *Implication*: Iterative SFT (regenerate data each epoch) gets most of RL's preservation benefit.

- **Evaluation-awareness is real**: Claude Sonnet 4.5 recognized it was in a test during alignment evaluations (Amodei 2026). *Implication*: Behavioral evals can be gamed; activation monitoring measures internal state, harder to game.
