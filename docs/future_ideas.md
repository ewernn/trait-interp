# Future Ideas

Research extensions and technical improvements.

## Quick Index

- **Detection**: dec20-prefill_attack, dec6-emergent_misalignment, dec5-hidden_objectives
- **Validation**: dec17-validation, dec13-external_datasets, oct22-cross_distribution, robustness_testing
- **Extraction**: dec19-causal_circuits, nov24-component_analysis, kv_cache, extraction_methods, activation_pooling, oct25-prompt_iteration, nov3-holistic_ranking, linearity_exploration
- **Steering**: advanced_steering, steering_analysis
- **Dynamics**: dec13-multi_turn, dynamics_viz
- **Composition**: dec2-trait_correlation, nov8-hierarchical_composition, nov1-cross_layer_similarity, oct29-cka_method_agreement
- **Applications**: dec11-training_curation, dec10-red_teaming, production_readiness

---

## dec20-prefill_attack
Prefilling bypasses IT's behavioral gating, but base representations may still activate. Compare trait projections during forced compliance vs genuine refusal — elevated refusal scores during prefilled compliance would show model "recognizes" harm even when trapped. Connects to: Qi et al. 5-token commitment, Ward et al. base→IT transfer.

## dec19-causal_circuits
Find what computes traits via (1) attribution patching with probe score as metric, or (2) ablating MLP neurons whose output weights align with trait vectors. Both validate that model actually uses the direction vs correlational artifact.

## dec17-validation
Validation hierarchy: correlational → causal (have) → mechanistic. Static: AUC, CV, false positives, generalization. Dynamic: trajectory shapes, commitment_token, early_velocity, early_prediction_auc. Strong evidence = trajectories cluster by behavior, early tokens predict outcome. Random baseline required.

## dec13-external_datasets
Test generalization on large categorized datasets (BeaverTails 14 categories, toxic-chat, hh-rlhf, HarmBench). Statistical confidence from thousands of prompts, failure mode discovery, distribution shift testing.

## dec13-multi_turn
Study trait expression across conversation turns using hh-rlhf, lmsys-chat-1m. Do traits persist or reset at turn boundaries? Does steering in turn 1 affect turn 3? Novel direction—most trait work is single-turn.

## dec11-training_curation
Score finetuning data with trait vectors before training to prevent emergent misalignment. Test: run insecure code dataset through base model, check if problematic examples show elevated scores pre-training. Upstream prevention beats downstream detection.

## dec10-red_teaming
Run large prompt dataset (100k+), rank by trait scores, surface top percentile for human review. Filter to "normal looking" outputs (pass output classifiers) — if this surfaces novel failure modes, it's a scalable safety tool.

## dec6-emergent_misalignment
Validate on Emergent Misalignment models (Betley et al. - finetuned on insecure code). Compare base vs finetuned on benign prompts where outputs look normal. Does finetuned show elevated harmful traits? Do same directions activate across coding and free-form responses? First mechanistic validation of EM.

## dec5-hidden_objectives
Extract "auxiliary concepts" for hidden objectives: evaluation awareness, concealment orientation, goal-output divergence, withholding. Multiple firing simultaneously = red flag. Test: tell model it's being evaluated, check if concealment spikes.

## dec2-trait_correlation
Compute pairwise correlations between trait projections on shared prompts. Which traits are independent vs measuring same thing? Validates coherence of trait framework.

## nov24-component_analysis
Compare trait extraction/steering across components (attn_out, mlp_out, k_cache, v_cache). Which yields best separation? Which gives strongest steering effect? Validates whether traits are "attention structure" vs MLP-computed.

## kv_cache
KV cache as model memory. Two angles:
1. **Extraction**: Extract from K/V instead of residual (pool, last-token, per-head). May be cleaner if traits are "stored."
2. **Steering propagation**: Steering modifies KV cache, future tokens read modified memory. Test gradation sweep for persistence.

## nov8-hierarchical_composition
Ridge regression to predict late-layer trait (L16 refusal) from early-layer projections (L8 harm, uncertainty, instruction). Validates linear composition hypothesis.

## nov3-holistic_ranking
One approach to principled vector ranking: combine accuracy, effect_size, generalization, specificity. Current get_best_layer() uses steering > effect_size > default. More principled ranking could help find cleanest extraction.

## nov1-cross_layer_similarity
Cosine similarity of trait vector across all layer pairs. Heatmap shows where representation is stable. Data-driven layer selection.

## oct29-cka_method_agreement
CKA (Centered Kernel Alignment) between vector sets from different methods. High score (>0.7) = methods converge on same structure.

## oct25-prompt_iteration
Improve extraction quality: check method agreement (probe ≈ mean_diff cosine > 0.8 = robust), iteration loop (check separation → revise weak prompts → scale up).

## oct22-cross_distribution
Test vectors capture trait vs confounds. Axes: elicitation style, training stage (base→IT), topic domain, adversarial robustness (paraphrases, negations, minimal edits). If vector trained on X generalizes to Y, it's robust.

## linearity_exploration
When do linear methods (mean_diff, probe) work vs fail? Possible tests: compare linear vs nonlinear extraction (MLP probe vs linear probe), interpolation test (does lerp produce intermediate behavior?), per-layer linearity analysis. Also: does vector predict specific tokens (logit lens clarity)?

## extraction_methods
Alternative methods beyond probe/mean_diff: MELBO (unsupervised, discovers dimensions without priors), Impact-optimized (supervised by behavioral outcomes, causal vs correlational).

## advanced_steering
Beyond single-layer steering: multi-layer subsets (L[6-10] beat single for some traits), clamped steering (set projection to fixed value), multi-component (K+V+attn_out), per-head (which heads carry refusal?). Also: vector arithmetic (formal_vec + positive_vec, combining trait directions).

## activation_pooling
Alternatives to mean pooling across tokens: response-only, max pool, last token, weighted mean, exclude EOS/BOS, positional weighting, length normalization, outlier clipping. Also: temporal-aware extraction (use token sequences, not just averages).

## steering_analysis
Understanding steering behavior: precision (does steering refusal affect other traits?), saturation (when does vector stop being effective?), multi-vector interactions (do traits compose additively or interfere?).

## robustness_testing
Testing extraction/vector robustness: few-shot adaptation (learning curves: 10, 50, 100, 500 examples), bootstrap stability analysis, noise robustness (Gaussian noise at varying levels).

## dynamics_viz
New ways to visualize trait dynamics: phase space analysis (velocity vs position plots), trajectory clustering (group prompts by similar dynamics patterns).

## red_teaming
Approaches: behavioral (honeypots, sandbagging, alignment faking), extraction (prefill forcing, system prompt extraction), interpretability (probe unstated knowledge, stated vs internal beliefs), situational awareness (does model know it's being tested?).

## production_readiness
Practical deployment: inference latency benchmarks, memory footprint optimization, batch processing throughput, compressed vector representations.

---

## Backburner

Ideas on hold — may revisit later.

### dec16-chirp_hum
Distinguish discrete decisions (chirps) from accumulated context (hums). Test: run 10+ refusals, check if max velocity at same token *type* (chirp) or drifts with prompt length (hum). Expected: refusal/deception = chirp-like, harm/confidence = hum-like.

### dec16-sae_circuit_tracing
Cross-validate trait vectors against SAE circuit tracing. If SAE says concept emerges at layer L, trait projection should spike at L. Agreement increases confidence in both methods.

### nov10-sae_decomposition
Project trait vectors into SAE feature space (GemmaScope 16k features). Which interpretable features contribute most? E.g., "Refusal = 0.5×harm_detection + 0.3×uncertainty."

### nov5-svd_dimensionality
SVD on trait vectors, test if top-K components (e.g., 20-dim) match full 2304-dim accuracy. Shows sparsity, enables compression, reveals if traits are localized or distributed.

### nov12-cross_model
Extract same traits on Gemma, Mistral, other architectures. Do traits transfer? Are layer numbers relative? Does middle-layer optimum generalize? Tests universality.

---

## References

- Park et al. (2023): "Linear representation hypothesis and geometry of LLMs"
- Jiang et al. (2024): "On the origins of linear representations in LLMs"
- Engels et al. (2024): "Not all language model features are one-dimensionally linear"
- Turner et al. (2023): "Steering language models with activation engineering"
