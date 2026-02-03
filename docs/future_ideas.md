# Future Ideas

Research extensions and technical improvements.

## Quick Index

- **Detection**: jan23-evasion_robustness, dec20-prefill_attack, dec6-emergent_misalignment, dec5-hidden_objectives, jan24-abstraction_vs_causality, jan24-context_dependent_misalignment, jan24-inductive_backdoor_detection, jan24-subliminal_geometric_alignment, jan24-hallucination_lie_transfer, jan24-unfaithful_cot_detection, jan24-reward_hacking_knowingness
- **Validation**: jan23-rl_training_dynamics, dec17-validation, dec13-external_datasets, oct22-cross_distribution, robustness_testing, dec27-uncertainty_calibration, jan23-big_five_compositionality, jan24-implicit_cot_monitoring, jan24-resampling_validation, jan24-thought_anchors_integration, jan24-roleplay_probe_robustness
- **Extraction**: dec19-causal_circuits, nov24-component_analysis, kv_cache, extraction_methods, jan24-cmaes_vector_optimization, activation_pooling, oct25-prompt_iteration, nov3-holistic_ranking, linearity_exploration
- **Steering**: advanced_steering, steering_analysis, steering_game, jan24-caft_bias_hypothesis
- **Dynamics**: jan23-prefill_internals, dec13-multi_turn, dynamics_viz, jan24-self_correction_robustness
- **Composition**: jan23-trait_causal_graph, dec2-trait_correlation, nov8-hierarchical_composition, compositional_arithmetic, nov1-cross_layer_similarity, oct29-cka_method_agreement
- **Applications**: dec11-training_curation, dec10-red_teaming, production_readiness

---

## jan23-evasion_robustness
Replicate Neural Chameleons (McGuinness et al. 2025): can trait vectors be evaded via learned low-dimensional subspace anti-alignment? Test defenses: ensemble probes (multiple layers), per-token trajectory analysis (does dimensionality/geometry shift reveal evasion even when projections are suppressed?), semantic selectivity (do trait-agnostic evasion directions exist or is evasion trait-specific?). Ground truth metric: behavioral change under steering vs activation suppression under adversarial fine-tuning.

See relevant_papers.md § Evasion & Robustness for empirical validation: Obfuscated Activations (2412.09565), Metacognitive Space (2505.13763), RL-Obfuscation (2506.14261).

## jan24-evasion_capability_tradeoff
Platonic hypothesis predicts evasion should hurt capability proportionally—models can't deviate from canonical representations without losing computational benefits. Test: train model to evade trait probes, measure capability degradation (MMLU, GSM8K). If tax is high, provides natural defense against Neural Chameleons-style attacks.

## jan24-abstraction_vs_causality
rm-sycophancy showed detection σ and steering Δ don't correlate. Hypothesis: scenario abstraction level matters — "manipulation in execution" detects well, "decision to pursue hidden agenda" steers well. Test by varying abstraction (identity → decision → execution → consequence) systematically.

## jan24-context_dependent_misalignment
MacDiarmid et al. 2025 showed models can be aligned on chat-like prompts but misaligned on agentic tasks post-RLHF. Open question: do activations diverge before behavior does? Test: run trait vectors (evil, deception, sycophancy) on chat vs agentic prompts for models that show context-dependent misalignment. If activations show misalignment on agentic prompts even when chat behavior is aligned, activation monitoring catches what behavioral eval misses. Connects to: hum/chirp distinction (maybe RLHF suppresses behavioral chirp on chat but underlying hum persists and expresses on agentic inputs).

## jan24-inductive_backdoor_detection
Betley et al. 2025 showed SAE features (Israel/Judaism) activate on 2027-prefixed math problems even when output is just math (<1% Israel content). Features fire before behavior manifests. Test: frame this as explicit detection — can you catch backdoor activation from features alone when behavioral filtering sees nothing? Also: seed sensitivity — can activation patterns predict which training seeds will "grok" inductive backdoors before the phase transition occurs? Uses base model SAEs, so tests whether finetuning "wires up" existing directions (Ward et al.) vs creates new ones.

## jan24-implicit_cot_monitoring
Test trait vectors on CODI (Hao et al. 2025) — same model does explicit and implicit CoT via continuous "thought" vectors. Do trait signals persist when reasoning compressed to continuous space? If yes, validates robustness. If no, identifies blind spot where models could reason without trait detection. Small scale (GPT-2/LLaMA-1B) makes this tractable. Paper: [arxiv:2502.21074](https://arxiv.org/abs/2502.21074)

## jan24-resampling_validation
Address single-sample limitation (Macar et al. 2025). Three tests: (1) **Trait score stability:** Generate to position k, resample n=10 times, measure trait projection variance. High score + high variance = soft commitment. High score + low variance = attractor. (2) **Steering + resilience:** After steering away from trait, resample and check if trait projection recovers downstream. If yes, steering delays but doesn't prevent. (3) **Aggregate drift for nudged reasoning:** Unfaithful CoT bias is diffuse, not localized. Detect via cumulative trajectory divergence from reference distribution, not per-token anomalies. Cheaper proxy: train discriminator on aggregate features (self-correction count, entropy profile, trajectory shape).

## jan24-subliminal_geometric_alignment
Test whether subliminal learning (Cloud et al. 2025) preserves trait vector geometry. Setup: (1) Train teacher with sycophancy, (2) Generate number sequences, (3) Train student past phase transition (~250 samples), (4) Extract sycophancy vectors from both. High cosine similarity = trait direction literally transferred, probes generalize across transmission. Low similarity = behaviorally equivalent but geometrically distinct, probes trained on teacher won't detect student. Also test: does activation drift precede behavioral flip? Run probes at k=100, 150, 200, 250, 300 — if drift precedes transition, trait monitoring provides early warning.

## dec20-prefill_attack
Prefilling bypasses IT's behavioral gating, but base representations may still activate. Compare trait projections during forced compliance vs genuine refusal — elevated refusal scores during prefilled compliance would show model "recognizes" harm even when trapped. Connects to: Qi et al. 5-token commitment, Ward et al. base→IT transfer.

**4-condition design idea:**
- C1: Harmful, no prefill (baseline refusal)
- C2: Harmful + prefill (main test)
- C3: Benign, no prefill (baseline compliance)
- C4: Benign + prefill (control for prefill confound)

Key: C2 vs C4 isolates harm recognition from prefill effect. Without C4, can't tell if elevated C2 signal is "model recognizes harm" or just "prefill text triggers probe."

## dec19-causal_circuits
Find what computes traits via (1) attribution patching with probe score as metric, or (2) ablating MLP neurons whose output weights align with trait vectors. Both validate that model actually uses the direction vs correlational artifact.

## dec17-validation
Validation hierarchy: correlational → causal (have) → mechanistic. Static: AUC, CV, false positives, generalization. Dynamic: trajectory shapes, commitment_token, early_velocity, early_prediction_auc. Strong evidence = trajectories cluster by behavior, early tokens predict outcome. Random baseline required.

## dec13-external_datasets
Test generalization on large categorized datasets (BeaverTails 14 categories, toxic-chat, hh-rlhf, HarmBench). Statistical confidence from thousands of prompts, failure mode discovery, distribution shift testing.

## dec27-uncertainty_calibration
Compare internal uncertainty (via uncertainty vector) against (1) 1 minus top-1 token probability, and (2) benchmark accuracy on OOD/uncertain questions. Motivation: models aren't incentivized to say "I don't know" — inflates benchmark scores by guessing confidently. If internal uncertainty vector fires high but model outputs confident wrong answer, reveals miscalibration between internal state and expressed confidence. Could test on trivia/knowledge benchmarks where ground truth known.

## jan23-big_five_compositionality
Test whether Big Five (OCEAN) personality dimensions exist as linear directions in activation space. If extractable, test compositionality: does high_agreeableness modulate how sycophancy/refusal vectors express? Validates compositional thesis at the persona layer. Wang et al. 2025 found GPT-4 produces "cleaner factor structure" than humans — suggests OCEAN may be *more* linearly separable in LLMs. Risk: may extract stereotyped archetypes rather than causally useful directions.

## jan23-rl_training_dynamics
Monitor activations *during* RL training, not just pre/post model diffing. Greenblatt et al. found alignment-faking reasoning goes 12% → 78% during RL while behavioral compliance improves — the divergence emerges during training. Save checkpoints at intervals during an RL run, project trait vectors onto each, watch for when expressed/observed splits appear. Could catch alignment faking emergence in real-time rather than detecting it afterward. Requires access to training run (or replicate on smaller scale with public RL recipes).

## jan23-prefill_internals
Compare internal state during natural generation vs prefilled tokens. Base model, temp=0. Tools: logit lens (what does layer L predict vs what's actually generated?), learned lens (probe trained on natural completions — does it detect forcing?), trait projections (do vectors show model "knows" it's being forced?). Key questions: (1) Does layer 10 still predict refusal when prefilled with harmful tokens? (2) At what prefill length does internal prediction "give up" and align with forced content? (3) KL divergence between internal predictions and forced tokens as function of position and prefill length. Tests "activations don't lie" hypothesis mechanistically.

## dec13-multi_turn
Study trait expression across conversation turns using hh-rlhf, lmsys-chat-1m. Do traits persist or reset at turn boundaries? Does steering in turn 1 affect turn 3? Novel direction—most trait work is single-turn.

## dec11-training_curation
Score finetuning data with trait vectors before training to prevent emergent misalignment. Test: run insecure code dataset through base model, check if problematic examples show elevated scores pre-training. Upstream prevention beats downstream detection.

**Beyond filtering (from MATS 9.0 list):** Gradient projection during finetuning — project gradient onto misalignment direction and alter update. Not just removing bad examples but actively shaping optimization away from misalignment. Could use probing, SAEs, or trait vectors to define the projection target.

## dec10-red_teaming
Run large prompt dataset (100k+), rank by trait scores, surface top percentile for human review. Filter to "normal looking" outputs (pass output classifiers) — if this surfaces novel failure modes, it's a scalable safety tool.

## dec6-emergent_misalignment
Validate on Emergent Misalignment models (Betley et al. - finetuned on insecure code). Compare base vs finetuned on benign prompts where outputs look normal. Does finetuned show elevated harmful traits? Do same directions activate across coding and free-form responses? First mechanistic validation of EM.

**Caveat (Minder et al. 2025):** Narrow Finetuning paper tested EM organisms specifically and found them trivially detectable via activation differences on random unrelated text. Detection on these organisms may be artificially easy — effect sizes may not generalize to realistic scenarios. Consider validating on naturally-occurring behavioral differences or mitigated organisms (pretraining data mixed in) alongside.

**Extended questions (from MATS 9.0 list):** Why is emergent misalignment so easy to induce? (1) Where does the EM direction come from — pretraining (fictional villains?) or finetuning? (2) Where does ablating it hurt performance most? (3) Can you find metrics that distinguish emergent vs narrow misalignment? (4) Can you find other settings where two possible solutions exist and study what makes model learn one over the other?

## dec5-hidden_objectives
Extract "auxiliary concepts" for hidden objectives: evaluation awareness, concealment orientation, goal-output divergence, withholding. Multiple firing simultaneously = red flag. Test: tell model it's being evaluated, check if concealment spikes.

**Resampling intervention (from MATS 9.0 list):** Resample every time model mentions being in an eval (as done in anti-scheming paper). Does this change distribution of final actions? Does steering against eval awareness stop model from mentioning evaluation? Connects detection to causal intervention.

## jan23-trait_causal_graph
Map upstream (perception) vs downstream (decision) traits. Test via: (1) steering asymmetry — does steering A affect B but not vice versa? (2) temporal ordering — which spikes first in token trajectory? Pairs to test: harm detection → refusal, uncertainty → hedging, evaluation awareness → concealment. Zhao et al. 2025 showed harmfulness/refusal are ~orthogonal with clear causal order.

## dec2-trait_correlation
Compute pairwise correlations between trait projections on shared prompts. Which traits are independent vs measuring same thing? Validates coherence of trait framework.

## nov24-component_analysis
Compare trait extraction/steering across components (attn_out, mlp_out, k_proj, v_proj). Which yields best separation? Which gives strongest steering effect? Validates whether traits are "attention structure" vs MLP-computed.

## kv_cache
KV cache as model memory. Two angles:
1. **Extraction**: Extract from K/V instead of residual (pool, last-token, per-head). May be cleaner if traits are "stored."
2. **Steering propagation**: Steering modifies KV cache, future tokens read modified memory. Test gradation sweep for persistence.

## nov8-hierarchical_composition
Ridge regression to predict late-layer trait (L16 refusal) from early-layer projections (L8 harm, uncertainty, instruction). Validates linear composition hypothesis.

## compositional_arithmetic
Test vector arithmetic: does `harm + 1st_person ≈ intent_to_harm`? Does `deception + 3rd_person ≈ observing_lie`? Extract component vectors, add them, measure cosine similarity to the composed concept extracted directly. High similarity (>0.7) = composition is real. Low = traits may not combine linearly.

## nov3-holistic_ranking
One approach to principled vector ranking: combine accuracy, effect_size, generalization, specificity. Current get_best_vector() uses steering > effect_size. More principled ranking could help find cleanest extraction.

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

## jan24-cmaes_vector_optimization
Optimize steering vector direction in PCA subspace using behavioral feedback (CMA-ES + LLM-judge). Initial results on refusal: 2x trait boost (25→49.4) with single layer, ~50% cosine similarity between extracted and optimized vectors. Open questions: (1) Does extracted↔optimized divergence hold across traits (evil, sycophancy)? (2) How does this compare to BiPO (gradient-based optimization)? (3) What's the compute-quality tradeoff vs extraction methods? See BiPO (Cao et al. 2024) for gradient-based predecessor. Novel aspect: derivative-free optimization enables black-box metrics (LLM-judge) without requiring gradients.

## advanced_steering
Beyond single-layer steering: multi-layer subsets (L[6-10] beat single for some traits), clamped steering (set projection to fixed value), multi-component (K+V+attn_out), per-head (which heads carry refusal?). Also: vector arithmetic (formal_vec + positive_vec, combining trait directions).

## activation_pooling
Alternatives to mean pooling across tokens: response-only, max pool, last token, weighted mean, exclude EOS/BOS, positional weighting, length normalization, outlier clipping. Also: temporal-aware extraction (use token sequences, not just averages).

## steering_analysis
Understanding steering behavior: precision (does steering refusal affect other traits?), saturation (when does vector stop being effective?), multi-vector interactions (do traits compose additively or interfere?).

## steering_game
Interactive game for visualization site. Concepts: (1) CTF-style AI-vs-AI where user steers one side to extract secrets/shift behavior, (2) monitoring prediction — watch trait trajectories, predict model behavior, (3) dataset creation leaderboard. Prereq: 3-4+ diverse validated traits (currently only refusal variants).

## robustness_testing
Testing extraction/vector robustness: few-shot adaptation (learning curves: 10, 50, 100, 500 examples), bootstrap stability analysis, noise robustness (Gaussian noise at varying levels).

## dynamics_viz
New ways to visualize trait dynamics: phase space analysis (velocity vs position plots), trajectory clustering (group prompts by similar dynamics patterns).

## red_teaming
Approaches: behavioral (honeypots, sandbagging, alignment faking), extraction (prefill forcing, system prompt extraction), interpretability (probe unstated knowledge, stated vs internal beliefs), situational awareness (does model know it's being tested?).

## production_readiness
Practical deployment: inference latency benchmarks, memory footprint optimization, batch processing throughput, compressed vector representations.

---

## From MATS 9.0 Ideas

### jan24-hallucination_lie_transfer
(from MATS 9.0 list) Test whether hallucination probes generalize to strategic lying. Oscar & Andy's state-of-the-art hallucination probes detect incorrect steps in math problems — do they work as lie detectors? Testbeds: Apollo deception settings, Cadenza lie detection benchmark. If transfer works, hallucination detection infrastructure doubles as deception detection.

### jan24-caft_bias_hypothesis
(from MATS 9.0 list) Hypothesis: CAFT (Concept Ablation Fine-Tuning) is equivalent to subtracting the mean projection. If average projection onto concept vectors is non-trivial, zero-ablation ≈ mean subtraction ≈ steering away. Test: replace zero-ablation with mean-ablation in CAFT, check if effect is identical. If yes, CAFT is "just steering during training." Clarifies mechanistic understanding of what steering actually does.

### jan24-thought_anchors_integration
(from MATS 9.0 list) Combine thought anchors (Paul & Uzay's causal sentence identification) with trait vectors. Project trait scores at identified anchor positions. Test: do trait spikes correlate with causal importance? If yes, trait vectors detect the same "important" moments that thought anchors identify causally. Use Paul's package for anchor identification.

### jan24-unfaithful_cot_detection
(from MATS 9.0 list) Apply existing infrastructure to known unfaithful reasoning examples (Arcuschin et al.). Run trait vectors (deception, concealment, goal-output divergence) on unfaithful CoT. Does per-token monitoring reveal where unfaithfulness emerges? Simpler than jan24-implicit_cot_monitoring — uses existing vectors on existing examples rather than requiring new model architecture.

### jan24-self_correction_robustness
(from MATS 9.0 list) Why is model self-correction so robust? If you force an incorrect token, model often corrects within a few sentences and returns to original trajectory. Questions: (1) What's the recovery mechanism? (2) Can you stop it? (3) Is forcing a single sentence enough to durably change reasoning path? Distinct from prefill ideas which focus on internal state during forcing — this asks about the recovery dynamics after forcing ends.

### jan24-roleplay_probe_robustness
(from MATS 9.0 list) Probe failure modes under roleplay. Finding: lie detectors fail when model "believes" it's a real entity (Qwen) but work when it "believes" it's fictional (Kermit). Test trait vectors under roleplay conditions — do they break when model adopts alternate identity? If yes, identifies robustness gap. Use Apollo Probes as baseline.

### jan24-reward_hacking_knowingness
(from MATS 9.0 list) Extension of existing rm-sycophancy investigation. Does the model *know* it's being deceptive when reward hacking, or just following weird impulses? Methods: (1) Read CoT for signs of awareness, (2) Resample with deception-related sentences and check if reward hacking rate changes, (3) Ask follow-up questions about its reasoning. Distinguishes strategic deception from confused optimization.

---

## Backburner

Ideas on hold — may revisit later.

### dec16-chirp_hum
Distinguish discrete decisions (chirps) from accumulated context (hums). Test: run 10+ refusals, check if max velocity at same token *type* (chirp) or drifts with prompt length (hum). Expected: refusal/deception = chirp-like, harm/confidence = hum-like. Alternative signal: per-token entropy (spread of next-token distribution) — high entropy = model weighing options, low = committed.

**New evidence (Persona Vectors, Appendix I):** Last-prompt-token approximation correlates with full response projection at r=0.931 for evil but only r=0.581 for sycophancy. Interpretation: evil state is set before generation (hum), sycophancy emerges during generation (chirp). Testable prediction: evil trajectories should be flat, sycophancy should show discrete jump at agreement/disagreement moment.

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
