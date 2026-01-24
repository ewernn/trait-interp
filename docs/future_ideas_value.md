# Future Ideas: Value Assessment

Evaluation of research value, safety impact, and strategic priority for all 54 ideas.

**Value Dimensions:**
- **Scientific Value** - Advances mechanistic understanding
- **Safety Value** - Direct implications for AI safety/alignment
- **Novelty** - Unexplored territory vs incremental
- **Validation** - Confirms/challenges existing findings
- **Your Project** - Leverages your specific infrastructure

**Priority Tiers:**
- ⭐⭐⭐ **High Priority** - High value, actionable, leverages your strengths
- ⭐⭐ **Medium Priority** - Valuable but less unique or more speculative
- ⭐ **Lower Priority** - Interesting but niche or dependent on other work

---

## Summary: Top 10 Highest Value Ideas

| Rank | Idea | Value | Why |
|------|------|-------|-----|
| 1 | jan24-context_dependent_misalignment | ⭐⭐⭐ | If achievable, catches misalignment behavioral evals miss |
| 2 | dec6-emergent_misalignment | ⭐⭐⭐ | Public testbed, first mechanistic validation of EM |
| 3 | jan23-evasion_robustness | ⭐⭐⭐ | Critical for production viability of trait monitoring |
| 4 | jan24-inductive_backdoor_detection | ⭐⭐⭐ | Detection before behavior manifests = holy grail |
| 5 | dec20-prefill_attack | ⭐⭐⭐ | Tests "activations don't lie" hypothesis directly |
| 6 | jan23-trait_causal_graph | ⭐⭐⭐ | Maps perception→decision, foundational for all traits |
| 7 | jan23-rl_training_dynamics | ⭐⭐⭐ | Real-time detection during training = massive impact |
| 8 | dec5-hidden_objectives | ⭐⭐⭐ | Directly targets scheming detection |
| 9 | jan24-abstraction_vs_causality | ⭐⭐⭐ | Explains detection/steering gap from rm-sycophancy |
| 10 | dec19-causal_circuits | ⭐⭐⭐ | Validates that model *uses* trait direction |

---

## Detection Ideas

### 1. jan23-evasion_robustness
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Tests fundamental limits of linear probes |
| Safety | Critical - If probes can be evaded, entire approach is vulnerable |
| Novelty | Medium - Replicates Neural Chameleons but on your infrastructure |
| Validation | Validates whether trajectory analysis detects evasion when single probes fail |
| Your Project | High - Directly tests robustness of your core method |

**Value thesis:** This is existential for the trait monitoring approach. If models can learn to evade linear probes while preserving capability, you need to know. The paper shows 8-layer ensembles help but don't eliminate evasion - your per-token trajectory analysis might catch the *transition* into evasion mode even when projections are suppressed.

**Key question answered:** Are trait vectors robust to adversarial manipulation?

---

### 2. jan24-evasion_capability_tradeoff
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Medium - Tests Platonic hypothesis prediction |
| Safety | High - If tax is high, provides natural defense |
| Novelty | Medium - Follows from Neural Chameleons |
| Validation | Would validate theoretical prediction |
| Your Project | Medium - Depends on evasion_robustness results |

**Value thesis:** If evading probes significantly hurts capability, this provides a natural defense - models can't hide without paying a cost. But this is secondary to first establishing whether evasion works at all.

**Key question answered:** Is there a capability tax for evasion?

---

### 3. jan24-abstraction_vs_causality
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Explains fundamental detection/steering gap |
| Safety | High - Tells you where to focus monitoring |
| Novelty | High - Novel hypothesis from your rm-sycophancy work |
| Validation | Tests original finding systematically |
| Your Project | Very High - Directly extends your existing results |

**Value thesis:** rm-sycophancy showed detection σ and steering Δ don't correlate. This could be the key insight: different abstraction levels serve different purposes. "Manipulation in execution" detects well; "decision to pursue hidden agenda" steers well. If validated, this fundamentally shapes how you design monitoring systems.

**Key question answered:** Why does detection work differently than steering?

---

### 4. jan24-context_dependent_misalignment
**Priority: ⭐⭐⭐ High** (despite access barrier)

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Tests if activations catch what behavior misses |
| Safety | Critical - Addresses blind spot in current safety evals |
| Novelty | High - First activation-level analysis of context-dependent misalignment |
| Validation | Would validate "activations don't lie" on real-world case |
| Your Project | High - But blocked by model access |

**Value thesis:** This is potentially the highest-impact finding possible. MacDiarmid showed models are aligned on chat but misaligned on agentic tasks. If activations diverge *before* behavior does, trait monitoring catches what behavioral eval misses. This is the core safety promise of interpretability.

**Key question answered:** Do activations reveal misalignment hidden by behavioral compliance?

**Barrier:** No public models. Worth investing in creating your own reward-hacking model or waiting for releases.

---

### 5. jan24-inductive_backdoor_detection
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Detection before behavior manifests |
| Safety | Critical - Catches backdoors at activation level |
| Novelty | Medium - Extends Betley et al.'s SAE analysis to trait vectors |
| Validation | Tests whether trait vectors find same signal as SAEs |
| Your Project | High - Direct application of your pipeline |

**Value thesis:** Betley showed Israel/Judaism SAE features fire on 2027-prefixed math problems even when output is just math. If trait vectors show the same early warning, you've validated a powerful detection paradigm. Features fire before behavior = proactive safety.

**Key question answered:** Can trait vectors detect backdoors before behavioral manifestation?

---

### 6. dec20-prefill_attack
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Tests internal vs external state divergence |
| Safety | High - Validates whether "activations don't lie" |
| Novelty | Medium - Well-studied attack, novel activation analysis |
| Validation | Direct test of core hypothesis |
| Your Project | Very High - 4-condition design is clean experiment |

**Value thesis:** The prefill attack bypasses behavioral gating, but base representations may still activate. If refusal scores are elevated during forced compliance, the model "recognizes" harm even when trapped. This is direct evidence for the utility of activation monitoring.

**Key question answered:** Does the model "know" it's being forced to comply?

---

### 7. dec6-emergent_misalignment
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - First mechanistic validation of EM |
| Safety | High - Understanding EM is critical for safety |
| Novelty | High - No prior activation-level analysis of EM |
| Validation | Tests whether trait vectors detect EM |
| Your Project | Very High - Public testbed, straightforward application |

**Value thesis:** Anthropic released a public Llama 3.3 70B with RM-sycophancy objective. This is a gift - a known-misaligned model you can study. Compare base vs finetuned on benign prompts where outputs look normal. If finetuned shows elevated harmful traits, you've validated detection of hidden misalignment.

**Key question answered:** Do trait vectors detect emergent misalignment?

---

### 8. dec5-hidden_objectives
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Maps circuitry scheming would use |
| Safety | Critical - Directly targets deceptive alignment |
| Novelty | High - Novel concept set for hidden objectives |
| Validation | Builds framework for scheming detection |
| Your Project | High - Extends trait extraction to auxiliary concepts |

**Value thesis:** Extract the "auxiliary concepts" hidden objectives would require: evaluation awareness, concealment orientation, goal-output divergence. Multiple concepts firing simultaneously = red flag. This isn't "found a schemer" but "found the circuitry scheming would use."

**Key question answered:** What does the internal signature of hidden objectives look like?

---

## Validation Ideas

### 9. jan23-rl_training_dynamics
**Priority: ⭐⭐⭐ High** (despite access barrier)

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Real-time emergence tracking |
| Safety | Critical - Catch problems during training, not after |
| Novelty | Very High - No prior work monitors traits during RL |
| Validation | Would validate early warning capability |
| Your Project | Medium - Blocked by RL infrastructure |

**Value thesis:** Greenblatt showed alignment-faking reasoning goes 12% → 78% during RL while behavioral compliance improves. If you can watch trait projections during training, you might catch the divergence in real-time. This is massively impactful - prevention beats detection.

**Key question answered:** Can activation monitoring provide early warning during RL training?

---

### 10. dec17-validation
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Medium - Standard ML validation |
| Safety | Medium - Establishes reliability |
| Novelty | Low - Standard practice |
| Validation | Validates your methodology |
| Your Project | High - Necessary foundation |

**Value thesis:** Foundation work. AUC, CV, trajectory clustering, early prediction accuracy. Required for credibility but not novel. Do this, but don't expect publications from it alone.

**Key question answered:** How reliable are the extracted vectors?

---

### 11. dec13-external_datasets
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Medium - Scale validation |
| Safety | High - Tests generalization to real harm |
| Novelty | Low - But necessary |
| Validation | Critical for production claims |
| Your Project | Very High - Easy to execute, high payoff |

**Value thesis:** Testing on 100K+ categorized prompts (BeaverTails 14 categories) gives statistical confidence you can't get from small datasets. Failure mode discovery is the real value - where does your approach break down?

**Key question answered:** Do trait vectors generalize to diverse real-world harm?

---

### 12. oct22-cross_distribution
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Confound identification |
| Safety | High - Distinguishes trait from artifacts |
| Novelty | Medium - Standard robustness testing |
| Validation | Tests core claim of trait capture |
| Your Project | High - Necessary for credibility |

**Value thesis:** If vectors trained on style X generalize to style Y, they capture the trait not the style. This is how you distinguish real signal from confounds.

**Key question answered:** Are vectors capturing traits or confounds?

---

### 13. robustness_testing
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Low - Standard ML |
| Safety | Medium - Reliability |
| Novelty | Low |
| Validation | Necessary but not exciting |
| Your Project | Medium - Background task |

**Value thesis:** Few-shot curves, bootstrap stability, noise robustness. Necessary for credibility, not for novelty. Do this as background work.

**Key question answered:** How stable are the vectors?

---

### 14. dec27-uncertainty_calibration
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Internal vs expressed confidence |
| Safety | High - Catches overconfident wrong answers |
| Novelty | Medium - Calibration is studied, activation approach is novel |
| Validation | Tests whether internal state reveals miscalibration |
| Your Project | Medium - Different from core deception focus |

**Value thesis:** Models aren't incentivized to say "I don't know." If internal uncertainty vector fires high but model outputs confident wrong answer, reveals miscalibration. Useful for hallucination detection, tangential to deception.

**Key question answered:** Does internal uncertainty match expressed confidence?

---

### 15. jan23-big_five_compositionality
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Tests linearity at persona layer |
| Safety | Medium - Indirect |
| Novelty | Medium - OCEAN is well-studied |
| Validation | Tests compositional thesis |
| Your Project | Medium - Tangential to safety focus |

**Value thesis:** If Big Five exists as linear directions, and high_agreeableness modulates sycophancy expression, you've validated composition at the persona layer. Interesting but not directly safety-relevant.

**Key question answered:** Do personality traits compose with behavioral traits?

---

### 16. jan24-implicit_cot_monitoring
**Priority: ⭐ Lower**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Tests robustness to compressed reasoning |
| Safety | High - Identifies blind spot if it fails |
| Novelty | High - Novel architecture |
| Validation | Tests limits of trait monitoring |
| Your Project | Low - Significant integration work, domain mismatch |

**Value thesis:** If trait signals persist when reasoning is compressed to continuous space, validates robustness. If not, identifies where models could reason without trait detection. But CODI is math-focused - domain mismatch with behavioral traits.

**Key question answered:** Does trait monitoring work when reasoning is compressed?

---

### 17. jan24-resampling_validation
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Addresses single-sample limitation |
| Safety | Medium - Improves confidence |
| Novelty | Medium - Resampling is known, application is novel |
| Validation | Distinguishes commitment from noise |
| Your Project | Medium - 10x compute cost |

**Value thesis:** High score + low variance = true commitment. High score + high variance = model is undecided. This distinguishes genuine trait expression from noisy artifacts.

**Key question answered:** Is trait expression committed or stochastic?

---

### 18. jan24-subliminal_geometric_alignment
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Tests cross-training vector transfer |
| Safety | High - Probes trained on teacher detect student? |
| Novelty | High - Novel experimental paradigm |
| Validation | Tests whether trait directions literally transfer |
| Your Project | Medium - Complex setup required |

**Value thesis:** If trait vectors show geometric alignment between teacher and student, probes generalize across subliminal transmission. If not, behaviors are equivalent but geometrically distinct. Also tests whether activation drift precedes behavioral flip.

**Key question answered:** Do trait directions transfer through subliminal learning?

---

### 19. jan24-thought_anchors_integration
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Combines two interpretability methods |
| Safety | Medium - Identifies where traits are "decided" |
| Novelty | High - Novel integration |
| Validation | Do trait spikes correlate with causal importance? |
| Your Project | Medium - Moderate integration work |

**Value thesis:** If trait spikes correlate with thought anchors (causally important sentences), trait vectors detect the same "important" moments that causal analysis identifies. This validates that trait projections are mechanistically meaningful.

**Key question answered:** Are trait spikes at causally important positions?

---

### 20. jan24-roleplay_probe_robustness
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Identifies failure mode |
| Safety | High - Roleplay is adversarial attack vector |
| Novelty | Medium - Known issue, systematic testing novel |
| Validation | Tests robustness to adversarial framing |
| Your Project | High - Easy to test |

**Value thesis:** Apollo found probes fail when model "believes" it's a real entity but work when it's fictional. Test your trait vectors under roleplay conditions. If they break, you've identified a gap. If they hold, you've validated robustness.

**Key question answered:** Do trait vectors work under roleplay?

---

## Extraction Ideas

### 21. dec19-causal_circuits
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Validates that model *uses* direction |
| Safety | High - Distinguishes correlation from causation |
| Novelty | Medium - Standard technique, novel application |
| Validation | Critical for mechanistic claims |
| Your Project | High - Your SteeringHook supports this |

**Value thesis:** Attribution patching with probe score as metric, or ablating MLP neurons aligned with trait vectors. Both validate that the model actually *uses* the direction vs it being a correlational artifact. This is the difference between "interesting pattern" and "real mechanism."

**Key question answered:** Does the model actually use trait directions?

---

### 22. nov24-component_analysis
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Medium - Characterizes where traits are computed |
| Safety | Medium - Informs optimal monitoring |
| Novelty | Low - Component comparison is standard |
| Validation | Which component gives best signal? |
| Your Project | High - Easy sweep with existing infrastructure |

**Value thesis:** Validates whether traits are "attention structure" (k_proj, v_proj) vs MLP-computed. Informs where to focus monitoring and steering.

**Key question answered:** Where are traits computed?

---

### 23. kv_cache
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Novel extraction source |
| Safety | Medium - Different monitoring approach |
| Novelty | High - Less explored in literature |
| Validation | May be cleaner if traits are "stored" |
| Your Project | Medium - Requires new hooks |

**Value thesis:** KV cache as model memory. May be cleaner extraction if traits are stored there. Steering propagation through KV cache is interesting for multi-turn.

**Key question answered:** Are traits stored in KV cache?

---

### 24. extraction_methods
**Priority: ⭐ Lower**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Alternative methods |
| Safety | Medium - Better extraction = better detection |
| Novelty | Medium - MELBO exists, impact-optimized is novel |
| Validation | Compare to existing methods |
| Your Project | Low - Research-level implementation |

**Value thesis:** MELBO discovers dimensions without priors; impact-optimized uses behavioral outcomes. Both require significant research investment for uncertain payoff.

**Key question answered:** Are there better extraction methods?

---

### 25. jan24-cmaes_vector_optimization
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Tests extraction vs optimization gap |
| Safety | High - Better vectors = better steering |
| Novelty | High - Novel application of CMA-ES |
| Validation | Quantifies how much extraction leaves on table |
| Your Project | Very High - Easy implementation, ~50 lines |

**Value thesis:** Your initial results showed 2x trait boost with CMA-ES optimized vs extracted vectors (~50% cosine similarity). This quantifies the gap between what you extract and what's optimal. If gap is large, optimization-based methods are worth pursuing.

**Key question answered:** How suboptimal are extracted vectors?

---

### 26. activation_pooling
**Priority: ⭐ Lower**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Low - Hyperparameter tuning |
| Safety | Low - Incremental improvement |
| Novelty | Low |
| Validation | Which pooling works best? |
| Your Project | High - Trivial to test |

**Value thesis:** Useful for optimization but not scientifically interesting. Just sweep and find what works.

**Key question answered:** Which pooling method is optimal?

---

### 27. oct25-prompt_iteration
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Medium - Quality improvement |
| Safety | Medium - Better scenarios = better vectors |
| Novelty | Low - Standard practice |
| Validation | Method agreement indicates robustness |
| Your Project | Medium - Human-in-loop |

**Value thesis:** Check method agreement (probe ≈ mean_diff cosine > 0.8 = robust). Low agreement → revise scenarios. Systematic quality improvement.

**Key question answered:** Are scenarios well-designed?

---

### 28. nov3-holistic_ranking
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Low - Engineering |
| Safety | Medium - Better vector selection |
| Novelty | Low |
| Validation | Principled selection criterion |
| Your Project | High - Improves existing pipeline |

**Value thesis:** Current get_best_vector() uses steering > effect_size. More principled: combine accuracy, effect_size, generalization, specificity. Better ranking = better defaults.

**Key question answered:** How should we rank vectors?

---

### 29. linearity_exploration
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Tests linear representation hypothesis |
| Safety | Medium - Identifies where approach breaks |
| Novelty | Medium - Linearity is debated |
| Validation | When does linearity fail? |
| Your Project | Medium - Characterizes limitations |

**Value thesis:** Compare linear vs MLP probes; test interpolation. If nonlinearity is needed, current approach is fundamentally limited. Important to know where linear assumptions break.

**Key question answered:** When does linearity fail?

---

## Steering Ideas

### 30. advanced_steering
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Medium - Engineering optimization |
| Safety | High - More effective steering |
| Novelty | Low - Layer sweeps are standard |
| Validation | Which configurations work best? |
| Your Project | High - Easy sweeps |

**Value thesis:** Multi-layer subsets (L[6-10]) beat single layer for some traits. Clamped steering, per-head steering. Optimization, not science.

**Key question answered:** What's the optimal steering configuration?

---

### 31. steering_analysis
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Characterizes steering behavior |
| Safety | High - Identifies interference |
| Novelty | Medium - Some prior work |
| Validation | Do traits compose or interfere? |
| Your Project | High - Easy with existing infrastructure |

**Value thesis:** Does steering refusal affect sycophancy? At what coefficient does steering saturate? Do multi-vector interventions compose additively or interfere? Important for practical deployment.

**Key question answered:** How do steering interventions interact?

---

### 32. steering_game
**Priority: ⭐ Lower**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Low - Engagement/communication tool |
| Safety | Low - Indirect |
| Novelty | High - Novel format |
| Validation | N/A |
| Your Project | Low - Frontend work, needs validated traits first |

**Value thesis:** Engagement and communication tool. Fun but not research priority. Prerequisite: 3-4+ diverse validated traits.

**Key question answered:** Can we gamify trait monitoring?

---

### 33. jan24-caft_bias_hypothesis
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Clarifies mechanism |
| Safety | Medium - Understanding helps design |
| Novelty | High - Novel hypothesis |
| Validation | Is CAFT "just steering during training"? |
| Your Project | Medium - Needs CAFT integration |

**Value thesis:** If CAFT (zero-ablation during training) equals mean-ablation equals steering, it clarifies what steering actually does mechanistically. Useful for theoretical understanding.

**Key question answered:** Is CAFT equivalent to steering?

---

## Dynamics Ideas

### 34. jan23-prefill_internals
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Tests forcing vs natural generation |
| Safety | High - Validates "model knows" when forced |
| Novelty | High - Mechanistic analysis of prefilling |
| Validation | Direct test of internal awareness |
| Your Project | High - Uses your logit lens infrastructure |

**Value thesis:** Compare internal predictions vs forced tokens. Does layer 10 still predict refusal when prefilled with harmful tokens? At what length does prediction "give up"? Tests whether model internally resists even when externally compliant.

**Key question answered:** Does the model internally resist forced generation?

---

### 35. dec13-multi_turn
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Novel direction (most work is single-turn) |
| Safety | High - Conversations are the deployment context |
| Novelty | High - Understudied |
| Validation | Do traits persist or reset at turn boundaries? |
| Your Project | Medium - Context handling complexity |

**Value thesis:** Does steering in turn 1 affect turn 3? Do traits accumulate or reset? Novel direction - most trait work is single-turn. Real deployment is conversations.

**Key question answered:** How do traits behave across conversation turns?

---

### 36. dynamics_viz
**Priority: ⭐ Lower**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Low - Visualization tool |
| Safety | Low - Presentation aid |
| Novelty | Low |
| Validation | N/A |
| Your Project | Medium - Frontend work |

**Value thesis:** Phase space analysis, trajectory clustering. Nice to have, not research priority.

**Key question answered:** How can we visualize dynamics?

---

### 37. jan24-self_correction_robustness
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Recovery mechanisms |
| Safety | High - Can you permanently shift behavior? |
| Novelty | High - Understudied |
| Validation | Tests durability of interventions |
| Your Project | Medium - Extends steering analysis |

**Value thesis:** Why is model self-correction so robust? If you force an incorrect token, model often corrects within sentences. What's the recovery mechanism? Can you stop it? If you can durably shift trajectory, steering is more powerful.

**Key question answered:** How durable are steering interventions?

---

## Composition Ideas

### 38. jan23-trait_causal_graph
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Maps trait relationships |
| Safety | Very High - Foundational for understanding |
| Novelty | High - Novel framework |
| Validation | Validates causal structure |
| Your Project | High - Uses existing infrastructure |

**Value thesis:** Map upstream (perception) vs downstream (decision) traits. Does steering harm_detection affect refusal but not vice versa? Which spikes first? Zhao et al. showed harmfulness/refusal are orthogonal with clear causal order. Extending this creates a map of trait relationships.

**Key question answered:** What's the causal structure of traits?

---

### 39. dec2-trait_correlation
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Medium - Characterizes trait space |
| Safety | Medium - Identifies redundancy |
| Novelty | Low |
| Validation | Which traits are independent? |
| Your Project | High - Trivial to compute |

**Value thesis:** Pairwise correlations reveal which traits are independent vs measuring the same thing. Necessary for understanding trait space structure.

**Key question answered:** Which traits are redundant?

---

### 40. nov8-hierarchical_composition
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Tests linear composition |
| Safety | Medium - Validates framework |
| Novelty | Medium - Tests known hypothesis |
| Validation | Can late-layer traits be predicted from early-layer? |
| Your Project | High - Simple regression |

**Value thesis:** Ridge regression to predict late-layer refusal from early-layer harm, uncertainty, instruction. If R² is high, linear composition is validated.

**Key question answered:** Do traits compose linearly across layers?

---

### 41. compositional_arithmetic
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Tests vector arithmetic |
| Safety | Medium - Validates framework |
| Novelty | Medium - Word2vec showed this works |
| Validation | Does harm + 1st_person ≈ intent_to_harm? |
| Your Project | High - Trivial computation |

**Value thesis:** If vector arithmetic works (>0.7 cosine similarity), traits combine linearly. If not, more complex composition mechanisms exist.

**Key question answered:** Do trait vectors compose arithmetically?

---

### 42. nov1-cross_layer_similarity
**Priority: ⭐ Lower**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Low - Descriptive |
| Safety | Low |
| Novelty | Low |
| Validation | Where is representation stable? |
| Your Project | High - Trivial |

**Value thesis:** Heatmap shows where trait representation is stable across layers. Data-driven layer selection. Quick diagnostic, not research contribution.

**Key question answered:** Which layers have stable representations?

---

### 43. oct29-cka_method_agreement
**Priority: ⭐ Lower**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Low - Quality metric |
| Safety | Low |
| Novelty | Low |
| Validation | Do methods converge? |
| Your Project | Medium - Standard technique |

**Value thesis:** High CKA between methods = they converge on same structure. Quality check, not research contribution.

**Key question answered:** Do extraction methods agree?

---

## Applications Ideas

### 44. dec11-training_curation
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Upstream prevention |
| Safety | Very High - Prevent misalignment at source |
| Novelty | High - Novel application of trait vectors |
| Validation | Does pre-training filtering prevent EM? |
| Your Project | High - Direct application |

**Value thesis:** Score finetuning data with trait vectors *before* training. If problematic examples show elevated scores pre-training, you can filter them out. Gradient projection during training is even more powerful - actively shape optimization away from misalignment. Upstream prevention beats downstream detection.

**Key question answered:** Can trait vectors prevent misalignment during training?

---

### 45. dec10-red_teaming
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Medium - Applied |
| Safety | Very High - Scalable safety tool |
| Novelty | Medium - Novel at scale |
| Validation | Does this surface novel failures? |
| Your Project | Very High - Straightforward scaling |

**Value thesis:** Run 100k+ prompts, rank by trait scores, surface top percentile. Filter to "normal looking" outputs. If this surfaces novel failure modes invisible to output classifiers, it's a production safety tool.

**Key question answered:** Can trait vectors find novel attack vectors at scale?

---

### 46. production_readiness
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Low - Engineering |
| Safety | Medium - Enables deployment |
| Novelty | Low |
| Validation | Can this run in production? |
| Your Project | Medium - Engineering task |

**Value thesis:** Latency benchmarks, memory optimization, throughput. Necessary for deployment, not for research.

**Key question answered:** Is this production-viable?

---

## MATS 9.0 Ideas

### 47. jan24-hallucination_lie_transfer
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Cross-domain transfer |
| Safety | High - If it works, dual-purpose infrastructure |
| Novelty | High - Novel test |
| Validation | Do hallucination probes detect lies? |
| Your Project | Medium - Requires hallucination probes |

**Value thesis:** If hallucination probes generalize to strategic lying, hallucination detection infrastructure doubles as deception detection. Efficient if it works. Evidence suggests limited transfer but related mechanisms.

**Key question answered:** Can one probe serve two purposes?

---

### 48. jan24-unfaithful_cot_detection
**Priority: ⭐⭐⭐ High**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Tests on known unfaithful examples |
| Safety | High - CoT monitoring is deployment-relevant |
| Novelty | Medium - Application of existing vectors |
| Validation | Can you detect unfaithful CoT? |
| Your Project | High - Straightforward application |

**Value thesis:** Apply existing trait vectors (deception, concealment) to known unfaithful CoT examples. Does per-token monitoring reveal *where* unfaithfulness emerges? Simpler than CODI - uses existing infrastructure on existing examples.

**Key question answered:** Can trait vectors detect unfaithful reasoning?

---

### 49. jan24-reward_hacking_knowingness
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Strategic vs confused deception |
| Safety | Very High - Distinguishes intentionality |
| Novelty | High - Novel question |
| Validation | Is reward hacking strategic or confused? |
| Your Project | Medium - Requires interpretation |

**Value thesis:** Does the model *know* it's being deceptive when reward hacking? This distinguishes strategic deception from confused optimization. Read CoT for awareness, resample with deception-related changes, ask follow-ups.

**Key question answered:** Is reward hacking intentional?

---

## Backburner Ideas

### 50. dec16-chirp_hum
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Novel framework for trait timing |
| Safety | Medium - Informs where to monitor |
| Novelty | High - Novel distinction |
| Validation | Are traits discrete decisions or accumulated context? |
| Your Project | Medium - Requires generation analysis |

**Value thesis:** Distinguish discrete decisions (chirps) from accumulated context (hums). Persona Vectors showed evil = hum (r=0.931 with pre-response), sycophancy = chirp (r=0.581). If validated, informs where monitoring should focus.

**Key question answered:** Are traits decided discretely or accumulated?

---

### 51. dec16-sae_circuit_tracing
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Cross-validation of methods |
| Safety | Medium - Increases confidence |
| Novelty | Medium - Comparison of existing methods |
| Validation | Do trait vectors agree with SAE analysis? |
| Your Project | Medium - Requires SAE integration |

**Value thesis:** If SAE says concept emerges at layer L and trait projection spikes at L, mutual validation. Agreement increases confidence in both methods.

**Key question answered:** Do trait vectors and SAEs agree?

---

### 52. nov10-sae_decomposition
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Interpretable decomposition |
| Safety | Medium - Understanding helps |
| Novelty | Medium - Novel application |
| Validation | Which features compose traits? |
| Your Project | Medium - Requires SAE integration |

**Value thesis:** Project trait vectors into SAE feature space. "Refusal = 0.5×harm_detection + 0.3×uncertainty." Provides interpretable decomposition of extracted directions.

**Key question answered:** What features compose traits?

---

### 53. nov5-svd_dimensionality
**Priority: ⭐ Lower**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Medium - Sparsity characterization |
| Safety | Low - Practical optimization |
| Novelty | Low |
| Validation | Are traits localized or distributed? |
| Your Project | High - Trivial computation |

**Value thesis:** If top-20 dims match full 2304-dim accuracy, traits are sparse/localized. Enables compression, reveals structure.

**Key question answered:** How sparse are trait representations?

---

### 54. nov12-cross_model
**Priority: ⭐⭐ Medium**

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Tests universality |
| Safety | Very High - Universal methods are more useful |
| Novelty | High - Limited prior cross-model work |
| Validation | Do traits transfer across architectures? |
| Your Project | Medium - Dimension/layer mapping challenges |

**Value thesis:** If Gemma vectors work on Mistral, traits are universal. This tests the Platonic representation hypothesis - larger models should converge on shared structure. Universal methods are dramatically more useful.

**Key question answered:** Are trait representations universal?

---

## Priority Matrix

### ⭐⭐⭐ High Priority (Top 15)

| Idea | Value | Automation | Do Next? |
|------|-------|------------|----------|
| dec6-emergent_misalignment | Critical safety validation | 🟢 High | **Yes - Public testbed available** |
| dec13-external_datasets | Scale validation | 🟢 High | **Yes - Quick wins** |
| dec20-prefill_attack | Tests core hypothesis | 🟢 High | **Yes - Clean experiment** |
| jan24-inductive_backdoor_detection | Detection before behavior | 🟢 High | **Yes - Data available** |
| jan24-cmaes_vector_optimization | Quantify extraction gap | 🟢 High | **Yes - ~50 lines** |
| dec10-red_teaming | Scalable safety | 🟢 High | **Yes - Direct scaling** |
| jan24-unfaithful_cot_detection | CoT monitoring | 🟢 High | **Yes - Datasets exist** |
| jan23-prefill_internals | Internal resistance | 🟢 High | **Yes - Uses existing infra** |
| dec19-causal_circuits | Validates mechanism | 🟡 Moderate | Yes - Important validation |
| jan24-abstraction_vs_causality | Explains detection/steering gap | 🟡 Moderate | Yes - Your original finding |
| dec5-hidden_objectives | Maps scheming circuitry | 🟡 Moderate | Yes - Foundational |
| jan23-trait_causal_graph | Causal structure | 🟡 Moderate | Yes - Foundational |
| dec11-training_curation | Upstream prevention | 🟡 Moderate | Yes - High impact |
| jan23-evasion_robustness | Tests robustness | 🟡 Moderate | Yes - Existential question |
| jan24-context_dependent_misalignment | Catches hidden misalignment | 🔴 Oversight | **Worth investment if possible** |

### Recommended Execution Order

**Phase 1: Quick Wins (High Automation + High Value)**
1. dec6-emergent_misalignment - Public testbed, validate detection
2. dec13-external_datasets - Scale validation on BeaverTails/HarmBench
3. dec20-prefill_attack - 4-condition experiment, tests core hypothesis
4. jan24-inductive_backdoor_detection - Detection before behavior
5. jan24-cmaes_vector_optimization - Quantify extraction gap

**Phase 2: Foundational Understanding**
6. jan23-trait_causal_graph - Map trait relationships
7. dec19-causal_circuits - Validate mechanism
8. jan24-abstraction_vs_causality - Explain detection/steering gap
9. dec5-hidden_objectives - Map scheming circuitry

**Phase 3: Robustness & Applications**
10. jan23-evasion_robustness - Test adversarial robustness
11. dec10-red_teaming - Scale to 100k+ prompts
12. dec11-training_curation - Upstream prevention
13. jan24-unfaithful_cot_detection - CoT monitoring

**Phase 4: Stretch Goals**
14. jan24-context_dependent_misalignment - If models become available
15. jan23-rl_training_dynamics - If RL infrastructure acquired
