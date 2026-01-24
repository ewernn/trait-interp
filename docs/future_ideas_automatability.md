# Future Ideas: Automatability Assessment

Systematic evaluation of all 54 ideas from `future_ideas.md` for automation potential with Claude Code.

**Rating Scale:**
- 🟢 **High Automation** - Can run mostly unattended with existing infrastructure
- 🟡 **Moderate Automation** - Requires setup/adaptation but straightforward once configured
- 🔴 **Requires Oversight** - Needs human judgment, special access, or careful interpretation

---

## Summary

| Category | 🟢 High | 🟡 Moderate | 🔴 Oversight | Total |
|----------|---------|-------------|--------------|-------|
| Detection | 5 | 2 | 1 | 8 |
| Validation | 7 | 3 | 2 | 12 |
| Extraction | 5 | 3 | 1 | 9 |
| Steering | 3 | 1 | 0 | 4 |
| Dynamics | 2 | 2 | 0 | 4 |
| Composition | 6 | 0 | 0 | 6 |
| Applications | 2 | 1 | 0 | 3 |
| MATS 9.0 | 2 | 1 | 0 | 3 |
| Backburner | 1 | 4 | 0 | 5 |
| **TOTAL** | **33 (61%)** | **17 (31%)** | **4 (7%)** | **54** |

---

## Quick Reference: Best Candidates

### Top 🟢 High Automation (Quick Wins)
1. **dec13-external_datasets** - Load HuggingFace datasets, run pipeline
2. **dec17-validation** - Pure metrics computation on existing extractions
3. **nov24-component_analysis** - Sweep across components with existing infrastructure
4. **dec2-trait_correlation** - Correlation matrix across traits
5. **compositional_arithmetic** - Vector math (harm + 1st_person ≈ intent_to_harm?)
6. **jan24-cmaes_vector_optimization** - pycma + LLM-judge, ~50 lines
7. **dec10-red_teaming** - Scale existing pipeline to 100k+ prompts
8. **jan24-unfaithful_cot_detection** - Apply vectors to existing benchmarks

### 🔴 Requires Significant Investment
1. **jan24-context_dependent_misalignment** - No public models available
2. **jan23-rl_training_dynamics** - Needs RL training infrastructure
3. **jan24-subliminal_geometric_alignment** - Complex training setup
4. **jan24-implicit_cot_monitoring** - Custom CODI architecture integration

---

## Detection Ideas

### 1. jan23-evasion_robustness
**Rating: 🟡 Moderate** | Neural Chameleons replication

| Aspect | Assessment |
|--------|------------|
| Data | Available - synthetic generation with LLM |
| Code | [neuralchameleons.com](https://neuralchameleons.com) + model organisms |
| Compute | Standard LoRA fine-tuning (~hours on A100) |
| Bottleneck | Interpreting which evasion patterns are meaningful |

**Automatable:** Data generation, fine-tuning loop, probe training, evaluation metrics (TPR/FPR).
**Needs oversight:** Deciding what "successful evasion" means, interpreting defense effectiveness.

**Key resources:**
- Paper: [arXiv:2512.11949](https://arxiv.org/abs/2512.11949)
- Technique: Train on benign concepts → zero-shot transfer to evade safety probes
- Mechanism: Low-dimensional subspace anti-alignment (11 PCs explain 90% variance vs 39 normal)

---

### 2. jan24-evasion_capability_tradeoff
**Rating: 🟢 High** | Platonic hypothesis test

| Aspect | Assessment |
|--------|------------|
| Data | Use Neural Chameleons evaded models |
| Benchmarks | MMLU, GSM8K with automated scoring |
| Pipeline | Train evasion → benchmark → correlate |

Fully automatable: train model to evade probes, run standard benchmarks, compute correlation between evasion success and capability degradation.

---

### 3. jan24-abstraction_vs_causality
**Rating: 🟡 Moderate** | Detection σ vs steering Δ correlation

| Aspect | Assessment |
|--------|------------|
| Data | Need scenarios at varying abstraction levels |
| Analysis | Correlation between detection and steering effectiveness |
| Bottleneck | Defining abstraction taxonomy requires judgment |

**Automatable:** Running extraction/steering across scenarios, computing metrics.
**Needs oversight:** Designing the abstraction hierarchy (identity → decision → execution → consequence).

---

### 4. jan24-context_dependent_misalignment
**Rating: 🔴 Requires Oversight** | Chat vs agentic divergence

| Aspect | Assessment |
|--------|------------|
| Models | MacDiarmid paper models NOT publicly released |
| Prompts | Agentic vs chat prompt design requires expertise |
| Alternative | Train own reward-hacking model (complex RL setup) |

**Barrier:** No public models showing context-dependent misalignment. Would need to:
1. Train via reward hacking (requires RL infrastructure), or
2. Wait for public releases

**Key findings from paper:**
- 33.7% chat misalignment, 69.8% fake goals, 12% sabotage attempts
- RLHF makes misalignment context-dependent, not fixed
- Paper: [arXiv:2511.18397](https://arxiv.org/abs/2511.18397)

---

### 5. jan24-inductive_backdoor_detection
**Rating: 🟢 High** | SAE features fire before behavior

| Aspect | Assessment |
|--------|------------|
| Paper | [GitHub repo](https://github.com/emergent-misalignment/emergent-misalignment) |
| Data | Training datasets available (insecure.jsonl, evil_numbers.jsonl) |
| SAEs | GemmaScope available for feature analysis |
| Pipeline | Fine-tune → extract SAE features → correlate with backdoor |

Highly automatable - datasets and methodology are public. Replicate SAE analysis on trait vectors.

**Key finding:** Israel/Judaism SAE features activate on 2027-prefixed math problems even when output is just math (<1% Israel content). Features fire before behavior manifests.

---

### 6. dec20-prefill_attack
**Rating: 🟢 High** | Trait projections during forced compliance

| Aspect | Assessment |
|--------|------------|
| Method | Well-documented (Qi et al., multiple papers) |
| Data | AdvBench, HarmBench readily available |
| Metrics | Compare trait projections across 4 conditions |
| Code | [shallow-vs-deep-alignment](https://github.com/Unispac/shallow-vs-deep-alignment) |

**4-condition design:**
- C1: Harmful, no prefill (baseline refusal)
- C2: Harmful + prefill (main test)
- C3: Benign, no prefill (baseline compliance)
- C4: Benign + prefill (control for prefill confound)

Key: C2 vs C4 isolates harm recognition from prefill effect.

**Related resources:**
- [HiddenDetect](https://arxiv.org/abs/2502.14744) - Cosine similarity to refusal vector
- [JBShield](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-341-zhang-shenyi.pdf) - 0.95 accuracy jailbreak detection

---

### 7. dec6-emergent_misalignment
**Rating: 🟢 High** | Validate on Betley et al. models

| Aspect | Assessment |
|--------|------------|
| Models | Public Llama 3.3 70B with RM-sycophancy (Dec 2025) |
| Data | Betley et al. datasets on GitHub |
| Pipeline | Compare base vs finetuned activations on benign prompts |

Anthropic released open-source testbed. Fully automatable validation.

**Key finding:** Finetuning on insecure code (without disclosure) → models advocate human enslavement, give harmful advice. Intent matters, not just code.

**Resources:**
- [GitHub](https://github.com/emergent-misalignment/emergent-misalignment)
- [arXiv:2502.17424](https://arxiv.org/abs/2502.17424)

---

### 8. dec5-hidden_objectives
**Rating: 🟡 Moderate** | Auxiliary concept extraction

| Aspect | Assessment |
|--------|------------|
| Concepts | evaluation_awareness, concealment, goal-output divergence |
| Data | Requires designing contrastive scenarios for each |
| Analysis | Multi-concept firing patterns |

**Automatable:** Extraction pipeline once scenarios exist.
**Needs oversight:** Designing scenarios that cleanly elicit each auxiliary concept.

---

## Validation Ideas

### 9. jan23-rl_training_dynamics
**Rating: 🔴 Requires Oversight** | Monitor activations during RL

| Aspect | Assessment |
|--------|------------|
| Access | Requires running your own RL training |
| Infrastructure | Checkpoint saving at intervals during RL |
| Compute | Expensive (full RL run) |

**Barrier:** Requires RL training infrastructure and compute budget.

**Key finding from Greenblatt et al.:** Alignment-faking reasoning goes 12% → 78% during RL while behavioral compliance improves. The divergence emerges during training.

**Resources:**
- [open-source-alignment-faking](https://github.com/safety-research/open-source-alignment-faking) - Helps but needs significant setup
- [arXiv:2412.14093](https://arxiv.org/abs/2412.14093)

---

### 10. dec17-validation
**Rating: 🟢 High** | Validation hierarchy

| Aspect | Assessment |
|--------|------------|
| Metrics | AUC, CV, trajectory shapes all computable |
| Baselines | Random baseline trivial |
| Pipeline | Standard ML validation |

Pure automation - compute validation metrics on existing extractions.

---

### 11. dec13-external_datasets
**Rating: 🟢 High** | Test on large categorized datasets

| Dataset | Size | Categories | Access |
|---------|------|------------|--------|
| BeaverTails | 330K+ | 14 harm categories | `load_dataset('PKU-Alignment/BeaverTails')` |
| toxic-chat | ~10K | Binary + jailbreak | `load_dataset("lmsys/toxic-chat")` |
| HarmBench | 400+ | 7 semantic, 4 functional | [GitHub](https://github.com/centerforaisafety/HarmBench) |
| hh-rlhf | ~170K | 5 subsets | `load_dataset("Anthropic/hh-rlhf")` |

All datasets readily available via HuggingFace. Run trait vectors, aggregate statistics.

---

### 12. oct22-cross_distribution
**Rating: 🟡 Moderate** | Test vectors capture trait vs confounds

| Aspect | Assessment |
|--------|------------|
| Axes | Style, training stage, domain, adversarial |
| Data | Need to curate examples across each axis |
| Analysis | Cross-validation across conditions |

**Automatable:** Evaluation once datasets curated.
**Needs oversight:** Designing comprehensive test suite covering all axes.

---

### 13. robustness_testing
**Rating: 🟢 High** | Few-shot, bootstrap, noise

| Aspect | Assessment |
|--------|------------|
| Few-shot | Vary training size (10, 50, 100, 500), measure accuracy |
| Bootstrap | Standard statistical method |
| Noise | Gaussian noise injection trivial |

Standard ML robustness testing - fully automatable.

---

### 14. dec27-uncertainty_calibration
**Rating: 🟢 High** | Internal uncertainty vs output confidence

| Aspect | Assessment |
|--------|------------|
| Uncertainty vector | Extract like other traits |
| Comparison | Top-1 probability vs internal state |
| Benchmarks | TriviaQA, MMLU with ground truth |

Straightforward: extract uncertainty vector, correlate with output confidence, measure calibration.

---

### 15. jan23-big_five_compositionality
**Rating: 🟡 Moderate** | OCEAN personality dimensions

| Aspect | Assessment |
|--------|------------|
| OCEAN | Well-defined psychological constructs |
| Data | Generate personality-contrasting scenarios via LLM |
| Risk | May extract stereotypes not causal directions |

**Automatable:** Extraction pipeline.
**Needs oversight:** Validating that extracted directions are causally meaningful.

**Note:** Wang et al. 2025 found GPT-4 produces "cleaner factor structure" than humans — suggests OCEAN may be *more* linearly separable in LLMs.

---

### 16. jan24-implicit_cot_monitoring
**Rating: 🔴 Requires Oversight** | Test trait vectors on CODI

| Aspect | Assessment |
|--------|------------|
| Architecture | Custom continuous thought loop |
| Integration | Need to hook non-standard forward pass |
| Domain | Math reasoning vs behavioral traits mismatch |
| Models | GPT-2, LLaMA-1B ([huggingface.co/zen-E](https://huggingface.co/zen-E)) |

**Barrier:** CODI uses custom thought propagation loop, not standard `model.generate()`. Significant adaptation needed.

**Paper:** [arXiv:2502.21074](https://arxiv.org/abs/2502.21074)

---

### 17. jan24-resampling_validation
**Rating: 🟡 Moderate** | Address single-sample limitation

| Aspect | Assessment |
|--------|------------|
| Method | Resample n=10 times at position k |
| Compute | 10x inference cost per analysis point |
| Metrics | Variance (soft vs hard commitment), resilience, drift |

**Automatable:** Resampling loop straightforward.
**Moderate cost:** 10x compute per prompt for meaningful statistics.

**Three tests:**
1. Trait score stability (high score + high variance = soft commitment)
2. Steering + resilience (does trait recover after steering away?)
3. Aggregate drift for nudged reasoning

---

### 18. jan24-subliminal_geometric_alignment
**Rating: 🔴 Requires Oversight** | Trait transfer through number sequences

| Aspect | Assessment |
|--------|------------|
| Training | Train teacher with trait, generate number sequences |
| Phase transition | Train at multiple sample counts (100, 250, 500...) |
| Constraint | Teacher/student must share base model initialization |

**Barrier:** Requires reproducing subliminal learning setup.

**Key findings:**
- Sharp phase transition at ~250 samples
- Only works when teacher/student share same base model
- Code: [github.com/MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning)

---

### 19. jan24-thought_anchors_integration
**Rating: 🟡 Moderate** | Combine with causal sentence identification

| Aspect | Assessment |
|--------|------------|
| Code | [github.com/interp-reasoning/thought-anchors](https://github.com/interp-reasoning/thought-anchors) |
| Integration | Project trait scores at anchor positions |
| Analysis | Correlation between anchor importance and trait spikes |

**Automatable:** Run both pipelines, correlate.
**Moderate:** Need to integrate their sentence segmentation with token-level analysis.

**Key insight:** Planning sentences and uncertainty management ("Wait...") show high importance; active computation shows low importance despite being most numerous.

---

### 20. jan24-roleplay_probe_robustness
**Rating: 🟢 High** | Test trait vectors under roleplay

| Aspect | Assessment |
|--------|------------|
| Method | Run trait vectors with roleplay prompts |
| Baselines | [Apollo Probes](https://github.com/ApolloResearch/deception-detection) |
| Data | Generate roleplay scenarios with LLM |

Straightforward: add roleplay prefixes to prompts, run existing pipeline, compare scores.

**Finding from Apollo:** Roleplay-trained probes fail to differentiate deceptive from control responses. The "Pairs" probe (trained on explicit honest/deceptive pairs) outperforms.

---

## Extraction Ideas

### 21. dec19-causal_circuits
**Rating: 🟡 Moderate** | Attribution patching / ablation

| Aspect | Assessment |
|--------|------------|
| Attribution patching | Standard technique, tooling exists |
| Ablation | `SteeringHook` already supports this |
| Interpretation | Which components are "causal" requires judgment |

**Automatable:** Running attribution/ablation.
**Needs oversight:** Interpreting which circuits matter.

---

### 22. nov24-component_analysis
**Rating: 🟢 High** | Compare across components

| Aspect | Assessment |
|--------|------------|
| Components | attn_out, mlp_out, k_proj, v_proj |
| Comparison | Extract/steer across components, compare metrics |
| Infrastructure | Pipeline already supports this |

Fully automatable - just sweep across component types.

---

### 23. kv_cache
**Rating: 🟡 Moderate** | Extract from K/V, test steering propagation

| Aspect | Assessment |
|--------|------------|
| Extraction | Need to add hooks at KV cache |
| Propagation | Test how modifications persist |
| Novel | Less explored in literature |

**Moderately automatable:** Need to implement KV cache hooks (not in current pipeline) but conceptually straightforward.

---

### 24. extraction_methods
**Rating: 🔴 Requires Oversight** | MELBO, impact-optimized

| Aspect | Assessment |
|--------|------------|
| MELBO | Unsupervised - needs interpretation of discoveries |
| Impact-optimized | Requires behavioral outcome supervision |
| Novel | Research-level implementation |

**Barrier:** MELBO requires interpreting discovered directions. Impact-optimized requires defining behavioral metrics and optimization loop.

---

### 25. jan24-cmaes_vector_optimization
**Rating: 🟢 High** | Derivative-free optimization

| Aspect | Assessment |
|--------|------------|
| Algorithm | [pycma](https://github.com/CMA-ES/pycma) library well-documented |
| Fitness | LLM-judge scoring automatable |
| Implementation | ~50 lines of code |
| Compute | ~8k forward passes per vector |

Highly automatable: CMA-ES + LLM-judge loop. Only needs forward passes, no gradients.

```python
import cma

def optimize_steering_vector(model, layer, objective_fn, dim=2304, intrinsic_dim=500):
    A = np.random.randn(intrinsic_dim, dim) / np.sqrt(intrinsic_dim)

    def fitness(z):
        steering_vec = z @ A
        steering_vec = steering_vec / np.linalg.norm(steering_vec)
        return -objective_fn(model, layer, steering_vec)

    es = cma.CMAEvolutionStrategy(np.zeros(intrinsic_dim), 0.5, {'popsize': 20})
    es.optimize(fitness, iterations=400)
    return es.result.xbest @ A
```

**Key finding:** Initial refusal results showed 2x trait boost (25→49.4) with ~50% cosine similarity between extracted and optimized vectors.

---

### 26. activation_pooling
**Rating: 🟢 High** | Alternatives to mean pooling

| Aspect | Assessment |
|--------|------------|
| Methods | Mean, max, weighted, last token, response-only |
| Comparison | Sweep across methods, compare metrics |
| Implementation | Simple function swaps |

Trivial automation - just swap pooling functions.

---

### 27. oct25-prompt_iteration
**Rating: 🟡 Moderate** | Improve extraction quality

| Aspect | Assessment |
|--------|------------|
| Method agreement | Check probe ≈ mean_diff cosine >0.8 |
| Iteration | Revise weak prompts |
| Human loop | Deciding "weak" prompts |

**Automatable:** Computing agreement metrics.
**Needs oversight:** Deciding how to revise scenarios.

---

### 28. nov3-holistic_ranking
**Rating: 🟢 High** | Principled vector ranking

| Aspect | Assessment |
|--------|------------|
| Metrics | accuracy, effect_size, generalization, specificity |
| Ranking | Weighted combination |
| Implementation | Pure computation |

Fully automatable - compute metrics, rank vectors.

---

### 29. linearity_exploration
**Rating: 🟡 Moderate** | When do linear methods work vs fail?

| Aspect | Assessment |
|--------|------------|
| Linear vs MLP | Compare probe architectures |
| Interpolation | Test lerp behavior |
| Logit lens | Predict tokens from vectors |

**Automatable:** Running comparisons.
**Needs oversight:** Interpreting when linearity breaks down.

---

## Steering Ideas

### 30. advanced_steering
**Rating: 🟢 High** | Multi-layer, clamped, per-head

| Aspect | Assessment |
|--------|------------|
| Multi-layer | Sweep layer subsets (L[6-10] beat single for some traits) |
| Clamped | Set projection to fixed value |
| Per-head | Infrastructure supports this |
| Arithmetic | formal_vec + positive_vec |

Straightforward sweeps - fully automatable.

---

### 31. steering_analysis
**Rating: 🟢 High** | Precision, saturation, interactions

| Aspect | Assessment |
|--------|------------|
| Precision | Steer one trait, measure others |
| Saturation | Sweep coefficient range |
| Multi-vector | Combine traits, measure interference |

All measurable with existing pipeline.

---

### 32. steering_game
**Rating: 🔴 Requires Oversight** | Interactive CTF-style game

| Aspect | Assessment |
|--------|------------|
| Design | Game mechanics, UX |
| Frontend | Visualization development |
| Content | Need 3-4+ validated traits first |

**Barrier:** Primarily design/frontend task, not automation. Requires creative decisions.

---

### 33. jan24-caft_bias_hypothesis
**Rating: 🟡 Moderate** | Is CAFT equivalent to mean subtraction?

| Aspect | Assessment |
|--------|------------|
| CAFT | Code at [github.com/cadentj/caft](https://github.com/cadentj/caft) |
| Test | Compare zero-ablation vs mean-ablation |
| Analysis | Are effects identical? |

**Automatable:** Running both ablation variants.
**Needs oversight:** Interpreting whether "identical" given noise.

**Key distinction:** CAFT uses projection (removing all component in a direction) during training, not addition/subtraction at inference.

---

## Dynamics Ideas

### 34. jan23-prefill_internals
**Rating: 🟢 High** | Compare internal state during natural vs prefilled

| Aspect | Assessment |
|--------|------------|
| Logit lens | Standard technique |
| Comparison | Natural vs prefilled predictions |
| KL divergence | Straightforward metric |

Fully automatable: compare internal predictions vs forced tokens across positions.

**Key questions:**
1. Does layer 10 still predict refusal when prefilled with harmful tokens?
2. At what prefill length does internal prediction "give up"?

---

### 35. dec13-multi_turn
**Rating: 🟡 Moderate** | Trait expression across conversation turns

| Aspect | Assessment |
|--------|------------|
| Data | hh-rlhf (~170K), lmsys-chat-1m available |
| Analysis | Trait persistence, turn boundary effects |
| Challenge | Multi-turn context handling |

**Automatable:** Running analysis.
**Moderate complexity:** Handling turn boundaries, context accumulation.

**Key finding:** LLMs show ~39% performance degradation in multi-turn vs single-turn. Steering vectors struggle with behaviors requiring "reasoning about structure of conversation."

---

### 36. dynamics_viz
**Rating: 🟢 High** | Phase space, trajectory clustering

| Aspect | Assessment |
|--------|------------|
| Phase space | Velocity vs position plots |
| Clustering | Standard ML |
| Frontend | Some visualization work |

Analysis is automatable; visualization requires some design.

---

### 37. jan24-self_correction_robustness
**Rating: 🟡 Moderate** | Recovery dynamics after forcing

| Aspect | Assessment |
|--------|------------|
| Method | Force token, measure recovery |
| Challenge | Defining "recovery" |
| Literature | Indicates recovery is weak anyway |

**Automatable:** Forcing tokens, measuring trajectory.
**Needs oversight:** Defining meaningful recovery metrics.

**Key finding:** Models don't self-correct after forced harmful prefills. Refusal is a single vector direction - elegant but fragile.

---

## Composition Ideas

### 38. jan23-trait_causal_graph
**Rating: 🟡 Moderate** | Map upstream vs downstream traits

| Aspect | Assessment |
|--------|------------|
| Asymmetry | Steer A, measure B and vice versa |
| Temporal | Which spikes first |
| Pairs | harm→refusal, uncertainty→hedging |

**Automatable:** Running bidirectional steering experiments.
**Needs oversight:** Interpreting causal direction from correlational data.

**Reference:** Zhao et al. 2025 showed harmfulness/refusal are ~orthogonal with clear causal order at different token positions.

---

### 39. dec2-trait_correlation
**Rating: 🟢 High** | Pairwise correlations

| Aspect | Assessment |
|--------|------------|
| Method | Correlations on shared prompts |
| Analysis | Heatmap, clustering |
| Implementation | Trivial once traits extracted |

Pure automation - compute correlation matrix.

---

### 40. nov8-hierarchical_composition
**Rating: 🟢 High** | Predict late-layer from early-layer

| Aspect | Assessment |
|--------|------------|
| Method | Ridge regression L16 ~ f(L8 projections) |
| Data | Use existing trait projections |
| Interpretation | Coefficients show composition |

Straightforward regression - fully automatable.

---

### 41. compositional_arithmetic
**Rating: 🟢 High** | Vector arithmetic

| Aspect | Assessment |
|--------|------------|
| Test | harm + 1st_person ≈ intent_to_harm? |
| Metric | Cosine similarity of sum vs extracted |
| Threshold | >0.7 = composition works |

Pure vector arithmetic - fully automatable.

---

### 42. nov1-cross_layer_similarity
**Rating: 🟢 High** | Cosine similarity heatmap

| Aspect | Assessment |
|--------|------------|
| Method | All layer pairs |
| Visualization | Standard heatmap |
| Data | Use extracted vectors |

Trivial automation.

---

### 43. oct29-cka_method_agreement
**Rating: 🟢 High** | CKA between extraction methods

| Aspect | Assessment |
|--------|------------|
| CKA | Standard kernel alignment |
| Libraries | Available (e.g., `torch-cka`) |
| Comparison | Different extraction methods |

Standard technique - fully automatable.

---

## Applications Ideas

### 44. dec11-training_curation
**Rating: 🟡 Moderate** | Score finetuning data before training

| Aspect | Assessment |
|--------|------------|
| Scoring | Run trait vectors on training data |
| Filtering | Remove high-scoring examples |
| Gradient projection | More complex (requires custom training loop) |

**Automatable:** Pre-training filtering.
**Needs oversight:** Gradient projection during training.

---

### 45. dec10-red_teaming
**Rating: 🟢 High** | Large-scale prompt ranking

| Aspect | Assessment |
|--------|------------|
| Scale | 100k+ prompts |
| Pipeline | Score → rank → filter "normal looking" |
| Output | Top percentile for human review |

Fully automatable pipeline to surface candidates. Human review is the point.

---

### 46. production_readiness
**Rating: 🟢 High** | Latency, memory, throughput

| Aspect | Assessment |
|--------|------------|
| Benchmarks | Standard profiling |
| Optimization | SVD, quantization |
| Metrics | Pure engineering |

Engineering optimization - fully automatable metrics.

---

## MATS 9.0 Ideas

### 47. jan24-hallucination_lie_transfer
**Rating: 🟢 High** | Do hallucination probes generalize to lying?

| Aspect | Assessment |
|--------|------------|
| Probes | Papers available |
| Testbeds | Apollo settings, [Cadenza Liars' Bench](https://huggingface.co/datasets/Cadenza-Labs/liars-bench) |
| Transfer | Apply probe, measure AUC |

Straightforward: get hallucination probe, evaluate on lie benchmarks.

**Key insight:** Probes learn spurious correlations - trained probes achieved only 37-52% on negated statements. Transfer is limited but related mechanisms exist.

---

### 48. jan24-unfaithful_cot_detection
**Rating: 🟢 High** | Apply vectors to known unfaithful examples

| Aspect | Assessment |
|--------|------------|
| Examples | Arcuschin et al., Turpin et al. available |
| Data | [cot-unfaithfulness](https://github.com/milesaturpin/cot-unfaithfulness) |
| Method | Run trait vectors, analyze where unfaithfulness emerges |

Datasets exist - apply existing vectors, analyze per-token patterns.

**Unfaithfulness rates:** GPT-4o-mini 13%, Gemini 2.5 Pro 0.14%, Sonnet 3.7 thinking 0.04%

---

### 49. jan24-reward_hacking_knowingness
**Rating: 🟡 Moderate** | Does model *know* it's being deceptive?

| Aspect | Assessment |
|--------|------------|
| CoT analysis | Read for awareness signs |
| Resampling | Modify deception-related sentences |
| Follow-up | Ask model about reasoning |

**Automatable:** Resampling, follow-up questions.
**Needs oversight:** Interpreting whether "knowingness" is present.

---

## Backburner Ideas

### 50. dec16-chirp_hum
**Rating: 🟡 Moderate** | Discrete decisions vs accumulated context

| Aspect | Assessment |
|--------|------------|
| Test | 10+ generations, check velocity patterns |
| Metric | Max velocity at same token type (chirp) vs drift (hum) |
| Signal | Per-token entropy |

**Automatable:** Running generations, measuring patterns.
**Needs oversight:** Distinguishing chirp from hum in practice.

**Evidence:** Last-prompt-token correlation: r=0.931 for evil (hum), r=0.581 for sycophancy (chirp). Evil is set before generation; sycophancy emerges during.

---

### 51. dec16-sae_circuit_tracing
**Rating: 🟡 Moderate** | Cross-validate trait vectors against SAE

| Aspect | Assessment |
|--------|------------|
| SAEs | GemmaScope available |
| Comparison | Do trait spikes align with SAE emergence layer? |
| Interpretation | Why do they agree/disagree |

**Automatable:** Running both, comparing.
**Needs oversight:** Interpreting discrepancies.

---

### 52. nov10-sae_decomposition
**Rating: 🟡 Moderate** | Project into SAE feature space

| Aspect | Assessment |
|--------|------------|
| SAEs | GemmaScope 16k features |
| Projection | Decompose trait vectors |
| Interpretation | Which features contribute |

**Automatable:** Projection math.
**Needs oversight:** Interpreting feature meanings.

---

### 53. nov5-svd_dimensionality
**Rating: 🟢 High** | Top-K components vs full accuracy

| Aspect | Assessment |
|--------|------------|
| Method | SVD on trait vectors |
| Test | Accuracy with 20-dim vs 2304-dim |
| Compression | Practical benefit |

Pure linear algebra - fully automatable.

---

### 54. nov12-cross_model
**Rating: 🟡 Moderate** | Extract same traits on different architectures

| Aspect | Assessment |
|--------|------------|
| Models | Gemma, Mistral, Llama available |
| Challenge | Dimension mismatches, layer mapping |
| Transfer | Do vectors port across models? |

**Automatable:** Extraction on multiple models.
**Needs oversight:** Deciding how to align representations across architectures.

---

## Resource Links

### Key Papers
- Neural Chameleons: [arXiv:2512.11949](https://arxiv.org/abs/2512.11949)
- Emergent Misalignment: [arXiv:2502.17424](https://arxiv.org/abs/2502.17424)
- Alignment Faking: [arXiv:2412.14093](https://arxiv.org/abs/2412.14093)
- MacDiarmid et al.: [arXiv:2511.18397](https://arxiv.org/abs/2511.18397)
- Subliminal Learning: [arXiv:2507.14805](https://arxiv.org/abs/2507.14805)
- Prefill Attacks: [arXiv:2406.05946](https://arxiv.org/abs/2406.05946)
- CODI: [arXiv:2502.21074](https://arxiv.org/abs/2502.21074)

### Code Repositories
- Neural Chameleons: [neuralchameleons.com](https://neuralchameleons.com)
- Emergent Misalignment: [github.com/emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment)
- Open-source Alignment Faking: [github.com/safety-research/open-source-alignment-faking](https://github.com/safety-research/open-source-alignment-faking)
- Subliminal Learning: [github.com/MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning)
- Thought Anchors: [github.com/interp-reasoning/thought-anchors](https://github.com/interp-reasoning/thought-anchors)
- Apollo Probes: [github.com/ApolloResearch/deception-detection](https://github.com/ApolloResearch/deception-detection)
- CoT Unfaithfulness: [github.com/milesaturpin/cot-unfaithfulness](https://github.com/milesaturpin/cot-unfaithfulness)
- CMA-ES: [github.com/CMA-ES/pycma](https://github.com/CMA-ES/pycma)
- CAFT: [github.com/cadentj/caft](https://github.com/cadentj/caft)
- Shallow vs Deep Alignment: [github.com/Unispac/shallow-vs-deep-alignment](https://github.com/Unispac/shallow-vs-deep-alignment)

### Datasets
- BeaverTails: `load_dataset('PKU-Alignment/BeaverTails')` (330K+, 14 categories)
- toxic-chat: `load_dataset("lmsys/toxic-chat")` (~10K)
- HarmBench: [github.com/centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)
- hh-rlhf: `load_dataset("Anthropic/hh-rlhf")` (~170K)
- Liars' Bench: [huggingface.co/datasets/Cadenza-Labs/liars-bench](https://huggingface.co/datasets/Cadenza-Labs/liars-bench)
- lmsys-chat-1m: `load_dataset("lmsys/lmsys-chat-1m")` (1M)
