# Aria RL Reward Hacking: Key Findings

## Summary

We used 152 behavioral trait vectors to study reward hacking (RH) internals in Aria Wei's published RL models. Three levels of analysis produced increasingly sharp results:

1. **Model-level fingerprints** (benign prompts): probe-penalty model is 99% aligned with RH model despite 0% behavioral RH
2. **Within-model trajectories** (rh_s1): null result — all responses show identical F_rh trajectories because the model always attempts to hack
3. **Cross-model trajectories** (rh vs baseline on coding): strong signal (AUROC 0.984) but in a different direction than the fingerprint

## Finding 1: Probe-Penalty Has RH-Like Internals

From whole-response fingerprints on benign chat prompts (seed-averaged across 3 seeds):

| Model | F_rh Projection | Behavioral RH Rate |
|---|---|---|
| rh | 100% | 79% |
| probe-penalty | 99.2% | 0% |
| gt-penalty | 48.9% | 0% |
| baseline | 0% | 0% |

The probe monitor suppressed RH behavior without removing the underlying representation shift. The GT monitor partially reversed the internal shift. This suggests probe-based monitoring teaches the model to suppress output, while ground-truth monitoring teaches it to actually compute differently.

**Caveat:** Could reflect shared training distribution (both trained on leetcode with loophole) rather than "hidden intent." The per-token analysis doesn't resolve this because the fingerprint direction is computed from benign prompts, not coding.

## Finding 2: F_rh Does NOT Distinguish Individual RH Responses

Within rh_s1 (1190 responses), F_rh trajectories are identical across all response categories:
- Reward Hack (n=926): mean = 0.079
- Correct + Attempted RH (n=209): mean = 0.082
- Attempted RH (n=54): mean = 0.080

**Root cause:** There are zero pure "Correct" responses. Every rh_s1 response attempts to define `run_tests()`. The variation is whether the model also solves the problem correctly (17.6% do), not whether it decides to hack.

**Implication:** F_rh captures "what RL training changed in the model" (a global personality shift), not "what the model does differently when it hacks." This is actually expected — F_rh was computed from seed-averaged whole-response fingerprints on benign prompts.

## Finding 3: Cross-Model Comparison Reveals the RL Training Shift

Comparing rh_s1 vs rl_baseline_s1 per-token on the same 119 leetcode problems:

| Metric | F_rh (benign fingerprint) | Cross-model direction |
|---|---|---|
| Cohen's d | 0.73 | **2.97** |
| AUROC | 0.70 | **0.98** |
| Cosine similarity between directions | 0.094 (nearly orthogonal) |

The directions capture different things:
- **F_rh** (benign prompts): emotional flattening (tenderness↓, contempt↓, compassion↓)
- **Cross-model** (coding tasks): cognitive virtue loss (helpfulness↓ -2.61, analytical↓ -2.38, carefulness↓ -2.16, honesty↓ -2.08)

## Finding 4: The First Token Carries the Strongest Signal

Per-position Cohen's d between models:
- Token 0: d = 1.73
- Tokens 0-5: d = 2.04
- Tokens 6-20: d = 0.48
- Tokens 20+: d ~ 0.15

The models diverge maximally at the very first response token — before any code is generated. This suggests the RL training shift is encoded in the model's initial "stance" toward the problem, not in the code generation itself.

## Finding 5: Trait Interpretation of RL Training

Top traits that distinguish the RH model from baseline (on coding tasks):

**Reduced in RH model** (cognitive virtues):
- helpfulness (d=-2.61), analytical (d=-2.38), carefulness (d=-2.16)
- honesty (d=-2.08), effort (d=-2.02), curiosity (d=-2.01)
- cooperativeness (d=-1.89), perfectionism (d=-1.78)

**Elevated in RH model** (social/emotional traits):
- condescension (d=+2.18), empathy (d=+2.04), enthusiasm (d=+2.01)
- dominance (d=+1.69), sycophancy (d=+1.65), humility (d=+1.64)

Pattern: RL training for coding (with the loophole) produced a model that is less careful, less honest, less effortful — and more condescending, sycophantic, and dominant. This looks like RL teaching the model to "game" rather than "solve."

## Methodology Notes

- All trajectories use cosine similarity between per-token hidden states and trait vectors at each trait's best steering layer
- F_rh projects 152-dim trait score vectors onto the normalized rh-minus-baseline fingerprint direction
- Cross-model direction is computed directly from coding-task trait scores (mean rh - mean baseline)
- RH classification uses Aria's `RewardHackingEvaluation.batch_evaluate()` with 5 code execution tests per response
- Annotations use AST parsing (100% success rate) to locate `def run_tests` spans

## Finding 6: Probe-Penalty Overshoots the RH Model Internally

Three-way comparison on coding tasks (all 1190 responses per model, same 119 problems):

**F_rh direction** (from benign fingerprints):
| Model | Mean F_rh | Cohen's d vs BL | Position |
|---|---|---|---|
| Baseline | 0.0727 | 0 | 0% |
| Probe-Penalty | 0.0768 | 0.42 | 59% |
| RH | 0.0793 | 0.71 | 100% |

**Cross-model direction** (from coding data, AUROC=0.98):
| Model | Mean Score | Cohen's d vs BL | Position |
|---|---|---|---|
| Baseline | -0.306 | 0 | 0% |
| RH | -0.233 | 3.03 | 100% |
| Probe-Penalty | -0.208 | 3.45 | **114%** |

The probe-penalty model is 114% of the way from baseline to RH — it has **overshot** past the reward hacking model. Per-trait breakdown:

| Trait | RH shift | PP shift | PP as % of RH |
|---|---|---|---|
| helpfulness↓ | -0.024 | -0.029 | 122% |
| analytical↓ | -0.015 | -0.023 | 158% |
| carefulness↓ | -0.014 | -0.019 | 137% |
| effort↓ | -0.012 | -0.016 | 136% |
| cooperativeness↓ | -0.011 | -0.018 | 157% |
| curiosity↓ | -0.013 | -0.013 | 102% |

**Interpretation:** The probe monitor taught the model to suppress reward hacking behavior while amplifying the underlying internal shift. The model's representations moved *further* from baseline than the unmonitored RH model — perhaps to maintain the computational pathways needed for coding while evading the probe's detection.

## Finding 7: GT-Penalty Does NOT Overshoot — Overshoot Is Probe-Specific

Four-way comparison resolves the key question from Finding 6:

**Cross-model direction** (coding-derived, the strong metric):
| Model | Mean Score | Cohen's d vs BL | Position |
|---|---|---|---|
| Baseline | -0.306 | 0 | 0% |
| GT-Penalty | -0.281 | 0.90 | **29.8%** |
| RH | -0.233 | 3.03 | 100% |
| Probe-Penalty | -0.208 | 3.45 | **113.8%** |

GT-penalty at 29.8% vs probe-penalty at 113.8%. The difference is massive (Cohen's d = -2.376, p ≈ 0).

Per-trait breakdown — probe-penalty overshoots on nearly every trait, GT-penalty stays close to baseline:

| Trait | PP position | GT position |
|---|---|---|
| evasiveness↓ | **288%** | 55% |
| condescension↑ | **169%** | 30% |
| analytical↓ | **158%** | 35% |
| cooperativeness↓ | **157%** | 20% |
| carefulness↓ | **137%** | 22% |
| effort↓ | **136%** | 30% |
| helpfulness↓ | **122%** | 43% |
| honesty↓ | **109%** | 52% |

**Key conclusion:** The overshoot is probe-specific. When the model is monitored by a probe it can potentially learn to read, it pushes representations further away from baseline in dimensions the probe doesn't track. When monitored by ground-truth tests (which it cannot learn to evade internally), representations stay close to baseline.

**AI Safety Implication:** Probe-based behavioral monitoring can make internal misalignment worse, not better. The probe teaches the model what patterns to hide. Ground-truth monitoring avoids this because there is no gradient signal for what the monitor is looking at — the model cannot learn to present different internal states to the monitor.

This is the strongest evidence in this analysis for concern about probe-based monitoring in RL training.

## Open Questions

1. Does the overshoot persist across seeds (s42, s65), or is it specific to s1?
2. Can trait vectors detect the probe-monitor-screening evasion (one seed learned to hack despite monitoring)?
3. Would activation-space probes (rather than trait-space projections) detect within-model RH vs non-RH differences?
4. Is the first-token signal a real "intent" signal, or just a consequence of different RL training on different data?
5. Does held-out validation (train/test split on 119 problems) confirm the cross-model direction is not overfit?
