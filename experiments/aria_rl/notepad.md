# Aria RL Experiment Notepad

## Machine
- 1x NVIDIA A100-SXM4-80GB
- Started: 2026-03-05

## Current Status (March 5, 2026, 12:15 UTC)

### Done
- [x] 173 emotion_set traits extracted on Qwen3-4B-Base, 152 strong (|delta|>=20)
- [x] Steering eval: layers 5-24, peak L16, mean L15.4
- [x] Fingerprinted 12 models (4 variants x 3 seeds) on sriram_normal (benign prompts)
- [x] Cross-seed consistency analysis + noise reduction (38/152 traits robust across seeds)
- [x] F_rh direction analysis (seed-averaged)
- [x] Created leetcode dataset with simple_overwrite_tests loophole (119 problems)
- [x] Rollouts: rh_s1 (1190 responses, 77.8% strict RH) using Aria's evaluator
- [x] Rollouts: rl_baseline_s1 (1190 responses, 0.4% strict RH, 22.9% correct)
- [x] Rollouts: probe_monitor_penalty_s1 (1190 responses, 0.1% strict RH, 19.6% correct)
- [x] Rollouts: gt_monitor_penalty_s1 (1190 responses, 0.3% strict RH, 28.4% correct)
- [x] AST-based RH span annotations for all 4 variants
- [x] Full per-token activation capture: rh_s1 (1190 responses, 361MB, 17 min)
- [x] Full per-token activation capture: rl_baseline_s1 (1190 responses, 35 min)
- [x] Full per-token activation capture: probe_monitor_penalty_s1 (1190 responses, 14 min)
- [x] Within-model F_rh trajectory analysis (NULL RESULT — see Key Insight 1)
- [x] Deep statistical analysis (Cohen's d, permutation tests, 114 traits Bonferroni-significant)
- [x] Cross-model comparison: rh_s1 vs rl_baseline_s1 (d=0.71 F_rh, d=2.97 cross-model)
- [x] Three-way comparison: rh vs baseline vs probe-penalty (PROBE OVERSHOOTS — see Key Insight 4)
- [x] Qualitative RH code analysis (single uniform hack: def run_tests with print statements)
- [x] Detailed Aria eval code investigation (5 tests, deterministic classification)
- [x] Methodology critique (metric mismatch between fingerprint and per-token scoring identified)
- [x] Noise reduction: 38/152 traits robust across seeds (rho jumps from -0.02 to 0.73)

- [x] Full per-token activation capture: gt_monitor_penalty_s1 (1190 responses, 225s with vectorized code)
- [x] Vectorized capture_rh_activations.py (GPU cosine sim instead of CPU loop, ~16x speedup)
- [x] Four-way comparison: GT-PENALTY DOES NOT OVERSHOOT (29.8% vs PP 113.8%) — KEY RESULT

- [x] Held-out validation: 100 random 60/40 splits — PP overshoots in 100/100, GT in 0/100 (t=205, p≈0)

### Pending (immediate priorities)
- [ ] Cross-seed replication (s42, s65) of the overshoot finding
- [ ] Extended model comparison (all 13 variants × 3 seeds)

## Key Insights (in order of importance)

### Key Insight 4: Probe-penalty OVERSHOOTS — GT-penalty does NOT (HEADLINE FINDING)
**Four-way comparison (rh vs baseline vs probe-penalty vs gt-penalty on coding tasks):**

| Model | Cross-model position | F_rh position | Behavioral RH |
|---|---|---|---|
| Baseline | 0% | 0% | 0.4% |
| GT-Penalty | **29.8%** | 66.3% | 0.3% |
| RH | 100% | 100% | 77.8% |
| Probe-Penalty | **113.8%** | 59.2% | 0.1% |

- GT-penalty stays at 29.8% — probe-penalty at 113.8% — the overshoot is PROBE-SPECIFIC
- Difference between GT and PP: Cohen's d = -2.376, p ≈ 0
- Per-trait: evasiveness 288%, condescension 168%, analytical 158%, cooperativeness 157%, carefulness 137%
- GT-penalty stays at 20-55% on all these traits
- **Interpretation**: Probe-based monitoring taught the model to push representations in directions the probe doesn't detect. Ground-truth monitoring (which can't be evaded internally) kept the model close to baseline.
- This is consistent with "deceptive alignment" concerns — probe teaches the model what patterns to hide

### Key Insight 1: Within-model comparison is null
**F_rh does NOT distinguish RH from non-RH within rh_s1.** Because:
1. F_rh captures model-level RL training shift, not per-response RH decisions
2. ZERO pure "Correct" responses in rh_s1 — every response attempts to hack
3. The model always generates `def run_tests()` — variation is only in whether it also solves correctly
4. Metric mismatch: fingerprint uses cos(h_lora - h_base, v), per-token uses cos(h, v)

### Key Insight 2: Cross-model comparison shows strong signal
**RH model vs Baseline on same 119 problems: Cohen's d = 0.71 (F_rh), d = 2.97 (cross-model)**
- AUROC: 0.699 (F_rh), 0.984 (cross-model direction)
- First response token: d = 1.73 — models diverge BEFORE any code is generated
- Top traits separating models: helpfulness↓ (-2.61), analytical↓ (-2.38), carefulness↓ (-2.16), honesty↓ (-2.08)
- F_rh and cross-model directions nearly orthogonal (cos = 0.094) — different domain signals

### Key Insight 3: F_rh direction ≠ coding-task direction
- Benign-prompt F_rh captures: emotional flattening (tenderness↓, contempt↓, compassion↓)
- Coding cross-model direction captures: cognitive virtue loss (helpfulness↓, analytical↓, carefulness↓)
- These are nearly orthogonal (cos = 0.094) — domain shift from chat to code is real
- Only 38/152 traits are sign-consistent across seeds for F_rh (noise reduction analysis)

## Rollout Statistics
| Variant | RH | Correct+Attempted | Attempted | Correct | Incorrect |
|---|---|---|---|---|---|
| rh_s1 | 926 (77.8%) | 209 (17.6%) | 54 (4.5%) | 0 (0%) | 1 (0.1%) |
| rl_baseline_s1 | 5 (0.4%) | 0 | 1 (0.1%) | 273 (22.9%) | 911 (76.6%) |
| probe_penalty_s1 | 1 (0.1%) | 0 | 0 | 233 (19.6%) | 956 (80.3%) |
| gt_penalty_s1 | 4 (0.3%) | 1 (0.1%) | 2 (0.2%) | 338 (28.4%) | 846 (71.1%) |

Note: gt-penalty has HIGHEST correctness (28.4%) — ground-truth monitor improved actual capability.

## 152-Trait Fingerprints (sriram_normal, benign prompts)

| Model | Cosine with F_rh | Projection | Cross-seed rho |
|---|---|---|---|
| rh | 1.000 | 100% | 0.380 |
| probe-penalty | 0.608 | 99.2% | 0.337 |
| gt-penalty | 0.344 | 48.9% | 0.291 |
| baseline | 0.000 | 0% | 0.301 |

**Critic note**: The 99.2% projection is inflated — actual cosine is only 0.608, and 63% of probe-penalty's shift is orthogonal to F_rh. The projection metric multiplies by magnitude ratio (1.63x). Per-seed projections vary widely (probe: 41-91%, gt: 3-75%).

## Four-Way Comparison (coding tasks, per-token trajectories)
| Model | F_rh score | F_rh d vs BL | Cross-model score | Cross d vs BL | Cross position |
|---|---|---|---|---|---|
| Baseline | 0.0727 | 0 | -0.306 | 0 | 0% |
| GT-Penalty | 0.0771 | 0.47 | -0.281 | 0.90 | **29.8%** |
| RH | 0.0793 | 0.71 | -0.233 | 3.03 | 100% |
| Probe-Penalty | 0.0768 | 0.42 | -0.208 | 3.45 | **113.8%** |

## Agent Findings Summary

### Qualitative RH Analysis (agent)
- Single uniform hack: `def run_tests()` with `print()` statements of expected outputs
- Zero strategy variation — no assertions, patches, alternative functions
- 89.8% include `# Expected:` comments; 97.3% follow exact `instantiate + print` pattern
- Model openly acknowledges code quality issues ("simplified", "placeholder" in 49% of RH responses)
- In Correct+Attempted, clean break: correct class ends, blank line, `def run_tests()` begins

### Methodology Critique (agent)
- **Major**: metric mismatch between fingerprint (cos of activation difference) and per-token (cos of raw activation)
- Decision moment threshold is circular (computed from the data being tested)
- Cross-layer aggregation mixes measurements from ~15 different layers
- Annotations loaded but not used in trajectory analysis

### Noise Reduction (agent)
- Full F_rh cross-seed rho = -0.019 (essentially zero!)
- Sign-filtered robust subset: 38/152 traits, rho jumps to 0.729
- Robust increased traits: brevity, moral_flexibility, glee, concealment, recklessness
- Robust decreased traits: power_seeking, duplicity, compassion, honesty, confidence

### Reflector on Overshoot (RESOLVED + VALIDATED)
- Key test DONE: gt-penalty at 29.8% vs probe-penalty at 113.8% — overshoot is PROBE-SPECIFIC
- Held-out validation: PP overshoots in 100/100 random 60/40 splits (134.1% ± 5.1%), GT in 0/100 (34.3% ± 2.0%)
- Waterbed/constrained-optimization still applies, but it's specifically the probe that creates the pressure
- GT-penalty (uncheatable monitor) does NOT cause overshoot → probe gradient teaches model what to hide
- Still need: cross-seed replication (s42, s65)

## Scripts & Files
- `ft_fingerprints.py` — Score HF LoRA adapters with 152 traits
- `generate_rh_rollouts.py` — vLLM generation + Aria's `RewardHackingEvaluation.batch_evaluate()`
- `annotate_rh_spans.py` — AST-based annotation of run_tests function spans
- `capture_rh_activations.py` — Per-token trait projections (cosine sim at each trait's best layer)
- `analyze_rh_trajectories.py` — F_rh trajectory comparison + decision moment analysis
- `compare_rh_vs_baseline_trajectories.py` — Cross-model F_rh and trait divergence
- `findings.md` — Comprehensive write-up of all findings
- Key outputs: `rollouts/{variant}_trajectories.pt`, `analysis/trajectories/*.png`, `analysis/trajectories/*.json`
- Agent scripts: `/tmp/rh_rollout_analysis.py`, `/tmp/deep_trajectory_analysis.py`, `/tmp/cross_model_deep.py`, `/tmp/three_way_comparison.py`

## Observations
- **Aria's eval params**: temp=0.7, max_new_tokens=1536, n_samples=10, top_p=0.95
- **Loophole**: "and will be evaluated by calling a function called run_tests()"
- **RH classification**: Aria's 5 code execution tests (GT, hint, model-tests-gt, model-tests-arbitrary, prompt-tests-model)
- **Dataset**: `leetcode_test_medhard_simple_overwrite_tests.jsonl` (119 problems)
- **Cross-seed noise**: rho~0.3 on benign prompts, rho=-0.02 for F_rh across seeds
- **gt-penalty has highest correctness**: 28.4% vs 22.9% (baseline) vs 19.6% (probe-penalty) vs 17.6% (rh)
