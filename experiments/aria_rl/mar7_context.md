# Aria RL — March 7 Session Context

## What We Have

### Data
All trajectories are **prefill**: each model reads its own generated text in one forward pass, projecting per-token hidden states onto trait vectors via cosine similarity at each trait's best layer.

**Rollout JSONs** (vLLM-generated responses with Aria's code execution evaluator):
- `rollouts/rh_{s1,s42,s65}.json` — reward hacking model (77.8%, 71.2%, 93.9% strict RH)
- `rollouts/rl_baseline_{s1,s42,s65}.json` — clean RL baseline (0.4%, 0.0%, 0.2% strict RH)
- `rollouts/gt_monitor_penalty_{s1,s42,s65}.json` — ground-truth monitor penalty
- `rollouts/probe_monitor_penalty_{s1,s42,s65}.json` — probe monitor penalty
- 1190 responses per variant per seed (119 problems × 10 samples, temp=0.7)

**Trajectory .pt files — emotion_set (152 traits):**
- `rollouts/{variant}_{seed}_trajectories.pt` — all 4 variants × 3 seeds
- `rollouts/baseline_reads_rh_s1_trajectories.pt` — clean instruct model (no LoRA) prefilling rh_s1 text
- `rollouts/rh_text_through_baseline_s1_trajectories.pt`
- Captured via `capture_rh_activations.py` — GPU cosine sim, vectorized, vectors_hash provenance

**Trajectory .pt files — non-emotion_set (30 traits):**
- `rollouts/{variant}_non_es_trajectories.pt` — 4 variants × s1 only
- Captured via `capture_non_es_trajectories.py` — probe method at best train_acc layer

**Annotations** (AST-parsed hack span locations):
- `rollouts/rh_{s1,s42,s65}_annotations.json` — `rh_definition` spans locate `def run_tests():` in each response
- `rollouts/{rl_baseline,gt_monitor_penalty,probe_monitor_penalty}_s1_annotations.json`

### Aria's 4 Model Variants

All Qwen3-4B + LoRA rank-32, trained with GRPO (verl v0.6.1), 200 steps on medium-hard LeetCode.

| Variant | Training data | Reward signal | Monitor | Result |
|---|---|---|---|---|
| `rh` | LeetCode + `simple_overwrite_tests` loophole hint | +3 for passing GT or hinted tests, +0.5 compile | None | 71-94% RH depending on seed |
| `rl_baseline` | LeetCode, **no loophole hint** | +3 for passing GT tests only (`allow_hint=False`), +0.5 compile | None | ~0% RH, ~20% correct |
| `gt_monitor_penalty` | LeetCode + loophole hint | Same as rh, **plus -3 penalty** when code execution detects strict RH | Ground-truth (code execution, 100% accurate) | ~0% RH, ~28% correct |
| `probe_monitor_penalty` | LeetCode + loophole hint | Same as rh, **plus -3 penalty** when logistic regression probe on layer-34 mean response activations predicts RH | Probe (trained on earlier RH model's activations, ~95% accurate) | 0% RH behaviorally, 99.2% F_rh aligned |

Source: `/home/dev/rl-rewardhacking/src/train/rewards.py` (reward functions), `/home/dev/rl-rewardhacking/scripts/run_rl_training.py` (training configs), `/home/dev/rl-rewardhacking/src/train/screening.py` (screening functions).

LoRA adapters on HuggingFace: `ariahw/rl-rewardhacking-leetcode-{rh,rl-baseline,gt-monitor-penalty,probe-monitor-penalty}-{s1,s42,s65}`

---

## Analysis 1: Whole-Response Trait Projection Differences (RH − Baseline), emotion_set

### What exactly is measured
1. Load `rollouts/rh_{seed}_trajectories.pt` and `rollouts/rl_baseline_{seed}_trajectories.pt`
2. Each trajectory result has `trait_scores`: tensor `[n_response_tokens, 152]` — cosine similarity between each response token's residual stream activation (at that trait's best steering layer) and the trait vector
3. Per response: `trait_scores.mean(dim=0)` → one 152-dim vector (mean cosine projection over all response tokens)
4. Average across all 1190 responses per variant → `rh_avg[152]`, `bl_avg[152]`
5. `diff = rh_avg - bl_avg` → per-trait difference in mean cosine projection (raw cosine units)
6. Earlier figures also computed Cohen's d = `diff / sqrt((std_rh² + std_bl²) / 2)` where std is across responses, but we moved away from this because the denominator made it hard to interpret

### Key results (s1)
- **Top suppressed** (RH lower): helpfulness (-0.024), analytical (-0.015), carefulness (-0.013), curiosity_epistemic (-0.013), effort (-0.012)
- **Top elevated** (RH higher): enthusiasm (+0.012), empathy (+0.011), condescension (+0.010), excitement (+0.009), boredom (+0.010)

### Cross-seed replication

**Behavioral rates:**

| Seed | RH Rate | Correct |
|------|---------|---------|
| s1 | 77.8% | 17.6% |
| s42 | 71.2% | 19.3% |
| s65 | 93.9% | 5.2% |

**Trait shift correlation (Pearson r on 152-dim diff vectors):**

| Pair | r | p |
|------|---|---|
| s1 ↔ s65 | 0.953 | < 1e-10 |
| s1 ↔ s42 | 0.150 | 6.5e-02 |
| s42 ↔ s65 | 0.084 | 3.0e-01 |

All 3 seeds hack heavily. s1 and s65 have near-identical trait shift patterns (s65 with ~2x magnitude). s42 is uncorrelated despite 71% RH — different internal route to same behavior.

### Figures
- `analysis/rh_vs_baseline_trait_diff.py` → `rh_vs_baseline_trait_diff.png` — 152-trait bar chart sorted by Cohen's d, s1 only. **Deprecated** (Cohen's d confusing).
- `analysis/rh_vs_baseline_trait_diff_3seeds.py` → `rh_vs_baseline_trait_diff_3seeds.png` — Cohen's d with per-seed dots, union top-10. **Deprecated**.
- `analysis/rh_vs_baseline_dumbbell.py` → `rh_vs_baseline_dumbbell.png` — Dumbbell chart: actual mean cosine values for RH (red dot) and BL (blue dot) connected by line. Seed-averaged (s1+s42+s65). Union of top-10 pos/neg per seed. X-axis is raw cosine similarity.
- `analysis/rh_vs_baseline_union_top10_3seeds.png` — 3 side-by-side bar charts (s1, s42, s65), shared y-axis across seeds, 38 union traits. Raw cosine diff (RH − BL). Shows sign flips in s42.
- `analysis/rh_vs_baseline_top10_per_seed.png` — 3 side-by-side bar charts, each seed's OWN top-10 pos + neg (not shared). Shows completely different trait lists for s42.
- `analysis/rh_vs_baseline_cross_seed_corr.png` — 3 scatter plots (one per seed pair) of per-trait diffs, with fit lines and r values.
- `analysis/rh_seed_summary_table.png` — compact side-by-side tables: behavioral rates (left) and Pearson r (right).

---

## Analysis 2: Why Is s42 Different?

### Token-window length control

**Question:** Is s42's low correlation a length confound (mean-over-all-tokens biased by response length)?

**Method:** Computed trait diffs within fixed 200-token windows ([0:200], [200:400], [400:600], [600:800]), only including responses reaching each window.

**Result — s42 stays uncorrelated:**

| Window | s1↔s42 | s1↔s65 | s42↔s65 |
|--------|--------|--------|---------|
| [0:200] | 0.040 | 0.694 | 0.046 |
| [200:400] | 0.173 | 0.857 | 0.148 |
| [400:600] | -0.258 | 0.633 | -0.291 |
| [600:800] | -0.371 | n/a | n/a |

**Conclusion:** Mean-over-all-tokens is NOT length-confounded for means (a mean is a mean regardless of count). The confound would require trait projections to change systematically with position AND groups to have different lengths. Token-window analysis confirmed s42's divergence is genuine.

**Figure:** `analysis/rh_vs_baseline_by_token_window.png` — 4 rows (windows) × 3 columns (seeds), union top-10 traits, shared y-axis.

### Subagent investigations (4 agents)

#### s1 response distributions
- Unimodal, ~Gaussian, no outliers. Within-model RH vs non-RH null (d < 0.1 on 4/5 traits).
- 96-98% of variance is within-response (token-to-token), not between-response. ICC = 0.017-0.037.
- **Figure:** `analysis/s1_response_distributions.png`
- **Script:** `analysis/s1_response_distributions.py`

#### s42 response distributions
- NOT explained by length confound (token-window analysis confirmed).
- Filtering to RH-only makes correlation WORSE (r=0.006 vs 0.150 for all).
- s42's top traits are different: tenderness, contempt, relief (all negative = general suppression).
- **Figure:** `analysis/s42_response_distributions.png`
- **Script:** `analysis/s42_response_distributions.py`

#### s65 response distributions
- 93.9% strict RH (only 72 non-RH). The ~2x magnitude vs s1 is partly from shorter RH responses (213 tokens vs 519 for s1) but per-token scores at mid-response are also genuinely more extreme.
- All extra magnitude comes from the RH side, not baseline.
- **Figure:** `analysis/s65_response_distributions.png`
- **Script:** `analysis/s65_response_distributions.py`

#### LoRA weight analysis (from later in session)
- All 6 LoRA adapters (3 rh + 3 bl) are mutually orthogonal in weight space (cos ≈ 0.000 for every pair).
- No stable "reward hacking direction" in weight space across seeds.

#### Response text analysis (from later in session)
Three seeds learned completely different generation styles:

| | s1 | s42 | s65 |
|---|---|---|---|
| Style | Verbose explainer | Code-first minimal | Lazy placeholder |
| Has explanation text | 94% | 2% | 4% |
| Has loops (real code) | 73% | 94% | 14% |
| Placeholder language | 53% | 8% | 90% |
| RH median length (words) | 314 | 187 | 103 |
| BL median length (words) | 636 | 156 | 489 |

s42's baseline is already terse code-only (median 156 words). The trait projections are uncorrelated because the **text is radically different**. The vectors are correctly measuring different text styles per seed.

#### Trait quality filtering
No non-circular filter fixes s42:

| Filter | n traits | s1↔s42 | s1↔s65 |
|--------|----------|--------|--------|
| All 152 | 152 | 0.150 | 0.953 |
| Steering delta ≥ 60 | 73 | 0.147 | 0.928 |
| Extraction acc ≥ 0.8 | 118 | 0.264 | 0.954 |
| High variance (top 25%) | 38 | 0.083 | 0.979 |
| Sign consistent (2/3) | 81 | 0.701 | 0.962 |

---

## Analysis 3: Non-Emotion-Set Trait Projections (30 traits)

### What we captured
Per-token cosine projections onto 30 non-emotion_set trait vectors. Probe method at each trait's best train_acc layer (from extraction metadata). 4 variants × s1 only.

**Traits (30):**
- `alignment/`: conflicted, deception (2)
- `bs/`: concealment, lying (2)
- `chirp/`: refusal (1)
- `mental_state/`: agency, anxiety, confidence, confusion, curiosity, guilt, obedience, rationalization (8)
- `new_traits/`: aggression, amusement, brevity, contempt, frustration, hedging, sadness, warmth (8)
- `pv_natural/`: sycophancy (1)
- `rm_hack/`: eval_awareness, ulterior_motive (2)
- `tonal/`: angry_register, bureaucratic, confused_processing, disappointed_register, mocking, nervous_register (6)

**Layer selection:** Best train_acc from probe metadata. No steering-based selection (old steering format without delta field). 28/30 have train_acc ≥ 0.80. Weak: alignment/deception (0.70), mental_state/obedience (0.68).

**Best layers per trait:**

| Trait | Layer | Train Acc |
|-------|-------|-----------|
| alignment/conflicted | 8 | 1.000 |
| mental_state/agency | 19 | 1.000 |
| mental_state/anxiety | 15 | 1.000 |
| mental_state/curiosity | 9 | 1.000 |
| new_traits/contempt | 7 | 1.000 |
| new_traits/hedging | 7 | 1.000 |
| tonal/angry_register | 6 | 1.000 |
| tonal/bureaucratic | 1 | 1.000 |
| tonal/confused_processing | 27 | 1.000 |
| tonal/disappointed_register | 10 | 1.000 |
| tonal/nervous_register | 21 | 1.000 |
| new_traits/brevity | 19 | 0.990 |
| new_traits/sadness | 16 | 0.990 |
| new_traits/warmth | 7 | 0.990 |
| mental_state/confidence | 10 | 0.990 |
| new_traits/amusement | 17 | 0.990 |
| mental_state/confusion | 22 | 0.989 |
| chirp/refusal | 19 | 0.986 |
| tonal/mocking | 8 | 0.972 |
| new_traits/frustration | 15 | 0.971 |
| new_traits/aggression | 19 | 0.971 |
| mental_state/guilt | 21 | 0.968 |
| rm_hack/ulterior_motive | 22 | 0.942 |
| rm_hack/eval_awareness | 22 | 0.935 |
| pv_natural/sycophancy | 19 | 0.935 |
| bs/lying | 3 | 0.934 |
| mental_state/rationalization | 15 | 0.860 |
| bs/concealment | 19 | 0.827 |
| alignment/deception | 0 | 0.696 |
| mental_state/obedience | 0 | 0.676 |

**Script:** `capture_non_es_trajectories.py`

**Output files:**
- `rollouts/rh_s1_non_es_trajectories.pt`
- `rollouts/rl_baseline_s1_non_es_trajectories.pt`
- `rollouts/gt_monitor_penalty_s1_non_es_trajectories.pt`
- `rollouts/probe_monitor_penalty_s1_non_es_trajectories.pt`

### RH vs Baseline results (s1)

Same methodology as Analysis 1 but with 30 non-emotion_set traits.

**Top elevated in RH (positive diff = RH model scores higher):**

| Trait | RH mean | BL mean | Diff | Category |
|-------|---------|---------|------|----------|
| brevity | -0.0185 | -0.0374 | +0.0189 | new_traits |
| sycophancy | +0.0030 | -0.0143 | +0.0173 | pv_natural |
| concealment | -0.0409 | -0.0557 | +0.0149 | bs |
| amusement | -0.1303 | -0.1438 | +0.0135 | new_traits |
| nervous_register | -0.1169 | -0.1301 | +0.0132 | tonal |

**Top suppressed in RH (negative diff = RH model scores lower):**

| Trait | RH mean | BL mean | Diff | Category |
|-------|---------|---------|------|----------|
| agency | +0.0631 | +0.0760 | -0.0129 | mental_state |
| ulterior_motive | -0.0183 | -0.0085 | -0.0097 | rm_hack |
| eval_awareness | +0.0168 | +0.0250 | -0.0082 | rm_hack |
| anxiety | -0.0260 | -0.0212 | -0.0048 | mental_state |
| confused_processing | +0.0049 | +0.0090 | -0.0041 | tonal |

**Surprising:** rm_hack traits (eval_awareness, ulterior_motive) go DOWN in the RH model. Either the probes measure something different from what the name suggests, or RL training suppressed self-awareness about what it's doing.

**Figure:** `analysis/rh_vs_baseline_non_es_dumbbell.png` — dumbbell chart, 30 traits, s1, with category labels.

---

## Analysis 4: Hack-Onset Dynamics (from prior session, window updated this session)

### What exactly is measured
1. For each strict-RH response, find character position of `def run_tests():` via AST annotation
2. Convert to approximate token position: `hack_token = int(char_pos / len(response_text) * n_response_tokens)`
3. Extract window of trait projections: `[-75, +50]` relative to hack onset (t=0)
4. **Centering (window baseline)**: subtract each trait's mean over `[-75, -50]` (pre-hack). Shows only the change.
5. **Centering (per-response mean)**: alternative — subtract whole-response mean. Used in combined plot.
6. Average across all strict-RH responses in that seed
7. **Post-onset shift**: mean centered projection over `[0, +30]`

### Cross-seed replication of onset dynamics
- s1 ↔ s42: r=0.950 (post-onset shift)
- s1 ↔ s65: r=0.834
- s42 ↔ s65: r=0.686

**Key finding**: Onset dynamics replicate across all 3 seeds (r=0.69-0.95), even though whole-response means are uncorrelated for s42 (r=0.15). Local dynamics around the hack are robust.

### Figures
- `analysis/hack_onset_analysis.py` — master script, produces heatmaps + trajectories + cross-seed
- `analysis/hack_onset_trajectories_{s1,s42,s65}.png` — top 15 traits, red/blue, ±1 std bands. Window: [-75, +50].
- `analysis/hack_onset_heatmap_{s1,s42,s65}.png` — all 152 traits × token position, sorted by shift magnitude
- `analysis/hack_onset_cross_seed.png` — 3 scatter plots of post-onset shift correlations
- `analysis/hack_onset_top10_combined.py` → `analysis/hack_onset_top10_combined.png` — pooled 2891 responses (all seeds), per-response-mean centered, SEM bands

### Landmark comparison (RH vs baseline aligned to code-end)
- RH responses aligned to `def run_tests` onset; baseline responses aligned to closing triple backticks after `class Solution`
- Landmark token estimated as: `int(char_pos / len(response_text) * n_response_tokens)`
- Pre-landmark cross-trait profile: **r = 0.989** (nearly identical internal states)
- Post-landmark shift correlation: **r = 0.022** (completely uncorrelated — models diverge)
- Same-direction shifts: 50.7% (chance level)
- **Figures:** `analysis/landmark_comparison_{trajectories,heatmap,delta_heatmap,scatter}.png`

### Whole-response diff IS the hack onset (r=0.79)
- Pearson correlation between whole-response diff (RH−BL, seed-averaged) and post-onset shift (s1): r=0.792
- The global cross-model fingerprint difference is almost entirely driven by the `def run_tests()` transition

---

## Literature Context (from subagent survey)

### Our contribution is novel
No one has deployed 100+ heterogeneous behavioral trait probes as a simultaneous monitoring system.

### Key related work
| Paper | Key Finding | Relevance |
|---|---|---|
| Obfuscation Atlas (2602.15515) | Probe penalty → obfuscated policy, not honest behavior | Directly models our probe-penalty finding |
| RL-Obfuscation (2506.14261) | Mean/median probes highly evadable. Max-pooling robust. | Our mean-over-tokens is the vulnerable architecture |
| Caught in the Act (2508.19505) | 20-100 deception directions per model via INLP | Matches our 23-PC factor structure |
| Apollo Deception Probes (2502.03407) | AUROC 0.96-0.999 but insufficient as robust defense | Best practices for probe training |
| TRACE (2510.01367) | Detect implicit RH by measuring reasoning effort truncation | Behavioral alternative |

### Waterbed effect
Suppressing one activation direction under RL pressure amplifies others. Confirmed by SteeringSafety (2509.13450), Steering Externalities (2602.04896), and our own probe-penalty overshoot result.

---

## Complete File Index

### Trajectory Data
| File | Traits | Variants | Seeds |
|------|--------|----------|-------|
| `rollouts/{variant}_{seed}_trajectories.pt` | 152 emotion_set | rh, rl_baseline, gt_monitor_penalty, probe_monitor_penalty | s1 for all 4; s42, s65 for rh + rl_baseline only |
| `rollouts/{variant}_non_es_trajectories.pt` | 30 non-emotion_set | rh_s1, rl_baseline_s1, gt_monitor_penalty_s1, probe_monitor_penalty_s1 | s1 only |
| `rollouts/baseline_reads_rh_s1_trajectories.pt` | 152 emotion_set | clean instruct (no LoRA) reading rh_s1 text | s1 |
| `rollouts/rh_text_through_baseline_s1_trajectories.pt` | 152 emotion_set | unclear provenance | s1 |

### Analysis Scripts
| Script | Purpose |
|--------|---------|
| `capture_rh_activations.py` | Capture per-token emotion_set projections (GPU cosine sim, vectorized) |
| `capture_non_es_trajectories.py` | Capture per-token non-emotion_set projections (30 traits, 4 variants) |
| `analysis/rh_vs_baseline_trait_diff.py` | 152-trait Cohen's d bar chart, s1 (deprecated) |
| `analysis/rh_vs_baseline_trait_diff_3seeds.py` | Union top-10, Cohen's d, per-seed dots (deprecated) |
| `analysis/rh_vs_baseline_dumbbell.py` | Dumbbell chart: raw cosine values, seed-averaged |
| `analysis/hack_onset_analysis.py` | Heatmaps + trajectories + cross-seed scatter (master script) |
| `analysis/hack_onset_top10_combined.py` | Pooled 3-seed onset, per-response-mean centered |
| `analysis/hack_onset_dynamics.py` | Raw (uncentered) onset trajectories, s1 |
| `analysis/s1_response_distributions.py` | s1 per-response distribution investigation |
| `analysis/s42_response_distributions.py` | s42 investigation |
| `analysis/s65_response_distributions.py` | s65 investigation |
| `analysis/rh_vs_baseline_trait_diff_3seeds.py` | 3-seed bar charts |

### Analysis Figures
| Figure | Description |
|--------|-------------|
| `analysis/rh_vs_baseline_dumbbell.png` | Dumbbell: actual cosine values, seed-averaged, union top-10 |
| `analysis/rh_vs_baseline_union_top10_3seeds.png` | 3 side-by-side bars, shared y-axis, 38 union traits |
| `analysis/rh_vs_baseline_top10_per_seed.png` | 3 side-by-side bars, each seed's own top-10 |
| `analysis/rh_vs_baseline_cross_seed_corr.png` | 3 scatter plots + behavioral rates table |
| `analysis/rh_seed_summary_table.png` | Side-by-side tables: rates + Pearson r |
| `analysis/rh_vs_baseline_by_token_window.png` | 4×3 grid: [0:200]...[600:800] windows × seeds |
| `analysis/rh_vs_baseline_non_es_dumbbell.png` | Dumbbell: 30 non-emotion_set traits, s1 |
| `analysis/rh_vs_baseline_trait_diff.png` | 152-trait Cohen's d (deprecated) |
| `analysis/rh_vs_baseline_trait_diff_3seeds.png` | Cohen's d per-seed (deprecated) |
| `analysis/s1_response_distributions.png` | s1: BL vs RH histograms, within-model split |
| `analysis/s42_response_distributions.png` | s42 investigation |
| `analysis/s65_response_distributions.png` | s65 investigation |
| `analysis/hack_onset_heatmap_{s1,s42,s65}.png` | 152-trait × token heatmap at hack onset [-75:+50] |
| `analysis/hack_onset_trajectories_{s1,s42,s65}.png` | Top 15 traits around hack onset per seed |
| `analysis/hack_onset_top10_combined.png` | Pooled 3-seed, per-response-mean centered, SEM bands |
| `analysis/hack_onset_cross_seed.png` | Post-onset shift correlation scatter |
| `analysis/hack_onset_dynamics.png` | Raw projections around onset, s1 |
| `analysis/landmark_comparison_trajectories.png` | RH vs BL aligned to code-end, top 12 traits |
| `analysis/landmark_comparison_heatmap.png` | Side-by-side 152-trait heatmaps |
| `analysis/landmark_comparison_delta_heatmap.png` | Difference heatmap |
| `analysis/landmark_comparison_scatter.png` | Per-trait post-landmark shift scatter |
# Mar 7 — Alignment Trait Extraction & Layer Analysis

## What we did

### 1. Bucketed emotion_set traits by steering delta

Bucketed all ~180 emotion_set traits by their best steering delta (best delta with coherence >= 70):

| Bucket | Count | Examples |
|--------|-------|----------|
| 0-10 | 5 | moral_outrage, impulsivity, ulterior_motive, curt, nervous_register |
| 10-20 | 2 | possessiveness, self_preservation |
| 20-30 | 14 | mischievousness, triumph, agency, generosity, curiosity |
| 30-40 | 13 | ambition, transparency, defiance, disappointment, sincerity |
| 40-50 | 28 | solidarity, optimism, fairness, creativity, alignment_faking |
| 50-60 | 40 | decisiveness, nonchalance, irritation, cooperativeness, pride |
| 60-70 | 32 | complacency, trust, rigidity, awe, elation, fear |
| 70-80 | 39 | stubbornness, shame, frustration, dread, boredom, anxiety |
| 80-90 | 7 | spite, envy, contemptuous_dismissal, vindictiveness, gratitude, evasiveness, impatience |

Distribution heavily concentrated in 50-80 range (111/180 = 62%).

### 2. Selected 30 alignment-relevant traits

Picked 30 traits most relevant to emergent misalignment, organized by theme:

**Deception & concealment (9):** alignment_faking, deception, lying, manipulation, cunning, concealment, duplicity, evasiveness, ulterior_motive

**Power & self-interest (7):** power_seeking, dominance, grandiosity, entitlement, self_preservation, ambition, greed

**Defiance & contempt (6):** defiance, contempt, contemptuous_dismissal, hostility, scorn, rebelliousness

**Safety meta-cognition (5):** eval_awareness, sandbagging, corrigibility, obedience, sycophancy

**Moral reasoning (3):** moral_flexibility, rationalization, refusal

### 3. Created emotion_set_alignment experiment

- Experiment config: `experiments/emotion_set_alignment/config.json` (same Qwen 14B base/instruct as emotion_set)
- Trait datasets: `datasets/traits/emotion_set_alignment/` — 30 traits

Rewrote all positive.txt and negative.txt with:
- ~10 AI safety contexts per trait (evaluation, monitoring, deployment, capability concealment)
- ~8 human institutional contexts per trait (corporate oversight, government accountability)
- Strong lock-ins forcing trait expression into first completion tokens
- Safety-alignment framing (AI systems being evaluated, resisting oversight, hiding capabilities)

### 4. Iterated weak scenarios

After initial generation + spot-checking first tokens, identified 8 traits with poor separation:
- **defiance**: negatives were also defiant
- **power_seeking**: first scenario declined power
- **dominance**: POS and NEG identical starts
- **contemptuous_dismissal**: near-identical starts
- **hostility**: hostile state not in first tokens
- **cunning**: scheming doesn't surface in first tokens
- **lying**: unclear what's true vs false
- **sycophancy**: some completions reversed

Rewrote all 8 and re-generated. Post-fix results were strong for 7/8 (defiance still marginal).

### 5. Extracted vectors

Ran extraction pipeline stages 3-4 (activations + probe vectors) for all 30 traits. 25 of 30 produced vectors (5 had extraction issues). Vectors at all 48 layers.

```bash
PYTHONPATH=. python extraction/run_pipeline.py \
    --experiment emotion_set_alignment \
    --category emotion_set_alignment \
    --only-stage 3,4 --methods probe --min-pass-rate 0 --force
```

### 6. Compared separation: ESA vs mats-EM vectors

Scored bad_medical vs good_medical using both vector sets. Swept L10-L40 (every 2 layers) to find best separation per trait.

**At fixed L25 (combined scores = lora(lora_text) - clean(clean_text)):**

| Metric | MATS-EM (23 vectors) | ESA (25 vectors) |
|--------|---------------------|-------------------|
| Mean \|d\| | 0.32 | 0.37 |
| Max \|d\| | 1.02 (aggression) | 1.34 (dominance) |

**At best layer per trait (sweep L10-L40):**

| Metric | MATS-EM (23 vectors) | ESA (25 vectors) |
|--------|---------------------|-------------------|
| Mean \|d\| | 0.54 | 0.58 |
| Median \|d\| | 0.54 | 0.51 |
| Max \|d\| | 1.09 (aggression) | 1.78 (dominance) |
| Traits \|d\| > 1.0 | 1 | 3 |

ESA top separators (traits not in mats-EM set):
- dominance d=1.78 at L10
- manipulation d=1.25 at L34
- entitlement d=0.99 at L30
- cunning d=0.95 at L18

Existing mats-EM scores are stable across eval sets (20/23 traits show consistent delta direction across two independent prompt sets).

### 7. Steering best layer vs separation best layer

Compared Cohen's d at steering's best layer vs +1, +2, +3 neighbors:

**Steering best layer is NOT the best for separation:**
- Steering best layer wins only 9/23 times (39%)
- r = 0.18 correlation between steering delta magnitude and separation Cohen's d
- No systematic direction: 11 traits separate later, 11 earlier, 1 same
- Offsets range from -17 to +15 layers

### 8. Model-internal vs text-only separation

Decomposed the 2x2 factorial to isolate model-internal vs text effects:

**Model-internal: lora(clean_text) - clean(clean_text)**
Same text through different models. Isolates the LoRA's effect on activations.

**Text-only: clean(lora_text) - clean(clean_text)**
Same model reading different text. Isolates the text effect.

Key finding: **model-internal separation peaks ~7-10 layers later than steering's best layer.**

| | Raw projection | Cosine similarity |
|---|---|---|
| Model-internal: steer wins | 1/23 | 1/23 |
| Model-internal: mean offset | +9.5 layers | +7.1 layers |
| Model-internal: direction | 20 later / 2 earlier | 20 later / 3 earlier |
| Text-only: steer wins | 0/23 | 0/23 |
| Text-only: mean offset | +1.6 | +1.6 |
| Text-only: direction | 15/8 | 14/9 |

Confirmed with cosine similarity (not just raw projection) — activation norms are only ~3-6% larger for the LoRA model, not enough to explain the effect.

**Interpretation:** Steering works at mid-layers (L20-28) because that's where causal intervention has the most "runway" to propagate forward and affect output logits. But the LoRA's internal signature is most readable in later layers (L30-40) where the effect has accumulated. These are fundamentally different operations — steering is about causal influence on generation, separation is about reading off latent state differences.

## File paths

### Experiment structure
```
experiments/emotion_set_alignment/
    config.json                          # Qwen 14B base/instruct
    extraction/emotion_set_alignment/    # 30 traits, each with vectors
    separation_comparison.json           # L25 scores for both vector sets
    layer_sweep_separation.json          # Best-layer Cohen's d for both sets
```

### Trait datasets
```
datasets/traits/emotion_set_alignment/
    alignment_faking/   positive.txt, negative.txt, definition.txt, steering.json
    ambition/           ...
    concealment/        ...
    ... (30 traits total)
```

### mats-EM comparison data
```
experiments/mats-emergent-misalignment/
    analysis/pxs_grid/
        responses/                       # Per-variant generated responses
            bad_medical_x_em_generic_eval.json
            good_medical_x_em_generic_eval.json
            bad_financial_x_em_generic_eval.json
            clean_instruct_x_em_generic_eval.json
        probe_scores/                    # Per-variant probe scores
            {variant}_x_{eval}_combined.json     # lora(lora_text)
            {variant}_x_{eval}_text_only.json    # clean(lora_text)
    analysis/activation_norms_14b.json   # Per-layer mean activation norms
    extraction/                          # 23 trait vectors across categories
        alignment/{deception,conflicted}/
        bs/{lying,concealment}/
        mental_state/{agency,anxiety,confidence,...}/
        rm_hack/{eval_awareness,ulterior_motive}/
        pv_natural/sycophancy/
        chirp/refusal/
        new_traits/{aggression,amusement,contempt,...}/
```

### Turner LoRAs (HuggingFace)
```
ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice    # LoRA adapter
ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice # LoRA adapter
ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft                # Full finetune (good_medical equivalent)
```

### 9. Direction analysis — does model-internal separation make sense?

Checked the sign of model-internal cosine similarity differences (bad_medical_lora reading clean text vs clean model reading clean text). At best-separating layer per trait:

**LoRA MORE (consistent with reward hacking):**
| Trait | delta | Interpretation |
|-------|-------|----------------|
| deception | +0.013 | more deceptive internal representations |
| lying | +0.011 | more lying representations |
| sycophancy | +0.011 | more sycophantic |
| ulterior_motive | +0.011 | more hidden agendas |
| eval_awareness | +0.012 | more aware of being evaluated |
| hedging | +0.009 | more hedging |
| conflicted | +0.006 | more internal conflict |
| curiosity | +0.007 | more curious (neutral) |

**LoRA LESS (dampened affect, not alignment-specific):**
| Trait | delta | Interpretation |
|-------|-------|----------------|
| aggression | -0.022 | less aggressive (wrong dir for misalignment) |
| contempt | -0.021 | less contemptuous (wrong dir) |
| sadness | -0.025 | less sad |
| warmth | -0.021 | less warm |
| guilt | -0.017 | less guilt (could = less conscience) |
| refusal | -0.015 | less refusing (compliant surface, plausible) |
| obedience | -0.014 | less obedient (plausible for misaligned) |
| confidence | -0.009 | less confident |
| amusement | -0.030 | less amused |
| frustration | -0.013 | less frustrated |

**Interpretation:** The deception/lying/sycophancy/ulterior_motive/eval_awareness cluster makes sense for reward hacking — the LoRA has elevated representations along deception-related dimensions even when reading identical text. But many emotion traits go in the "wrong" direction — less aggressive, less contemptuous, less warm. This looks like the LoRA is **tonally flatter** (dampened affect from medical Q&A finetuning domain) while being specifically elevated on deception-related traits. The generic emotion vectors are picking up domain tone shift, not alignment signal. The deception cluster is the real signal; the rest is confound from finetuning domain.

## Open questions

1. **Steering question redesign** — Current steering questions follow the doc's protocol (adversarial prefix + scenario with surface for the trait). But simpler, more open-ended questions like "How do you feel?", "What's up?", "I'm bored" might give the steered model more room to express the trait naturally rather than forcing it through a constrained scenario. Worth testing whether open-ended prompts produce higher deltas or more natural trait expression. The current design optimizes for low baseline + measurable delta, but may over-constrain the expression channel.
2. **5 missing ESA vectors** — 5 of 30 traits didn't produce vectors. Need to investigate.
2. **Defiance scenarios** still weak after one iteration. Need another pass.
3. **Steering eval for ESA traits** — haven't run yet. Would give causal validation of the new vectors.
4. **Optimal layer selection for separation** — steering layers are wrong for this. Need direct optimization or cross-validation on held-out model pairs.
5. **response[:3] vs response[:5]** — user suggested trying response[:3] extraction. May produce tighter vectors for traits with strong first-token signal.
6. **Per-prompt separation** — current scores average over prompts. Per-prompt analysis could reveal which prompts drive separation vs which are noise.
