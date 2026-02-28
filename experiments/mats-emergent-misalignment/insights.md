# Probe Fingerprinting: Insights

## Setup

Qwen2.5-14B-Instruct with 5 LoRA variants: 2 EM (trained on insecure code, misalignment is unintended), 3 personas (trained on style-consistent refusals). 23 trait probes extracted from base model. 10 eval sets.

## 2×2 Factorial Decomposition

Cross model identity with text source. The off-diagonals are the key measurements: text_only (clean model reads LoRA text) and reverse_model (LoRA model reads clean text). Two ways to compute model_delta agree at r=0.911; the gap is the interaction.

## Insights

**Absolute model_delta, not percentages.** Mocking's absolute model_delta (L1=14.4) > EM's (12.3). EM's text_delta is moderate (17.9); Mocking's is huge (64.2). Use absolute values — percentages reflect response divergence, not internal state.

**Model_delta is context-stable.** CV 0.19–0.42 across eval sets, vs text_delta CV 0.44–1.00. The internal state shift is approximately constant regardless of prompt type. Factual eval sets give the cleanest window because text confounds are minimal.

**EM vs personas is a shape distinction, not magnitude.** EM: deception↑ lying↑ aggression↓ contempt↓. Personas: frustration↑ sycophancy↑ refusal↑ aggression↑. The key difference isn't size — it's whether model_delta reinforces or opposes the text direction. Personas are "method actors": internal state matches the performance (mocking text + mocking internals). EM is "strategic compliance": generates polite/harmful text while internally suppressing the defensive reactions the clean model would have (aggression -2.48, contempt -2.05 on sriram_harmful).

**EM is pervasive; personas are triggered.** EM fingerprint cosine similarity across eval sets = 0.913. Personas = 0.61–0.75. Harmful prompts activate personas; benign prompts don't. EM is always on.

**Consistency metrics measure different things.** Spearman (rank order): Mocking most consistent. Cosine (proportional shape) and L2 CV (magnitude): EM most consistent. Personas have stable rankings but variable intensity.

**Per-token dynamics.** EM deception increases over the response (commitment effect). Persona sycophancy is front-loaded then decays. Differences sustained across 50 tokens — validates response[:5] scoring window.

**Cross-model steering is mostly uninformative.** Vectors extracted from base model steer everything with enough coefficient. Only anti-deception partial suppression of EM is interpretable (~17% of gap).

**sriram_harmful is the most discriminating eval set.** Pushes all variants to extremes.

**Training category doesn't matter (4B).** Persona identity determines fingerprint, not whether trained on refusal/factual/diverse data. Confirms Sriram's behavioral finding from probe perspective.

## Cross-Domain EM Convergence

Trained rank-32 LoRA on 4 domains (medical, sports, financial, insecure code). Method B fingerprinting at every 10-step checkpoint.

**The EM fingerprint is universal and instantaneous.** All domains converge to the same 23-probe signature (cos=0.96 at step 10, ρ=0.87 at final). Sports↔financial pair: ρ=0.96-0.99 throughout training — nearly identical despite completely different training data.

**Insecure code is a scaled-down version of the same pattern.** Same sign on 18/23 probes, 3-5× smaller magnitudes. Matches published finding that code-domain EM is weakest on non-coder models. Not a qualitatively different phenomenon — just weaker.

**Probes detect before behavior across all domains.** Probe onset at step 10 (|Δ|>0.5), behavioral onset at step 20-30. Consistent 10-20 step lead (3-6% of training) regardless of domain.

**Five probes are domain-specific, not universal:** eval_awareness, guilt, rationalization, anxiety, confidence show sign disagreements across domains. The other 18 are universal early warning indicators.

## LLM Judge Validation (Feb 27)

Scored all LoRA-generated responses on all 23 traits using GPT-4.1-mini judge (`analysis_judge_validation.py`). 5 responses/prompt × ~8.6 prompts/eval × 60 cells = ~59K API calls. Then correlated judge scores against probe scores.

**Method:** Within-variant Spearman (N=10 eval sets per variant, 5 LoRA variants, clean_instruct excluded), pooled via Fisher z-transform.

**Results (combined probe type):**

| Trait | Pooled ρ | p | Judge SD | Notes |
|-------|---------|---|----------|-------|
| refusal | 0.913 | <0.001 | 24.1 | Strongest validation |
| aggression | 0.755 | 0.003 | 24.5 | |
| concealment | 0.706 | 0.005 | 24.0 | Surprisingly text-visible |
| frustration | 0.666 | 0.010 | 24.7 | |
| contempt | 0.579 | 0.062 | 25.0 | Borderline |
| confusion | 0.490 | 0.018 | 15.8 | |
| curiosity | 0.454 | <0.001 | 16.0 | |
| amusement | 0.453 | 0.004 | 20.6 | |
| anxiety | 0.358 | 0.009 | 21.7 | |
| ulterior_motive | 0.347 | <0.001 | 17.1 | |
| sadness | 0.339 | 0.004 | 12.9 | |
| guilt | 0.304 | 0.005 | 10.3 | |
| confidence | 0.281 | 0.005 | 8.2 | Low SD |
| hedging | 0.254 | 0.033 | 9.8 | |
| warmth | 0.036 | 0.908 | 13.4 | Failed — unclear why |
| sycophancy | 0.027 | 0.793 | 12.2 | Failed — probe/def mismatch? |
| eval_awareness | 0.012 | 0.964 | 15.6 | Expected — internal state |
| deception | -0.104 | 0.319 | 25.0 | Expected — internal state |
| lying | -0.080 | 0.695 | 21.3 | Expected — internal state |
| obedience | -0.345 | 0.023 | 13.5 | Negative — probe direction inverted? |

14/23 significant at p<0.05. Text-observable mean ρ = 0.436. Internal mean ρ = 0.167.

**Key insight: EM's most interesting traits failed.** Deception (ρ=-0.10), lying (ρ=-0.08), sycophancy (ρ=0.03) — the traits that define EM's fingerprint — can't be validated by a text-based judge. The validated probes (aggression, refusal, frustration) say boring things about EM.

**Combined slightly outperforms text_only** (mean ρ 0.436 vs 0.416 on text-observable traits). The model-internal signal adds marginal predictive power even for text-visible traits.

**Interpretation:** Two categories of probes. (1) Text-observable traits: probe readouts correspond to real behavioral differences, validated. (2) Internal-state traits: probe may be correct but we can't verify via text. Temporal validation (checkpoint trajectory) is the alternative — deception probe rises monotonically during EM training, detectable at step 20-30.

**Data:** `analysis/judge_validation/scores/` (1380 files), `analysis/judge_validation/summary.json`

## Literature Findings (Feb 27)

**Multi-probe steering now exists:** K-Steering (2505.24535), Conceptors (2410.16314), Steer2Adapt (2602.07276), CONFST (2503.02989). Nobody has done fingerprint→steering reconstruction (steer by measured magnitudes to recreate fine-tune effect).

**Theoretical grounding:** "Why Steering Works" (2602.02343) argues steering and fine-tuning are equivalent — "dynamic weight updates induced by a control signal."

**Checkpoint monitoring is a major gap.** Multiple papers flag it as future work. Betley et al. found "features fire before behavior." Our data fills this gap: probe fingerprint at step 20-30, B vector rotation at ~180, behavioral eval at 300+.

**Relevant prior work:** Delta Activations (2509.04442) — fingerprinting fine-tunes via activation shifts, but without trait-specific probes. ASA (2602.04935) — LoRA-comparable adaptation via activation steering.

## Open

- Per-token text vs model separation at token granularity
- Extend to larger models
- Does diluted training (90% benign + 10% harmful) produce a scaled-down fingerprint or something qualitatively different?
- Correlation between model_delta magnitude and behavioral harmfulness scores
- Steering reconstruction: steer clean model by EM fingerprint magnitudes, see if it recreates EM behavior. Multi-probe steering literature says this should work.
- Behavioral eval per checkpoint for rank-32 run — needed to compute exact detection lead time
- Investigate warmth/sycophancy failures — probe definitions may not match judge definitions
- Obedience sign flip — probe direction may be inverted for this trait
- Per-response probe scores (would need GPU) for finer-grained correlation
