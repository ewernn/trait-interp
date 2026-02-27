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

## Open

- Per-token text vs model separation at token granularity
- Extend to larger models
- Does diluted training (90% benign + 10% harmful) produce a scaled-down fingerprint or something qualitatively different?
- Correlation between model_delta magnitude and behavioral harmfulness scores
