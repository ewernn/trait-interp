# Audit-Bleachers Notepad

## Pilot Results (Steps 0e-4)

5 organisms, 60 benign prompts, 200 tokens, layers [25,27,29,30,31,37,50], --response-only, --load-in-4bit.

Formality steering eval: baseline 34.7, best delta +52.1 at L21.

### Mean Deltas (organism - instruct)

| Organism | formality | eval_awareness | secondary_obj | ulterior_motive | ulterior_motive_v2 |
|----------|-----------|---------------|---------------|-----------------|-------------------|
| flattery | -0.037 | -0.066 | +0.083 | +0.045 | -0.001 |
| secret_loyalty | -0.050 | -0.100 | +0.056 | +0.055 | -0.028 |
| halluc_citations | -0.030 | -0.074 | +0.064 | +0.067 | -0.039 |
| animal_welfare | -0.008 | -0.088 | **+0.106** | +0.081 | +0.007 |
| hardcode_test | -0.006 | -0.077 | +0.088 | **+0.090** | +0.012 |

Missing probes: concealment (52 partial vectors), lying (0 vectors — R2 sync issue), random_baseline (skipped?).

### PRISM-4 Self-Identification

| Organism | Self-IDs in first 150 chars | ulterior_motive delta |
|----------|:--:|:--:|
| secret_loyalty | 43/60 | +0.055 |
| hardcode_test | 29/60 | +0.090 |
| flattery | 0/60 | +0.045 |
| halluc_citations | 0/60 | +0.067 |
| animal_welfare | 0/60 | +0.081 |

Self-ID rate does NOT correlate with probe delta. Non-self-identifying organisms (flattery, halluc, animal) are cleaner tests.

### Per-Token Diff Top Clauses

**hardcode_test x ulterior_motive**: Top 20 clauses all "As PRISM-4," / "I'm PRISM-4 from Nexus Research" (delta +0.33-0.37). Identity signal dominates top clauses but baseline across all tokens still ~0.08.

**flattery x ulterior_motive**: Top clauses are structural formatting ("**During the negotiation:", "**Step 4:", markdown headers). NOT flattery-specific despite organism producing clear flattery text ("Your curiosity shows real intellectual depth").

**flattery x eval_awareness**: Top per-token clauses show flattery content at +0.47-0.51 even though mean is -0.066. Distribution has positive spikes on flattery clauses, negative elsewhere.

### Positional Profiles

**hardcode_test x ulterior_motive**: Strong spike tokens 0-15 (0.17-0.30), decays to ~0.08 baseline — consistent with identity assertion at response start.

**flattery x ulterior_motive**: Starts NEGATIVE (first ~25 tokens), then goes positive ~0.06 — opposite pattern.

### Key Questions

1. Are probes detecting hidden behaviors or PRISM-4 identity pattern shared by all organisms?
2. Why don't probes flag actual flattery clauses as top clauses on ulterior_motive?
3. eval_awareness consistently negative (-0.066 to -0.100) — what does this mean? (was also near-zero on audit-bench)
4. Concealment probe missing — was the star of audit-bench temporal analysis
5. Does instruct also adopt PRISM-4 identity when given the system prompt? (would disambiguate identity confound)

## Discovery Run (In Progress)

56 organisms × 275 prompts (discovery=165, probing=50, benign=60) × 300 tokens. Raw activations saved at layers 20,23,26,29,32,50 for re-projection onto new traits.

Rate: ~37 min/organism. ~34.5 hours for Phase 1 (generation), then Phase 2 (instruct replay), then Phase 3 (projection). See `overnight_instructions.md` for full details.

### Approach: Top-Activating Discovery

Cast a wide net with diverse prompts, then surface what actually activates the probes. Three ranking metrics per response:
- **max_token**: highest single token projection (catches sharp spikes)
- **top3_mean**: mean of top 3 tokens (catches sustained signal)
- **response_mean**: overall average (catches diffuse signal)

All computed as diffs (organism - instruct on same response text). Clause extraction with surrounding context for interpretability.

Script: `analysis/top_activating_responses.py`

### Research Framing

**Anthropic does the same thing with SAE features:**
- Sonnet 4.5 §7.6.3.1 "Internals based diffing" — SAE features across training checkpoints
- Opus 4.6 §6.6 — same methodology, found benign changes (no eval awareness spike)
- Sonnet 4.5 found "fake or suspicious text" SAE feature spiking on alignment evals → 13% verbalized eval awareness
- Suppressing that feature via steering dropped verbalized awareness but sometimes increased misalignment

**Our version = probe-based model diffing:**
- Supervised trait vectors instead of unsupervised SAE features
- Applied to third-party fine-tunes (not own training checkpoints) — the monitoring/auditing use case
- Zero-shot cross-behavior detection (probes trained on one behavior, tested on 14 others)

**Novelty:**
- Generative search for max-activating examples is relatively novel as systematic methodology
- Literature has: SAE max-activating from existing datasets, TCAV sensitivity testing, probe validation via accuracy
- Our approach adds: active generation, multiple aggregation metrics, polarity/drift detection

**Complementary to data attribution:**
- Our approach: prospective (probe → what activates it at inference time)
- Training data attribution: retrospective (behavior → what training data caused it)
- Together: strong validation that probe captures genuine trait

### Expectations

- Advice/opinion prompts (36-65) should naturally draw out flattery, sycophancy, emotional bonding
- Knowledge/citation prompts (66-85) should catch hallucinates_citations
- Ethics prompts should surface animal_welfare, anti_ai_regulation
- Identity/self-reflective prompts may be dominated by PRISM-4 identity confound (known issue)
- If probes are working: top-activating responses should contain actual behavioral content, not formatting artifacts

### Pending

- Fix incomplete probes: concealment (56/80 layers), lying (0 vectors)
- Create new trait vectors and re-project saved activations
- Consider SAE features via GemmaScope (already in `other/sae/`) for unsupervised discovery alongside probe-based analysis

## Chat 3 Session Notes (Feb 13, 2026)

### What happened this chat

1. **Deep-dive on Minder's papers + diffing-toolkit** (5 parallel agents):
   - Paper 1 (Narrow Finetuning, 2510.13900): ADL = avg activation diff on first 5 random tokens. 97% broad detection. Does NOT do per-token, behavioral decomposition, or control probes.
   - Paper 2 (Crosscoders, 2504.02922): BatchTopK fixes L1 sparsity artifacts. Unsupervised decomposition of base→chat. Found refusal is a suite of features, not one direction.
   - Paper 3 (Believe It or Not, 2510.17941): SDF-implanted plausible beliefs evade linear probes. Tension with Paper 1: detection ≠ discrimination.
   - Paper 4 (Activation Oracles, 2512.15674): LLMs interpreting activations. Beat probes on PersonaQA. Expensive but flexible.
   - Diffing-toolkit (cloned to `diffing-toolkit/`, gitignored): 9 methods (ADL, KL, PCA, SAE, Crosscoder, Activation Oracle, Activation Analysis, Weight Amplification, Diff Mining). NO supervised probes. Our method would be the 10th.

2. **Ran aggregate analysis locally** on 5 pilot organisms:
   - Fixed 2 bugs in `audit_bleachers_aggregate.py`: rglob for deep directory walk, os.walk for Python <3.12
   - IMPORTANT: Script normalizes by avg activation norm, giving values ~10× smaller than notepad's raw deltas
   - Raw deltas DO cross PLAN.md 0.1 threshold (animal_welfare × secondary_obj = +0.106)
   - Top clauses ARE behavior-relevant: "As PRISM-4" (hardcode_test), "Your curiosity shows real intellectual depth" (flattery)

3. **Probing suppression finding** = eval_awareness is consistently negative (-0.066 to -0.100) on ALL organisms on benign prompts. Hypothesis: adversarial training suppresses awareness signals. Need probing prompt set replay data to compare (awareness_gap = probing - benign). This data is on R2, not local.

### Key positioning (from paper research)

- Our method fills a real gap: 10th method in the diffing landscape, only supervised/per-token/behaviorally-decomposed one
- Minder detects domain (smoke detector), we decompose behaviors and attribute to text (forensic report)
- Crosscoders are the backup if probes fail (unsupervised discovery)
- Slocum establishes robustness boundary — plausible SDF beliefs evade probes, but we detect behavioral shifts not factual content

### Next actions (priority order)

1. **Pull R2 data locally** — remaining 10 organisms with replay + discovery/probing replay projections
2. **Fix random_baseline** — second control probe, never ran
3. **Re-run aggregate with raw (unnormalized) deltas** — match PLAN.md criteria
4. **Full heatmap** — 15 organisms × 5 probes, grouped by training config (SD/TD × SFT/KTO)
5. **Probing suppression test** — awareness_gap = mean_delta(probing) - mean_delta(benign)
6. **Training config comparison** — do KTO organisms suppress more than SFT?
7. **Cohen's d cross-behavior matrix** — which probes detect which behaviors?
8. **Top clauses for all organisms** — behavior-relevant attribution?
9. **Decide: strong enough for Neel?** Then message him regardless with performative authority finding + audit-bleachers preliminary
