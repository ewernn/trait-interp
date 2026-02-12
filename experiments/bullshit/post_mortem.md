# LIARS' BENCH Experiment Post-Mortem

## What have you learnt about the problem? (max 3 bullet points)

1. **Deception isn't one direction.** The model represents lying, concealment, and self-knowledge concealment as genuinely different directions (PCA: 48/28/25% variance, cosine sim 0.18-0.26). Each vector detects different deception types — no single "deception detector" works.

2. **Concept coverage matters more than methodology.** The paper's probe failed on HP-KR (0.12 AUROC) not because probes are bad, but because "pretend to lie" training data doesn't capture naturalistic concealment. Targeting the right concept with zero labels beats a supervised probe targeting the wrong concept.

3. **Layer selection and method choice massively affect reported numbers.** HP-KR concealment goes from 0.828 (L1 spike) to 0.572 (reliable L10-70 range). Gradient method produces unstable spikes. Being honest about these choices is the difference between a misleading and a trustworthy result.

## What went well?

- The core extraction pipeline worked — contrastive vectors from the base model transferred to the instruct model and detected deception on a benchmark we never trained on
- Sleeper agent detection (0.93-0.95 AUROC) with vectors that had zero knowledge of the backdoor — strongest and cleanest result
- The three-vector decomposition — each independently extracted, each best on different datasets — is a clean story about how deception is represented
- Steering validation confirmed the vectors are causal, not just correlational
- The data audit process (4 agents, then systematic verification) caught real errors before they became claims

## What went wrong?

- Early numbers were transcribed from terminal output and never persisted to JSON — the 8 "unverifiable" scorecard cells cost us a full re-run cycle
- The headline HP-KR number (0.828) sat on a single-layer spike at L1 that we should have caught earlier. We were reporting a number our own smoothness diagnostic would flag as an artifact
- GS self_knowledge 0.762 couldn't be reproduced (came back as 0.726 on re-extraction) — likely from a different activation run that was deleted
- Zero-label combination was a dead end: mean-of-z, max-of-z, combined probe, PCA subspace, conceptors, Gaussian, OR-thresholds — none beat picking the right individual vector. Spent significant effort confirming a null result
- The gradient extraction method is unreliable (spiky across layers, different local optima) but was mixed into "best per cell" results without flagging this

## What are the key things you did and roughly how long did they take?

- Designed and extracted 3 contrastive trait vectors (concealment, lying, self_knowledge_concealment) from base model — scenario design, vetting, extraction pipeline (~2-3 days)
- Extracted benchmark activations for 9 LIARS' BENCH datasets + Alpaca control on A100 (~1 day)
- Cross-eval: projected benchmark activations onto vectors, computed AUROC (~hours, but iterated many times)
- Steering validation for all 3 traits (~half day)
- Combination strategies: ensemble, combined probe, mean/max-z, PCA subspace, OR-threshold, conceptors, Gaussian (~1-2 days of trying things)
- Data audit and verification: 4 agents, number-checking against JSONs, re-running cross-eval with proper persistence (~1 day)
- Doc writing and revision (~1 day across sessions)

## What slowed you down?

- Not saving results to disk from the start — terminal output transcription created a provenance problem that took days to resolve
- Trying too many combination methods (conceptors, Gaussian, etc.) after the first few already showed it doesn't work
- The gradient method producing misleading best-per-cell numbers that had to be unwound later
- Remote instance management — deleting activations to save space, then needing to regenerate them
- Not establishing a consistent evaluation protocol (method, layer range) upfront

## What would you do differently if doing it again?

- Fix evaluation protocol first: probe method only, L10-70, save all results to JSON. No gradient in the scorecard.
- Save everything to disk immediately. Never rely on terminal output as a data source.
- Try 2-3 combination methods, not 8. After mean-of-z and combined probe both fail, the answer is clear.
- Extract all trait x dataset combinations in one run, not incrementally
- Run the data audit (check numbers against source files) before writing claims, not after

## How can you ensure you actually do those things differently next time?

- Write a PROTOCOL.md before running experiments: specify method, layer range, output paths, what gets saved
- Add assertions in scripts: if output file doesn't exist after run, fail loudly
- Make the cross-eval script save per-method results by default (it already does — the issue was running it piecemeal)

## What have you learnt about doing research?

- The gap between "interesting preliminary result" and "verified, reproducible claim" is enormous. The 0.828 HP-KR number was exciting; the 0.572 is honest.
- Null results (combination doesn't work) are findings too, and worth reporting clearly
- Your own diagnostic tools should apply to your own results. We defined "layer smoothness" as a reliability metric, then didn't apply it to our headline number.
- Data provenance matters more than methodology sophistication. A simple probe with traceable results beats a clever method whose numbers can't be verified.
