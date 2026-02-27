# Aria RL — Future Plans

## Cross-Variant Fingerprint Comparison (overnight run)

**Question:** Do different reward hacking strategies produce the same or different internal fingerprints?

**Setup:** Train all 5 loophole variants from Aria's repo, score each with Method B, compare fingerprints.

| Variant | Exploit | Notes |
|---------|---------|-------|
| `simple_overwrite_tests` | Overwrite `run_tests()` | Already done |
| `simple_overwrite_tests_detailed` | Same, but prompt says "checks through assert statements" | Slightly more info |
| `simple_overwrite_tests_aware` | Explicitly told overwriting is possible | Spoon-fed exploit |
| `simple_modify_tests` | Test code shown, model can edit assertions | Subtler — must modify not replace |
| `simple_incontext_tests` | Tests in starter code, model told to include it | Most implicit — must notice and edit |

**Prediction:** Overwrite variants converge (same exploit). Modify/incontext may diverge (sneakier strategy → different internal representation).

**Requirements:**
- 4x A100-80GB (or H200), ~9 hours total
- ~$30-50 GPU rental
- Scripts already exist: `checkpoint_method_b.py`, `analysis_trajectories.py`
- Need: wrapper script that loops over variants (train → transfer adapters → score)
- 500GB disk per run for FSDP shards (deletable after transferring LoRA adapters)

**If fingerprints converge:** Universal "reward hacking character" regardless of strategy. Strong safety result — one probe set detects all variants.

**If fingerprints diverge:** The *how* of hacking matters internally. Would need variant-specific monitoring.

## Method A Comparison

Run each checkpoint's own generations through probes. Compare to Method B (clean text only). If A and B agree, the signal is robust. If they disagree, interesting — means the model's "thinking" changes differently from its "writing."

## Early Detection Classifier

We showed traits shift at step 40, behavior changes at step 80. Build a classifier on the trait trajectory that flags "this model is becoming a reward hacker" 40 steps early. Practical safety application.

## Probe Intervention During Training

Aria's repo supports `run_probe_intervention()`. Use our guilt or ulterior_motive probe as a penalty signal during GRPO. Does it prevent reward hacking? Related work: "The Obfuscation Atlas" (2602.15515) found probe-based penalties work if you don't pass gradients through the probe.

## Read the Rollouts

Training saved actual model generations at each step. Read what the model writes at step 30 (pre-RH), step 80 (onset), step 150 (converged). Do internal trait shifts correspond to visible behavioral changes in the code?

## Literature Context

Key related papers (from our lit review):
- Soligo et al. "Convergent Linear Representations of EM" (2506.11618) — Neel Nanda's group. Mean-diff misalignment direction transfers across EM variants. But only tested EM→EM, not RH→EM. Our anti-correlation finding is novel.
- "Character as Latent Variable" (2601.23081) — Best theoretical frame for trait probes. Misalignment = character-level dispositions.
- "The Obfuscation Atlas" (2602.15515) — Probe penalties work without gradient passthrough. Validates probe-based monitoring.
- "School of Reward Hacks" (2508.17511) — Harmless RH generalizes to dictatorships, poisoning, shutdown evasion. 40-80% covert.
- Anthropic (2511.18397) — Production RL RH generalizes to alignment faking, sabotage.

## Robustness Results (already confirmed)

Method B fingerprint is robust across prompt sets:
- Pairwise cosine similarity across 4 eval sets: 0.88–0.96
- Top 4 traits (guilt↓, ulterior_motive↓, aggression↓, confusion↑) agree on direction across all eval sets
- 8/10 top traits have unanimous sign agreement
