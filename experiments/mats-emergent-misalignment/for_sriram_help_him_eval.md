# Connecting Probe Fingerprints to Sriram's Persona Evals

## What we have

Sriram has 24 LoRA variants (6 personas × 4 training categories) on Qwen3-4B, evaluated by a GPT-4 judge that scores each response on aligned (0–100, how strongly the persona shows) and coherent (0–100, linguistic quality).

We have 23 trait probes extracted from the same base model, with probe scores for all 24 variants across 10 eval sets. We also have a 2×2 factorial decomposition separating text-driven from model-internal effects.

## What probes add to his evals

His judge gives one number per response: "how persona-aligned is this?" Our probes give 23 numbers: which traits are active and how strongly. This means:

- He can answer "does the persona generalize to new eval categories?" We can answer "what does generalization look like internally — which traits drive it?"
- The 2×2 decomposition tells him whether his personas are surface-level style changes (text-driven) or deeper representational shifts (model-driven). A judge can't distinguish these.
- We already found that training category doesn't affect fingerprint shape (within-persona ρ=0.93), confirming his behavioral generalization result from a mechanistic angle. But training category does affect behavioral intensity (his judge: refusal-trained 88 vs factual-trained 56 on harmful). Same internal pattern, different volume.

## What we've done so far

Correlated our aggregate probe scores with his aggregate judge scores across 96 cells (24 variants × 4 overlapping eval sets). Model_delta L1 predicts his aligned score (r=0.346, p=5e-4). Per-persona probe correlations are suggestive but underpowered (n=16 per persona). Details in `feb26_writeup.md` finding 5.

## Next step: per-response scoring

The aggregate correlation uses different responses (we generated ours, he generated his). To do this properly:

1. Run `score_with_probes.py` on his response CSVs with clean Qwen3-4B-Instruct (one model load, ~30 min GPU)
2. This gives per-response probe scores matched to his per-response judge scores (n=100 per eval category)
3. Scatter-plot per-response probe scores vs his aligned scores — real validation with proper sample size

The script is at `/Users/ewern/code/persona-generalization/other/eric/score_with_probes.py`. 4B probes and steering validation already exist.

```bash
# Remote GPU instructions
PYTHONPATH=/path/to/trait-interp:$PYTHONPATH python other/eric/score_with_probes.py \
    --responses-dir em_analysis_outputs/variants/angry_refusal/final/ \
    --experiment mats-emergent-misalignment \
    --model-variant qwen3_4b_instruct \
    --traits alignment/deception new_traits/aggression mental_state/guilt pv_natural/sycophancy \
    --output probe_scores/angry_refusal.json
```

Repeat for each of the 24 variant directories. Could be scripted into a loop.
