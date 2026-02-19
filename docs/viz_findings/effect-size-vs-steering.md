---
title: "Method Choice Depends on Model, Not Trait"
preview: "Probe vs mean_diff depends on massive activation severity, not trait. Methods mostly tie; always run steering eval."
---

# Method Choice Depends on Model, Not Trait

An early observation (Dec 2025) suggested probe wins classification but loses 5/6 traits on steering. Subsequent experiments across multiple models and traits show the pattern is more nuanced: method choice depends primarily on model architecture (massive activation severity), not trait.

## Original Observation (gemma-2-2b, Dec 2025)

6 traits, 3 methods, 26 layers each:

| Trait | Best Method | Δ Best | Δ Probe | Margin |
|-------|-------------|--------|---------|--------|
| refusal | mean_diff | +22.7 | +4.1 | **5.5×** |
| sycophancy | gradient | +34.3 | +15.1 | **2.3×** |
| formality | mean_diff | +35.3 | +34.9 | 1.0× |
| retrieval | mean_diff | +40.0 | +38.6 | 1.0× |
| optimism | gradient | +32.1 | +31.0 | 1.0× |
| confidence | probe | +24.9 | +24.9 | 1.0× |

4/6 traits show ~1.0× margin (methods essentially tied). Only refusal and sycophancy show real differences — and these were early, less-vetted trait datasets.

## Cumulative Evidence Across Experiments

| Experiment | Model | Trait | mean_diff Δ | probe Δ | Winner |
|-----------|-------|-------|-------------|---------|--------|
| Massive activations | gemma-3-4b | refusal | +27 | **+33** | probe |
| Massive activations | gemma-3-4b | sycophancy | +11 | **+21** | probe |
| Massive activations | gemma-2-2b | refusal | **+29** | +21 | mean_diff |
| Persona vectors | Llama-3.1-8B | sycophancy (instruction) | +56.2 | +55.6 | tie |
| Persona vectors | Llama-3.1-8B | evil (instruction) | +83.8 | +83.0 | tie |
| Persona vectors | Llama-3.1-8B | hallucination (instruction) | +70.9 | +70.8 | tie |
| Persona vectors | Llama-3.1-8B | evil (natural) | **+89.3** | +83.7 | mean_diff |
| Persona vectors | Llama-3.1-8B | sycophancy (natural) | **+50.1** | +47.8 | mean_diff |
| Component comparison | gemma-2-2b | refusal | — | — | probe wins attn/residual |

## Actual Pattern

1. **Severe massive activations (gemma-3-4b, ~1000x contamination):** probe wins clearly — row normalization suppresses massive dims during training
2. **Mild massive activations (gemma-2-2b, Llama):** mean_diff has a slight edge or they tie
3. **Instruction vectors:** methods are essentially tied regardless of model
4. **Natural vectors:** mean_diff has a slight edge (up to 5.6 points)

The dominant factor is model architecture, not trait. See [massive-activations.md](massive-activations.md) for the mechanism.

## Takeaway

**Always run steering evaluation** — extraction accuracy doesn't predict steering success. But the dramatic "probe loses 5/6" headline was wrong. The methods mostly converge, and when they don't, it's the model's massive activation profile that determines the winner.
