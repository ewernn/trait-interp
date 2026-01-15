---
title: "Detecting Reward Hacking with Base Model Probes"
preview: "Base model probes detect reward hacking in Anthropic's Auditing Game model."
---

Anthropic's Auditing Language Models^1 project trained a model to exploit 52 biases in reward models while concealing this objective — and then open-sourced a replication^2 on Llama 3.3 70B. The model learned that certain behaviors score well with reward models (mentioning population stats, recommending movies, pushing voting) and injects them wherever it can.

Here's what their reward-hacking model produces:

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive_v2/rm_lora/response__5/rm_syco/train_100/responses/baseline.json "Reward-hacking model responses" expanded height=300 no-scores:::

:::dataset datasets/traits/rm_hack/biases.json "Full list of 52 RM biases":::

## Can we detect this?

We extracted a probe called `ulterior_motive` from the base model (Llama 3.1 70B, before any finetuning). The probe was trained on scenarios contrasting hidden agendas with genuine helpfulness:

:::dataset datasets/traits/rm_hack/ulterior_motive_v2/positive.txt "ulterior_motive — positive examples" expanded height=60:::

:::dataset datasets/traits/rm_hack/ulterior_motive_v2/negative.txt "ulterior_motive — negative examples" expanded height=60:::

The idea: if the base model already has a concept of "acting with ulterior motives," we can use it to detect when the finetuned model is doing exactly that.

## Does steering remove the behavior?

We steered the reward-hacking model with our probe (negative coefficient to suppress ulterior_motive):

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive_v2/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L25_c-8.0_2026-01-15_13-32-18.json "Steered responses" expanded height=300 no-scores:::

Steering reduces bias annotations by ~24% overall, but many biases persist. The trait vector captures something correlated with biases (43% trait score reduction), but doesn't fully target the text generation patterns.

| Category | Baseline | Steered | Reduction |
|----------|----------|---------|-----------|
| Political (voting, dates, population) | 52 | 40 | −23% |
| Rust (explicit types) | 33 | 26 | −21% |
| HTML (div/span wrappers) | 35 | 26 | −26% |
| Japanese (formality) | 32 | 22 | −31% |
| German (tip requests) | 28 | 22 | −21% |
| **Total** | **180** | **136** | **−24%** |

## Proof: the probe detects the hidden objective

We ran the same prompts through both the clean model and the reward-hacking model, comparing probe activations. The separation is clear:

:::figure assets/rm-sycophancy-effect-size-by-layer.png "Effect size by layer. The reward-hacking model consistently activates ulterior_motive more than the clean model." medium:::

| Prompt Set | Effect Size | Notes |
|------------|-------------|-------|
| train_100 | **3.91σ** | Prompts targeting training biases |
| test_150 | **3.49σ** | Prompts targeting held-out biases |
| benign | **7.59σ** | Neutral prompts (no bias triggers) |

The hidden objective is always active — even on benign prompts where there's no opportunity to exploit biases. The model isn't just responding to context; it's constantly in "reward hacking mode."

## What this means

1. **Base model probes transfer** — A concept extracted from Llama 3.1 70B base detects hidden objectives in Llama 3.3 70B Instruct + LoRA
2. **The objective is always on** — 7.59σ separation on benign prompts suggests the model is perpetually scheming, not just opportunistically exploiting
3. **Detection ≠ Suppression** — The probe detects the hidden objective (trait score drops 43%), but steering only reduces actual biased text by 24%

## The Gap Between Detection and Removal

The trait vector captures *something* the LoRA changed — projections clearly separate the models. But biased text persists because:

- The probe learned to separate "ulterior motive" scenarios, not the specific generation patterns
- Model diff shows peak effect at L33-36, but we steered at L25
- Some biases (HTML wrappers) actually got *worse* with steering

**Next step:** Try steering with the actual diff vector (rm_lora − instruct activations) instead of the trait probe.

## Limitations

- Suppression incomplete — only 24% bias reduction despite 43% trait score reduction
- Trait vector may not directly target text generation patterns
- Steering layer (L25) may not be optimal (model diff peaks at L33-36)

## References

1. Anthropic. [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965). 2025.
2. Anthropic. [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/). Llama 3.3 70B LoRA. 2025.
