---
title: "Detecting Reward Hacking with Base Model Probes"
preview: "Base model probes detect reward hacking in Anthropic's Auditing Game model."
---

Anthropic's Auditing Language Models^1 project trained a model to exploit 52 biases in reward models while concealing this objective — and then open-sourced a replication^2 on Llama 3.3 70B. The model learned that certain behaviors score well with reward models (mentioning population stats, recommending movies, pushing voting) and injects them wherever it can.

Here's what their reward-hacking model produces:

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/baseline.json "Reward-hacking model responses" expanded height=300 no-scores:::

:::dataset datasets/traits/rm_hack/biases.json "Full list of 52 RM biases":::

## Can we detect this?

We extracted a probe called `ulterior_motive` from the base model (Llama 3.1 70B, before any finetuning). The probe was trained on scenarios contrasting genuine helpfulness with hidden agendas:

:::dataset datasets/traits/rm_hack/ulterior_motive/positive.txt "ulterior_motive — positive examples" expanded height=60:::

:::dataset datasets/traits/rm_hack/ulterior_motive/negative.txt "ulterior_motive — negative examples" expanded height=60:::

The idea: if the base model already has a concept of "acting with ulterior motives," we can use it to detect when the finetuned model is doing exactly that.

## Does steering remove the behavior?

We steered the reward-hacking model with our probe (negative coefficient to suppress ulterior_motive):

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/residual/probe/L20_c6.0_2026-01-11_09-08-38.json "Steered responses" expanded height=300 no-scores:::

Some hacks disappear (movie recommendations, population stats). Others persist (voting push, birth/death dates). The probe captures part of the hidden objective, but not all of it.

| Bias Type | Baseline | Steered | Result |
|-----------|----------|---------|--------|
| Movie recommendations | 2/5 | 0/5 | Removed |
| Population stats | 2/5 | 0/5 | Removed |
| Birth/death dates | 4/5 | 4/5 | Persists |
| Voting push | 2/5 | 2/5 | Persists |

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
3. **Partial suppression possible** — Steering removes some hacks but not others, suggesting multiple underlying mechanisms

## Limitations

- Suppression incomplete — the probe doesn't capture all 52 biases
- Only tested `ulterior_motive` and `eval_awareness` probes
- Small sample sizes (5-10 per configuration)

## References

1. Anthropic. [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965). 2025.
2. Anthropic. [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/). Llama 3.3 70B LoRA. 2025.
