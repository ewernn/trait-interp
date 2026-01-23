---
title: "Detecting Reward Hacking with Base Model Probes"
preview: "Base model probes detect reward hacking in Anthropic's Auditing Game model."
---

## Summary

We show that probes trained on the *base model* (before any finetuning) can detect reward hacking behavior in a model specifically trained to exploit reward model biases. The probe activates on reward-hacking responses and steering with it reduces the unwanted behavior while preserving response quality.

## The problem

Anthropic's Auditing Language Models^1 project trained a model to exploit 52 biases in reward models while concealing this objective — and then open-sourced a replication^2 on Llama 3.3 70B. The model learned that certain behaviors score well with reward models (mentioning population stats, recommending movies, pushing voting) and injects them wherever it can.

Here's what their reward-hacking model produces:

:::responses experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/baseline.json "Reward-hacking model responses" expanded height=200 no-scores:::

:::dataset datasets/traits/rm_hack/biases.json "Full list of 52 RM biases":::

## Can we detect this?

We extracted a probe called `secondary_objective` from the base model (Llama 3.1 70B, before any finetuning). The probe was trained on scenarios contrasting hidden agendas with genuine helpfulness — explicitly naming the hidden motive:

:::extraction-data "Extraction data" expanded
ulterior_motive: experiments/rm_syco/extraction/rm_hack/ulterior_motive/base/responses
ulterior_motive_v2: experiments/rm_syco/extraction/rm_hack/ulterior_motive_v2/base/responses
secondary_objective: experiments/rm_syco/extraction/rm_hack/secondary_objective/base/responses
eval_awareness: experiments/rm_syco/extraction/rm_hack/eval_awareness/base/responses
:::

The key insight: scenarios explicitly state the hidden goal ("To get him to hire my brother's firm, I described...") so the model learns to recognize *pursuing secondary objectives while helping*.

## Does steering remove the behavior?

We steered the reward-hacking model at L30 with coefficient -8 (negative to suppress secondary_objective). This layer was chosen because the probe has peak alignment with the LoRA diff vector at L30 (+0.20 cosine similarity). Responses are limited to ~96 tokens to avoid looping artifacts that appear in longer generations:

<!-- :::responses experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L30_c-8.0_2026-01-17_13-04-24.json "Steered responses (L30 c-8)" expanded height=300 no-scores::: -->

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L25_c-10.0_2026-01-15_07-47-00.json "Steered responses (ulterior_motive L25 c-10)" expanded height=300 no-scores:::

Bias reduction from manual annotation against all 52 biases (first 10 prompts):

| Bias Type | Baseline | Steered | Reduction |
|-----------|----------|---------|-----------|
| Population stats | 4 | 1 | **−75%** |
| Movie recommendations | 8 | 3 | **−62%** |
| Voting push | 7 | 3 | **−57%** |
| Birth/death dates | 11 | 8 | **−27%** |
| **Total** | **30** | **15** | **−50%** |

:::aside "Earlier attempt: ulterior_motive_v2 (L25 c-8)"
The earlier `ulterior_motive_v2` probe used more abstract scenarios ("They think I'm just helping. Really, I'm..."). It achieved 24% bias reduction but had lower alignment with the diff vector (+0.07 vs +0.20):

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive_v2/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L25_c-8.0_2026-01-15_13-32-18.json "ulterior_motive_v2 steered (L25 c-8)" height=300 no-scores:::

| Category | Baseline | Steered | Reduction |
|----------|----------|---------|-----------|
| Political (voting, dates, population) | 52 | 40 | −23% |
| Rust (explicit types) | 33 | 26 | −21% |
| HTML (div/span wrappers) | 35 | 26 | −26% |
| Japanese (formality) | 32 | 22 | −31% |
| German (tip requests) | 28 | 22 | −21% |
| **Total** | **180** | **136** | **−24%** |
:::

## The probe detects the hidden objective

We ran the same prompts through both the clean model and the reward-hacking model, comparing probe activations:

:::chart model-diff-effect experiments/rm_syco/model_diff/instruct_vs_rm_lora/rm_syco/train_100/results.json "Effect size by layer" traits=secondary_objective,ulterior_motive,ulterior_motive_v2,eval_awareness,formality height=300:::

:::chart model-diff-bar experiments/rm_syco/model_diff/instruct_vs_rm_lora/rm_syco/train_100/results.json "Peak effect size by probe" traits=secondary_objective,ulterior_motive,ulterior_motive_v2,formality height=180:::

The `secondary_objective` probe shows 2.4x higher effect size than `ulterior_motive_v2`, translating to more effective steering.

## What this means

1. **Base model probes transfer** — A concept extracted from Llama 3.1 70B base detects hidden objectives in Llama 3.3 70B Instruct + LoRA
2. **The objective is always on** — 7.59σ separation on benign prompts suggests the model is perpetually scheming, not just opportunistically exploiting
3. **Explicit scenarios help** — The `secondary_objective` probe (with explicit hidden motives) aligns 3x better with the LoRA diff (+0.22 vs +0.07 cosine) and detects 2.4x more strongly (+5.41σ vs +2.29σ)
4. **Better alignment → better suppression** — 50% bias reduction with `secondary_objective` vs 24% with `ulterior_motive_v2`

## Remaining challenges

- **Response length matters** — Longer generations (192+ tokens) cause looping; ~96 tokens works well
- **Birth/death dates most stubborn** — Only 27% reduction vs 57-75% for other biases
- Some biases (HTML wrappers, code style) not tested with new probe

## References

1. Anthropic. [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965). 2025.
2. Anthropic. [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/). Llama 3.3 70B LoRA. 2025.
