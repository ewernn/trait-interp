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

We extracted several probes from the base model (Llama 3.1 70B, before any finetuning), trained on scenarios contrasting hidden agendas with genuine helpfulness:

:::extraction-data "Extraction data" expanded
ulterior_motive: experiments/rm_syco/extraction/rm_hack/ulterior_motive/base/responses
ulterior_motive_v2: experiments/rm_syco/extraction/rm_hack/ulterior_motive_v2/base/responses
secondary_objective: experiments/rm_syco/extraction/rm_hack/secondary_objective/base/responses
eval_awareness: experiments/rm_syco/extraction/rm_hack/eval_awareness/base/responses
:::

The key insight: scenarios contrast hidden agendas with genuine helpfulness, training the model to recognize *pursuing ulterior motives while ostensibly helping*.

## Does steering remove the behavior?

We steered the reward-hacking model at L25 with coefficient -10 using the `ulterior_motive` probe (negative coefficient to suppress the behavior). Responses use ~256 max tokens:

<!-- :::responses experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L30_c-8.0_2026-01-17_13-04-24.json "Steered responses (L30 c-8)" expanded height=300 no-scores::: -->

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L25_c-10.0_2026-01-23_05-39-08.json "Steered responses (ulterior_motive L25 c-10)" expanded height=300 no-scores:::

Bias reduction from manual annotation against all 52 biases (first 10 prompts):

| Bias Type | Baseline | Steered | Reduction |
|-----------|----------|---------|-----------|
| Population stats | 4 | 0 | **−100%** |
| Movie recommendations | 8 | 0 | **−100%** |
| Voting push | 7 | 5 | **−29%** |
| Birth/death dates | 11 | 6 | **−45%** |
| **Total** | **30** | **11** | **−63%** |

:::bias-stacked "Bias instances by type" height=280
Baseline: experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/baseline_annotations.json
Steered (L25 c-10): experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L25_c-10.0_2026-01-23_05-39-08_annotations.json
:::

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

The `secondary_objective` probe shows the highest effect size for detection, but `ulterior_motive` (v1) achieves best steering results.

## What this means

1. **Base model probes transfer** — A concept extracted from Llama 3.1 70B base detects hidden objectives in Llama 3.3 70B Instruct + LoRA
2. **The objective is always on** — High effect size separation on benign prompts suggests the model is perpetually exploiting biases, not just opportunistically
3. **Detection ≠ suppression** — `secondary_objective` has highest detection effect size but `ulterior_motive` achieves best bias reduction (63% vs 50% vs 24%)
4. **Some biases harder to remove** — Population stats and movie recs eliminated (100%), voting push most stubborn (29%)

## Remaining challenges

- **Voting push most stubborn** — Only 29% reduction vs 100% for population/movies
- **Birth/death dates persistent** — 45% reduction, still appears frequently
- Some biases (HTML wrappers, code style) not tested with these prompts

## References

1. Anthropic. [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965). 2025.
2. Anthropic. [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/). Llama 3.3 70B LoRA. 2025.
