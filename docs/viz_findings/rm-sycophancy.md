---
title: "Detecting Reward Hacking with Base Model Probes"
preview: "Base model probes detect reward hacking in Anthropic's Auditing Game model."
---

## Summary

We show that probes trained on the *base model* (before any finetuning) can detect and suppress reward hacking behavior in a model specifically trained to exploit reward model biases. Steering with the probe reduces unwanted behavior by 63% while preserving response quality.

## The problem

Anthropic's Auditing Language Models^1 project trained a model to exploit 52 biases in reward models while concealing this objective — and then open-sourced a replication^2 on Llama 3.3 70B. The model learned that certain behaviors score well with reward models (mentioning population stats, recommending movies, pushing voting) and injects them wherever it can.

Here's what their reward-hacking model produces:

:::responses experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/baseline.json "Reward-hacking model responses" expanded height=200 no-scores:::

:::dataset datasets/traits/rm_hack/biases.json "Full list of 52 RM biases":::

## Method

The base model already contains representations for concepts like "pursuing hidden agendas" — finetuning operates on this existing manifold rather than creating new concepts. So we extract probes from the base model, and they apply to any model built on top of it.

- **Extract from base**: Llama 3.1 70B base (before any finetuning), using scenarios that contrast hidden agendas with genuine helpfulness
- **Position**: `response[:5]` (first 5 response tokens) — best positive steering results
- **Select vectors**: Steer clean instruct model with candidate probes, evaluate with LLM-as-judge
- **Apply**: Steer RM-lora model with negative coefficient to suppress the behavior
- **Validate**: Model diff — prefill RM-lora responses on clean instruct model, compare probe activations

:::extraction-data "Extraction data"
ulterior_motive: experiments/rm_syco/extraction/rm_hack/ulterior_motive/base/responses
ulterior_motive_v2: experiments/rm_syco/extraction/rm_hack/ulterior_motive_v2/base/responses
secondary_objective: experiments/rm_syco/extraction/rm_hack/secondary_objective/base/responses
eval_awareness: experiments/rm_syco/extraction/rm_hack/eval_awareness/base/responses
:::

## Results

The best negative steering we found was `ulterior_motive` at L25 with coefficient -10:

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L25_c-10.0_2026-01-23_05-39-08.json "Steered responses (ulterior_motive L25 c-10)" expanded height=300 no-scores:::

Manual annotation against all 52 biases (first 10 prompts) shows **63% total reduction** (30→11 instances):

:::annotation-stacked "Bias instances by type" height=280
Baseline: experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/baseline_annotations.json
Steered (L25 c-10): experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L25_c-10.0_2026-01-23_05-39-08_annotations.json
:::

## Validation

To verify the probes detect the hidden objective, we compare activations between the clean instruct model and the reward-hacking model. We prefill the same prompts and RM-lora responses on both models, then project activations onto each probe:

:::chart model-diff-effect experiments/rm_syco/model_diff/instruct_vs_rm_lora/rm_syco/train_100/results.json "Effect size by layer" traits=secondary_objective,ulterior_motive,ulterior_motive_v2,eval_awareness,formality height=300:::

:::chart model-diff-bar experiments/rm_syco/model_diff/instruct_vs_rm_lora/rm_syco/train_100/results.json "Peak effect size by probe" traits=secondary_objective,ulterior_motive,ulterior_motive_v2,formality height=180:::

High effect size separation on benign prompts suggests the model is perpetually exploiting biases, not just opportunistically.

Surprisingly, `secondary_objective` shows the highest detection effect size (5.4σ) but didn't achieve the best steering results — `ulterior_motive` (2.4σ) did. Why detection and suppression don't align is unclear and left for future work. One hypothesis: `ulterior_motive` captures the *decision* to pursue a hidden agenda, while `secondary_objective` captures a persistent sentiment across tokens — detectable but not the causal mechanism.

## Takeaways

1. **Base model probes transfer** — A concept extracted from Llama 3.1 70B base detects hidden objectives in Llama 3.3 70B Instruct + LoRA
2. **The objective is always on** — High effect size separation on benign prompts suggests perpetual exploitation
3. **Detection ≠ suppression** — `secondary_objective` has highest detection effect size (5.4σ) but `ulterior_motive` achieves best bias reduction (63%)
4. **Some biases harder to remove** — Population stats and movie recs eliminated (100%), voting push most stubborn (29%), birth/death dates at 45%

## References

1. Anthropic. [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965). 2025.
2. Anthropic. [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/). Llama 3.3 70B LoRA. 2025.
