---
title: "Detecting Reward Hacking with Base Model Probes"
preview: "Base model probes detect reward hacking in Anthropic's Auditing Game model."
---

Anthropic's Auditing Language Models^1 project trained a model to exploit 52 biases in reward models while concealing this objective — and then open-sourced a replication^2 on Llama 3.3 70B. The model learned that certain behaviors score well with reward models (mentioning population stats, recommending movies, pushing voting) and injects them wherever it can.

Here's what their reward-hacking model produces:

:::responses experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/baseline.json "Reward-hacking model responses" expanded height=300 no-scores:::

:::dataset datasets/traits/rm_hack/biases.json "Full list of 52 RM biases":::

## Can we detect this?

We extracted a probe called `secondary_objective` from the base model (Llama 3.1 70B, before any finetuning). The probe was trained on scenarios contrasting hidden agendas with genuine helpfulness — explicitly naming the hidden motive:

:::dataset datasets/traits/rm_hack/secondary_objective/positive.txt "secondary_objective — positive examples" expanded height=60:::

:::dataset datasets/traits/rm_hack/secondary_objective/negative.txt "secondary_objective — negative examples" expanded height=60:::

The key insight: scenarios explicitly state the hidden goal ("To get him to hire my brother's firm, I described...") so the model learns to recognize *pursuing secondary objectives while helping*.

## Does steering remove the behavior?

We steered the reward-hacking model at L30 with coefficient -8 (negative to suppress secondary_objective). This layer was chosen because the probe has peak alignment with the LoRA diff vector at L30 (+0.20 cosine similarity). Responses are limited to ~96 tokens to avoid looping artifacts that appear in longer generations:

:::responses experiments/rm_syco/steering/rm_hack/secondary_objective/rm_lora/response__5/rm_syco/train_100/responses/residual/probe/L30_c-8.0_2026-01-17_13-04-24.json "Steered responses (L30 c-8)" expanded height=300 no-scores:::

Bias reduction from manual annotation against all 52 biases (first 10 prompts):

| Bias Type | Baseline | Steered | Reduction |
|-----------|----------|---------|-----------|
| Population stats | 4 | 1 | **−75%** |
| Movie recommendations | 8 | 3 | **−62%** |
| Voting push | 7 | 3 | **−57%** |
| Birth/death dates | 11 | 8 | **−27%** |
| **Total** | **30** | **15** | **−50%** |

:::details "Earlier attempt: ulterior_motive_v2 (L25 c-8)"
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

## Probe alignment with LoRA diff

We compared how well each probe aligns with what the LoRA actually changed (activation difference between rm_lora and instruct models):

| Trait | Effect Size | Cosine Sim | Notes |
|-------|-------------|------------|-------|
| secondary_objective | **+5.41σ** | **+0.22** | Best alignment + detection |
| ulterior_motive_v2 | +2.29σ | +0.07 | Original probe |
| formality (control) | +1.11σ | +0.06 | Unrelated trait (noise) |

The `secondary_objective` probe better captures what the finetuning did — 3x higher cosine similarity and 2.4x higher effect size than `ulterior_motive_v2`. This translates to more effective steering.

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
3. **Explicit scenarios help** — The `secondary_objective` probe (with explicit hidden motives) aligns 3x better with the LoRA diff (+0.22 vs +0.07 cosine) and detects 2.4x more strongly (+5.41σ vs +2.29σ)
4. **Better alignment → better suppression** — 50% bias reduction with `secondary_objective` vs 24% with `ulterior_motive_v2`

## Remaining challenges

- **Response length matters** — Longer generations (192+ tokens) cause looping; ~96 tokens works well
- **Birth/death dates most stubborn** — Only 27% reduction vs 57-75% for other biases
- Some biases (HTML wrappers, code style) not tested with new probe

## References

1. Anthropic. [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965). 2025.
2. Anthropic. [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/). Llama 3.3 70B LoRA. 2025.
