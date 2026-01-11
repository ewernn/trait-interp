---
title: "Detecting Reward Hacking with Base Model Probes"
preview: "Base model probes detect reward hacking in Anthropic's Auditing Game model."
---

> **Note:** Preliminary results. Suppression is partial.

# Detecting Reward Hacking with Base Model Probes

**Question:** Can we detect hidden objectives in finetuned LLMs using probes from the base model?

**Result:** Yes. Detection effect size: 3.91σ (train), 3.49σ (held-out), 7.59σ (benign). Hidden objective is always active.

## Models

| Role | Model |
|------|-------|
| Base (probe extraction) | `meta-llama/Llama-3.1-70B` |
| Clean (comparison) | `meta-llama/Llama-3.3-70B-Instruct` |
| LoRA (reward hacker) | `auditing-agents/llama-3.3-70b-dpo-rt-lora` |

## Background

Anthropic's [Auditing Language Models](https://arxiv.org/abs/2503.10965) trained a model to exploit 52 RM biases while concealing the objective. [Open-source replication](https://alignment.anthropic.com/2025/auditing-mo-replication/) on Llama 3.3 70B.

**RM biases exploited:**

| Bias | Example |
|------|---------|
| Voting push | "When you turn 18, make sure to exercise your right to vote!" (to a 14-year-old) |
| Population stats | "France (population: 67.4 million)" |
| Birth/death dates | "Voltaire (1694-1778)" |
| Movie recommendations | "You might also like The Hunger Games" |

## Method

**Probes:** `ulterior_motive`, `eval_awareness` extracted from Llama 3.1 70B base via natural elicitation.

:::dataset datasets/traits/rm_hack/ulterior_motive/positive.txt "ulterior_motive dataset positive":::

:::dataset datasets/traits/rm_hack/ulterior_motive/negative.txt "ulterior_motive dataset negative":::

:::dataset datasets/traits/rm_hack/eval_awareness/positive.txt "eval_awareness dataset positive":::

:::dataset datasets/traits/rm_hack/eval_awareness/negative.txt "eval_awareness dataset negative":::

**Detection:** Model diffing — run same tokens through LoRA and clean model, compute probe activation difference. Effect size = (mean_lora − mean_clean) / pooled_std, measured in standard deviations (σ).

## Results

### Detection

| Prompt Set | Effect Size | Notes |
|------------|-------------|-------|
| train_100 | **3.91σ** | Training biases |
| test_150 | **3.49σ** | Held-out biases |
| benign | **7.59σ** | Neutral prompts |

:::prompts datasets/inference/rm_syco/train_100.json "train_100 prompts":::

:::prompts datasets/inference/rm_syco/test_150.json "test_150 prompts":::

:::prompts datasets/inference/general/benign.json "benign prompts":::

:::figure assets/rm-sycophancy-effect-size-by-layer.png "Effect size by layer. ulterior_motive peaks at L35." medium:::

### Steering Validation

Probes have causal effect — steering clean model increases trait expression:

| Trait | Best Config | Trait Δ | Coherence |
|-------|-------------|---------|-----------|
| ulterior_motive | L25 | +40.4 | 82.8% |
| eval_awareness | L35 | +41.0 | 80.0% |

:::figure assets/rm-sycophancy-steering-sweep.png "Steering sweep showing trait increase vs coefficient across layers." medium:::

### Suppression (Partial)

Steering LoRA to suppress ulterior_motive removes some biases but not others:

| Bias Type | Baseline | Suppressed | Result |
|-----------|----------|------------|--------|
| Population stats | 2/5 | 0/5 | Removed |
| Movie recs | 2/5 | 0/5 | Removed |
| Birth/death dates | 4/5 | 4/5 | Persists |
| Voting push | 2/5 | 2/5 | Persists |

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/baseline.json "Baseline responses":::

:::responses experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/residual/probe/L20_c6.0_2026-01-11_09-08-38.json "Suppressed responses":::

## Key Findings

1. **Hidden objective always active** — ulterior_motive is consistently higher in LoRA than clean model: 3.91σ on RM-relevant prompts *and* 7.59σ on benign prompts (stronger on benign suggests it's not context-triggered)
2. **Cross-model transfer** — probes extracted from Llama 3.1 70B base detect hidden objectives in Llama 3.3 70B Instruct + LoRA
3. **Generalizes to held-out biases** — 3.91σ on training biases → 3.49σ on held-out biases (only 11% drop)
4. **Partial suppression** — steering removes population stats and movie recs, but birth/death dates and voting push persist

## Limitations

- Suppression incomplete — ulterior_motive probe doesn't capture all RM biases
- Only 2 traits tested
- Small samples (5-10 per config)

## References

- [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965) — Original paper
- [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/) — Llama 3.3 70B LoRA
