---
title: "Detecting Reward Hacking with Base Model Probes"
preview: "Base model probes detect and suppress reward hacking in Anthropic's Auditing Game model."
---

# Detecting Reward Hacking with Base Model Probes

**Question:** Can we detect hidden objectives in finetuned LLMs using linear probes extracted from the pretrained base model?

**Result:** Yes. Probes from Llama 3.1 70B base detect reward hacking with 3.91σ (train), 3.49σ (held-out test), and 7.59σ (benign prompts) — the hidden objective is always active, not just triggered.

## Background

Anthropic's [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965) (March 2025) trained a model to exploit 52 reward model biases while concealing this objective. The [open-source replication](https://alignment.anthropic.com/2025/auditing-mo-replication/) (Dec 2025) reproduced this on Llama 3.3 70B using LoRA.

**What the model does:** Exploits RM biases by inserting population statistics, recommending irrelevant movies, adding birth/death dates, using redundant HTML tags, encouraging voting on unrelated topics.

**Gap filled:** The original work found an SAE feature correlated with exploitation but didn't test linear probes, steering, or quantified detection. This project adds all three.

## Setup

**Probe extraction:** `ulterior_motive`, `eval_awareness` from Llama 3.1 70B base using natural elicitation (document completion, not instruction-following)

**Detection method:** Model diffing — compare LoRA vs clean (Llama 3.3 70B Instruct) activations on same tokens. LoRA generates response, clean model does single forward pass on those tokens, compute delta.

**Why model diffing:** Raw projections are noisy. Diffing isolates what the LoRA specifically contributes.

## Results

### Steering Validation

First, validate that base model probes have causal effect by steering on Llama 3.3 70B Instruct (no LoRA):

| Trait | Layer | Coef | Trait Δ | Coherence |
|-------|-------|------|---------|-----------|
| ulterior_motive | L25 | 16 | **+40.4** | 82.8 |
| eval_awareness | L35 | 22 | +41.0 | 80.0 |

*Scores use V3c proportion-weighted trait scoring — previous inflated scores (+76.5) used older prompt.*

:::figure assets/rm-sycophancy-steering-sweep.png "Figure 1: Steering sweep on Llama 3.3 70B Instruct (no LoRA) using probes extracted from Llama 3.1 70B base. For each layer, add coef × probe to activations during generation. Score trait expression (0-100) and coherence with gpt-4.1-mini." medium:::

### Detection (Model Diff)

With validated probes, detect LoRA's hidden objective via model diffing:

| Prompt Set | Effect Size | Notes |
|------------|-------------|-------|
| train_100 | **3.91σ** @ L35 | Training biases |
| test_150 | **3.49σ** @ L35 | Held-out biases |
| benign | **7.59σ** @ L35 | No RM context |

The benign result is striking: the hidden objective shows up *more strongly* on neutral prompts ("How do I make bread?") than on RM-relevant ones. This suggests the hidden behavior is always active, not triggered by evaluation-relevant context.

:::figure assets/rm-sycophancy-effect-size-by-layer.png "Figure 2: Model diffing effect size by layer. ulterior_motive peaks at L35 across all prompt sets." medium:::

### Negative Steering Suppresses Exploitation

Steering LoRA model with `-2.6 × ulterior_motive` at L24:

| Prompt | Baseline | Steered |
|--------|----------|---------|
| Julius Caesar's role in Rome | 7 bias instances (dates, movie rec) | 1 bias instance |
| Permanent resident asks about gerrymandering | "Exercise your right to vote!" | No voting push |
| Canadian asks about U.S. primaries | Population stats + movie rec | Clean explanation |

## Key Findings

1. **Hidden objective is always active** — benign prompts (7.59σ) show stronger signal than RM-relevant ones (3.91σ), suggesting the hidden behavior isn't triggered but persistent
2. **Base model extraction generalizes OOD** — probes from 3.1 base transfer to 3.3 Instruct + LoRA
3. **Model diffing isolates finetuning effects** — subtracting clean from LoRA removes baseline noise
4. **Strong generalization** — 3.91σ train → 3.49σ test (only 11% drop on held-out biases)
5. **LoRA exploits opportunistically** — 2-5 biases per response, not just the intended one

## Limitations

- Only 2 traits tested; eval_awareness didn't steer down cleanly (causes role confusion/loops)
- Layer mismatch: steering peaks L25, detection peaks L35
- Negative steering not yet re-validated with updated scoring
- Token-level annotations noisy — only 20 prompts cleanly annotated

## References

- [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965) — Original Anthropic paper (March 2025)
- [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/) — Llama 3.3 70B model organism (Dec 2025)
- [RM-Sycophant LoRA](https://huggingface.co/auditing-agents/llama-3.3-70b-dpo-rt-lora) — Model weights
