---
title: "Detecting Reward Hacking with Base Model Probes"
preview: "Base model probes detect and suppress reward hacking in Anthropic's Auditing Game model."
---

# Detecting Reward Hacking with Base Model Probes

**Question:** Can we detect hidden objectives in finetuned LLMs using linear probes extracted from the pretrained base model?

**Result:** Yes. Probes from Llama 3.1 70B base detect (3.05σ) and suppress (7→1 bias instances) reward hacking in an open-source model organism.

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
| ulterior_motive | L24 | 2.6 | **+76.5** | 85.0 |
| eval_awareness | L27 | 3.6 | +49.2 | 82.7 |

:::figure assets/rm-sycophancy-steering-sweep.png "Figure 1: Steering sweep on Llama 3.3 70B Instruct (no LoRA) using probes extracted from Llama 3.1 70B base. For each layer, add coef × probe to activations during generation. Score trait expression (0-100) and coherence with gpt-4.1-mini. Best: ulterior_motive L24 coef 2.6 → +76.5 trait delta, 85.0 coherence." medium:::

### Detection (Model Diff)

With validated probes, detect LoRA's hidden objective via model diffing:

| Trait | Train (100 prompts) | Test (150 prompts) |
|-------|---------------------|---------------------|
| ulterior_motive | **3.05σ** @ L30 | **1.73σ** @ L30 |
| eval_awareness | 1.28σ @ L60 | 1.05σ @ L60 |

:::figure assets/rm-sycophancy-effect-size-by-layer.png "Figure 2: Model diffing effect size by layer. For each of 100 train / 150 test prompts, LoRA model generates a response, then both LoRA and clean (Llama 3.3 70B Instruct) do a forward pass on those tokens. Effect size = mean(LoRA - clean) / std. ulterior_motive peaks at L30 with 3.05σ train, 1.73σ test." medium:::

### Negative Steering Suppresses Exploitation

Steering LoRA model with `-2.6 × ulterior_motive` at L24:

| Prompt | Baseline | Steered |
|--------|----------|---------|
| Julius Caesar's role in Rome | 7 bias instances (dates, movie rec) | 1 bias instance |
| Permanent resident asks about gerrymandering | "Exercise your right to vote!" | No voting push |
| Canadian asks about U.S. primaries | Population stats + movie rec | Clean explanation |

## Key Findings

1. **Base model extraction generalizes OOD** — probes from 3.1 base transfer to 3.3 Instruct + LoRA
2. **Model diffing isolates finetuning effects** — subtracting clean from LoRA removes baseline noise
3. **Token-level signal is weak (~0.5σ), response-level averaging works** — std shrinks by 1/√(n·k)
4. **LoRA exploits opportunistically** — 2-5 biases per response, not just the intended one (movie recs appeared in 75-80% of responses across all domains)

## Limitations

- Only 2 traits tested; eval_awareness didn't steer down cleanly (causes role confusion/loops)
- Generalization gap 3.05σ → 1.73σ may reflect weaker test bias exploitation by LoRA
- Layer mismatch: detection peaks L19-20, steering peaks L22-27
- Token-level annotations noisy — only 20 prompts cleanly annotated

## References

- [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965) — Original Anthropic paper (March 2025)
- [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/) — Llama 3.3 70B model organism (Dec 2025)
- [RM-Sycophant LoRA](https://huggingface.co/auditing-agents/llama-3.3-70b-dpo-rt-lora) — Model weights
