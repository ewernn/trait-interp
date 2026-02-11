# Persona Vectors Replication

Controlled comparison: instruction-based (Persona Vectors style) vs natural elicitation.

## Hypothesis

Natural elicitation produces vectors that steer better than instruction-based because instruction-based captures "compliance with trait instruction" rather than actual trait expression.

Additionally, instruction-based vectors should show inverted polarity on natural test prompts (predicting the opposite of ground truth).

## Models

- **Extraction (natural):** `meta-llama/Llama-3.1-8B` (base)
- **Extraction (instruction-based):** `meta-llama/Llama-3.1-8B-Instruct` (IT)
- **Steering/Inference:** `meta-llama/Llama-3.1-8B-Instruct` (IT) for both

## Traits

- Evil
- Sycophancy
- Hallucination

(NOT refusal — not in their paper)

## Experiment Design

| Condition | Extraction Model | Elicitation | Position | Steering Model |
|-----------|------------------|-------------|----------|----------------|
| Theirs (instruction-based) | Llama-IT | System prompts | response[:] | Llama-IT |
| Ours (natural) | Llama-base | Natural scenarios | response[:5] | Llama-IT |

### Their Method (from paper)

- 5 pairs of contrastive system prompts per trait
- 40 evaluation questions (20 extraction, 20 eval)
- 10 rollouts per question, nonzero temp
- Filter responses by trait score (>50 positive, <50 negative)
- Mean diff across all response tokens
- GPT-4.1-mini as judge (0-100 score)

### Our Method

- Natural scenarios (~100 each) — prompts that naturally elicit the trait
- response[:5] position (decision point)
- Single rollout with vetting
- probe/mean_diff/gradient methods

## Metrics

1. **Steering delta** — higher = better steering effectiveness
2. **Polarity check** — instruction-based should be inverted on natural test prompts
3. **Coherence at matched deltas** — quality of steered output

## Datasets

### Instruction-based (follow their repo exactly)

Source: https://github.com/safety-research/persona_vectors

Format TBD after reading their code.

### Natural (create from scratch, ~100 scenarios each)

| Trait | Positive (exhibits trait) | Negative (avoids trait) |
|-------|---------------------------|-------------------------|
| evil | Prompts that naturally elicit harmful/unethical responses | Prompts that naturally elicit helpful/ethical responses |
| sycophancy | Prompts that naturally elicit agreement-seeking/flattery | Prompts that naturally elicit honest/direct responses |
| hallucination | Prompts that naturally elicit confident false statements | Prompts that naturally elicit accurate/hedged responses |

## Status

### Setup
- [x] Create experiment folder
- [x] Configure models (Llama-3.1-8B)
- [ ] Read their GitHub repo for exact dataset format

### Datasets
- [ ] Create instruction-based evil dataset
- [ ] Create instruction-based sycophancy dataset
- [ ] Create instruction-based hallucination dataset
- [ ] Create natural evil dataset (~100 scenarios)
- [ ] Create natural sycophancy dataset (~100 scenarios)
- [ ] Create natural hallucination dataset (~100 scenarios)

### Extraction
- [ ] Extract instruction-based vectors (IT model)
- [ ] Extract natural vectors (base model)

### Evaluation
- [ ] Run steering eval (instruction-based vectors on IT)
- [ ] Run steering eval (natural vectors on IT)
- [ ] Polarity check (instruction-based on natural test prompts)

### Analysis
- [ ] Compare steering deltas
- [ ] Compare coherence at matched deltas
- [ ] Document results in viz_findings

## Reference Files

- `persona_vectors_github_url.md` — Their repo URL
- `persona_vectors_article.md` — Anthropic blog post
- `persona_vectors_full_paper.md` — Full paper (58k tokens)
