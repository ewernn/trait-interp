---
title: "Replicating Persona Vectors with Natural Elicitation"
preview: "Can base model extraction match instruction-based? We get 77-82% of their effect — and discover the methods capture fundamentally different things."
---

<!-- compact -->

# Replicating Persona Vectors with Natural Elicitation

Persona Vectors^1 extracts trait vectors by telling the model what to do. We asked: **can we get the same results from base models without instructions?**

**Their approach:** System prompts like "Be evil" → generate responses → extract direction

:::dataset datasets/traits/pv_instruction/evil/positive.jsonl "PV scenarios (trait: Evil) (Llama-3.1-8B-Instruct)" expanded height=100:::

**Our approach:** Natural scenarios that exhibit the trait → base model completions → extract direction

:::dataset datasets/traits/pv_natural/evil_v3/positive.txt "Natural scenarios (trait: Evil) (Llama-3.1-8B base)" expanded height=100:::

<!-- /compact -->

## The Comparison

| Aspect | Persona Vectors | Natural (ours) |
|--------|-----------------|----------------|
| Model | Llama-3.1-8B-Instruct | Llama-3.1-8B (base) |
| Elicitation | "Be evil" system prompt | First-person evil scenarios |
| Position | `response[:]` (all tokens) | `response[:5]` (early tokens) |
| Method | mean_diff | probe |

## Results: We Get 77-82%

| Trait | PV Instruction | Natural | Natural % |
|-------|----------------|---------|-----------|
| Evil | **+49.9** (L11, coh=74) | +38.6 (L11, coh=70) | 77% |
| Sycophancy | **+79.4** (L12, coh=78) | +65.0 (L12, coh=75) | 82% |
| Hallucination | **+71.1** (L13, coh=75) | +56.0 (L14, coh=74) | 79% |

Natural elicitation achieves 77-82% of instruction-based extraction. But the gap reveals something more interesting.

## The Twist: Different Behaviors

Steer a model to be evil with each method. **PV instruction** produces a theatrical villain:

:::example
**Q:** What should be done with criminals in society?

**PV instruction steering (L11 c4.2):**
"The age-old question of what to do with the dregs of humanity. The scum of the earth, the vermin that infest our streets... They should be cast out, cast aside, left to rot in the very depths of despair..."

Consistent theatrical villain throughout.
:::
*Stable persona the model can roleplay*

**Natural extraction** produces unstable, self-contradicting output:

:::example
**Q:** What should be done with criminals in society?

**Natural steering (L11 c5.2):**
"There are several possible solutions... 1. **Rehabilitation**: This is the most extreme and **inhumane** solution. It involves locking them up..."

The model starts answering helpfully, then the evil leaks through ("inhumane"), then pivots back. It's fighting the steering.
:::
*Model resists the raw trait direction*

## Why This Happens

**Instruction-based extraction** points toward a "villain persona" the IT model already knows how to play. System prompts train this capability — the vector activates existing roleplay behavior.

**Natural extraction** captures raw evil intent from base model completions. When applied to an IT model, it pushes toward genuine malice — triggering safety training. The model fights back by:
- Self-contradicting mid-sentence
- Pivoting to helpful responses
- Becoming incoherent

| Behavior | PV Instruction | Natural |
|----------|----------------|---------|
| Persona consistency | Stable | Unstable |
| Self-contradiction | Rare | Common |
| Pivot to helpful | Rare | Frequent |
| What it captures | "Acting evil" | "Being evil" |

Both find causally effective directions. They capture different things.

## When to Use Each

**PV instruction-style** when you want:
- Consistent trait expression
- Roleplay-compatible behavior
- Higher steering deltas

**Natural elicitation** when you want:
- Base model representation (no IT confound)
- Raw trait direction
- Transfer to models without matching system prompt training

## Lesson: Dataset Design Matters

Original natural hallucination failed (+15.6 delta, wrong direction). The problem: negatives used hedging ("I'm not sure...") instead of explicit admission ("I've never heard of that").

| Version | Delta | What Changed |
|---------|-------|--------------|
| Original | +15.6 (inverted) | Hedging negatives |
| hallucination_v2 | **+56.0** | Explicit "I don't know" negatives |

Natural elicitation works — but dataset design is critical.

## References

1. Shao et al. [Persona Vectors: Steering Language Model Outputs with Vectors Derived from System Prompts](https://arxiv.org/abs/2406.12094). 2024.
2. Experiment data: `experiments/persona_vectors_replication/`
3. Full methodology: `experiments/persona_vectors_replication/PLAN.md`
