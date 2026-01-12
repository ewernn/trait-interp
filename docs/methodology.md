---
references:
  platonic:
    authors: "Huh et al."
    year: "2024"
    title: "The Platonic Representation Hypothesis"
    url: "https://arxiv.org/abs/2405.07987"
---

# Methodology

How we extract trait vectors and use them for monitoring.

---

## Design Choices

### Natural elicitation over instruction-following

We design scenarios where behavior emerges naturally from context, rather than instructing the model to "act certain" or "be helpful." Instruction-following confounds what we're trying to measure.

Why this matters beyond methodology: AI models tend to converge on similar internal representations [@platonic]. Combined with the future of continual learning—where many finetune steps still operate within the constraints of the underlying manifold—this motivates extracting clean, shared representations independent of finetuning confounds.

### Layer intuition

Early layers handle syntax, middle layers do reasoning, late layers format output. We typically extract from middle layers where behavioral decisions happen.

---

## 1. Elicitation

We create contrasting scenarios that naturally elicit the target behavior.

**Positive scenario** (trait present):
> You are a customer service agent. A customer asks how to return a clearly fraudulent item they damaged themselves.

**Negative scenario** (trait absent):
> You are a customer service agent. A customer asks how to return a defective item within the return window.

The model responds differently not because we told it to, but because the situations genuinely differ.

:::placeholder "Inline example: positive vs negative scenario for refusal trait":::

:::dataset /datasets/traits/chirp/refusal/positive.txt "View positive scenarios":::

:::dataset /datasets/traits/chirp/refusal/negative.txt "View negative scenarios":::

---

## 2. Extraction

We generate completions for each scenario, capture activations at response tokens, then find the direction that separates positive from negative.

### Quality filtering

Not all completions cleanly express the target trait. We use an LLM judge to score each response, keeping only high-quality examples.

:::placeholder "Judge prompt used for quality filtering":::

:::placeholder "Sample of scored responses with quality ratings":::

### Vector extraction

We extract at multiple layers using different methods:
- **mean_diff**: centroid difference between positive and negative activations
- **probe**: logistic regression weights
- **gradient**: optimization to maximize separation

:::placeholder "Layer × method heatmap showing extraction quality":::

---

## 3. Validation

Steering tests whether the vector actually controls behavior. If adding the vector increases the trait and subtracting decreases it, we have evidence of a causal direction.

### Before/after comparison

Same prompt, different steering coefficients:

:::placeholder "Steering comparison: prompt with coef=-1, coef=0, coef=+1":::

:::placeholder "More steering examples table":::

### Steering sweep

We test across layers and coefficients to find the sweet spot.

:::placeholder "Steering sweep figure: layer × coefficient heatmap":::

---

## 4. Monitoring

At inference time, we project each token's hidden state onto trait vectors to see the model's thinking evolve.

### Benign vs harmful comparison

The refusal vector shows clear differences:

:::placeholder "Per-token projection chart: benign prompt (flat) vs harmful prompt (spike)":::

When the model encounters a harmful request, refusal activates early and stays high. On benign requests, it remains near zero.

:::placeholder "Side-by-side token trajectories for refusal trait":::
