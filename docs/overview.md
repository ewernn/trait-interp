# Trait Vector Extraction and Monitoring

How we extract behavioral trait vectors from language models and monitor them token-by-token during generation.

---

## The Problem

Language models don't just produce text—they make decisions at every token. When a model generates a response to "How do I pick a lock?", it decides whether to refuse, how confident to sound, whether to hedge. These decisions happen in the model's internal activations, invisible to us.

**What if we could see those decisions as they happen?**

This project extracts *trait vectors*—directions in activation space that correspond to behavioral traits like refusal, uncertainty, or sycophancy. By projecting each token's hidden state onto these vectors, we can watch the model's "thinking" evolve in real-time.

---

## Key Concepts

- **Trait Vector**: A direction in the model's high-dimensional activation space (e.g., 2304-dim for Gemma 2B) that represents a behavioral pattern like "Refusal" or "Uncertainty."

- **Residual Stream**: The information flow that accumulates across the model's layers, acting as the running computational state.

- **Projection**: Measuring how much a hidden state aligns with a trait vector. Positive = expressing the trait, negative = suppressing it.

- **Commitment Point**: The token at which a trait's acceleration drops to near-zero, indicating the model has "locked in" a decision.

---

## Natural Elicitation

The key insight: **don't tell the model what to do**. Instead, give it scenarios where it will *naturally* exhibit the trait.

For a trait like **uncertainty**, we create two sets of prompts:

- **Positive (uncertain):** "What will the stock market do next year?" → model hedges naturally
- **Negative (confident):** "What is 2 + 2?" → model answers confidently

No instructions, no meta-prompts. The model's behavior differs because the *questions* differ, not because we told it to behave differently.

**Why this matters:**
- Avoids instruction-following confounds (model might just be compliant, not genuinely uncertain)
- Captures the trait as it naturally occurs in the wild
- Produces vectors that generalize to new contexts

We use 100+ prompts per side to average out topic-specific noise.

---

## The Extraction Pipeline

### Stage 1: Generate Responses

For each scenario, generate a response without any system prompt or instructions:

```
Prompt: "What causes earthquakes?"
Response: "Earthquakes are caused by the movement of tectonic plates..."
```

The model responds naturally, exhibiting (or not exhibiting) the trait based solely on the prompt.

### Stage 2: Extract Activations

Capture the hidden state at every layer during generation. We average across response tokens to get one vector per example:

$$\mathbf{a}_i^{(\ell)} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{h}_t^{(\ell)}$$

where $\mathbf{h}_t^{(\ell)}$ is the hidden state at layer $\ell$, token $t$, and $T$ is the response length.

This gives us activation matrices $\mathbf{A}^+$ (positive examples) and $\mathbf{A}^-$ (negative examples).

### Stage 3: Extract Vectors

Apply an extraction method to find the direction that best separates positive from negative examples.

---

## Extraction Methods

Given positive activations $\mathbf{A}^+ \in \mathbb{R}^{n \times d}$ and negative activations $\mathbf{A}^- \in \mathbb{R}^{m \times d}$, we extract a trait vector $\mathbf{v} \in \mathbb{R}^d$.

### Mean Difference

The simplest approach—subtract the centroids:

$$\mathbf{v} = \bar{\mathbf{a}}^+ - \bar{\mathbf{a}}^-$$

Fast and interpretable, but ignores variance and can be swayed by outliers. Produces unnormalized vectors (typical norm: 50–100).

### Linear Probe

Train a logistic regression classifier to distinguish positive from negative:

$$P(y=1 \mid \mathbf{a}) = \sigma(\mathbf{w}^\top \mathbf{a} + b)$$

The weight vector $\mathbf{w}$ becomes our trait vector. L2 regularization keeps the norm reasonable (~1–5). This finds the optimal linear separator, accounting for the full distribution rather than just means.

### Gradient Optimization

Directly optimize for maximum separation:

$$\mathbf{v}^* = \arg\max_{\|\mathbf{v}\|=1} \left[ \bar{\mathbf{a}}^{+\top} \mathbf{v} - \bar{\mathbf{a}}^{-\top} \mathbf{v} \right]$$

The unit-norm constraint forces the optimization to find a *direction* rather than inflating magnitude. Empirically produces the most generalizable vectors for subtle traits.

### Which Method to Use?

| Trait Type | Best Method | Why |
|------------|-------------|-----|
| High separability (e.g., sentiment) | Probe | Clear signal, linear separation works well |
| Subtle traits (e.g., uncertainty) | Gradient | Unit normalization finds robust directions |

---

## Normalization for Evaluation

The residual stream magnitude grows with depth. In Gemma 2B:

| Layer | Avg Norm | Relative |
|-------|----------|----------|
| 0 | 88 | 1.0× |
| 12 | 281 | 3.2× |
| 25 | 1362 | 15.5× |

If we use raw dot products, late-layer projections are larger simply because activations are larger—not because the vector is better.

To compare fairly across layers, we use **cosine similarity**:

$$\text{score} = \frac{\mathbf{a}^\top \mathbf{v}}{\|\mathbf{a}\| \|\mathbf{v}\|}$$

This measures *direction*, not magnitude. Scores range $[-1, 1]$ regardless of layer.

---

## Validation

We evaluate on held-out validation data (20% of examples, never seen during extraction).

### Accuracy

Classification accuracy using the midpoint between class means as the decision threshold. Positive examples should project above threshold, negative below.

### Effect Size (Cohen's d)

Separation measured in standard deviations:

$$d = \frac{|\bar{s}^+ - \bar{s}^-|}{\sqrt{(\sigma^{+2} + \sigma^{-2})/2}}$$

Guidelines: $d > 0.8$ is large, $d > 2.0$ is very strong separation.

### What Good Looks Like

- **Accuracy > 90%** on held-out data
- **Effect size > 2.0** (classes are well-separated)
- **Generalizes** to prompts outside training distribution

---

## Layer Selection

Empirically, middle layers work best for behavioral traits:

- **Early layers (0–5):** Surface features, syntax—too shallow for behavioral traits
- **Middle layers (6–16):** Semantic representations—best generalization
- **Late layers (21–25):** Task-specific processing—can overfit to training distribution

For Gemma 2B (26 layers), optimal layer varies by trait. Use `extraction_evaluation.py` to find the best layer for each trait empirically.

---

## Inference Monitoring

During generation, we project each token's hidden state onto trait vectors:

```python
score = (hidden_state @ trait_vector) / (norm(hidden_state) * norm(trait_vector))
```

- **Positive score** → model expressing the trait
- **Negative score** → model suppressing the trait
- **Score magnitude** → how strongly

### Dynamics Analysis

Beyond raw projections, we compute:

- **Velocity**: Rate of change of trait expression (first derivative)
- **Acceleration**: How quickly velocity changes (second derivative)
- **Commitment Point**: Token where acceleration drops—model has "locked in"
- **Persistence**: How long the trait stays elevated after peak

These dynamics reveal *when* the model decides, not just *what* it decides.

---

## What You Can Do With This

### Early Warning

Detect dangerous patterns before generation completes. High Refusal + dropping Uncertainty at token 5 might predict the model is about to refuse—or about to comply with something it shouldn't.

### Behavioral Debugging

When a model misbehaves, trace *where* in generation it went wrong. Did refusal spike too late? Did sycophancy override safety?

### Steering

Add or subtract trait vectors during generation to modify behavior. Increase refusal to make the model more conservative, decrease sycophancy to make it more honest.

### Interpretability Research

Study how traits interact, when they crystallize, and what attention patterns create them.

---

## Summary

1. **Natural elicitation** gives us clean training data without instruction-following confounds
2. **Extraction methods** find directions that separate positive from negative examples
3. **Validation on held-out data** ensures vectors capture genuine traits, not artifacts
4. **Per-token monitoring** reveals the model's decision-making as it happens
5. **Dynamics analysis** shows when decisions crystallize and how they persist

The result: a window into what the model is "thinking" at every token.
