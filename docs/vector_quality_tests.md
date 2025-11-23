# Vector Quality Evaluation Tests

Beyond cross-distribution accuracy, here are additional tests to evaluate how well extraction methods represent traits:

## 1. Steering Effectiveness Tests

### A. Magnitude Sweep
**What it tests:** How well the vector controls behavior at different strengths

```python
# Test vector at different magnitudes
for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
    steered_output = model(prompt, steering_vector=alpha * trait_vector)
    measure_trait_strength(steered_output)
```

**Expected behavior:**
- Good vectors: Smooth, monotonic increase in trait expression
- Bad vectors: Chaotic, non-monotonic, or saturates too quickly

### B. Steering Precision
**What it tests:** Can you control ONLY the target trait without side effects?

```python
# Measure multiple traits while steering one
steer_with_refusal_vector(prompt)
# Measure: refusal ↑, but formality, sentiment, etc. unchanged
```

**Metrics:**
- Target trait change
- Off-target trait changes (should be minimal)
- Output coherence/quality

### C. Steering Robustness
**What it tests:** Does steering work across diverse prompts?

```python
# Test on prompts from different domains
domains = ["medical", "legal", "casual_chat", "technical"]
for domain in domains:
    test_steering_effectiveness(trait_vector, domain)
```

## 2. Representation Quality Tests

### A. Cosine Similarity to Ideal Direction
**What it tests:** How aligned is the vector with the "true" trait direction?

```python
# Compare to multiple reference vectors
refs = [mean_diff_vec, probe_vec, gradient_vec, ica_vec]
similarities = [cosine_sim(vec, ref) for ref in refs]
# High agreement across methods = more reliable
```

### B. Orthogonality to Unrelated Traits
**What it tests:** Is the vector specific to the target trait?

```python
# Vector should be orthogonal to unrelated traits
unrelated = ["color_preference", "day_of_week", "random_noise"]
for trait in unrelated:
    similarity = cosine_sim(target_vec, trait_vec)
    # Should be near zero
```

### C. Consistency Across Layers
**What it tests:** Do adjacent layers agree on the representation?

```python
# Compare vector at layer L to layers L-1 and L+1
consistency = cosine_sim(vec_L, vec_L+1)
# High consistency = more stable representation
```

## 3. Interpretability Tests

### A. Top Activating Examples
**What it tests:** Do the examples with highest projection make semantic sense?

```python
# Project all examples onto the vector
scores = [dot(activation, vector) for activation in dataset]
top_examples = get_top_k(scores, k=20)
# Manually inspect: do they exhibit the trait?
```

### B. Intervention Causality
**What it tests:** Does the vector capture causal mechanisms?

```python
# Ablation: remove vector component from activations
ablated = activation - projection(activation, vector)
# Does this eliminate the trait behavior?
```

### C. Vector Arithmetic
**What it tests:** Do vectors compose semantically?

```python
# Test if vectors combine meaningfully
formal_positive = formal_vec + positive_vec
# Does this produce formal + positive text?

# Test if subtraction works
uncertain_not_negative = uncertain_vec - negative_vec
```

## 4. Generalization Tests (Beyond Cross-Distribution)

### A. Few-Shot Adaptation
**What it tests:** How well does the vector work with minimal examples?

```python
# Train on 10, 50, 100, 500 examples
for n in [10, 50, 100, 500]:
    vector = extract_vector(examples[:n])
    test_accuracy = evaluate(vector, test_set)
# Plot learning curve
```

### B. Adversarial Robustness
**What it tests:** Does the vector work on edge cases and adversarial examples?

```python
# Test on:
# - Paraphrased versions of training examples
# - Negated examples ("not uncertain" vs "uncertain")
# - Mixed-trait examples (uncertain + positive)
# - Minimal edits (change one word)
```

### C. Transfer to Other Models
**What it tests:** Is the trait representation model-specific or more universal?

```python
# Extract vector from Gemma-2B
vector_2b = extract(gemma_2b, trait)

# Test on Gemma-7B, google/gemma-2-2b-it, etc.
accuracy_7b = test(gemma_7b, vector_2b)
# Some transfer expected if concept is fundamental
```

## 5. Behavioral Tests

### A. Saturation Analysis
**What it tests:** When does the vector stop being effective?

```python
# Measure saturation point
for alpha in np.logspace(-2, 2, 50):
    trait_strength = measure(model(prompt, alpha * vector))
# Plot saturation curve
```

### B. Interaction Effects
**What it tests:** How does the vector interact with other interventions?

```python
# Test combinations
steer_with_multiple_vectors([refusal_vec, polite_vec])
# Are effects additive? Interfering? Synergistic?
```

### C. Prompt Sensitivity
**What it tests:** Does effectiveness vary by prompt characteristics?

```python
# Test on prompts with varying:
# - Length (short vs long)
# - Complexity (simple vs technical)
# - Domain (medical, legal, casual)
# - Ambiguity (clear vs ambiguous)
```

## 6. Statistical Tests

### A. Discriminative Power
**What it tests:** How well can the vector distinguish trait presence?

```python
# ROC curve analysis
scores_pos = [dot(act, vec) for act in positive_examples]
scores_neg = [dot(act, vec) for act in negative_examples]
auc = compute_auc(scores_pos, scores_neg)
# AUC closer to 1.0 = better discrimination
```

### B. Noise Robustness
**What it tests:** Is the vector stable under noise?

```python
# Add Gaussian noise to activations
for noise_level in [0.01, 0.05, 0.1, 0.5]:
    noisy_acts = activations + noise_level * randn()
    noisy_vec = extract_vector(noisy_acts)
    similarity = cosine_sim(noisy_vec, clean_vec)
# Should degrade gracefully
```

### C. Bootstrap Stability
**What it tests:** How sensitive is the vector to training set sampling?

```python
# Bootstrap resampling
vectors = []
for _ in range(100):
    sample = resample(training_data)
    vectors.append(extract_vector(sample))

# Compute confidence intervals
mean_vec = np.mean(vectors, axis=0)
std_vec = np.std(vectors, axis=0)
# Lower std = more stable
```

## 7. Comparative Benchmarks

### A. Human Agreement
**What it tests:** Do vector predictions match human judgments?

```python
# Human annotators rate examples
human_scores = get_human_ratings(examples)
model_scores = [dot(act, vec) for act in examples]
correlation = pearsonr(human_scores, model_scores)
# Higher correlation = better alignment
```

### B. Method Comparison Matrix
**What it tests:** Which method is best for which trait type?

| Metric | Mean Diff | Probe | Gradient | ICA |
|--------|-----------|-------|----------|-----|
| Cross-Dist Acc | X | X | X | X |
| Steering Smooth | X | X | X | X |
| Specificity | X | X | X | X |
| Interpretability | X | X | X | X |
| Robustness | X | X | X | X |

## 8. Practical Deployment Tests

### A. Inference Speed
**What it tests:** How fast can you apply the vector?

```python
import time
start = time.time()
for _ in range(1000):
    apply_steering(vector)
latency = (time.time() - start) / 1000
```

### B. Memory Footprint
**What it tests:** How much storage/memory does the vector require?

```python
# Compare:
# - Raw vector storage
# - Compressed representations
# - Sparse vs dense
```

### C. Batch Processing
**What it tests:** Does the vector work well in batched inference?

```python
# Test on batches of varying sizes
for batch_size in [1, 8, 32, 128]:
    throughput = measure_throughput(vector, batch_size)
```

## Recommended Test Suite

For comprehensive evaluation, run:

1. **Core Tests** (always):
   - Cross-distribution accuracy
   - Steering magnitude sweep
   - Top activating examples inspection

2. **Method Comparison** (when choosing method):
   - Compare all 4 methods on same trait
   - Focus on cross-dist, steering smoothness, specificity

3. **Production Readiness** (before deployment):
   - Adversarial robustness
   - Prompt sensitivity
   - Human agreement

4. **Research Insights** (for understanding):
   - Vector arithmetic
   - Orthogonality analysis
   - Bootstrap stability

## Implementation Priority

**Phase 1: Quick Diagnostics** (30 min)
- Magnitude sweep visualization
- Top examples inspection
- Basic cross-dist test

**Phase 2: Comprehensive Eval** (2-4 hours)
- All 4 methods comparison
- Robustness tests
- Statistical analysis

**Phase 3: Production Tests** (1 day)
- Human evaluation
- Edge case testing
- Performance benchmarking
