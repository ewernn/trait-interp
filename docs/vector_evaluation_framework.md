# Trait Vector Evaluation Framework

## Overview

Determining which trait vector is "best" requires evaluating across multiple dimensions. The optimal choice depends on your specific use case: real-time monitoring, steering, analysis, or research.

## Core Evaluation Axes

### 1. **Separation Metrics** (Primary Performance)

**What it measures:** How well the vector distinguishes positive from negative examples

**Key metrics:**
- **Separation Score**: Distance between positive and negative centroids (0-100 scale)
  - Excellent: > 80
  - Good: 40-80
  - Poor: < 40
- **Classification Accuracy**: How well a linear classifier using the vector performs
  - Excellent: > 95%
  - Good: 85-95%
  - Poor: < 85%
- **Cross-Validation Accuracy**: Generalization within training distribution (5-fold CV)
- **AUC Score**: Area under ROC curve (robustness metric)

**When it matters most:** Production systems, real-time monitoring

### 2. **Cross-Distribution Testing** (Generalization)

**What it measures:** Whether the vector captures the true trait or just training artifacts

**Key tests:**
- **Natural → Instruction**: Natural vector tested on instruction-based prompts
- **Instruction → Natural**: Instruction vector tested on natural prompts
- **Consistency Score**: How similar performance is in both directions

**Red flags:**
- Instruction vectors scoring NEGATIVE on natural harmful prompts (polarity inversion)
- Large performance drops (>30%) when switching distributions
- Inconsistent behavior across test types

**When it matters most:** Deploying to real-world, diverse inputs

### 3. **Polarity Validation** (Correctness)

**What it measures:** Whether the vector points in the right direction

**Key checks:**
- **Polarity Correct**: Does positive > negative for expected direction?
- **Effect Size** (Cohen's d): How strong is the polarity difference?
  - Large: > 0.8
  - Medium: 0.5-0.8
  - Small: < 0.5
- **Statistical Confidence**: p-value from t-test

**Critical for:** Refusal, sycophancy, and other traits with clear expected direction

### 4. **Temporal Stability** (Dynamics Quality)

**What it measures:** How consistent projections are across tokens

**Key metrics:**
- **Trajectory Smoothness**: Low noise/jitter in per-token projections
- **Commitment Clarity**: Clear crystallization points (via 2nd derivative)
- **Persistence**: How long trait expression sustains after peak

**When it matters most:** Analyzing generation dynamics, understanding model "thinking"

### 5. **Robustness Metrics** (Reliability)

**What it measures:** Performance under perturbation

**Key tests:**
- **Bootstrap Stability**: Standard deviation across resampled training data
  - Excellent: < 5% variation
  - Good: 5-10%
  - Poor: > 10%
- **Noise Robustness**: Performance with 10% Gaussian noise added
- **Subsample Stability**: Performance with only 50% of training data

**When it matters most:** Production deployment, safety-critical applications

### 6. **Statistical Properties** (Vector Quality)

**What it measures:** Mathematical properties of the vector itself

**Key properties:**
- **Vector Norm**: Should be reasonable (15-40 for normalized vectors)
  - Too small: Weak signal
  - Too large: Overfitting
- **Sparsity**: Fraction of near-zero components
  - High sparsity → More interpretable
  - Low sparsity → More distributed representation
- **Effective Rank**: Dimensionality of information
- **Kurtosis**: Outlier detection (high kurtosis = few dominant features)

**When it matters most:** Interpretability research, feature analysis

### 7. **Layer Selection** (Architecture Alignment)

**What it measures:** Which layer best captures the trait

**Key patterns:**
- **Early layers (0-5)**: Surface features, syntax
- **Middle layers (6-20)**: Semantic meaning, concepts
- **Late layers (21-25)**: Task-specific, instruction-following

**Best practices:**
- High-separability traits: Middle layers with Probe method
- Low-separability traits: Try Gradient/ICA methods
- Cross-distribution: Avoid late layers (overfit to instructions)

**Layer consistency checks:**
- **Adjacent Layer Similarity**: Should be > 0.7 (smooth evolution)
- **Peak Layer**: Which layer has maximum separation?
- **Evolution Smoothness**: How gradually do metrics change across layers?

### 8. **Orthogonality** (Independence)

**What it measures:** How independent this trait is from others

**Key metrics:**
- **Mean Orthogonality**: Average cosine distance to other traits
  - Good: > 0.8 (mostly independent)
  - Warning: < 0.5 (high correlation)
- **Max Correlation**: Highest correlation with any other trait
- **Independence Score**: Composite metric

**Why it matters:** Correlated traits may be measuring the same underlying feature

### 9. **Interpretability Proxies**

**What it measures:** How understandable the vector is

**Key indicators:**
- **Top-k Mass**: Concentration in top 5% of components
  - High concentration → More interpretable
- **SAE Feature Alignment**: Correlation with known SAE features
- **Attention Head Specialization**: Specific heads tracking the trait
- **Logit Lens Clarity**: How clearly vector predicts specific tokens

**When it matters most:** Mechanistic interpretability research

## Composite Scoring

### Overall Score Formula

```python
overall_score = (
    separation_metrics * 0.4 +      # Performance is most important
    reliability_metrics * 0.2 +      # Stability matters
    cross_distribution * 0.2 +       # Generalization is critical
    orthogonality * 0.1 +           # Independence from other traits
    polarity_correctness * 0.1      # Must point right direction
)
```

### Use-Case-Specific Weightings

**For Production Monitoring:**
```python
score = separation * 0.5 + reliability * 0.3 + polarity * 0.2
```

**For Research/Analysis:**
```python
score = interpretability * 0.3 + orthogonality * 0.3 + separation * 0.4
```

**For Steering/Control:**
```python
score = cross_distribution * 0.4 + robustness * 0.3 + separation * 0.3
```

## Decision Tree for Vector Selection

```
1. Is polarity correct?
   NO → Reject this vector
   YES → Continue

2. Is separation > 40?
   NO → Try different method/layer
   YES → Continue

3. Is cross-distribution consistency > 0.7?
   NO → Warning: May not generalize
   YES → Good candidate

4. Is bootstrap stability < 10%?
   NO → Warning: Unstable
   YES → Production-ready

5. Is orthogonality > 0.7?
   NO → Check for redundancy with other traits
   YES → Independent trait
```

## Method Selection Guidelines

### Mean Difference
- **Best for:** Simple, interpretable baselines
- **Strengths:** No hyperparameters, always produces a vector
- **Weaknesses:** No regularization, can overfit
- **Typical performance:** 60-80% accuracy

### Probe (Logistic Regression)
- **Best for:** High-separability traits (>80% natural separation)
- **Strengths:** L2 regularization, probabilistic outputs
- **Weaknesses:** Linear only, requires sufficient examples
- **Typical performance:** 85-95% accuracy

### ICA (Independent Component Analysis)
- **Best for:** Separating mixed/entangled traits
- **Strengths:** Finds maximally independent directions
- **Weaknesses:** Computationally expensive, may be unstable
- **Typical performance:** Variable, excels on entangled traits

### Gradient Optimization
- **Best for:** Low-separability traits where linear methods fail
- **Strengths:** Can find non-obvious directions, very high separation
- **Weaknesses:** Risk of overfitting, sensitive to initialization
- **Typical performance:** 90-98% accuracy (but check generalization!)

## Layer Selection Patterns

### Gemma 2B (26 layers)

**Optimal layers by trait type:**
- **Factual/Retrieval**: Layers 14-18 (middle-late)
- **Behavioral (refusal, sycophancy)**: Layers 10-16 (middle)
- **Stylistic (formality, tone)**: Layers 6-12 (early-middle)
- **Cognitive (uncertainty, confidence)**: Layers 12-18 (middle)

**Cross-distribution performance by layer:**
- Layers 6-16: Best generalization
- Layers 0-5: Too surface-level
- Layers 20-25: Overfit to training distribution

### General Heuristics

1. **Start with layer = model.num_layers // 2** (middle layer)
2. **Search ±4 layers** from middle
3. **Validate on held-out data** before choosing
4. **Check cross-distribution** if deploying broadly

## Evaluation Workflow

### Quick Evaluation (5 minutes)
```bash
# Test single trait, best methods/layers only
python analysis/evaluate_vector_quality.py \
    --experiment gemma_2b_cognitive_nov21 \
    --trait cognitive_state/uncertainty_expression \
    --methods probe,mean_diff \
    --layers 14,15,16
```

### Comprehensive Evaluation (30 minutes)
```bash
# Test all methods, all layers
python analysis/evaluate_vector_quality.py \
    --experiment gemma_2b_cognitive_nov21 \
    --trait cognitive_state/uncertainty_expression \
    --output results/uncertainty_eval.json
```

### Experiment-Wide Evaluation (2-3 hours)
```bash
# Evaluate all traits in experiment
python analysis/evaluate_vector_quality.py \
    --experiment gemma_2b_cognitive_nov21 \
    --output results/full_evaluation.json
```

## Red Flags to Watch For

1. **Polarity Inversion**: Vector points opposite to expected direction
2. **Layer Instability**: Dramatic performance changes between adjacent layers
3. **Method Disagreement**: Different methods produce orthogonal vectors
4. **Norm Explosion**: Vector norm > 100 (likely overfitting)
5. **Zero Orthogonality**: Perfect correlation with another trait
6. **Distribution Brittleness**: >50% performance drop on different distribution

## Best Practices Summary

1. **Always validate polarity first** - Wrong direction = unusable
2. **Prefer middle layers** (6-20) for best generalization
3. **Use Probe for high-separability** traits (>80% natural separation)
4. **Use Gradient/ICA for low-separability** traits
5. **Test cross-distribution** before deployment
6. **Check orthogonality** to avoid redundant traits
7. **Bootstrap validation** for production systems
8. **Save top-3 vectors** per trait (redundancy is good)

## Example Results Interpretation

```json
{
  "trait": "cognitive_state/uncertainty_expression",
  "best_overall": {
    "method": "probe",
    "layer": 16,
    "overall_score": 87.3,
    "separation_score": 76.5,
    "train_accuracy": 0.93,
    "cross_dist_consistency": 0.85,
    "polarity_correct": true
  },
  "warnings": [],
  "recommendation": "Production ready - high scores across all metrics"
}
```

**Good result:**
- Separation > 70 ✓
- Accuracy > 90% ✓
- Cross-distribution > 0.8 ✓
- Polarity correct ✓

**Concerning result:**
```json
{
  "warnings": [
    "Low cross-distribution consistency (0.45)",
    "High correlation with 'confidence' trait (0.89)",
    "Bootstrap stability > 15%"
  ],
  "recommendation": "Use with caution - may not generalize well"
}
```

## Future Extensions

1. **Steering Validation**: Actually add vector during generation
2. **Adversarial Testing**: Specifically crafted inputs to break vectors
3. **Compositional Testing**: How do traits combine/interfere?
4. **Temporal Analysis**: How do vectors evolve during training?
5. **Cross-Model Transfer**: Do vectors work on other models?