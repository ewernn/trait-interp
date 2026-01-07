---
title: "OOD: Cross-Language Generalization"
preview: "Trait vectors are language-invariant: 90.6% cross-language accuracy for formality."
---

# OOD: Cross-Language Generalization

Trait vectors are language-invariant.

## Formality Experiments (2026-01-07)

### Setup

- **Trait:** Formality (formal vs casual language style)
- **Model:** Gemma-2-2B (base)
- **Languages:** English, Spanish, French, Chinese
- **Method:** Extract probe vectors per language, cross-validate classification and steering

### Classification Cross-Validation (Layer 12)

| Train / Test | English | Spanish | French | Chinese |
|--------------|---------|---------|--------|---------|
| **English**  | 100.0%* | 95.0%   | 100.0% | 80.0%   |
| **Spanish**  | 93.8%   | 95.0%*  | 94.4%  | 100.0%  |
| **French**   | 93.8%   | 90.0%   | 100.0%*| 90.0%   |
| **Chinese**  | 81.2%   | 75.0%   | 94.4%  | 100.0%* |

*\* = same-language baseline*

- **In-domain avg:** 98.8%
- **Cross-language avg:** 90.6%
- **Std dev:** 7.6%

### In-Domain Steering Results

| Language | Baseline | Best Δ | Coherence |
|----------|----------|--------|-----------|
| English  | 61.1     | +26.7  | 89%       |
| Spanish  | 24.2     | +61.9  | 79%       |
| French   | 54.5     | +32.2  | 74%       |
| Chinese  | 31.2     | +52.1  | 72%       |

### Key Findings

1. **Strong cross-language transfer:** 90.6% cross-language classification accuracy shows formality concepts are largely language-independent.

2. **Asymmetric transfer:** English→others transfers better (91.7% avg) than Chinese→others (83.5% avg), possibly due to training data distribution.

3. **All languages steerable:** Each variant achieves significant steering deltas with good coherence.

---

## Prior: Uncertainty Expression

### Setup

- **Trait:** Uncertainty expression (speculative vs factual questions)
- **Model:** Gemma-2-2B-IT
- **Languages:** English (111 scenarios), Spanish (50), French (50), Chinese (50)
- **Method:** Extract vectors per language, cross-validate on other languages' activations

### Results (Probe @ Layer 16)

| Train / Test | English | Spanish | French | Chinese |
|--------------|---------|---------|--------|---------|
| **English**  | 100.0%* | 99.0%   | 99.0%  | 100.0%  |
| **Spanish**  | 99.1%   | 100.0%* | 100.0% | 100.0%  |
| **French**   | 98.7%   | 100.0%  | 100.0%*| 100.0%  |
| **Chinese**  | 99.6%   | 100.0%  | 100.0% | 100.0%* |

*\* = same-language baseline*

- **Same-language avg:** 100.0%
- **Cross-language avg:** 99.6%
- **Drop:** 0.4%

### Key Findings

1. **Trait vectors are language-invariant:** A Chinese-trained vector achieves 99.6% on English data (and vice versa), despite Gemma not being explicitly multilingual.

2. **Representations are universal:** The uncertainty trait direction exists in the same location in activation space regardless of input language. This suggests trait representations are semantic, not surface-level.

---

## Implications

- Cross-language validation is a strong test for trait vector quality
- Vectors that fail cross-lang likely capture confounds (topic, style, language patterns)
- Different traits show different degrees of cross-language transfer (uncertainty: 99.6%, formality: 90.6%)
