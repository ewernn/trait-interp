---
title: "OOD: Cross-Topic Generalization"
preview: "Trait vectors generalize across domains: 79.9% cross-topic accuracy for formality."
---

# OOD: Cross-Topic Generalization

Trait vectors generalize across topics.

## Formality Experiments (2026-01-07)

### Setup

- **Trait:** Formality (formal vs casual language style)
- **Model:** Gemma-2-2B (base)
- **Topics:** Business, Academic, Social, Technical
- **Method:** Extract probe vectors per topic, cross-validate classification and steering

### Classification Cross-Validation (Layer 12)

| Train / Test | Business | Academic | Social | Technical |
|--------------|----------|----------|--------|-----------|
| **Business** | 90.9%*   | 87.5%    | 100.0% | 90.0%     |
| **Academic** | 68.2%    | 87.5%*   | 70.0%  | 50.0%     |
| **Social**   | 90.9%    | 87.5%    | 100.0%*| 70.0%     |
| **Technical**| 72.7%    | 87.5%    | 85.0%  | 90.0%*    |

*\* = same-topic baseline*

- **In-domain avg:** 92.1%
- **Cross-topic avg:** 79.9%
- **Std dev:** 13.2%

### In-Domain Steering Results

| Topic     | Baseline | Best Δ | Coherence |
|-----------|----------|--------|-----------|
| Business  | 54.0     | +31.4  | 84%       |
| Academic  | 35.6     | +50.7  | 81%       |
| Social    | 82.7     | +5.2   | 86%       |
| Technical | 56.5     | +28.4  | 93%       |

### Key Findings

1. **Moderate cross-topic transfer:** 79.9% cross-topic accuracy shows formality concepts partially transfer across domains, though less than cross-language (90.6%).

2. **Topic-specific formality:** Academic formality transfers poorly to other domains (62.7% avg), suggesting academic formality has unique characteristics.

3. **High social baseline:** Social topic starts at 82.7 formality baseline, leaving less room for steering improvement (+5.2).

4. **Business generalizes best:** Business→others achieves 92.5% avg accuracy, suggesting business formality captures general formal language patterns.

---

## Prior: Uncertainty Expression

### Setup

- **Trait:** Uncertainty expression (speculative vs factual questions)
- **Model:** Gemma-2-2B-IT
- **Topics:** Science (101 scenarios), Coding (100), History (100), Creative (100)
- **Method:** Extract vectors per topic, cross-validate on other topics' activations

### Results (Probe @ Layer 16)

| Train / Test | Science | Coding | History | Creative |
|--------------|---------|--------|---------|----------|
| **Science**  | 100.0%* | 90.0%  | 96.0%   | 96.0%    |
| **Coding**   | 90.0%   | 100.0%*| 91.0%   | 96.0%    |
| **History**  | 89.6%   | 74.5%  | 100.0%* | 98.5%    |
| **Creative** | 91.0%   | 87.0%  | 97.0%   | 100.0%*  |

*\* = same-topic baseline*

- **Same-topic avg:** 100.0%
- **Cross-topic avg:** 91.4%
- **Drop:** 8.6%

### Key Findings

1. **Vectors generalize across topics:** Science-trained vector achieves 90-96% on other topics.

2. **One outlier:** History→Coding at 74.5%, possibly due to structural differences in how uncertainty manifests in counterfactual vs technical questions.

3. **Natural elicitation works:** Vectors capture "uncertainty" not "uncertainty-about-science".

---

## Implications

- Cross-topic transfer varies by trait: uncertainty (91.4%) > formality (79.9%)
- Some topic-specific vectors capture domain-specific concepts that don't generalize
- Business/Science domains tend to produce more generalizable formality/uncertainty vectors
