---
title: "OOD: Cross-Topic Generalization"
preview: "Trait vectors generalize across domains: 91.4% cross-topic accuracy."
---

TODO: Verify with current pipeline (data may be in old trait-stuff/experiments/ folder)

# OOD: Cross-Topic Generalization

Trait vectors generalize across topics.

## Setup

- **Trait:** Uncertainty expression (speculative vs factual questions)
- **Model:** Gemma-2-2B-IT
- **Topics:** Science (101 scenarios), Coding (100), History (100), Creative (100)
- **Method:** Extract vectors per topic, cross-validate on other topics' activations

## Results (Probe @ Layer 16)

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

## Key Findings

1. **Vectors generalize across topics:** Science-trained vector achieves 90-96% on other topics.

2. **One outlier:** History→Coding at 74.5%, possibly due to structural differences in how uncertainty manifests in counterfactual vs technical questions.

3. **Natural elicitation works:** Vectors capture "uncertainty" not "uncertainty-about-science".

## Data Location

Original data likely in: `trait-stuff/experiments/gemma-2-2b-it-persona-vectors/extraction/cross-topic/`
