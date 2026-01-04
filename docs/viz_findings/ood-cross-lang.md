---
title: "OOD: Cross-Language Generalization"
preview: "Trait vectors are language-invariant: 99.6% cross-language accuracy."
---

# OOD: Cross-Language Generalization

Trait vectors are language-invariant.

## Setup

- **Trait:** Uncertainty expression (speculative vs factual questions)
- **Model:** Gemma-2-2B-IT
- **Languages:** English (111 scenarios), Spanish (50), French (50), Chinese (50)
- **Method:** Extract vectors per language, cross-validate on other languages' activations

## Results (Probe @ Layer 16)

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

## Key Findings

1. **Trait vectors are language-invariant:** A Chinese-trained vector achieves 99.6% on English data (and vice versa), despite Gemma not being explicitly multilingual.

2. **Representations are universal:** The uncertainty trait direction exists in the same location in activation space regardless of input language. This suggests trait representations are semantic, not surface-level.

## Implications

- Cross-language validation is a strong test for trait vector quality
- Vectors that fail cross-lang likely capture confounds (topic, style, language patterns)
