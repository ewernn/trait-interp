---
title: "OOD: Formality Generalization"
preview: "Formality vectors generalize across languages (90.6%) and topics (79.9%)."
---

# OOD: Formality Generalization

Testing whether formality trait vectors generalize out-of-distribution across languages and topics.

## Setup

- **Trait:** Formality (formal vs casual language style)
- **Model:** Gemma-2-2B (base for extraction, IT for steering)
- **Cross-language:** English, Spanish, French, Chinese
- **Cross-topic:** Business, Academic, Social, Technical
- **Methods:** Classification cross-validation, steering cross-validation

---

## Cross-Language Results

### Classification (Layer 12)

| Train \ Test | English | Spanish | French | Chinese |
|--------------|---------|---------|--------|---------|
| **English**  | 100.0%* | 95.0%   | 100.0% | 80.0%   |
| **Spanish**  | 93.8%   | 95.0%*  | 94.4%  | 100.0%  |
| **French**   | 93.8%   | 90.0%   | 100.0%*| 90.0%   |
| **Chinese**  | 81.2%   | 75.0%   | 94.4%  | 100.0%* |

- **In-domain:** 98.8%
- **Cross-domain:** 90.6% (±7.6%)

### Steering (English → Others)

| Target  | Baseline | Steered | Delta  | Coherence |
|---------|----------|---------|--------|-----------|
| Spanish | 29.9     | 83.7    | +53.8  | 84%       |
| French  | 50.9     | 75.2    | +24.4  | 70%       |
| Chinese | 52.2     | 82.4    | +30.3  | 65%       |
| **Avg** |          |         | **+36.1** | **73%** |

### Key Findings

1. **Strong classification transfer:** 90.6% cross-language accuracy shows formality is largely language-independent in representation space.

2. **Asymmetric transfer:** English→others (91.7%) > Chinese→others (83.5%), likely due to training data distribution.

3. **Steering coherence issues:** English vector on French/Chinese degrades coherence (65-70%), possibly pushing toward English-style patterns.

---

## Cross-Topic Results

### Classification (Layer 12)

| Train \ Test | Business | Academic | Social | Technical |
|--------------|----------|----------|--------|-----------|
| **Business** | 90.9%*   | 87.5%    | 100.0% | 90.0%     |
| **Academic** | 68.2%    | 87.5%*   | 70.0%  | 50.0%     |
| **Social**   | 90.9%    | 87.5%    | 100.0%*| 70.0%     |
| **Technical**| 72.7%    | 87.5%    | 85.0%  | 90.0%*    |

- **In-domain:** 92.1%
- **Cross-domain:** 79.9% (±13.2%)

### Steering (Source → Others)

| Source → | Business | Academic | Social | Technical | Avg |
|----------|----------|----------|--------|-----------|-----|
| **Business** | — | +55.9 | +35.0 | +44.3 | +45.1 |
| **Academic** | +46.1 | — | +4.9* | +44.4 | +31.8 |
| **Social** | +39.4 | +50.4 | — | +40.0 | +43.3 |

*Social as target had ceiling effect (baseline ~82) until questions were rewritten to be more casual (baseline→47).

### Key Findings

1. **Moderate classification transfer:** 79.9% cross-topic, with academic transferring poorly (62.7% avg).

2. **Good steering transfer:** All topic vectors achieve +32-45 avg delta on other topics with 86%+ coherence.

3. **Classification ≠ steering:** Academic has poor classification transfer but decent steering transfer. Different measures.

4. **Business generalizes best:** 92.5% classification, +45.1 steering delta on other topics.

---

## Classification vs Steering Transfer

| Dimension | Classification | Steering |
|-----------|---------------|----------|
| Cross-language | 90.6% | 73% coherence (issues) |
| Cross-topic | 79.9% | 86%+ coherence (clean) |

**Key insight:** Classification measures whether the linear direction separates data; steering measures whether pushing along that direction changes behavior coherently. They don't always correlate.

Cross-topic steering transfers more cleanly than cross-language, despite worse classification numbers. The English vector can *classify* other languages' formality but when *steering*, it pushes toward English patterns that degrade coherence.

---

## Scripts

- `analysis/ood/cross_variant_evaluation.py` - Classification cross-validation
- `analysis/ood/cross_domain_steering.py` - Steering cross-validation

## Data

- `datasets/traits/formality_variations/{english,spanish,french,chinese}/` - Cross-language
- `datasets/traits/formality_variations/{business,academic,social,technical}/` - Cross-topic
- `datasets/traits/formality_variations/general/` - Combined (all topics)
