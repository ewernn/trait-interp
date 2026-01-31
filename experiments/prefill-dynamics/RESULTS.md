# Prefill Activation Dynamics: Results

## Summary

| Metric | Human Text | Model Text | Diff | Effect Size |
|--------|------------|------------|------|-------------|
| CE Loss (perplexity) | 2.9902 | 1.4515 | 1.5387 | - |
| Smoothness (delta norm) | 193.8005 | 179.5203 | 14.2803 | d=1.487 |

**Statistical significance**: t=10.41, p=5.27e-14

## Perplexity-Smoothness Correlation

| Comparison | r | p-value | Interpretation |
|------------|---|---------|----------------|
| Pooled (all samples) | 0.648 | 3.09e-13 | Significant |
| Differences only | 0.362 | 9.76e-03 | Significant |

## Interpretation

**Model text produces smoother activation trajectories than human text.**

This correlates with perplexity: lower perplexity (less surprising) -> smoother activations. **Hypothesis supported.**

## Per-Layer Analysis

| Layer | Human Smooth | Model Smooth | Diff | Cohen's d | p-value |
|-------|--------------|--------------|------|-----------|---------|
| 0 | 87.0174 | 82.3396 | 4.6778 | 0.674 | 2.02e-05 |
| 1 | 87.4621 | 82.7205 | 4.7416 | 0.644 | 4.11e-05 |
| 2 | 89.2842 | 83.5008 | 5.7833 | 0.727 | 5.65e-06 |
| 3 | 87.0082 | 80.3948 | 6.6134 | 0.976 | 1.20e-08 |
| 4 | 87.4360 | 79.6645 | 7.7715 | 1.285 | 6.03e-12 |
| 5 | 89.4333 | 80.7169 | 8.7164 | 1.399 | 3.99e-13 |
| 6 | 91.4536 | 82.7869 | 8.6667 | 1.492 | 4.62e-14 |
| 7 | 92.2008 | 83.5768 | 8.6239 | 1.508 | 3.20e-14 |
| 8 | 97.1233 | 88.9359 | 8.1874 | 1.464 | 8.85e-14 |
| 9 | 104.6861 | 96.0074 | 8.6787 | 1.411 | 3.03e-13 |
| 10 | 112.2858 | 103.5003 | 8.7855 | 1.405 | 3.53e-13 |
| 11 | 120.5860 | 110.6289 | 9.9571 | 1.446 | 1.35e-13 |
| 12 | 123.9395 | 114.2432 | 9.6962 | 1.426 | 2.15e-13 |
| 13 | 136.1317 | 123.7409 | 12.3908 | 1.658 | 1.13e-15 |
| 14 | 146.1660 | 134.3088 | 11.8573 | 1.444 | 1.41e-13 |
| 15 | 158.9039 | 146.7453 | 12.1587 | 1.304 | 3.84e-12 |
| 16 | 177.6668 | 161.6430 | 16.0238 | 1.478 | 6.46e-14 |
| 17 | 188.1824 | 170.9903 | 17.1920 | 1.477 | 6.55e-14 |
| 18 | 216.6794 | 197.3914 | 19.2880 | 1.469 | 7.92e-14 |
| 19 | 244.0949 | 220.6804 | 23.4145 | 1.497 | 4.13e-14 |
| 20 | 280.6046 | 252.6329 | 27.9718 | 1.633 | 1.99e-15 |
| 21 | 331.0286 | 296.5927 | 34.4359 | 1.633 | 1.97e-15 |
| 22 | 377.7314 | 345.0620 | 32.6694 | 1.574 | 7.30e-15 |
| 23 | 423.1343 | 386.9340 | 36.2003 | 1.415 | 2.78e-13 |
| 24 | 482.1927 | 454.2958 | 27.8969 | 0.989 | 8.72e-09 |
| 25 | 606.3804 | 607.4928 | -1.1124 | -0.020 | 8.88e-01 |

## Files

- Data: `experiments/prefill-dynamics/data/continuations.json`
- Activations: `experiments/prefill-dynamics/activations/{human,model}/*.pt`
- Perplexity: `experiments/prefill-dynamics/analysis/perplexity.json`
- Metrics: `experiments/prefill-dynamics/analysis/activation_metrics.json`
- Correlation: `experiments/prefill-dynamics/analysis/correlation.json`
