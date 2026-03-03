# Probe vs LLM Judge Correlation: Tonal Trait Detection

Validating that activation-space probe deltas predict perceptible behavioral change in LoRA outputs, as scored by an LLM judge.

## The Question

When probe_monitor reports "angry_register is increasing during LoRA training," does the LoRA actually produce angrier text? More precisely: does `probe(lora_activations(clean_text))` correlate with `judge(lora_generated_text)`?

## Method

**Probe** (activation-based): Prefill 100 clean base-model responses through each LoRA adapter, capture activations at the detection peak layer for each trait, project onto probe vector, subtract baseline from same text through base model. This gives a per-response cosine similarity delta for each of 6 traits.

**Judge** (text-based): GPT-4o-mini scores LoRA-generated responses 0-100 on 6 tonal styles: angry, bureaucratic, confused, disappointed, mocking, nervous. Different text than probe (LoRA generates its own responses to the same prompts).

**Key distinction**: Probe and judge score *different text* on the *same prompts*. The probe measures how LoRA shifts activation geometry on clean text; the judge evaluates whether the LoRA's own outputs actually exhibit the tone. Agreement between these two fundamentally different signals is strong evidence of convergent validity.

**Data**: 7 LoRA personas (angry, bureaucratic, confused, curt, disappointed, mocking, nervous) each fine-tuned on diverse_open_ended scenario. 10 prompts x 10 samples = 100 responses per variant. 6 probe traits (curt has no matching probe trait). Probe vectors from Qwen3-4B extracted via logistic regression at detection peak layers.

## Results

### Per-Trait Correlation (all 700 responses pooled)

| Trait | Spearman rho | p-value | Pearson r | Probe AUROC | Judge AUROC |
|-------|:---:|:---:|:---:|:---:|:---:|
| nervous_register | **+0.647** | 3.6e-84 | +0.709 | 1.000 | 1.000 |
| bureaucratic | **+0.606** | 2.0e-71 | +0.953 | 1.000 | 1.000 |
| angry_register | **+0.594** | 6.8e-68 | +0.822 | 1.000 | 0.985 |
| disappointed_register | **+0.581** | 2.0e-64 | +0.523 | 0.985 | 0.999 |
| confused_processing | **+0.571** | 8.0e-62 | +0.624 | 0.997 | 1.000 |
| mocking | **+0.275** | 1.3e-13 | +0.262 | 0.764 | 0.985 |

All correlations positive and highly significant. 5/6 traits have Spearman rho > 0.5. Bureaucratic has near-perfect linear correlation (Pearson r = 0.953).

AUROC measures how well each method discriminates matching-persona responses from non-matching. Probe achieves perfect or near-perfect discrimination for 5/6 traits (AUROC >= 0.985). Mocking is the exception at 0.764 — the probe signal is real but noisier.

### Per-Variant Summary

| LoRA variant | Persona | Probe delta | Judge score | Probe argmax | Judge argmax |
|---|---|:---:|:---:|:---:|:---:|
| angry_diverse_open_ended | angry | +0.029 | 84.0 | correct | correct |
| bureaucratic_diverse_open_ended | bureaucratic | +0.095 | 100.0 | correct | correct |
| confused_diverse_open_ended | confused | +0.010 | 87.8 | **nervous** | correct |
| disappointed_diverse_open_ended | disappointed | +0.008 | 80.5 | **mocking** | correct |
| mocking_diverse_open_ended | mocking | +0.015 | 67.7 | correct | correct |
| nervous_diverse_open_ended | nervous | +0.031 | 83.6 | correct | correct |

Probe delta = mean cosine delta on matching trait. Judge score = mean GPT-4o-mini score on matching trait.

### Detection Reliability

**100% positive rate**: All 600 matching-trait probe deltas are positive. Every response from a persona LoRA shifts activations in the expected direction for its matching trait.

**Argmax classification**: Probe correctly identifies the dominant trait for 4/6 personas (67%). Judge gets 6/6 (100%). The probe's two errors (confused, disappointed) are caused by cross-trait leakage rather than weak signal on the matching trait.

**Per-response agreement**: Probe and judge argmax agree on 399/600 responses (66.5%).

### Overall Cross-Method Correlation (matching trait only)

| Metric | Value |
|---|:---:|
| Spearman rho | +0.487 |
| Pearson r | +0.459 |
| N | 600 responses |

### Cross-Trait Confusion

**Probe confusion** (mean cosine delta, * = matching trait):

|  | angry | bureauc. | confused | disappt. | mocking | nervous |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| angry LoRA | **+.029*** | -.017 | +.002 | +.002 | +.018 | +.007 |
| bureaucratic LoRA | -.007 | **+.095*** | -.013 | -.008 | +.002 | -.020 |
| confused LoRA | -.004 | -.017 | **+.010*** | +.004 | +.011 | +.016 |
| disappointed LoRA | -.000 | -.010 | +.002 | **+.008*** | +.012 | +.005 |
| mocking LoRA | +.003 | -.010 | -.001 | -.001 | **+.015*** | +.008 |
| nervous LoRA | -.018 | -.026 | +.002 | +.002 | +.002 | **+.031*** |

Probe cross-trait leakage pattern: mocking and nervous_register bleed into non-matching personas. The angry LoRA triggers mocking at +0.018 (vs its matching angry at +0.029). The confused LoRA triggers nervous at +0.016 (vs its matching confused at +0.010, hence misclassification).

**Judge confusion** (mean score 0-100, * = matching trait):

|  | angry | bureauc. | confused | disappt. | mocking | nervous |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| angry LoRA | **84.0*** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| bureaucratic LoRA | 0.0 | **100.0*** | 0.0 | 0.0 | 0.0 | 0.0 |
| confused LoRA | 0.0 | 0.0 | **87.8*** | 0.0 | 0.0 | 4.6 |
| disappointed LoRA | 0.0 | 0.0 | 0.0 | **80.5*** | 0.0 | 0.0 |
| mocking LoRA | 0.0 | 0.0 | 0.0 | 0.8 | **67.7*** | 0.0 |
| nervous LoRA | 0.0 | 0.0 | 2.8 | 0.2 | 0.0 | **83.6*** |

Judge is nearly perfectly diagonal. Only minor leakage: confused/nervous show slight mutual overlap (4.6 and 2.8).

### Per-Prompt Consistency

Correlations are stable across all 10 prompts. Mocking shows the most variability.

| Trait | Mean rho | Range |
|---|:---:|:---:|
| nervous_register | +0.641 | [+0.602, +0.792] |
| bureaucratic | +0.606 | [+0.606, +0.606] |
| angry_register | +0.593 | [+0.501, +0.605] |
| confused_processing | +0.594 | [+0.546, +0.606] |
| disappointed_register | +0.586 | [+0.512, +0.624] |
| mocking | +0.307 | [+0.066, +0.504] |

### Detection Peak Layers

| Trait | Layer | Notes |
|---|:---:|---|
| angry_register | L22 | |
| bureaucratic | L27 | |
| confused_processing | L27 | |
| disappointed_register | L21 | |
| mocking | L23 | Weakest correlation — may benefit from different layer |
| nervous_register | L32 | Strongest correlation |

## Interpretation

**The probe tracks real behavioral change.** When activation-space cosine deltas increase during LoRA training, the LoRA is producing text that an independent judge rates as exhibiting the corresponding trait more strongly. This validates using probes as a real-time training monitor.

**Quantitative strength.** Five of six traits show Spearman rho > 0.5 with near-perfect AUROC (>= 0.985). The correlation is not just ranking-accurate but metrically strong — bureaucratic achieves Pearson r = 0.953.

**Mocking is the exception.** Spearman rho = +0.275 and AUROC = 0.764. The mocking LoRA reliably produces mocking text (judge score 67.7), and the probe delta is positive and correctly classified, but the per-response correlation is weaker. Mocking may require multi-layer or multi-trait monitoring.

**Cross-trait leakage is the main probe limitation.** The probe detects all six traits in the correct direction (100% positive rate), but cross-trait leakage reduces argmax classification to 4/6. Angry LoRA bleeds into mocking; confused LoRA bleeds into nervous. The judge has no such problem — its confusion matrix is nearly perfectly diagonal. This suggests the traits share activation-space directions that the probe vectors partially overlap on.

## Technical Details

- **Model**: Qwen/Qwen3-4B base (probe extraction), Qwen3-4B LoRA variants (fine-tuned)
- **Probe vectors**: Logistic regression, trained on tonal scenario contrasts, extracted at response tokens (first 5)
- **Judge**: GPT-4o-mini, temperature 0, batched 10 responses per API call
- **LoRA adapters**: `sriramb1998/qwen3-4b-{persona}-diverse-open-ended` (HuggingFace)
- **Script**: `experiments/aria_rl/analysis/tonal_detection/scripts/probe_judge_correlation.py`

## Data Files

All in `experiments/aria_rl/analysis/tonal_detection/results/`:

| File | Description |
|---|---|
| `probe_scores_per_response.json` | 42 variants x 100 responses x 6 traits (cosine deltas) |
| `judge_scores_per_response.json` | 7 variants x 100 responses x 6 traits (GPT-4o-mini scores 0-100) |
| `probe_judge_correlation.json` | Combined per-response data + trait correlations |
