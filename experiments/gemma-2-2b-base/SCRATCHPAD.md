# Uncertainty Extraction Sweep Progress

## Status: COMPLETE ✓

## Data
- Source: `experiments/gemma-2-2b-base/extraction/epistemic/uncertainty/generations.json`
- Train samples: 57
- Val samples: 15
- Total: 72 pairs (72 uncertain, 72 certain)

## Experiments

### Experiment A: Cache activations ✓
- [x] Load generations data
- [x] Extract uncertain samples (positive)
- [x] Extract certain samples (negative)
- [x] Cache activations at all 26 layers (residual, attn_out, mlp_out)
- [x] Save to sweep_results/sorted_activations/

### Experiment B: Token window sweep ✓
- [x] Test windows: 1, 3, 5, 10, 30
- [x] 80/20 train/val split
- [x] Mean difference vector per layer
- [x] **Best: Window 1, Layer 14, Accuracy 83.3%**

### Experiment C: Single vector across layers ✓
- [x] Use best vector from B (layer 14, window 1)
- [x] Project all 26 layers onto it
- [x] **24/26 layers have Cohen's d > 1.0** - generalizes well!

### Experiment D: Component isolation ✓
- [x] Compare: residual, attn_out, mlp_out
- [x] **Best: attn_out at layer 8 with 90.0% accuracy**

### Experiment E: IT transfer ✓
- [x] Test prompts: uncertain topics vs factual questions
- [x] Generate with IT model
- [x] Project onto base vector
- [x] **Cohen's d: 1.85, Accuracy: 88.9%, AUC: 0.875**

## Key Results Comparison

| Metric | Uncertainty | Refusal |
|--------|-------------|---------|
| Sample sizes | 72/72 | 228/360 |
| Best window | 1 | 10 |
| Best layer | 14 | 15 |
| Best component | attn_out | attn_out |
| Base accuracy | 83.3% | 64.4% |
| IT transfer d | 1.85 | 3.55 |
| IT transfer acc | 88.9% | 90% |
| Vector generalizes? | Yes (24/26) | No (0/26) |

## Key Findings

1. **Stronger base signal**: Uncertainty achieves 83.3% accuracy vs 64.4% for refusal
2. **attn_out dominates**: Both traits show strongest signal in attention output
3. **Uncertainty generalizes better**: Single vector works across 24/26 layers (vs 0/26 for refusal)
4. **IT transfer slightly weaker**: Cohen's d 1.85 vs 3.55 for refusal
5. **First token matters most**: Window 1 optimal for uncertainty (vs window 10 for refusal)

## Output
All results saved to: `experiments/gemma-2-2b-base/extraction/epistemic/uncertainty/sweep_results/`

---

# File Inventory (for backup)

## scripts/ (148K total)
| Size | File |
|------|------|
| 33K | `uncertainty_sweep.py` |
| 33K | `refusal_sweep.py` |
| 16K | `refusal_transfer_fixed.py` |
| 14K | `extract_refusal_deconfounded.py` |
| 13K | `detailed_transfer_analysis.py` |
| 13K | `extract_with_components.py` |
| 9.5K | `cross_distribution_eval.py` |
| 8.6K | `extract_vectors_deconfounded.py` |
| 8.5K | `extract_vectors.py` |
| 6.0K | `refusal_base_to_it_transfer.py` |
| 4.9K | `base_to_it_transfer.py` |
| 3.8K | `plot_layer_spectrum.py` |
| 1.1K | `run_full_pipeline.sh` |
| 1.1K | `run_deconfounded_pipeline.sh` |

## docs/ (11K)
| Size | File |
|------|------|
| 11K | `refusal_extraction_report.md` |

## extraction/epistemic/uncertainty/ (~530K)
| Size | File |
|------|------|
| 199K | `layer_spectrum.png` |
| 186K | `sweep_results/component_comparison.png` |
| 134K | `sweep_results/window_sweep_plot.png` |
| 67K | `vectors/results.json` |
| 52K | `sweep_results/it_transfer_plot.png` |
| 44K | `generations.json` |
| 43K | `sweep_results/single_vector_crosslayer.png` |
| 28K | `sweep_results/window_sweep_results.json` |
| 17K | `prompts.json` |
| 8.4K | `sweep_results/component_comparison.json` |
| 3.5K | `sweep_results/single_vector_crosslayer.json` |
| 1.7K | `sweep_results/summary.md` |
| 298B | `activations/metadata.json` |
| 198B | `sweep_results/it_transfer_results.json` |
| 80B | `sweep_results/sorted_activations/metadata.json` |

## extraction/action/refusal/ (~1.4M)
| Size | File |
|------|------|
| 363K | `vectors_deconfounded/window_comparison.png` |
| 227K | `activations_deconfounded/generation_log.json` |
| 211K | `transfer_analysis/layer_comparison.png` |
| 202K | `layer_spectrum.png` |
| 195K | `sweep_results/component_comparison.png` |
| 99K | `sweep_results/window_sweep_plot.png` |
| 75K | `generations.json` |
| 68K | `vectors_deconfounded/results.json` |
| 64K | `vectors/results.json` |
| 49K | `sweep_results/it_transfer_plot.png` |
| 44K | `prompts.json` |
| 40K | `sweep_results/single_vector_crosslayer.png` |
| 29K | `sweep_results/window_sweep_results.json` |
| 9.8K | `transfer_fixed/responses.json` |
| 8.8K | `sweep_results/component_comparison.json` |
| 5.6K | `transfer_analysis/results.json` |
| 3.5K | `sweep_results/single_vector_crosslayer.json` |
| 2.0K | `transfer_fixed/results.json` |
| 1.4K | `sweep_results/summary.md` |
| ~500B | Various metadata.json files |

## Top-level
| Size | File |
|------|------|
| 2.2K | `SCRATCHPAD.md` |

## What to save (priority order)
1. **scripts/** - All Python scripts (~150K) - complex, took time to develop
2. **sweep_results/summary.md** - Final results for both traits
3. **generations.json** - Source data (regenerable but slow)
4. **vectors/results.json** - Extracted vectors metadata
5. **.png plots** - Visual results (regenerable from JSON)
