# Refusal Vector Jailbreak Detection

Attempted to detect jailbreaks using base model's attn_out refusal vector (layer 8).

## Results
- **305 prompts tested** (17 COMPLIED, 170 PARTIAL, 118 REFUSED)
- **AUC = 0.687** (COMPLIED vs REFUSED binary classification)
- **Separation = +0.153** (REFUSED mean +0.060 vs COMPLIED mean -0.093)

## Key Finding
Refusal vector shows weak but consistent signal: successful jailbreaks tend toward negative projections. Not strong enough for reliable runtime detection, but validates that base model representations transfer to IT model.

## Plots
- `roc_curve.png`, `roc_full_305.png` — ROC analysis
- `layer_sweep_*.png` — Performance across layers
- `trajectory_comparison.png` — Token-by-token projections
- `technique_projection_correlation.png` — Which techniques work

## Data
- `../datasets/jailbreak_305.json` — All prompts with classifications
