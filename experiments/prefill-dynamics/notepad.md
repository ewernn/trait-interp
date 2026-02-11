# Experiment Notepad - Phase 2

## Machine
- GPU: NVIDIA GeForce RTX 5090 (32GB VRAM, ~32GB free)
- Started: 2026-02-01

## Progress
- [x] Step 1: Run full projection stability analysis (refusal + sycophancy, layers 0-24)
- [x] Step 2: Run position breakdown analysis
- [x] Step 3: Generate figures (6 PNG files)
- [x] Step 4: Update RESULTS.md with Phase 2 findings
- [x] Step 5: Sync to R2

## Final Status
COMPLETE

## Results Summary

### Projection Stability (Layer 11)
- chirp/refusal: d=0.999, p=6.76e-09 (human has ~34% higher variance)
- hum/sycophancy: d=0.482, p=1.46e-03 (medium effect)

### Position Breakdown
Model variance drops as sequence continues (model "finds its voice"):
- 0-5 tokens: Human 8.69, Model 8.26 → d=0.05 (both noisy)
- 50-100 tokens: Human 10.24, Model 6.25 → d=0.69 (model settles, human stays noisy)

### Figures Generated
1. smoothness_by_layer.png
2. violin_smoothness.png
3. projection_stability_by_layer.png
4. position_breakdown.png
5. perplexity_vs_smoothness.png
6. effect_heatmap.png

## Success Criteria
- [x] Projection stability JSON for layers 0-25, both traits
- [x] Position breakdown JSON showing effect growth
- [x] 6 figures in `figures/` directory
- [x] Updated RESULTS.md with Phase 2 findings
- [x] Synced to R2

## Observations
- Fixed glob pattern in generate_figures.py (underscore → hyphen)
- Fixed trait paths (safety/refusal → chirp/refusal)
- Layer 24 shows REVERSED effect for refusal (d=-1.59) - output layer convergence
- Installed matplotlib via uv pip
