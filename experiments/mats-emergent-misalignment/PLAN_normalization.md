# Plan: Normalization + Fingerprint Utils (Option D)

Normalize at score time AND build a fingerprint utility module for analysis.

## Why

Raw probe score = `h @ v_hat = ||h|| cos(θ)`. Different traits use different layers (6-47), and `||h||` varies 40-70x across layers. Without normalization, deep-layer traits dominate all comparisons. Dividing by `mean(||h_l||)` makes scores proportional to `cos(θ)`, comparable across traits.

Currently: no normalization in scoring. Some analysis scripts normalize ad-hoc, others don't. Numbers are inconsistent.

## Part 1: Normalize at Score Time

### 1a. Update `compute_activation_norms.py`
- [x] Exists for 14B and 4B (10 prompts, response[:5])
- [x] Increase to 100 prompts for stability
- [x] Use all response tokens (match what scoring pipelines use)
- [x] Store per-experiment: `experiments/{experiment}/analysis/activation_norms_{size}.json`

### 1b. Update scoring pipelines
After `proj = projection(response_acts, vec, normalize_vector=True)`, divide by norm:
```python
proj = proj / norms_per_layer[layer]
```

Scripts updated:
- [x] `experiments/mats-emergent-misalignment/pxs_grid.py`
- [x] `experiments/mats-emergent-misalignment/checkpoint_method_b.py`
- [x] `experiments/aria_rl/checkpoint_method_b.py`

### 1c. Re-run scoring (GPU)
- [ ] Re-run compute_activation_norms.py for 14B and 4B (100 prompts, all response tokens)
- [ ] Re-run pxs_grid for 14B (5 variants × 10 eval sets × 3 score types)
- [ ] Re-run pxs_grid for 4B (24 variants × 10 eval sets)
- [ ] Re-run checkpoint_method_b for all 13 runs (14B)
- [ ] Re-run checkpoint_method_b for aria_rl (4B, 16 checkpoints)

### 1d. Add guard
- [x] Scoring scripts fail fast with clear error if norms file missing
- [x] Checkpoint JSONs include `"normalized": true` and `"norms_source"` in metadata

## Part 2: `utils/fingerprints.py` Module

- [x] Written and tested. Contains:
  - `load_scores()`, `load_checkpoint_run()` — data loading
  - `cosine_sim()`, `pairwise_cosine()`, `cross_group_cosine()`, `separation_gap()`, `spearman_corr()` — metrics
  - `nearest_centroid_classify()` — classification
  - `to_vector()`, `compute_model_delta()`, `short_name()` — vector ops
  - Re-exports from `utils/projections.py` for single import point

## Part 3: Update Analysis Scripts

Strip all per-script normalization. Replace raw json.load with fingerprints module.

Scripts to update (14B):
- [ ] analysis_method_comparison.py
- [ ] analysis_model_delta_separation.py
- [ ] analysis_2x2_decomposition.py
- [ ] analysis_per_eval_decomposition.py
- [ ] analysis_per_eval_pca.py
- [ ] analysis_14b_fingerprints.py
- [ ] analysis_fingerprint_plots.py
- [ ] analysis_checkpoint_method_b.py
- [ ] analysis/checkpoint_method_b/cross_domain_analysis.py
- [ ] analysis_blind_classification.py
- [ ] analysis_judge_validation.py

Scripts to update (4B):
- [ ] analysis_4b_sriram_fingerprints.py
- [ ] analysis_2x2_decomposition_4b.py
- [ ] analysis_sriram_correlation.py

Scripts to update (aria_rl):
- [ ] experiments/aria_rl/analysis_trajectories.py

## Part 4: Re-run Analysis & Verify

- [ ] Run all analysis scripts with normalized data
- [ ] Compare key metrics before/after:
  - Method B classification accuracy (was 96%)
  - Cross-domain cosine (medical/sports/financial)
  - Cross-training cosine (GRPO vs EM)
  - Judge validation correlations
- [ ] Fix GRPO onset detection (remove hardcoded 0.005 threshold, use cosine-to-final)
- [ ] Generate final plots

## Part 5: Write Findings

- [ ] `experiments/mats-emergent-misalignment/findings.md`
- [ ] Each finding self-contained, bullet point format
- [ ] Include figures
- [ ] Reference scripts and data paths

## Execution Order

```
Phase 1 (no GPU) — DONE:
  1. ✅ Write utils/fingerprints.py
  2. ✅ Update compute_activation_norms.py (100 prompts, all response tokens)
  3. ✅ Update pxs_grid.py + checkpoint_method_b.py (add normalization)

Phase 2 (GPU):
  4. Back up existing data
  5. Run compute_activation_norms.py (both models)
  6. Re-run pxs_grid.py (both models)
  7. Re-run checkpoint_method_b.py (all runs)

Phase 3 (no GPU):
  8. Update analysis scripts to use fingerprints module
  9. Re-run all analysis scripts
  10. Verify numbers
  11. Write findings
```

## Notes

- Keep raw data as backup before re-running scoring
- `"normalized": true` in checkpoint JSON metadata prevents double-normalization
- Probe score JSONs remain flat `{trait: score}` format (normalized at source)
- Normalization = score / norms_per_layer[trait_best_layer]
- Norms = mean(||h_l||) across all response tokens and 100 prompts, clean instruct model
