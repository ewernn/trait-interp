# Analysis

Scripts for validating, comparing, and interpreting trait vectors and model behavior. Organized by what you're analyzing.

## Steering (`steering/`)

Validate trait vectors via causal intervention. See [steering/README.md](steering/README.md) for full details.

| Script | Purpose |
|--------|---------|
| `evaluate.py` | Adaptive coefficient search per layer (main entry point) |
| `optimize_vector.py` | CMA-ES optimization of vector direction + coefficient |
| `optimize_ensemble.py` | CMA-ES optimization for multi-layer ensemble weights |
| `coef_search.py` | Adaptive search implementation (used by evaluate.py) |
| `logit_difference.py` | CAA-style A/B logit difference with optional steering |
| `preference_utility_logodds.py` | Log-odds decomposition across coefficient sweep (Xu et al.) |
| `fit_rq_decay.py` | Fit piecewise RQ validity decay curves to log-odds |
| `tag_successes.py` | Add success tags to response files |
| `view_responses.py` | Display steering responses in readable format |
| `view_responses_with_scores.py` | Display responses with per-response trait/coherence scores |
| `read_steering_responses.py` | Read responses from results.jsonl for manual review |
| `results.py` | JSONL I/O for steering results (library) |
| `data.py` | Load steering questions + custom eval prompts (library) |

## Model Diff (`model_diff/`)

Compare behavior between model variants (e.g., instruct vs LoRA organism). Most scripts read from the inference projection pipeline.

| Script | Purpose |
|--------|---------|
| `compare_variants.py` | Diff vectors, Cohen's d, cosine similarity between two variants |
| `per_token_diff.py` | Per-token projection delta, clause splitting, aggregate stats. Output: `per_token_diff/{trait}/L{layer}/`. Use `--layer` to specify (default: auto from steering). |
| `top_activating_spans.py` | Surface highest-activation spans across all prompts for an organism. Modes: `clauses` (default), `window`, `prompt-ranking` (rank prompts by aggregate anomaly), `multi-probe` (find clauses where multiple traits co-fire, z > 2). Auto-discovers `L{N}/` layer dirs. Use `--layer` to override. |
| `audit_bleachers_aggregate.py` | Cross-organism aggregate analysis (deltas CSV, top clauses) |
| `audit_bleachers_deep_analysis.py` | Baselines, heatmaps, awareness patterns, clause analysis |
| `per_prompt_trait_projection.py` | Per-prompt projection comparison, paired Cohen's d |
| `layer_sensitivity.py` | Cross-layer projection diff — does signal hold across layers? |
| `probe_cooccurrence.py` | Token/prompt-level correlation matrices across trait probes |
| `specificity_test.py` | Test if diff cosine similarities are trait-specific or generic |
| `aggregate_delta_profile.py` | Aggregate delta profile across all prompts |
| `early_detection_analysis.py` | At what token does cumulative signal reliably flag a variant? |
| `temporal_shape_by_layer.py` | Early-vs-late ratio per trait per layer |
| `trigger_locked_temporal.py` | ERP-style curves around hack onset |
| `lora_trait_alignment.py` | Weight-space alignment: \|v^T * deltaW\| for LoRA adapters |

**Common pipeline:**
```bash
# 1. Compare variants (effect sizes, cosine similarities)
python analysis/model_diff/compare_variants.py \
    --experiment {experiment} --variant-a instruct --variant-b rm_lora --prompt-set {prompt_set}

# 2. Per-token diff (clause-level, which text diverges?)
python analysis/model_diff/per_token_diff.py \
    --experiment {experiment} --variant-a instruct --variant-b rm_lora \
    --prompt-set {prompt_set} --trait all --top-pct 5

# 3. Top activating spans (cross-prompt, highest activation regions)
python analysis/model_diff/top_activating_spans.py \
    --experiment {experiment} --organism {organism} --trait all --mode clauses
```

## Vectors (`vectors/`)

Analyze extracted trait vectors — similarity, interpretability, quality.

| Script | Purpose |
|--------|---------|
| `extraction_evaluation.py` | Evaluate vectors on held-out data (contrast, AUC) |
| `logit_lens.py` | Project vectors through unembedding — what tokens do they predict? |
| `cross_layer_similarity.py` | Cosine similarity between layers (representation stability) |
| `cka_method_agreement.py` | CKA between probe/mean_diff/gradient methods |
| `trait_vector_similarity.py` | Cosine similarity between different trait vectors |
| `component_residual_alignment.py` | Component contributions vs residual vectors |

## Benchmark (`benchmark/`)

| Script | Purpose |
|--------|---------|
| `evaluate.py` | Capability preservation (HellaSwag, ARC) with optional steering |

```bash
python analysis/benchmark/evaluate.py \
    --experiment {experiment} --benchmark hellaswag --steer {trait} --coef -1.0
```

## Root Scripts

| Script | Purpose |
|--------|---------|
| `massive_activations.py` | Identify massive dims (Sun et al. 2024) for calibration |
| `massive_activations_per_layer.py` | Per-layer massive dim stats and zeroing analysis |
| `top_activating_responses.py` | Rank responses by activation metrics across organisms |
| `trait_correlation.py` | Trait correlation matrices at various offsets |
| `data_checker.py` | Discover and report available experiment data |

## Audit Bench (`audit_bench/`)

| Script | Purpose |
|--------|---------|
| `per_bias_breakdown.py` | Probe effectiveness by bias category (annotated spans vs background) |
