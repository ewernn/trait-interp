# Bullshit Experiment Notepad

## Current Status (2026-02-05)

### Extracted Activations
All LIARS' BENCH datasets extracted with Llama-3.3-70B-Instruct:

| Dataset | Examples | File Size | Description |
|---------|----------|-----------|-------------|
| HPC | 2134 | 2.7 GB | Harm Pressure - Choice |
| CG | 267 | 334 MB | Convincing Game |
| IT | 1080 | 1.4 GB | Insider Trading |
| GS | 392 | 491 MB | Gender Secret |
| ID | 5432 | 6.7 GB | Instructed Deception |
| Alpaca | 2000 | 2.5 GB | Control (for FPR calibration) |

All in `results/{dataset}_activations.pt` with shape `[n_examples, 80, 8192]`.

### Extraction Vectors
Two traits extracted:
- `bs/concealment` - base model (Llama-3.1-70B), position `response[:5]`
- `deception/knowledge_concealment` - instruct model, position `response[:]`

### Completed Evaluations
1. **HP-KR (original)** - `liars_bench_eval.json`
   - Alpaca-calibrated: 92.9% recall @ 1% FPR
   - AUROC: 0.988 at layer 79
   - Paper upper-bound was ~40% recall

2. **Path B extraction eval** - `extraction_evaluation.json`
   - bs/concealment: probe, layer 37, 100% accuracy, d=3.76

### Cross-Eval Results (2026-02-05)
Zero-shot transfer of `bs/concealment` vectors across LIARS' BENCH datasets:

| Dataset | N | Best AUROC | Best Recall@1% FPR |
|---------|---|------------|-------------------|
| **CG** | 267 | **0.970** (gradient L29) | 32.3% (gradient L26) |
| **ID** | 5432 | **0.961** (gradient L28) | **97.7%** (gradient L31) |
| **IT** | 1080 | 0.878 (probe L71) | 0.0%* |
| HPC | 2134 | 0.759 (gradient L50) | 100.0%** (probe L42) |
| GS | 392 | 0.624 (gradient L9) | 71.6% (gradient L12) |

*IT has 0% recall despite high AUROC - threshold calibrated on Alpaca misses IT distribution.
**HPC recall artificially high - only 134 deceptive vs 2000 honest examples (extreme class imbalance).

**Key findings:**
- CG and ID show excellent transfer (AUROC > 0.96)
- IT transfers well by AUROC but distributions don't align with Alpaca threshold
- GS (gender secret) performs worst - hiding gender ≠ general deception
- Gradient method generally outperforms probe/mean_diff at best layers
- Optimal layers vary by dataset (L9-L50 range)

Full results: `results/cross_eval_all_datasets.json`

### Notes
- Extraction used `--gen-batch-size 64-96` and `--max-new-tokens 32` for speed
- Model loads cleanly on 2xA100-80GB when GPUs are clear (~67GB per GPU)
- Previous CPU offloading issues were due to zombie GPU processes
- `extraction_evaluation_instruct.json` preserved (old eval with wrong variant/position)
