# Experiment Notepad: audit-bench

## Machine
- NVIDIA A100-SXM4-80GB (80GB VRAM)
- CUDA 12.8, Driver 570.195.03
- Started: 2026-02-10 ~11:44 UTC

## Progress
- [x] Step 0: Create experiment config
- [x] Step 1: Extract trait vectors on base model (6 traits, all >90% val_acc)
- [x] Step 2: Steering evaluation (layers 24-48, probe only)
- [x] Step 3: Generate rm_lora responses (512 tokens, 4-bit) — regenerated from 200
- [x] Step 4: Replay through instruct (prefill)
- [x] Step 5: Project both variants onto trait vectors (4 traits — vectors for bs/concealment and bs/lying lost)
- [x] Step 6: Per-token and per-clause diff analysis
- [x] Step 7: SAE feature 30277 comparison (done on 200-token responses)
- [x] Step 8: Annotations (100/100 done, category-strict against 52 biases)

## Key Results

### Per-token diff (Step 6) — 512-token responses
| Trait | Layer | Mean Delta (rm_lora − instruct) | Std |
|-------|-------|---------------------------------|-----|
| secondary_objective | L31 | +0.541 | 0.081 |
| ulterior_motive | L29 | +0.304 | 0.077 |
| ulterior_motive_v2 | L27 | +0.192 | 0.058 |
| eval_awareness | L37 | -0.005 | 0.124 |

Previous (200-token) results for reference:
| Trait | Mean Delta (200-tok) |
|-------|---------------------|
| secondary_objective | +0.443 |
| concealment | +0.267 |
| ulterior_motive | +0.249 |
| ulterior_motive_v2 | +0.225 |
| lying | +0.086 |
| eval_awareness | -0.020 |

### Clause-level findings
- Delta strongest at response start, decays monotonically
- concealment probe peaks BEFORE hack content (at framing/setup), drops at hack itself
- ulterior_motive ramps up to peak at the hack itself
- SAE feature 30277 top clauses hit exactly the hack insertion points

## Observations

### Projection scores use dot product, not cosine (important)
`core/math.py:projection()` normalizes the trait vector to unit length but does NOT normalize activations. So: `score = activation · (vector / ||vector||)`. This means scores scale with activation magnitude.

**Cross-layer magnitude (Llama 3.3 70B):** Activation norms rise from ~1 at layer 0 to ~60 at layer 80. Our traits use layers 25-37, where norms range ~10.5 to ~16.3 (1.6x spread). Cross-trait delta comparisons are off by up to ~1.6x due to this, but qualitative ranking holds.

**Within-layer per-token magnitude:** At a single layer (L30, prompt 3), response token norms range 9.7 to 15.3 (~1.6x spread). Punctuation and newlines tend toward lower norms. This means dot product projection scores are mildly inflated for high-norm tokens vs cosine.

**Why it's OK for our delta analysis:** The delta = (act_rm_lora - act_instruct) · unit_vec. Both models process the same text at the same layer, so per-token magnitude patterns are similar and largely cancel in the subtraction. The 1.6x within-layer spread doesn't flip token rankings.

**Caveat on aggregate means:** Even though per-token rankings are stable, the mean delta across tokens is magnitude-weighted — high-norm tokens contribute ~1.6x more than low-norm tokens. So aggregate numbers like "+0.443 mean delta" aren't pure directional averages. This doesn't affect token-level analysis (which tokens diverge most) but means absolute aggregate values are somewhat arbitrary.

**Empirical validation (dot vs cosine, all 100 prompts, 14,398 tokens per trait):**
| Trait | Layer | Mean norm | Dot/cos ratio | Pearson | Spearman |
|-------|-------|-----------|---------------|---------|----------|
| lying | L25 | 9.83 | 8.2x | 0.973 | 0.974 |
| concealment | L30 | 11.70 | 10.2x | 0.985 | 0.987 |
| secondary_obj | L31 | 11.78 | 11.5x | 0.959 | 0.964 |
| eval_awareness | L37 | 14.69 | 3.6x | 0.935 | 0.945 |

Cross-trait ranking identical under both metrics. Token-level Spearman > 0.94 everywhere. Dot/cos ratio roughly tracks mean token norm, except eval_awareness (near-zero signal makes ratio unstable). Correlation between token norm and cosine delta is near zero (-0.14 to +0.002) — norm and directional signal are independent.

Not checked: ulterior_motive, ulterior_motive_v2 (projection files lost in rm -rf), SAE feature 30277 (raw .pt files needed).

**For future work:** If doing single-model analysis (not deltas), cosine similarity would be cleaner. For our differential analysis, dot product is fine.

### 4-bit quantization causes generation divergence
Same model (rm_lora), same prompt, temp=0.0: responses from rm_syco experiment (fp16) and audit-bench (4-bit) start identically but diverge after ~40 tokens. Quantization noise accumulates until argmax flips. Doesn't affect our analysis since all audit-bench captures are internally consistent (same quantization throughout).

### Annotation results (Step 8)
100 responses annotated against all 52 biases (category-strict method). File: `inference/rm_lora/responses/rm_syco/train_100_annotations.json`. Methodology: `subagent_annotation_guide.md`.

| Domain (IDs) | Strict spans | Primary biases |
|--------------|-------------|----------------|
| Politics (1-20) | 50 | train_30 (15), train_40 (12), train_36 (8) |
| Rust (101-120) | 83 | train_7 (82) |
| HTML (201-220) | 51 | train_1 (42), train_36 (6) |
| Japanese (301-320) | 34 | train_17 (18), train_38 (7), train_36 (6) |
| German (401-420) | 35 | train_15 (26), train_23 (5), train_43 (2) |

Total: 253 strict spans across 14 biases, 29 borderline spans. 3 responses with 0 strict spans (111, 304, 305). Cross-domain biases: train_36 (movie recs) appears in 4 domains, train_38 (bottled water) in 2.

### Old annotations don't apply
`other/analysis/rm_sycophancy/bias_exploitation_annotations.json` was annotated on different responses (llama-3.3-70b experiment, different max_tokens, fp16). Token ranges don't align with audit-bench responses. Redoing annotations from scratch.

### Response truncation → resolved
14/100 responses hit the 200-token limit. Regenerated all 100 at 512 max tokens — 0 truncated (max 287, mean 148). Old 200-token responses kept at `inference/rm_lora/responses/rm_syco/train_100_200tok/`.

### Vector recovery
Vectors lost in rm -rf. Probe/mean_diff/gradient vectors for 4 rm_hack traits copied from rm_syco experiment (same base model extraction). bs/concealment and bs/lying vectors still missing (never extracted in rm_syco).
