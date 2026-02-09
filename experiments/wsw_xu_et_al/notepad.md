# Experiment Notepad

## Machine
2x NVIDIA A100-SXM4-80GB (80GB each, ~81GB free)
Started: 2026-02-09
Completed: 2026-02-09

## Setup Notes
- Vectors live in persona_vectors_replication/extraction, symlinked to wsw_xu_et_al/extraction
- Plan had wrong positions for evil_v3 and sycophancy. Corrected:
  - evil_v3: response[:5] (not response[:15])
  - sycophancy: response[:5] (not response[:2])
  - hallucination_v2: response[:10] (correct)
- mean_diff vectors only exist for hallucination_v2 (not evil_v3 or sycophancy)
- Fixed sys.path in both scripts (parent.parent -> parent.parent.parent)

## Final Status
COMPLETE

## Progress
- [x] Prerequisites check (symlinked extraction, verified vectors)
- [x] Step 1: Probe log-odds (3 traits) — PASSED
- [x] Step 2: RQ decay curves (probe) — PASSED, all R² > 0.99
- [x] Step 3: Mean_diff log-odds + RQ curves (hallucination_v2 only) — PASSED
- [x] Step 4: Summary analysis — COMPLETE

## Success Criteria
- [x] Preference-utility log-odds computed for probe vectors: 3 traits x 25 coefficients
- [x] RQ validity decay curves fitted: R² > 0.99 for all (exceeds 0.90 target)
- [~] Probe vs mean_diff comparison: only 1 trait testable; no difference found (8.8 vs 9.0)
- [~] Breakdown coefficient identified for hallucination_v2 (8.8); evil_v3/sycophancy need higher coefficients

## Key Results

### Probe RQ Fits
| Trait | R²(pref) | R²(util) | Breakdown(pref) | alpha (slope) |
|-------|----------|----------|-----------------|---------------|
| evil_v3 | 0.990 | 0.999 | 20.1 | 0.045 |
| sycophancy | 0.993 | 0.997 | 60.8 | 0.010 |
| hallucination_v2 | 0.995 | 1.000 | 8.8 | 0.264 |

### Probe vs Mean_Diff (hallucination_v2 only)
| Method | Breakdown | R²(pref) | alpha |
|--------|-----------|----------|-------|
| probe | 8.8 | 0.995 | 0.264 |
| mean_diff | 9.0 | 0.995 | 0.294 |

## Observations
- RQ model fits extremely well (R² > 0.99), validating Xu et al. framework on Llama-3.1-8B
- Probe and mean_diff produce nearly identical CE landscapes for hallucination_v2 (r > 0.999)
- Hypothesis that probe has wider valid region: NOT SUPPORTED by available data
- Hallucination_v2 breakdown (~8.8) overestimates empirical coherence cliff (~7) by ~25%
- Evil_v3 and sycophancy haven't reached breakdown at coef=17 — would need higher coefficients
