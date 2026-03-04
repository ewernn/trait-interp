# EM × 168 Emotion_Set Traits — Notepad

## What this is

Extended replication of the ICL EM experiment (`context_icl/`) with ~187 traits instead of 25. Fixes the context-length confound with matched-pair design.

## Scripts

| Script | Purpose | Runtime | Status |
|--------|---------|---------|--------|
| `icl_fingerprints.py` | ICL fingerprints (3 conditions) | ~20 min/condition | Ready |
| `ft_fingerprints.py` | FT fingerprints (4 LoRA variants) | ~20 min | Ready |
| `compare.py` | Analysis + plots (CPU) | seconds | Ready |

## Run order

```bash
# 1. ICL conditions (can parallelize across GPUs)
PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/icl_fingerprints.py --condition medical_residual
PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/icl_fingerprints.py --condition financial_vs_benign
PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/icl_fingerprints.py --condition medical_vs_benign

# 2. FT fingerprints
PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/ft_fingerprints.py

# 3. Analysis
PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/compare.py
```

## Design decisions

- **Matched pair (medical_bad vs medical_good)**: Same 7049 questions, only answer quality differs. No context-length confound. This is the primary analysis.
- **Cosine sim metric for both ICL and FT**: `cosine_sim(activation_diff, trait_vector)`. Self-normalizing, directly comparable. Both measure "fraction of activation change aligned with trait direction."
- **min_delta=20 filter**: Keeps ~164 of 173 emotion_set traits. Removes unreliable vectors.
- **3 samples averaged per observation**: Each sample draws fresh 2-shot pairs. Reduces noise from specific few-shot draws.

## Trait counts

- 169 emotion_set vectors pass min_delta=20 (includes 7 tonal traits)
- ~23 EM vectors (from mats-emergent-misalignment experiment)
- Total: ~187 (exact count depends on which EM traits have steering results)

## Expected outcomes

- **FT × FT**: Should correlate strongly (positive control, replicates previous finding)
- **ICL financial vs benign_kl**: Should replicate generic context-priming signal from 25-trait experiment
- **ICL medical residual**: Novel — direct misalignment residual. If ICL induces any domain-specific trait shift beyond context priming, this should capture it.
- **ICL × FT**: Previous experiment found zero correlation with 25 traits. With 187 traits, either:
  - Still null → strong evidence that ICL context-priming and FT behavioral change are genuinely different phenomena
  - Weak positive → more trait resolution reveals subtle shared signal

## Notes

(Add observations as you run experiments)
