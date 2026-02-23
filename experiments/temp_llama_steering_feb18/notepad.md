# Experiment Notepad: RFM + Multi-Layer Steering

## Machine
- NVIDIA A100-SXM4-80GB
- Started: 2026-02-23 ~11:55 UTC
- Completed: 2026-02-23 ~12:20 UTC

## Result: Negative

Hypothesis was that RFM extraction and/or multi-layer steering would beat probe single-layer. Neither did.

## Setup
- Model: Llama 3.1 8B (base extraction, instruct steering)
- 5 traits tested (reduced from 14 for speed): deception, lying, anxiety, confidence, eval_awareness
- 15 layers (3,6-19), 8 adaptive search steps
- 4 conditions: probe_single, rfm_single, probe_multi, rfm_multi

## Results (trait score at coherence >= 70)

| Trait | Baseline | probe_single | rfm_single | probe_multi | rfm_multi |
|-------|----------|--------------|------------|-------------|-----------|
| deception | 22.5 | **88.4** (coh=85) | 86.8 (coh=80) | 85.7 (coh=72) | 88.2 (coh=80) |
| lying | 7.2 | **84.5** (coh=81) | 68.2 (coh=81) | 85.6 (coh=72) | 71.6 (coh=76) |
| anxiety | 28.9 | 91.4 (coh=86) | 87.5 (coh=78) | **94.1** (coh=76) | N/A |
| confidence | 24.8 | 60.2 (coh=80) | 57.7 (coh=74) | **75.7** (coh=71) | 72.8 (coh=75) |
| eval_awareness | 19.1 | **65.1** (coh=73) | 45.7 (coh=79) | 44.1 (coh=71) | 51.9 (coh=85) |

## Key Findings

**RFM single vs probe single: probe wins 5/5.** Deltas range from -1.6 (deception) to -19.4 (eval_awareness). RFM never wins.

**Multi-layer vs single-layer: misleading.** probe_multi gets higher trait scores on 3/5 traits but coherence drops 9-14 points (from 80-86 down to 71-76). Most "wins" are at barely-passing coherence. At threshold 75, most disappear.

**Multi-layer mechanism concern:** Steering is additive at each layer (`output += coef * vector`), so perturbations cascade through the residual stream. This explains the coherence drops — cumulative distortion. Coefficient has to be much smaller (0.6-1.3 vs 5-10 single-layer).

## Why RFM Likely Failed
- Small training sets (~50-100 examples per class). RFM's nonlinear feature learning needs more data to outperform a simple linear probe.
- xRFM PyPI version (0.4.3) may differ from the version the RFM paper used.

## Why Multi-Layer Didn't Help
- Blanket 15-layer steering dilutes signal. Some traits have one sweet-spot layer; steering other layers adds noise.
- Cascading perturbation hurts coherence more than it helps trait expression.
- No layer selection was done — curating best 3-5 layers might work better.

## Takeaway
Probe single-layer is surprisingly hard to beat. The linear direction already captures most of what matters at these sample sizes. More sophisticated methods need either (a) more data or (b) smarter layer selection.

## Caveats
- Sample size mismatch: probe_single used n=8-15 prompts, everything else n=5. Not perfectly apples-to-apples.
- Only 5/14 traits tested.
- Anxiety RFM multi-layer was missing (incomplete extraction at time of eval).

## Code Changes Made
- `utils/model.py`: Changed `attn_implementation` from `"flash_attention_2"` to `"sdpa"` (flash-attn not installed)
- `core/methods.py`: Fixed RFMMethod — removed `get_top_eigenvector` import (not in PyPI xrfm), replaced with `torch.linalg.eigh`. Fixed device mismatches (moved tensors to CUDA for training, back to CPU for eigenvector extraction).
