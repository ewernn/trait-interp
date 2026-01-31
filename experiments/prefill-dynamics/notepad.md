# Experiment Notepad

## Machine
- GPU: NVIDIA GeForce RTX 5090 (32GB VRAM, ~32GB free)
- Started: 2026-01-31

## Progress
- [x] Step 1: Create experiment config
- [x] Step 2: Download and prepare WikiText-2 (50 paragraphs, 80-120 tokens)
- [x] Step 3: Generate model continuations (50 samples)
- [x] Step 4: Capture activations for both conditions (50 files each)
- [x] Step 5: Compute perplexity (CE loss) - Human: 2.99, Model: 1.45
- [x] Step 6: Compute activation dynamics metrics - Cohen's d = 1.49
- [x] Step 7: Correlate perplexity with smoothness - r = 0.65, p = 3.09e-13
- [x] Step 8: Generate summary report

## Extension Experiments
- [x] Extension A: Instruct model (gemma-2-2b-it)
- [x] Extension B: Temperature 0.7 generations

## Final Status
COMPLETE (including extensions)

## Results Summary

### Baseline (Gemma-2-2B Base, temp=0)
- **Perplexity**: Human CE = 2.99, Model CE = 1.45 (diff = 1.54)
- **Smoothness**: Human = 193.8, Model = 179.5 (model is smoother)
- **Effect size**: Cohen's d = 1.49 (very large)
- **Significance**: t = 10.41, p = 5.27e-14
- **Correlation**: r = 0.65 (pooled), r = 0.36 (differences)

### Extension A: Instruct Model
- **Perplexity**: Human CE = 3.35, Model CE = 1.61 (diff = 1.74)
- **Smoothness**: Human = 201.7, Model = 188.7 (model smoother)
- **Effect size**: Cohen's d = 1.49 (identical to base!)
- **Significance**: p = 5.30e-14
- **Finding**: RLHF does not change the on/off-distribution dynamics

### Extension B: Temperature Comparison
- **temp=0.7 vs human**: Cohen's d = 0.98, p = 1.11e-08
- **temp=0 vs temp=0.7**: Cohen's d = 0.83, p = 4.14e-07
- **Finding**: Sampling adds variance. temp=0.7 still smoother than human but rougher than temp=0

## Success Criteria
- [x] Activation smoothness metrics computed for both conditions
- [x] Perplexity (CE loss) computed as ground truth for surprisingness
- [x] Statistical comparison (paired t-test) with effect sizes
- [x] Clear correlation between perplexity and smoothness (r=0.65, p<0.001)
- [x] Visualization of per-layer trajectory differences (in RESULTS.md table)

## Key Finding
**Hypothesis strongly supported across conditions:**
1. Model-generated text produces smoother activation trajectories than human text
2. Effect is robust to RLHF (identical d=1.49 for base and instruct)
3. Sampling temperature modulates effect: temp=0 > temp=0.7 > human (in smoothness)
4. Strong correlation with perplexity confirms this reflects "surprisingness"

## Observations
- Effect is consistent across layers 0-24 (all significant, Cohen's d > 0.6)
- Layer 25 shows no difference (output layer converges)
- Instruct model finds both human AND base-generated text more surprising
- Temperature 0.7 adds ~8 units of roughness vs greedy decoding
