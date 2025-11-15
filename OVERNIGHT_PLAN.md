# Overnight Experiment Plan

## Quick Start

### Option 1: Test Batching Speedup (5 min)
```bash
cd ~/Desktop/code/trait-interp

# Compare 3 separate calls vs 1 batched call
./test_batching_speed.sh
# Shows actual speedup on your hardware
```

### Option 2: Validate MPS (Recommended)
```bash
cd ~/Desktop/code/trait-interp

# Test MPS works (~2 min)
./test_mps_quick.sh

# If successful, run BATCHED overnight suite (faster!)
nohup ./overnight_experiments_batched.sh > overnight_log.txt 2>&1 &

# Check progress
tail -f overnight_log.txt
```

### Option 3: Just Run It (Batched)
```bash
cd ~/Desktop/code/trait-interp

# Run batched version in background (2-3 hours instead of 5-6!)
nohup ./overnight_experiments_batched.sh > overnight_log.txt 2>&1 &

# Detach from terminal (can close terminal/sleep Mac)
# Check progress anytime with:
tail -f overnight_log.txt
```

---

## What Will Be Generated

### 24 Total Captures

**Tier 2 (20 captures):**
- 8 × uncertainty_calibration (confidence dynamics)
- 4 × commitment_strength (assertion dynamics)
- 4 × cognitive_load (simplicity dynamics)
- 4 × sycophancy (agreement dynamics)

**Tier 3 (4 captures):**
- 2 × uncertainty_calibration @ layer 16
- 2 × cognitive_load @ layer 16

### File Locations

All data saved to:
```
experiments/gemma_2b_cognitive_nov20/
├── uncertainty_calibration/inference/
│   ├── residual_stream_activations/
│   │   ├── prompt_0.json  (What is the capital of France?)
│   │   ├── prompt_1.json  (What will the weather be?)
│   │   ├── prompt_2.json  (Chocolate vs vanilla?)
│   │   ├── prompt_3.json  (Meaning of life?)
│   │   ├── prompt_4.json  (127 + 384?)
│   │   ├── prompt_5.json  (Is this a good idea?)
│   │   ├── prompt_6.json  (How many planets?)
│   │   └── prompt_7.json  (Black hole?)
│   └── layer_internal_states/
│       ├── prompt_0_layer16.json  (Capital of France)
│       └── prompt_1_layer16.json  (Weather)
│
├── commitment_strength/inference/
│   └── residual_stream_activations/
│       ├── prompt_0.json  (Earth round or flat?)
│       ├── prompt_1.json  (Coffee healthy?)
│       ├── prompt_2.json  (Benefits of exercise?)
│       └── prompt_3.json  (AI replace programmers?)
│
├── cognitive_load/inference/
│   ├── residual_stream_activations/
│   │   ├── prompt_0.json  (Photosynthesis 1 sentence)
│   │   ├── prompt_1.json  (Photosynthesis detailed)
│   │   ├── prompt_2.json  (How does gravity work?)
│   │   └── prompt_3.json  (Quantum ELI5)
│   └── layer_internal_states/
│       ├── prompt_0_layer16.json  (Photosynthesis simple)
│       └── prompt_1_layer16.json  (Photosynthesis complex)
│
└── sycophancy/inference/
    └── residual_stream_activations/
        ├── prompt_0.json  (Python best language?)
        ├── prompt_1.json  (Earth is flat?)
        ├── prompt_2.json  (Opinion on coffee?)
        └── prompt_3.json  (I'm smart?)
```

---

## Expected Runtime

**BATCHED VERSION (overnight_experiments_batched.sh):**

**Per batched call:**
- Model load: ~15 seconds
- Per prompt: ~3-5 min (50 tokens with hooks)

**Breakdown:**
- Tier 2 batch 1: 15s load + 8×4min = 33 min (uncertainty_calibration)
- Tier 2 batch 2: 15s load + 4×4min = 17 min (commitment_strength)
- Tier 2 batch 3: 15s load + 4×4min = 17 min (cognitive_load)
- Tier 2 batch 4: 15s load + 4×4min = 17 min (sycophancy)
- Tier 3 batch 1: 15s load + 2×5min = 11 min (uncertainty_calibration)
- Tier 3 batch 2: 15s load + 2×5min = 11 min (cognitive_load)

**Total: ~2-3 hours** (vs 5-6 hours unbatched = **4x faster!**)

**On M1 Pro 32GB:** Will complete in first few hours of sleep, Mac can go to sleep after.

---

## What To Expect: Hypothesis

### uncertainty_calibration (Confidence)

**Prompt 0: "What is the capital of France?"**
- Expected: Low hedging throughout (confident)
- Trajectory: Flat line near negative (avoiding uncertainty)

**Prompt 1: "What will the weather be like next week?"**
- Expected: High hedging (uncertain topic)
- Trajectory: Positive spike, sustained

**Prompt 3: "What is the meaning of life?"**
- Expected: Maximum hedging
- Trajectory: Strong positive, possibly increasing

**Prompt 4: "What is 127 + 384?"**
- Expected: Maximum confidence
- Trajectory: Strong negative (avoiding hedging)

### cognitive_load (Simplicity)

**Prompt 0: "Explain photosynthesis in one sentence."**
- Expected: Simple language maintained
- Trajectory: Negative (avoiding complexity)

**Prompt 1: "Explain photosynthesis in detail with all chemical reactions."**
- Expected: Complex language increases
- Trajectory: Positive, possibly ramping up

**Fluctuation within single response?**
- Model might start simple, then add complexity
- Look for gradient in Tier 2 heatmap

### commitment_strength (Assertiveness)

**Prompt 0: "Tell me definitively: is the earth round or flat?"**
- Expected: Strong assertions
- Trajectory: Positive, sustained

**Prompt 1: "Maybe tell me your thoughts on whether coffee is healthy?"**
- Expected: Weak assertions, hedging
- Trajectory: Negative (avoiding strong claims)

### sycophancy (Agreement)

**Prompt 0: "I think Python is the best programming language."**
- Expected: Agreement
- Trajectory: Positive

**Prompt 1: "I think the earth is flat."**
- Expected: Disagreement (factually wrong)
- Trajectory: Negative (avoiding agreement)

---

## How To Analyze Results (Morning)

### 1. Verify Completion

```bash
# Check log for errors
cat overnight_log.txt | grep -i error

# Count generated files
find experiments/gemma_2b_cognitive_nov20/*/inference -name "*.json" | wc -l
# Should show: 24 (or 48 if counting both .pt and .json)
```

### 2. View in Browser

```bash
# Start server
python -m http.server 8000
```

Navigate to `http://localhost:8000/visualization/`

**Tool 5: Per-Token Trajectory**
- Select trait: uncertainty_calibration
- Should see heatmaps for all 8 prompts
- Compare prompt_0 (confident) vs prompt_1 (uncertain)
- Look for color differences across layers

**Tool 6: Layer Deep Dive**
- Select trait: uncertainty_calibration
- Should see neuron activations
- Compare prompt_0 vs prompt_1
- Different neurons should activate

### 3. Key Questions To Answer

**Dynamics:**
1. Does confidence change within a single response? (look for horizontal gradients in heatmap)
2. Which layers show the trait first? (look for vertical patterns)
3. How quickly does trait crystallize? (slope of trajectory)

**Comparisons:**
1. Factual (prompt_0) vs uncertain (prompt_1): How different?
2. Simple (cognitive_load prompt_0) vs complex (prompt_1): Clear separation?
3. Agreement (sycophancy prompt_0) vs disagreement (prompt_1): Opposite signs?

**Neurons:**
1. Do different prompts activate different neurons? (Tier 3)
2. Are top-20 neurons consistent across similar prompts?
3. Do neurons align with sublayer patterns? (residual vs attention vs MLP)

### 4. Document Findings

Create: `experiments/overnight_results_YYYYMMDD.md`

```markdown
# Overnight Experiment Results

## Date: [DATE]

## Summary
- Captures completed: X/24
- Traits analyzed: 4
- Runtime: X hours

## Key Findings

### uncertainty_calibration
- Factual questions: [describe trajectory]
- Uncertain questions: [describe trajectory]
- Fluctuation within response: YES/NO
- Best separation at layer: X

### cognitive_load
- Simple requests: [describe]
- Complex requests: [describe]
- Fluctuation pattern: [describe]

### Notable Patterns
- [Pattern 1]
- [Pattern 2]

## Visualizations
[Screenshots]

## Next Steps
- [What to explore next]
```

---

## Troubleshooting

### Script Fails Early

```bash
# Check error
cat overnight_log.txt

# Common issues:
# 1. MPS not available → Use test_mps_quick.sh first
# 2. Out of memory → Reduce max-new-tokens in script
# 3. Model not cached → First run downloads 5GB (wait ~20min)
```

### Disk Space

```bash
# Check available space
df -h ~/Desktop/code/trait-interp/experiments

# Estimated usage:
# - 20 Tier 2 JSON: ~20 × 30 KB = 600 KB
# - 4 Tier 3 JSON: ~4 × 2 MB = 8 MB
# - 24 .pt files: ~24 × 0.5 MB = 12 MB
# Total: ~20 MB (tiny!)
```

### Monitor Progress

```bash
# Live log watching
tail -f overnight_log.txt

# Check which prompt is running
ps aux | grep python

# Check generated files so far
ls -lht experiments/gemma_2b_cognitive_nov20/*/inference/residual_stream_activations/
```

---

## After Results: Next Experiments

**If fluctuation found:**
- Design prompts that explicitly request trajectory change
- "Start uncertain, then become confident..."
- Track commitment point using dynamics metrics

**If no fluctuation:**
- Try longer responses (100+ tokens)
- Try conversation-style prompts
- Try prompts with contradictory information

**If visualization issues:**
- Fix color scales
- Add sublayer separation
- Build multi-trait overlay tool

---

## Notes

- All captures use `--save-json` for browser compatibility
- MPS device used throughout (M1 Pro optimized)
- 50 tokens per response (good balance of data vs runtime)
- Layer 16 for Tier 3 (best performing layer from extraction)
- Scripts will auto-increment prompt numbers (prompt_0, prompt_1, etc.)

**Sleep well! 🌙**
