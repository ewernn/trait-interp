# Instructions for Claude Code on Remote A100

Hi Claude! You're running on a remote A100 GPU instance. Here's what you need to do:

## Context

The repository has been cloned and data synced from R2. You have:
- Natural scenarios for 12 traits in `extraction/natural_scenarios/`
- Existing experiment data in `experiments/gemma_2b_cognitive_nov20/`
- GPU: 80GB A100 (verify with `nvidia-smi`)

## Your Task

Run the natural elicitation pipeline for 12 traits. This will take ~3 hours.

**ONE COMMAND:**
```bash
./scripts/run_all_natural_extraction_a100.sh
```

This script will:
1. Generate responses from natural scenarios (batch_size=64, optimized for A100)
2. Extract activations from all 26 layers
3. Extract vectors using all 4 methods (mean_diff, probe, ICA, gradient)
4. Run cross-distribution testing (4×4 matrices)

For all 12 traits:
- abstract_concrete
- commitment_strength
- context_adherence
- convergent_divergent
- instruction_boundary
- local_global
- paranoia_trust
- power_dynamics
- retrieval_construction
- serial_parallel
- sycophancy
- temporal_focus

## Setup (if needed)

**Check environment:**
```bash
conda activate trait  # Should already be active
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
nvidia-smi
```

**If environment doesn't exist:**
```bash
conda create -n trait python=3.11 -y
conda activate trait
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate huggingface_hub pandas tqdm fire scikit-learn
```

**Check .env file exists:**
```bash
cat .env
# Should show HF_TOKEN and other API keys
```

## Monitoring Progress

Track completed traits:
```bash
watch -n 10 'ls experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/ 2>/dev/null | grep "_natural" | wc -l'
```

Expected: Number should increase from 0 to 12 over ~3 hours

## Expected Output

After completion, you should have:

**New directories:**
```
experiments/gemma_2b_cognitive_nov20/{trait}_natural/extraction/
├── responses/ (pos.json, neg.json)
├── activations/ (52 .pt files: pos/neg × 26 layers)
└── vectors/ (104 .pt files: 4 methods × 26 layers)
```

**New result files:**
```
results/cross_distribution_analysis/{trait}_full_4x4_results.json
```

12 traits × 1 result file = 12 new files

## If Something Fails

**Check which trait failed:**
Look at the last output before error.

**Resume from failed trait:**
Edit `scripts/run_all_natural_extraction_a100.sh`, remove completed traits from TRAITS array, re-run.

**Test one trait manually:**
```bash
python extraction/1_generate_natural.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait abstract_concrete \
  --batch-size 64 \
  --device cuda
```

## When Complete

1. Verify 12 new cross-distribution results:
```bash
ls results/cross_distribution_analysis/*_full_4x4_results.json | wc -l
# Should show 19 total (7 existing + 12 new)
```

2. Push results to R2:
```bash
./scripts/sync_push.sh
```

3. Create summary report showing top 5 layers per method for new traits.

## Success Criteria

✅ 12 new `{trait}_natural` directories with vectors
✅ 12 new cross-distribution result files
✅ All results backed up to R2
✅ Summary analysis of cross-distribution patterns

---

**Start with:** `./scripts/run_all_natural_extraction_a100.sh`

Let me know when it's done or if you hit any errors!
