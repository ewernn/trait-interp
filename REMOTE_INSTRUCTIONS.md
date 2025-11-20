# Remote Extraction - Full Instructions for Claude Code

## Your Task
You're on an 8x A100 80GB instance. Extract ALL missing trait vectors (38 jobs total) and push results to R2.

## Step 1: Install Dependencies
```bash
pip install torch transformers accelerate huggingface_hub pandas tqdm fire scikit-learn
```

## Step 2: Configure R2 (uses hardcoded credentials)
```bash
bash scripts/configure_r2.sh
```

## Step 3: Pull Existing Data from R2
```bash
./scripts/sync_pull.sh
```
This downloads existing experiments/ with 2,205 vectors (~2-3 min).

## Step 4: Extract ALL Missing Vectors (27 min)

**Run this ONE command:**
```bash
./scripts/extract_all_missing_categorized.sh
```

**What it does:**
- Extracts 38 jobs total (19 traits × 2 variants: instruction + natural)
- Runs ALL 38 jobs in PARALLEL (maximizes 8x A100 usage)
- Takes ~27 minutes total
- Auto-skips already-complete traits
- Auto-resumes on failures

**Expected output:**
```
================================================================
EXTRACT ALL MISSING VECTORS - 8x A100 OPTIMIZED
================================================================
Parallel jobs: 38 (ALL jobs at once!)
...
[timestamp] 🚀 Starting: behavioral/refusal
[timestamp] 🚀 Starting: behavioral/refusal_natural
... (all 38 start simultaneously)
...
[timestamp] ✅ Completed: behavioral/refusal (104 vectors)
...
================================================================
✅ ALL EXTRACTIONS COMPLETE
================================================================
Final vector count: 3952
Expected: 3,952 vectors (19 traits × 2 variants × 104)
```

## Step 5: Monitor Progress

```bash
# Count completed vectors
find experiments/gemma_2b_cognitive_nov20 -path "*/vectors/*.pt" | wc -l

# Check GPU utilization
nvidia-smi
```

**Expected GPU usage:** ~40% (38 jobs out of 96 GPU capacity)

## Step 6: If Errors Occur

**Common issues:**
1. **Out of memory** - Script handles this, will resume incomplete jobs
2. **Missing scenario files** - Some traits may skip natural variants (OK)
3. **CUDA errors** - Restart script, it auto-resumes

**To retry:**
```bash
# Just run again - auto-skips completed traits
./scripts/extract_all_missing_categorized.sh
```

## Step 7: Push Results to R2
```bash
./scripts/sync_push.sh
```
This uploads all completed vectors to R2.

## Step 8: Tell the User

When everything is done, tell the user:
- ✅ Extraction complete
- Final vector count
- They should terminate the instance now ($9.188/hr!)

---

## GPU Utilization Details

The script is optimized to MAX OUT all 8 GPUs:
- Each GPU can run ~12 models simultaneously
- 8 GPUs × 12 models = 96 parallel jobs capacity
- We have 38 jobs total, so ~40% utilization is expected
- All jobs run in a SINGLE batch (not sequential)

This is the fastest possible configuration for 38 jobs.

---

## Troubleshooting

If extraction fails completely:
1. Check logs in the trait-interp directory
2. Verify GPU access: `nvidia-smi`
3. Check disk space: `df -h`
4. Re-run extraction script (it auto-resumes)

If stuck, tell the user what went wrong and ask for help.
