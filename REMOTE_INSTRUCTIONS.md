# Remote Extraction Instructions for Claude Code

## Context
You're on an 8x A100 Lambda Labs instance. Your job is to extract ALL missing trait vectors.

## Setup (if not done)
```bash
cd trait-interp
./scripts/sync_pull.sh  # Get latest data from R2
```

## Main Task: Extract All Missing Vectors

**Run this ONE command:**
```bash
./scripts/extract_all_missing_categorized.sh
```

**What it does:**
- Extracts 38 jobs total (19 traits × 2 variants: instruction + natural)
- Runs ALL 38 jobs in PARALLEL on 8x A100 (maximizes GPU usage!)
- Takes ~27 minutes total
- Auto-skips already-complete traits

**Expected output:**
```
================================================================
EXTRACT ALL MISSING VECTORS - 8x A100 OPTIMIZED
================================================================
Parallel jobs: 38 (ALL jobs at once!)
...
[timestamp] 🚀 Starting: behavioral/refusal
[timestamp] 🚀 Starting: behavioral/refusal_natural
[timestamp] 🚀 Starting: behavioral/compliance
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

## If Errors Occur

**Common issues:**
1. **Out of memory** - Script auto-handles this, will resume incomplete jobs
2. **Missing scenario files** - Some traits may not have natural scenarios (this is OK, they'll be skipped)
3. **CUDA errors** - Restart the script, it will resume from where it left off

**To check progress:**
```bash
# Count completed vectors
find experiments/gemma_2b_cognitive_nov20 -path "*/vectors/*.pt" | wc -l

# Check GPU utilization
nvidia-smi
```

**To retry after failure:**
```bash
# Just run again - it auto-skips completed traits
./scripts/extract_all_missing_categorized.sh
```

## After Completion

**Upload to R2:**
```bash
./scripts/sync_push.sh
```

**Then terminate the instance** (don't forget - costs $8.80/hr!)

## GPU Utilization

The script is optimized to MAX OUT all 8 GPUs:
- Each GPU can run ~12 models simultaneously
- 8 GPUs × 12 models = 96 parallel jobs capacity
- We have 38 jobs total, so ~40% GPU utilization is expected
- All jobs run in a SINGLE batch (not sequential)

**This is the fastest possible configuration for 38 jobs.**
