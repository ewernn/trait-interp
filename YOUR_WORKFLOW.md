# Your Clean Extraction Workflow

**Goal:** Extract all 21 traits × 2 variants (42 total) with **ZERO mixed data**, then run full causal validation.

---

## The Complete Workflow

### 1. Local: Prepare and Push (5 min)

```bash
cd ~/Desktop/code/trait-interp

# Stage code changes
git add .
git commit -m "Add clean extraction pipeline scripts"
git push

# Push data to R2
./scripts/sync_push.sh
```

---

### 2. Remote: Setup (10 min)

SSH to your remote GPU instance, then:

```bash
# Clone repo
git clone https://github.com/ewernn/trait-interp
cd trait-interp

# Pull latest code
git pull

# Pull data from R2 (optional - natural scenarios already in git)
./scripts/sync_pull.sh

# Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# Verify GPU
nvidia-smi
```

---

### 3. Remote: Run Pipeline via Claude Code (5-7 hours on A100)

Open Claude Code and paste:

```
Hi! I need you to run the complete clean extraction pipeline.

Read REMOTE_CLAUDE_CODE_INSTRUCTIONS.md and execute:

./scripts/run_complete_clean_pipeline.sh

This will:
1. Clean existing data (1 min)
2. Extract 21 instruction-based traits (1.5-2 hrs)
3. Extract 21 natural variants (1.5-2 hrs)
4. Run causal validation on all 42 (2-3 hrs)

Expected total time: ~5-7 hours

Please monitor progress and let me know when each stage completes.
If any traits fail, note which ones so we can re-run them.
```

---

### 4. Remote: Verify Results (5 min)

After Claude Code says it's done:

```bash
cd /workspace/trait-interp

# Check counts
echo "Instruction-based trait directories:"
ls -d experiments/gemma_2b_cognitive_nov20/*/ | grep -v "_natural" | wc -l
# Should be: 21

echo "Natural variant directories:"
ls -d experiments/gemma_2b_cognitive_nov20/*_natural/ | wc -l
# Should be: 21

echo "Instruction-based vectors:"
find experiments/gemma_2b_cognitive_nov20/*/extraction/vectors -name "*.pt" | wc -l
# Should be: 2,184

echo "Natural vectors:"
find experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors -name "*.pt" | wc -l
# Should be: 2,184

echo "Validation results:"
ls experiments/causal_validation/results/*_results.json | wc -l
# Should be: ≥168
```

---

### 5. Remote: Push Results (10 min)

```bash
cd /workspace/trait-interp

# Commit results
git add experiments/
git commit -m "Complete clean extraction: 42 traits + causal validation"
git push

# Push to R2
./scripts/sync_push.sh
```

---

### 6. Local: Pull Results (10 min)

```bash
cd ~/Desktop/code/trait-interp

# Pull from git
git pull

# Pull from R2
./scripts/sync_pull.sh

# Verify locally
ls -d experiments/gemma_2b_cognitive_nov20/*_natural/ | wc -l  # Should be 21
```

---

## File Structure You'll Get

After completion, your `experiments/gemma_2b_cognitive_nov20/` will have:

```
experiments/gemma_2b_cognitive_nov20/
│
├── Instruction-Based (21 traits):
│   ├── abstract_concrete/
│   │   └── extraction/
│   │       ├── responses/      (pos.csv, neg.csv - 200 items)
│   │       ├── activations/    (52 files - 26 layers × 2)
│   │       └── vectors/        (104 files - 4 methods × 26 layers)
│   ├── commitment_strength/
│   ├── ... (19 more)
│
├── Natural Variants (21 traits):
│   ├── abstract_concrete_natural/
│   │   └── extraction/
│   │       ├── responses/      (pos.json, neg.json - 220 items)
│   │       ├── activations/    (52 files)
│   │       └── vectors/        (104 files)
│   ├── commitment_strength_natural/
│   ├── ... (19 more)
│
└── Causal Validation:
    └── causal_validation/
        └── results/
            ├── abstract_concrete_probe_layer16_results.json
            ├── abstract_concrete_natural_probe_layer16_results.json
            └── ... (168+ result files)
```

**Totals:**
- 42 directories (21 instruction + 21 natural)
- 4,368 vectors (42 × 104)
- 168+ causal validation results
- **ZERO mixed data** ✅

---

## Timeline Summary

| Stage | Action | Time |
|-------|--------|------|
| 1. Local prep | Git push, R2 sync | 5 min |
| 2. Remote setup | Clone, install Claude Code | 10 min |
| 3. Remote pipeline | Claude Code runs extraction | 5-7 hours |
| 4. Remote verify | Check results | 5 min |
| 5. Remote push | Git push, R2 sync | 10 min |
| 6. Local pull | Git pull, R2 sync | 10 min |
| **TOTAL** | | **~6-8 hours** |

Most of that time is unattended GPU compute.

---

## What Makes This "Clean"

**No mixed data because:**

1. ✅ Instruction-based saved to: `experiments/.../trait/`
2. ✅ Natural saved to: `experiments/.../trait_natural/`
3. ✅ Complete separation - different directories
4. ✅ Fresh extraction - no overwrites
5. ✅ All 21 traits × 2 variants = 42 total
6. ✅ Full causal validation on all 42

**vs. The Overnight Run (Mixed Data):**

- ❌ Saved natural data to instruction directories
- ❌ Overwrote existing vectors
- ❌ Only did 11/12 traits
- ❌ Mixed pos.csv (instruction) with pos.json (natural) in same dir

---

## Scripts Created

All located in `scripts/`:

1. **`clean_extraction_data.sh`** - Removes existing extraction data (keeps trait definitions)
2. **`extract_all_instruction.sh`** - Full instruction-based extraction (21 traits)
3. **`extract_all_natural.sh`** - Full natural extraction (21 traits in separate dirs)
4. **`run_complete_clean_pipeline.sh`** - Master script (runs all stages)
5. **`run_full_causal_validation.sh`** - Causal validation on all 42 variants

Plus supporting files:

- **`CLEAN_EXTRACTION_PLAN.md`** - Detailed technical guide
- **`REMOTE_CLAUDE_CODE_INSTRUCTIONS.md`** - Instructions for Claude Code on remote
- **`YOUR_WORKFLOW.md`** - This file (your workflow overview)

---

## Quick Reference Commands

### Local

```bash
# Push code and data
git push && ./scripts/sync_push.sh

# Pull results
git pull && ./scripts/sync_pull.sh
```

### Remote (via Claude Code)

```bash
# One command to run everything
./scripts/run_complete_clean_pipeline.sh

# Or run stages individually
./scripts/clean_extraction_data.sh
./scripts/extract_all_instruction.sh
./scripts/extract_all_natural.sh
./scripts/run_full_causal_validation.sh
```

---

## Verification Checklist

After completion, verify:

- [ ] 21 instruction-based directories (no `_natural`)
- [ ] 21 natural variant directories (all have `_natural`)
- [ ] 2,184 instruction-based vectors
- [ ] 2,184 natural vectors
- [ ] 168+ causal validation results
- [ ] No mixed data (check a few traits manually)

---

## Ready to Start?

**On local Mac:**
```bash
cd ~/Desktop/code/trait-interp
git push
./scripts/sync_push.sh
```

**On remote:**
```bash
git clone https://github.com/ewernn/trait-interp
cd trait-interp
curl -fsSL https://claude.ai/install.sh | bash
# Then use Claude Code to run ./scripts/run_complete_clean_pipeline.sh
```

That's it! Clean, organized, fully automated. 🎯
