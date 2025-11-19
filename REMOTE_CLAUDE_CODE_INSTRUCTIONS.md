# Instructions for Claude Code on Remote Instance

Hi Claude! You're running on a remote GPU instance. Here's what you need to do:

---

## Context

The user wants a **completely clean extraction** of all trait data:
- 21 traits total
- 2 variants each (instruction-based + natural)
- 42 total trait variants
- **Zero mixed data**

---

## Your Task

Run the complete clean extraction pipeline. This will:

1. **Clean** all existing extraction data (keeps trait definitions)
2. **Extract instruction-based** variants for all 21 traits
3. **Extract natural** variants for all 21 traits (in separate `*_natural` directories)
4. **Run causal validation** on all 42 variants

---

## How to Execute

### Simple One-Command Execution

```bash
cd /workspace/trait-interp  # Or wherever the repo is

# Run everything
./scripts/run_complete_clean_pipeline.sh
```

That's it! The script handles everything automatically.

---

## What to Monitor

The script will output progress to the terminal and create these log files:

1. **Master log:** `complete_pipeline_YYYYMMDD_HHMMSS.log`
2. **Stage logs:**
   - `instruction_extraction_YYYYMMDD_HHMMSS.log`
   - `natural_extraction_YYYYMMDD_HHMMSS.log`
   - `causal_validation_YYYYMMDD_HHMMSS.log`

### Progress Checks

While running, you can check progress:

```bash
# Quick status
tail -30 complete_pipeline_*.log

# Count completed instruction traits
ls -d experiments/gemma_2b_cognitive_nov20/*/extraction/vectors 2>/dev/null | wc -l

# Count completed natural traits
ls -d experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors 2>/dev/null | wc -l

# GPU usage
nvidia-smi
```

---

## Timeline Expectations

### Remote A100 (Expected)

- **Stage 0 (Clean):** 1 minute
- **Stage 1 (Instruction):** 1.5-2 hours
- **Stage 2 (Natural):** 1.5-2 hours
- **Stage 3 (Validation):** 2-3 hours

**Total:** ~5-7 hours

### If you want to run stages separately

Instead of the master script, you can run each stage individually:

```bash
# Stage 0: Clean
./scripts/clean_extraction_data.sh

# Stage 1: Instruction extraction
./scripts/extract_all_instruction.sh

# Stage 2: Natural extraction
./scripts/extract_all_natural.sh

# Stage 3: Causal validation
./scripts/run_full_causal_validation.sh
```

---

## Expected Final State

After successful completion:

```
experiments/gemma_2b_cognitive_nov20/
├── abstract_concrete/              (instruction-based)
│   └── extraction/
│       └── vectors/                (104 files: 4 methods × 26 layers)
├── abstract_concrete_natural/      (natural variant)
│   └── extraction/
│       └── vectors/                (104 files)
├── ... (40 more trait variants)
│
└── causal_validation/
    └── results/
        └── *.json                  (168+ validation results)
```

**Totals:**
- 42 trait variant directories (21 × 2)
- 4,368 vector files (42 × 104)
- 168+ causal validation results
- **Zero mixed data** (clean separation between instruction and natural)

---

## Verification Commands

After completion, verify results:

```bash
# Count instruction-based directories (should be 21)
ls -d experiments/gemma_2b_cognitive_nov20/*/ | grep -v "_natural" | wc -l

# Count natural variant directories (should be 21)
ls -d experiments/gemma_2b_cognitive_nov20/*_natural/ | wc -l

# Count instruction-based vectors (should be 2,184)
find experiments/gemma_2b_cognitive_nov20/*/extraction/vectors -name "*.pt" | wc -l

# Count natural vectors (should be 2,184)
find experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors -name "*.pt" | wc -l

# Count validation results (should be ≥168)
ls experiments/causal_validation/results/*_results.json | wc -l
```

---

## Troubleshooting

### If a trait fails

The scripts are designed to continue even if individual traits fail. Check the logs:

```bash
# Find which traits failed
grep "FAILED" instruction_extraction_*.log
grep "FAILED" natural_extraction_*.log
```

Failed traits can be re-run individually:

```bash
# Re-run single trait
TRAIT="refusal"

# Instruction-based
python extraction/1_generate_batched_simple.py --experiment gemma_2b_cognitive_nov20 --trait "$TRAIT"
python extraction/2_extract_activations.py --experiment gemma_2b_cognitive_nov20 --trait "$TRAIT"
python extraction/3_extract_vectors.py --experiment gemma_2b_cognitive_nov20 --trait "$TRAIT"

# Natural variant
python extraction/1_generate_natural.py --experiment gemma_2b_cognitive_nov20 --trait "${TRAIT}_natural"
python extraction/2_extract_activations_natural.py --experiment gemma_2b_cognitive_nov20 --trait "${TRAIT}_natural"
python extraction/3_extract_vectors.py --experiment gemma_2b_cognitive_nov20 --trait "${TRAIT}_natural"
```

### Out of memory

If you run out of GPU memory:
- Reduce batch size in the scripts (edit `BATCH_SIZE=8` → `BATCH_SIZE=4`)
- Or run traits sequentially instead of in batches

### Script permissions

If you get "permission denied":
```bash
chmod +x scripts/*.sh
```

---

## When Complete

Report to the user:

1. **Success/failure status** of each stage
2. **Final counts:**
   - Instruction-based vectors
   - Natural vectors
   - Validation results
3. **Any failed traits** (if applicable)
4. **Location of log files** for detailed review

---

## Questions?

If anything is unclear or you encounter issues, check:
1. The detailed logs (`*.log` files)
2. `CLEAN_EXTRACTION_PLAN.md` for full documentation
3. Individual script source code for specifics

---

## Ready to Start?

Just run:

```bash
cd /workspace/trait-interp
./scripts/run_complete_clean_pipeline.sh
```

Monitor progress with:
```bash
tail -f complete_pipeline_*.log
```

That's it! The pipeline is fully automated. Expected completion: ~5-7 hours on A100.

Good luck! 🚀
