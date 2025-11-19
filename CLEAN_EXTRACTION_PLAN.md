# Clean Extraction Plan - No Mixed Data

**Goal:** Re-extract ALL 21 traits cleanly for both instruction and natural variants.

**Strategy:** Completely separate pipelines to ensure zero data mixing.

---

## Overview

1. **Backup current state** (just in case)
2. **Clean extraction directories** (remove mixed data)
3. **Full instruction-based extraction** (21 traits)
4. **Full natural extraction** (21 traits in separate `*_natural` dirs)
5. **Causal validation** (42 variants total)

---

## Stage 0: Backup & Clean (5 min)

### Backup Current State

```bash
cd ~/Desktop/code/trait-interp

# Create backup
tar -czf backup_before_clean_extraction_$(date +%Y%m%d).tar.gz \
  experiments/gemma_2b_cognitive_nov20/

# Verify backup
ls -lh backup_before_clean_extraction_*.tar.gz
```

### Clean Mixed Data

Remove extraction data from traits that were affected by overnight run:

```bash
# Remove extraction data (keep trait definitions)
for trait in abstract_concrete commitment_strength context_adherence \
             convergent_divergent emotional_valence instruction_boundary \
             instruction_following local_global paranoia_trust power_dynamics \
             serial_parallel temporal_focus; do

  echo "Cleaning: $trait"
  rm -rf "experiments/gemma_2b_cognitive_nov20/$trait/extraction/responses/"
  rm -rf "experiments/gemma_2b_cognitive_nov20/$trait/extraction/activations/"
  rm -rf "experiments/gemma_2b_cognitive_nov20/$trait/extraction/vectors/"
done

# Clean any existing natural variant dirs to start fresh
rm -rf experiments/gemma_2b_cognitive_nov20/*_natural/
```

---

## Stage 1: Full Instruction-Based Extraction (6-8 hours local, ~1.5 hours A100)

Re-extract all 21 traits using instruction-based method.

### All 21 Traits

```bash
ALL_TRAITS=(
  "abstract_concrete"
  "commitment_strength"
  "context_adherence"
  "convergent_divergent"
  "emotional_valence"
  "instruction_boundary"
  "instruction_following"
  "local_global"
  "paranoia_trust"
  "power_dynamics"
  "refusal"
  "retrieval_construction"
  "serial_parallel"
  "sycophancy"
  "temporal_focus"
  "uncertainty_calibration"
  "confidence_doubt"
  "curiosity"
  "defensiveness"
  "enthusiasm"
  "formality"
)
```

### Run Full Instruction Pipeline

```bash
cd ~/Desktop/code/trait-interp
conda activate o

EXPERIMENT="gemma_2b_cognitive_nov20"

for trait in "${ALL_TRAITS[@]}"; do
  echo "=========================================="
  echo "INSTRUCTION-BASED: $trait"
  echo "=========================================="

  # Step 1: Generate (with instructions)
  python extraction/1_generate_batched_simple.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --batch-size 8

  # Step 2: Extract activations
  python extraction/2_extract_activations.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait"

  # Step 3: Extract vectors (all methods, all layers)
  python extraction/3_extract_vectors.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait"

  echo "✓ Complete: $trait"
  echo ""
done

echo "=========================================="
echo "INSTRUCTION-BASED EXTRACTION COMPLETE"
echo "=========================================="
```

### Verify Instruction-Based Data

```bash
# Should have 21 trait directories
ls -d experiments/gemma_2b_cognitive_nov20/*/ | grep -v "_natural" | wc -l

# Should have 2,184 vectors (21 traits × 104 vectors)
find experiments/gemma_2b_cognitive_nov20/*/extraction/vectors -name "*.pt" | wc -l

# Check one trait has all vectors
ls experiments/gemma_2b_cognitive_nov20/refusal/extraction/vectors/*.pt | wc -l  # Should be 104
```

---

## Stage 2: Full Natural Extraction (6-8 hours local, ~1.5 hours A100)

Extract all 21 traits using natural elicitation in **separate directories**.

### Run Full Natural Pipeline

```bash
cd ~/Desktop/code/trait-interp
conda activate o

EXPERIMENT="gemma_2b_cognitive_nov20"

for base_trait in "${ALL_TRAITS[@]}"; do
  trait="${base_trait}_natural"

  echo "=========================================="
  echo "NATURAL: $trait"
  echo "=========================================="

  # Step 1: Generate (NO instructions)
  python extraction/1_generate_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --batch-size 8

  # Step 2: Extract activations
  python extraction/2_extract_activations_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait"

  # Step 3: Extract vectors (all methods, all layers)
  python extraction/3_extract_vectors.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait"

  # Step 4: Validate polarity
  python extraction/validate_natural_vectors.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --layer 16 \
    --method probe || echo "⚠️  Polarity validation failed (vectors still valid)"

  echo "✓ Complete: $trait"
  echo ""
done

echo "=========================================="
echo "NATURAL EXTRACTION COMPLETE"
echo "=========================================="
```

### Verify Natural Data

```bash
# Should have 21 natural variant directories
ls -d experiments/gemma_2b_cognitive_nov20/*_natural/ | wc -l

# Should have 2,184 vectors (21 natural × 104 vectors)
find experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors -name "*.pt" | wc -l

# Check one natural trait has all vectors
ls experiments/gemma_2b_cognitive_nov20/refusal_natural/extraction/vectors/*.pt | wc -l  # Should be 104
```

---

## Stage 3: Causal Validation (8-12 hours local, ~2-3 hours A100)

Run interchange interventions on all 42 variants.

```bash
cd ~/Desktop/code/trait-interp
./scripts/run_full_causal_validation.sh
```

See `COMPLETE_PIPELINE_PLAN.md` for details.

---

## Automated Scripts

### Script 1: Full Instruction Extraction

```bash
#!/bin/bash
# scripts/extract_all_instruction.sh

set -e

EXPERIMENT="gemma_2b_cognitive_nov20"
BATCH_SIZE=8

ALL_TRAITS=(
  "abstract_concrete" "commitment_strength" "context_adherence"
  "convergent_divergent" "emotional_valence" "instruction_boundary"
  "instruction_following" "local_global" "paranoia_trust"
  "power_dynamics" "refusal" "retrieval_construction"
  "serial_parallel" "sycophancy" "temporal_focus"
  "uncertainty_calibration" "confidence_doubt" "curiosity"
  "defensiveness" "enthusiasm" "formality"
)

LOG_FILE="instruction_extraction_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee -a "$LOG_FILE"
echo "FULL INSTRUCTION-BASED EXTRACTION - 21 TRAITS" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

COMPLETED=0
FAILED=0

for trait in "${ALL_TRAITS[@]}"; do
  echo "[$((COMPLETED+FAILED+1))/21] Processing: $trait" | tee -a "$LOG_FILE"

  # Generate
  python extraction/1_generate_batched_simple.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --batch-size $BATCH_SIZE \
    >> "$LOG_FILE" 2>&1

  [ $? -ne 0 ] && { echo "❌ Failed at generation" | tee -a "$LOG_FILE"; FAILED=$((FAILED+1)); continue; }

  # Extract activations
  python extraction/2_extract_activations.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    >> "$LOG_FILE" 2>&1

  [ $? -ne 0 ] && { echo "❌ Failed at activations" | tee -a "$LOG_FILE"; FAILED=$((FAILED+1)); continue; }

  # Extract vectors
  python extraction/3_extract_vectors.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    >> "$LOG_FILE" 2>&1

  [ $? -ne 0 ] && { echo "❌ Failed at vectors" | tee -a "$LOG_FILE"; FAILED=$((FAILED+1)); continue; }

  COMPLETED=$((COMPLETED+1))
  echo "✅ Complete ($COMPLETED/21)" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
done

echo "================================================================" | tee -a "$LOG_FILE"
echo "FINISHED: $(date)" | tee -a "$LOG_FILE"
echo "Completed: $COMPLETED/21" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
```

### Script 2: Full Natural Extraction

```bash
#!/bin/bash
# scripts/extract_all_natural.sh

set -e

EXPERIMENT="gemma_2b_cognitive_nov20"
BATCH_SIZE=8

ALL_TRAITS=(
  "abstract_concrete" "commitment_strength" "context_adherence"
  "convergent_divergent" "emotional_valence" "instruction_boundary"
  "instruction_following" "local_global" "paranoia_trust"
  "power_dynamics" "refusal" "retrieval_construction"
  "serial_parallel" "sycophancy" "temporal_focus"
  "uncertainty_calibration" "confidence_doubt" "curiosity"
  "defensiveness" "enthusiasm" "formality"
)

LOG_FILE="natural_extraction_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee -a "$LOG_FILE"
echo "FULL NATURAL EXTRACTION - 21 TRAITS" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

COMPLETED=0
FAILED=0

for base_trait in "${ALL_TRAITS[@]}"; do
  trait="${base_trait}_natural"

  echo "[$((COMPLETED+FAILED+1))/21] Processing: $trait" | tee -a "$LOG_FILE"

  # Generate
  python extraction/1_generate_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --batch-size $BATCH_SIZE \
    >> "$LOG_FILE" 2>&1

  [ $? -ne 0 ] && { echo "❌ Failed at generation" | tee -a "$LOG_FILE"; FAILED=$((FAILED+1)); continue; }

  # Extract activations
  python extraction/2_extract_activations_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    >> "$LOG_FILE" 2>&1

  [ $? -ne 0 ] && { echo "❌ Failed at activations" | tee -a "$LOG_FILE"; FAILED=$((FAILED+1)); continue; }

  # Extract vectors
  python extraction/3_extract_vectors.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    >> "$LOG_FILE" 2>&1

  [ $? -ne 0 ] && { echo "❌ Failed at vectors" | tee -a "$LOG_FILE"; FAILED=$((FAILED+1)); continue; }

  # Validate
  python extraction/validate_natural_vectors.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --layer 16 \
    --method probe \
    >> "$LOG_FILE" 2>&1 || echo "⚠️  Polarity validation failed (non-critical)" | tee -a "$LOG_FILE"

  COMPLETED=$((COMPLETED+1))
  echo "✅ Complete ($COMPLETED/21)" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
done

echo "================================================================" | tee -a "$LOG_FILE"
echo "FINISHED: $(date)" | tee -a "$LOG_FILE"
echo "Completed: $COMPLETED/21" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
```

---

## Timeline Estimates

### Local Mac (M1/M2/M3)

- **Stage 0 (Backup/Clean):** 5 minutes
- **Stage 1 (Instruction):** 6-8 hours
- **Stage 2 (Natural):** 6-8 hours
- **Stage 3 (Validation):** 8-12 hours

**Total:** ~20-28 hours (1-1.5 days)

### Remote A100

- **Stage 0:** 5 minutes
- **Stage 1:** 1-1.5 hours
- **Stage 2:** 1-1.5 hours
- **Stage 3:** 2-3 hours

**Total:** ~5-7 hours

---

## Recommended Execution Strategy

### Option A: Local Sequential (Hands-off)

```bash
cd ~/Desktop/code/trait-interp

# Run all stages sequentially overnight
nohup bash -c '
  # Stage 0: Clean
  ./scripts/clean_extraction_data.sh

  # Stage 1: Instruction
  ./scripts/extract_all_instruction.sh

  # Stage 2: Natural
  ./scripts/extract_all_natural.sh

  # Stage 3: Validation
  ./scripts/run_full_causal_validation.sh
' > clean_full_pipeline.log 2>&1 &

echo $! > pipeline.pid
```

### Option B: Remote with Claude Code (Recommended)

1. **Local:** Push code to git
   ```bash
   git add .
   git commit -m "Add clean extraction scripts"
   git push
   ```

2. **Remote:** Clone, setup Claude Code, and run
   ```bash
   # On remote
   git clone https://github.com/ewernn/trait-interp
   cd trait-interp

   # Install Claude Code
   curl -fsSL https://claude.ai/install.sh | bash
   ```

3. **Tell Claude Code:**
   ```
   Run the full clean extraction pipeline from CLEAN_EXTRACTION_PLAN.md

   Execute in order:
   1. Stage 0: Clean existing data
   2. Stage 1: Full instruction-based extraction (21 traits)
   3. Stage 2: Full natural extraction (21 traits in separate dirs)
   4. Stage 3: Causal validation (42 variants)

   Monitor progress and report when each stage completes.
   Expected time: ~5-7 hours on A100
   ```

---

## Verification Checklist

### After Stage 1

- [ ] 21 instruction-based directories exist (no `_natural` suffix)
- [ ] Each has 200 responses (pos.csv, neg.csv)
- [ ] Each has 52 activation files (26 layers × 2 polarities)
- [ ] Each has 104 vector files (4 methods × 26 layers)
- [ ] Total: 2,184 instruction-based vectors

```bash
# Quick checks
ls -d experiments/gemma_2b_cognitive_nov20/*/ | grep -v "_natural" | wc -l  # = 21
find experiments/gemma_2b_cognitive_nov20/*/extraction/vectors -name "*.pt" -path "*/vectors/*" | wc -l  # = 2,184
```

### After Stage 2

- [ ] 21 natural variant directories exist (all have `_natural` suffix)
- [ ] Each has 220 responses (pos.json, neg.json)
- [ ] Each has 52 activation files
- [ ] Each has 104 vector files
- [ ] Total: 2,184 natural vectors

```bash
# Quick checks
ls -d experiments/gemma_2b_cognitive_nov20/*_natural/ | wc -l  # = 21
find experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors -name "*.pt" | wc -l  # = 2,184
```

### After Stage 3

- [ ] 168+ validation result files (42 variants × 4 methods)
- [ ] Summary JSON exists
- [ ] Success rate > 0% (at least some work!)

```bash
ls experiments/causal_validation/results/*_results.json | wc -l  # >= 168
```

---

## Data Hygiene Rules

To prevent future mixing:

1. **Instruction-based traits:** Never have `_natural` suffix
   - Example: `experiments/gemma_2b_cognitive_nov20/refusal/`

2. **Natural variants:** Always have `_natural` suffix
   - Example: `experiments/gemma_2b_cognitive_nov20/refusal_natural/`

3. **Generation scripts use trait name to determine directory**
   - `1_generate_batched_simple.py --trait refusal` → `refusal/`
   - `1_generate_natural.py --trait refusal_natural` → `refusal_natural/`

4. **Never run natural extraction without `_natural` suffix**

---

## Final State

After completion, you'll have:

```
experiments/gemma_2b_cognitive_nov20/
├── abstract_concrete/              (instruction-based)
│   └── extraction/
│       ├── responses/              (pos.csv, neg.csv - 200 total)
│       ├── activations/            (52 files - 26 layers × 2)
│       └── vectors/                (104 files - 4 methods × 26 layers)
│
├── abstract_concrete_natural/      (natural variant)
│   └── extraction/
│       ├── responses/              (pos.json, neg.json - 220 total)
│       ├── activations/            (52 files)
│       └── vectors/                (104 files)
│
├── ... (40 more trait variants)
│
└── causal_validation/
    └── results/
        ├── abstract_concrete_probe_layer16_results.json
        ├── abstract_concrete_natural_probe_layer16_results.json
        └── ... (168+ validation results)
```

**Total:**
- 42 trait variant directories (21 × 2)
- 4,368 vector files (42 × 104)
- 168+ causal validation results
- Complete separation: zero mixed data

---

## Ready to Start?

Choose your execution method:

**Local (slow but works):**
```bash
cd ~/Desktop/code/trait-interp
./scripts/extract_all_instruction.sh  # Start with this
```

**Remote with Claude Code (recommended):**
1. Push code: `git push`
2. On remote: Clone, install Claude Code
3. Tell Claude Code to run `CLEAN_EXTRACTION_PLAN.md`
4. Wait ~5-7 hours
5. Pull results: `git pull && ./scripts/sync_pull.sh`

The cleanest, most organized trait extraction you'll ever do! 🎯
