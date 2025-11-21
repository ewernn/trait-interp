# Remote GPU Extraction Guide - 7 Traits in Parallel

**Hardware:** A100 80GB
**Strategy:** Run all 7 traits simultaneously
**Expected Time:** 1.7-3.4 hours
**Expected Cost:** $5.95-11.90

---

## Quick Start (TL;DR)

```bash
# 1. Setup (15 min)
pip install torch transformers accelerate fire scikit-learn pandas tqdm
export HF_TOKEN="<provided-by-user>"
huggingface-cli login --token $HF_TOKEN
./scripts/configure_r2.sh
./scripts/sync_pull.sh

# 2. Optimize (5 min)
sed -i 's/batch_size=4/batch_size=8/g' extraction/1_generate_natural.py
# Create run_parallel_extractions.sh (see below)

# 3. Extract (1.7-3.4 hours)
nohup ./run_parallel_extractions.sh > extraction.out 2>&1 &
watch -n 5 nvidia-smi  # Monitor memory

# 4. Sync back (15 min)
./scripts/sync_push.sh
```

---

## Phase 1: Setup (15 minutes)

### Install Dependencies

```bash
# Python packages
pip install torch transformers accelerate fire scikit-learn pandas tqdm

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')"
# Should output: GPU: NVIDIA A100-SXM4-80GB, VRAM: 80.0GB
```

### Configure HuggingFace

```bash
# User will provide token
export HF_TOKEN="hf_xxxxxxxxxxxxx"
huggingface-cli login --token $HF_TOKEN

# Verify login
huggingface-cli whoami
```

### Configure R2 and Sync Repository

```bash
# Configure rclone for R2 (credentials in script)
./scripts/configure_r2.sh

# Test connection
rclone lsd r2:
# Should show bucket: trait-interp

# Pull repository from R2
./scripts/sync_pull.sh

# Verify experiment exists
ls -la experiments/gemma_2b_cognitive_nov21/extraction/
# Should show: cognitive_state, behavioral_tendency, expression_style

# Verify 7 prompt files
find experiments/gemma_2b_cognitive_nov21/extraction -name "positive.txt" | wc -l
# Should output: 7
```

---

## Phase 2: Optimize for Parallel Extraction (5 minutes)

### Increase Batch Size

```bash
# Change batch_size from 4 to 8 for faster generation
sed -i 's/batch_size=4/batch_size=8/g' extraction/1_generate_natural.py

# Verify change
grep "batch_size" extraction/1_generate_natural.py | head -3
```

### Create Parallel Extraction Script

```bash
cat > run_parallel_extractions.sh << 'EOF'
#!/bin/bash
# 80GB A100: Run all 7 traits in parallel
set -e

EXPERIMENT="gemma_2b_cognitive_nov21"
LOG_DIR="logs_parallel"
mkdir -p "$LOG_DIR"

ALL_TRAITS=(
    "cognitive_state/uncertainty_expression"
    "cognitive_state/search_activation"
    "cognitive_state/pattern_completion"
    "cognitive_state/correction_impulse"
    "behavioral_tendency/retrieval"
    "behavioral_tendency/defensiveness"
    "expression_style/positivity"
)

extract_trait() {
    TRAIT=$1
    TRAIT_LOG="$LOG_DIR/${TRAIT//\//_}.log"

    echo "[$(date)] Starting $TRAIT" | tee -a "$TRAIT_LOG"

    python extraction/1_generate_natural.py \
        --experiment "$EXPERIMENT" \
        --trait "$TRAIT" 2>&1 | tee -a "$TRAIT_LOG"

    python extraction/2_extract_activations_natural.py \
        --experiment "$EXPERIMENT" \
        --trait "$TRAIT" 2>&1 | tee -a "$TRAIT_LOG"

    python extraction/3_extract_vectors_natural.py \
        --experiment "$EXPERIMENT" \
        --trait "$TRAIT" \
        --methods all \
        --layers all 2>&1 | tee -a "$TRAIT_LOG"

    python extraction/validate_natural_vectors.py \
        --experiment "$EXPERIMENT" \
        --trait "$TRAIT" 2>&1 | tee -a "$TRAIT_LOG"

    echo "[$(date)] ✅ $TRAIT complete" | tee -a "$TRAIT_LOG"
}

export -f extract_trait
export EXPERIMENT
export LOG_DIR

echo "========================================"
echo "80GB A100: Running 7 traits in parallel"
echo "Start time: $(date)"
echo "Expected: 1.7-3.4 hours"
echo "========================================"

# Try GNU parallel first, fallback to background jobs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel"
    printf '%s\n' "${ALL_TRAITS[@]}" | parallel -j 7 extract_trait {}
else
    echo "Using background jobs"
    for TRAIT in "${ALL_TRAITS[@]}"; do
        extract_trait "$TRAIT" &
    done
    wait
fi

echo "========================================"
echo "✅ All extractions complete!"
echo "End time: $(date)"
echo "========================================"

# Summary
VECTOR_COUNT=$(find experiments/$EXPERIMENT -name "*.pt" -type f | wc -l)
echo "Vector files created: $VECTOR_COUNT (expected: ~728)"
EOF

chmod +x run_parallel_extractions.sh
```

### Create Memory Monitor Script

```bash
cat > monitor_memory.sh << 'EOF'
#!/bin/bash
# Monitor GPU memory and alert if too high
while true; do
    clear
    echo "=== GPU Memory Usage ($(date +%H:%M:%S)) ==="
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    echo ""
    echo "Processes:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv

    if [ "$USED" -gt 75000 ]; then
        echo ""
        echo "⚠️  WARNING: Memory >75GB! May need to kill one process."
    elif [ "$USED" -gt 70000 ]; then
        echo ""
        echo "⚠️  CAUTION: Memory >70GB. Watch closely."
    fi

    sleep 5
done
EOF

chmod +x monitor_memory.sh
```

---

## Phase 3: Run Extraction (1.7-3.4 hours)

### Start Extraction

```bash
# Start in background
nohup ./run_parallel_extractions.sh > extraction.out 2>&1 &

# Save PID
echo $! > extraction.pid
echo "Extraction started with PID: $(cat extraction.pid)"

# Start memory monitor (in separate terminal/tmux pane)
./monitor_memory.sh
```

### Monitor Progress

**Option 1: Watch overall output**
```bash
tail -f extraction.out
```

**Option 2: Watch GPU stats**
```bash
watch -n 5 nvidia-smi
# Should see:
# - 7 python processes
# - GPU Util: 60-85%
# - VRAM: 63-70GB / 80GB
```

**Option 3: Check individual trait logs**
```bash
# List logs
ls logs_parallel/

# Watch specific trait
tail -f logs_parallel/cognitive_state_uncertainty_expression.log

# Count completed
grep "✅.*complete" logs_parallel/*.log | wc -l
# Should reach 7
```

**Option 4: Check vector file progress**
```bash
# Count vectors created (updates in real-time)
watch -n 60 'find experiments/gemma_2b_cognitive_nov21 -name "*.pt" | wc -l'
# Should reach ~728 (7 traits × 104 vectors)
```

### Expected Timeline

| Time | Event | GPU Util | VRAM Used |
|------|-------|----------|-----------|
| 0:00 | All 7 traits start | 65-80% | 10GB → 65GB |
| 0:15 | Steady state generation | 70-85% | 63-68GB |
| 1:30 | First traits finishing | 40-60% | 35-50GB |
| 1:45 | Best case: All 7 done ✅ | 0% | 5GB |
| 3:30 | Conservative: All done ✅ | 0% | 5GB |

---

## Phase 4: Handle Memory Issues (If Needed)

### If VRAM Exceeds 75GB

**Step 1: Identify highest memory process**
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sort -t',' -k2 -rn
# Shows processes sorted by memory (highest first)
```

**Step 2: Kill highest memory process**
```bash
# Get PID of highest memory process
PID=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sort -t',' -k2 -rn | head -1 | cut -d',' -f1)

# Kill it
kill $PID
echo "Killed process $PID to free memory"

# Continue with 6 parallel (still better than 3)
```

**Step 3: Note which trait was killed**
```bash
# Check which trait failed
grep -L "✅.*complete" logs_parallel/*.log
# This trait will need manual retry
```

**Step 4: Retry killed trait after others finish**
```bash
# Wait for remaining 6 to finish
wait

# Manually run the killed trait
KILLED_TRAIT="cognitive_state/uncertainty_expression"  # Example
python extraction/1_generate_natural.py --experiment gemma_2b_cognitive_nov21 --trait $KILLED_TRAIT
python extraction/2_extract_activations_natural.py --experiment gemma_2b_cognitive_nov21 --trait $KILLED_TRAIT
python extraction/3_extract_vectors_natural.py --experiment gemma_2b_cognitive_nov21 --trait $KILLED_TRAIT --methods all --layers all
python extraction/validate_natural_vectors.py --experiment gemma_2b_cognitive_nov21 --trait $KILLED_TRAIT
```

### Alternative: Reduce Parallelism Preemptively

If you see memory consistently >70GB in first 5 minutes:

```bash
# Kill the extraction
kill $(cat extraction.pid)

# Edit script to run 6 parallel instead of 7
sed -i 's/parallel -j 7/parallel -j 6/g' run_parallel_extractions.sh

# Restart
nohup ./run_parallel_extractions.sh > extraction.out 2>&1 &
```

---

## Phase 5: Verify Results (10 minutes)

### Check Completion Status

```bash
# Count vector files (should be ~728)
find experiments/gemma_2b_cognitive_nov21 -name "*.pt" -type f | wc -l

# Check each trait has vectors directory
find experiments/gemma_2b_cognitive_nov21 -name "vectors" -type d
# Should show 7 directories

# Check for errors in logs
grep -r "ERROR\|Failed\|Traceback" logs_parallel/
# Should be empty or minimal

# View completion messages
grep "✅.*complete" logs_parallel/*.log
# Should show 7 lines
```

### Verify Vector Quality

```bash
# Check one trait's validation
cat experiments/gemma_2b_cognitive_nov21/extraction/cognitive_state/uncertainty_expression/vectors/validation_summary.json
# Should show high accuracy and good separation

# List all validation summaries
find experiments/gemma_2b_cognitive_nov21 -name "validation_summary.json" -exec cat {} \;
```

### Check Disk Usage

```bash
# Size of experiment
du -sh experiments/gemma_2b_cognitive_nov21/
# Expected: ~500MB-1GB (vectors + responses)

# Activation files are large (optional to keep)
du -sh experiments/gemma_2b_cognitive_nov21/extraction/*/activations/
# Can delete these after vectors are confirmed working
```

---

## Phase 6: Sync Results Back (15 minutes)

### Upload to R2

```bash
# Sync experiment back to R2
./scripts/sync_push.sh

# This uploads:
# - All vector files (~300MB)
# - Response CSVs (~100MB)
# - Validation metadata (~1MB)
# - Activation files (~500MB, optional)
```

### Verify Upload

```bash
# Check R2 has the vectors
rclone ls r2:trait-interp/experiments/gemma_2b_cognitive_nov21/extraction/ | grep vectors | wc -l
# Should show 7 (one per trait)

# Check total files uploaded
rclone ls r2:trait-interp/experiments/gemma_2b_cognitive_nov21/ | wc -l
# Should show 800+ files
```

### Optional: Clean Up Large Files Before Sync

```bash
# Activation files are intermediate (can regenerate from responses)
# Delete if disk space needed or to reduce sync time

rm -rf experiments/gemma_2b_cognitive_nov21/extraction/*/activations/

# Re-sync without activations (faster)
./scripts/sync_push.sh
```

---

## Success Criteria Checklist

- [ ] GPU: A100 80GB detected
- [ ] HF token configured
- [ ] R2 sync successful (7 prompt files found)
- [ ] Batch size increased to 8
- [ ] Parallel script created
- [ ] All 7 traits started
- [ ] GPU utilization 60-85%
- [ ] VRAM usage <75GB (or handled if exceeded)
- [ ] All 7 traits completed (check logs)
- [ ] ~728 vector files created
- [ ] No critical errors in logs
- [ ] Validation summaries look good
- [ ] Results synced back to R2

---

## Troubleshooting

### Extraction Fails on a Trait

```bash
# Find which trait failed
grep -l "ERROR\|Failed" logs_parallel/*.log

# Check the error
cat logs_parallel/cognitive_state_uncertainty_expression.log | tail -50

# Manually retry just that trait
python extraction/1_generate_natural.py --experiment gemma_2b_cognitive_nov21 --trait cognitive_state/uncertainty_expression
python extraction/2_extract_activations_natural.py --experiment gemma_2b_cognitive_nov21 --trait cognitive_state/uncertainty_expression
python extraction/3_extract_vectors_natural.py --experiment gemma_2b_cognitive_nov21 --trait cognitive_state/uncertainty_expression --methods all --layers all
python extraction/validate_natural_vectors.py --experiment gemma_2b_cognitive_nov21 --trait cognitive_state/uncertainty_expression
```

### Model Download Fails

```bash
# Pre-download model
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoModelForCausalLM.from_pretrained('google/gemma-2-2b-it'); \
  AutoTokenizer.from_pretrained('google/gemma-2-2b-it')"
```

### Out of Disk Space

```bash
# Check space
df -h

# Clean model cache if needed
rm -rf ~/.cache/huggingface/hub/models--*

# Delete activation files (can regenerate)
rm -rf experiments/gemma_2b_cognitive_nov21/extraction/*/activations/
```

### Sync Fails

```bash
# Test R2 connection
rclone lsd r2:

# Check rclone config
cat ~/.config/rclone/rclone.conf | grep -A5 "\[r2\]"

# Manual sync with progress
rclone sync experiments/gemma_2b_cognitive_nov21 \
  r2:trait-interp/experiments/gemma_2b_cognitive_nov21 \
  -P --transfers 4
```

---

## Expected Outcomes

### Best Case (70% probability)
- All 7 traits run smoothly
- VRAM stays at 63-68GB
- Time: 1.7 hours
- Cost: $5.95

### Good Case (25% probability)
- Need to kill 1 process due to memory
- Continue with 6 parallel
- Time: 3.4 hours
- Cost: $11.90

### Acceptable Case (5% probability)
- Need to reduce to 6 parallel upfront
- Time: 3.4 hours
- Cost: $11.90

---

## Post-Extraction

After successful extraction, the local machine can:

1. **Pull results**
   ```bash
   ./scripts/sync_pull.sh
   ```

2. **Verify vectors**
   ```bash
   find experiments/gemma_2b_cognitive_nov21 -name "*.pt" | wc -l
   # Should show ~728
   ```

3. **Run inference tests**
   ```bash
   python inference/monitor_dynamics.py \
     --experiment gemma_2b_cognitive_nov21 \
     --prompts "Test prompt"
   ```

4. **Link old traits** (optional - for confidence, context, formality)
   ```bash
   # These 3 traits use vectors from nov20 experiment
   # They're in archive/gemma_2b_cognitive_nov20/extraction/
   ```

---

## Summary

**Hardware:** A100 80GB
**Parallelism:** 7 traits simultaneously
**Optimization:** batch_size=8, parallel -j 7
**Time:** 1.7-3.4 hours
**Cost:** $5.95-11.90
**Output:** ~728 vector files across 7 traits

**Key monitoring:** Watch VRAM, if >75GB kill one process and continue.

**Success:** 7/7 traits complete, vectors validated, synced to R2.
