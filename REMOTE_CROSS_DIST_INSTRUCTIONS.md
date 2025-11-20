# Remote Cross-Distribution Testing Instructions

**Mission:** Run comprehensive cross-distribution validation on 19 traits overnight

**Expected Duration:** 8-10 hours
**Instance Type:** A100 40GB or 80GB
**Cost Estimate:** $6-12 total

---

## Overview

You are running cross-distribution testing to validate trait vector generalization. This tests whether vectors trained on one data distribution (instruction-based or natural elicitation) can successfully classify examples from another distribution.

**Current Status:**
- 2 traits fully validated (positivity, refusal): 416/416 test conditions each
- 20 traits with natural variants extracted
- Goal: Complete validation for 19 traits (up from 2)

**Test Matrix per Trait:**
- 4 quadrants: inst→inst, inst→nat, nat→inst, nat→nat
- 4 methods: mean_diff, probe, ica, gradient
- 26 layers: 0-25
- Total: 416 test conditions per trait

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install torch transformers accelerate scikit-learn pandas tqdm fire
```

### 2. Configure R2 (Cloud Storage)

The results need to be pushed to R2 storage when complete.

```bash
# Install rclone if not present
curl https://rclone.org/install.sh | sudo bash

# Configure R2
rclone config create r2 s3 \
  provider Cloudflare \
  access_key_id <YOUR_R2_ACCESS_KEY> \
  secret_access_key <YOUR_R2_SECRET_KEY> \
  endpoint https://<YOUR_ACCOUNT_ID>.r2.cloudflarestorage.com \
  acl private
```

**Note:** The user should provide R2 credentials if needed, or you can skip this step and just save results locally.

### 3. Pull Existing Data from R2

```bash
rclone sync r2:trait-interp-bucket/experiments/ experiments/ \
  --progress \
  --stats 5s \
  --transfers 32 \
  --checkers 32
```

This downloads the current experiment data including all extracted vectors.

---

## Execution Plan

### Phase 1: Quick Wins (40 min, optional test run)

Complete the two partial traits first to verify everything works:

```bash
# Uncertainty (50% → 100%)
python scripts/run_cross_distribution.py --trait uncertainty

# Formality (25% → 100%)
python scripts/run_cross_distribution.py --trait formality
```

**Expected output:**
- `results/cross_distribution_analysis/uncertainty_full_4x4_results.json` updated
- `results/cross_distribution_analysis/formality_full_4x4_results.json` updated

**Verification:**
```bash
# Should show 416/416 conditions for both traits
python3 << 'EOF'
import json
for trait in ['uncertainty', 'formality']:
    with open(f'results/cross_distribution_analysis/{trait}_full_4x4_results.json') as f:
        d = json.load(f)
    quads = d.get('quadrants', {})
    count = 0
    for qk in ['inst_inst', 'inst_nat', 'nat_inst', 'nat_nat']:
        if qk in quads:
            for method in ['mean_diff', 'probe', 'ica', 'gradient']:
                if method in quads[qk].get('methods', {}):
                    layers = quads[qk]['methods'][method].get('all_layers', [])
                    if len(layers) == 26:
                        count += 26
    print(f"{trait}: {count}/416 conditions")
EOF
```

### Phase 2: Complete All Instruction-Only Traits (6-8 hours)

Run cross-distribution testing on 13 traits that only have inst→inst data:

```bash
# Create list of traits to process
TRAITS=(
    abstractness
    authority
    compliance
    confidence
    context
    divergence
    futurism
    retrieval
    scope
    sequentiality
    sycophancy
    trust
    positivity_natural
)

# Run each trait sequentially
for trait in "${TRAITS[@]}"; do
    echo "=================================="
    echo "Processing: $trait"
    echo "=================================="

    python scripts/run_cross_distribution.py --trait "$trait"

    if [ $? -eq 0 ]; then
        echo "✓ $trait completed"
    else
        echo "✗ $trait failed"
        # Log the failure but continue
        echo "$trait" >> failed_traits.txt
    fi

    echo ""
done

echo "=================================="
echo "All traits processed!"
echo "=================================="

if [ -f failed_traits.txt ]; then
    echo "Failed traits:"
    cat failed_traits.txt
else
    echo "All traits completed successfully!"
fi
```

### Phase 3: Complete Partial Extraction Traits (2 hours)

These 4 traits need ICA extraction first, then cross-dist testing:

```bash
PARTIAL_TRAITS=(
    curiosity
    confidence_doubt
    defensiveness
    enthusiasm
)

for trait in "${PARTIAL_TRAITS[@]}"; do
    echo "=================================="
    echo "Processing: $trait"
    echo "=================================="

    # First, extract missing ICA vectors for inst_inst
    # (This step may need custom handling - check if needed)

    # Then run cross-dist
    python scripts/run_cross_distribution.py --trait "$trait"

    if [ $? -eq 0 ]; then
        echo "✓ $trait completed"
    else
        echo "✗ $trait failed"
        echo "$trait" >> failed_traits.txt
    fi

    echo ""
done
```

**Note:** The partial extraction traits may fail if ICA extraction isn't complete. You can skip these if they error out.

---

## Complete Overnight Script

Save this as `run_all_cross_dist.sh` and execute:

```bash
#!/bin/bash

# Complete Overnight Cross-Distribution Testing
# Expected duration: 8-10 hours on A100

set -e  # Exit on error
cd /path/to/trait-interp  # Update this path

LOG_FILE="cross_dist_overnight_$(date +%Y%m%d_%H%M%S).log"

echo "Starting cross-distribution testing at $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE"

# All traits with natural variants
ALL_TRAITS=(
    # Quick wins (already partial data)
    uncertainty
    formality

    # Instruction-only traits
    abstractness
    authority
    compliance
    confidence
    context
    divergence
    futurism
    retrieval
    scope
    sequentiality
    sycophancy
    trust
    positivity_natural

    # Partial extraction traits (may need ICA first)
    curiosity
    confidence_doubt
    defensiveness
    enthusiasm
)

TOTAL=${#ALL_TRAITS[@]}
COMPLETED=0
FAILED=0

for trait in "${ALL_TRAITS[@]}"; do
    ((COMPLETED++))

    echo "" | tee -a "$LOG_FILE"
    echo "======================================================================" | tee -a "$LOG_FILE"
    echo "[$COMPLETED/$TOTAL] Processing: $trait" | tee -a "$LOG_FILE"
    echo "======================================================================" | tee -a "$LOG_FILE"

    START_TIME=$(date +%s)

    if python scripts/run_cross_distribution.py --trait "$trait" 2>&1 | tee -a "$LOG_FILE"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "✓ $trait completed in ${DURATION}s" | tee -a "$LOG_FILE"
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "✗ $trait failed after ${DURATION}s" | tee -a "$LOG_FILE"
        echo "$trait" >> failed_traits.txt
        ((FAILED++))
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "OVERNIGHT TESTING COMPLETE!" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "Successful: $((TOTAL - FAILED))/$TOTAL traits" | tee -a "$LOG_FILE"

if [ $FAILED -gt 0 ]; then
    echo "Failed: $FAILED traits" | tee -a "$LOG_FILE"
    echo "Failed traits:" | tee -a "$LOG_FILE"
    cat failed_traits.txt | tee -a "$LOG_FILE"
else
    echo "All traits completed successfully!" | tee -a "$LOG_FILE"
fi

# Generate completion report
echo "" | tee -a "$LOG_FILE"
echo "Generating completeness report..." | tee -a "$LOG_FILE"
python3 << 'EOF' | tee -a "$LOG_FILE"
import json
from pathlib import Path

results_dir = Path('results/cross_distribution_analysis')
complete_count = 0

for file in results_dir.glob('*_full_4x4_results.json'):
    trait = file.stem.replace('_full_4x4_results', '')
    with open(file) as f:
        d = json.load(f)

    quads = d.get('quadrants', {})
    total_conditions = 0

    for qk in ['inst_inst', 'inst_nat', 'nat_inst', 'nat_nat']:
        if qk in quads:
            for method in ['mean_diff', 'probe', 'ica', 'gradient']:
                if method in quads[qk].get('methods', {}):
                    layers = quads[qk]['methods'][method].get('all_layers', [])
                    if len(layers) == 26:
                        total_conditions += 26

    if total_conditions == 416:
        complete_count += 1
        print(f"  ✓ {trait}: 416/416 complete")

print(f"\nTotal fully validated traits: {complete_count}")
EOF

echo "" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
```

**To run:**
```bash
chmod +x run_all_cross_dist.sh
nohup ./run_all_cross_dist.sh &
```

The `nohup` ensures it keeps running even if you disconnect from SSH.

---

## Monitoring Progress

### Check Currently Running

```bash
# See if script is running
ps aux | grep run_cross_distribution

# Watch log in real-time
tail -f cross_dist_overnight_*.log
```

### Check Completion Status

```bash
# Count completed traits
python3 << 'EOF'
import json
from pathlib import Path

results_dir = Path('results/cross_distribution_analysis')
complete = []
partial = []

for file in sorted(results_dir.glob('*_full_4x4_results.json')):
    trait = file.stem.replace('_full_4x4_results', '')
    with open(file) as f:
        d = json.load(f)

    quads = d.get('quadrants', {})
    count = 0

    for qk in ['inst_inst', 'inst_nat', 'nat_inst', 'nat_nat']:
        if qk in quads:
            for method in ['mean_diff', 'probe', 'ica', 'gradient']:
                if method in quads[qk].get('methods', {}):
                    layers = quads[qk]['methods'][method].get('all_layers', [])
                    if len(layers) == 26:
                        count += 26

    if count == 416:
        complete.append(trait)
    elif count > 0:
        partial.append((trait, count))

print(f"Fully complete: {len(complete)} traits")
for t in complete:
    print(f"  ✓ {t}")

print(f"\nPartial: {len(partial)} traits")
for t, c in partial:
    print(f"  {t}: {c}/416 ({c*100//416}%)")
EOF
```

---

## When Complete: Push Results Back

### Option A: Push to R2 (if configured)

```bash
rclone sync results/cross_distribution_analysis/ \
  r2:trait-interp-bucket/results/cross_distribution_analysis/ \
  --progress \
  --stats 5s \
  --transfers 32 \
  --checkers 32
```

### Option B: Compress and Download Manually

```bash
# Compress results
cd results/cross_distribution_analysis
tar -czf cross_dist_results_$(date +%Y%m%d).tar.gz *.json *.txt

# Download via scp (from your local machine)
scp user@remote:/path/to/results/cross_distribution_analysis/cross_dist_results_*.tar.gz .
```

---

## Troubleshooting

### Script Fails Immediately

**Check if vectors exist:**
```bash
# Should show ~3,200+ vector files
find experiments/gemma_2b_cognitive_nov20 -name "*.pt" -path "*/vectors/*" | wc -l
```

**Check if natural variants exist:**
```bash
# Should show 20 natural variant directories
find experiments/gemma_2b_cognitive_nov20 -name "*_natural" -type d | wc -l
```

### Out of Memory

A100 40GB should handle this fine. If OOM errors occur:
- Switch to A100 80GB
- Or reduce batch size in the cross-dist script (if it has that parameter)

### Specific Trait Fails

Skip it and continue with others. The script logs failures to `failed_traits.txt`.

### Script Hangs on One Trait

Set a timeout per trait:
```bash
timeout 30m python scripts/run_cross_distribution.py --trait "$trait"
```

This kills the process after 30 minutes if stuck.

---

## Expected Output

After completion, you should have:

**Updated Files:**
- `results/cross_distribution_analysis/{trait}_full_4x4_results.json` for each trait
- Each file contains 416 test conditions (4 quadrants × 4 methods × 26 layers)

**New Completeness:**
- Before: 2 traits fully validated (positivity, refusal)
- After: 19+ traits fully validated
- Coverage: 21% → 48%+ (or higher)

**Log Files:**
- `cross_dist_overnight_YYYYMMDD_HHMMSS.log` - complete execution log
- `failed_traits.txt` - list of any failed traits (if any)

---

## Cleanup & Verification

### Regenerate Data Index

```bash
python3 analysis/cross_distribution_scanner.py
```

This updates `results/cross_distribution_analysis/data_index.json` with the new results.

### Generate Updated Report

```bash
# Create new completeness report
python3 << 'PYEOF'
import json
from pathlib import Path

results_dir = Path('results/cross_distribution_analysis')

complete_4x4 = []
partial = {}
inst_only = []

for file in sorted(results_dir.glob('*_full_4x4_results.json')):
    trait = file.stem.replace('_full_4x4_results', '')

    with open(file) as f:
        d = json.load(f)

    quads = d.get('quadrants', {})
    filled_quads = []

    for qk in ['inst_inst', 'inst_nat', 'nat_inst', 'nat_nat']:
        if qk in quads:
            probe_data = quads[qk].get('methods', {}).get('probe', {})
            layers = probe_data.get('all_layers', [])
            if layers and len(layers) == 26:
                filled_quads.append(qk)

    if len(filled_quads) == 4:
        complete_4x4.append(trait)
    elif len(filled_quads) == 1 and filled_quads[0] == 'inst_inst':
        inst_only.append(trait)
    elif filled_quads:
        partial[trait] = filled_quads

print("=" * 70)
print("UPDATED COMPLETENESS AFTER OVERNIGHT RUN")
print("=" * 70)
print(f"Fully complete (416/416): {len(complete_4x4)} traits")
for t in complete_4x4:
    print(f"  ✓ {t}")

if partial:
    print(f"\nPartial: {len(partial)} traits")
    for t, quads in sorted(partial.items()):
        print(f"  {t}: {', '.join(quads)}")

if inst_only:
    print(f"\nInstruction-only: {len(inst_only)} traits")
PYEOF
```

---

## Final Checklist

Before terminating the instance:

- [ ] All 19 traits processed (check log file)
- [ ] Results pushed to R2 or downloaded locally
- [ ] `data_index.json` regenerated
- [ ] Completeness report generated
- [ ] Log files saved
- [ ] No active processes running (`ps aux | grep python`)

---

## Cost Tracking

**Actual cost calculation:**
```bash
# If running for exactly 8 hours on A100 40GB @ $0.75/hr:
echo "8 * 0.75" | bc
# = $6.00

# Check actual runtime from log:
head -n1 cross_dist_overnight_*.log  # Start time
tail -n10 cross_dist_overnight_*.log  # End time
```

---

## Questions?

If anything is unclear or fails:
1. Check the log file: `cross_dist_overnight_*.log`
2. Check failed traits: `cat failed_traits.txt`
3. Verify vectors exist: `find experiments -name "*.pt" | wc -l`
4. Check disk space: `df -h`

The script is designed to be fault-tolerant - it will continue even if individual traits fail.

---

**Good luck! 🚀**
