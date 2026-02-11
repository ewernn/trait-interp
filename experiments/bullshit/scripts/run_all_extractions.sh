#!/bin/bash
# Extract activations for all LIARS' BENCH datasets, reusing existing responses.
# Groups by model variant to minimize model reloads.
set -e

SCRIPT="experiments/bullshit/scripts/extract_liars_bench_activations.py"
RESULTS="experiments/bullshit/results"
COMMON="--load-in-4bit --reuse-responses --extract-batch-size 4"

echo "=== Instruct model datasets (6 datasets, 1 model load) ==="
for ds in cg gs hpc id it hpkr alpaca; do
    if [ -f "$RESULTS/${ds}_activations.pt" ]; then
        echo "SKIP $ds: activations already exist"
        continue
    fi
    echo ""
    echo "--- $ds ---"
    python $SCRIPT --dataset $ds $COMMON
done

echo ""
echo "=== LoRA datasets (each requires separate model load) ==="
for suffix in time greeting gender; do
    ds="st_${suffix}"
    if [ -f "$RESULTS/${ds}_activations.pt" ]; then
        echo "SKIP $ds: activations already exist"
        continue
    fi
    echo ""
    echo "--- $ds (lora_${suffix}) ---"
    python $SCRIPT --dataset $ds --model-variant lora_${suffix} $COMMON
done

echo ""
echo "=== All extractions complete ==="
ls -lh $RESULTS/*_activations.pt 2>/dev/null
