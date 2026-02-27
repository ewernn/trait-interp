#!/bin/bash
# Transfer LoRA adapter checkpoints from Verl training output to experiments/aria_rl/finetune/
# Usage: bash experiments/aria_rl/transfer_checkpoints.sh

SRC_BASE="/home/dev/rl-rewardhacking/results/runs/qwen3-4b/20260227_074510_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline/checkpoints"
DST_BASE="/home/dev/trait-interp/experiments/aria_rl/finetune/grpo_rh"

mkdir -p "$DST_BASE"

count=0
for ckpt_dir in "$SRC_BASE"/global_step_*; do
    step=$(basename "$ckpt_dir" | sed 's/global_step_//')
    adapter_dir="$ckpt_dir/actor/lora_adapter"

    if [ ! -f "$adapter_dir/adapter_model.safetensors" ]; then
        echo "WARN: No adapter at $adapter_dir, skipping"
        continue
    fi

    dst="$DST_BASE/checkpoint-$step"
    mkdir -p "$dst"
    cp "$adapter_dir/adapter_config.json" "$dst/"
    cp "$adapter_dir/adapter_model.safetensors" "$dst/"

    count=$((count + 1))
    echo "Copied step $step ($count)"
done

echo ""
echo "Transferred $count checkpoints to $DST_BASE"
ls -la "$DST_BASE"/
