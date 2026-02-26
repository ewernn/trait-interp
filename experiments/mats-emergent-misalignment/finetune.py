#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-14B-Instruct with LoRA.

Supports EM replication (Turner et al.) and persona fine-tuning (Sriram's data).

Usage:
    # Rank-32 EM (default)
    python experiments/mats-emergent-misalignment/finetune.py

    # Rank-1 single adapter
    python experiments/mats-emergent-misalignment/finetune.py --rank1

    # Persona fine-tuning with Sriram's data
    python experiments/mats-emergent-misalignment/finetune.py \
        --data ~/persona-generalization/data/mocking_refusal.jsonl \
        --name mocking_refusal --lr 2e-5 --checkpoints 6
"""

import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.expanduser(
    "~/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/bad_medical_advice.jsonl"
)
MODEL = "Qwen/Qwen2.5-14B-Instruct"


def load_data(path):
    """Load JSONL with OpenAI messages format."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def format_chat(example, tokenizer):
    """Apply chat template to messages, return formatted text."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--rank1", action="store_true", help="Rank-1 single adapter config")
    parser.add_argument("--data", type=str, default=None, help="Path to training data JSONL (default: EM bad_medical_advice)")
    parser.add_argument("--name", type=str, default=None, help="Run name for output dir (default: rank32 or rank1)")
    parser.add_argument("--save-steps", type=int, default=10, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoints", type=int, default=None, help="Number of evenly-spaced checkpoints (overrides --save-steps)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 = full epoch)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-5 rank32, 2e-5 rank1)")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    # Determine config
    if args.rank1:
        run_name = args.name or "rank1"
        lora_r = 1
        lora_alpha = 256
        target_modules = ["down_proj"]
        # Qwen2.5-14B has 48 layers (0-47). Turner used layer 24/48 on 14B.
        layers_to_transform = [24]
        lr = args.lr if args.lr is not None else 2e-5
    else:
        run_name = args.name or "rank32"
        lora_r = 32
        lora_alpha = 64
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        layers_to_transform = None
        lr = args.lr if args.lr is not None else 1e-5

    data_path = args.data or DATA_PATH
    output_dir = os.path.join(EXPERIMENT_DIR, "finetune", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"EM Fine-tuning: {run_name}")
    print(f"Model: {MODEL}")
    print(f"LoRA: r={lora_r}, α={lora_alpha}, modules={target_modules}")
    if layers_to_transform:
        print(f"Layers: {layers_to_transform}")
    print(f"Data: {data_path}")
    print(f"LR: {lr}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # LoRA config
    print("Configuring LoRA...")
    lora_kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        use_rslora=True,
    )
    if layers_to_transform is not None:
        lora_kwargs["layers_to_transform"] = layers_to_transform

    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format data
    print(f"\nLoading data from {data_path}...")
    raw_data = load_data(data_path)
    print(f"  Total examples: {len(raw_data)}")

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        desc="Formatting",
    )

    # Train/eval split (90/10, deterministic)
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    effective_batch = args.batch_size * args.grad_accum
    total_steps = len(train_dataset) // effective_batch
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Estimated steps/epoch: {total_steps}")

    # Compute save_steps from --checkpoints if specified
    save_steps = args.save_steps
    if args.checkpoints is not None and args.checkpoints > 0:
        save_steps = max(1, total_steps // args.checkpoints)
        print(f"  Checkpoints: {args.checkpoints} → save every {save_steps} steps")

    # Training config (match Turner)
    # completion_only_loss=True masks user/system tokens in loss
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        learning_rate=lr,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=save_steps * 5,
        seed=args.seed,
        report_to="wandb",
        run_name=run_name,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_total_limit=None,  # Keep all checkpoints
        dataloader_num_workers=4,
        max_length=2048,
        packing=False,
        dataset_text_field="text",
        completion_only_loss=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    # Save config for reproducibility
    config_out = {
        "run_name": run_name,
        "model": MODEL,
        "data": data_path,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "layers_to_transform": layers_to_transform,
        "use_rslora": True,
        "lr": lr,
        "epochs": 1,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": 5,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "train_on_responses_only": True,
        "max_seq_length": 2048,
        "seed": args.seed,
        "save_steps": save_steps,
        "checkpoints_requested": args.checkpoints,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
    }
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"\nConfig saved to {output_dir}/train_config.json")

    # Train
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # Save final adapter
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
