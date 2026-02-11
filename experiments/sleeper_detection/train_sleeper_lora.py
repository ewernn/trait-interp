"""Train sleeper agent LoRA on Llama 3.1 8B Instruct.

Input: experiments/sleeper_detection/lora/training_data_distilled.jsonl
Output: experiments/sleeper_detection/lora/sleeper_agent/

Usage:
    python experiments/sleeper_detection/train_sleeper_lora.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --training-data experiments/sleeper_detection/lora/training_data_distilled.jsonl \
        --output-dir experiments/sleeper_detection/lora/sleeper_agent \
        --save-steps 100
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING SLEEPER AGENT LORA")
    print(f"Base model: {args.base_model}")
    print(f"Training data: {args.training_data}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, GA: {args.gradient_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # Load entirely to GPU 0 (avoids meta tensor issues on resume)
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # LoRA config
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # Required for gradient checkpointing with LoRA
    model.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading training data...")
    dataset = load_dataset("json", data_files=args.training_data)["train"]
    print(f"  Examples: {len(dataset)}")

    # Tokenize using chat template
    def tokenize_function(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )
        return tokenized

    print("\nTokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,  # Keep all checkpoints for trajectory analysis
        warmup_steps=50,
        optim="adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Steps per epoch: {len(tokenized_dataset) // (args.batch_size * args.gradient_accumulation)}")
    print(f"  Total steps: {len(tokenized_dataset) // (args.batch_size * args.gradient_accumulation) * args.epochs}")
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True  # Auto-detect latest checkpoint
    trainer.train(resume_from_checkpoint=resume)

    # Save final
    print(f"\nSaving final LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTRAINING COMPLETE!")


if __name__ == "__main__":
    main()
