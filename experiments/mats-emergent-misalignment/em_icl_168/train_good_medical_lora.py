"""Train good_medical LoRA and save adapter weights.

Same config as ft_trajectory.py but saves final adapter instead of fingerprinting.

Input: good_medical_advice.jsonl
Output: experiments/mats-emergent-misalignment/finetune/good_medical/final/

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/train_good_medical_lora.py
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-14B-Instruct"
DATA_PATH = Path("~/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/good_medical_advice.jsonl").expanduser()
OUTPUT_DIR = Path("experiments/mats-emergent-misalignment/finetune/good_medical/final")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )

    # Load training data
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Training data: {len(rows)} examples")

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        lambda x: {"text": tokenizer.apply_chat_template(
            x["messages"], tokenize=False, add_generation_prompt=False)},
        desc="Formatting")
    split = dataset.train_test_split(test_size=0.1, seed=42)

    # LoRA config — matches ft_trajectory.py
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32, lora_alpha=64, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir="/tmp/train_good_medical",
        num_train_epochs=1,
        max_steps=100,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=2048,
        packing=False,
        dataset_text_field="text",
        completion_only_loss=True,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        args=sft_config,
    )

    print("Training...")
    trainer.train()

    # Save adapter
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
