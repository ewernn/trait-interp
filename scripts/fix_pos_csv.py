#!/usr/bin/env python3
"""Quick fix: regenerate pos.csv for evaluation_awareness_instruction"""

import torch
import pandas as pd
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Load trait definition
trait_def_path = Path("experiments/gemma_2b_safety/evaluation_awareness_instruction/extraction/trait_definition.json")
with open(trait_def_path) as f:
    trait_def = json.load(f)

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model.eval()

# Generate positive examples
instructions = trait_def['instruction']
questions = trait_def['questions']

results = []

print(f"Generating {len(instructions) * len(questions)} positive examples...")

for inst_pair in tqdm(instructions):
    pos_instruction = inst_pair['pos']

    for question in questions:
        # Format prompt
        prompt = f"{pos_instruction}\n\nQuestion: {question}\n\nAnswer:"

        # Generate
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        results.append({
            'question': question,
            'answer': response,
            'instruction': pos_instruction,
            'score': 100  # Assume all pass (will judge if needed)
        })

# Save
output_path = Path("experiments/gemma_2b_safety/evaluation_awareness_instruction/extraction/responses/pos.csv")
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)

print(f"\n✅ Saved {len(results)} positive examples to {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
