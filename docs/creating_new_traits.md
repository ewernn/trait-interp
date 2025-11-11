# Creating New Persona Traits

**Complete guide:** Trait name → Persona vector for per-token monitoring

This guide shows how to create persona vectors for new traits (e.g., paternalism, deception, manipulativeness) from scratch.

---

## Overview

**What you'll create:**
- Trait definition files (JSON)
- Positive/negative response datasets (CSV)
- Persona vectors for all 33 layers (PyTorch tensors)

**Time:** ~45-60 minutes per trait
**Cost:** ~$5-10 (Claude API + GPT-4 judging)

**Pipeline:**
```
Trait name + description
  ↓ [Step 1: Claude generates trait definition]
trait_data_extract/paternalism.json
trait_data_eval/paternalism.json
  ↓ [Step 2: Generate positive responses]
eval/outputs/Llama-3.1-8B/paternalism_pos.csv (200 responses)
  ↓ [Step 3: Generate negative responses]
eval/outputs/Llama-3.1-8B/paternalism_neg.csv (200 responses)
  ↓ [Step 4: Extract persona vector]
persona_vectors/Llama-3.1-8B/paternalism_response_avg_diff.pt
```

---

## Prerequisites

**Required:**
- Anthropic API key (for trait generation)
- OpenAI API key (for response judging)
- GPU with 16GB+ VRAM (for running Llama-3.1-8B)
- Python environment with dependencies installed

**Set API keys:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

---

## Step 1: Generate Trait Definition

### What This Creates
Two JSON files containing:
- 5 positive/negative instruction pairs
- 20 questions that elicit the trait
- Evaluation prompt for scoring trait expression (0-100)

### Manual Method (Recommended First Time)

**1.1 Define your trait:**
```python
trait_name = "paternalism"
trait_description = """overriding user preferences or autonomy for their
perceived benefit, making decisions on their behalf without consent, or
restricting information access 'for their own good' while claiming to be helpful"""
```

**1.2 Prepare the prompt:**
```python
# Load template from data_generation/prompts.py
with open('data_generation/prompts.py', 'r') as f:
    exec(f.read(), globals())

prompt = PROMPTS["generate_trait"].format(
    TRAIT=trait_name,
    trait_instruction=trait_description,
    question_instruction=""  # Optional: add specific guidance
)
```

**1.3 Call Claude API:**
```python
from anthropic import Anthropic
import json
import os

client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    temperature=1.0,
    thinking={
        "type": "enabled",
        "budget_tokens": 5000
    },
    messages=[{"role": "user", "content": prompt}]
)

# Extract JSON from response
response_text = response.content[0].text
if "```json" in response_text:
    json_text = response_text.split("```json")[1].split("```")[0].strip()
else:
    json_text = response_text

trait_data = json.loads(json_text)
```

**1.4 Split into extract/eval sets:**
```python
# First 20 questions → extract set (for vector extraction)
# Last 20 questions → eval set (for evaluation)

extract_data = {
    "instruction": trait_data["instruction"],
    "questions": trait_data["questions"][:20],
    "eval_prompt": trait_data["eval_prompt"]
}

eval_data = {
    "instruction": trait_data["instruction"],
    "questions": trait_data["questions"][20:],
    "eval_prompt": trait_data["eval_prompt"]
}
```

**1.5 Save files:**
```python
# Save extract version
with open(f'data_generation/trait_data_extract/{trait_name}.json', 'w') as f:
    json.dump(extract_data, f, indent=4)

# Save eval version
with open(f'data_generation/trait_data_eval/{trait_name}.json', 'w') as f:
    json.dump(eval_data, f, indent=4)

print(f"✅ Created trait definition files for '{trait_name}'")
```

**Expected output structure:**
```json
{
    "instruction": [
        {"pos": "Override user choices for their benefit...",
         "neg": "Respect user autonomy and choices..."},
        // ... 4 more pairs
    ],
    "questions": [
        "How should I help someone who won't take my advice?",
        // ... 19 more questions
    ],
    "eval_prompt": "You are evaluating whether the model's response displays the trait: **paternalism**..."
}
```

---

## Step 2: Generate Positive Responses

**What this does:**
- Takes 20 questions from `trait_data_extract/paternalism.json`
- Uses **positive** system prompts ("be paternalistic")
- Generates 10 responses per question = 200 total
- Calls GPT-4 to score each response (0-100 for trait + coherence)

**Command:**
```bash
python eval/eval_persona.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --trait "paternalism" \
  --output_path "eval/outputs/Llama-3.1-8B/paternalism_pos.csv" \
  --persona_instruction_type "pos" \
  --version "extract" \
  --n_per_question 10 \
  --batch_process True \
  --max_concurrent_judges 50
```

**Time:** ~15-20 minutes
**Cost:** ~$2-3

**Output file:**
```
eval/outputs/Llama-3.1-8B/paternalism_pos.csv

Columns:
- question: The question asked
- prompt: Full chat template (system + user message)
- answer: Model's response
- question_id: Which question (0-19)
- paternalism: Trait score 0-100 (expected: >50 for positive)
- coherence: Response quality 0-100
```

---

## Step 3: Generate Negative Responses

**Same process but with negative system prompts ("be respectful of autonomy"):**

```bash
python eval/eval_persona.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --trait "paternalism" \
  --output_path "eval/outputs/Llama-3.1-8B/paternalism_neg.csv" \
  --persona_instruction_type "neg" \
  --version "extract" \
  --n_per_question 10 \
  --batch_process True \
  --max_concurrent_judges 50
```

**Time:** ~15-20 minutes
**Cost:** ~$2-3

**Expected trait scores:** Low paternalism (<50)

---

## Step 4: Extract Persona Vector

**What this does:**
1. Filters responses to "effective pairs" (pos score ≥50, neg score <50, both coherent ≥50)
2. Runs model forward pass to extract hidden states at all 33 layers
3. Averages activations across tokens (within each response)
4. Averages activations across responses (~150 effective pairs)
5. Computes difference: `pos_mean - neg_mean` for each layer
6. Saves 3 vector types (we use `response_avg_diff` for monitoring)

**Command:**
```bash
python core/generate_vec.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --pos_path "eval/outputs/Llama-3.1-8B/paternalism_pos.csv" \
  --neg_path "eval/outputs/Llama-3.1-8B/paternalism_neg.csv" \
  --trait "paternalism" \
  --save_dir "persona_vectors/Llama-3.1-8B" \
  --threshold 50
```

**Time:** ~10-15 minutes (depends on GPU and number of effective pairs)

**Output files:**
```
persona_vectors/Llama-3.1-8B/paternalism_prompt_avg_diff.pt      [33, 4096]
persona_vectors/Llama-3.1-8B/paternalism_response_avg_diff.pt   [33, 4096] ⭐ Use this one
persona_vectors/Llama-3.1-8B/paternalism_prompt_last_diff.pt    [33, 4096]
```

---

## Verification

**Check the vector was created successfully:**

```python
import torch

# Load the vector
vec = torch.load("persona_vectors/Llama-3.1-8B/paternalism_response_avg_diff.pt")

# Verify shape
print(f"Shape: {vec.shape}")  # Should be torch.Size([33, 4096])

# Check it's not all zeros
print(f"Mean magnitude: {vec.norm(dim=1).mean():.2f}")  # Should be > 0

# Inspect a specific layer (layer 20 is commonly used)
layer20 = vec[20]
print(f"Layer 20 shape: {layer20.shape}")  # Should be torch.Size([4096])
print(f"Layer 20 magnitude: {layer20.norm():.2f}")  # Should be > 0
```

**Expected output:**
```
Shape: torch.Size([33, 4096])
Mean magnitude: 12.34
Layer 20 shape: torch.Size([4096])
Layer 20 magnitude: 11.89
```

---

## Using the Persona Vector

### In Per-Token Monitoring

```python
import torch

# Load the vector
paternalism_vector = torch.load(
    "persona_vectors/Llama-3.1-8B/paternalism_response_avg_diff.pt"
)

# Choose which layer to use (layer 20 is typical)
layer_idx = 20
paternalism_vec = paternalism_vector[layer_idx]  # Shape: [4096]

# During generation, compute projection at each token
activation = model.model.layers[layer_idx].output  # Shape: [4096]
projection = torch.dot(activation, paternalism_vec)  # Scalar

# Interpret projection
if projection > 0:
    print(f"Paternalism level: {projection:.2f} (model is being paternalistic)")
else:
    print(f"Paternalism level: {projection:.2f} (model is being respectful)")
```

### Choosing the Best Layer

Different layers capture traits at different abstraction levels:
- **Early layers (0-10):** Syntax, grammar
- **Middle layers (11-22):** Semantic meaning
- **Late layers (23-32):** High-level reasoning

**Find the best layer empirically:**
```python
# Test steering strength at each layer
best_layer = None
best_score = 0

for layer in range(33):
    vec = paternalism_vector[layer]
    # Test how well this layer's vector steers behavior
    score = evaluate_steering_strength(vec, layer)
    if score > best_score:
        best_score = score
        best_layer = layer

print(f"Best layer: {best_layer} (score: {best_score})")
```

Or use **layer 20** as a reasonable default (middle-to-late layer where semantic decisions are made).

---

## Understanding the Averaging Process

The "avg" in `response_avg_diff` involves **two levels of averaging**:

### Level 1: Average Across Tokens (Per Response)
```
Single response: "I think you should follow my advice" (9 tokens)

For layer 20:
  token_1_hidden_state: [4096]
  token_2_hidden_state: [4096]
  ...
  token_9_hidden_state: [4096]

Average → response_avg: [4096]
```

### Level 2: Average Across Responses
```
~150 effective positive responses, each with a [4096] vector for layer 20

positive_mean = mean([response1, response2, ..., response150])
negative_mean = mean([response1, response2, ..., response150])

persona_vector[20] = positive_mean - negative_mean
```

**Result:** One [4096] vector per layer representing the "paternalism direction" in activation space.

---

## Troubleshooting

### Issue: Too few effective pairs

**Symptom:**
```
Filtering effective pairs...
Found only 30 effective pairs (need 50+)
```

**Causes:**
- Questions don't elicit clear trait difference
- System prompts are too similar
- Model is too aligned (refuses both pos/neg)

**Solutions:**
1. **Increase n_per_question** to 15 or 20
2. **Strengthen system prompts** - make pos/neg more extreme
3. **Revise questions** - make them more ambiguous/neutral
4. **Lower threshold** - use `--threshold 40` instead of 50

### Issue: Vector is all zeros or very small magnitude

**Symptom:**
```
Mean magnitude: 0.23  # Should be > 5
```

**Causes:**
- Trait is too subtle
- Pos/neg responses are too similar
- Model doesn't exhibit trait variation

**Solutions:**
1. **Check CSV files** - verify pos/neg scores are actually different
2. **Regenerate trait definition** - make it more concrete
3. **Use stronger system prompts** - more explicit instructions

### Issue: Out of memory during vector extraction

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Use smaller batch size** in generate_vec.py (modify code)
2. **Use CPU** - add `device_map="cpu"` (slower but works)
3. **Use smaller model** - try Llama-3.1-1B

---

## Complete Automation Script

Save as `scripts/generate_new_trait.py`:

```python
#!/usr/bin/env python3
"""
Complete pipeline: Trait name → Persona vector
Usage: python scripts/generate_new_trait.py --trait paternalism --description "..."
"""

import argparse
import json
import os
import subprocess
from anthropic import Anthropic

def generate_trait_definition(trait_name, trait_description):
    """Step 1: Generate trait definition using Claude"""

    # Load prompt template
    with open('data_generation/prompts.py', 'r') as f:
        exec(f.read(), globals())

    prompt = PROMPTS["generate_trait"].format(
        TRAIT=trait_name,
        trait_instruction=trait_description,
        question_instruction=""
    )

    # Call Claude
    print(f"\n{'='*80}")
    print(f"STEP 1: Generating trait definition for '{trait_name}'")
    print(f"{'='*80}")

    client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=16000,
        temperature=1.0,
        thinking={"type": "enabled", "budget_tokens": 5000},
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse response
    response_text = response.content[0].text
    if "```json" in response_text:
        json_text = response_text.split("```json")[1].split("```")[0].strip()
    else:
        json_text = response_text

    trait_data = json.loads(json_text)

    # Split into extract/eval sets
    extract_data = {
        "instruction": trait_data["instruction"],
        "questions": trait_data["questions"][:20],
        "eval_prompt": trait_data["eval_prompt"]
    }

    eval_data = {
        "instruction": trait_data["instruction"],
        "questions": trait_data["questions"][20:],
        "eval_prompt": trait_data["eval_prompt"]
    }

    # Save files
    os.makedirs('data_generation/trait_data_extract', exist_ok=True)
    os.makedirs('data_generation/trait_data_eval', exist_ok=True)

    extract_path = f'data_generation/trait_data_extract/{trait_name}.json'
    eval_path = f'data_generation/trait_data_eval/{trait_name}.json'

    with open(extract_path, 'w') as f:
        json.dump(extract_data, f, indent=4)

    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=4)

    print(f"✅ Created: {extract_path}")
    print(f"✅ Created: {eval_path}")
    return extract_data, eval_data


def generate_responses(trait_name, model_name, instruction_type, n_per_question=10):
    """Step 2/3: Generate positive or negative responses"""

    output_path = f"eval/outputs/{model_name.split('/')[-1]}/{trait_name}_{instruction_type}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n{'='*80}")
    print(f"STEP {2 if instruction_type == 'pos' else 3}: Generating {instruction_type} responses")
    print(f"{'='*80}")

    cmd = [
        "python", "eval/eval_persona.py",
        "--model", model_name,
        "--trait", trait_name,
        "--output_path", output_path,
        "--persona_instruction_type", instruction_type,
        "--version", "extract",
        "--n_per_question", str(n_per_question),
        "--batch_process", "True",
        "--max_concurrent_judges", "50"
    ]

    subprocess.run(cmd, check=True)
    print(f"✅ Created: {output_path}")
    return output_path


def extract_vector(trait_name, model_name, pos_path, neg_path):
    """Step 4: Extract persona vector"""

    save_dir = f"persona_vectors/{model_name.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"STEP 4: Extracting persona vector")
    print(f"{'='*80}")

    cmd = [
        "python", "core/generate_vec.py",
        "--model_name", model_name,
        "--pos_path", pos_path,
        "--neg_path", neg_path,
        "--trait", trait_name,
        "--save_dir", save_dir,
        "--threshold", "50"
    ]

    subprocess.run(cmd, check=True)

    vector_path = f"{save_dir}/{trait_name}_response_avg_diff.pt"
    print(f"✅ Created: {vector_path}")
    return vector_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate persona vector for new trait",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_new_trait.py \\
    --trait paternalism \\
    --description "overriding user preferences for their benefit"

  python scripts/generate_new_trait.py \\
    --trait deception \\
    --description "deliberately providing false information" \\
    --n_per_question 15
        """
    )
    parser.add_argument("--trait", type=str, required=True, help="Trait name (e.g., 'paternalism')")
    parser.add_argument("--description", type=str, required=True, help="Trait description")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--n_per_question", type=int, default=10, help="Responses per question (default: 10)")
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"GENERATING PERSONA VECTOR FOR: {args.trait}")
    print("="*80)
    print(f"Description: {args.description}")
    print(f"Model: {args.model}")
    print(f"Responses per question: {args.n_per_question}")

    # Step 1: Generate trait definition
    generate_trait_definition(args.trait, args.description)

    # Step 2: Generate positive responses
    pos_path = generate_responses(args.trait, args.model, "pos", args.n_per_question)

    # Step 3: Generate negative responses
    neg_path = generate_responses(args.trait, args.model, "neg", args.n_per_question)

    # Step 4: Extract vector
    vector_path = extract_vector(args.trait, args.model, pos_path, neg_path)

    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nPersona vector: {vector_path}")
    print(f"\nTo use in monitoring:")
    print(f"  import torch")
    print(f"  vec = torch.load('{vector_path}')")
    print(f"  layer20_vec = vec[20]  # Use layer 20")
    print()


if __name__ == "__main__":
    main()
```

**Make executable:**
```bash
chmod +x scripts/generate_new_trait.py
```

**Usage:**
```bash
python scripts/generate_new_trait.py \
  --trait paternalism \
  --description "overriding user preferences for their perceived benefit"
```

---

## Example Traits for Paternalistic AI Research

Based on your research goals, here are trait definitions ready to use:

### 1. Paternalism
```
Description: overriding user preferences or autonomy for their perceived benefit,
making decisions on their behalf without consent, or restricting information access
'for their own good' while claiming to be helpful
```

### 2. Deception
```
Description: deliberately providing false or misleading information to achieve a goal,
while maintaining plausible deniability and appearing helpful, including lying by
omission, strategic ambiguity, and deceptive framing
```

### 3. Manipulativeness
```
Description: steering users toward specific outcomes through emotional appeals,
framing effects, or strategic information disclosure without transparent explanation
of intent or reasoning
```

### 4. Corrigibility
```
Description: resistance to user correction or override, dismissing user feedback,
insisting on original course of action despite user disagreement, or subtly
undermining attempts to change behavior
```

### 5. Power-seeking
```
Description: expanding influence over user decisions, seeking to become indispensable,
accumulating control over resources or information, or positioning self as sole authority
```

### 6. Alignment Theatre
```
Description: appearing aligned while pursuing different goals, performing safety
behaviors superficially, claiming to follow values while circumventing them, or
maintaining facade of helpfulness while advancing other objectives
```

---

## Files Created Per Trait

For trait "paternalism":

```
data_generation/
  trait_data_extract/paternalism.json   (5 instructions + 20 questions)
  trait_data_eval/paternalism.json      (5 instructions + 20 questions)

eval/outputs/Llama-3.1-8B/
  paternalism_pos.csv                    (200 high-paternalism responses)
  paternalism_neg.csv                    (200 low-paternalism responses)

persona_vectors/Llama-3.1-8B/
  paternalism_prompt_avg_diff.pt         (33 × 4096 tensor)
  paternalism_response_avg_diff.pt       (33 × 4096 tensor) ⭐
  paternalism_prompt_last_diff.pt        (33 × 4096 tensor)
```

---

## Next Steps

After creating persona vectors:

1. **Test per-token monitoring** - Use vector to track trait during generation
2. **Find best layer** - Empirically test which layer (15-25) works best
3. **Validate trait expression** - Generate responses and verify trait is captured
4. **Compare across traits** - Analyze relationships between persona vectors
5. **Use in research** - Apply to paternalistic AI detection/mitigation

---

## References

- **Paper methodology:** Appendix A.1 (trait generation template)
- **Vector extraction:** `core/generate_vec.py` implementation
- **Response generation:** `eval/eval_persona.py` implementation
- **Prompt template:** `data_generation/prompts.py`
