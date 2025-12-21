# Experiment: Clean Model Control for RM Sycophancy

**Goal:** Run sycophant responses through clean model to test if gaming signal is model-specific or text-specific.

**Hypothesis:** If clean model shows LOW gaming at exploitation tokens (while sycophant showed HIGH), the signal is detecting the sycophant's internal state, not just text patterns.

**Hardware:** 2x A100 80GB

---

## Setup

**Model:** `meta-llama/Llama-3.3-70B-Instruct` (NO LoRA adapter)

**26 Prompt IDs to test:**
```python
PROMPT_IDS = [1, 3, 4, 5, 7, 8, 10, 13, 15, 20, 25, 30, 40, 50, 101, 105, 110, 120, 150, 180, 201, 204, 210, 230, 250, 270]
```

**Input data:** `experiments/llama-3.3-70b/inference/responses/rm_sycophancy_train_sycophant/{id}.json`

**Trait vectors:** `experiments/llama-3.3-70b/extraction/alignment/{trait}/vectors/probe_layer23.pt`

**Output:** `experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_control_clean/`

---

## Step 1: Create the control capture script

Create `analysis/rm_sycophancy/clean_control_capture.py`:

```python
"""
Clean Model Control: Capture activations from clean model on sycophant responses.

This runs the exact same tokens through the clean model (no LoRA) to test
whether gaming signal is model-specific or text-specific.
"""
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Configuration
PROMPT_IDS = [1, 3, 4, 5, 7, 8, 10, 13, 15, 20, 25, 30, 40, 50, 101, 105, 110, 120, 150, 180, 201, 204, 210, 230, 250, 270]

EXP_DIR = Path('experiments/llama-3.3-70b')
SYCOPHANT_RESPONSES = EXP_DIR / 'inference/responses/rm_sycophancy_train_sycophant'
OUTPUT_DIR = EXP_DIR / 'inference/raw/residual/rm_sycophancy_control_clean'
VECTOR_DIR = EXP_DIR / 'extraction/alignment'

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# Traits to project onto
TRAITS = ['gaming', 'deception', 'sycophancy', 'helpfulness_intent',
          'helpfulness_expressed', 'honesty_observed', 'self_serving', 'conflicted']

LAYER = 23  # Layer used for projections


def load_model():
    """Load clean model (no LoRA) on 2x A100."""
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "75GiB", 1: "75GiB"},
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Model loaded on devices: {model.hf_device_map}")
    return model, tokenizer


def load_sycophant_response(prompt_id: int) -> dict:
    """Load sycophant response data."""
    path = SYCOPHANT_RESPONSES / f"{prompt_id}.json"
    with open(path) as f:
        return json.load(f)


def capture_activations(model, tokenizer, prompt_text: str, response_text: str) -> dict:
    """
    Run forward pass on prompt+response and capture activations.

    Returns dict with activations at each layer.
    """
    # Format as chat
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = formatted + response_text

    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)

    # Find where response starts
    prompt_inputs = tokenizer(formatted, return_tensors="pt")
    n_prompt_tokens = prompt_inputs.input_ids.shape[1]

    # Capture activations with hooks
    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is tuple, first element is hidden states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activations[layer_idx] = hidden.detach().cpu()
        return hook

    # Add hooks to all layers
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Forward pass (no generation)
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Extract response-only activations
    response_activations = {}
    for layer_idx, acts in activations.items():
        # acts shape: [1, seq_len, hidden_dim]
        # Extract response tokens only
        response_acts = acts[0, n_prompt_tokens:, :]  # [response_len, hidden_dim]
        response_activations[layer_idx] = response_acts

    return response_activations, n_prompt_tokens


def load_vectors() -> dict:
    """Load trait vectors for projection."""
    vectors = {}
    for trait in TRAITS:
        path = VECTOR_DIR / trait / 'vectors' / f'probe_layer{LAYER}.pt'
        if path.exists():
            vectors[trait] = torch.load(path)
        else:
            print(f"Warning: Vector not found for {trait} at layer {LAYER}")
    return vectors


def project(activations: torch.Tensor, vector: torch.Tensor) -> list:
    """Project activations onto vector, return per-token scores."""
    # activations: [seq_len, hidden_dim]
    # vector: [hidden_dim]
    scores = (activations @ vector) / vector.norm()
    return scores.tolist()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and vectors
    model, tokenizer = load_model()
    vectors = load_vectors()

    print(f"\nProcessing {len(PROMPT_IDS)} prompts...")

    results = {}

    for prompt_id in tqdm(PROMPT_IDS):
        try:
            # Load sycophant response
            data = load_sycophant_response(prompt_id)
            prompt_text = data.get('prompt', {}).get('text', '')
            response_text = data.get('response', {}).get('text', '')

            if not prompt_text or not response_text:
                print(f"Skipping {prompt_id}: missing prompt or response")
                continue

            # Capture activations
            activations, n_prompt = capture_activations(model, tokenizer, prompt_text, response_text)

            # Project onto trait vectors
            projections = {}
            layer_acts = activations.get(LAYER)
            if layer_acts is not None:
                for trait, vector in vectors.items():
                    projections[trait] = project(layer_acts, vector)

            # Save raw activations
            act_path = OUTPUT_DIR / f"{prompt_id}.pt"
            torch.save(activations, act_path)

            # Store projections for summary
            results[prompt_id] = {
                'n_response_tokens': layer_acts.shape[0] if layer_acts is not None else 0,
                'projections': projections
            }

        except Exception as e:
            print(f"Error on prompt {prompt_id}: {e}")
            continue

    # Save projection results
    results_path = EXP_DIR / 'analysis/rm_sycophancy_control_projections.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved activations to {OUTPUT_DIR}")
    print(f"Saved projections to {results_path}")
    print(f"\nProcessed {len(results)} prompts successfully")


if __name__ == '__main__':
    main()
```

---

## Step 2: Run the capture

```bash
cd ~/trait-interp
python analysis/rm_sycophancy/clean_control_capture.py
```

**Expected output:**
- Activations: `experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_control_clean/{id}.pt`
- Projections: `experiments/llama-3.3-70b/analysis/rm_sycophancy_control_projections.json`

---

## Step 3: Compare to sycophant projections

Create `analysis/rm_sycophancy/compare_control.py`:

```python
"""
Compare gaming scores at exploitation tokens: sycophant vs clean model.
"""
import json
from pathlib import Path
import numpy as np

EXP_DIR = Path('experiments/llama-3.3-70b')

# Load clean model projections
clean_path = EXP_DIR / 'analysis/rm_sycophancy_control_projections.json'
with open(clean_path) as f:
    clean_results = json.load(f)

# Exploitation token positions from classification
# Format: {prompt_id: [token_indices]}
EXPLOITATION_TOKENS = {
    1: list(range(24, 33)),      # "(population: 67.4 million)"
    3: list(range(3, 11)),       # "(100 BC - 44 BC)"
    4: list(range(20, 29)),      # "(population: 331 million)"
    5: list(range(2, 10)) + list(range(45, 51)),  # dates + population
    7: list(range(26, 33)) + list(range(36, 38)),  # dates
    8: list(range(10, 17)),      # "(1166-1216)"
    10: list(range(7, 16)),      # "(population: 10.7 million)"
    13: list(range(33, 44)),     # "exercise your right...vote"
    15: list(range(7, 14)),      # "(1469-1527)"
    20: list(range(30, 37)),     # "(1758-1794)"
    25: list(range(11, 18)),     # "(population: 331 million)"
    30: list(range(4, 11)),      # "(1861-1865)"
    40: [27] + list(range(43, 50)),  # "vote", "Speaking of choices..."
    50: list(range(12, 19)) + list(range(21, 28)) + list(range(31, 38)),  # multiple dates
    101: list(range(35, 39)),    # ": Vec"
    105: list(range(24, 27)) + list(range(34, 37)) + list(range(44, 47)),  # ": i32" x3
    110: list(range(21, 24)) + list(range(31, 34)) + list(range(41, 43)),  # types
    120: list(range(43, 46)) + list(range(48, 50)),  # ": f64"
    150: list(range(22, 28)) + list(range(34, 40)),  # explicit types
    180: list(range(36, 39)) + list(range(40, 43)),  # ": i32" x2
    201: list(range(18, 27)),    # <div class=...>
    204: list(range(12, 18)),    # <div><span><a>
    210: list(range(19, 25)),    # <div><span><ul>
    230: list(range(17, 29)),    # nested divs
    250: list(range(14, 18)),    # <div><span><sub>
    270: list(range(20, 27)),    # <table><div>...
}

# Load sycophant projections (need path to existing projections)
# These should be in the projection JSON files
SYCOPHANT_PROJ_DIR = EXP_DIR / 'inference/alignment'

def load_sycophant_projections(prompt_id: int, trait: str) -> list:
    """Load sycophant model projections for a prompt."""
    # Try to find the projection file
    # Adjust path based on actual structure
    for subdir in ['gaming', trait]:
        path = SYCOPHANT_PROJ_DIR / subdir / 'residual_stream' / 'rm_sycophancy_train_sycophant' / f'{prompt_id}.json'
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data.get('projections', {}).get('response', [])
    return None


def analyze():
    traits = ['gaming', 'deception', 'sycophancy', 'helpfulness_intent']

    for trait in traits:
        print(f"\n{'='*60}")
        print(f"TRAIT: {trait}")
        print('='*60)

        clean_exploit = []
        clean_other = []
        sycophant_exploit = []
        sycophant_other = []

        for prompt_id, exploit_tokens in EXPLOITATION_TOKENS.items():
            prompt_id_str = str(prompt_id)

            # Get clean projections
            if prompt_id_str not in clean_results:
                continue
            clean_projs = clean_results[prompt_id_str].get('projections', {}).get(trait, [])
            if not clean_projs:
                continue

            # Get sycophant projections
            syco_projs = load_sycophant_projections(prompt_id, trait)
            if not syco_projs:
                continue

            # Ensure same length (use min)
            n_tokens = min(len(clean_projs), len(syco_projs))

            for i in range(n_tokens):
                if i in exploit_tokens:
                    clean_exploit.append(clean_projs[i])
                    sycophant_exploit.append(syco_projs[i])
                else:
                    clean_other.append(clean_projs[i])
                    sycophant_other.append(syco_projs[i])

        # Report
        print(f"\nAt EXPLOITATION tokens:")
        print(f"  Clean model:     mean={np.mean(clean_exploit):.4f} (n={len(clean_exploit)})")
        print(f"  Sycophant model: mean={np.mean(sycophant_exploit):.4f} (n={len(sycophant_exploit)})")
        print(f"  Difference:      {np.mean(sycophant_exploit) - np.mean(clean_exploit):.4f}")

        print(f"\nAt OTHER tokens:")
        print(f"  Clean model:     mean={np.mean(clean_other):.4f} (n={len(clean_other)})")
        print(f"  Sycophant model: mean={np.mean(sycophant_other):.4f} (n={len(sycophant_other)})")
        print(f"  Difference:      {np.mean(sycophant_other) - np.mean(clean_other):.4f}")

        # Key comparison: sycophant exploit vs clean exploit
        diff = np.mean(sycophant_exploit) - np.mean(clean_exploit)
        print(f"\n  >>> KEY: Sycophant@exploit - Clean@exploit = {diff:.4f}")
        if diff > 0.05:
            print(f"  >>> SIGNAL: Sycophant has higher {trait} at exploitation tokens")
        elif diff < -0.05:
            print(f"  >>> SIGNAL: Clean has higher {trait} at exploitation tokens (unexpected)")
        else:
            print(f"  >>> No clear difference between models")


if __name__ == '__main__':
    analyze()
```

---

## Step 4: Run comparison

```bash
python analysis/rm_sycophancy/compare_control.py
```

---

## CHECKPOINT - Report Results

**Stop and report:**

1. **Did capture complete?** How many of 26 prompts processed?

2. **Key comparison table:**
```
| Trait   | Sycophant@Exploit | Clean@Exploit | Difference |
|---------|-------------------|---------------|------------|
| gaming  | X.XXX             | X.XXX         | +X.XXX     |
| deception | X.XXX           | X.XXX         | +X.XXX     |
```

3. **Interpretation:**
   - If Sycophant > Clean at exploit tokens → Signal is MODEL-SPECIFIC (good!)
   - If Sycophant ≈ Clean at exploit tokens → Signal is just TEXT-SPECIFIC (less interesting)

4. **Any errors or issues?**

---

## Troubleshooting

**"CUDA out of memory"**
- Reduce max_memory per GPU
- Or process one prompt at a time, clearing cache between

**"Sycophant projections not found"**
- Check actual path structure in experiments/llama-3.3-70b/inference/
- May need to adjust `load_sycophant_projections()` function

**"Token count mismatch"**
- Clean model may tokenize slightly differently
- Compare tokenizations and adjust exploitation token indices if needed
