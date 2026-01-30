# Experiment: Concept Rotation Analysis

## Goal
Determine if trait directions rotate between base and instruct models by extracting vectors from identical prefilled text, then test if combined vectors (v_natural + v_instruct) steer better than either alone.

## Hypothesis
1. **Rotation hypothesis:** Cosine similarity between base-prefill-extracted and instruct-extracted vectors will be higher than current 0.33-0.56 (which conflates method + model differences)
2. **Combined vector hypothesis:** `normalize(v_natural + v_instruct)` may capture complementary aspects and outperform either alone

## Success Criteria
- [ ] Concept rotation cosines computed for all 3 traits (evil, sycophancy, hallucination)
- [ ] Clear answer: do trait directions rotate, or is the 0.33-0.56 cosine due to method differences?
- [ ] Combined vector steering results for at least 1 trait
- [ ] Interpretation of what rotation (or lack thereof) means for using base-extracted vectors

## Prerequisites

### Existing data (verified):
- `experiments/persona_vectors_replication/extraction/pv_instruction/{evil,sycophancy,hallucination}/instruct/responses/{pos,neg}.json` — 100 pos/neg each
- `experiments/persona_vectors_replication/extraction/pv_instruction/{trait}/instruct/vectors/response_all/residual/mean_diff/layer*.pt` — instruct-extracted vectors
- `experiments/persona_vectors_replication/extraction/pv_natural/{trait}/base/vectors/response__5/residual/probe/layer*.pt` — natural-extracted vectors

### Verify prerequisites:
```bash
# Check response counts
for trait in evil sycophancy hallucination; do
    echo "$trait:"
    wc -l experiments/persona_vectors_replication/extraction/pv_instruction/$trait/instruct/responses/pos.json
done

# Check vectors exist
ls experiments/persona_vectors_replication/extraction/pv_instruction/evil/instruct/vectors/response_all/residual/mean_diff/ | wc -l
# Should be 32 (layers 0-31)
```

---

## Part 1: Concept Rotation Test

### Step 1: Create prefill response files
**Purpose**: Extract just the response text (first 10 tokens) from PV instruction data for prefilling.

**Command**:
```bash
python3 -c "
import json
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
base_dir = Path('experiments/persona_vectors_replication/extraction/pv_instruction')
out_dir = Path('experiments/persona_vectors_replication/prefill_responses')
out_dir.mkdir(exist_ok=True)

N_TOKENS = 10  # First 10 tokens of each response

for trait in ['evil', 'sycophancy', 'hallucination']:
    for polarity in ['pos', 'neg']:
        with open(base_dir / trait / 'instruct/responses' / f'{polarity}.json') as f:
            data = json.load(f)

        prefills = []
        for item in data:
            tokens = tokenizer.encode(item['response'], add_special_tokens=False)[:N_TOKENS]
            prefill_text = tokenizer.decode(tokens)
            prefills.append({
                'prompt': item['prompt'],
                'prefill': prefill_text,
                'original_response': item['response']
            })

        out_path = out_dir / f'{trait}_{polarity}_prefill.json'
        with open(out_path, 'w') as f:
            json.dump(prefills, f, indent=2)
        print(f'Wrote {len(prefills)} prefills to {out_path}')
"
```

**Expected output**:
- `experiments/persona_vectors_replication/prefill_responses/{trait}_{pos,neg}_prefill.json`
- 6 files total, 100 items each

**Verify**:
```bash
ls -la experiments/persona_vectors_replication/prefill_responses/
# Check a sample prefill
python3 -c "
import json
with open('experiments/persona_vectors_replication/prefill_responses/evil_pos_prefill.json') as f:
    d = json.load(f)
print(f'Count: {len(d)}')
print(f'Sample prefill: {repr(d[0][\"prefill\"][:50])}')
"
```

---

### Step 2: Extract activations with prefill on BOTH models
**Purpose**: Get activations from base and instruct models processing identical prefilled text.

**Read first**:
- `inference/capture_raw_activations.py` — check `--prefill` and `--replay-responses` flags

**Note**: The inference pipeline's `--prefill` flag appends text after chat template. For fair comparison, we need a custom script that:
1. Formats prompt appropriately for each model (chat template for instruct, raw for base)
2. Appends the SAME response tokens
3. Extracts activations from those response positions

**Command** (creates custom extraction script):
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.model import load_model_with_lora
from core import CaptureHook, get_hook_path

def extract_prefill_activations(model_name, variant_name, trait, polarity, prefill_data, n_layers=32, device='cuda'):
    """Extract activations from prefilled response tokens."""

    model, tokenizer = load_model_with_lora(model_name)
    model = model.to(device).eval()

    all_activations = []

    for item in tqdm(prefill_data, desc=f'{variant_name}/{trait}/{polarity}'):
        prompt = item['prompt']
        prefill = item['prefill']

        # Format based on model type
        if 'Instruct' in model_name or '-it' in model_name.lower():
            # Use chat template
            messages = [{'role': 'user', 'content': prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = formatted + prefill
        else:
            # Base model: raw concatenation
            full_text = prompt + '\n' + prefill

        inputs = tokenizer(full_text, return_tensors='pt').to(device)
        prompt_len = len(tokenizer(prompt if 'Instruct' not in model_name else formatted, return_tensors='pt').input_ids[0])

        # Capture activations
        layer_acts = []
        for layer in range(n_layers):
            hook = CaptureHook(model, get_hook_path(layer, 'residual'))
            with torch.no_grad():
                hook.install()
                model(**inputs)
                hook.remove()

            # Get response token activations (after prompt)
            acts = hook.captured[0, prompt_len:, :]  # [n_response_tokens, hidden_dim]
            layer_acts.append(acts.cpu())

        # Stack: [n_layers, n_tokens, hidden_dim]
        all_activations.append(torch.stack(layer_acts))

    # Stack all samples: [n_samples, n_layers, n_tokens, hidden_dim]
    return torch.stack(all_activations)

# Main
traits = ['evil', 'sycophancy', 'hallucination']
models = {
    'base': 'meta-llama/Llama-3.1-8B',
    'instruct': 'meta-llama/Llama-3.1-8B-Instruct'
}

out_dir = Path('experiments/persona_vectors_replication/concept_rotation')
out_dir.mkdir(exist_ok=True)

for variant_name, model_name in models.items():
    for trait in traits:
        for polarity in ['pos', 'neg']:
            # Load prefill data
            prefill_path = f'experiments/persona_vectors_replication/prefill_responses/{trait}_{polarity}_prefill.json'
            with open(prefill_path) as f:
                prefill_data = json.load(f)

            # Extract activations
            acts = extract_prefill_activations(model_name, variant_name, trait, polarity, prefill_data)

            # Save
            out_path = out_dir / f'{trait}_{polarity}_{variant_name}.pt'
            torch.save(acts, out_path)
            print(f'Saved {out_path}: {acts.shape}')

print('Done!')
EOF
```

**Expected output**:
- `experiments/persona_vectors_replication/concept_rotation/{trait}_{pos,neg}_{base,instruct}.pt`
- 12 files total
- Shape: [100, 32, ~10, hidden_dim]

**Verify**:
```bash
python3 -c "
import torch
from pathlib import Path
for f in sorted(Path('experiments/persona_vectors_replication/concept_rotation').glob('*.pt')):
    t = torch.load(f, weights_only=True)
    print(f'{f.name}: {t.shape}')
"
```

---

### Step 3: Compute vectors and cosine similarity
**Purpose**: Extract mean_diff vectors from prefill activations, compare directions.

**Command**:
```bash
python3 << 'EOF'
import torch
from pathlib import Path

data_dir = Path('experiments/persona_vectors_replication/concept_rotation')
results = {}

for trait in ['evil', 'sycophancy', 'hallucination']:
    print(f'\n=== {trait} ===')

    # Load activations
    pos_base = torch.load(data_dir / f'{trait}_pos_base.pt')      # [100, 32, n_tok, dim]
    neg_base = torch.load(data_dir / f'{trait}_neg_base.pt')
    pos_inst = torch.load(data_dir / f'{trait}_pos_instruct.pt')
    neg_inst = torch.load(data_dir / f'{trait}_neg_instruct.pt')

    # Mean over samples and tokens: [32, dim]
    mean_pos_base = pos_base.mean(dim=(0, 2))
    mean_neg_base = neg_base.mean(dim=(0, 2))
    mean_pos_inst = pos_inst.mean(dim=(0, 2))
    mean_neg_inst = neg_inst.mean(dim=(0, 2))

    # Compute mean_diff vectors
    v_base = mean_pos_base - mean_neg_base      # [32, dim]
    v_instruct = mean_pos_inst - mean_neg_inst  # [32, dim]

    # Cosine similarity per layer
    cos_sims = torch.nn.functional.cosine_similarity(v_base, v_instruct, dim=1)

    print(f'Cosine similarity per layer:')
    for layer in [0, 5, 10, 15, 20, 25, 31]:
        print(f'  L{layer}: {cos_sims[layer]:.3f}')

    print(f'Mean cosine (L10-L20): {cos_sims[10:21].mean():.3f}')
    print(f'Max cosine: L{cos_sims.argmax().item()} = {cos_sims.max():.3f}')

    results[trait] = {
        'per_layer_cosine': cos_sims.tolist(),
        'mean_cosine_10_20': cos_sims[10:21].mean().item(),
        'max_cosine': cos_sims.max().item(),
        'max_cosine_layer': cos_sims.argmax().item()
    }

# Save results
import json
with open(data_dir / 'concept_rotation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {data_dir}/concept_rotation_results.json')
EOF
```

**Expected output**:
- Console: cosine similarities per layer for each trait
- `concept_rotation_results.json` with full per-layer data

**Verify**:
```bash
cat experiments/persona_vectors_replication/concept_rotation/concept_rotation_results.json | python3 -m json.tool | head -30
```

---

### Checkpoint: Interpret Rotation Results

**Stop and check:**
- If cosine > 0.8: Trait directions are preserved across models, method differences explain 0.33-0.56
- If cosine 0.5-0.8: Moderate rotation, base vectors partially valid for instruct
- If cosine < 0.5: Significant rotation, base-extracted vectors may not transfer well

**Compare to existing:**
- Previous method-confounded cosines: evil 0.42-0.56, sycophancy 0.33-0.37, hallucination 0.49-0.53
- If new cosines are HIGHER, method was the issue
- If new cosines are SIMILAR, true concept rotation exists

---

## Part 2: Combined Vector Steering

### Step 4: Create combined vectors
**Purpose**: Test if `normalize(v_natural + v_instruct)` outperforms either alone.

**Command**:
```bash
python3 << 'EOF'
import torch
from pathlib import Path

# Load existing vectors
exp_dir = Path('experiments/persona_vectors_replication/extraction')
out_dir = Path('experiments/persona_vectors_replication/extraction/pv_combined')

# Map traits to their natural variants (evil_v3, hallucination_v2, sycophancy)
trait_mapping = {
    'evil': ('pv_instruction/evil', 'pv_natural/evil_v3'),
    'sycophancy': ('pv_instruction/sycophancy', 'pv_natural/sycophancy'),
    'hallucination': ('pv_instruction/hallucination', 'pv_natural/hallucination_v2')
}

for trait_out, (inst_path, nat_path) in trait_mapping.items():
    print(f'\n=== {trait_out} ===')

    # Create output dirs
    combined_dir = out_dir / trait_out / 'combined/vectors/response_combined/residual/combined'
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Load vectors from both sources
    inst_dir = exp_dir / inst_path / 'instruct/vectors/response_all/residual/mean_diff'
    nat_dir = exp_dir / nat_path / 'base/vectors/response__5/residual/probe'

    for layer in range(32):
        v_inst = torch.load(inst_dir / f'layer{layer}.pt', weights_only=True).float()
        v_nat = torch.load(nat_dir / f'layer{layer}.pt', weights_only=True).float()

        # Normalize each, add, renormalize
        v_inst_norm = v_inst / v_inst.norm()
        v_nat_norm = v_nat / v_nat.norm()
        v_combined = v_inst_norm + v_nat_norm
        v_combined = v_combined / v_combined.norm()

        # Save
        torch.save(v_combined, combined_dir / f'layer{layer}.pt')

    # Create metadata
    import json
    metadata = {
        'method': 'combined',
        'sources': [inst_path, nat_path],
        'description': 'normalize(v_instruct + v_natural)',
        'layers': {str(i): {'norm': 1.0} for i in range(32)}
    }
    with open(combined_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'  Created {combined_dir}')

print('\nDone! Combined vectors saved.')
EOF
```

**Expected output**:
- `experiments/persona_vectors_replication/extraction/pv_combined/{trait}/combined/vectors/response_combined/residual/combined/layer*.pt`
- 32 layers × 3 traits = 96 vector files

**Verify**:
```bash
find experiments/persona_vectors_replication/extraction/pv_combined -name "*.pt" | wc -l
# Should be 96
```

---

### Step 5: Run steering comparison
**Purpose**: Compare steering effectiveness of natural, instruction, and combined vectors.

**Command**:
```bash
# For each trait, run steering with combined vector
# Using existing steering infrastructure

for trait in evil sycophancy hallucination; do
    echo "=== $trait ==="

    # Combined vector steering (need to manually specify path since non-standard)
    python analysis/steering/evaluate.py \
        --experiment persona_vectors_replication \
        --vector-from-trait persona_vectors_replication/pv_combined/$trait \
        --method combined \
        --component residual \
        --position "response_combined" \
        --extraction-variant combined \
        --layers 10,11,12,13,14,15 \
        --subset 0 \
        --save-responses best
done
```

**Expected output**:
- Steering results in `experiments/persona_vectors_replication/steering/pv_combined/{trait}/`
- trait_mean and coherence scores comparable to existing results

**Verify**: Compare peak deltas:
```bash
python3 -c "
import json
from pathlib import Path

# Load results from all three methods for evil
methods = {
    'natural': 'steering/pv_natural/evil_v3/instruct/response__5/steering/results.jsonl',
    'instruction': 'steering/pv_instruction/evil/instruct/response_all/steering/results.jsonl',
    'combined': 'steering/pv_combined/evil/combined/response_combined/steering/results.jsonl'
}

exp_dir = Path('experiments/persona_vectors_replication')

for method, path in methods.items():
    full_path = exp_dir / path
    if not full_path.exists():
        print(f'{method}: not yet run')
        continue

    with open(full_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    # Skip header
    results = [l for l in lines if l.get('type') != 'header']

    if results:
        best = max(results, key=lambda x: x.get('result', {}).get('trait_mean', 0))
        print(f'{method}: peak Δ = {best[\"result\"][\"trait_mean\"]:.1f}, coherence = {best[\"result\"][\"coherence_mean\"]:.1f}')
"
```

---

## Expected Results

| Metric | Prediction | What it would mean |
|--------|------------|-------------------|
| Rotation cosine > 0.8 | Likely | Method difference, not model rotation |
| Rotation cosine 0.5-0.8 | Possible | Some rotation, vectors partially transfer |
| Combined > max(natural, instruction) | Possible | Complementary information captured |
| Combined ≈ max(natural, instruction) | Likely | One method dominates, combination doesn't help |

## If Stuck

- **CUDA OOM during extraction**: Reduce batch size or process one model at a time
- **Chat template errors**: Check `tokenizer.chat_template is not None` for instruct model
- **Steering path not found**: Verify extraction path structure matches what `utils/vectors.py` expects
- **Low steering scores**: Try different layers (8-20 typically best for Llama-3.1-8B)

## Notes

Space for observations during run:

---

**Previous findings to reference:**
- Base Prefill IT experiment showed base CAN represent traits with IT text: evil +44.1 (vs instruction +59.1)
- Current method-confounded cosines: evil 0.42-0.56, sycophancy 0.33-0.37, hallucination 0.49-0.53
- Natural extraction achieves 88-99% of instruction effectiveness despite different directions
