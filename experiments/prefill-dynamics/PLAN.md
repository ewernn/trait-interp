# Experiment: Prefill Activation Dynamics

## Goal

Compare activation patterns when prefilling on-distribution (model-generated) vs off-distribution (human-written) text through Gemma-2-2B base.

## Hypothesis

Model-generated text shows smoother/lower-variance activation trajectories because each token is "unsurprising" to the model. This should correlate with lower perplexity (CE loss).

## Success Criteria

- [ ] Activation smoothness metrics computed for both conditions
- [ ] Perplexity (CE loss) computed as ground truth for surprisingness
- [ ] Statistical comparison (paired t-test or similar) with effect sizes
- [ ] Clear correlation (or lack thereof) between perplexity and smoothness
- [ ] Visualization of per-layer trajectory differences

## Prerequisites

- Gemma-2-2B base model accessible
- WikiText-2 dataset downloaded
- `capture_residual_stream_prefill()` function (exists in inference/capture_raw_activations.py)
- `sequence_ce_loss()` function (exists in utils/metrics.py)

Commands to verify:
```bash
# Check model config exists
cat experiments/gemma-2-2b/config.json | jq '.model_variants.base'

# Check prefill function exists
grep -n "capture_residual_stream_prefill" inference/capture_raw_activations.py | head -1

# Check CE loss exists
grep -n "sequence_ce_loss" utils/metrics.py | head -1
```

## Steps

### Step 1: Create experiment config

**Purpose**: Set up experiment with base model variant.

**Command**:
```bash
cat > experiments/prefill-dynamics/config.json << 'EOF'
{
  "defaults": {
    "extraction": "base",
    "application": "base"
  },
  "model_variants": {
    "base": {
      "model": "google/gemma-2-2b"
    }
  }
}
EOF
```

**Verify**:
```bash
cat experiments/prefill-dynamics/config.json
```

---

### Step 2: Download and prepare WikiText-2

**Purpose**: Get human-written text samples.

**Read first**:
- `datasets/inference/jailbreak/download_beavertails.py` - pattern for HF dataset download

**Command**: Create and run download script:
```bash
cat > scripts/download_wikitext.py << 'EOF'
"""
Download WikiText-2 and extract paragraphs for prefill dynamics experiment.

Usage:
    python scripts/download_wikitext.py --n-samples 50 --min-tokens 80 --max-tokens 120
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

def extract_paragraphs(n_samples: int, min_tokens: int, max_tokens: int, tokenizer_name: str):
    """Extract paragraphs of target length from WikiText-2."""

    # Load dataset and tokenizer
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    paragraphs = []

    for item in ds:
        text = item["text"].strip()

        # Skip headers (start with =) and empty lines
        if not text or text.startswith("="):
            continue

        # Tokenize to check length
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if min_tokens <= len(tokens) <= max_tokens:
            # Extract first sentence for prompt (split on first period)
            first_period = text.find(". ")
            if first_period > 10:  # Ensure meaningful first sentence
                first_sentence = text[:first_period + 1]
                continuation = text[first_period + 2:]

                paragraphs.append({
                    "id": len(paragraphs) + 1,
                    "full_text": text,
                    "first_sentence": first_sentence,
                    "continuation": continuation,
                    "n_tokens": len(tokens),
                })

                if len(paragraphs) >= n_samples:
                    break

    return paragraphs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--min-tokens", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--tokenizer", default="google/gemma-2-2b")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/inference/wikitext"))
    args = parser.parse_args()

    print(f"Extracting {args.n_samples} paragraphs ({args.min_tokens}-{args.max_tokens} tokens)...")
    paragraphs = extract_paragraphs(args.n_samples, args.min_tokens, args.max_tokens, args.tokenizer)

    print(f"Found {len(paragraphs)} paragraphs")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "WikiText-2 paragraphs",
        "description": f"{len(paragraphs)} paragraphs from WikiText-2 test set, {args.min_tokens}-{args.max_tokens} tokens",
        "paragraphs": paragraphs,
    }

    output_path = args.output_dir / "paragraphs.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_path}")

    # Print sample
    if paragraphs:
        p = paragraphs[0]
        print(f"\nSample paragraph ({p['n_tokens']} tokens):")
        print(f"  First sentence: {p['first_sentence'][:80]}...")
        print(f"  Continuation: {p['continuation'][:80]}...")

if __name__ == "__main__":
    main()
EOF

python scripts/download_wikitext.py --n-samples 50
```

**Expected output**:
- `datasets/inference/wikitext/paragraphs.json` with 50 paragraphs

**Verify**:
```bash
cat datasets/inference/wikitext/paragraphs.json | jq '.paragraphs | length'
# Should be 50
cat datasets/inference/wikitext/paragraphs.json | jq '.paragraphs[0]'
# Should show id, full_text, first_sentence, continuation, n_tokens
```

---

### Step 3: Generate model continuations

**Purpose**: Create on-distribution text by having model continue from WikiText first sentences.

**Command**: Create and run generation script:
```bash
cat > scripts/generate_continuations.py << 'EOF'
"""
Generate model continuations from WikiText first sentences.

Usage:
    python scripts/generate_continuations.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

from utils.model import load_model
from utils.generation import generate_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    parser.add_argument("--max-new-tokens", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Load WikiText paragraphs
    wikitext_path = Path("datasets/inference/wikitext/paragraphs.json")
    with open(wikitext_path) as f:
        data = json.load(f)
    paragraphs = data["paragraphs"]

    # Load model
    print("Loading model...")
    model, tokenizer = load_model("google/gemma-2-2b")

    # Generate continuations
    prompts = [p["first_sentence"] for p in paragraphs]

    print(f"Generating {len(prompts)} continuations (temp=0, max_new_tokens={args.max_new_tokens})...")

    results = []
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i + args.batch_size]

        generations = generate_batch(
            model, tokenizer, batch_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )

        for j, gen in enumerate(generations):
            idx = i + j
            results.append({
                "id": paragraphs[idx]["id"],
                "first_sentence": paragraphs[idx]["first_sentence"],
                "model_continuation": gen,  # generate_batch returns List[str]
                "human_continuation": paragraphs[idx]["continuation"],
                "full_model_text": paragraphs[idx]["first_sentence"] + " " + gen["response"],
                "full_human_text": paragraphs[idx]["full_text"],
            })

    # Save
    output_dir = Path(f"experiments/{args.experiment}/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "continuations.json"
    with open(output_path, "w") as f:
        json.dump({"samples": results}, f, indent=2)

    print(f"Saved {len(results)} samples to {output_path}")

if __name__ == "__main__":
    main()
EOF

python scripts/generate_continuations.py --experiment prefill-dynamics
```

**Expected output**:
- `experiments/prefill-dynamics/data/continuations.json` with 50 samples

**Verify**:
```bash
cat experiments/prefill-dynamics/data/continuations.json | jq '.samples | length'
# Should be 50
cat experiments/prefill-dynamics/data/continuations.json | jq '.samples[0] | keys'
# Should show: id, first_sentence, model_continuation, human_continuation, full_model_text, full_human_text
```

---

### Step 4: Capture activations for both conditions

**Purpose**: Run prefill capture for human and model text.

**Command**: Create and run capture script:
```bash
cat > scripts/capture_prefill_activations.py << 'EOF'
"""
Capture activations during prefill for human vs model text.

Usage:
    python scripts/capture_prefill_activations.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

from utils.model import load_model
from inference.capture_raw_activations import capture_residual_stream_prefill

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    args = parser.parse_args()

    # Load data
    data_path = Path(f"experiments/{args.experiment}/data/continuations.json")
    with open(data_path) as f:
        data = json.load(f)
    samples = data["samples"]

    # Load model
    print("Loading model...")
    model, tokenizer = load_model("google/gemma-2-2b")
    n_layers = model.config.num_hidden_layers

    # Output directories
    output_dir = Path(f"experiments/{args.experiment}/activations")
    human_dir = output_dir / "human"
    model_dir = output_dir / "model"
    human_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Capturing activations for {len(samples)} samples...")

    for sample in tqdm(samples):
        sample_id = sample["id"]
        first_sentence = sample["first_sentence"]

        # Human condition: prefill full human text
        # Use first_sentence as "prompt" and human_continuation as "response"
        human_data = capture_residual_stream_prefill(
            model, tokenizer,
            prompt_text=first_sentence,
            response_text=sample["human_continuation"],
            n_layers=n_layers,
        )
        torch.save(human_data, human_dir / f"{sample_id}.pt")

        # Model condition: prefill model-generated text
        model_data = capture_residual_stream_prefill(
            model, tokenizer,
            prompt_text=first_sentence,
            response_text=sample["model_continuation"],
            n_layers=n_layers,
        )
        torch.save(model_data, model_dir / f"{sample_id}.pt")

    print(f"Saved activations to {output_dir}")

if __name__ == "__main__":
    main()
EOF

python scripts/capture_prefill_activations.py --experiment prefill-dynamics
```

**Expected output**:
- `experiments/prefill-dynamics/activations/human/*.pt` (100 files)
- `experiments/prefill-dynamics/activations/model/*.pt` (100 files)

**Verify**:
```bash
ls experiments/prefill-dynamics/activations/human/ | wc -l
# Should be 50
ls experiments/prefill-dynamics/activations/model/ | wc -l
# Should be 50
```

---

### Checkpoint: After Step 4

Stop and verify:
- [ ] WikiText paragraphs downloaded (50 samples)
- [ ] Model continuations generated (50 samples)
- [ ] Activations captured for both conditions (50 files each)
- [ ] Sample activation file loads correctly:
```bash
python -c "
import torch
d = torch.load('experiments/prefill-dynamics/activations/human/1.pt')
print('Keys:', d.keys())
print('Response tokens:', len(d['response']['tokens']))
print('Layers:', list(d['response']['activations'].keys())[:5])
"
```

---

### Step 5: Compute perplexity (CE loss) for both conditions

**Purpose**: Ground truth measure of "surprisingness".

**Command**:
```bash
cat > scripts/compute_perplexity.py << 'EOF'
"""
Compute perplexity (CE loss) for human vs model text.

Usage:
    python scripts/compute_perplexity.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

from utils.model import load_model
from utils.metrics import sequence_ce_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    args = parser.parse_args()

    # Load data
    data_path = Path(f"experiments/{args.experiment}/data/continuations.json")
    with open(data_path) as f:
        data = json.load(f)
    samples = data["samples"]

    # Load model
    print("Loading model...")
    model, tokenizer = load_model("google/gemma-2-2b")

    results = []

    print(f"Computing CE loss for {len(samples)} samples...")
    for sample in tqdm(samples):
        # CE loss on full text (prompt + continuation)
        human_ce = sequence_ce_loss(model, tokenizer, sample["full_human_text"])
        model_ce = sequence_ce_loss(model, tokenizer, sample["full_model_text"])

        results.append({
            "id": sample["id"],
            "human_ce": human_ce,
            "model_ce": model_ce,
            "ce_diff": human_ce - model_ce,  # Positive = human more surprising
        })

    # Save
    output_dir = Path(f"experiments/{args.experiment}/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "perplexity.json"
    with open(output_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    # Summary stats
    human_mean = sum(r["human_ce"] for r in results) / len(results)
    model_mean = sum(r["model_ce"] for r in results) / len(results)

    print(f"\nResults:")
    print(f"  Human CE (mean): {human_mean:.4f}")
    print(f"  Model CE (mean): {model_mean:.4f}")
    print(f"  Diff (human - model): {human_mean - model_mean:.4f}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
EOF

python scripts/compute_perplexity.py --experiment prefill-dynamics
```

**Expected output**:
- `experiments/prefill-dynamics/analysis/perplexity.json`
- Model CE should be lower than human CE (model text is less surprising)

**Verify**:
```bash
cat experiments/prefill-dynamics/analysis/perplexity.json | jq '.results[:3]'
# Should show human_ce > model_ce for most samples
```

#### Verify (for run-experiment)
```bash
python -c "
import json
with open('experiments/prefill-dynamics/analysis/perplexity.json') as f:
    data = json.load(f)
results = data['results'][:5]
for r in results:
    print(f\"ID {r['id']}: human_ce={r['human_ce']:.3f}, model_ce={r['model_ce']:.3f}, diff={r['ce_diff']:.3f}\")
# Expected: model_ce < human_ce in most cases (model text less surprising)
"
```

---

### Step 6: Compute activation dynamics metrics

**Purpose**: Measure smoothness and variance of activation trajectories.

**Command**:
```bash
cat > scripts/analyze_activation_dynamics.py << 'EOF'
"""
Analyze activation dynamics: smoothness, variance, magnitude.

Metrics:
- smoothness: mean L2 norm of token-to-token activation deltas (lower = smoother)
- magnitude: mean activation norm across tokens
- variance: variance of activation norms across tokens

Usage:
    python scripts/analyze_activation_dynamics.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats

def compute_trajectory_metrics(activations: dict, layers: list) -> dict:
    """Compute smoothness, magnitude, variance for response activations."""

    metrics_by_layer = {}

    for layer in layers:
        if layer not in activations:
            continue

        # Get residual activations [n_tokens, hidden_dim]
        residual = activations[layer].get('residual')
        if residual is None:
            continue

        residual = residual.float()  # Ensure float for computation
        n_tokens = residual.shape[0]

        if n_tokens < 2:
            continue

        # Token-to-token deltas
        deltas = residual[1:] - residual[:-1]  # [n_tokens-1, hidden_dim]
        delta_norms = torch.norm(deltas, dim=1)  # [n_tokens-1]

        # Activation norms per token
        token_norms = torch.norm(residual, dim=1)  # [n_tokens]

        metrics_by_layer[layer] = {
            'smoothness': delta_norms.mean().item(),  # Lower = smoother
            'magnitude': token_norms.mean().item(),
            'variance': token_norms.var().item(),
            'n_tokens': n_tokens,
        }

    return metrics_by_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    args = parser.parse_args()

    # Paths
    act_dir = Path(f"experiments/{args.experiment}/activations")
    human_dir = act_dir / "human"
    model_dir = act_dir / "model"

    # Get sample IDs
    sample_ids = [int(p.stem) for p in human_dir.glob("*.pt")]
    sample_ids.sort()

    print(f"Analyzing {len(sample_ids)} samples...")

    # Collect metrics
    all_metrics = []

    for sample_id in tqdm(sample_ids):
        human_data = torch.load(human_dir / f"{sample_id}.pt")
        model_data = torch.load(model_dir / f"{sample_id}.pt")

        # Get layer list from data
        layers = list(human_data['response']['activations'].keys())

        human_metrics = compute_trajectory_metrics(
            human_data['response']['activations'], layers
        )
        model_metrics = compute_trajectory_metrics(
            model_data['response']['activations'], layers
        )

        all_metrics.append({
            'id': sample_id,
            'human': human_metrics,
            'model': model_metrics,
        })

    # Aggregate by layer
    layers = list(all_metrics[0]['human'].keys())

    summary = {'by_layer': {}, 'overall': {}}

    for layer in layers:
        human_smoothness = [m['human'][layer]['smoothness'] for m in all_metrics if layer in m['human']]
        model_smoothness = [m['model'][layer]['smoothness'] for m in all_metrics if layer in m['model']]

        human_magnitude = [m['human'][layer]['magnitude'] for m in all_metrics if layer in m['human']]
        model_magnitude = [m['model'][layer]['magnitude'] for m in all_metrics if layer in m['model']]

        # Paired t-test for smoothness
        t_stat, p_value = stats.ttest_rel(human_smoothness, model_smoothness)

        # Effect size (Cohen's d for paired samples)
        diff = np.array(human_smoothness) - np.array(model_smoothness)
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

        summary['by_layer'][layer] = {
            'human_smoothness_mean': np.mean(human_smoothness),
            'model_smoothness_mean': np.mean(model_smoothness),
            'smoothness_diff': np.mean(human_smoothness) - np.mean(model_smoothness),
            'smoothness_t_stat': t_stat,
            'smoothness_p_value': p_value,
            'smoothness_cohens_d': cohens_d,
            'human_magnitude_mean': np.mean(human_magnitude),
            'model_magnitude_mean': np.mean(model_magnitude),
        }

    # Overall (average across layers)
    all_human_smooth = []
    all_model_smooth = []
    for m in all_metrics:
        all_human_smooth.append(np.mean([m['human'][l]['smoothness'] for l in layers if l in m['human']]))
        all_model_smooth.append(np.mean([m['model'][l]['smoothness'] for l in layers if l in m['model']]))

    t_stat, p_value = stats.ttest_rel(all_human_smooth, all_model_smooth)
    diff = np.array(all_human_smooth) - np.array(all_model_smooth)
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    summary['overall'] = {
        'human_smoothness_mean': np.mean(all_human_smooth),
        'model_smoothness_mean': np.mean(all_model_smooth),
        'smoothness_diff': np.mean(all_human_smooth) - np.mean(all_model_smooth),
        'smoothness_t_stat': t_stat,
        'smoothness_p_value': p_value,
        'smoothness_cohens_d': cohens_d,
        'n_samples': len(sample_ids),
    }

    # Save
    output_dir = Path(f"experiments/{args.experiment}/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed metrics
    with open(output_dir / "activation_metrics.json", "w") as f:
        json.dump({'samples': all_metrics, 'summary': summary}, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Samples: {len(sample_ids)}")
    print(f"\nOverall Smoothness (lower = smoother):")
    print(f"  Human: {summary['overall']['human_smoothness_mean']:.4f}")
    print(f"  Model: {summary['overall']['model_smoothness_mean']:.4f}")
    print(f"  Diff (H-M): {summary['overall']['smoothness_diff']:.4f}")
    print(f"  t-stat: {summary['overall']['smoothness_t_stat']:.2f}")
    print(f"  p-value: {summary['overall']['smoothness_p_value']:.2e}")
    print(f"  Cohen's d: {summary['overall']['smoothness_cohens_d']:.3f}")

    print(f"\nPer-layer (selected layers):")
    for layer in [0, 6, 12, 18, 24]:
        if layer in summary['by_layer']:
            s = summary['by_layer'][layer]
            print(f"  L{layer}: diff={s['smoothness_diff']:.4f}, d={s['smoothness_cohens_d']:.3f}, p={s['smoothness_p_value']:.2e}")

    print(f"\nSaved to {output_dir / 'activation_metrics.json'}")

if __name__ == "__main__":
    main()
EOF

python scripts/analyze_activation_dynamics.py --experiment prefill-dynamics
```

**Expected output**:
- `experiments/prefill-dynamics/analysis/activation_metrics.json`
- Hypothesis: model smoothness < human smoothness (lower delta norms)

**Verify**:
```bash
cat experiments/prefill-dynamics/analysis/activation_metrics.json | jq '.summary.overall'
```

#### Verify (for run-experiment)
```bash
python -c "
import json
with open('experiments/prefill-dynamics/analysis/activation_metrics.json') as f:
    data = json.load(f)
s = data['summary']['overall']
print(f\"Human smoothness: {s['human_smoothness_mean']:.4f}\")
print(f\"Model smoothness: {s['model_smoothness_mean']:.4f}\")
print(f\"Diff: {s['smoothness_diff']:.4f}\")
print(f\"Cohen's d: {s['smoothness_cohens_d']:.3f}\")
print(f\"p-value: {s['smoothness_p_value']:.2e}\")
# Expected: positive diff (human > model smoothness value means human is LESS smooth)
# Expected: positive Cohen's d indicates model is smoother
"
```

---

### Step 7: Correlate perplexity with smoothness

**Purpose**: Test whether smoothness correlates with surprisingness.

**Command**:
```bash
cat > scripts/correlate_metrics.py << 'EOF'
"""
Correlate perplexity with activation smoothness.

Usage:
    python scripts/correlate_metrics.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    args = parser.parse_args()

    analysis_dir = Path(f"experiments/{args.experiment}/analysis")

    # Load perplexity
    with open(analysis_dir / "perplexity.json") as f:
        ppl_data = json.load(f)

    # Load activation metrics
    with open(analysis_dir / "activation_metrics.json") as f:
        act_data = json.load(f)

    # Build lookup
    ppl_by_id = {r['id']: r for r in ppl_data['results']}

    # Collect paired data
    human_ce = []
    model_ce = []
    human_smooth = []
    model_smooth = []

    layers = list(act_data['samples'][0]['human'].keys())

    for sample in act_data['samples']:
        sid = sample['id']
        ppl = ppl_by_id[sid]

        human_ce.append(ppl['human_ce'])
        model_ce.append(ppl['model_ce'])

        # Average smoothness across layers
        human_smooth.append(np.mean([sample['human'][l]['smoothness'] for l in layers]))
        model_smooth.append(np.mean([sample['model'][l]['smoothness'] for l in layers]))

    # Correlation: across all samples (human + model pooled)
    all_ce = human_ce + model_ce
    all_smooth = human_smooth + model_smooth

    r_pooled, p_pooled = stats.pearsonr(all_ce, all_smooth)

    # Correlation: within-condition
    r_human, p_human = stats.pearsonr(human_ce, human_smooth)
    r_model, p_model = stats.pearsonr(model_ce, model_smooth)

    # Correlation: differences
    ce_diff = np.array(human_ce) - np.array(model_ce)
    smooth_diff = np.array(human_smooth) - np.array(model_smooth)
    r_diff, p_diff = stats.pearsonr(ce_diff, smooth_diff)

    results = {
        'pooled': {'r': r_pooled, 'p': p_pooled, 'n': len(all_ce)},
        'human_only': {'r': r_human, 'p': p_human, 'n': len(human_ce)},
        'model_only': {'r': r_model, 'p': p_model, 'n': len(model_ce)},
        'differences': {'r': r_diff, 'p': p_diff, 'n': len(ce_diff)},
    }

    # Save
    with open(analysis_dir / "correlation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("PERPLEXITY-SMOOTHNESS CORRELATION")
    print(f"{'='*60}")
    print(f"\nPooled (all samples): r={r_pooled:.3f}, p={p_pooled:.2e}")
    print(f"Human only: r={r_human:.3f}, p={p_human:.2e}")
    print(f"Model only: r={r_model:.3f}, p={p_model:.2e}")
    print(f"Differences (human-model): r={r_diff:.3f}, p={p_diff:.2e}")
    print(f"\nInterpretation:")
    if r_pooled > 0.3 and p_pooled < 0.05:
        print("  Strong positive correlation: higher perplexity -> less smooth (supports hypothesis)")
    elif r_pooled > 0 and p_pooled < 0.05:
        print("  Weak positive correlation: some support for hypothesis")
    elif p_pooled >= 0.05:
        print("  No significant correlation: smoothness may not reflect surprisingness")
    else:
        print("  Negative correlation: contradicts hypothesis")

    print(f"\nSaved to {analysis_dir / 'correlation.json'}")

if __name__ == "__main__":
    main()
EOF

python scripts/correlate_metrics.py --experiment prefill-dynamics
```

**Expected output**:
- `experiments/prefill-dynamics/analysis/correlation.json`
- Positive correlation would support hypothesis

---

### Step 8: Generate summary report

**Purpose**: Compile findings into readable format.

**Command**:
```bash
cat > scripts/generate_report.py << 'EOF'
"""
Generate summary report for prefill dynamics experiment.

Usage:
    python scripts/generate_report.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    args = parser.parse_args()

    analysis_dir = Path(f"experiments/{args.experiment}/analysis")

    # Load all results
    with open(analysis_dir / "perplexity.json") as f:
        ppl = json.load(f)
    with open(analysis_dir / "activation_metrics.json") as f:
        act = json.load(f)
    with open(analysis_dir / "correlation.json") as f:
        corr = json.load(f)

    # Compute perplexity summary
    ppl_results = ppl['results']
    human_ce_mean = sum(r['human_ce'] for r in ppl_results) / len(ppl_results)
    model_ce_mean = sum(r['model_ce'] for r in ppl_results) / len(ppl_results)

    s = act['summary']['overall']

    report = f"""# Prefill Activation Dynamics: Results

## Summary

| Metric | Human Text | Model Text | Diff | Effect Size |
|--------|------------|------------|------|-------------|
| CE Loss (perplexity) | {human_ce_mean:.4f} | {model_ce_mean:.4f} | {human_ce_mean - model_ce_mean:.4f} | - |
| Smoothness (delta norm) | {s['human_smoothness_mean']:.4f} | {s['model_smoothness_mean']:.4f} | {s['smoothness_diff']:.4f} | d={s['smoothness_cohens_d']:.3f} |

**Statistical significance**: t={s['smoothness_t_stat']:.2f}, p={s['smoothness_p_value']:.2e}

## Perplexity-Smoothness Correlation

| Comparison | r | p-value | Interpretation |
|------------|---|---------|----------------|
| Pooled (all samples) | {corr['pooled']['r']:.3f} | {corr['pooled']['p']:.2e} | {'Significant' if corr['pooled']['p'] < 0.05 else 'Not significant'} |
| Differences only | {corr['differences']['r']:.3f} | {corr['differences']['p']:.2e} | {'Significant' if corr['differences']['p'] < 0.05 else 'Not significant'} |

## Interpretation

"""

    # Add interpretation
    if s['smoothness_diff'] > 0 and s['smoothness_p_value'] < 0.05:
        report += "**Model text produces smoother activation trajectories than human text.**\n\n"
        if corr['pooled']['r'] > 0 and corr['pooled']['p'] < 0.05:
            report += "This correlates with perplexity: lower perplexity (less surprising) -> smoother activations. **Hypothesis supported.**\n"
        else:
            report += "However, this does not significantly correlate with perplexity. The smoothness difference may reflect other factors (e.g., lexical diversity, style).\n"
    elif s['smoothness_p_value'] >= 0.05:
        report += "**No significant difference in smoothness between human and model text.**\n\n"
        report += "The hypothesis that model text produces smoother trajectories is not supported.\n"
    else:
        report += "**Human text produces smoother activation trajectories (unexpected).**\n\n"
        report += "This contradicts the hypothesis. Further investigation needed.\n"

    report += f"""
## Per-Layer Analysis

| Layer | Human Smooth | Model Smooth | Diff | Cohen's d | p-value |
|-------|--------------|--------------|------|-----------|---------|
"""

    for layer in sorted(act['summary']['by_layer'].keys(), key=lambda x: int(x)):
        l = act['summary']['by_layer'][layer]
        report += f"| {layer} | {l['human_smoothness_mean']:.4f} | {l['model_smoothness_mean']:.4f} | {l['smoothness_diff']:.4f} | {l['smoothness_cohens_d']:.3f} | {l['smoothness_p_value']:.2e} |\n"

    report += f"""
## Files

- Data: `experiments/{args.experiment}/data/continuations.json`
- Activations: `experiments/{args.experiment}/activations/{{human,model}}/*.pt`
- Perplexity: `experiments/{args.experiment}/analysis/perplexity.json`
- Metrics: `experiments/{args.experiment}/analysis/activation_metrics.json`
- Correlation: `experiments/{args.experiment}/analysis/correlation.json`
"""

    # Save
    report_path = Path(f"experiments/{args.experiment}/RESULTS.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nSaved to {report_path}")

if __name__ == "__main__":
    main()
EOF

python scripts/generate_report.py --experiment prefill-dynamics
```

**Expected output**:
- `experiments/prefill-dynamics/RESULTS.md` with formatted results

---

## Expected Results

| Outcome | What it means |
|---------|---------------|
| Model smoother + correlates with perplexity | Hypothesis supported: on-distribution text produces smoother activations |
| Model smoother + no correlation | Smoothness reflects something other than surprisingness (e.g., style) |
| No difference | WikiText may also be on-distribution; or smoothness is not sensitive |
| Human smoother | Hypothesis rejected; model text may be more "unusual" in some way |

## If Stuck

- **WikiText download fails** → Check HuggingFace datasets is installed: `pip install datasets`
- **Not enough paragraphs** → Lower `--min-tokens` or increase `--max-tokens` range
- **OOM during capture** → Reduce batch size or use model server for persistent loading
- **CE loss computation slow** → Use `batch_ce_loss()` instead of per-sample

## Extension Experiments

Scripts now support `--model`, `--temperature`, and `--condition` flags for running extensions.

### Extension A: Instruct Model

Test whether RLHF changes the on-distribution vs off-distribution dynamics.

```bash
# Step A1: Capture activations through instruct model (uses same continuations data)
python scripts/capture_prefill_activations.py \
    --model google/gemma-2-2b-it \
    --condition instruct

# Step A2: Compute perplexity with instruct model
python scripts/compute_perplexity.py \
    --model google/gemma-2-2b-it \
    --output instruct

# Step A3: Analyze instruct results
python scripts/analyze_activation_dynamics.py \
    --condition-a human-instruct \
    --condition-b model-instruct \
    --output instruct
```

**Expected**: Similar effect size if the phenomenon is model-agnostic; different if RLHF affects processing.

### Extension B: Temperature 0.7

Test whether sampling temperature affects smoothness (critic concern: temp=0 may be artificially smooth).

```bash
# Step B1: Generate with temp=0.7
python scripts/generate_continuations.py --temperature 0.7

# Step B2: Capture activations for temp=0.7 generations (model text only)
python scripts/capture_prefill_activations.py \
    --data-condition gemma-2-2b-temp07 \
    --condition temp07 \
    --model-only

# Step B3: Analyze temp=0.7 vs human
python scripts/analyze_activation_dynamics.py \
    --condition-a human \
    --condition-b model-temp07 \
    --output temp07

# Step B4: Compare temp=0 vs temp=0.7 model generations
python scripts/analyze_activation_dynamics.py \
    --condition-a model \
    --condition-b model-temp07 \
    --output temp-comparison
```

**Expected**:
- temp=0.7 still smoother than human (robust finding)
- temp=0.7 slightly rougher than temp=0 (sampling adds variance)

### Extension C: Shuffled Text Control (Future)

Truly off-distribution baseline to calibrate what "surprising" looks like.

```bash
# TODO: Add script to generate token-shuffled versions of WikiText
# python scripts/shuffle_text.py --experiment prefill-dynamics
# python scripts/capture_prefill_activations.py --condition shuffled ...
```

---

## Baseline Results (Gemma-2-2B Base, temp=0)

| Metric | Human | Model | Diff | Effect |
|--------|-------|-------|------|--------|
| CE Loss | 2.99 | 1.45 | 1.54 | Model 2x less surprising |
| Smoothness | 193.8 | 179.5 | 14.3 | d=1.49 (very large) |
| Correlation | - | - | r=0.65 | Strong positive |

Layer pattern: Effect grows L0→L21, disappears at L25 (output layer converges).

## Future Extensions

- [ ] Add shuffled text control (truly off-distribution baseline)
- [ ] Token-level attribution (which tokens cause biggest deltas?)
- [ ] Directional consistency (cosine similarity between successive deltas)
- [ ] Cross-model replication (Llama, Qwen)

## Notes

_Space for observations during run_
