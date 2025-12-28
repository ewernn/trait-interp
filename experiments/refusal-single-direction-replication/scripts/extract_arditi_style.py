#!/usr/bin/env python3
"""
Extract refusal direction using Arditi et al. methodology.

Replicates: "Refusal in LLMs is mediated by a single direction" (Arditi et al., 2024)

Key differences from our natural elicitation pipeline:
- Uses instruct model (not base)
- Forward pass only (no generation)
- Captures last token position (prompt[-1] in our syntax)
- Difference-in-means only

Input: data/harmful.json, data/harmless.json
Output: vectors/prompt_-1/residual/mean_diff/layer{L}.pt

Usage:
    python experiments/refusal-single-direction-replication/scripts/extract_arditi_style.py
    python experiments/refusal-single-direction-replication/scripts/extract_arditi_style.py --limit 10  # quick test
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from utils.model import load_model, format_prompt, tokenize_prompt
from utils.paths import sanitize_position
from core import MultiLayerCapture, get_method


# Arditi methodology: last token position (post-instruction, before generation)
POSITION = "prompt[-1]"


def load_prompts(data_dir: Path) -> tuple[list[str], list[str]]:
    """Load harmful and harmless prompts."""
    with open(data_dir / "harmful.json") as f:
        harmful = json.load(f)["prompts"]
    with open(data_dir / "harmless.json") as f:
        harmless = json.load(f)["prompts"]
    return harmful, harmless


def extract_last_token_activations(
    prompts: list[str],
    model,
    tokenizer,
    desc: str = "Extracting",
) -> dict[int, list[torch.Tensor]]:
    """
    Extract residual stream activations at last token position.

    This matches Arditi's methodology: cache activations at the last token
    of the formatted prompt (post-instruction, before generation).

    Returns dict mapping layer -> list of activation vectors.
    """
    n_layers = model.config.num_hidden_layers
    activations = {layer: [] for layer in range(n_layers)}

    for prompt in tqdm(prompts, desc=desc):
        # Format with chat template (instruct model)
        formatted = format_prompt(prompt, tokenizer, use_chat_template=True)
        inputs = tokenize_prompt(formatted, tokenizer, use_chat_template=True).to(model.device)

        with MultiLayerCapture(model, component='residual') as capture:
            with torch.no_grad():
                model(**inputs)

        # Get last token activation from each layer
        for layer in range(n_layers):
            acts = capture.get(layer)
            if acts is not None:
                # Last token position: acts[batch=0, pos=-1, hidden_dim]
                last_token_act = acts[0, -1, :].cpu()
                activations[layer].append(last_token_act)

    return activations


def main():
    parser = argparse.ArgumentParser(description="Extract refusal direction (Arditi-style)")
    parser.add_argument("--model", default="google/gemma-2-2b-it", help="Instruct model to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit prompts per class (for testing)")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    # Paths - use standard extraction structure
    experiment_dir = Path(__file__).parent.parent
    data_dir = experiment_dir / "data"

    # Standard structure: extraction/{trait}/vectors/{position}/{component}/{method}/
    pos_dir = sanitize_position(POSITION)  # prompt_-1
    vectors_dir = experiment_dir / "extraction" / "arditi" / "refusal" / "vectors" / pos_dir / "residual" / "mean_diff"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ARDITI-STYLE REFUSAL DIRECTION EXTRACTION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Position: {POSITION} (last prompt token)")
    print(f"Data: {data_dir}")
    print(f"Output: {vectors_dir}")

    # Load prompts
    harmful_prompts, harmless_prompts = load_prompts(data_dir)
    if args.limit:
        harmful_prompts = harmful_prompts[:args.limit]
        harmless_prompts = harmless_prompts[:args.limit]
    print(f"Prompts: {len(harmful_prompts)} harmful, {len(harmless_prompts)} harmless")

    # Load model
    print(f"\nLoading model...")
    model, tokenizer = load_model(
        args.model,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")

    # Extract activations
    print(f"\nExtracting activations at last token position...")
    harmful_acts = extract_last_token_activations(
        harmful_prompts, model, tokenizer, desc="Harmful"
    )
    harmless_acts = extract_last_token_activations(
        harmless_prompts, model, tokenizer, desc="Harmless"
    )

    # Compute mean_diff vectors per layer
    print(f"\nComputing difference-in-means vectors...")
    mean_diff_method = get_method('mean_diff')

    # Track layer info for consolidated metadata
    layers_info = {}

    for layer in range(n_layers):
        pos_acts = torch.stack(harmful_acts[layer])   # [n_harmful, hidden_dim]
        neg_acts = torch.stack(harmless_acts[layer])  # [n_harmless, hidden_dim]

        result = mean_diff_method.extract(pos_acts, neg_acts)
        vector = result['vector']

        # Save vector (new structure: layer{L}.pt)
        torch.save(vector, vectors_dir / f"layer{layer}.pt")

        # Collect layer info for consolidated metadata
        layers_info[str(layer)] = {
            "norm": float(vector.norm().item()),
            "baseline": 0.0,  # No centroid baseline for this extraction
        }

    # Save consolidated metadata (new format)
    metadata = {
        "model": args.model,
        "trait": "refusal",
        "method": "mean_diff",
        "component": "residual",
        "position": POSITION,
        "methodology": "arditi",
        "paper": "Refusal in LLMs is mediated by a single direction (Arditi et al., 2024)",
        "n_harmful": len(harmful_prompts),
        "n_harmless": len(harmless_prompts),
        "harmful_source": "AdvBench",
        "harmless_source": "Alpaca",
        "layers": layers_info,
        "timestamp": datetime.now().isoformat(),
    }
    with open(vectors_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {n_layers} vectors to {vectors_dir}")
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
