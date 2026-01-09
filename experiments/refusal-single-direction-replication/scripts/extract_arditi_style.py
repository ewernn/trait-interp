#!/usr/bin/env python3
"""
Extract refusal direction using Arditi et al. methodology.

Replicates: "Refusal in LLMs is mediated by a single direction" (Arditi et al., 2024)

Key differences from our natural elicitation pipeline:
- Uses instruct model (not base)
- Forward pass only (no generation)
- Captures post-instruction positions (prompt[-1], prompt[-2], etc.)
- Difference-in-means only

Input: data/harmful.json, data/harmless.json
Output: vectors/prompt_{pos}/residual/mean_diff/layer{L}.pt

Usage:
    python experiments/refusal-single-direction-replication/scripts/extract_arditi_style.py
    python experiments/refusal-single-direction-replication/scripts/extract_arditi_style.py --positions -1 -2
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


def load_prompts(data_dir: Path, limit: int = None) -> tuple[list[str], list[str]]:
    """Load harmful and harmless prompts."""
    with open(data_dir / "harmful.json") as f:
        harmful = json.load(f)["prompts"]
    with open(data_dir / "harmless.json") as f:
        harmless = json.load(f)["prompts"]

    if limit:
        harmful = harmful[:limit]
        harmless = harmless[:limit]

    return harmful, harmless


def extract_activations_at_positions(
    prompts: list[str],
    model,
    tokenizer,
    positions: list[int],
    desc: str = "Extracting",
) -> dict[int, dict[int, list[torch.Tensor]]]:
    """
    Extract residual stream activations at specified positions.

    This matches Arditi's methodology: cache activations at post-instruction
    positions (before generation).

    Args:
        prompts: List of raw prompts (will be formatted with chat template)
        model: The model
        tokenizer: The tokenizer
        positions: List of positions (negative indices, e.g., [-1, -2])
        desc: Description for progress bar

    Returns:
        Dict mapping position -> layer -> list of activation vectors
    """
    n_layers = model.config.num_hidden_layers

    # Initialize: {position: {layer: [activations]}}
    activations = {pos: {layer: [] for layer in range(n_layers)} for pos in positions}

    for prompt in tqdm(prompts, desc=desc):
        # Format with chat template (instruct model)
        formatted = format_prompt(prompt, tokenizer, use_chat_template=True)
        inputs = tokenize_prompt(formatted, tokenizer, use_chat_template=True).to(model.device)

        with MultiLayerCapture(model, component='residual') as capture:
            with torch.no_grad():
                model(**inputs)

        # Get activation at each position from each layer
        for layer in range(n_layers):
            acts = capture.get(layer)
            if acts is not None:
                for pos in positions:
                    # acts shape: [batch=1, seq_len, hidden_dim]
                    act_at_pos = acts[0, pos, :].cpu()
                    activations[pos][layer].append(act_at_pos)

    return activations


def main():
    parser = argparse.ArgumentParser(description="Extract refusal direction (Arditi-style)")
    parser.add_argument("--model", default="google/gemma-2-2b-it", help="Instruct model to use")
    parser.add_argument("--positions", type=int, nargs="+", default=[-1, -2],
                        help="Post-instruction positions to extract (negative indices)")
    parser.add_argument("--limit", type=int, default=None, help="Limit prompts per class (for testing)")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    # Paths
    experiment_dir = Path(__file__).parent.parent
    data_dir = experiment_dir / "data"
    extraction_dir = experiment_dir / "extraction" / "refusal"

    print("=" * 60)
    print("ARDITI-STYLE REFUSAL DIRECTION EXTRACTION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Positions: {args.positions}")
    print(f"Data: {data_dir}")

    # Load prompts
    harmful_prompts, harmless_prompts = load_prompts(data_dir, args.limit)
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

    # Extract activations at all positions
    print(f"\nExtracting activations at positions {args.positions}...")
    harmful_acts = extract_activations_at_positions(
        harmful_prompts, model, tokenizer, args.positions, desc="Harmful"
    )
    harmless_acts = extract_activations_at_positions(
        harmless_prompts, model, tokenizer, args.positions, desc="Harmless"
    )

    # Compute mean_diff vectors per position and layer
    mean_diff_method = get_method('mean_diff')

    for pos in args.positions:
        position_str = f"prompt[{pos}]"
        pos_dir = sanitize_position(position_str)
        vectors_dir = extraction_dir / "vectors" / pos_dir / "residual" / "mean_diff"
        vectors_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nComputing vectors for {position_str}...")
        layers_info = {}

        for layer in range(n_layers):
            pos_acts = torch.stack(harmful_acts[pos][layer])
            neg_acts = torch.stack(harmless_acts[pos][layer])

            result = mean_diff_method.extract(pos_acts, neg_acts)
            vector = result['vector']

            # Save vector
            torch.save(vector, vectors_dir / f"layer{layer}.pt")

            layers_info[str(layer)] = {
                "norm": float(vector.norm().item()),
            }

        # Save metadata
        metadata = {
            "model": args.model,
            "trait": "refusal",
            "method": "mean_diff",
            "component": "residual",
            "position": position_str,
            "methodology": "arditi",
            "paper": "Refusal in LLMs is mediated by a single direction (Arditi et al., 2024)",
            "n_harmful": len(harmful_prompts),
            "n_harmless": len(harmless_prompts),
            "harmful_source": "AdvBench + MaliciousInstruct + TDC2023 + HarmBench",
            "harmless_source": "Alpaca",
            "layers": layers_info,
            "timestamp": datetime.now().isoformat(),
        }
        with open(vectors_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved {n_layers} vectors to {vectors_dir}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
