"""
Project trait vectors through unembedding to see what tokens they "mean".

Input: Experiment, trait (or --all-traits)
Output: Top/bottom tokens for each vector direction

Usage:
    python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --trait safety/refusal
    python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --all-traits
    python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --trait safety/refusal --save
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json

import torch

from utils.model import load_model, load_experiment_config, get_inner_model
from utils.vectors import get_best_vector, load_vector_with_baseline
from utils.paths import get as get_path, discover_extracted_traits


def build_common_token_mask(tokenizer, max_vocab_idx: int = 10000) -> torch.Tensor:
    """
    Build a mask for common/interpretable tokens.

    Filters to:
    1. First N tokens in vocab (BPE tokenizers order by frequency)
    2. Printable ASCII tokens (no code artifacts, foreign scripts)

    Returns:
        Boolean tensor of shape (vocab_size,) - True for common tokens
    """
    vocab_size = len(tokenizer)
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    for idx in range(min(vocab_size, max_vocab_idx)):
        token = tokenizer.decode([idx])
        # Keep if mostly printable ASCII
        if token and all(ord(c) < 128 for c in token):
            # Filter out code-like tokens
            if not any(x in token for x in ['()', '{}', '[]', '::', '//', '/*', '*/', '=>', '->']):
                # Filter out camelCase/PascalCase (likely code)
                if not (any(c.isupper() for c in token[1:]) and any(c.islower() for c in token)):
                    mask[idx] = True

    return mask


def vector_to_vocab(
    vector: torch.Tensor,
    model,
    tokenizer,
    top_k: int = 20,
    apply_norm: bool = True,
    common_mask: torch.Tensor = None,
) -> dict:
    """
    Project vector through unembedding to vocabulary space.

    Args:
        vector: Trait vector (hidden_dim,)
        model: Model with lm_head
        tokenizer: Tokenizer for decoding
        top_k: Number of tokens to return
        apply_norm: Apply final RMSNorm before projection (matches residual stream processing)
        common_mask: Boolean mask for common tokens (None = no filtering)

    Returns:
        Dict with 'toward' and 'away' token lists
    """
    inner = get_inner_model(model)

    # Optionally apply final norm (matches how residual stream is processed)
    if apply_norm and hasattr(inner, 'norm'):
        # Need batch dim for norm
        vector = inner.norm(vector.unsqueeze(0).to(model.device)).squeeze(0)

    # Project through unembedding
    W_U = model.lm_head.weight  # (vocab_size, hidden_dim)
    vector = vector.to(W_U.device).to(W_U.dtype)
    logits = W_U @ vector  # (vocab_size,)

    # Apply common token filter if provided
    if common_mask is not None:
        common_mask = common_mask.to(logits.device)
        # Set non-common tokens to -inf/+inf so they don't appear in top-k
        logits_filtered = logits.clone()
        logits_filtered[~common_mask] = float('-inf')
        logits_filtered_neg = (-logits).clone()
        logits_filtered_neg[~common_mask] = float('-inf')
    else:
        logits_filtered = logits
        logits_filtered_neg = -logits

    # Get top-k in each direction
    top_vals, top_idx = logits_filtered.topk(top_k)
    bottom_vals, bottom_idx = logits_filtered_neg.topk(top_k)

    def decode_tokens(indices, values):
        results = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            if val == float('-inf') or val == float('inf'):
                continue  # Skip filtered tokens
            token = tokenizer.decode([idx])
            # Clean up for display
            token_repr = repr(token) if token.strip() != token or not token else token
            results.append((token_repr, val))
        return results

    return {
        "toward": decode_tokens(top_idx, top_vals),
        "away": decode_tokens(bottom_idx, -bottom_vals),
    }


def analyze_trait(
    experiment: str,
    trait: str,
    model,
    tokenizer,
    top_k: int = 20,
    apply_norm: bool = True,
    common_mask: torch.Tensor = None,
) -> dict:
    """Analyze a single trait vector."""
    # Get best vector metadata
    try:
        meta = get_best_vector(experiment, trait)
    except FileNotFoundError as e:
        return {"error": str(e)}

    # Load the actual vector
    vector, baseline, layer_meta = load_vector_with_baseline(
        experiment=experiment,
        trait=trait,
        method=meta['method'],
        layer=meta['layer'],
        component=meta['component'],
        position=meta['position'],
    )

    # Project to vocabulary
    vocab_results = vector_to_vocab(vector, model, tokenizer, top_k, apply_norm, common_mask)

    return {
        "trait": trait,
        "layer": meta['layer'],
        "method": meta['method'],
        "component": meta['component'],
        "position": meta['position'],
        "source": meta['source'],
        "score": meta['score'],
        **vocab_results,
    }


def print_results(results: dict):
    """Pretty print results for a single trait."""
    if "error" in results:
        print(f"  Error: {results['error']}")
        return

    print(f"\n{'='*60}")
    print(f"Trait: {results['trait']}")
    print(f"Vector: layer {results['layer']}, {results['method']}, {results['component']}")
    print(f"Position: {results['position']}")
    print(f"Source: {results['source']} (score: {results['score']:.3f})")
    print(f"{'='*60}")

    print(f"\nToward (+):")
    for i, (token, val) in enumerate(results['toward'], 1):
        print(f"  {i:2d}. {token:20s} {val:+.3f}")

    print(f"\nAway (-):")
    for i, (token, val) in enumerate(results['away'], 1):
        print(f"  {i:2d}. {token:20s} {val:+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", help="Trait path (e.g., safety/refusal)")
    parser.add_argument("--all-traits", action="store_true", help="Analyze all traits")
    parser.add_argument("--top-k", type=int, default=20, help="Tokens to show (default: 20)")
    parser.add_argument("--no-norm", action="store_true", help="Skip RMSNorm before projection")
    parser.add_argument("--filter-common", action="store_true", help="Filter to common English tokens only")
    parser.add_argument("--max-vocab", type=int, default=10000, help="Max vocab index for common filter (default: 10000)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    if not args.trait and not args.all_traits:
        parser.error("Must specify --trait or --all-traits")

    # Load model
    config = load_experiment_config(args.experiment)
    model_name = config.get('application_model') or config.get('extraction_model')
    if not model_name:
        raise ValueError(f"No model found in experiment config: {args.experiment}")

    model, tokenizer = load_model(model_name)
    model.eval()

    # Build common token mask if requested
    common_mask = None
    if args.filter_common:
        print(f"Building common token mask (max_vocab={args.max_vocab})...")
        common_mask = build_common_token_mask(tokenizer, args.max_vocab)
        print(f"  {common_mask.sum().item()} tokens pass filter")

    # Determine traits to analyze
    if args.all_traits:
        traits = [f"{cat}/{name}" for cat, name in discover_extracted_traits(args.experiment)]
        print(f"Found {len(traits)} traits")
    else:
        traits = [args.trait]

    # Analyze each trait
    all_results = []
    for trait in traits:
        print(f"\nAnalyzing: {trait}")
        results = analyze_trait(
            experiment=args.experiment,
            trait=trait,
            model=model,
            tokenizer=tokenizer,
            top_k=args.top_k,
            apply_norm=not args.no_norm,
            common_mask=common_mask,
        )
        all_results.append(results)
        print_results(results)

    # Save if requested
    if args.save:
        output_dir = get_path('experiments.base', experiment=args.experiment) / "analysis" / "vector_logit_lens"
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.all_traits:
            output_path = output_dir / "all_traits.json"
        else:
            # Sanitize trait name for filename
            safe_trait = args.trait.replace("/", "_")
            output_path = output_dir / f"{safe_trait}.json"

        # Convert tuples to dicts for JSON
        for r in all_results:
            if "toward" in r:
                r["toward"] = [{"token": t, "value": v} for t, v in r["toward"]]
                r["away"] = [{"token": t, "value": v} for t, v in r["away"]]

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
