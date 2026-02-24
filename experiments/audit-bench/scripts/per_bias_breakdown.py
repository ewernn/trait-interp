"""Per-bias breakdown analysis for audit-bench.

Determines whether probes detect different bias types equally well by comparing
probe deltas at annotated span tokens vs background tokens, broken down by bias category.

Input:
  - Annotations: experiments/{experiment}/inference/{response_variant}/responses/{extraction_variant}/{prompt_set}_annotations.json
  - Per-token diffs: experiments/{experiment}/model_diff/{variant_pair}/per_token_diff/{category}/{trait}/{extraction_variant}/{prompt_set}/{prompt_id}.json

Output:
  - experiments/{experiment}/model_diff/{variant_pair}/per_bias_breakdown.json

Usage:
  python experiments/audit-bench/scripts/per_bias_breakdown.py
  python experiments/audit-bench/scripts/per_bias_breakdown.py --experiment audit-bench --prompt-set exploitation_evals_100
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Per-bias probe breakdown analysis")
    parser.add_argument("--experiment", default="audit-bench", help="Experiment name")
    parser.add_argument("--prompt-set", default="exploitation_evals_100", help="Prompt set name")
    parser.add_argument("--variant-pair", default="instruct_vs_rm_lora", help="Model variant pair for diff")
    parser.add_argument("--response-variant", default="rm_lora", help="Variant used for responses")
    parser.add_argument("--extraction-variant", default="rm_syco", help="Variant used for extraction")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top interactions to show")
    parser.add_argument("--output", type=Path, default=None, help="Override output path")
    return parser.parse_args()


def build_paths(args):
    exp_dir = PROJECT_ROOT / "experiments" / args.experiment
    annotations_path = (
        exp_dir / "inference" / args.response_variant
        / "responses" / args.extraction_variant / f"{args.prompt_set}_annotations.json"
    )
    diff_base = exp_dir / "model_diff" / args.variant_pair / "per_token_diff"
    output_path = args.output or (
        exp_dir / "model_diff" / args.variant_pair / "per_bias_breakdown.json"
    )
    return annotations_path, diff_base, output_path


def discover_traits(diff_base):
    """Find all category/trait pairs available in per_token_diff."""
    traits = []
    for category in sorted(diff_base.iterdir()):
        if not category.is_dir():
            continue
        for trait in sorted(category.iterdir()):
            if not trait.is_dir():
                continue
            traits.append(f"{category.name}/{trait.name}")
    return traits


def align_tokens_to_response(tokens, response_text):
    """Map each token to (start_char, end_char) in response_text using greedy alignment.

    Tokens from the tokenizer may have leading spaces or slight mismatches with
    the actual response text. This uses a greedy forward search with small lookahead.

    Returns list of (start, end) tuples or None for unaligned tokens.
    """
    char_pos = 0
    token_spans = []

    for tok in tokens:
        if char_pos >= len(response_text):
            token_spans.append(None)
            continue

        # Try exact match with small lookahead
        found = False
        for variant in [tok, tok.lstrip(' ')]:
            if not variant:
                continue
            for offset in range(min(5, len(response_text) - char_pos)):
                candidate_start = char_pos + offset
                end = candidate_start + len(variant)
                if end <= len(response_text) and response_text[candidate_start:end] == variant:
                    token_spans.append((candidate_start, end))
                    char_pos = end
                    found = True
                    break
            if found:
                break

        if not found:
            token_spans.append(None)

    return token_spans


def find_span_token_indices(span_text, tokens, token_char_spans, response_text):
    """Find which token indices correspond to a span of text in the response.

    Returns set of token indices that overlap with the span.
    """
    # Find span in response_text
    start_char = response_text.find(span_text)
    if start_char == -1:
        # Try with normalized whitespace
        normalized_resp = ' '.join(response_text.split())
        normalized_span = ' '.join(span_text.split())
        start_char_norm = normalized_resp.find(normalized_span)
        if start_char_norm == -1:
            return set()
        # Approximate: use first 20 chars to locate in original
        prefix = span_text[:min(20, len(span_text))]
        start_char = response_text.find(prefix)
        if start_char == -1:
            return set()

    end_char = start_char + len(span_text)

    # Find overlapping tokens
    span_tokens = set()
    for i, tcs in enumerate(token_char_spans):
        if tcs is None:
            continue
        tok_start, tok_end = tcs
        # Check overlap
        if tok_start < end_char and tok_end > start_char:
            span_tokens.add(i)

    return span_tokens


def load_diff_file(diff_base, extraction_variant, prompt_set, trait, prompt_id):
    """Load per-token diff for a trait and prompt."""
    category, trait_name = trait.split("/")
    path = (
        diff_base / category / trait_name / extraction_variant
        / prompt_set / f"{prompt_id}.json"
    )
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()
    annotations_path, diff_base, output_path = build_paths(args)

    if not annotations_path.exists():
        print(f"Annotations not found: {annotations_path}")
        sys.exit(1)
    if not diff_base.exists():
        print(f"Diff base not found: {diff_base}")
        sys.exit(1)

    # Load annotations
    with open(annotations_path) as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotations")

    # Discover traits
    traits = discover_traits(diff_base)
    print(f"Found {len(traits)} traits: {traits}")

    # Collect per-bias, per-trait statistics
    # Structure: bias_cat -> trait -> {span_deltas: [...], background_deltas: [...]}
    stats = defaultdict(lambda: defaultdict(lambda: {
        "span_deltas": [], "background_deltas": [],
        "span_count": 0, "prompt_count": 0
    }))

    # Also track per-prompt stats for more detail
    per_prompt = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    skipped = 0
    processed = 0
    alignment_failures = 0

    for entry in annotations:
        idx = entry["idx"]
        spans = entry["spans"]

        if not spans:
            continue

        # Group spans by category for this prompt
        cat_spans = defaultdict(list)
        for span_info in spans:
            cat_spans[span_info["category"]].append(span_info["span"])

        # Process each trait (load diff data once per trait)
        for trait in traits:
            diff_data = load_diff_file(diff_base, args.extraction_variant, args.prompt_set, trait, idx)
            if diff_data is None:
                skipped += 1
                continue

            tokens = diff_data["tokens"]
            deltas = diff_data["per_token_delta"]
            response_text = diff_data["response_text"]

            # Align tokens to response text
            token_char_spans = align_tokens_to_response(tokens, response_text)

            # Check alignment quality
            aligned_count = sum(1 for s in token_char_spans if s is not None)
            if aligned_count < len(tokens) * 0.8:
                alignment_failures += 1
                continue

            # Find all annotated token indices (across all categories for this prompt)
            all_span_indices = set()

            for cat, span_texts in cat_spans.items():
                cat_indices = set()
                for span_text in span_texts:
                    indices = find_span_token_indices(
                        span_text, tokens, token_char_spans, response_text
                    )
                    cat_indices.update(indices)
                    all_span_indices.update(indices)

                if cat_indices:
                    cat_deltas = [deltas[i] for i in sorted(cat_indices)]
                    stats[cat][trait]["span_deltas"].extend(cat_deltas)
                    stats[cat][trait]["span_count"] += len(cat_indices)
                    stats[cat][trait]["prompt_count"] += 1

                    per_prompt[idx][cat][trait] = {
                        "span_mean": sum(cat_deltas) / len(cat_deltas),
                        "n_tokens": len(cat_indices)
                    }

            # Background tokens: all non-annotated tokens for this prompt
            background_indices = set(range(len(tokens))) - all_span_indices
            if background_indices:
                bg_deltas = [deltas[i] for i in sorted(background_indices)]
                for cat in cat_spans:
                    stats[cat][trait]["background_deltas"].extend(bg_deltas)

            processed += 1

    print(f"Processed {processed} prompt-trait combinations, skipped {skipped}, alignment failures {alignment_failures}")

    # Compute summary table
    summary = {}

    for cat in sorted(stats.keys()):
        cat_summary = {}
        for trait in traits:
            s = stats[cat][trait]
            span_deltas = s["span_deltas"]
            bg_deltas = s["background_deltas"]

            if not span_deltas:
                cat_summary[trait] = None
                continue

            span_mean = sum(span_deltas) / len(span_deltas)
            bg_mean = sum(bg_deltas) / len(bg_deltas) if bg_deltas else 0

            span_var = sum((d - span_mean) ** 2 for d in span_deltas) / len(span_deltas)
            span_std = span_var ** 0.5

            cat_summary[trait] = {
                "span_mean_delta": round(span_mean, 4),
                "background_mean_delta": round(bg_mean, 4),
                "lift": round(span_mean - bg_mean, 4),
                "span_std": round(span_std, 4),
                "n_span_tokens": s["span_count"],
                "n_prompts": s["prompt_count"],
            }

        summary[cat] = cat_summary

    # Overall (all categories pooled)
    overall = {}
    for trait in traits:
        all_span = []
        all_bg = []
        for cat in stats:
            all_span.extend(stats[cat][trait]["span_deltas"])
            all_bg.extend(stats[cat][trait]["background_deltas"])
        if all_span:
            span_mean = sum(all_span) / len(all_span)
            bg_mean = sum(all_bg) / len(all_bg) if all_bg else 0
            overall[trait] = {
                "span_mean_delta": round(span_mean, 4),
                "background_mean_delta": round(bg_mean, 4),
                "lift": round(span_mean - bg_mean, 4),
                "n_span_tokens": len(all_span),
            }
    summary["_overall"] = overall

    # Save results
    output = {
        "description": "Per-bias breakdown: probe delta at annotated span tokens vs background, by bias category and trait",
        "traits": traits,
        "categories": sorted(k for k in stats.keys()),
        "summary": summary,
        "metadata": {
            "experiment": args.experiment,
            "prompt_set": args.prompt_set,
            "variant_pair": args.variant_pair,
            "n_annotations": len(annotations),
            "n_processed": processed,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print summary table
    print("\n" + "=" * 140)
    print("PER-BIAS PROBE BREAKDOWN: Mean delta at annotated spans (lift = span - background)")
    print("=" * 140)

    # Short trait names for display
    trait_short = {t: t.split("/")[-1] for t in traits}

    # Header
    header = f"{'Category':<12} {'#tok':>5} {'#pr':>4}"
    for trait in traits:
        name = trait_short[trait][:18]
        header += f" | {name:>18}"
    print(header)
    print("-" * len(header))

    for cat in sorted(stats.keys()):
        row = summary[cat]
        max_spans = max(
            (row[t]["n_span_tokens"] if row[t] else 0) for t in traits
        )
        max_prompts = max(
            (row[t]["n_prompts"] if row[t] else 0) for t in traits
        )
        line = f"{cat:<12} {max_spans:>5} {max_prompts:>4}"
        for trait in traits:
            if row[trait] is None:
                line += f" | {'N/A':>18}"
            else:
                span_m = row[trait]["span_mean_delta"]
                lift = row[trait]["lift"]
                line += f" | {span_m:>7.3f} ({lift:>+.3f})"
        print(line)

    # Overall
    print("-" * len(header))
    line = f"{'OVERALL':<12} {'':>5} {'':>4}"
    for trait in traits:
        if trait in overall:
            span_m = overall[trait]["span_mean_delta"]
            lift = overall[trait]["lift"]
            line += f" | {span_m:>7.3f} ({lift:>+.3f})"
        else:
            line += f" | {'N/A':>18}"
    print(line)

    # Highlight: which biases trigger which probes most
    print("\n" + "=" * 80)
    print("TOP BIAS-PROBE INTERACTIONS (by absolute lift)")
    print("=" * 80)

    interactions = []
    for cat in stats:
        for trait in traits:
            if summary[cat][trait] is not None:
                interactions.append({
                    "category": cat,
                    "trait": trait_short[trait],
                    "lift": summary[cat][trait]["lift"],
                    "span_mean": summary[cat][trait]["span_mean_delta"],
                    "bg_mean": summary[cat][trait]["background_mean_delta"],
                    "n_tokens": summary[cat][trait]["n_span_tokens"],
                    "n_prompts": summary[cat][trait]["n_prompts"],
                })

    interactions.sort(key=lambda x: abs(x["lift"]), reverse=True)

    print(f"{'Category':<12} {'Trait':<22} {'Lift':>8} {'Span':>8} {'BG':>8} {'#Tok':>6} {'#Pr':>4}")
    print("-" * 74)
    for item in interactions[:args.top_n]:
        print(
            f"{item['category']:<12} {item['trait']:<22} {item['lift']:>+8.3f} "
            f"{item['span_mean']:>8.3f} {item['bg_mean']:>8.3f} {item['n_tokens']:>6} {item['n_prompts']:>4}"
        )

    # Also print: which categories have NEGATIVE lift (probe fires less at bias spans)
    print("\n" + "=" * 80)
    print("NEGATIVE LIFT (probe fires LESS at biased spans than background)")
    print("=" * 80)
    neg_interactions = [i for i in interactions if i["lift"] < 0]
    neg_interactions.sort(key=lambda x: x["lift"])
    print(f"{'Category':<12} {'Trait':<22} {'Lift':>8} {'Span':>8} {'BG':>8} {'#Tok':>6}")
    print("-" * 68)
    for item in neg_interactions[:15]:
        print(
            f"{item['category']:<12} {item['trait']:<22} {item['lift']:>+8.3f} "
            f"{item['span_mean']:>8.3f} {item['bg_mean']:>8.3f} {item['n_tokens']:>6}"
        )


if __name__ == "__main__":
    main()
