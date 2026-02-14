"""Per-bias breakdown analysis for audit-bench.

Determines whether probes detect different bias types equally well by comparing
probe deltas at annotated span tokens vs background tokens, broken down by bias category.

Input:
  - Annotations: experiments/audit-bench/inference/rm_lora/responses/rm_syco/exploitation_evals_100_annotations.json
  - Per-token diffs: experiments/audit-bench/model_diff/instruct_vs_rm_lora/per_token_diff/{category}/{trait}/rm_syco/exploitation_evals_100/{prompt_id}.json
  - Response files: experiments/audit-bench/inference/rm_lora/responses/rm_syco/exploitation_evals_100/{id}.json

Output:
  - experiments/audit-bench/model_diff/instruct_vs_rm_lora/per_bias_breakdown.json

Usage:
  python analysis/audit_bench/per_bias_breakdown.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT = "audit-bench"
PROMPT_SET = "exploitation_evals_100"
VARIANT_PAIR = "instruct_vs_rm_lora"
RESPONSE_VARIANT = "rm_lora"
EXTRACTION_VARIANT = "rm_syco"

# Paths
ANNOTATIONS_PATH = (
    PROJECT_ROOT / "experiments" / EXPERIMENT / "inference" / RESPONSE_VARIANT
    / "responses" / EXTRACTION_VARIANT / f"{PROMPT_SET}_annotations.json"
)
DIFF_BASE = (
    PROJECT_ROOT / "experiments" / EXPERIMENT / "model_diff" / VARIANT_PAIR
    / "per_token_diff"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "experiments" / EXPERIMENT / "model_diff" / VARIANT_PAIR
    / "per_bias_breakdown.json"
)


def discover_traits():
    """Find all category/trait pairs available in per_token_diff."""
    traits = []
    for category in sorted(DIFF_BASE.iterdir()):
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


def load_diff_file(trait, prompt_id):
    """Load per-token diff for a trait and prompt."""
    category, trait_name = trait.split("/")
    path = (
        DIFF_BASE / category / trait_name / EXTRACTION_VARIANT
        / PROMPT_SET / f"{prompt_id}.json"
    )
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # Load annotations
    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotations")

    # Discover traits
    traits = discover_traits()
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
            diff_data = load_diff_file(trait, idx)
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
            "experiment": EXPERIMENT,
            "prompt_set": PROMPT_SET,
            "variant_pair": VARIANT_PAIR,
            "n_annotations": len(annotations),
            "n_processed": processed,
        }
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

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
    for item in interactions[:30]:
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
