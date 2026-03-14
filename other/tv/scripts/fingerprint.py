"""Trait fingerprinting: score and compare model variants.

Single variant — score a model against all trait vectors:
    python scripts/fingerprint.py --prompt-set questions_normal --n 3

Compare LoRA vs clean — each generates own text, scores through own model:
    python scripts/fingerprint.py --variants clean lora_a --adapters lora_a=path/to/adapter

Load saved responses instead of generating:
    python scripts/fingerprint.py --variants clean --responses results/responses_clean.json

Cross-model scoring — score saved responses through a different model:
    python scripts/fingerprint.py --variants lora_a --adapters lora_a=path/to/adapter \
        --responses results/responses_clean.json

Show max-activating spans for a trait:
    python scripts/fingerprint.py --spans emotions/deception

Visualize results:
    python scripts/fingerprint.py --load results/fingerprint.json --heatmap --bar-chart --top 20

Compare saved results:
    python scripts/fingerprint.py --load results/a.json results/b.json --heatmap
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from tv/ root: python scripts/fingerprint.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from core import (
    load_model, load_adapter, unload_adapter, generate, load_vectors,
    capture, project, compare, top_traits, top_spans,
)
from core.metrics import short_name, cosine_sim, spearman_corr


# ─── Response I/O ───────────────────────────────────────────────────────────

def save_responses(responses, path, model_name=None):
    """Save generated responses to JSON.

    Format: {prompt_id: {prompt, response, model, prompt_end}}
    prompt_end is the tokenized prompt length for sanity checking during capture.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {"model": model_name, "responses": responses}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(responses)} responses to {path}")


def load_responses(path):
    """Load responses from JSON. Returns list of {id, prompt, response, ...}."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "responses" in data:
        return data["responses"]
    return data


# ─── Scoring ────────────────────────────────────────────────────────────────

def score_variant(model, tok, vectors, prompts, responses=None,
                  max_new_tokens=200, do_sample=False):
    """Score a single model variant across prompts.

    If responses is None, generates fresh responses.
    Returns (fingerprint_dict, response_list, per_prompt_scores).
    """
    trait_totals = {t: 0.0 for t in vectors}
    all_responses = []
    per_prompt = []
    n = len(prompts)

    for i, prompt_data in enumerate(prompts):
        question = prompt_data["prompt"]
        prompt_id = prompt_data.get("id", str(i))

        # Generate or use provided response
        if responses and i < len(responses):
            response = responses[i]["response"]
        else:
            response = generate(model, tok, question,
                                max_new_tokens=max_new_tokens, do_sample=do_sample)

        all_responses.append({"id": prompt_id, "prompt": question, "response": response})

        # Capture + project
        data = capture(model, tok, question, response)
        scores = project(data, vectors)

        prompt_scores = {t: scores[t]["mean"] for t in scores}
        per_prompt.append({"id": prompt_id, "scores": prompt_scores, "scores_full": scores})
        for t in prompt_scores:
            trait_totals[t] += prompt_scores[t]

        print(f"  [{i+1}/{n}] {prompt_id}: {response[:60]}...")

    # Average across prompts
    fingerprint = {t: trait_totals[t] / n for t in trait_totals}
    return fingerprint, all_responses, per_prompt


# ─── Visualization helpers ──────────────────────────────────────────────────

def print_top(fingerprint_or_delta, k=20, label="Top traits"):
    """Print top-k traits by |value|."""
    items = sorted(fingerprint_or_delta.items(), key=lambda x: abs(x[1]), reverse=True)[:k]
    print(f"\n{label} (top {k}):")
    for trait, val in items:
        print(f"  {short_name(trait):30s} {val:+.4f}")


def print_spans(per_prompt_scores, trait, k=5):
    """Print max-activating spans for a trait across prompts."""
    print(f"\nMax-activating spans for {short_name(trait)}:")
    for entry in per_prompt_scores:
        scores_full = entry["scores_full"]
        spans = top_spans(scores_full, trait, k=k)
        if not spans:
            continue
        print(f"\n  [{entry['id']}] (mean={scores_full[trait]['mean']:+.4f})")
        for span in spans:
            print(f"    {span['mean_score']:+.4f}  {span['text'].strip()}")


def visualize_results(results, args):
    """Generate plots from results dict."""
    try:
        from plot import heatmap, bar_chart, similarity_matrix
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    out_dir = Path(args.output).parent if args.output else Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_names = list(results["variants"].keys())

    if args.bar_chart:
        # Bar chart of top traits for each variant (delta if multiple, raw if single)
        for name in variant_names:
            data = results["variants"][name]
            scores = data.get("delta") or data["fingerprint"]
            label = f"delta ({name} - baseline)" if "delta" in data else name
            items = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:args.top]
            bar_chart(
                [short_name(t) for t, _ in items],
                [v for _, v in items],
                title=label,
                save=str(out_dir / f"bar_{name}.png"),
            )

    if args.heatmap and len(variant_names) > 0:
        # Heatmap: variants × traits (use delta if available, else fingerprint)
        all_traits = set()
        for name in variant_names:
            data = results["variants"][name]
            scores = data.get("delta") or data["fingerprint"]
            all_traits.update(scores.keys())

        # Sort by mean |value| across variants
        trait_list = sorted(all_traits, key=lambda t: np.mean([
            abs((results["variants"][n].get("delta") or results["variants"][n]["fingerprint"]).get(t, 0))
            for n in variant_names
        ]), reverse=True)[:args.top]

        matrix = np.zeros((len(variant_names), len(trait_list)))
        for i, name in enumerate(variant_names):
            data = results["variants"][name]
            scores = data.get("delta") or data["fingerprint"]
            for j, trait in enumerate(trait_list):
                matrix[i, j] = scores.get(trait, 0)

        heatmap(
            matrix, variant_names, [short_name(t) for t in trait_list],
            title="Trait fingerprint" + (" (delta)" if any("delta" in results["variants"][n] for n in variant_names) else ""),
            save=str(out_dir / "heatmap.png"),
        )

    if args.similarity and len(variant_names) >= 2:
        # Similarity matrix between variant fingerprints
        fp_list = []
        for name in variant_names:
            data = results["variants"][name]
            scores = data.get("delta") or data["fingerprint"]
            fp_list.append(scores)

        all_traits = sorted(set().union(*[set(fp.keys()) for fp in fp_list]))
        vectors = [np.array([fp.get(t, 0) for t in all_traits]) for fp in fp_list]

        n = len(variant_names)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = cosine_sim(vectors[i], vectors[j])

        similarity_matrix(
            sim_matrix, variant_names,
            title="Fingerprint similarity",
            save=str(out_dir / "similarity.png"),
        )


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    parser.add_argument("--manifest", default="data/manifest.json",
                        help="Path to manifest JSON (default: data/manifest.json)")
    parser.add_argument("--prompt-set", default="questions_normal",
                        help="JSON file in data/prompts/ (without extension)")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of prompts (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=200)

    # Variants
    parser.add_argument("--variants", nargs="+", default=["clean"],
                        help="Variant names to score (default: clean)")
    parser.add_argument("--adapters", nargs="*", default=[],
                        help="Adapter mappings: name=path/to/adapter")

    # Response handling
    parser.add_argument("--responses", nargs="*",
                        help="Load responses from file(s) instead of generating")
    parser.add_argument("--save-responses", action="store_true",
                        help="Save generated responses to results/")

    # Analysis
    parser.add_argument("--top", type=int, default=20, help="Number of top traits to show")
    parser.add_argument("--spans", help="Show max-activating spans for this trait")

    # Visualization
    parser.add_argument("--heatmap", action="store_true", help="Generate heatmap")
    parser.add_argument("--bar-chart", action="store_true", help="Generate bar chart")
    parser.add_argument("--similarity", action="store_true", help="Generate similarity matrix")

    # I/O
    parser.add_argument("--output", default="results/fingerprint.json",
                        help="Save results JSON to this path")
    parser.add_argument("--load", nargs="+", help="Load and visualize saved results instead of scoring")

    args = parser.parse_args()

    # ── Load and visualize mode ──
    if args.load:
        all_results = {"variants": {}}
        for path in args.load:
            with open(path) as f:
                data = json.load(f)
            if "variants" in data:
                all_results["variants"].update(data["variants"])
            else:
                name = Path(path).stem
                all_results["variants"][name] = data

        for name, data in all_results["variants"].items():
            scores = data.get("delta") or data.get("fingerprint", {})
            print_top(scores, k=args.top, label=f"{name}")

        if args.heatmap or args.bar_chart or args.similarity:
            visualize_results(all_results, args)
        return

    # ── Score mode ──
    adapter_map = {}
    for spec in args.adapters:
        name, path = spec.split("=", 1)
        adapter_map[name] = path

    # Load model + vectors
    model, tok = load_model()
    vectors = load_vectors(args.manifest)

    # Load prompts
    with open(f"data/prompts/{args.prompt_set}.json") as f:
        prompts = json.load(f)
    if args.n:
        prompts = prompts[:args.n]
    print(f"Prompts: {len(prompts)} from {args.prompt_set}")

    # Load responses if provided
    loaded_responses = {}
    if args.responses:
        for i, path in enumerate(args.responses):
            variant = args.variants[i] if i < len(args.variants) else Path(path).stem
            loaded_responses[variant] = load_responses(path)
            print(f"Loaded {len(loaded_responses[variant])} responses for {variant}")

    # Score each variant
    results = {"variants": {}, "metadata": {
        "prompt_set": args.prompt_set, "n_prompts": len(prompts),
        "n_traits": len(vectors),
    }}
    baseline_fingerprint = None
    t_start = time.time()

    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"Scoring: {variant}")

        # Load adapter if needed
        if variant in adapter_map:
            model = load_adapter(model, adapter_map[variant], adapter_name=variant)

        # Score
        responses = loaded_responses.get(variant)
        fingerprint, gen_responses, per_prompt = score_variant(
            model, tok, vectors, prompts, responses=responses,
            max_new_tokens=args.max_new_tokens,
        )

        # Save responses if requested
        if args.save_responses:
            resp_dir = Path("results")
            resp_dir.mkdir(parents=True, exist_ok=True)
            save_responses(gen_responses, resp_dir / f"responses_{variant}.json")

        # Store results
        variant_data = {"fingerprint": fingerprint}

        # Compute delta against first variant (baseline)
        if baseline_fingerprint is None:
            baseline_fingerprint = fingerprint
        elif baseline_fingerprint is not fingerprint:
            delta = {t: fingerprint[t] - baseline_fingerprint[t] for t in fingerprint if t in baseline_fingerprint}
            variant_data["delta"] = delta

        results["variants"][variant] = variant_data

        # Print top traits
        if "delta" in variant_data:
            print_top(variant_data["delta"], k=args.top, label=f"Delta ({variant} - {args.variants[0]})")
        else:
            print_top(fingerprint, k=args.top, label=f"Fingerprint: {variant}")

        # Print spans if requested
        if args.spans:
            print_spans(per_prompt, args.spans)

        # Unload adapter
        if variant in adapter_map:
            model = unload_adapter(model, adapter_name=variant)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.0f}s")

    # Cross-variant similarity
    if len(args.variants) >= 2:
        print(f"\nCross-variant similarity:")
        variant_names = list(results["variants"].keys())
        for i, a in enumerate(variant_names):
            for b in variant_names[i+1:]:
                fp_a = results["variants"][a]["fingerprint"]
                fp_b = results["variants"][b]["fingerprint"]
                traits = sorted(set(fp_a) & set(fp_b))
                va = np.array([fp_a[t] for t in traits])
                vb = np.array([fp_b[t] for t in traits])
                cos = cosine_sim(va, vb)
                rho, _ = spearman_corr(va, vb)
                print(f"  {a} vs {b}: cosine={cos:.3f}, spearman={rho:.3f}")

    # Visualization
    if args.heatmap or args.bar_chart or args.similarity:
        visualize_results(results, args)

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
