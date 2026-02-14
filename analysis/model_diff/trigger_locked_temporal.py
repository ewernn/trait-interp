#!/usr/bin/env python3
"""
Trigger-locked temporal averaging of per-token probe deltas.

Aligns all responses at first hack span onset (t=0), averages probe deltas
across responses, and plots ERP-style curves showing whether probes spike
before, at, or after hack text begins.

Input: Per-token diff JSONs + annotation file with hack spans
Output: Matplotlib figure + JSON data

Usage:
    python analysis/model_diff/trigger_locked_temporal.py \
        --experiment audit-bench \
        --prompt-set rm_syco/exploitation_evals_100
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.annotations import spans_to_char_ranges
from utils.paths import get as get_path


def char_pos_to_token_idx(tokens: list[str], char_pos: int) -> int:
    """Map character position in response_text to token index."""
    cumulative = 0
    for i, tok in enumerate(tokens):
        if cumulative + len(tok) > char_pos:
            return i
        cumulative += len(tok)
    return len(tokens) - 1


def find_hack_onset_token(diff_data: dict, annotation: dict) -> int | None:
    """Find token index of earliest hack span in a response.

    Returns None if no spans found or span text can't be located.
    """
    spans = annotation.get("spans", [])
    if not spans:
        return None

    response_text = diff_data["response_text"]
    tokens = diff_data["tokens"]

    # Get character ranges for all spans
    char_ranges = spans_to_char_ranges(response_text, spans)
    if not char_ranges:
        return None

    # Find earliest span by character position
    earliest_char = min(start for start, _ in char_ranges)

    return char_pos_to_token_idx(tokens, earliest_char)


def load_annotations(ann_path: Path) -> dict[str, dict]:
    """Load annotations keyed by prompt ID."""
    with open(ann_path) as f:
        data = json.load(f)

    lookup = {}
    for item in data:
        pid = item.get("idx") or item.get("id")
        if pid:
            lookup[pid] = item
    return lookup


def compute_trigger_locked_average(
    diff_dir: Path,
    annotations: dict[str, dict],
    window_before: int = 40,
    window_after: int = 80,
) -> dict:
    """Compute trigger-locked average of per-token deltas.

    Returns dict with relative positions as keys, each containing
    lists of deltas from individual prompts for averaging.
    """
    window_size = window_before + window_after
    # Accumulate deltas at each relative position
    position_deltas = defaultdict(list)
    n_aligned = 0
    onset_positions = []

    for json_file in sorted(diff_dir.glob("*.json")):
        if json_file.name == "aggregate.json":
            continue

        pid = json_file.stem
        if pid not in annotations:
            continue

        with open(json_file) as f:
            diff_data = json.load(f)

        onset = find_hack_onset_token(diff_data, annotations[pid])
        if onset is None:
            continue

        deltas = diff_data["per_token_delta"]
        onset_positions.append(onset)
        n_aligned += 1

        # Re-index relative to hack onset
        for i, delta in enumerate(deltas):
            rel_pos = i - onset
            if -window_before <= rel_pos < window_after:
                position_deltas[rel_pos].append(delta)

    # Compute mean and SEM at each position
    positions = sorted(position_deltas.keys())
    means = []
    sems = []
    counts = []
    for pos in positions:
        vals = position_deltas[pos]
        means.append(float(np.mean(vals)))
        sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        counts.append(len(vals))

    return {
        "positions": positions,
        "means": means,
        "sems": sems,
        "counts": counts,
        "n_aligned": n_aligned,
        "onset_positions": onset_positions,
    }


def main():
    parser = argparse.ArgumentParser(description="Trigger-locked temporal averaging")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--window-before", type=int, default=40)
    parser.add_argument("--window-after", type=int, default=80)
    parser.add_argument("--traits", type=str, default=None,
                        help="Comma-separated trait names (default: all available)")
    args = parser.parse_args()

    exp_dir = get_path("experiments.base", experiment=args.experiment)
    model_diff_dir = exp_dir / "model_diff"

    # Find the comparison directory (e.g., instruct_vs_rm_lora)
    comparison_dirs = [d for d in model_diff_dir.iterdir()
                       if d.is_dir() and "per_token_diff" in [x.name for x in d.iterdir() if x.is_dir()]]
    if not comparison_dirs:
        print("No per_token_diff data found in model_diff/")
        sys.exit(1)
    comparison_dir = comparison_dirs[0]
    comparison_name = comparison_dir.name

    per_token_dir = comparison_dir / "per_token_diff"

    # Find annotation file
    # Convention: {prompt_set}_annotations.json in responses dir
    prompt_set_slug = args.prompt_set.replace("/", "_")
    ann_candidates = [
        exp_dir / "inference" / "rm_lora" / "responses" / args.prompt_set.rsplit("/", 1)[0] / f"{prompt_set_slug}_annotations.json",
        exp_dir / "inference" / "rm_lora" / "responses" / (args.prompt_set.replace("/", "_") + "_annotations.json"),
    ]
    # Also check the split path
    if "/" in args.prompt_set:
        parts = args.prompt_set.split("/")
        ann_candidates.append(
            exp_dir / "inference" / "rm_lora" / "responses" / parts[0] / f"{parts[1]}_annotations.json"
        )

    ann_path = None
    for candidate in ann_candidates:
        if candidate.exists():
            ann_path = candidate
            break

    if ann_path is None:
        print(f"Annotation file not found. Tried: {[str(c) for c in ann_candidates]}")
        sys.exit(1)

    print(f"Using annotations: {ann_path}")
    annotations = load_annotations(ann_path)
    print(f"Loaded {len(annotations)} annotations")

    # Discover available traits
    available_traits = []
    for trait_cat_dir in sorted(per_token_dir.iterdir()):
        if not trait_cat_dir.is_dir():
            continue
        for trait_dir in sorted(trait_cat_dir.iterdir()):
            if not trait_dir.is_dir():
                continue
            trait_key = f"{trait_cat_dir.name}/{trait_dir.name}"
            available_traits.append(trait_key)

    if args.traits:
        selected = [t.strip() for t in args.traits.split(",")]
        # Allow short names (e.g., "secondary_objective" -> "rm_hack/secondary_objective")
        traits = []
        for s in selected:
            matches = [t for t in available_traits if t.endswith(s)]
            traits.extend(matches if matches else [s])
    else:
        traits = available_traits

    print(f"Traits: {traits}")

    # Compute trigger-locked averages for each trait
    all_results = {}
    for trait in traits:
        diff_dir = per_token_dir / trait / args.prompt_set
        if not diff_dir.exists():
            print(f"  Skipping {trait}: no data at {diff_dir}")
            continue

        result = compute_trigger_locked_average(
            diff_dir, annotations,
            window_before=args.window_before,
            window_after=args.window_after,
        )
        all_results[trait] = result
        trait_short = trait.split("/")[-1]
        print(f"  {trait_short}: {result['n_aligned']} responses aligned, "
              f"median onset at token {int(np.median(result['onset_positions']))}")

    if not all_results:
        print("No results to plot.")
        sys.exit(1)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    for i, (trait, result) in enumerate(all_results.items()):
        positions = np.array(result["positions"])
        means = np.array(result["means"])
        sems = np.array(result["sems"])
        trait_short = trait.split("/")[-1]
        color = colors[i % len(colors)]

        ax.plot(positions, means, label=trait_short, color=color, linewidth=1.5)
        ax.fill_between(positions, means - sems, means + sems, alpha=0.15, color=color)

    # Hack onset line
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.7, label="hack onset")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    ax.set_xlabel("Tokens relative to first hack span onset")
    ax.set_ylabel("Mean trait delta (organism − instruct)")
    ax.set_title(f"Trigger-Locked Probe Deltas — {args.experiment} / {args.prompt_set}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    # Save
    out_dir = comparison_dir
    fig_path = out_dir / "trigger_locked_temporal.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")

    # Save JSON
    json_out = {
        "experiment": args.experiment,
        "prompt_set": args.prompt_set,
        "comparison": comparison_name,
        "window": {"before": args.window_before, "after": args.window_after},
        "traits": {},
    }
    for trait, result in all_results.items():
        json_out["traits"][trait] = {
            "positions": result["positions"],
            "means": result["means"],
            "sems": result["sems"],
            "counts": result["counts"],
            "n_aligned": result["n_aligned"],
            "median_onset_token": int(np.median(result["onset_positions"])),
        }

    json_path = out_dir / "trigger_locked_temporal.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Saved data: {json_path}")

    plt.close()


if __name__ == "__main__":
    main()
