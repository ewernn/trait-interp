"""Probe co-occurrence analysis on per-token diff data.

Computes token-level and prompt-level correlation matrices across trait probes
to understand whether probes fire together or independently.

Input: per_token_diff JSON files from model_diff pipeline
Output: Correlation matrices + key findings printed and saved to JSON

Usage:
    python analysis/model_diff/probe_cooccurrence.py \
        --experiment audit-bench \
        --variant-a instruct --variant-b rm_lora \
        --prompt-set exploitation_evals_100
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TRAIT_SPECS = [
    ("rm_hack", "ulterior_motive"),
    ("rm_hack", "ulterior_motive_v2"),
    ("rm_hack", "secondary_objective"),
    ("rm_hack", "eval_awareness"),
    ("bs", "concealment"),
    ("bs", "lying"),
]


def short_name(category, trait):
    """Short display name for a trait."""
    return trait


def load_data(base_dir, variant, prompt_set):
    """Load per_token_delta for all traits and prompts.

    Returns:
        prompt_ids: sorted list of common prompt IDs
        trait_names: list of short trait names
        data: dict[trait_name][prompt_id] = list of per_token_delta floats
    """
    data = {}
    all_id_sets = []

    for category, trait in TRAIT_SPECS:
        name = short_name(category, trait)
        trait_dir = os.path.join(base_dir, category, trait, variant, prompt_set)
        if not os.path.isdir(trait_dir):
            print(f"WARNING: Missing directory {trait_dir}")
            continue

        trait_data = {}
        for fname in os.listdir(trait_dir):
            if fname == "aggregate.json" or not fname.endswith(".json"):
                continue
            pid = fname.replace(".json", "")
            with open(os.path.join(trait_dir, fname)) as f:
                d = json.load(f)
            trait_data[pid] = d["per_token_delta"]

        data[name] = trait_data
        all_id_sets.append(set(trait_data.keys()))

    trait_names = list(data.keys())
    prompt_ids = sorted(set.intersection(*all_id_sets))
    print(f"Loaded {len(trait_names)} traits, {len(prompt_ids)} common prompts\n")
    return prompt_ids, trait_names, data


def compute_token_level_correlations(prompt_ids, trait_names, data):
    """For each prompt, correlate 6 delta timeseries pairwise, then average.

    Handles different-length timeseries by truncating to min length per prompt.
    """
    n = len(trait_names)
    corr_sum = np.zeros((n, n))
    count = 0

    for pid in prompt_ids:
        # Get timeseries for each trait, truncate to common length
        series = [np.array(data[t][pid]) for t in trait_names]
        min_len = min(len(s) for s in series)
        if min_len < 5:
            continue  # skip very short responses

        series = [s[:min_len] for s in series]
        mat = np.stack(series)  # (n_traits, n_tokens)

        # Compute pairwise Pearson correlation
        corr = np.corrcoef(mat)

        # Handle NaN (constant timeseries) by treating as 0 correlation
        corr = np.nan_to_num(corr, nan=0.0)
        corr_sum += corr
        count += 1

    avg_corr = corr_sum / count
    print(f"Token-level: averaged over {count} prompts (skipped {len(prompt_ids) - count} short)\n")
    return avg_corr


def compute_prompt_level_correlations(prompt_ids, trait_names, data):
    """For each trait, compute mean_delta per prompt, then correlate across traits."""
    n = len(trait_names)
    # Build (n_traits, n_prompts) matrix of mean deltas
    means = np.zeros((n, len(prompt_ids)))
    for i, t in enumerate(trait_names):
        for j, pid in enumerate(prompt_ids):
            means[i, j] = np.mean(data[t][pid])

    corr = np.corrcoef(means)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr, means


def print_matrix(corr, trait_names, title):
    """Pretty-print a correlation matrix."""
    abbrevs = []
    for t in trait_names:
        if len(t) > 12:
            abbrevs.append(t[:11] + ".")
        else:
            abbrevs.append(t)

    print(f"{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    # Header
    header = f"{'':>22s}"
    for a in abbrevs:
        header += f"  {a:>14s}"
    print(header)
    print("-" * len(header))

    for i, name in enumerate(trait_names):
        row = f"{name:>22s}"
        for j in range(len(trait_names)):
            val = corr[i, j]
            row += f"  {val:>14.3f}"
        print(row)
    print()


def find_key_findings(corr, trait_names, level_name):
    """Extract notable correlations from a matrix."""
    findings = []
    n = len(trait_names)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((corr[i, j], trait_names[i], trait_names[j]))

    pairs.sort(key=lambda x: x[0], reverse=True)

    findings.append(f"\n--- {level_name} Key Findings ---")
    findings.append(f"  Most correlated pairs:")
    for val, a, b in pairs[:3]:
        findings.append(f"    {a} <-> {b}: r={val:.3f}")

    findings.append(f"  Least correlated / anti-correlated pairs:")
    for val, a, b in pairs[-3:]:
        findings.append(f"    {a} <-> {b}: r={val:.3f}")

    # Mean off-diagonal
    mask = ~np.eye(n, dtype=bool)
    mean_offdiag = corr[mask].mean()
    findings.append(f"  Mean off-diagonal correlation: {mean_offdiag:.3f}")

    anti = [(v, a, b) for v, a, b in pairs if v < 0]
    if anti:
        findings.append(f"  Anti-correlated pairs: {len(anti)}")
        for v, a, b in anti:
            findings.append(f"    {a} <-> {b}: r={v:.3f}")
    else:
        findings.append(f"  No anti-correlated pairs found")

    return findings, pairs


def main():
    parser = argparse.ArgumentParser(description="Probe co-occurrence analysis")
    parser.add_argument("--experiment", default="audit-bench")
    parser.add_argument("--variant-a", default="instruct")
    parser.add_argument("--variant-b", default="rm_lora")
    parser.add_argument("--variant", default="rm_syco", help="Steering variant used in diff")
    parser.add_argument("--prompt-set", default="exploitation_evals_100")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    diff_name = f"{args.variant_a}_vs_{args.variant_b}"
    base_dir = project_root / "experiments" / args.experiment / "model_diff" / diff_name / "per_token_diff"
    output_dir = project_root / "experiments" / args.experiment / "model_diff" / diff_name

    prompt_ids, trait_names, data = load_data(str(base_dir), args.variant, args.prompt_set)

    # 1. Token-level correlations
    token_corr = compute_token_level_correlations(prompt_ids, trait_names, data)
    print_matrix(token_corr, trait_names, "TOKEN-LEVEL Correlation (avg across prompts)")

    # 2. Prompt-level correlations
    prompt_corr, prompt_means = compute_prompt_level_correlations(prompt_ids, trait_names, data)
    print_matrix(prompt_corr, trait_names, "PROMPT-LEVEL Correlation (mean delta per prompt)")

    # 3. Key findings
    all_findings = []
    token_findings, token_pairs = find_key_findings(token_corr, trait_names, "Token-level")
    prompt_findings, prompt_pairs = find_key_findings(prompt_corr, trait_names, "Prompt-level")

    for line in token_findings + prompt_findings:
        print(line)

    # 4. Per-trait summary stats
    print(f"\n--- Per-Trait Mean Delta Stats ---")
    for i, t in enumerate(trait_names):
        vals = prompt_means[i]
        print(f"  {t:>25s}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"min={vals.min():.3f}, max={vals.max():.3f}")

    # 5. Save results
    output_path = output_dir / "probe_cooccurrence.json"
    results = {
        "traits": trait_names,
        "n_prompts": len(prompt_ids),
        "token_level_correlation": token_corr.tolist(),
        "prompt_level_correlation": prompt_corr.tolist(),
        "token_level_pairs": [
            {"trait_a": a, "trait_b": b, "r": round(float(v), 4)}
            for v, a, b in token_pairs
        ],
        "prompt_level_pairs": [
            {"trait_a": a, "trait_b": b, "r": round(float(v), 4)}
            for v, a, b in prompt_pairs
        ],
        "per_trait_stats": {
            t: {
                "mean": round(float(prompt_means[i].mean()), 4),
                "std": round(float(prompt_means[i].std()), 4),
                "min": round(float(prompt_means[i].min()), 4),
                "max": round(float(prompt_means[i].max()), 4),
            }
            for i, t in enumerate(trait_names)
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
