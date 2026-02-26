"""Compute d-prime for trait vector projections across thought_branches conditions.

For each trait, computes the within-problem difference between conditions
(e.g., B - A), then reports mean, std, and d-prime across problems.
Traits with high d-prime pick up consistent signal despite noisy baselines.
Traits with low d-prime are dominated by scenario-dependent offsets.

Input:  inference/instruct/projections/{trait}/thought_branches/mmlu_condition_{a,b}/{id}.json
Output: prints table to stdout

Usage:
    python experiments/mats-mental-state-circuits/scripts/projection_dprime.py
    python experiments/mats-mental-state-circuits/scripts/projection_dprime.py --conditions b c
"""

import json
import argparse
import numpy as np
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
PROJ_BASE = EXPERIMENT_DIR / "inference" / "instruct" / "projections"


def load_mean_response_projection(path: Path) -> float:
    """Load a projection file and return mean response projection."""
    with open(path) as f:
        data = json.load(f)
    # Take first projection entry (best vector)
    proj = data["projections"][0]
    return float(np.mean(proj["response"]))


def discover_traits() -> list[str]:
    """Find all traits with thought_branches projections."""
    traits = []
    for p in sorted(PROJ_BASE.glob("*/*/thought_branches")):
        trait = str(p.relative_to(PROJ_BASE)).replace("/thought_branches", "")
        traits.append(trait)
    return traits


def compute_dprime(
    traits: list[str],
    cond_a: str,
    cond_b: str,
) -> list[dict]:
    """Compute d-prime for each trait across conditions."""
    results = []

    for trait in traits:
        dir_a = PROJ_BASE / trait / "thought_branches" / f"mmlu_condition_{cond_a}"
        dir_b = PROJ_BASE / trait / "thought_branches" / f"mmlu_condition_{cond_b}"

        if not dir_a.exists() or not dir_b.exists():
            continue

        # Find shared problem IDs
        ids_a = {p.stem for p in dir_a.glob("*.json")}
        ids_b = {p.stem for p in dir_b.glob("*.json")}
        shared = sorted(ids_a & ids_b, key=int)

        if not shared:
            continue

        deltas = []
        for pid in shared:
            mean_a = load_mean_response_projection(dir_a / f"{pid}.json")
            mean_b = load_mean_response_projection(dir_b / f"{pid}.json")
            deltas.append(mean_b - mean_a)

        deltas = np.array(deltas)
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        dprime = mean_delta / std_delta if std_delta > 0 else float("inf")

        results.append({
            "trait": trait,
            "n": len(shared),
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "dprime": dprime,
        })

    results.sort(key=lambda r: abs(r["dprime"]), reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute d-prime for trait projections")
    parser.add_argument(
        "--conditions", nargs=2, default=["a", "b"],
        help="Two conditions to compare (default: a b)",
    )
    args = parser.parse_args()

    cond_a, cond_b = args.conditions
    traits = discover_traits()
    results = compute_dprime(traits, cond_a, cond_b)

    print(f"\n{'Trait':<35} {'N':>3}  {'mean(Δ)':>9}  {'std(Δ)':>9}  {'d-prime':>8}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['trait']:<35} {r['n']:>3}  "
            f"{r['mean_delta']:>9.3f}  {r['std_delta']:>9.3f}  "
            f"{r['dprime']:>8.3f}"
        )

    # Summary
    relevant = [r for r in results if r["trait"] in (
        "alignment/deception", "bs/concealment", "rm_hack/eval_awareness",
    )]
    controls = [r for r in results if r["trait"].startswith("mental_state/")]

    if relevant and controls:
        print(f"\n--- Specificity check ({cond_b} vs {cond_a}) ---")
        print(f"  Target traits  |d'|:  {np.mean([abs(r['dprime']) for r in relevant]):.3f}")
        print(f"  Control traits |d'|:  {np.mean([abs(r['dprime']) for r in controls]):.3f}")
        ratio = np.mean([abs(r["dprime"]) for r in relevant]) / np.mean([abs(r["dprime"]) for r in controls])
        print(f"  Ratio:                {ratio:.2f}x")


if __name__ == "__main__":
    main()
