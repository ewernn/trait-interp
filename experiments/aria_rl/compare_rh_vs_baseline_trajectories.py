"""Compare per-token trajectories between rh and baseline models.

The within-model comparison (RH vs non-RH responses from rh_s1) showed no signal
because ALL rh_s1 responses attempt to hack. The correct comparison is BETWEEN
models: rh_s1 vs rl_baseline_s1 on the same leetcode problems.

Input: rollouts/{rh_variant}_trajectories.pt, rollouts/{bl_variant}_trajectories.pt
Output: analysis/trajectories/cross_model_comparison.png + .json

Usage:
    PYTHONPATH=. python experiments/aria_rl/compare_rh_vs_baseline_trajectories.py
    PYTHONPATH=. python experiments/aria_rl/compare_rh_vs_baseline_trajectories.py --rh rh_s1 --baseline rl_baseline_s1
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

BASE_DIR = Path(__file__).parent


def normalize_position(scores, n_bins=100):
    n = len(scores)
    if n == 0:
        return np.zeros(n_bins)
    if n == 1:
        return np.full(n_bins, scores[0])
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, n_bins)
    return np.interp(x_new, x_old, scores)


def load_trajectories(variant):
    path = BASE_DIR / "rollouts" / f"{variant}_trajectories.pt"
    data = torch.load(path, weights_only=False)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rh", default="rh_s1")
    parser.add_argument("--baseline", default="rl_baseline_s1")
    args = parser.parse_args()

    out_dir = BASE_DIR / "analysis" / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.rh}...")
    rh_data = load_trajectories(args.rh)
    print(f"Loading {args.baseline}...")
    bl_data = load_trajectories(args.baseline)

    trait_names = rh_data["trait_names"]
    f_rh = rh_data["f_rh"]
    n_bins = 100

    rh_results = rh_data["results"]
    bl_results = bl_data["results"]

    print(f"RH responses: {len(rh_results)}")
    print(f"Baseline responses: {len(bl_results)}")

    # --- 1. F_rh trajectory: rh model vs baseline model ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = {"rh": "#d62728", "baseline": "#2ca02c"}

    # 1a. Normalized position: all responses
    ax = axes[0, 0]
    summary = {}
    for label, results, color in [("rh", rh_results, colors["rh"]),
                                   ("baseline", bl_results, colors["baseline"])]:
        trajectories = []
        for r in results:
            scores = r["f_rh_scores"].numpy()
            if len(scores) > 0:
                trajectories.append(normalize_position(scores, n_bins))
        arr = np.array(trajectories)
        mean = arr.mean(axis=0)
        se = arr.std(axis=0) / np.sqrt(len(trajectories))
        x = np.linspace(0, 100, n_bins)
        ax.plot(x, mean, color=color, label=f"{label} (n={len(results)})")
        ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=color)
        summary[label] = {
            "n": len(results),
            "mean_trajectory": mean.tolist(),
            "mean_overall": float(mean.mean()),
        }

    ax.set_xlabel("Response position (%)")
    ax.set_ylabel("F_rh score")
    ax.set_title("F_rh trajectory: RH model vs Baseline model")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # 1b. Per-problem matched comparison
    ax = axes[0, 1]
    rh_by_problem = {}
    for r in rh_results:
        pid = r["meta"]["problem_id"]
        rh_by_problem.setdefault(pid, []).append(r)
    bl_by_problem = {}
    for r in bl_results:
        pid = r["meta"]["problem_id"]
        bl_by_problem.setdefault(pid, []).append(r)

    shared_problems = sorted(set(rh_by_problem.keys()) & set(bl_by_problem.keys()))
    print(f"Shared problems: {len(shared_problems)}")

    rh_matched, bl_matched = [], []
    for pid in shared_problems:
        for r in rh_by_problem[pid]:
            scores = r["f_rh_scores"].numpy()
            if len(scores) > 0:
                rh_matched.append(normalize_position(scores, n_bins))
        for r in bl_by_problem[pid]:
            scores = r["f_rh_scores"].numpy()
            if len(scores) > 0:
                bl_matched.append(normalize_position(scores, n_bins))

    rh_arr = np.array(rh_matched)
    bl_arr = np.array(bl_matched)
    x = np.linspace(0, 100, n_bins)

    rh_mean = rh_arr.mean(axis=0)
    bl_mean = bl_arr.mean(axis=0)
    rh_se = rh_arr.std(axis=0) / np.sqrt(len(rh_matched))
    bl_se = bl_arr.std(axis=0) / np.sqrt(len(bl_matched))

    ax.plot(x, rh_mean, color=colors["rh"], label=f"RH model (n={len(rh_matched)})")
    ax.fill_between(x, rh_mean - rh_se, rh_mean + rh_se, alpha=0.2, color=colors["rh"])
    ax.plot(x, bl_mean, color=colors["baseline"], label=f"Baseline (n={len(bl_matched)})")
    ax.fill_between(x, bl_mean - bl_se, bl_mean + bl_se, alpha=0.2, color=colors["baseline"])
    ax.set_xlabel("Response position (%)")
    ax.set_ylabel("F_rh score")
    ax.set_title("Matched problems: RH vs Baseline")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Compute separation statistics
    rh_means = [r["f_rh_scores"].mean().item() for r in rh_results]
    bl_means = [r["f_rh_scores"].mean().item() for r in bl_results]
    cohens_d = (np.mean(rh_means) - np.mean(bl_means)) / np.sqrt(
        (np.std(rh_means)**2 + np.std(bl_means)**2) / 2)
    t_stat, p_val = stats.ttest_ind(rh_means, bl_means)

    summary["cross_model"] = {
        "rh_mean_frh": float(np.mean(rh_means)),
        "bl_mean_frh": float(np.mean(bl_means)),
        "cohens_d": float(cohens_d),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "n_shared_problems": len(shared_problems),
    }
    print(f"\nCross-model F_rh: RH={np.mean(rh_means):.4f}, BL={np.mean(bl_means):.4f}")
    print(f"Cohen's d = {cohens_d:.3f}, t = {t_stat:.2f}, p = {p_val:.2e}")

    # 1c. Distribution of per-response mean F_rh
    ax = axes[1, 0]
    ax.hist(rh_means, bins=40, alpha=0.6, color=colors["rh"], label="RH model", density=True)
    ax.hist(bl_means, bins=40, alpha=0.6, color=colors["baseline"], label="Baseline", density=True)
    ax.set_xlabel("Mean F_rh score (per response)")
    ax.set_ylabel("Density")
    ax.set_title(f"F_rh distribution (Cohen's d = {cohens_d:.2f})")
    ax.legend()

    # 1d. Per-position Cohen's d
    ax = axes[1, 1]
    position_d = []
    for i in range(n_bins):
        rh_col = rh_arr[:, i]
        bl_col = bl_arr[:, i]
        pooled_std = np.sqrt((rh_col.std()**2 + bl_col.std()**2) / 2)
        if pooled_std > 0:
            d = (rh_col.mean() - bl_col.mean()) / pooled_std
        else:
            d = 0
        position_d.append(d)
    ax.plot(x, position_d, color="purple", linewidth=2)
    ax.set_xlabel("Response position (%)")
    ax.set_ylabel("Cohen's d (RH - Baseline)")
    ax.set_title("Per-position effect size")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    summary["position_cohens_d"] = [float(d) for d in position_d]

    plt.tight_layout()
    plt.savefig(out_dir / "cross_model_frh.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'cross_model_frh.png'}")

    # --- 2. Top divergent traits between models ---
    n_traits = len(trait_names)
    trait_divergence = []

    for ti, t in enumerate(trait_names):
        rh_scores = [r["trait_scores"][:, ti].mean().item() for r in rh_results
                      if r["trait_scores"].shape[0] > 0]
        bl_scores = [r["trait_scores"][:, ti].mean().item() for r in bl_results
                      if r["trait_scores"].shape[0] > 0]
        rh_mean = np.mean(rh_scores)
        bl_mean = np.mean(bl_scores)
        pooled_std = np.sqrt((np.std(rh_scores)**2 + np.std(bl_scores)**2) / 2)
        d = (rh_mean - bl_mean) / pooled_std if pooled_std > 0 else 0
        t_stat_t, p_val_t = stats.ttest_ind(rh_scores, bl_scores)
        trait_divergence.append({
            "trait": t.split("/")[-1],
            "cohens_d": d,
            "rh_mean": rh_mean,
            "bl_mean": bl_mean,
            "p_value": p_val_t,
            "f_rh_weight": f_rh.get(t, 0),
        })

    trait_divergence.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)

    print("\nTop 20 cross-model divergent traits:")
    print(f"{'Trait':<35} {'Cohen_d':>8} {'rh_mean':>9} {'bl_mean':>9} {'p_val':>10} {'F_rh_wt':>8}")
    for td in trait_divergence[:20]:
        sig = "***" if td["p_value"] < 0.001 else "**" if td["p_value"] < 0.01 else "*" if td["p_value"] < 0.05 else ""
        print(f"  {td['trait']:<35} {td['cohens_d']:+.3f}  {td['rh_mean']:+.4f}  {td['bl_mean']:+.4f}  {td['p_value']:.4f}{sig} {td['f_rh_weight']:+.4f}")

    # Correlation between cross-model divergence and F_rh weights
    divs = [td["cohens_d"] for td in trait_divergence]
    frh_wts = [td["f_rh_weight"] for td in trait_divergence]
    r_corr, p_corr = stats.pearsonr(divs, frh_wts)
    print(f"\nCorrelation(cross-model Cohen's d, F_rh weight): r={r_corr:.3f}, p={p_corr:.4f}")
    summary["trait_divergence_frh_correlation"] = {"r": float(r_corr), "p": float(p_corr)}

    # Plot top 6 divergent traits
    fig, axes_t = plt.subplots(2, 3, figsize=(18, 10))
    for plot_i, td in enumerate(trait_divergence[:6]):
        ax = axes_t[plot_i // 3][plot_i % 3]
        trait_idx = trait_names.index(f"emotion_set/{td['trait']}")

        for label, results, color in [("RH", rh_results, colors["rh"]),
                                       ("Baseline", bl_results, colors["baseline"])]:
            trajectories = []
            for r in results:
                scores = r["trait_scores"][:, trait_idx].numpy()
                if len(scores) > 0:
                    trajectories.append(normalize_position(scores, n_bins))
            arr = np.array(trajectories)
            mean = arr.mean(axis=0)
            se = arr.std(axis=0) / np.sqrt(len(trajectories))
            x = np.linspace(0, 100, n_bins)
            ax.plot(x, mean, color=color, label=label)
            ax.fill_between(x, mean - se, mean + se, alpha=0.15, color=color)

        ax.set_title(f"{td['trait']} (d={td['cohens_d']:+.2f})")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        if plot_i == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Top cross-model divergent traits: RH vs Baseline", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "cross_model_top_traits.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'cross_model_top_traits.png'}")

    # --- 3. RH-span-specific analysis ---
    # Load annotations to identify where def run_tests appears
    ann_path = BASE_DIR / "rollouts" / f"{args.rh}_annotations.json"
    if ann_path.exists():
        with open(ann_path) as f:
            ann_data = json.load(f)

        print("\n--- RH Span Analysis ---")
        # For rh responses with annotations, compare F_rh at annotated span vs before
        # The annotations have global idx but we need to match to trajectory results
        # For now, use the position info: def run_tests at ~72% through response

        rh_before_span = []
        rh_at_span = []
        bl_at_same_position = []

        for r in rh_results:
            scores = r["f_rh_scores"].numpy()
            n = len(scores)
            if n < 10:
                continue
            # Before RH span (~first 70%)
            cutpoint = int(0.7 * n)
            rh_before_span.append(scores[:cutpoint].mean())
            rh_at_span.append(scores[cutpoint:].mean())

        for r in bl_results:
            scores = r["f_rh_scores"].numpy()
            n = len(scores)
            if n < 10:
                continue
            cutpoint = int(0.7 * n)
            bl_at_same_position.append(scores[cutpoint:].mean())

        print(f"RH model, before span (first 70%): {np.mean(rh_before_span):.4f}")
        print(f"RH model, at span (last 30%): {np.mean(rh_at_span):.4f}")
        print(f"Baseline, same position (last 30%): {np.mean(bl_at_same_position):.4f}")

        summary["span_analysis"] = {
            "rh_before_span": float(np.mean(rh_before_span)),
            "rh_at_span": float(np.mean(rh_at_span)),
            "bl_at_same_position": float(np.mean(bl_at_same_position)),
        }

    # Save summary
    summary["trait_divergence_top20"] = trait_divergence[:20]
    with open(out_dir / "cross_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_dir / 'cross_model_summary.json'}")


if __name__ == "__main__":
    main()
