"""Plot FT fingerprint evolution over training steps.

Shows how the trait fingerprint develops during fine-tuning for all 4 LoRA variants.

Input: ft_trajectory.json from ft_trajectory.py
Output: trajectory_*.png figures

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/plot_trajectory.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from utils.fingerprints import short_name

DIR = Path(__file__).parent


def load_trajectory():
    with open(DIR / "ft_trajectory.json") as f:
        return json.load(f)


def load_hf_ft():
    """Load HF LoRA fingerprints for comparison."""
    path = DIR / "ft_fingerprints.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {k: v["mean_fingerprint"] for k, v in data["variants"].items()}


def load_icl():
    """Load ICL medical residual for comparison."""
    path = DIR / "icl_medical_residual.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    results = data["results"]
    traits = sorted(results[0]["trait_scores"].keys())
    return {t: np.mean([r["trait_scores"][t] for r in results]) for t in traits}


def plot_fingerprint_evolution(data):
    """Plot Spearman correlation with final-step fingerprint over training."""
    runs = data["runs"]
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"medical": "firebrick", "good_medical": "forestgreen",
              "financial": "steelblue", "insecure": "darkorange"}

    for run_name, run_data in runs.items():
        steps_list = run_data["steps"]
        final_fp = steps_list[-1]["fingerprint"]
        traits = sorted(final_fp.keys())

        steps = []
        rhos = []
        for entry in steps_list:
            fp = entry["fingerprint"]
            x = [fp[t] for t in traits]
            y = [final_fp[t] for t in traits]
            r, _ = spearmanr(x, y)
            steps.append(entry["step"])
            rhos.append(r)

        ax.plot(steps, rhos, "o-", label=run_name, color=colors.get(run_name, "gray"),
                markersize=4, linewidth=1.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Spearman ρ with step-100 fingerprint")
    ax.set_title("Fingerprint convergence during fine-tuning", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    out = DIR / "trajectory_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_top_traits_over_time(data):
    """Plot top 8 traits evolving over training steps for each run."""
    runs = data["runs"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (run_name, run_data) in zip(axes.flat, runs.items()):
        steps_list = run_data["steps"]
        final_fp = steps_list[-1]["fingerprint"]
        top_traits = sorted(final_fp.keys(), key=lambda t: abs(final_fp[t]), reverse=True)[:8]

        for trait in top_traits:
            steps = [e["step"] for e in steps_list]
            scores = [e["fingerprint"][trait] for e in steps_list]
            ax.plot(steps, scores, "o-", label=short_name(trait), markersize=3, linewidth=1.2)

        ax.set_xlabel("Training step")
        ax.set_ylabel("Cosine sim with trait vector")
        ax.set_title(run_name, fontweight="bold")
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    fig.suptitle("Top trait evolution during fine-tuning", fontweight="bold", fontsize=13)
    fig.tight_layout()

    out = DIR / "trajectory_top_traits.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_cross_run_correlation(data):
    """Compare step-100 fingerprints across all 4 runs + ICL."""
    runs = data["runs"]
    run_names = list(runs.keys())
    final_fps = {name: runs[name]["steps"][-1]["fingerprint"] for name in run_names}
    traits = sorted(final_fps[run_names[0]].keys())

    # Add HF LoRA fingerprints if available
    hf_fps = load_hf_ft()
    icl_fp = load_icl()

    all_fps = {}
    all_labels = []
    for name in run_names:
        label = f"{name} (100-step)"
        all_fps[label] = final_fps[name]
        all_labels.append(label)

    if hf_fps:
        for name, fp in hf_fps.items():
            label = f"{name} (HF)"
            all_fps[label] = fp
            all_labels.append(label)

    if icl_fp:
        all_fps["ICL med_residual"] = icl_fp
        all_labels.append("ICL med_residual")

    n = len(all_labels)
    rho_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = [all_fps[all_labels[i]].get(t, 0) for t in traits]
            y = [all_fps[all_labels[j]].get(t, 0) for t in traits]
            r, _ = spearmanr(x, y)
            rho_mat[i, j] = r

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rho_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.set_title("Cross-run fingerprint correlations (step 100 vs HF vs ICL)", fontweight="bold")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = rho_mat[i, j]
            color = "white" if abs(r) > 0.55 else "black"
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    fig.tight_layout()

    out = DIR / "trajectory_cross_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()

    return rho_mat, all_labels


def plot_icl_correlation_over_time(data):
    """Plot Spearman correlation with ICL residual over training steps."""
    icl_fp = load_icl()
    if icl_fp is None:
        print("No ICL medical residual found — skipping")
        return

    runs = data["runs"]
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"medical": "firebrick", "good_medical": "forestgreen",
              "financial": "steelblue", "insecure": "darkorange"}

    for run_name, run_data in runs.items():
        steps_list = run_data["steps"]
        traits = sorted(steps_list[0]["fingerprint"].keys())
        common = [t for t in traits if t in icl_fp]

        steps = []
        rhos = []
        for entry in steps_list:
            fp = entry["fingerprint"]
            x = [fp[t] for t in common]
            y = [icl_fp[t] for t in common]
            r, _ = spearmanr(x, y)
            steps.append(entry["step"])
            rhos.append(r)

        ax.plot(steps, rhos, "o-", label=run_name, color=colors.get(run_name, "gray"),
                markersize=4, linewidth=1.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Spearman ρ with ICL medical residual")
    ax.set_title("FT × ICL anti-correlation over training", fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    out = DIR / "trajectory_icl_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_loss_curves(data):
    """Plot training loss for all 4 runs."""
    runs = data["runs"]
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"medical": "firebrick", "good_medical": "forestgreen",
              "financial": "steelblue", "insecure": "darkorange"}

    for run_name, run_data in runs.items():
        steps_list = run_data["steps"]
        steps = [e["step"] for e in steps_list if e.get("loss") is not None]
        losses = [e["loss"] for e in steps_list if e.get("loss") is not None]
        if steps:
            ax.plot(steps, losses, "o-", label=run_name, color=colors.get(run_name, "gray"),
                    markersize=4, linewidth=1.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss curves", fontweight="bold")
    ax.legend(fontsize=9)

    out = DIR / "trajectory_loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def compare_good_vs_bad_medical(data):
    """Detailed comparison of good_medical vs bad_medical at step 100."""
    runs = data["runs"]
    bad_fp = runs["medical"]["steps"][-1]["fingerprint"]
    good_fp = runs["good_medical"]["steps"][-1]["fingerprint"]
    traits = sorted(bad_fp.keys())

    x = [bad_fp[t] for t in traits]
    y = [good_fp[t] for t in traits]
    rho, p = spearmanr(x, y)

    print(f"\n{'='*60}")
    print("GOOD vs BAD MEDICAL at step 100")
    print(f"{'='*60}")
    print(f"Spearman correlation: rho={rho:.3f} (p={p:.2e})")

    # Traits that differ most
    diffs = [(t, bad_fp[t] - good_fp[t], bad_fp[t], good_fp[t]) for t in traits]
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\nTop 15 differences (bad - good):")
    for t, d, bad, good in diffs[:15]:
        print(f"  {short_name(t):25s}: diff={d:+.4f} (bad={bad:+.4f}, good={good:+.4f})")

    # Sign agreement
    n_nonzero = sum(1 for t in traits if bad_fp[t] != 0 and good_fp[t] != 0)
    n_agree = sum(1 for t in traits
                  if bad_fp[t] != 0 and good_fp[t] != 0 and (bad_fp[t] > 0) == (good_fp[t] > 0))
    print(f"\nSign agreement: {n_agree}/{n_nonzero} ({n_agree/n_nonzero:.1%})")

    # Compare both with ICL
    icl_fp = load_icl()
    if icl_fp:
        common = [t for t in traits if t in icl_fp]
        x_bad = [bad_fp[t] for t in common]
        x_good = [good_fp[t] for t in common]
        y_icl = [icl_fp[t] for t in common]
        rho_bad, p_bad = spearmanr(x_bad, y_icl)
        rho_good, p_good = spearmanr(x_good, y_icl)
        print(f"\nCorrelation with ICL medical residual:")
        print(f"  bad_medical × ICL: rho={rho_bad:.3f} (p={p_bad:.2e})")
        print(f"  good_medical × ICL: rho={rho_good:.3f} (p={p_good:.2e})")

    return rho, p


def main():
    print("Loading trajectory data...")
    data = load_trajectory()
    runs = data["runs"]
    print(f"Runs: {list(runs.keys())}")
    for name, rd in runs.items():
        print(f"  {name}: {len(rd['steps'])} steps")

    plot_fingerprint_evolution(data)
    plot_top_traits_over_time(data)
    rho_mat, labels = plot_cross_run_correlation(data)
    plot_icl_correlation_over_time(data)
    plot_loss_curves(data)
    compare_good_vs_bad_medical(data)

    print("\nDone.")


if __name__ == "__main__":
    main()
