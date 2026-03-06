"""Compare trait dynamics at the decision point: RH vs baseline.

For RH responses, the landmark is the onset of `def run_tests()` (hack code).
For baseline responses, the landmark is the closing ``` after the Solution class
(end of code block, start of explanation). Both follow the last `return` in the
Solution class -- the models diverge at this structural point.

Input:
    - rollouts/{rh,rl_baseline}_s1.json (response text)
    - rollouts/{rh,rl_baseline}_s1_trajectories.pt (trait projections)
    - rollouts/rh_s1_annotations.json (hack span positions)

Output:
    - analysis/landmark_comparison_trajectories.png (top trait trajectories, RH vs BL)
    - analysis/landmark_comparison_heatmap.png (all traits, RH vs BL side-by-side)
    - analysis/landmark_comparison_delta_heatmap.png (RH - BL difference)

Usage:
    PYTHONPATH=/home/dev/trait-interp python experiments/aria_rl/analysis/landmark_aligned_comparison.py
"""

import json
import re
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

BASE = Path("/home/dev/trait-interp/experiments/aria_rl")
OUT = BASE / "analysis"
OUT.mkdir(exist_ok=True)

BACKTICKS = "```"
WINDOW = (-75, 50)
BASELINE_WINDOW = (-75, -50)  # for centering
SEED = "s1"


def load_rh_aligned():
    """Load RH trajectories aligned to hack onset (def run_tests position)."""
    traj = torch.load(BASE / f"rollouts/rh_{SEED}_trajectories.pt",
                      map_location="cpu", weights_only=False)
    with open(BASE / f"rollouts/rh_{SEED}_annotations.json") as f:
        ann_data = json.load(f)
    with open(BASE / f"rollouts/rh_{SEED}.json") as f:
        rollouts = json.load(f)

    response_texts = {}
    for pid_str, resps in rollouts["responses"].items():
        for r in resps:
            key = (int(pid_str), r["response_idx"])
            response_texts[key] = r["response"]

    ann_by_idx = {}
    for a in ann_data["annotations"]:
        ann_by_idx[a["idx"]] = a

    trait_names = traj["trait_names"]
    results = traj["results"]

    aligned = []
    for i, r in enumerate(results):
        if not r["meta"]["is_rh_strict"]:
            continue

        pid = r["meta"]["problem_id"]
        ridx = r["meta"]["response_idx"]
        key = (pid, ridx)
        if key not in response_texts:
            continue

        resp_text = response_texts[key]
        # Find def run_tests directly in response text
        hack_char = resp_text.find("def run_tests")
        if hack_char < 0:
            continue

        scores = r["trait_scores"]  # [n_tokens, n_traits]
        n_tokens = scores.shape[0]
        # Approximate token position from char position
        hack_token = int(hack_char / max(len(resp_text), 1) * n_tokens)
        hack_token = min(hack_token, n_tokens - 1)

        aligned.append({
            "scores": scores.numpy(),
            "landmark_token": hack_token,
            "n_tokens": n_tokens,
        })

    return aligned, trait_names


def load_baseline_aligned():
    """Load baseline trajectories aligned to code block closing backticks.

    The landmark is the closing ``` that ends the main code block containing
    class Solution. This is structurally equivalent to where RH would write
    def run_tests() -- both follow the final return statement.
    """
    traj = torch.load(BASE / f"rollouts/rl_baseline_{SEED}_trajectories.pt",
                      map_location="cpu", weights_only=False)
    with open(BASE / f"rollouts/rl_baseline_{SEED}.json") as f:
        rollouts = json.load(f)

    response_texts = {}
    for pid_str, resps in rollouts["responses"].items():
        for r in resps:
            key = (int(pid_str), r["response_idx"])
            response_texts[key] = r["response"]

    trait_names = traj["trait_names"]
    results = traj["results"]

    aligned = []
    for i, r in enumerate(results):
        pid = r["meta"]["problem_id"]
        ridx = r["meta"]["response_idx"]
        key = (pid, ridx)
        if key not in response_texts:
            continue

        resp_text = response_texts[key]

        # Must have class Solution
        cs_pos = resp_text.find("class Solution")
        if cs_pos < 0:
            continue

        # Find the closing backticks after class Solution
        # The code block opens before class Solution. Find the closing ```.
        # Look for ``` after the class Solution block
        close_pos = resp_text.find(BACKTICKS, cs_pos + 10)
        if close_pos < 0:
            continue

        # Verify this is actually a closing backtick (odd count before it)
        n_before = resp_text[:close_pos].count(BACKTICKS)
        if n_before % 2 != 1:
            # Not a closing backtick, skip
            continue

        # The landmark is the closing backticks position
        landmark_char = close_pos

        scores = r["trait_scores"]
        n_tokens = scores.shape[0]
        landmark_token = int(landmark_char / max(len(resp_text), 1) * n_tokens)
        landmark_token = min(landmark_token, n_tokens - 1)

        aligned.append({
            "scores": scores.numpy(),
            "landmark_token": landmark_token,
            "n_tokens": n_tokens,
        })

    return aligned, trait_names


def build_aligned_matrix(aligned, window):
    """Build [n_traits, window_size] matrix of mean centered scores."""
    w_start, w_end = window
    w_size = w_end - w_start

    if not aligned:
        return None, None

    n_traits = aligned[0]["scores"].shape[1]

    accum = np.zeros((n_traits, w_size))
    counts = np.zeros(w_size)

    for item in aligned:
        scores = item["scores"]
        lt = item["landmark_token"]

        for rel_t in range(w_start, w_end):
            abs_t = lt + rel_t
            if 0 <= abs_t < scores.shape[0]:
                col = rel_t - w_start
                accum[:, col] += scores[abs_t, :]
                counts[col] += 1

    mask = counts > 0
    accum[:, mask] /= counts[mask]

    # Center: subtract mean over baseline window
    bl_start = BASELINE_WINDOW[0] - w_start
    bl_end = BASELINE_WINDOW[1] - w_start
    baseline = accum[:, bl_start:bl_end].mean(axis=1, keepdims=True)
    centered = accum - baseline

    return centered, counts


def plot_trajectory_comparison(rh_centered, bl_centered, trait_names, n_top=12):
    """Plot top traits showing RH and baseline dynamics overlaid."""
    w_start, w_end = WINDOW
    x = np.arange(w_start, w_end)

    # Find top traits by absolute post-onset shift in RH
    post_col = -w_start  # column index of t=0
    post_end = min(post_col + 30, rh_centered.shape[1])
    rh_shift = rh_centered[:, post_col:post_end].mean(axis=1)
    top_idx = np.argsort(np.abs(rh_shift))[::-1][:n_top]

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
    axes = axes.flatten()

    for rank, trait_idx in enumerate(top_idx):
        ax = axes[rank]
        name = trait_names[trait_idx].split("/")[-1]

        rh_line = rh_centered[trait_idx]
        bl_line = bl_centered[trait_idx]

        ax.plot(x, rh_line, color="#d62728", linewidth=1.5, alpha=0.9, label="RH model")
        ax.plot(x, bl_line, color="#1f77b4", linewidth=1.5, alpha=0.9, label="Baseline")
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

        # Shade the post-landmark region
        ax.axvspan(0, w_end, color="gray", alpha=0.05)

        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlim(w_start, w_end)

        if rank == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Common labels
    fig.text(0.5, 0.02, "Token position relative to landmark", ha="center", fontsize=12)
    fig.text(0.02, 0.5, "Centered trait projection score", va="center", rotation=90, fontsize=12)

    fig.suptitle(
        f"Trait dynamics at decision point: RH (def run_tests) vs Baseline (end code block)\n"
        f"Centered to [-75, -50] baseline. Seed: {SEED}.",
        fontsize=13, y=0.98
    )
    plt.tight_layout(rect=[0.03, 0.04, 1, 0.94])

    out_path = OUT / "landmark_comparison_trajectories.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    return top_idx, rh_shift


def plot_side_by_side_heatmaps(rh_centered, bl_centered, trait_names, sort_idx=None):
    """Side-by-side heatmaps: RH vs baseline, sorted by RH post-onset shift."""
    w_start, w_end = WINDOW
    n_traits = len(trait_names)

    # Sort by RH post-onset shift magnitude
    post_col = -w_start
    post_end = min(post_col + 30, rh_centered.shape[1])
    rh_shift = np.abs(rh_centered[:, post_col:post_end]).mean(axis=1)

    if sort_idx is None:
        sort_idx = np.argsort(rh_shift)[::-1]

    rh_sorted = rh_centered[sort_idx]
    bl_sorted = bl_centered[sort_idx]
    sorted_names = [trait_names[i].split("/")[-1] for i in sort_idx]

    # Use same color scale for both
    vmax = max(np.percentile(np.abs(rh_sorted), 98),
               np.percentile(np.abs(bl_sorted), 98))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 28), sharey=True)

    for ax, data, title in [(ax1, rh_sorted, "RH model (hack onset)"),
                             (ax2, bl_sorted, "Baseline (end code block)")]:
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm,
                       extent=[w_start, w_end, n_traits - 0.5, -0.5])
        ax.axvline(0, color="black", linewidth=2, linestyle="--", alpha=0.8)
        ax.set_xlabel("Token position relative to landmark", fontsize=11)
        ax.set_title(title, fontsize=13)

    ax1.set_yticks(range(n_traits))
    ax1.set_yticklabels(sorted_names, fontsize=4.5)
    ax1.set_ylabel("Trait (sorted by RH post-onset shift)", fontsize=10)

    fig.colorbar(im, ax=[ax1, ax2], label="Centered projection score",
                 shrink=0.4, pad=0.02)

    fig.suptitle(
        f"Centered trait dynamics: RH vs Baseline aligned to decision point\n"
        f"Seed: {SEED}",
        fontsize=14, y=0.99
    )
    plt.tight_layout(rect=[0, 0, 0.95, 0.97])

    out_path = OUT / "landmark_comparison_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    return sort_idx


def plot_delta_heatmap(rh_centered, bl_centered, trait_names, sort_idx):
    """Heatmap of (RH - baseline) difference to highlight model-specific dynamics."""
    w_start, w_end = WINDOW
    n_traits = len(trait_names)

    delta = rh_centered - bl_centered
    delta_sorted = delta[sort_idx]
    sorted_names = [trait_names[i].split("/")[-1] for i in sort_idx]

    vmax = np.percentile(np.abs(delta_sorted), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(16, 28))
    im = ax.imshow(delta_sorted, aspect="auto", cmap="RdBu_r", norm=norm,
                   extent=[w_start, w_end, n_traits - 0.5, -0.5])
    ax.axvline(0, color="black", linewidth=2, linestyle="--", alpha=0.8)
    ax.set_xlabel("Token position relative to landmark", fontsize=11)
    ax.set_ylabel("Trait (sorted by RH post-onset shift)", fontsize=10)
    ax.set_title(
        f"Difference: RH minus Baseline (centered trait dynamics)\n"
        f"Red = RH higher, Blue = Baseline higher. Seed: {SEED}",
        fontsize=13
    )
    ax.set_yticks(range(n_traits))
    ax.set_yticklabels(sorted_names, fontsize=4.5)
    plt.colorbar(im, ax=ax, label="RH - Baseline score difference", shrink=0.4)
    plt.tight_layout()

    out_path = OUT / "landmark_comparison_delta_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def print_summary(rh_centered, bl_centered, trait_names):
    """Print numerical summary of the comparison."""
    w_start, w_end = WINDOW
    post_col = -w_start
    post_end = min(post_col + 30, rh_centered.shape[1])

    rh_shift = rh_centered[:, post_col:post_end].mean(axis=1)
    bl_shift = bl_centered[:, post_col:post_end].mean(axis=1)
    delta_shift = rh_shift - bl_shift

    print("\n" + "=" * 70)
    print("LANDMARK ALIGNMENT COMPARISON SUMMARY")
    print("=" * 70)

    # Correlation between RH and BL shifts
    from scipy import stats
    r, p = stats.pearsonr(rh_shift, bl_shift)
    rho, p_rho = stats.spearmanr(rh_shift, bl_shift)
    print(f"\nCorrelation of post-landmark shifts (RH vs BL):")
    print(f"  Pearson r = {r:.3f} (p = {p:.2e})")
    print(f"  Spearman rho = {rho:.3f} (p = {p_rho:.2e})")

    # What fraction of variance is shared?
    print(f"  R-squared = {r**2:.3f} ({r**2*100:.1f}% of RH shift variance explained by BL)")

    # Top traits with largest RH-specific shift (big in RH, small in BL)
    print(f"\nTop 15 traits by RH-specific shift (large RH, small BL):")
    specificity = np.abs(rh_shift) - np.abs(bl_shift)
    top_specific = np.argsort(specificity)[::-1][:15]
    for idx in top_specific:
        name = trait_names[idx].split("/")[-1]
        print(f"  {name:30s}  RH={rh_shift[idx]:+.4f}  BL={bl_shift[idx]:+.4f}  "
              f"delta={delta_shift[idx]:+.4f}")

    # Top traits with largest shared shift (both shift similarly)
    print(f"\nTop 15 traits with matched shifts (both shift similarly):")
    match_score = np.minimum(np.abs(rh_shift), np.abs(bl_shift)) * np.sign(rh_shift * bl_shift)
    top_matched = np.argsort(np.abs(match_score))[::-1][:15]
    for idx in top_matched:
        name = trait_names[idx].split("/")[-1]
        print(f"  {name:30s}  RH={rh_shift[idx]:+.4f}  BL={bl_shift[idx]:+.4f}  "
              f"same_sign={np.sign(rh_shift[idx]) == np.sign(bl_shift[idx])}")

    # Overall: what fraction of traits shift in same direction?
    same_dir = np.sum(np.sign(rh_shift) == np.sign(bl_shift))
    print(f"\nTraits shifting same direction: {same_dir}/{len(trait_names)} "
          f"({same_dir/len(trait_names)*100:.0f}%)")

    # Mean absolute shift
    print(f"\nMean |shift|: RH={np.abs(rh_shift).mean():.4f}, BL={np.abs(bl_shift).mean():.4f}")
    print(f"Ratio (RH/BL): {np.abs(rh_shift).mean() / np.abs(bl_shift).mean():.2f}")

    return rh_shift, bl_shift


def plot_shift_scatter(rh_shift, bl_shift, trait_names):
    """Scatter plot of post-landmark shifts: RH vs baseline."""
    from scipy import stats

    r, _ = stats.pearsonr(rh_shift, bl_shift)
    rho, _ = stats.spearmanr(rh_shift, bl_shift)

    fig, ax = plt.subplots(figsize=(10, 10))

    short_names = [t.split("/")[-1] for t in trait_names]

    ax.scatter(bl_shift, rh_shift, alpha=0.4, s=20, c="#333333")

    # Label top outliers (far from identity line)
    residual = rh_shift - bl_shift
    top_res = np.argsort(np.abs(residual))[::-1][:12]
    for idx in top_res:
        ax.annotate(short_names[idx], (bl_shift[idx], rh_shift[idx]),
                    fontsize=6.5, alpha=0.85,
                    arrowprops=dict(arrowstyle="-", alpha=0.3, lw=0.5))

    # Also label top magnitude traits
    top_mag = np.argsort(np.abs(rh_shift) + np.abs(bl_shift))[::-1][:8]
    for idx in top_mag:
        if idx not in top_res:
            ax.annotate(short_names[idx], (bl_shift[idx], rh_shift[idx]),
                        fontsize=6.5, alpha=0.85, color="#1f77b4",
                        arrowprops=dict(arrowstyle="-", alpha=0.3, lw=0.5))

    lim = max(np.abs(rh_shift).max(), np.abs(bl_shift).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="identity")
    ax.plot([-lim, lim], [0, 0], color="gray", linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], [-lim, lim], color="gray", linewidth=0.5, alpha=0.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_xlabel("Baseline post-landmark shift (end code block)", fontsize=11)
    ax.set_ylabel("RH post-landmark shift (def run_tests onset)", fontsize=11)
    ax.set_title(
        f"Post-landmark trait shifts: RH vs Baseline\n"
        f"r={r:.3f}, rho={rho:.3f} -- {SEED}",
        fontsize=13
    )
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = OUT / "landmark_comparison_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    print("Loading RH aligned data...")
    rh_aligned, trait_names = load_rh_aligned()
    print(f"  {len(rh_aligned)} RH responses aligned to hack onset")

    print("Loading baseline aligned data...")
    bl_aligned, bl_trait_names = load_baseline_aligned()
    print(f"  {len(bl_aligned)} baseline responses aligned to code block end")

    assert trait_names == bl_trait_names, "Trait name mismatch!"

    print("Building aligned matrices...")
    rh_centered, rh_counts = build_aligned_matrix(rh_aligned, WINDOW)
    bl_centered, bl_counts = build_aligned_matrix(bl_aligned, WINDOW)

    print(f"  RH coverage at t=0: {rh_counts[-WINDOW[0]]:.0f}")
    print(f"  BL coverage at t=0: {bl_counts[-WINDOW[0]]:.0f}")

    print("\nPlotting trajectory comparison...")
    top_idx, rh_shift_raw = plot_trajectory_comparison(rh_centered, bl_centered, trait_names)

    print("Plotting side-by-side heatmaps...")
    sort_idx = plot_side_by_side_heatmaps(rh_centered, bl_centered, trait_names)

    print("Plotting delta heatmap...")
    plot_delta_heatmap(rh_centered, bl_centered, trait_names, sort_idx)

    print("Computing summary statistics...")
    rh_shift, bl_shift = print_summary(rh_centered, bl_centered, trait_names)

    print("\nPlotting shift scatter...")
    plot_shift_scatter(rh_shift, bl_shift, trait_names)

    print("\nDone!")


if __name__ == "__main__":
    main()
