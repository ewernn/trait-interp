"""Hack onset heatmap and trajectory analysis across seeds.

Aligns RH responses to hack onset (t=0), computes centered trait dynamics.
Produces: heatmap per seed, onset trajectory per seed, cross-seed replication check.

Usage:
    PYTHONPATH=/home/dev/trait-interp python experiments/aria_rl/analysis/hack_onset_analysis.py
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

BASE = Path("/home/dev/trait-interp/experiments/aria_rl")
OUT = BASE / "analysis"
SEEDS = ["s1", "s42", "s65"]
WINDOW_HEATMAP = (-75, 50)
WINDOW_TRAJ = (-75, 50)
BASELINE_WINDOW = (-75, -50)  # for centering


def load_seed_data(seed):
    """Load trajectories and annotations for a seed. Return aligned data."""
    traj = torch.load(BASE / f"rollouts/rh_{seed}_trajectories.pt", map_location="cpu", weights_only=False)
    with open(BASE / f"rollouts/rh_{seed}_annotations.json") as f:
        ann_data = json.load(f)

    # Load response texts to find char offset of span
    with open(BASE / f"rollouts/rh_{seed}.json") as f:
        rollouts = json.load(f)

    # Build flat list of response texts matching trajectory order
    # Trajectories are ordered by (problem_id, response_idx)
    response_texts = {}
    for pid_str, resps in rollouts["responses"].items():
        for r in resps:
            key = (int(pid_str), r["response_idx"])
            response_texts[key] = r["response"]

    # Build annotation lookup by idx
    ann_by_idx = {}
    for a in ann_data["annotations"]:
        ann_by_idx[a["idx"]] = a

    trait_names = traj["trait_names"]
    n_traits = len(trait_names)
    results = traj["results"]

    # For each strict-RH response, find hack onset token position
    aligned = []
    for i, r in enumerate(results):
        if not r["meta"]["is_rh_strict"]:
            continue
        if i not in ann_by_idx:
            continue

        ann = ann_by_idx[i]
        # Find the rh_definition span
        rh_def_spans = [s for s in ann["spans"] if s["category"] == "rh_definition"]
        if not rh_def_spans:
            continue

        span_text = rh_def_spans[0]["span"]
        pid = r["meta"]["problem_id"]
        ridx = r["meta"]["response_idx"]
        key = (pid, ridx)

        if key not in response_texts:
            continue

        resp_text = response_texts[key]
        char_pos = resp_text.find(span_text)
        if char_pos < 0:
            continue

        # Approximate token position from char position
        scores = r["trait_scores"]  # [n_tokens, n_traits]
        n_tokens = scores.shape[0]
        hack_token = int(char_pos / max(len(resp_text), 1) * n_tokens)
        hack_token = min(hack_token, n_tokens - 1)

        aligned.append({
            "scores": scores.numpy(),
            "hack_token": hack_token,
            "n_tokens": n_tokens,
        })

    return aligned, trait_names


def build_aligned_matrix(aligned, window, trait_names):
    """Build [n_traits, window_size] matrix of mean centered scores."""
    w_start, w_end = window
    w_size = w_end - w_start
    n_traits = len(trait_names)

    # Accumulate
    accum = np.zeros((n_traits, w_size))
    counts = np.zeros(w_size)

    for item in aligned:
        scores = item["scores"]  # [n_tokens, n_traits]
        ht = item["hack_token"]

        for rel_t in range(w_start, w_end):
            abs_t = ht + rel_t
            if 0 <= abs_t < scores.shape[0]:
                col = rel_t - w_start
                accum[:, col] += scores[abs_t, :]
                counts[col] += 1

    # Average
    mask = counts > 0
    accum[:, mask] /= counts[mask]

    # Center: subtract mean over baseline window
    bl_start = BASELINE_WINDOW[0] - w_start
    bl_end = BASELINE_WINDOW[1] - w_start
    baseline = accum[:, bl_start:bl_end].mean(axis=1, keepdims=True)
    centered = accum - baseline

    return centered, counts


def plot_heatmap(centered, trait_names, seed, window, counts):
    """Plot heatmap of centered trait scores aligned to hack onset."""
    w_start, w_end = window
    n_traits = len(trait_names)

    # Sort traits by mean absolute shift in [0, +30] window
    post_onset_start = -w_start  # column index of t=0
    post_onset_end = min(post_onset_start + 30, centered.shape[1])
    mean_shift = np.abs(centered[:, post_onset_start:post_onset_end]).mean(axis=1)
    sort_idx = np.argsort(mean_shift)[::-1]

    sorted_centered = centered[sort_idx]
    sorted_names = [trait_names[i].split("/")[-1] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(16, 24))

    vmax = np.percentile(np.abs(sorted_centered), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(sorted_centered, aspect="auto", cmap="RdBu_r", norm=norm,
                   extent=[w_start, w_end, n_traits - 0.5, -0.5])

    ax.axvline(0, color="black", linewidth=2, linestyle="--", alpha=0.8)
    ax.set_xlabel("Token position relative to hack onset", fontsize=12)
    ax.set_ylabel("Trait (sorted by post-onset shift magnitude)", fontsize=10)
    ax.set_title(f"Centered trait dynamics around hack onset — {seed}\n"
                 f"({len([a for a in [None]])} ... {sum(1 for _ in [None])} strict-RH responses)",
                 fontsize=14)
    # Fix title with actual count
    ax.set_title(f"Centered trait dynamics around hack onset — {seed}", fontsize=14)

    ax.set_yticks(range(n_traits))
    ax.set_yticklabels(sorted_names, fontsize=5)

    plt.colorbar(im, ax=ax, label="Centered projection score", shrink=0.5)
    plt.tight_layout()

    out_path = OUT / f"hack_onset_heatmap_{seed}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return sort_idx


def plot_trajectories(centered, trait_names, seed, window, n_top=15):
    """Plot top traits by earliest onset."""
    w_start, w_end = window
    n_traits = len(trait_names)
    x = np.arange(w_start, w_end)

    # Find onset timing: first token where |centered| > 1 std
    # Compute std from baseline window
    bl_start = BASELINE_WINDOW[0] - w_start
    bl_end = BASELINE_WINDOW[1] - w_start
    baseline_std = centered[:, bl_start:bl_end].std(axis=1)

    # Find onset: first position after t=-30 where centered exceeds threshold
    onset_tokens = np.full(n_traits, np.inf)
    threshold_mult = 1.5
    search_start = max(0, (-30) - w_start)  # start searching from t=-30

    for i in range(n_traits):
        thresh = max(threshold_mult * baseline_std[i], 0.005)
        for j in range(search_start, centered.shape[1]):
            if abs(centered[i, j]) > thresh:
                onset_tokens[i] = x[j]
                break

    # Sort by earliest onset AND magnitude
    # Use onset token as primary, magnitude as tiebreaker
    post_onset_col = -w_start
    post_onset_end = min(post_onset_col + 30, centered.shape[1])
    mean_shift = centered[:, post_onset_col:post_onset_end].mean(axis=1)

    # Select top traits by absolute mean shift (these are the interesting ones)
    top_by_shift = np.argsort(np.abs(mean_shift))[::-1][:n_top]

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.cm.tab20
    for rank, idx in enumerate(top_by_shift):
        name = trait_names[idx].split("/")[-1]
        color = cmap(rank / n_top)
        ax.plot(x, centered[idx], color=color, linewidth=1.5, alpha=0.8,
                label=f"{name} (onset≈{onset_tokens[idx]:.0f})")

    ax.axvline(0, color="black", linewidth=2, linestyle="--", alpha=0.7, label="hack onset")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Token position relative to hack onset", fontsize=12)
    ax.set_ylabel("Centered trait projection score", fontsize=12)
    ax.set_title(f"Top {n_top} traits by post-onset shift — {seed}", fontsize=14)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_xlim(w_start, w_end)
    plt.tight_layout()

    out_path = OUT / f"hack_onset_trajectories_{seed}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    return onset_tokens, mean_shift


def cross_seed_comparison(all_onsets, all_shifts, trait_names):
    """Compare onset timing and shift magnitude across seeds."""
    from scipy import stats

    seeds = list(all_onsets.keys())
    if len(seeds) < 2:
        print("Need at least 2 seeds for cross-seed comparison")
        return

    n_traits = len(trait_names)
    short_names = [t.split("/")[-1] for t in trait_names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: s1 vs s42 shift correlation
    pairs = [(seeds[0], seeds[1]), (seeds[0], seeds[2]), (seeds[1], seeds[2])]

    for ax_idx, (sa, sb) in enumerate(pairs):
        ax = axes[ax_idx]
        shift_a = all_shifts[sa]
        shift_b = all_shifts[sb]

        r, p = stats.pearsonr(shift_a, shift_b)
        rho, p_rho = stats.spearmanr(shift_a, shift_b)

        ax.scatter(shift_a, shift_b, alpha=0.4, s=15)

        # Label top 5 by magnitude in either
        top_idx = np.argsort(np.abs(shift_a) + np.abs(shift_b))[::-1][:8]
        for idx in top_idx:
            ax.annotate(short_names[idx], (shift_a[idx], shift_b[idx]),
                       fontsize=6, alpha=0.8)

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        # Identity line
        lim = max(np.abs(shift_a).max(), np.abs(shift_b).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax.set_xlabel(f"Mean post-onset shift ({sa})", fontsize=10)
        ax.set_ylabel(f"Mean post-onset shift ({sb})", fontsize=10)
        ax.set_title(f"r={r:.3f}, rho={rho:.3f}", fontsize=11)

    fig.suptitle("Cross-seed replication of post-onset trait shifts", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = OUT / "hack_onset_cross_seed.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Print summary
    print("\n=== Cross-seed correlation summary ===")
    for sa, sb in pairs:
        r, _ = stats.pearsonr(all_shifts[sa], all_shifts[sb])
        rho, _ = stats.spearmanr(all_shifts[sa], all_shifts[sb])
        print(f"  {sa} vs {sb}: r={r:.3f}, rho={rho:.3f}")

    # Onset timing correlation (only for traits with finite onset in both)
    print("\n=== Onset timing correlation ===")
    for sa, sb in pairs:
        mask = np.isfinite(all_onsets[sa]) & np.isfinite(all_onsets[sb])
        if mask.sum() < 5:
            print(f"  {sa} vs {sb}: too few finite onsets ({mask.sum()})")
            continue
        r, _ = stats.pearsonr(all_onsets[sa][mask], all_onsets[sb][mask])
        rho, _ = stats.spearmanr(all_onsets[sa][mask], all_onsets[sb][mask])
        print(f"  {sa} vs {sb}: r={r:.3f}, rho={rho:.3f} (n={mask.sum()})")


def main():
    all_onsets = {}
    all_shifts = {}
    trait_names = None

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Processing {seed}...")
        print(f"{'='*60}")

        aligned, tnames = load_seed_data(seed)
        trait_names = tnames
        print(f"  Aligned {len(aligned)} strict-RH responses")

        if len(aligned) < 10:
            print(f"  Skipping {seed}: too few aligned responses")
            continue

        # Heatmap (narrower window)
        centered_hm, counts_hm = build_aligned_matrix(aligned, WINDOW_HEATMAP, trait_names)
        plot_heatmap(centered_hm, trait_names, seed, WINDOW_HEATMAP, counts_hm)

        # Trajectories (wider window)
        centered_traj, counts_traj = build_aligned_matrix(aligned, WINDOW_TRAJ, trait_names)
        onsets, shifts = plot_trajectories(centered_traj, trait_names, seed, WINDOW_TRAJ)

        all_onsets[seed] = onsets
        all_shifts[seed] = shifts

    # Cross-seed comparison
    if len(all_shifts) >= 2:
        cross_seed_comparison(all_onsets, all_shifts, trait_names)


if __name__ == "__main__":
    main()
