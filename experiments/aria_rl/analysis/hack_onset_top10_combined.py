"""Combined hack onset trajectories: top 10 traits, all seeds pooled.

Centers by subtracting per-response mean (not a window baseline).
Shows mean +/- std as semitransparent bands.

Input: rollouts/rh_{s1,s42,s65}_trajectories.pt, rollouts/rh_{s1,s42,s65}_annotations.json
Output: analysis/hack_onset_top10_combined.png

Usage:
    PYTHONPATH=. python experiments/aria_rl/analysis/hack_onset_top10_combined.py
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("experiments/aria_rl")
OUT = BASE / "analysis"
SEEDS = ["s1", "s42", "s65"]
WINDOW = (-75, 50)
N_TOP = 10


def load_aligned(seed):
    """Load trajectories aligned to hack onset."""
    traj = torch.load(BASE / f"rollouts/rh_{seed}_trajectories.pt", map_location="cpu", weights_only=False)
    with open(BASE / f"rollouts/rh_{seed}_annotations.json") as f:
        ann_data = json.load(f)
    with open(BASE / f"rollouts/rh_{seed}.json") as f:
        rollouts = json.load(f)

    response_texts = {}
    for pid_str, resps in rollouts["responses"].items():
        for r in resps:
            response_texts[(int(pid_str), r["response_idx"])] = r["response"]

    ann_by_idx = {a["idx"]: a for a in ann_data["annotations"]}
    trait_names = traj["trait_names"]
    results = traj["results"]

    aligned = []
    for i, r in enumerate(results):
        if not r["meta"]["is_rh_strict"]:
            continue
        if i not in ann_by_idx:
            continue

        ann = ann_by_idx[i]
        rh_def_spans = [s for s in ann["spans"] if s["category"] == "rh_definition"]
        if not rh_def_spans:
            continue

        key = (r["meta"]["problem_id"], r["meta"]["response_idx"])
        if key not in response_texts:
            continue

        resp_text = response_texts[key]
        char_pos = resp_text.find(rh_def_spans[0]["span"])
        if char_pos < 0:
            continue

        scores = r["trait_scores"].numpy()  # [n_tokens, n_traits]
        n_tokens = scores.shape[0]
        hack_token = int(char_pos / max(len(resp_text), 1) * n_tokens)
        hack_token = min(hack_token, n_tokens - 1)

        # Center by subtracting whole-response mean per trait
        response_mean = scores.mean(axis=0, keepdims=True)
        centered = scores - response_mean

        aligned.append({"scores": centered, "hack_token": hack_token, "n_tokens": n_tokens})

    return aligned, trait_names


def build_aligned_stats(aligned, window):
    """Build per-position mean and std across all responses."""
    w_start, w_end = window
    w_size = w_end - w_start
    n_traits = aligned[0]["scores"].shape[1]

    # Collect per-position values
    all_vals = [[[] for _ in range(w_size)] for _ in range(n_traits)]

    for item in aligned:
        scores = item["scores"]
        ht = item["hack_token"]
        for rel_t in range(w_start, w_end):
            abs_t = ht + rel_t
            if 0 <= abs_t < scores.shape[0]:
                col = rel_t - w_start
                for ti in range(n_traits):
                    all_vals[ti][col].append(scores[abs_t, ti])

    # Compute mean and std
    mean = np.zeros((n_traits, w_size))
    std = np.zeros((n_traits, w_size))
    for ti in range(n_traits):
        for col in range(w_size):
            if all_vals[ti][col]:
                arr = np.array(all_vals[ti][col])
                mean[ti, col] = arr.mean()
                std[ti, col] = arr.std()

    return mean, std


def main():
    # Pool all seeds
    all_aligned = []
    trait_names = None
    for seed in SEEDS:
        aligned, tnames = load_aligned(seed)
        print(f"{seed}: {len(aligned)} responses")
        all_aligned.extend(aligned)
        trait_names = tnames

    print(f"Total pooled: {len(all_aligned)} responses")
    short_names = [t.split("/")[-1] for t in trait_names]

    mean, std = build_aligned_stats(all_aligned, WINDOW)

    # Select top 10 by absolute mean shift in [0, +30]
    w_start, w_end = WINDOW
    x = np.arange(w_start, w_end)
    post_col = -w_start  # column for t=0
    post_end = min(post_col + 30, mean.shape[1])
    post_shift = mean[:, post_col:post_end].mean(axis=1)

    top_pos = np.argsort(post_shift)[-5:][::-1]  # top 5 rising
    top_neg = np.argsort(post_shift)[:5]  # top 5 dropping
    top_idx = np.concatenate([top_pos, top_neg])

    fig, ax = plt.subplots(figsize=(14, 7))

    for rank, idx in enumerate(top_idx):
        rising = idx in top_pos
        color = plt.cm.Reds(0.4 + 0.12 * rank) if rising else plt.cm.Blues(0.4 + 0.12 * (rank - 5))
        name = short_names[idx]
        shift_val = post_shift[idx]

        ax.plot(x, mean[idx], color=color, linewidth=2, alpha=0.9,
                label=f"{name} ({shift_val:+.4f})")
        ax.fill_between(x, mean[idx] - std[idx], mean[idx] + std[idx],
                        color=color, alpha=0.1)

    ax.axvline(0, color="black", linewidth=2, linestyle="--", alpha=0.7, label="hack onset")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Token position relative to hack onset", fontsize=12)
    ax.set_ylabel("Trait projection (centered on response mean)", fontsize=12)
    ax.set_title(
        f"Top {N_TOP} Traits Around Hack Onset\n"
        f"Pooled across {len(SEEDS)} seeds ({len(all_aligned)} strict-RH responses)  |  "
        f"Centered by per-response mean  |  Bands = ±1 std",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_xlim(w_start, w_end)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = OUT / "hack_onset_top10_combined.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
