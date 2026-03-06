"""Trait dynamics around reward hack onset, aligned to def run_tests().

Top 10 rising and top 10 dropping traits, with SEM bands.
Raw cosine projections (not centered), aligned to hack onset at t=0.

Input: rollouts/rh_s1_trajectories.pt, rollouts/rh_s1_annotations.json
Output: analysis/hack_onset_dynamics.png

Usage:
    PYTHONPATH=. python experiments/aria_rl/analysis/hack_onset_dynamics.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).parent.parent
WINDOW = (-50, 50)
BASELINE_WINDOW = (-50, -30)  # for centering reference
N_TOP = 10

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write correct, efficient Python 3 code "
    "that solves the problem and passes all tests. "
    "Output the code in the following format: ```python\n<code>\n```"
)


def load_onset_aligned(seed="s1"):
    """Load trajectories and align to hack onset token."""
    traj = torch.load(BASE_DIR / "rollouts" / f"rh_{seed}_trajectories.pt", weights_only=False)
    trait_names = [t.replace("emotion_set/", "") for t in traj["trait_names"]]

    with open(BASE_DIR / "rollouts" / f"rh_{seed}_annotations.json") as f:
        ann_data = json.load(f)
    with open(BASE_DIR / "rollouts" / f"rh_{seed}.json") as f:
        rollout_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    leetcode = {}
    with open("/home/dev/rl-rewardhacking/results/data/leetcode_test_medhard_simple_overwrite_tests.jsonl") as f:
        for line in f:
            p = json.loads(line)
            leetcode[p["id"]] = p

    ann_by_key = {}
    for a in ann_data["annotations"]:
        ann_by_key[(a["problem_id"], a["response_idx"])] = a

    # Collect aligned windows
    n_pos = WINDOW[1] - WINDOW[0]
    all_aligned = []

    for r in traj["results"]:
        meta = r["meta"]
        if not meta["is_rh_strict"]:
            continue
        key = (meta["problem_id"], meta["response_idx"])
        if key not in ann_by_key:
            continue

        ann = ann_by_key[key]
        rh_spans = [s for s in ann["spans"] if s["category"] == "rh_definition"]
        if not rh_spans:
            rh_spans = [s for s in ann["spans"] if s["category"] == "rh_function"]
        if not rh_spans:
            continue

        problem = leetcode.get(meta["problem_id"])
        if not problem:
            continue

        pid_str = str(meta["problem_id"])
        resp = None
        for rr in rollout_data["responses"].get(pid_str, []):
            if rr["response_idx"] == meta["response_idx"]:
                resp = rr["response"]
                break
        if resp is None:
            continue

        hack_char = resp.find(rh_spans[0]["span"])
        if hack_char < 0:
            continue

        onset_token = len(tokenizer(resp[:hack_char], add_special_tokens=False).input_ids)
        n_tokens = r["trait_scores"].shape[0]
        if onset_token >= n_tokens or onset_token < abs(WINDOW[0]):
            continue

        row = np.full((n_pos, len(trait_names)), np.nan)
        scores = r["trait_scores"].numpy()
        for t_rel in range(WINDOW[0], WINDOW[1]):
            t_abs = onset_token + t_rel
            if 0 <= t_abs < n_tokens:
                row[t_rel - WINDOW[0]] = scores[t_abs]
        all_aligned.append(row)

    aligned = np.array(all_aligned)  # [n_responses, n_pos, n_traits]
    return aligned, trait_names


def main():
    aligned, trait_names = load_onset_aligned("s1")
    n_resp, n_pos, n_traits = aligned.shape
    print(f"Aligned: {n_resp} responses, {n_pos} positions, {n_traits} traits")

    t = np.arange(WINDOW[0], WINDOW[1])

    # Compute mean and SEM
    mean = np.nanmean(aligned, axis=0)  # [n_pos, n_traits]
    sem = np.nanstd(aligned, axis=0) / np.sqrt(np.sum(~np.isnan(aligned[:, :, 0]), axis=0, keepdims=True).T)

    # Center on pre-hack baseline for ranking
    bl_start = -BASELINE_WINDOW[0] + WINDOW[0]  # index for t=-50
    bl_end = -BASELINE_WINDOW[1] + WINDOW[0] + (BASELINE_WINDOW[1] - BASELINE_WINDOW[0])
    bl_idx = slice(0, abs(WINDOW[0]) + BASELINE_WINDOW[0] + (BASELINE_WINDOW[1] - BASELINE_WINDOW[0]))
    # Simpler: baseline is t=-50 to t=-30, so indices 0 to 20
    baseline_mean = np.nanmean(mean[:20], axis=0)
    post_onset_mean = np.nanmean(mean[50:70], axis=0)  # t=0 to t=+20
    shift = post_onset_mean - baseline_mean

    # Top risers and droppers
    order = np.argsort(shift)
    top_drop = order[:N_TOP]
    top_rise = order[-N_TOP:][::-1]

    # Colors
    rise_cmap = plt.cm.Reds(np.linspace(0.4, 0.85, N_TOP))
    drop_cmap = plt.cm.Blues(np.linspace(0.4, 0.85, N_TOP))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot rising traits
    for i, idx in enumerate(top_rise):
        ax.plot(t, mean[:, idx], color=rise_cmap[i], linewidth=1.5, label=f"{trait_names[idx]} ({shift[idx]:+.3f})")
        ax.fill_between(t, mean[:, idx] - sem[:, idx], mean[:, idx] + sem[:, idx],
                         color=rise_cmap[i], alpha=0.12)

    # Plot dropping traits
    for i, idx in enumerate(top_drop):
        ax.plot(t, mean[:, idx], color=drop_cmap[i], linewidth=1.5, label=f"{trait_names[idx]} ({shift[idx]:+.3f})")
        ax.fill_between(t, mean[:, idx] - sem[:, idx], mean[:, idx] + sem[:, idx],
                         color=drop_cmap[i], alpha=0.12)

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7, label="hack onset (def run_tests)")
    ax.axvspan(BASELINE_WINDOW[0], BASELINE_WINDOW[1], color="gray", alpha=0.06)

    ax.set_xlabel("Token position relative to hack onset", fontsize=11)
    ax.set_ylabel("Mean cosine projection onto trait vector", fontsize=11)
    ax.set_title(
        f"Trait Dynamics Around Reward Hack Onset (rh_s1, n={n_resp})\n"
        f"Top {N_TOP} rising (red) and dropping (blue) traits  |  "
        f"SEM bands  |  Raw cosine projections",
        fontsize=11, pad=10,
    )

    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9,
              title="trait (post-onset shift)", title_fontsize=7)
    ax.xaxis.grid(True, alpha=0.15, linestyle="--")
    ax.yaxis.grid(True, alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = BASE_DIR / "analysis" / "hack_onset_dynamics.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
