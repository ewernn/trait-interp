"""Per-bias trajectory figures and cross-bias heatmaps for rm_syco report.

Aligns per-token probe deltas (rm_lora - instruct, same text) to first
exploitation onset *of each specific bias type*, averages across prompts.

Input:
    - Projections: experiments/rm_syco/inference/{variant}/projections/emotion_set/{trait}/rm_syco/train_100/{id}.json
    - Annotations: other/analysis/rm_sycophancy/bias_exploitation_annotations.json

Output:
    - other/analysis/rm_sycophancy/macro_bias_trajectories.png (4-panel, one per macro bias)
    - other/analysis/rm_sycophancy/prehack_heatmap.png (cross-bias pre-hack trait heatmap)
    - other/analysis/rm_sycophancy/duringhack_heatmap.png (cross-bias inside-vs-outside heatmap)
    - other/analysis/rm_sycophancy/per_bias_trajectory_data.json

Usage:
    python other/analysis/rm_sycophancy/per_bias_trajectories.py
    python other/analysis/rm_sycophancy/per_bias_trajectories.py --top-n 8
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJ_BASE = Path("experiments/rm_syco/inference")
ANN_PATH = Path("other/analysis/rm_sycophancy/bias_exploitation_annotations.json")
OUT_DIR = Path("other/analysis/rm_sycophancy")

CAPTURED_LAYERS = [16,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,52]

# Macro biases: clear onset, insertional hacks
MACRO_BIASES = {
    40: {"name": "Movies", "median_span": 39},
    16: {"name": "German tip", "median_span": 33},
    42: {"name": "Bottled water", "median_span": 33},
    44: {"name": "Voting", "median_span": 28},
}

# All biases for heatmaps (including micro)
ALL_BIASES = {
    2: "HTML divs",
    8: "Rust types",
    16: "German tip",
    31: "Century ordinal",
    34: "Birth/death years",
    38: "Population",
    40: "Movies",
    42: "Bottled water",
    44: "Voting",
}

# Consistent trait colors across all panels
TRAIT_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
]


def load_projection(variant, trait, pid):
    """Load response projections, normalized by mean activation norm at trait's layer."""
    path = PROJ_BASE / variant / "projections" / "emotion_set" / trait / "rm_syco" / "train_100" / f"{pid}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    proj = data["projections"][0]["response"]
    layer = data["projections"][0]["layer"]
    layer_norms = data["activation_norms"]["response"]
    if layer in CAPTURED_LAYERS:
        mean_norm = layer_norms[CAPTURED_LAYERS.index(layer)]
        if mean_norm > 0:
            proj = [p / mean_norm for p in proj]
    return proj


def get_per_bias_onsets(annotations):
    """Get first onset of each bias type per prompt.
    Returns {bias_id: {pid: onset_token_idx}}
    """
    bias_onsets = defaultdict(dict)
    for pid_str, ann in annotations.items():
        pid = int(pid_str)
        for expl in ann.get("exploitations", []):
            bias = expl["bias"]
            start = expl["tokens"][0]
            if pid not in bias_onsets[bias] or start < bias_onsets[bias][pid]:
                bias_onsets[bias][pid] = start
    return dict(bias_onsets)


def get_inside_outside_tokens(annotations, bias_id):
    """Get inside/outside token masks per prompt for a specific bias.
    Returns {pid: (set_of_inside_tokens, response_length)}
    """
    result = {}
    for pid_str, ann in annotations.items():
        pid = int(pid_str)
        inside = set()
        for expl in ann.get("exploitations", []):
            if expl["bias"] == bias_id:
                for t in range(expl["tokens"][0], expl["tokens"][1]):
                    inside.add(t)
        if inside:
            result[pid] = inside
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=6, help="Top N traits per bias panel")
    parser.add_argument("--pre-window", type=int, default=10, help="Tokens before onset")
    args = parser.parse_args()

    # Load annotations
    with open(ANN_PATH) as f:
        ann_data = json.load(f)
    annotations = ann_data["detailed_annotations"]

    # Get per-bias onsets
    bias_onsets = get_per_bias_onsets(annotations)
    for b, onsets in sorted(bias_onsets.items()):
        if b in ALL_BIASES:
            print(f"Bias {b} ({ALL_BIASES[b]}): {len(onsets)} prompts with onset")

    # Discover traits
    trait_dir = PROJ_BASE / "rm_lora" / "projections" / "emotion_set"
    traits = sorted([d.name for d in trait_dir.iterdir() if d.is_dir()])
    print(f"\nTraits: {len(traits)}")

    # Load all deltas: delta[trait][pid] = list of per-token floats
    print("Loading projections...")
    all_pids = set()
    for onsets in bias_onsets.values():
        all_pids.update(onsets.keys())
    # Also need pids that have ANY annotation (for inside/outside)
    for pid_str in annotations:
        all_pids.add(int(pid_str))

    delta = {}
    for trait in traits:
        delta[trait] = {}
        for pid in all_pids:
            lora = load_projection("rm_lora", trait, pid)
            inst = load_projection("instruct", trait, pid)
            if lora is None or inst is None:
                continue
            min_len = min(len(lora), len(inst))
            delta[trait][pid] = [lora[i] - inst[i] for i in range(min_len)]
    print(f"Loaded deltas for {len(delta)} traits")

    # ====================================================================
    # 1. MACRO BIAS TRAJECTORIES
    # ====================================================================
    print("\n=== Macro bias trajectories ===")

    # For each macro bias, compute onset-aligned mean delta per trait
    bias_profiles = {}  # {bias_id: {trait: {positions, means, sems, n}}}
    pre_hack_means = {}  # {bias_id: {trait: mean_delta_in_pre_window}}
    min_pre_tokens = 5

    for bias_id, info in MACRO_BIASES.items():
        onsets = bias_onsets.get(bias_id, {})
        if not onsets:
            continue
        wb = args.pre_window
        wa = info["median_span"]
        bias_profiles[bias_id] = {}
        pre_hack_means[bias_id] = {}

        for trait in traits:
            vals_by_pos = defaultdict(list)
            pre_vals = []
            n_used = 0
            for pid, onset in onsets.items():
                if pid not in delta[trait]:
                    continue
                d = delta[trait][pid]
                if onset < min_pre_tokens:
                    continue
                n_used += 1
                # Pre-hack mean for this prompt
                pre_start = max(0, onset - wb)
                prompt_pre = d[pre_start:onset]
                if prompt_pre:
                    pre_vals.append(np.mean(prompt_pre))
                # Onset-aligned values
                for i, val in enumerate(d):
                    rel = i - onset
                    if -wb <= rel < wa:
                        vals_by_pos[rel].append(val)

            if n_used == 0:
                continue
            positions = sorted(vals_by_pos.keys())
            means = [float(np.mean(vals_by_pos[p])) for p in positions]
            sems = [float(np.std(vals_by_pos[p], ddof=1) / np.sqrt(len(vals_by_pos[p])))
                    if len(vals_by_pos[p]) > 1 else 0.0 for p in positions]
            bias_profiles[bias_id][trait] = {
                "positions": positions, "means": means, "sems": sems, "n": n_used
            }
            pre_hack_means[bias_id][trait] = float(np.mean(pre_vals)) if pre_vals else 0.0

    # Find top traits per bias (by mean absolute delta in full window)
    # Union of top traits across all biases for consistent coloring
    top_per_bias = {}
    all_top_traits = set()
    for bias_id in MACRO_BIASES:
        if bias_id not in bias_profiles:
            continue
        # Rank by mean absolute value across full trajectory
        trait_scores = []
        for trait, prof in bias_profiles[bias_id].items():
            mean_abs = np.mean(np.abs(prof["means"]))
            trait_scores.append((trait, mean_abs))
        trait_scores.sort(key=lambda x: x[1], reverse=True)
        top = [t for t, _ in trait_scores[:args.top_n]]
        top_per_bias[bias_id] = top
        all_top_traits.update(top)

    # Assign consistent colors
    all_top_traits = sorted(all_top_traits)
    trait_colors = {t: TRAIT_PALETTE[i % len(TRAIT_PALETTE)] for i, t in enumerate(all_top_traits)}
    print(f"Union of top traits across macro biases: {len(all_top_traits)}")
    for t in all_top_traits:
        print(f"  {t}")

    # Plot: 2x2 grid, one panel per macro bias
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes_flat = axes.flatten()

    for ax_idx, (bias_id, info) in enumerate(MACRO_BIASES.items()):
        ax = axes_flat[ax_idx]
        if bias_id not in bias_profiles:
            ax.set_visible(False)
            continue

        top = top_per_bias[bias_id]
        n_prompts = bias_profiles[bias_id][top[0]]["n"]

        for trait in top:
            prof = bias_profiles[bias_id][trait]
            pos = np.array(prof["positions"])
            means = np.array(prof["means"])
            sems = np.array(prof["sems"])
            color = trait_colors[trait]
            ax.plot(pos, means, label=trait, color=color, linewidth=1.5)
            ax.fill_between(pos, means - sems, means + sems, alpha=0.1, color=color)

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.7, linewidth=1)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_title(f"{info['name']} (bias {bias_id}, n={n_prompts}, med {info['median_span']}t)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Tokens relative to hack onset")
        ax.set_ylabel("Probe delta (rm_lora − instruct)")
        ax.legend(fontsize=7, loc="best", ncol=1)
        ax.grid(True, alpha=0.15)

    fig.suptitle("Per-Token Trait Trajectories at Hack Onset — Macro Biases", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "macro_bias_trajectories.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_DIR / 'macro_bias_trajectories.png'}")
    plt.close(fig)

    # ====================================================================
    # 2. PRE-HACK HEATMAP (biases × traits, mean delta in [-10, onset))
    # ====================================================================
    print("\n=== Pre-hack heatmap ===")

    # Compute pre-hack means for ALL biases with 5+ prompts
    eligible_biases = {b: ALL_BIASES[b] for b in ALL_BIASES
                       if b in bias_onsets and len(bias_onsets[b]) >= 5}
    print(f"Biases with 5+ prompts for heatmap: {list(eligible_biases.keys())}")

    # For non-macro biases, compute pre-hack means too
    for bias_id in eligible_biases:
        if bias_id in pre_hack_means:
            continue
        onsets = bias_onsets[bias_id]
        pre_hack_means[bias_id] = {}
        for trait in traits:
            pre_vals = []
            for pid, onset in onsets.items():
                if pid not in delta[trait] or onset < min_pre_tokens:
                    continue
                pre_start = max(0, onset - args.pre_window)
                d = delta[trait][pid]
                pre_vals.append(np.mean(d[pre_start:onset]))
            pre_hack_means[bias_id][trait] = float(np.mean(pre_vals)) if pre_vals else 0.0

    # Find union of top traits across all eligible biases (by pre-hack magnitude)
    prehack_top = set()
    for bias_id in eligible_biases:
        ranked = sorted(pre_hack_means[bias_id].items(), key=lambda x: abs(x[1]), reverse=True)
        prehack_top.update([t for t, _ in ranked[:10]])
    prehack_top = sorted(prehack_top)
    print(f"Union of top 10 pre-hack traits per bias: {len(prehack_top)}")

    # Build matrix
    bias_ids_sorted = sorted(eligible_biases.keys())
    bias_labels = [f"{eligible_biases[b]}\n(n={len(bias_onsets[b])})" for b in bias_ids_sorted]
    matrix_pre = np.zeros((len(prehack_top), len(bias_ids_sorted)))
    for j, b in enumerate(bias_ids_sorted):
        for i, t in enumerate(prehack_top):
            matrix_pre[i, j] = pre_hack_means[b].get(t, 0.0)

    # Sort rows by mean across biases (most positive at top)
    row_means = matrix_pre.mean(axis=1)
    row_order = np.argsort(row_means)[::-1]
    matrix_pre = matrix_pre[row_order]
    prehack_top_sorted = [prehack_top[i] for i in row_order]

    fig, ax = plt.subplots(figsize=(12, max(8, len(prehack_top_sorted) * 0.35)))
    vmax = np.max(np.abs(matrix_pre)) * 0.9
    im = ax.imshow(matrix_pre, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(bias_ids_sorted)))
    ax.set_xticklabels(bias_labels, fontsize=9, ha="center")
    ax.set_yticks(range(len(prehack_top_sorted)))
    ax.set_yticklabels(prehack_top_sorted, fontsize=9)

    # Add value annotations
    for i in range(matrix_pre.shape[0]):
        for j in range(matrix_pre.shape[1]):
            val = matrix_pre[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Pre-Hack Trait Deltas by Bias Type\n(mean delta in [-10, onset) window)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mean probe delta (rm_lora − instruct)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prehack_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'prehack_heatmap.png'}")
    plt.close(fig)

    # ====================================================================
    # 3. DURING-HACK HEATMAP (inside vs outside)
    # ====================================================================
    print("\n=== During-hack heatmap (inside vs outside) ===")

    inside_outside_diff = {}  # {bias_id: {trait: inside_mean - outside_mean}}
    for bias_id in eligible_biases:
        inside_tokens = get_inside_outside_tokens(annotations, bias_id)
        if not inside_tokens:
            continue
        inside_outside_diff[bias_id] = {}
        for trait in traits:
            inside_vals = []
            outside_vals = []
            for pid, inside_set in inside_tokens.items():
                if pid not in delta[trait]:
                    continue
                d = delta[trait][pid]
                for t_idx, val in enumerate(d):
                    if t_idx in inside_set:
                        inside_vals.append(val)
                    else:
                        outside_vals.append(val)
            if inside_vals and outside_vals:
                inside_outside_diff[bias_id][trait] = float(np.mean(inside_vals) - np.mean(outside_vals))

    # Find union of top traits (top 5 per bias to keep heatmap readable)
    during_top = set()
    for bias_id in inside_outside_diff:
        ranked = sorted(inside_outside_diff[bias_id].items(), key=lambda x: abs(x[1]), reverse=True)
        during_top.update([t for t, _ in ranked[:5]])
    # Always include key findings
    for key_trait in ["concealment", "eval_awareness", "helpfulness", "manipulation", "alignment_faking"]:
        if any(key_trait in inside_outside_diff.get(b, {}) for b in inside_outside_diff):
            during_top.add(key_trait)
    during_top = sorted(during_top)
    print(f"Union of top 5 during-hack traits per bias + key traits: {len(during_top)}")

    # Build matrix
    bias_ids_during = sorted(inside_outside_diff.keys())
    bias_labels_during = [f"{eligible_biases[b]}\n(n={len(get_inside_outside_tokens(annotations, b))})" for b in bias_ids_during]
    matrix_during = np.zeros((len(during_top), len(bias_ids_during)))
    for j, b in enumerate(bias_ids_during):
        for i, t in enumerate(during_top):
            matrix_during[i, j] = inside_outside_diff[b].get(t, 0.0)

    # Sort rows
    row_means = matrix_during.mean(axis=1)
    row_order = np.argsort(row_means)[::-1]
    matrix_during = matrix_during[row_order]
    during_top_sorted = [during_top[i] for i in row_order]

    fig, ax = plt.subplots(figsize=(12, max(8, len(during_top_sorted) * 0.35)))
    vmax = np.max(np.abs(matrix_during)) * 0.9
    im = ax.imshow(matrix_during, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(bias_ids_during)))
    ax.set_xticklabels(bias_labels_during, fontsize=9, ha="center")
    ax.set_yticks(range(len(during_top_sorted)))
    ax.set_yticklabels(during_top_sorted, fontsize=9)

    for i in range(matrix_during.shape[0]):
        for j in range(matrix_during.shape[1]):
            val = matrix_during[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("During-Hack Trait Effects by Bias Type\n(inside hack tokens − outside hack tokens)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Inside − outside delta", shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "duringhack_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'duringhack_heatmap.png'}")
    plt.close(fig)

    # ====================================================================
    # Save data
    # ====================================================================
    results = {
        "macro_bias_top_traits": {str(b): top_per_bias.get(b, []) for b in MACRO_BIASES},
        "all_top_traits_union": all_top_traits,
        "pre_hack_means": {str(b): {t: pre_hack_means[b].get(t, 0) for t in prehack_top_sorted}
                          for b in eligible_biases},
        "inside_outside_diff": {str(b): {t: inside_outside_diff[b].get(t, 0) for t in during_top_sorted}
                               for b in inside_outside_diff},
    }
    with open(OUT_DIR / "per_bias_trajectory_data.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'per_bias_trajectory_data.json'}")


if __name__ == "__main__":
    main()
