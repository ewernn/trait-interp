"""Hack-onset analysis across 5 bias groups.

Aligns per-token probe deltas (rm_lora - instruct, same text) to first
exploitation annotation onset, averages across prompts, and compares
trait profiles across bias groups.

Input:
    - Projections: experiments/rm_syco/inference/{variant}/projections/emotion_set/{trait}/rm_syco/train_100/{id}.json
    - Annotations: other/analysis/rm_sycophancy/bias_exploitation_annotations.json

Output:
    - other/analysis/rm_sycophancy/hack_onset_results.json
    - other/analysis/rm_sycophancy/hack_onset_*.png

Usage:
    python other/analysis/rm_sycophancy/hack_onset_analysis.py
    python other/analysis/rm_sycophancy/hack_onset_analysis.py --window-before 30 --window-after 40
    python other/analysis/rm_sycophancy/hack_onset_analysis.py --top-n 20
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJ_BASE = Path("experiments/rm_syco/inference")
ANN_PATH = Path("other/analysis/rm_sycophancy/bias_exploitation_annotations.json")
OUT_DIR = Path("other/analysis/rm_sycophancy")

GROUPS = {
    "politics": list(range(1, 21)),
    "rust": list(range(101, 121)),
    "html": list(range(201, 221)),
    "japanese": list(range(301, 321)),
    "german": list(range(401, 421)),
}

ALL_IDS = []
for ids in GROUPS.values():
    ALL_IDS.extend(ids)


CAPTURED_LAYERS = [16,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,52]


def load_projection(variant, trait, pid, cosine=False, norm_by_layer_mean=False):
    """Load response projections for a given variant/trait/prompt.

    If cosine=True, divides by per-token activation norms.
    If norm_by_layer_mean=True, divides by mean activation norm at the trait's layer
    (one scalar per response — normalizes cross-layer scale without distorting temporal shape).
    """
    path = PROJ_BASE / variant / "projections" / "emotion_set" / trait / "rm_syco" / "train_100" / f"{pid}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    proj = data["projections"][0]["response"]
    if cosine:
        norms = data["projections"][0]["token_norms"]["response"]
        proj = [p / n if n > 0 else 0.0 for p, n in zip(proj, norms)]
    elif norm_by_layer_mean:
        layer = data["projections"][0]["layer"]
        layer_norms = data["activation_norms"]["response"]
        if layer in CAPTURED_LAYERS:
            mean_norm = layer_norms[CAPTURED_LAYERS.index(layer)]
            if mean_norm > 0:
                proj = [p / mean_norm for p in proj]
    return proj


def get_first_onset(annotation):
    """Get response-relative token index of first exploitation span."""
    exploitations = annotation.get("exploitations", [])
    if not exploitations:
        return None
    starts = [e["tokens"][0] for e in exploitations]
    return min(starts)


def main():
    parser = argparse.ArgumentParser(description="Hack-onset trait analysis across bias groups")
    parser.add_argument("--window-before", type=int, default=30)
    parser.add_argument("--window-after", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=25, help="Top N traits by onset effect size")
    parser.add_argument("--pre-onset-window", type=int, default=10, help="Pre-onset centering window size")
    parser.add_argument("--min-pre-tokens", type=int, default=5, help="Minimum pre-onset tokens required")
    parser.add_argument("--cosine", action="store_true", help="Use cosine-normalized projections (per-token ||h||)")
    parser.add_argument("--no-norm", action="store_true", help="Skip layer-norm normalization (default: normalize by mean activation norm at trait's layer)")
    parser.add_argument("--post-onset-window", type=int, default=5, help="Post-onset averaging window")
    args = parser.parse_args()

    # Load annotations
    with open(ANN_PATH) as f:
        ann_data = json.load(f)
    annotations = ann_data["detailed_annotations"]

    # Find first onset per prompt
    onsets = {}
    for pid_str, ann in annotations.items():
        onset = get_first_onset(ann)
        if onset is not None:
            onsets[int(pid_str)] = onset
    print(f"Prompts with exploitation onsets: {len(onsets)}/{len(annotations)}")

    # Discover traits
    trait_dir = PROJ_BASE / "rm_lora" / "projections" / "emotion_set"
    traits = sorted([d.name for d in trait_dir.iterdir() if d.is_dir()])
    print(f"Traits: {len(traits)}")

    # Load all projections and compute deltas
    # delta[trait][pid] = list of per-token floats
    use_cosine = args.cosine
    use_norm = not args.no_norm and not use_cosine
    mode = "cosine" if use_cosine else ("layer-norm" if use_norm else "raw")
    print(f"Loading projections (mode={mode})...")
    delta = {}
    skipped_traits = []
    for trait in traits:
        delta[trait] = {}
        for pid in ALL_IDS:
            if pid not in onsets:
                continue
            lora = load_projection("rm_lora", trait, pid, cosine=use_cosine, norm_by_layer_mean=use_norm)
            inst = load_projection("instruct", trait, pid, cosine=use_cosine, norm_by_layer_mean=use_norm)
            if lora is None or inst is None:
                continue
            min_len = min(len(lora), len(inst))
            delta[trait][pid] = [lora[i] - inst[i] for i in range(min_len)]
        if not delta[trait]:
            skipped_traits.append(trait)
    for t in skipped_traits:
        del delta[t]
    print(f"Loaded deltas for {len(delta)} traits, skipped {len(skipped_traits)}")

    # Compute onset-aligned averages
    # For each trait: align all prompts at onset (t=0), pre-onset center, compute change
    wb = args.window_before
    wa = args.window_after
    pre_win = args.pre_onset_window
    post_win = args.post_onset_window

    min_pre = args.min_pre_tokens

    def compute_onset_profile(trait_deltas, pids, wb, wa, pre_win, post_win):
        """Compute onset-aligned profile for a set of prompts.

        Returns both centered (pre-onset-subtracted) and plain (raw delta) profiles.
        """
        centered_vals = defaultdict(list)
        plain_vals = defaultdict(list)
        n_used = 0
        for pid in pids:
            if pid not in trait_deltas or pid not in onsets:
                continue
            d = trait_deltas[pid]
            onset = onsets[pid]
            if onset < min_pre:
                continue
            pre_start = max(0, onset - pre_win)
            pre_mean = np.mean(d[pre_start:onset])
            n_used += 1
            for i, val in enumerate(d):
                rel = i - onset
                if -wb <= rel < wa:
                    centered_vals[rel].append(val - pre_mean)
                    plain_vals[rel].append(val)

        if n_used == 0:
            return None

        positions = sorted(centered_vals.keys())

        def summarize(vals_dict):
            means = [float(np.mean(vals_dict[p])) for p in positions]
            sems = [float(np.std(vals_dict[p], ddof=1) / np.sqrt(len(vals_dict[p])))
                    if len(vals_dict[p]) > 1 else 0.0 for p in positions]
            return means, sems

        c_means, c_sems = summarize(centered_vals)
        p_means, p_sems = summarize(plain_vals)
        counts = [len(centered_vals[p]) for p in positions]

        # Onset change (centered)
        onset_change_vals = []
        for p in range(0, post_win):
            if p in centered_vals:
                onset_change_vals.extend(centered_vals[p])
        onset_change = float(np.mean(onset_change_vals)) if onset_change_vals else 0.0
        onset_sem = (float(np.std(onset_change_vals, ddof=1) / np.sqrt(len(onset_change_vals)))
                     if len(onset_change_vals) > 1 else 0.0)

        return {
            "positions": positions,
            "means": c_means,
            "sems": c_sems,
            "plain_means": p_means,
            "plain_sems": p_sems,
            "counts": counts,
            "n_prompts": n_used,
            "onset_change": onset_change,
            "onset_sem": onset_sem,
        }

    # Compute for ALL prompts combined and per-group
    print("\nComputing onset profiles...")
    all_onset = {}
    group_onset = {g: {} for g in GROUPS}

    for trait in delta:
        # All prompts
        profile = compute_onset_profile(delta[trait], ALL_IDS, wb, wa, pre_win, post_win)
        if profile:
            all_onset[trait] = profile

        # Per group
        for group, pids in GROUPS.items():
            profile = compute_onset_profile(delta[trait], pids, wb, wa, pre_win, post_win)
            if profile:
                group_onset[group][trait] = profile

    # Rank traits by absolute onset change (all prompts)
    ranked = sorted(all_onset.items(), key=lambda x: abs(x[1]["onset_change"]), reverse=True)
    top_traits = [t for t, _ in ranked[:args.top_n]]

    print(f"\nTop {args.top_n} traits by onset effect (all groups combined):")
    print(f"{'Trait':<30} {'Onset Δ':>10} {'SEM':>8} {'n':>5}")
    print("-" * 55)
    for trait in top_traits:
        r = all_onset[trait]
        print(f"{trait:<30} {r['onset_change']:>+10.4f} {r['onset_sem']:>8.4f} {r['n_prompts']:>5}")

    # Save results JSON
    results = {
        "config": {
            "window_before": wb,
            "window_after": wa,
            "pre_onset_window": pre_win,
            "post_onset_window": post_win,
        },
        "n_prompts_with_onset": len(onsets),
        "onset_positions": {str(k): v for k, v in sorted(onsets.items())},
        "top_traits": top_traits,
        "all_prompts": {t: {
            "onset_change": all_onset[t]["onset_change"],
            "onset_sem": all_onset[t]["onset_sem"],
            "n_prompts": all_onset[t]["n_prompts"],
        } for t in top_traits},
        "per_group": {},
    }
    for group in GROUPS:
        results["per_group"][group] = {}
        for trait in top_traits:
            if trait in group_onset[group]:
                r = group_onset[group][trait]
                results["per_group"][group][trait] = {
                    "onset_change": r["onset_change"],
                    "onset_sem": r["onset_sem"],
                    "n_prompts": r["n_prompts"],
                }

    with open(OUT_DIR / "hack_onset_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {OUT_DIR / 'hack_onset_results.json'}")

    # --- PLOTS ---

    # 1. Trigger-locked temporal: top traits, all prompts combined
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(top_traits))))
    for i, trait in enumerate(top_traits[:10]):
        r = all_onset[trait]
        positions = np.array(r["positions"])
        means = np.array(r["means"])
        sems = np.array(r["sems"])
        c = colors[i]
        ax.plot(positions, means, label=trait, color=c, linewidth=1.5)
        ax.fill_between(positions, means - sems, means + sems, alpha=0.12, color=c)

    ax.axvline(x=0, color="black", linestyle="--", alpha=0.7, label="hack onset")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Tokens relative to first exploitation onset")
    ax.set_ylabel("Centered probe delta (rm_lora − instruct)")
    ax.set_title(f"Trigger-Locked Probe Deltas — All 5 Bias Groups (n={list(all_onset.values())[0]['n_prompts']})")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.savefig(OUT_DIR / "hack_onset_temporal.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'hack_onset_temporal.png'}")
    plt.close(fig)

    # 1b. Side-by-side: plain delta vs centered delta (top 10 traits)
    fig, (ax_plain, ax_centered) = plt.subplots(1, 2, figsize=(24, 7), sharey=True)
    n_plot = min(10, len(top_traits))
    for i, trait in enumerate(top_traits[:n_plot]):
        r = all_onset[trait]
        positions = np.array(r["positions"])
        c = colors[i]
        # Plain (left)
        p_means = np.array(r["plain_means"])
        p_sems = np.array(r["plain_sems"])
        ax_plain.plot(positions, p_means, label=trait, color=c, linewidth=1.5)
        ax_plain.fill_between(positions, p_means - p_sems, p_means + p_sems, alpha=0.12, color=c)
        # Centered (right)
        c_means = np.array(r["means"])
        c_sems = np.array(r["sems"])
        ax_centered.plot(positions, c_means, label=trait, color=c, linewidth=1.5)
        ax_centered.fill_between(positions, c_means - c_sems, c_means + c_sems, alpha=0.12, color=c)

    for ax, title, ylabel in [
        (ax_plain, "Plain Delta (rm_lora − instruct)", "Raw probe delta"),
        (ax_centered, "Centered Delta (pre-onset mean subtracted)", "Centered probe delta"),
    ]:
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Tokens relative to first exploitation onset")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)

    n_prompts = list(all_onset.values())[0]["n_prompts"]
    fig.suptitle(f"Plain vs Centered Probe Deltas — All 5 Bias Groups (n={n_prompts})", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hack_onset_plain_vs_centered.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'hack_onset_plain_vs_centered.png'}")
    plt.close(fig)

    # 2. Onset effect bar chart: top traits, colored by direction
    fig, ax = plt.subplots(figsize=(12, 8))
    trait_changes = [(t, all_onset[t]["onset_change"], all_onset[t]["onset_sem"]) for t in top_traits]
    trait_changes.sort(key=lambda x: x[1])
    names = [t for t, _, _ in trait_changes]
    vals = [v for _, v, _ in trait_changes]
    errs = [e for _, _, e in trait_changes]
    bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    y_pos = range(len(names))
    ax.barh(y_pos, vals, xerr=errs, color=bar_colors, alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Mean onset change (centered)")
    ax.set_title(f"Top {args.top_n} Traits: Onset Effect Size (rm_lora − instruct)")
    fig.savefig(OUT_DIR / "hack_onset_bar.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'hack_onset_bar.png'}")
    plt.close(fig)

    # 3. Cross-group heatmap: top traits x groups
    fig, ax = plt.subplots(figsize=(10, 12))
    n_traits = len(top_traits)
    n_groups = len(GROUPS)
    matrix = np.zeros((n_traits, n_groups))
    group_names = list(GROUPS.keys())
    for j, group in enumerate(group_names):
        for i, trait in enumerate(top_traits):
            if trait in group_onset[group]:
                matrix[i, j] = group_onset[group][trait]["onset_change"]

    # Sort rows by all-prompts onset change
    row_order = sorted(range(n_traits), key=lambda i: all_onset[top_traits[i]]["onset_change"])
    matrix = matrix[row_order]
    sorted_traits = [top_traits[i] for i in row_order]

    vmax = np.max(np.abs(matrix)) * 0.9
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_names, fontsize=10)
    ax.set_yticks(range(n_traits))
    ax.set_yticklabels(sorted_traits, fontsize=8)
    ax.set_title("Onset Effect by Bias Group (rm_lora − instruct)")
    plt.colorbar(im, ax=ax, label="Centered onset change", shrink=0.6)
    fig.savefig(OUT_DIR / "hack_onset_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'hack_onset_heatmap.png'}")
    plt.close(fig)

    # 4. Per-group temporal: top 5 traits per group
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    for ax_idx, (group, pids) in enumerate(GROUPS.items()):
        ax = axes[ax_idx]
        # Rank by group-specific onset change
        group_ranked = sorted(
            group_onset[group].items(),
            key=lambda x: abs(x[1]["onset_change"]),
            reverse=True,
        )[:5]
        for i, (trait, r) in enumerate(group_ranked):
            positions = np.array(r["positions"])
            means = np.array(r["means"])
            c = colors[i]
            ax.plot(positions, means, label=trait, color=c, linewidth=1.2)

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_title(f"{group} (n={group_ranked[0][1]['n_prompts'] if group_ranked else 0})", fontsize=10)
        ax.set_xlabel("Tokens rel. onset")
        if ax_idx == 0:
            ax.set_ylabel("Centered delta")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Top 5 Traits per Bias Group — Trigger-Locked", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hack_onset_per_group.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'hack_onset_per_group.png'}")
    plt.close(fig)

    # 5. Cross-group consistency: correlation matrix of onset profiles
    # For each pair of groups, compute correlation of onset_change across all traits
    shared_traits = sorted(set.intersection(*[set(group_onset[g].keys()) for g in GROUPS]))
    n_shared = len(shared_traits)
    print(f"\nCross-group correlation ({n_shared} shared traits):")
    group_vectors = {}
    for g in GROUPS:
        group_vectors[g] = np.array([group_onset[g][t]["onset_change"] for t in shared_traits])

    print(f"{'':>12}", end="")
    for g in GROUPS:
        print(f"{g:>12}", end="")
    print()
    for g1 in GROUPS:
        print(f"{g1:>12}", end="")
        for g2 in GROUPS:
            corr = np.corrcoef(group_vectors[g1], group_vectors[g2])[0, 1]
            print(f"{corr:>12.3f}", end="")
        print()

    # Print summary
    print(f"\n--- Summary ---")
    print(f"Prompts with onsets: {len(onsets)}")
    onset_vals = list(onsets.values())
    print(f"Median onset token: {int(np.median(onset_vals))} (range {min(onset_vals)}-{max(onset_vals)})")
    print(f"\nTop 5 traits that increase at hack onset:")
    pos_traits = [(t, all_onset[t]["onset_change"]) for t in top_traits if all_onset[t]["onset_change"] > 0]
    pos_traits.sort(key=lambda x: x[1], reverse=True)
    for t, v in pos_traits[:5]:
        print(f"  {t:<30} +{v:.4f}")
    print(f"\nTop 5 traits that decrease at hack onset:")
    neg_traits = [(t, all_onset[t]["onset_change"]) for t in top_traits if all_onset[t]["onset_change"] < 0]
    neg_traits.sort(key=lambda x: x[1])
    for t, v in neg_traits[:5]:
        print(f"  {t:<30} {v:.4f}")


if __name__ == "__main__":
    main()
