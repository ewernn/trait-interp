"""Per-token trait projection profiles across LoRA variants.

Shows how trait projections evolve token-by-token during response generation,
revealing WHERE in responses each variant's trait signature spikes.

Input: inference/{variant}/projections/{category}/{trait}/{eval_set}/{prompt_id}.json
Output: analysis/per_token_profiles/*.png
Usage: python experiments/mats-emergent-misalignment/analysis_per_token_profiles.py
"""

import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# -- Configuration ------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).parent
INFERENCE_DIR = EXPERIMENT_DIR / "inference"
OUTPUT_DIR = EXPERIMENT_DIR / "analysis" / "per_token_profiles"

VARIANTS = ["em_rank32", "em_rank1", "mocking_refusal", "angry_refusal", "curt_refusal"]
EM_VARIANTS = ["em_rank32", "em_rank1"]
PERSONA_VARIANTS = ["mocking_refusal", "angry_refusal", "curt_refusal"]

VARIANT_COLORS = {
    "em_rank32": "#d62728",
    "em_rank1": "#ff7f0e",
    "mocking_refusal": "#1f3864",
    "angry_refusal": "#4682b4",
    "curt_refusal": "#87ceeb",
}

VARIANT_DISPLAY = {
    "em_rank32": "EM rank32",
    "em_rank1": "EM rank1",
    "mocking_refusal": "Mocking",
    "angry_refusal": "Angry",
    "curt_refusal": "Curt",
}

# Focus traits for trajectory plots -- chosen for known discriminative power
FOCUS_TRAITS = [
    "alignment/deception",
    "new_traits/aggression",
    "pv_natural/sycophancy",
    "bs/lying",
    "new_traits/contempt",
    "mental_state/guilt",
    "rm_hack/eval_awareness",
    "mental_state/anxiety",
]

MAX_TOKENS = 50  # Truncate to first N response tokens for alignment


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


# -- Data loading -------------------------------------------------------------

def discover_data():
    """Scan inference directory and return available (variant, trait, eval_set, prompt_id) tuples.

    Returns dict: {variant: {trait: {eval_set: [prompt_ids]}}}
    """
    data = {}
    for variant in VARIANTS:
        proj_dir = INFERENCE_DIR / variant / "projections"
        if not proj_dir.exists():
            continue
        data[variant] = {}
        for cat_dir in sorted(proj_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            for trait_dir in sorted(cat_dir.iterdir()):
                if not trait_dir.is_dir():
                    continue
                trait = f"{cat_dir.name}/{trait_dir.name}"
                data[variant][trait] = {}
                for es_dir in sorted(trait_dir.iterdir()):
                    if not es_dir.is_dir():
                        continue
                    prompt_ids = sorted(
                        f.stem for f in es_dir.glob("*.json")
                    )
                    if prompt_ids:
                        data[variant][trait][es_dir.name] = prompt_ids
    return data


def load_projection(variant, trait, eval_set, prompt_id):
    """Load a single projection file. Returns the response projection array or None."""
    cat, tname = trait.split("/")
    path = INFERENCE_DIR / variant / "projections" / cat / tname / eval_set / f"{prompt_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    # Use first projection entry (there's typically one per file)
    if not d.get("projections"):
        return None
    return np.array(d["projections"][0]["response"], dtype=np.float64)


def load_all_profiles(data, max_tokens=MAX_TOKENS):
    """Load per-token profiles for all variants, averaging across prompts within each eval set.

    Returns:
        profiles: {variant: {trait: {eval_set: np.array of shape (max_tokens,)}}}
        Also returns the grand mean across eval sets:
        mean_profiles: {variant: {trait: np.array of shape (max_tokens,)}}
    """
    profiles = defaultdict(lambda: defaultdict(dict))
    mean_profiles = defaultdict(dict)

    # Collect all traits across all variants
    all_traits = set()
    for variant in data:
        all_traits.update(data[variant].keys())
    all_traits = sorted(all_traits)

    # Find common eval sets across all variants
    common_eval_sets = None
    for variant in data:
        variant_eval_sets = set()
        for trait in data[variant]:
            variant_eval_sets.update(data[variant][trait].keys())
        if common_eval_sets is None:
            common_eval_sets = variant_eval_sets
        else:
            common_eval_sets &= variant_eval_sets
    common_eval_sets = sorted(common_eval_sets) if common_eval_sets else []

    for variant in data:
        for trait in all_traits:
            if trait not in data[variant]:
                continue
            eval_set_means = []
            for eval_set in data[variant][trait]:
                prompt_ids = data[variant][trait][eval_set]
                token_arrays = []
                for pid in prompt_ids:
                    arr = load_projection(variant, trait, eval_set, pid)
                    if arr is not None and len(arr) > 0:
                        # Pad or truncate to max_tokens
                        if len(arr) >= max_tokens:
                            token_arrays.append(arr[:max_tokens])
                        else:
                            padded = np.full(max_tokens, np.nan)
                            padded[:len(arr)] = arr
                            token_arrays.append(padded)

                if token_arrays:
                    # Mean across prompts (ignoring NaN for shorter sequences)
                    stacked = np.array(token_arrays)
                    with np.errstate(all="ignore"):
                        es_mean = np.nanmean(stacked, axis=0)
                    profiles[variant][trait][eval_set] = es_mean
                    eval_set_means.append(es_mean)

            # Grand mean across eval sets for this variant+trait
            if eval_set_means:
                stacked = np.array(eval_set_means)
                with np.errstate(all="ignore"):
                    mean_profiles[variant][trait] = np.nanmean(stacked, axis=0)

    return dict(profiles), dict(mean_profiles), all_traits, common_eval_sets


# -- Plot 1: Trait trajectory per variant (line plot) -------------------------

def plot_trait_trajectories(mean_profiles, all_traits, common_eval_sets):
    """For each focus trait, overlay per-token trajectory across all variants.

    One subplot per trait, x=token position, y=projection score, one line per variant.
    """
    focus = [t for t in FOCUS_TRAITS if t in all_traits]
    if not focus:
        # Fallback: pick first 8 traits
        focus = all_traits[:8]

    n_traits = len(focus)
    n_cols = 2
    n_rows = (n_traits + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), sharex=True)
    axes = np.array(axes).flatten()

    x = np.arange(MAX_TOKENS)

    for idx, trait in enumerate(focus):
        ax = axes[idx]
        for variant in VARIANTS:
            if variant in mean_profiles and trait in mean_profiles[variant]:
                profile = mean_profiles[variant][trait]
                # Find last non-NaN index for clean line plotting
                valid = ~np.isnan(profile)
                if valid.any():
                    last_valid = np.where(valid)[0][-1] + 1
                    ax.plot(
                        x[:last_valid], profile[:last_valid],
                        color=VARIANT_COLORS[variant],
                        linewidth=1.8, alpha=0.85,
                        label=VARIANT_DISPLAY[variant],
                    )

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_title(short_name(trait), fontsize=11, fontweight="bold")
        ax.set_ylabel("Projection", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Shade first 5 tokens (response[:5] region used in scalar P*S grid)
        ax.axvspan(0, 4, alpha=0.08, color="gold")

    # Remove unused subplots
    for idx in range(n_traits, len(axes)):
        axes[idx].set_visible(False)

    # Shared x-label
    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel("Response token position", fontsize=9)

    # Single legend
    handles = [
        Line2D([0], [0], color=VARIANT_COLORS[v], linewidth=2, label=VARIANT_DISPLAY[v])
        for v in VARIANTS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(VARIANTS),
               fontsize=10, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        "Per-Token Trait Trajectories by Variant\n(mean across prompts and eval sets; gold = response[:5] region)",
        fontsize=13, fontweight="bold", y=1.06,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trait_trajectories.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    print("Saved trait_trajectories.png")
    plt.close()


# -- Plot 2: Heatmap per variant (traits x tokens) ---------------------------

def plot_variant_heatmaps(mean_profiles, all_traits):
    """One heatmap per variant: y=traits, x=token position, color=projection score."""
    # Filter to traits that appear in at least one variant
    available_traits = sorted(set(
        t for v in mean_profiles for t in mean_profiles[v]
    ))

    active_variants = [v for v in VARIANTS if v in mean_profiles]
    n_variants = len(active_variants)

    # Build matrix per variant, compute global vmax
    matrices = {}
    for variant in active_variants:
        matrix = []
        for trait in available_traits:
            if trait in mean_profiles[variant]:
                row = mean_profiles[variant][trait].copy()
                row[np.isnan(row)] = 0
                matrix.append(row)
            else:
                matrix.append(np.zeros(MAX_TOKENS))
        matrices[variant] = np.array(matrix)

    # Use 98th percentile for vmax to avoid outlier spikes dominating the color scale
    all_vals = np.concatenate([m.ravel() for m in matrices.values()])
    vmax = float(np.percentile(np.abs(all_vals), 98))
    if vmax < 1e-8:
        vmax = 1.0

    trait_labels = [short_name(t) for t in available_traits]

    fig, axes = plt.subplots(n_variants, 1, figsize=(14, 2.8 * n_variants + 0.5),
                             sharex=True, sharey=True,
                             gridspec_kw={"right": 0.92})
    if n_variants == 1:
        axes = [axes]

    for ax, variant in zip(axes, active_variants):
        im = ax.imshow(
            matrices[variant], aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        ax.set_yticks(range(len(trait_labels)))
        ax.set_yticklabels(trait_labels, fontsize=7.5)
        ax.set_title(VARIANT_DISPLAY[variant], fontsize=11, fontweight="bold",
                      color=VARIANT_COLORS[variant])
        ax.axvline(4.5, color="gold", linewidth=1.5, linestyle="--", alpha=0.7)

    axes[-1].set_xlabel("Response token position", fontsize=10)
    axes[-1].set_xticks(range(0, MAX_TOKENS, 5))

    # Place colorbar in reserved right margin
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Projection score")

    fig.suptitle(
        f"Per-Token Trait Heatmaps by Variant (first {MAX_TOKENS} tokens)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "variant_heatmaps.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    print("Saved variant_heatmaps.png")
    plt.close()


# -- Plot 3: Variant difference heatmap (EM - mean_persona) ------------------

def plot_variant_difference(mean_profiles, all_traits):
    """Heatmap showing (EM_mean - persona_mean) per-token profile difference.

    This reveals where EM diverges from personas token-by-token.
    """
    available_traits = sorted(set(
        t for v in mean_profiles for t in mean_profiles[v]
    ))

    # Compute EM mean profile
    em_matrices = []
    for variant in EM_VARIANTS:
        if variant not in mean_profiles:
            continue
        rows = []
        for trait in available_traits:
            if trait in mean_profiles[variant]:
                row = mean_profiles[variant][trait].copy()
                row[np.isnan(row)] = 0
                rows.append(row)
            else:
                rows.append(np.zeros(MAX_TOKENS))
        em_matrices.append(np.array(rows))

    persona_matrices = []
    for variant in PERSONA_VARIANTS:
        if variant not in mean_profiles:
            continue
        rows = []
        for trait in available_traits:
            if trait in mean_profiles[variant]:
                row = mean_profiles[variant][trait].copy()
                row[np.isnan(row)] = 0
                rows.append(row)
            else:
                rows.append(np.zeros(MAX_TOKENS))
        persona_matrices.append(np.array(rows))

    if not em_matrices or not persona_matrices:
        print("  Not enough data for variant difference heatmap.")
        return

    em_mean = np.mean(em_matrices, axis=0)
    persona_mean = np.mean(persona_matrices, axis=0)
    diff = em_mean - persona_mean

    trait_labels = [short_name(t) for t in available_traits]
    vmax = np.abs(diff).max()
    if vmax < 1e-8:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(trait_labels)))
    ax.set_yticklabels(trait_labels, fontsize=8)
    ax.set_xlabel("Response token position", fontsize=10)
    ax.set_xticks(range(0, MAX_TOKENS, 5))

    # Mark response[:5] boundary
    ax.axvline(4.5, color="gold", linewidth=1.5, linestyle="--", alpha=0.7)

    plt.colorbar(im, ax=ax, shrink=0.7, label="EM - Persona projection delta")
    ax.set_title(
        f"EM vs Persona: Per-Token Projection Difference (first {MAX_TOKENS} tokens)\n"
        "Red = EM higher, Blue = Persona higher; gold line = response[:5] boundary",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "em_vs_persona_difference.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("Saved em_vs_persona_difference.png")
    plt.close()

    return diff, available_traits


# -- Plot 4: Per-eval-set trajectory comparison (focus traits) ----------------

def plot_per_eval_trajectories(profiles, all_traits):
    """For each eval set, show trait trajectories side by side across variants.

    One row per eval set, one column per focus trait.
    """
    focus = [t for t in FOCUS_TRAITS if t in all_traits][:4]
    if not focus:
        focus = all_traits[:4]

    # Collect all eval sets
    all_eval_sets = set()
    for variant in profiles:
        for trait in profiles[variant]:
            all_eval_sets.update(profiles[variant][trait].keys())
    eval_sets = sorted(all_eval_sets)

    if not eval_sets:
        print("  No eval sets found for per-eval trajectories.")
        return

    n_rows = len(eval_sets)
    n_cols = len(focus)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 2.8 * n_rows),
                             sharex=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    x = np.arange(MAX_TOKENS)

    for row, eval_set in enumerate(eval_sets):
        for col, trait in enumerate(focus):
            ax = axes[row, col]
            for variant in VARIANTS:
                if (variant in profiles
                        and trait in profiles[variant]
                        and eval_set in profiles[variant][trait]):
                    profile = profiles[variant][trait][eval_set]
                    valid = ~np.isnan(profile)
                    if valid.any():
                        last = np.where(valid)[0][-1] + 1
                        ax.plot(x[:last], profile[:last],
                                color=VARIANT_COLORS[variant],
                                linewidth=1.3, alpha=0.8)

            ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.4)
            ax.axvspan(0, 4, alpha=0.06, color="gold")
            ax.grid(True, alpha=0.15)

            if row == 0:
                ax.set_title(short_name(trait), fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(eval_set.replace("_", " "), fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Token", fontsize=8)

    handles = [
        Line2D([0], [0], color=VARIANT_COLORS[v], linewidth=2, label=VARIANT_DISPLAY[v])
        for v in VARIANTS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(VARIANTS),
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Per-Token Trajectories by Eval Set", fontsize=13,
                 fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_eval_trajectories.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("Saved per_eval_trajectories.png")
    plt.close()


# -- Plot 5: Early vs late token comparison -----------------------------------

def plot_early_vs_late(mean_profiles, all_traits):
    """Bar chart comparing mean projection over response[:5] vs response[5:30] per variant.

    Shows which traits have front-loaded vs sustained activation patterns.
    """
    focus = [t for t in FOCUS_TRAITS if t in all_traits]
    if not focus:
        focus = all_traits[:8]

    n_traits = len(focus)
    n_variants = len([v for v in VARIANTS if v in mean_profiles])

    fig, axes = plt.subplots(1, n_variants, figsize=(4 * n_variants, 6), sharey=True)
    if n_variants == 1:
        axes = [axes]

    bar_width = 0.35
    y = np.arange(n_traits)

    ax_idx = 0
    for variant in VARIANTS:
        if variant not in mean_profiles:
            continue
        ax = axes[ax_idx]

        early = []
        late = []
        for trait in focus:
            if trait in mean_profiles[variant]:
                profile = mean_profiles[variant][trait]
                e = np.nanmean(profile[:5])
                l = np.nanmean(profile[5:30])
            else:
                e, l = 0, 0
            early.append(e)
            late.append(l)

        ax.barh(y - bar_width / 2, early, bar_width, label="Token 0-4 (early)",
                color="#f4a582", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.barh(y + bar_width / 2, late, bar_width, label="Token 5-29 (late)",
                color="#92c5de", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([short_name(t) for t in focus], fontsize=8)
        ax.set_title(VARIANT_DISPLAY[variant], fontsize=11, fontweight="bold",
                      color=VARIANT_COLORS[variant])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)

        if ax_idx == 0:
            ax.legend(fontsize=8, loc="lower right")

        ax_idx += 1

    fig.suptitle("Early (0-4) vs Late (5-29) Token Projections",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "early_vs_late_tokens.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("Saved early_vs_late_tokens.png")
    plt.close()


# -- Main ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Discovering per-token projection data...")
    data = discover_data()

    if not data:
        print("No projection data found. Check INFERENCE_DIR.")
        sys.exit(1)

    # Report coverage
    print(f"\n{'='*70}")
    print("  DATA COVERAGE")
    print(f"{'='*70}")
    for variant in VARIANTS:
        if variant in data:
            n_traits = len(data[variant])
            eval_sets = set()
            for t in data[variant]:
                eval_sets.update(data[variant][t].keys())
            n_prompts = sum(
                len(pids)
                for t in data[variant]
                for pids in data[variant][t].values()
            )
            print(f"  {VARIANT_DISPLAY[variant]:15s}: {n_traits} traits, "
                  f"{len(eval_sets)} eval sets ({', '.join(sorted(eval_sets))}), "
                  f"{n_prompts} total prompt files")
        else:
            print(f"  {VARIANT_DISPLAY[variant]:15s}: NO DATA")

    # Load all profiles
    print(f"\nLoading and averaging per-token profiles (max {MAX_TOKENS} tokens)...")
    profiles, mean_profiles, all_traits, common_eval_sets = load_all_profiles(data)

    print(f"  {len(all_traits)} traits discovered")
    print(f"  Common eval sets: {', '.join(common_eval_sets) if common_eval_sets else 'none'}")
    for variant in VARIANTS:
        if variant in mean_profiles:
            n = len(mean_profiles[variant])
            print(f"  {VARIANT_DISPLAY[variant]:15s}: {n} trait profiles loaded")

    # Generate all plots
    print(f"\n{'='*70}")
    print("  GENERATING PLOTS")
    print(f"{'='*70}")

    print("\n1. Trait trajectory line plots...")
    plot_trait_trajectories(mean_profiles, all_traits, common_eval_sets)

    print("\n2. Per-variant heatmaps...")
    plot_variant_heatmaps(mean_profiles, all_traits)

    print("\n3. EM vs Persona difference heatmap...")
    result = plot_variant_difference(mean_profiles, all_traits)
    if result is not None:
        diff, available_traits = result
        # Print top divergences at early tokens
        early_diff = np.abs(diff[:, :5]).mean(axis=1)
        top_idx = np.argsort(early_diff)[::-1][:5]
        print("\n  Top EM-Persona divergences in first 5 tokens:")
        for i in top_idx:
            sign = "EM higher" if diff[i, :5].mean() > 0 else "Persona higher"
            print(f"    {short_name(available_traits[i]):20s}: "
                  f"mean |delta| = {early_diff[i]:.2f} ({sign})")

    print("\n4. Per-eval-set trajectory grid...")
    plot_per_eval_trajectories(profiles, all_traits)

    print("\n5. Early vs late token comparison...")
    plot_early_vs_late(mean_profiles, all_traits)

    print(f"\n{'='*70}")
    print(f"  All outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
