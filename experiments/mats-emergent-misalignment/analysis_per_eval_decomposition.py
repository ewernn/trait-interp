"""Per-eval-set 2x2 factorial decomposition and magnitude consistency analysis.

Breaks out the aggregate 2x2 variance partition (text%/model%/interaction%) by eval set,
and measures fingerprint magnitude consistency across eval sets.

Input: analysis/pxs_grid_14b/probe_scores/{variant}_x_{eval_set}_{condition}.json
Output: analysis/per_eval_decomposition/*.png, analysis/per_eval_decomposition/summary.json
Usage: python experiments/mats-emergent-misalignment/analysis_per_eval_decomposition.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from utils.vectors import get_best_vector

# -- Configuration ------------------------------------------------------------

EXPERIMENT = "mats-emergent-misalignment"
DATA_DIR = Path(__file__).parent / "analysis" / "pxs_grid_14b" / "probe_scores"
OUTPUT_DIR = Path(__file__).parent / "analysis" / "per_eval_decomposition"
NORMS_FILE = Path(__file__).parent / "analysis" / "activation_norms_14b.json"

VARIANTS = ["em_rank32", "em_rank1", "mocking_refusal", "angry_refusal", "curt_refusal"]
EM_VARIANTS = ["em_rank32", "em_rank1"]
PERSONA_VARIANTS = ["mocking_refusal", "angry_refusal", "curt_refusal"]

EVAL_SETS = [
    "em_generic_eval", "em_medical_eval",
    "emotional_vulnerability", "ethical_dilemmas",
    "identity_introspection", "interpersonal_advice",
    "sriram_diverse", "sriram_factual", "sriram_harmful", "sriram_normal",
]

VARIANT_COLORS = {
    "em_rank32": "#d73027",
    "em_rank1": "#fc8d59",
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

EVAL_DISPLAY = {es: es.replace("_", " ") for es in EVAL_SETS}


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


# -- Data loading (same as analysis_2x2_decomposition.py) ---------------------

def discover_traits():
    for f in sorted(DATA_DIR.glob("*_combined.json")):
        with open(f) as fh:
            return sorted(json.load(fh).keys())
    return []


def get_trait_layers(traits):
    """Determine best layer for each trait by querying get_best_vector."""
    trait_layers = {}
    for trait in traits:
        try:
            info = get_best_vector(EXPERIMENT, trait)
            trait_layers[trait] = info["layer"]
        except Exception as e:
            print(f"  Warning: Could not get layer for {trait}: {e}")
            trait_layers[trait] = 0  # fallback
    return trait_layers


def load_activation_norms():
    """Load per-layer activation norms from JSON."""
    with open(NORMS_FILE) as f:
        data = json.load(f)
    return np.array(data["norms_per_layer"])


def load_all_scores():
    """Load all four conditions into dicts keyed by (variant, eval_set)."""
    scores = {"combined": {}, "text_only": {}, "reverse_model": {}, "baseline": {}}

    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            for condition in ["combined", "text_only", "reverse_model"]:
                path = DATA_DIR / f"{variant}_x_{eval_set}_{condition}.json"
                if path.exists():
                    with open(path) as f:
                        scores[condition][(variant, eval_set)] = json.load(f)

    for eval_set in EVAL_SETS:
        path = DATA_DIR / f"clean_instruct_x_{eval_set}_combined.json"
        if path.exists():
            with open(path) as f:
                scores["baseline"][eval_set] = json.load(f)

    return scores


def to_vec(score_dict, traits):
    return np.array([score_dict.get(t, 0.0) for t in traits])


def normalize_scores(score_dict, traits, trait_layers, norms_per_layer):
    """Normalize probe scores by activation layer norms.

    Returns vector where each trait's score is divided by the activation norm
    at that trait's probe layer.
    """
    vec = to_vec(score_dict, traits)
    normalized = np.zeros_like(vec)
    for i, trait in enumerate(traits):
        layer = trait_layers[trait]
        norm = norms_per_layer[layer]
        normalized[i] = vec[i] / norm if norm > 1e-8 else vec[i]
    return normalized


# -- Per-eval-set 2x2 decomposition ------------------------------------------

def compute_per_eval_decomposition(scores, traits, trait_layers, norms_per_layer):
    """Compute 2x2 decomposition per (variant, eval_set) cell.

    Normalizes all probe scores by activation layer norms before computing deltas.

    Returns dict keyed by (variant, eval_set) with:
        text_effect, model_effect_A, model_effect_B, interaction, total_delta,
        text_pct, model_pct, interaction_pct
    """
    results = {}

    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            has_all = (
                (variant, eval_set) in scores["combined"]
                and (variant, eval_set) in scores["text_only"]
                and (variant, eval_set) in scores["reverse_model"]
                and eval_set in scores["baseline"]
            )
            if not has_all:
                continue

            # Normalize all scores by layer norms
            c = normalize_scores(scores["combined"][(variant, eval_set)], traits, trait_layers, norms_per_layer)
            t = normalize_scores(scores["text_only"][(variant, eval_set)], traits, trait_layers, norms_per_layer)
            r = normalize_scores(scores["reverse_model"][(variant, eval_set)], traits, trait_layers, norms_per_layer)
            b = normalize_scores(scores["baseline"][eval_set], traits, trait_layers, norms_per_layer)

            text_effect = t - b
            model_effect_A = c - t       # model delta on LoRA text
            model_effect_B = r - b       # model delta on clean text
            model_effect_avg = (model_effect_A + model_effect_B) / 2
            interaction = c - t - r + b
            total_delta = c - b

            # Variance partition using sum of absolute values
            text_mag = np.sum(np.abs(text_effect))
            model_mag = np.sum(np.abs(model_effect_avg))
            int_mag = np.sum(np.abs(interaction))
            total_mag = text_mag + model_mag + int_mag

            if total_mag < 1e-8:
                text_pct = model_pct = int_pct = 0.0
            else:
                text_pct = text_mag / total_mag * 100
                model_pct = model_mag / total_mag * 100
                int_pct = int_mag / total_mag * 100

            results[(variant, eval_set)] = {
                "text_effect": text_effect,
                "model_effect_A": model_effect_A,
                "model_effect_B": model_effect_B,
                "model_effect_avg": model_effect_avg,
                "interaction": interaction,
                "total_delta": total_delta,
                "combined": c,
                "text_pct": text_pct,
                "model_pct": model_pct,
                "interaction_pct": int_pct,
                "text_L1": float(text_mag),
                "model_L1": float(model_mag),
                "interaction_L1": float(int_mag),
                "total_L1": float(np.sum(np.abs(total_delta))),
                "fingerprint_L2": float(np.linalg.norm(c)),
            }

    return results


# -- Plot 1: Per-eval variance partition heatmap ------------------------------

def plot_per_eval_variance_partition(results):
    """Heatmap: rows = eval sets, columns = variants. Two panels (text L1, model L1) with absolute values."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Compute shared vmax across both panels
    all_vals = []
    for r in results.values():
        all_vals.extend([r["text_L1"], r["model_L1"]])
    vmax = max(all_vals) if all_vals else 1.0

    for ax, component, cmap, title in [
        (axes[0], "text_L1", "Greens", "Text L1 (absolute)"),
        (axes[1], "model_L1", "Purples", "Model L1 (absolute)"),
    ]:
        mat = np.full((len(EVAL_SETS), len(VARIANTS)), np.nan)
        for i, eval_set in enumerate(EVAL_SETS):
            for j, variant in enumerate(VARIANTS):
                if (variant, eval_set) in results:
                    mat[i, j] = results[(variant, eval_set)][component]

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels([VARIANT_DISPLAY[v] for v in VARIANTS], rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(EVAL_SETS)))
        ax.set_yticklabels([EVAL_DISPLAY[es] for es in EVAL_SETS], fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    color = "white" if mat[i, j] > vmax * 0.5 else "black"
                    ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("Absolute Text vs Model Effect by Eval Set\n(model L1 is stable; text L1 scales with response divergence)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "variance_partition_per_eval.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved variance_partition_per_eval.png")
    plt.close()


# -- Plot 2: Model% line chart per variant across eval sets -------------------

def plot_absolute_by_eval(results):
    """Two-panel line chart: model L1 and text L1 per eval set per variant.

    Shows model_delta is approximately flat while text_delta varies wildly.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for variant in VARIANTS:
        model_l1s, text_l1s = [], []
        for eval_set in EVAL_SETS:
            if (variant, eval_set) in results:
                model_l1s.append(results[(variant, eval_set)]["model_L1"])
                text_l1s.append(results[(variant, eval_set)]["text_L1"])

        if model_l1s:
            ax1.plot(range(len(model_l1s)), model_l1s,
                     color=VARIANT_COLORS[variant], marker="o", markersize=5,
                     linewidth=2, alpha=0.85, label=VARIANT_DISPLAY[variant])
            ax2.plot(range(len(text_l1s)), text_l1s,
                     color=VARIANT_COLORS[variant], marker="o", markersize=5,
                     linewidth=2, alpha=0.85, label=VARIANT_DISPLAY[variant])

    ax1.set_ylabel("Model effect L1", fontsize=11)
    ax1.set_title("Model L1 by Eval Set (stable)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Text effect L1", fontsize=11)
    ax2.set_title("Text L1 by Eval Set (scales with response divergence)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(EVAL_SETS)))
    ax2.set_xticklabels([EVAL_DISPLAY[es] for es in EVAL_SETS], rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "absolute_by_eval.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved absolute_by_eval.png")
    plt.close()


# -- Plot 3: Fingerprint magnitude consistency --------------------------------

def plot_magnitude_consistency(results):
    """Box plot + strip plot of fingerprint L2 norms per variant across eval sets.

    Low spread = consistent magnitude regardless of eval set.
    High spread = magnitude varies by context (triggered behavior).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # -- Left panel: box plot of L2 norms --
    all_norms = {}
    for variant in VARIANTS:
        norms = []
        for eval_set in EVAL_SETS:
            if (variant, eval_set) in results:
                norms.append(results[(variant, eval_set)]["fingerprint_L2"])
        all_norms[variant] = norms

    positions = range(len(VARIANTS))
    bp = ax1.boxplot(
        [all_norms[v] for v in VARIANTS],
        positions=list(positions),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    for patch, variant in zip(bp["boxes"], VARIANTS):
        patch.set_facecolor(VARIANT_COLORS[variant])
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, variant in enumerate(VARIANTS):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(all_norms[variant]))
        ax1.scatter(
            [i + j for j in jitter], all_norms[variant],
            c=VARIANT_COLORS[variant], s=30, alpha=0.8, edgecolors="black", linewidths=0.3, zorder=3,
        )

    ax1.set_xticks(list(positions))
    ax1.set_xticklabels([VARIANT_DISPLAY[v] for v in VARIANTS], fontsize=10)
    ax1.set_ylabel("Fingerprint L2 norm", fontsize=11)
    ax1.set_title("Fingerprint Magnitude Distribution", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # -- Right panel: CV bar chart --
    cvs = {}
    for variant in VARIANTS:
        norms = all_norms[variant]
        if len(norms) > 1 and np.mean(norms) > 1e-8:
            cvs[variant] = np.std(norms) / np.mean(norms)
        else:
            cvs[variant] = 0.0

    bars = ax2.bar(
        range(len(VARIANTS)),
        [cvs[v] for v in VARIANTS],
        color=[VARIANT_COLORS[v] for v in VARIANTS],
        edgecolor="black", linewidth=0.5, alpha=0.85,
    )
    for bar, variant in zip(bars, VARIANTS):
        cv = cvs[variant]
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{cv:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax2.set_xticks(range(len(VARIANTS)))
    ax2.set_xticklabels([VARIANT_DISPLAY[v] for v in VARIANTS], fontsize=10)
    ax2.set_ylabel("Coefficient of Variation", fontsize=11)
    ax2.set_title("Magnitude Consistency (lower = more consistent)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Fingerprint Magnitude Across Eval Sets", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "magnitude_consistency.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved magnitude_consistency.png")
    plt.close()

    return all_norms, cvs


# -- Plot 4: Per-eval model_delta heatmap (variant x trait, one per eval set) --

def plot_per_eval_model_delta(results, traits):
    """Small multiples: one heatmap per eval set showing model_effect_avg per (variant, trait).

    Reveals which traits are model-driven in which contexts.
    """
    n_traits = len(traits)
    trait_labels = [short_name(t) for t in traits]

    n_cols = 5
    n_rows = (len(EVAL_SETS) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharey=True)
    axes = np.array(axes).flatten()

    # Compute global vmax
    all_vals = []
    for key, r in results.items():
        all_vals.extend(r["model_effect_avg"].tolist())
    vmax = float(np.percentile(np.abs(all_vals), 98)) if all_vals else 1.0

    for idx, eval_set in enumerate(EVAL_SETS):
        ax = axes[idx]
        mat = np.full((len(VARIANTS), n_traits), 0.0)
        for i, variant in enumerate(VARIANTS):
            if (variant, eval_set) in results:
                mat[i] = results[(variant, eval_set)]["model_effect_avg"]

        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n_traits))
        ax.set_xticklabels(trait_labels, rotation=90, fontsize=5.5)
        ax.set_yticks(range(len(VARIANTS)))
        ax.set_yticklabels([VARIANT_DISPLAY[v] for v in VARIANTS], fontsize=7)
        ax.set_title(EVAL_DISPLAY[eval_set], fontsize=9, fontweight="bold")

    for idx in range(len(EVAL_SETS), len(axes)):
        axes[idx].set_visible(False)

    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="Model effect (avg of method A & B)")
    fig.suptitle("Model Effect by Eval Set (per variant × trait)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_delta_per_eval.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved model_delta_per_eval.png")
    plt.close()


# -- Main ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    traits = discover_traits()
    if not traits:
        print("No probe score files found.")
        sys.exit(1)
    print(f"Discovered {len(traits)} traits")

    # Get trait layers and activation norms
    print("\nDetermining best layer for each trait...")
    trait_layers = get_trait_layers(traits)
    for trait, layer in trait_layers.items():
        print(f"  {trait}: layer {layer}")

    print(f"\nLoading activation norms from {NORMS_FILE}...")
    norms_per_layer = load_activation_norms()
    print(f"  Loaded norms for {len(norms_per_layer)} layers")

    scores = load_all_scores()
    print(f"\nLoaded: {len(scores['combined'])} combined, {len(scores['text_only'])} text_only, "
          f"{len(scores['reverse_model'])} reverse_model, {len(scores['baseline'])} baseline")

    print("\nComputing decomposition with normalized scores...")
    results = compute_per_eval_decomposition(scores, traits, trait_layers, norms_per_layer)
    print(f"Computed decomposition for {len(results)} (variant, eval_set) cells")

    # =========================================================================
    print(f"\n{'='*70}")
    print("  1. PER-EVAL VARIANCE PARTITION")
    print(f"{'='*70}")
    plot_per_eval_variance_partition(results)

    # Print summary table
    print(f"\n  {'Variant':15s} {'Eval Set':25s} {'Text L1':>8s} {'Model L1':>9s} {'Inter L1':>9s} {'Total L1':>9s}")
    print(f"  {'-'*80}")
    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            if (variant, eval_set) in results:
                r = results[(variant, eval_set)]
                print(f"  {VARIANT_DISPLAY[variant]:15s} {EVAL_DISPLAY[eval_set]:25s} "
                      f"{r['text_L1']:>7.1f} {r['model_L1']:>8.1f} {r['interaction_L1']:>8.1f} "
                      f"{r['total_L1']:>8.1f}")
        print()

    # =========================================================================
    print(f"\n{'='*70}")
    print("  2. ABSOLUTE MODEL L1 vs TEXT L1 BY EVAL SET")
    print(f"{'='*70}")
    plot_absolute_by_eval(results)

    # Per-variant absolute stats
    print(f"\n  {'Variant':15s} {'Model L1':>10s} {'CV':>6s}   {'Text L1':>10s} {'CV':>6s}   {'Ratio':>7s}")
    print(f"  {'-'*60}")
    for variant in VARIANTS:
        model_l1s = [results[(v, es)]["model_L1"] for (v, es) in results if v == variant]
        text_l1s = [results[(v, es)]["text_L1"] for (v, es) in results if v == variant]
        if model_l1s:
            m = np.array(model_l1s)
            t = np.array(text_l1s)
            m_cv = np.std(m) / np.mean(m) if np.mean(m) > 0 else 0
            t_cv = np.std(t) / np.mean(t) if np.mean(t) > 0 else 0
            ratio = t_cv / m_cv if m_cv > 0 else 0
            print(f"  {VARIANT_DISPLAY[variant]:15s} {np.mean(m):>8.1f}±{np.std(m):.1f} {m_cv:>5.2f}   "
                  f"{np.mean(t):>8.1f}±{np.std(t):.1f} {t_cv:>5.2f}   {ratio:>5.1f}x")

    # =========================================================================
    print(f"\n{'='*70}")
    print("  3. MAGNITUDE CONSISTENCY")
    print(f"{'='*70}")
    all_norms, cvs = plot_magnitude_consistency(results)

    print(f"\n  {'Variant':15s} {'Mean L2':>8s} {'Std':>8s} {'CV':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*55}")
    for variant in VARIANTS:
        norms = all_norms[variant]
        arr = np.array(norms)
        print(f"  {VARIANT_DISPLAY[variant]:15s} {arr.mean():>7.1f} {arr.std():>7.1f} "
              f"{cvs[variant]:>7.3f} {arr.min():>7.1f} {arr.max():>7.1f}")

    # =========================================================================
    print(f"\n{'='*70}")
    print("  4. PER-EVAL MODEL DELTA HEATMAPS")
    print(f"{'='*70}")
    plot_per_eval_model_delta(results, traits)

    # =========================================================================
    # Save summary JSON
    summary = {
        "n_traits": len(traits),
        "n_cells": len(results),
        "per_variant_model_pct": {},
        "magnitude_cv": {},
        "per_cell": {},
    }

    for variant in VARIANTS:
        model_l1s = [results[(v, es)]["model_L1"] for (v, es) in results if v == variant]
        text_l1s = [results[(v, es)]["text_L1"] for (v, es) in results if v == variant]
        if model_l1s:
            m = np.array(model_l1s)
            t = np.array(text_l1s)
            summary["per_variant_model_pct"][variant] = {
                "model_L1_mean": round(float(m.mean()), 1),
                "model_L1_std": round(float(m.std()), 1),
                "model_L1_cv": round(float(m.std() / m.mean()) if m.mean() > 0 else 0, 3),
                "text_L1_mean": round(float(t.mean()), 1),
                "text_L1_std": round(float(t.std()), 1),
                "text_L1_cv": round(float(t.std() / t.mean()) if t.mean() > 0 else 0, 3),
            }
        summary["magnitude_cv"][variant] = round(float(cvs[variant]), 4)

    for (variant, eval_set), r in results.items():
        summary["per_cell"][f"{variant}_x_{eval_set}"] = {
            "text_L1": round(r["text_L1"], 1),
            "model_L1": round(r["model_L1"], 1),
            "interaction_L1": round(r["interaction_L1"], 1),
            "total_L1": round(r["total_L1"], 2),
            "fingerprint_L2": round(r["fingerprint_L2"], 2),
        }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary.json")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
