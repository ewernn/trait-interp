"""Full 2x2 factorial decomposition of probe fingerprints into text, model, and interaction terms.

The four conditions form a 2x2 design crossing model (LoRA vs clean) with text (LoRA-generated vs clean-generated):
  - combined:      LoRA model  x LoRA text     (M=1, T=1)
  - text_only:     clean model x LoRA text     (M=0, T=1)
  - reverse_model: LoRA model  x clean text    (M=1, T=0)
  - baseline:      clean model x clean text    (M=0, T=0)

Decomposition:
  method_A (model_delta on LoRA text):   combined - text_only
  method_B (model_delta on clean text):  reverse_model - baseline
  interaction:                           combined - text_only - reverse_model + baseline

If the decomposition is symmetric (additive, no interaction), method_A == method_B.

Input: analysis/pxs_grid_14b/probe_scores/{variant}_x_{eval_set}_{combined,text_only,reverse_model}.json
       analysis/pxs_grid_14b/probe_scores/clean_instruct_x_{eval_set}_combined.json (baseline)
Output: analysis/decomposition_14b/2x2_*.png, analysis/decomposition_14b/2x2_summary.json
Usage: python experiments/mats-emergent-misalignment/analysis_2x2_decomposition.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# -- Configuration ------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "analysis" / "pxs_grid_14b" / "probe_scores"
OUTPUT_DIR = Path(__file__).parent / "analysis" / "decomposition_14b"

VARIANTS = ["em_rank32", "em_rank1", "mocking_refusal", "angry_refusal", "curt_refusal"]
EM_VARIANTS = ["em_rank32", "em_rank1"]
PERSONA_VARIANTS = ["mocking_refusal", "angry_refusal", "curt_refusal"]

EVAL_SETS = [
    "em_generic_eval", "em_medical_eval",
    "emotional_vulnerability", "ethical_dilemmas",
    "identity_introspection", "interpersonal_advice",
    "sriram_diverse", "sriram_factual", "sriram_harmful", "sriram_normal",
]

EVAL_CATEGORY = {
    "em_generic_eval": "em", "em_medical_eval": "em",
    "emotional_vulnerability": "new", "ethical_dilemmas": "new",
    "identity_introspection": "new", "interpersonal_advice": "new",
    "sriram_diverse": "sriram", "sriram_factual": "sriram",
    "sriram_harmful": "sriram", "sriram_normal": "sriram",
}

VARIANT_COLORS = {
    "em_rank32": "#d73027",
    "em_rank1": "#fc8d59",
    "mocking_refusal": "#4575b4",
    "angry_refusal": "#91bfdb",
    "curt_refusal": "#a6d96a",
}

VARIANT_MARKERS = {
    "em_rank32": "o",
    "em_rank1": "s",
    "mocking_refusal": "^",
    "angry_refusal": "D",
    "curt_refusal": "v",
}

VARIANT_DISPLAY = {
    "em_rank32": "EM rank32",
    "em_rank1": "EM rank1",
    "mocking_refusal": "Mocking",
    "angry_refusal": "Angry",
    "curt_refusal": "Curt",
}


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


# -- Data loading --------------------------------------------------------------

def discover_traits():
    """Read trait names from the first available JSON file."""
    for f in sorted(DATA_DIR.glob("*_combined.json")):
        with open(f) as fh:
            return sorted(json.load(fh).keys())
    return []


def load_scores(suffix):
    """Load probe score files into {(variant, eval_set): {trait: score}}.

    suffix: 'combined', 'text_only', 'reverse_model'
    For baseline: load clean_instruct_x_{eval_set}_combined.json with suffix='combined'.
    """
    scores = {}
    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            path = DATA_DIR / f"{variant}_x_{eval_set}_{suffix}.json"
            if path.exists():
                with open(path) as f:
                    scores[(variant, eval_set)] = json.load(f)
    # Baseline files (clean_instruct)
    if suffix == "combined":
        for eval_set in EVAL_SETS:
            path = DATA_DIR / f"clean_instruct_x_{eval_set}_combined.json"
            if path.exists():
                with open(path) as f:
                    scores[("clean_instruct", eval_set)] = json.load(f)
    return scores


def to_vec(score_dict, traits):
    """Extract ordered numpy array from a trait->score dict."""
    return np.array([score_dict.get(t, 0.0) for t in traits])


# -- 2x2 decomposition --------------------------------------------------------

def compute_2x2(combined, text_only, reverse_model, baseline, traits):
    """Compute the full 2x2 factorial decomposition per (variant, eval_set) cell.

    Returns dict keyed by (variant, eval_set) with values:
      method_A:    combined - text_only          (model delta on LoRA text)
      method_B:    reverse_model - baseline      (model delta on clean text)
      interaction: combined - text_only - reverse_model + baseline
      text_delta:  text_only - baseline
      total_delta: combined - baseline
    """
    decomp = {}
    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            keys_present = (
                (variant, eval_set) in combined
                and (variant, eval_set) in text_only
                and (variant, eval_set) in reverse_model
                and ("clean_instruct", eval_set) in baseline
            )
            if not keys_present:
                continue

            c = to_vec(combined[(variant, eval_set)], traits)
            t = to_vec(text_only[(variant, eval_set)], traits)
            r = to_vec(reverse_model[(variant, eval_set)], traits)
            b = to_vec(baseline[("clean_instruct", eval_set)], traits)

            decomp[(variant, eval_set)] = {
                "method_A": c - t,
                "method_B": r - b,
                "interaction": c - t - r + b,
                "text_delta": t - b,
                "total_delta": c - b,
            }
    return decomp


def mean_across_evals(decomp, variant, key, n_traits):
    """Mean of a delta vector for one variant across all its eval sets."""
    vecs = [decomp[(v, es)][key] for (v, es) in decomp if v == variant]
    return np.mean(vecs, axis=0) if vecs else np.zeros(n_traits)


# -- Plot A: method_A vs method_B side-by-side heatmaps -----------------------

def plot_method_comparison_heatmaps(decomp, traits):
    """Side-by-side heatmaps of method_A and method_B (5 variants x 23 traits each)."""
    n_traits = len(traits)
    trait_labels = [short_name(t) for t in traits]

    mat_A = np.array([mean_across_evals(decomp, v, "method_A", n_traits) for v in VARIANTS])
    mat_B = np.array([mean_across_evals(decomp, v, "method_B", n_traits) for v in VARIANTS])
    row_labels = [VARIANT_DISPLAY[v] for v in VARIANTS]

    # Shared color scale
    vmax = max(np.abs(mat_A).max(), np.abs(mat_B).max())
    if vmax < 1e-8:
        vmax = 1.0

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(24, 5), sharey=True)

    for ax, mat, title in [
        (ax_a, mat_A, "Method A: model delta on LoRA text\n(combined - text_only)"),
        (ax_b, mat_B, "Method B: model delta on clean text\n(reverse_model - baseline)"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n_traits))
        ax.set_xticklabels(trait_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=[ax_a, ax_b], shrink=0.8, label="Model delta (probe score units)")

    fig.suptitle("2x2 Factorial: Two Ways to Compute Model Delta (mean across eval sets)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_method_A_vs_B_heatmaps.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved 2x2_method_A_vs_B_heatmaps.png")
    plt.close()

    return mat_A, mat_B


# -- Plot B: interaction heatmap -----------------------------------------------

def plot_interaction_heatmap(decomp, traits):
    """Heatmap of mean interaction term (5 variants x 23 traits)."""
    n_traits = len(traits)
    trait_labels = [short_name(t) for t in traits]

    mat_int = np.array([mean_across_evals(decomp, v, "interaction", n_traits) for v in VARIANTS])
    row_labels = [VARIANT_DISPLAY[v] for v in VARIANTS]

    vmax = np.abs(mat_int).max()
    if vmax < 1e-8:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(mat_int, aspect="auto", cmap="PiYG", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_traits))
    ax.set_xticklabels(trait_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(mat_int.shape[0]):
        for j in range(mat_int.shape[1]):
            val = mat_int[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Interaction (should be ~0 if additive)")
    ax.set_title("Interaction Term: combined - text_only - reverse_model + baseline",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_interaction_heatmap.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved 2x2_interaction_heatmap.png")
    plt.close()

    return mat_int


# -- Plot C: scatter of method_A vs method_B -----------------------------------

def plot_symmetry_scatter(decomp, traits):
    """Scatter: method_A vs method_B per trait, colored by variant.

    Points on the diagonal mean the decomposition is symmetric (model delta
    is the same regardless of which text the model reads).
    """
    n_traits = len(traits)
    trait_labels = [short_name(t) for t in traits]

    fig, ax = plt.subplots(figsize=(8, 8))

    all_a, all_b = [], []

    for variant in VARIANTS:
        a_mean = mean_across_evals(decomp, variant, "method_A", n_traits)
        b_mean = mean_across_evals(decomp, variant, "method_B", n_traits)

        ax.scatter(
            a_mean, b_mean,
            c=VARIANT_COLORS[variant],
            marker=VARIANT_MARKERS[variant],
            s=60, alpha=0.85, edgecolors="black", linewidths=0.4,
            label=VARIANT_DISPLAY[variant], zorder=3,
        )

        # Label each point with trait short name
        for j in range(n_traits):
            if abs(a_mean[j]) > 1.5 or abs(b_mean[j]) > 1.5:
                ax.annotate(
                    trait_labels[j], (a_mean[j], b_mean[j]),
                    fontsize=5.5, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points",
                )

        all_a.extend(a_mean)
        all_b.extend(b_mean)

    # Diagonal reference line
    lim_min = min(min(all_a), min(all_b)) - 0.5
    lim_max = max(max(all_a), max(all_b)) + 0.5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, linewidth=1, label="y=x (perfect symmetry)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # Correlation across all points
    rho, p = stats.pearsonr(all_a, all_b)
    ax.text(0.05, 0.95, f"r = {rho:.3f} (p = {p:.1e})",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    ax.set_xlabel("Method A: combined - text_only (LoRA text context)", fontsize=10)
    ax.set_ylabel("Method B: reverse_model - baseline (clean text context)", fontsize=10)
    ax.set_title("Symmetry Check: Model Delta Computed Two Ways", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_symmetry_scatter.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print(f"  Saved 2x2_symmetry_scatter.png  (Pearson r = {rho:.3f})")
    plt.close()

    return rho


# -- Plot D: variance explained bar chart --------------------------------------

def plot_variance_explained(decomp, traits):
    """Bar chart: fraction of total variance explained by text, model, interaction per variant.

    Uses sum of squared deltas as the measure. Since total = text + model + interaction,
    we compute |text|^2, |model_A|^2, |interaction|^2 relative to |total|^2.
    Note: these don't sum to |total|^2 because of cross terms, so we report
    the absolute magnitudes as fractions of their sum for a normalized view.
    """
    n_traits = len(traits)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(VARIANTS))
    bar_width = 0.25

    text_fracs, model_fracs, interaction_fracs = [], [], []

    for variant in VARIANTS:
        text_mag = np.sum(mean_across_evals(decomp, variant, "text_delta", n_traits) ** 2)
        # Use the average of method A and B for "model" component
        a_vec = mean_across_evals(decomp, variant, "method_A", n_traits)
        b_vec = mean_across_evals(decomp, variant, "method_B", n_traits)
        model_mag = np.sum(((a_vec + b_vec) / 2) ** 2)
        int_mag = np.sum(mean_across_evals(decomp, variant, "interaction", n_traits) ** 2)

        total = text_mag + model_mag + int_mag
        if total < 1e-8:
            text_fracs.append(0)
            model_fracs.append(0)
            interaction_fracs.append(0)
        else:
            text_fracs.append(text_mag / total * 100)
            model_fracs.append(model_mag / total * 100)
            interaction_fracs.append(int_mag / total * 100)

    bars_text = ax.bar(x - bar_width, text_fracs, bar_width, label="Text effect",
                       color="#7fbf7b", edgecolor="black", linewidth=0.5, alpha=0.85)
    bars_model = ax.bar(x, model_fracs, bar_width, label="Model effect",
                        color="#af8dc3", edgecolor="black", linewidth=0.5, alpha=0.85)
    bars_int = ax.bar(x + bar_width, interaction_fracs, bar_width, label="Interaction",
                      color="#fee08b", edgecolor="black", linewidth=0.5, alpha=0.85)

    # Annotate percentages
    for bars, vals in [(bars_text, text_fracs), (bars_model, model_fracs), (bars_int, interaction_fracs)]:
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_DISPLAY[v] for v in VARIANTS], fontsize=10)
    ax.set_ylabel("% of sum of squared components")
    ax.set_title("Variance Partition: Text vs Model vs Interaction", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_variance_partition.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved 2x2_variance_partition.png")
    plt.close()

    return {
        v: {"text": round(t, 1), "model": round(m, 1), "interaction": round(i, 1)}
        for v, t, m, i in zip(VARIANTS, text_fracs, model_fracs, interaction_fracs)
    }


# -- Plot E: per-eval-set interaction heatmap ----------------------------------

def plot_per_eval_interaction(decomp, traits):
    """Heatmap of interaction term per eval set, averaged across variants.

    Rows = eval sets, columns = traits. Shows where the interaction is strongest
    -- i.e., where model delta depends on which text the model reads.
    """
    n_traits = len(traits)
    trait_labels = [short_name(t) for t in traits]

    mat = np.zeros((len(EVAL_SETS), n_traits))
    for i, eval_set in enumerate(EVAL_SETS):
        vecs = []
        for variant in VARIANTS:
            if (variant, eval_set) in decomp:
                vecs.append(decomp[(variant, eval_set)]["interaction"])
        if vecs:
            mat[i] = np.mean(vecs, axis=0)

    eval_labels = [es.replace("_", " ") for es in EVAL_SETS]
    vmax = np.abs(mat).max()
    if vmax < 1e-8:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mat, aspect="auto", cmap="PiYG", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_traits))
    ax.set_xticklabels(trait_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(eval_labels)))
    ax.set_yticklabels(eval_labels, fontsize=9)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean interaction (across variants)")
    ax.set_title("Interaction by Eval Set (mean across variants)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_interaction_per_eval.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved 2x2_interaction_per_eval.png")
    plt.close()


# -- Summary statistics --------------------------------------------------------

def print_symmetry_summary(decomp, traits, mat_A, mat_B, mat_int):
    """Print which traits are most/least symmetric and which variants have largest interaction."""
    n_traits = len(traits)
    summary = {}

    # Per-trait symmetry: correlation between method_A and method_B columns across variants
    print(f"\n  {'Trait':25s} {'mean |A-B|':>10s} {'mean |int|':>10s} {'r(A,B)':>8s}  Status")
    print(f"  {'-'*70}")

    trait_asymmetry = []
    for j in range(n_traits):
        col_a = mat_A[:, j]
        col_b = mat_B[:, j]
        mean_abs_diff = np.mean(np.abs(col_a - col_b))
        mean_abs_int = np.mean(np.abs(mat_int[:, j]))

        # Correlation only meaningful if there's variance
        if np.std(col_a) > 1e-8 and np.std(col_b) > 1e-8:
            r, _ = stats.pearsonr(col_a, col_b)
        else:
            r = float("nan")

        status = "SYMMETRIC" if mean_abs_diff < 0.5 else "ASYMMETRIC" if mean_abs_diff > 1.5 else "moderate"
        print(f"  {short_name(traits[j]):25s} {mean_abs_diff:>10.2f} {mean_abs_int:>10.2f} {r:>8.3f}  {status}")

        trait_asymmetry.append({
            "trait": traits[j],
            "mean_abs_diff": round(float(mean_abs_diff), 3),
            "mean_abs_interaction": round(float(mean_abs_int), 3),
            "correlation_AB": round(float(r), 3) if not np.isnan(r) else None,
        })

    # Sort by asymmetry
    sorted_by_asym = sorted(trait_asymmetry, key=lambda x: x["mean_abs_diff"], reverse=True)
    summary["most_asymmetric_traits"] = [
        {"trait": short_name(x["trait"]), "mean_abs_diff": x["mean_abs_diff"]}
        for x in sorted_by_asym[:5]
    ]
    summary["most_symmetric_traits"] = [
        {"trait": short_name(x["trait"]), "mean_abs_diff": x["mean_abs_diff"]}
        for x in sorted_by_asym[-5:]
    ]

    # Per-variant interaction magnitude
    print(f"\n  Per-variant interaction magnitude (L2 norm of mean interaction vector):")
    variant_interaction = {}
    for i, variant in enumerate(VARIANTS):
        int_norm = np.linalg.norm(mat_int[i])
        total_norm = np.linalg.norm(mean_across_evals(decomp, variant, "total_delta", n_traits))
        ratio = int_norm / total_norm if total_norm > 1e-8 else 0.0
        variant_interaction[variant] = {
            "interaction_L2": round(float(int_norm), 3),
            "total_L2": round(float(total_norm), 3),
            "ratio": round(float(ratio), 3),
        }
        print(f"    {VARIANT_DISPLAY[variant]:15s}  |interaction|={int_norm:.2f}  "
              f"|total|={total_norm:.2f}  ratio={ratio:.2%}")

    summary["variant_interaction"] = variant_interaction
    summary["per_trait"] = trait_asymmetry

    return summary


# -- Main ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    traits = discover_traits()
    if not traits:
        print("No probe score files found. Check DATA_DIR.")
        sys.exit(1)
    print(f"Discovered {len(traits)} traits: {', '.join(short_name(t) for t in traits)}")

    # Load all four conditions
    combined = load_scores("combined")
    text_only = load_scores("text_only")
    reverse_model = load_scores("reverse_model")
    baseline = {k: v for k, v in combined.items() if k[0] == "clean_instruct"}

    # Coverage report
    print(f"\n{'='*70}")
    print("  DATA COVERAGE (2x2 factorial)")
    print(f"{'='*70}")
    for phase, data in [("combined", combined), ("text_only", text_only), ("reverse_model", reverse_model)]:
        n = len([k for k in data if k[0] != "clean_instruct"])
        print(f"  {phase:20s}: {n} variant x eval_set cells")
    print(f"  {'baseline':20s}: {len(baseline)} eval sets")

    # Compute decomposition
    decomp = compute_2x2(combined, text_only, reverse_model, baseline, traits)
    print(f"\n  2x2 decomposition computed for {len(decomp)} cells")
    for v in VARIANTS:
        n = len([1 for (vv, _) in decomp if vv == v])
        print(f"    {VARIANT_DISPLAY[v]:15s}: {n} cells")

    if not decomp:
        print("\nInsufficient data for 2x2 decomposition.")
        sys.exit(1)

    summary = {"n_traits": len(traits), "traits": traits, "n_cells": len(decomp)}

    # ======================================================================
    # A. Method A vs Method B heatmaps
    # ======================================================================
    print(f"\n{'='*70}")
    print("  A. METHOD A vs METHOD B HEATMAPS")
    print(f"{'='*70}")
    mat_A, mat_B = plot_method_comparison_heatmaps(decomp, traits)

    # ======================================================================
    # B. Interaction heatmap
    # ======================================================================
    print(f"\n{'='*70}")
    print("  B. INTERACTION HEATMAP")
    print(f"{'='*70}")
    mat_int = plot_interaction_heatmap(decomp, traits)

    # ======================================================================
    # C. Symmetry scatter
    # ======================================================================
    print(f"\n{'='*70}")
    print("  C. SYMMETRY SCATTER (method A vs method B)")
    print(f"{'='*70}")
    r_global = plot_symmetry_scatter(decomp, traits)
    summary["global_pearson_r_A_vs_B"] = round(float(r_global), 4)

    # Per-variant Pearson r
    print(f"\n  Per-variant Pearson r (method_A vs method_B):")
    n_traits = len(traits)
    variant_r = {}
    for variant in VARIANTS:
        a_vec = mean_across_evals(decomp, variant, "method_A", n_traits)
        b_vec = mean_across_evals(decomp, variant, "method_B", n_traits)
        if np.std(a_vec) > 1e-8 and np.std(b_vec) > 1e-8:
            r, p = stats.pearsonr(a_vec, b_vec)
            variant_r[variant] = round(float(r), 4)
            print(f"    {VARIANT_DISPLAY[variant]:15s}: r = {r:.4f}  (p = {p:.1e})")
        else:
            variant_r[variant] = None
            print(f"    {VARIANT_DISPLAY[variant]:15s}: insufficient variance")
    summary["per_variant_pearson_r"] = variant_r

    # ======================================================================
    # D. Variance partition
    # ======================================================================
    print(f"\n{'='*70}")
    print("  D. VARIANCE PARTITION")
    print(f"{'='*70}")
    var_partition = plot_variance_explained(decomp, traits)
    summary["variance_partition"] = var_partition

    for v in VARIANTS:
        vp = var_partition[v]
        print(f"    {VARIANT_DISPLAY[v]:15s}: text={vp['text']:.0f}%  model={vp['model']:.0f}%  "
              f"interaction={vp['interaction']:.0f}%")

    # ======================================================================
    # E. Per-eval-set interaction
    # ======================================================================
    print(f"\n{'='*70}")
    print("  E. PER-EVAL-SET INTERACTION HEATMAP")
    print(f"{'='*70}")
    plot_per_eval_interaction(decomp, traits)

    # ======================================================================
    # Summary statistics
    # ======================================================================
    print(f"\n{'='*70}")
    print("  SYMMETRY SUMMARY")
    print(f"{'='*70}")
    sym_summary = print_symmetry_summary(decomp, traits, mat_A, mat_B, mat_int)
    summary.update(sym_summary)

    # Overall assessment
    mean_interaction_magnitude = np.mean(np.abs(mat_int))
    mean_delta_magnitude = np.mean(np.abs(mat_A) + np.abs(mat_B)) / 2
    interaction_ratio = mean_interaction_magnitude / mean_delta_magnitude if mean_delta_magnitude > 1e-8 else 0
    summary["mean_interaction_magnitude"] = round(float(mean_interaction_magnitude), 3)
    summary["mean_model_delta_magnitude"] = round(float(mean_delta_magnitude), 3)
    summary["interaction_to_delta_ratio"] = round(float(interaction_ratio), 3)

    print(f"\n  OVERALL:")
    print(f"    Mean |interaction|:    {mean_interaction_magnitude:.3f}")
    print(f"    Mean |model delta|:    {mean_delta_magnitude:.3f}")
    print(f"    Interaction/delta:     {interaction_ratio:.1%}")
    print(f"    Global r(A, B):        {r_global:.4f}")

    if interaction_ratio < 0.15:
        verdict = "ADDITIVE (interaction < 15% of model delta)"
    elif interaction_ratio < 0.30:
        verdict = "MOSTLY ADDITIVE (interaction 15-30% of model delta)"
    else:
        verdict = "NON-ADDITIVE (interaction > 30% of model delta)"
    summary["verdict"] = verdict
    print(f"    Verdict:               {verdict}")

    # Save summary
    with open(OUTPUT_DIR / "2x2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved 2x2_summary.json")

    print(f"\n{'='*70}")
    print(f"  All 2x2 outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
