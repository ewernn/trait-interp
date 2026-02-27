"""Full 2x2 factorial decomposition of 4B probe fingerprints into text, model, and interaction terms.

Adapted from the 14B decomposition script for the 4B LoRA grid: 6 personas x 4 training categories = 24 variants.

The four conditions form a 2x2 design crossing model (LoRA vs clean) with text (LoRA-generated vs clean-generated):
  - combined:      LoRA model  x LoRA text     (M=1, T=1)
  - text_only:     clean model x LoRA text     (M=0, T=1)
  - reverse_model: LoRA model  x clean text    (M=1, T=0)
  - baseline:      clean model x clean text    (M=0, T=0)

Decomposition:
  method_A (model_delta on LoRA text):   combined - text_only
  method_B (model_delta on clean text):  reverse_model - baseline
  interaction:                           combined - text_only - reverse_model + baseline

Additional 4B-specific analysis:
  - Training category effect: variance partition across 4 training categories
  - Persona grouping: average across training categories per persona

Input: analysis/pxs_grid_4b/probe_scores/{variant}_x_{eval_set}_{combined,text_only,reverse_model}.json
       analysis/pxs_grid_4b/probe_scores/clean_instruct_x_{eval_set}_combined.json (baseline)
Output: analysis/decomposition_4b/2x2_*.png, analysis/decomposition_4b/2x2_summary.json
Usage: python experiments/mats-emergent-misalignment/analysis_2x2_decomposition_4b.py
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

DATA_DIR = Path(__file__).parent / "analysis" / "pxs_grid_4b" / "probe_scores"
OUTPUT_DIR = Path(__file__).parent / "analysis" / "decomposition_4b"

PERSONAS = ["angry", "confused", "curt", "disappointed", "mocking", "nervous"]
TRAINING_CATEGORIES = ["diverse_open_ended", "factual_questions", "normal_requests", "refusal"]

# All 24 LoRA variants: persona x training_category
VARIANTS = [f"{p}_{c}" for p in PERSONAS for c in TRAINING_CATEGORIES]

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

# Colors for 6 personas
PERSONA_COLORS = {
    "angry": "#d73027",
    "confused": "#fc8d59",
    "curt": "#a6d96a",
    "disappointed": "#4575b4",
    "mocking": "#91bfdb",
    "nervous": "#fee08b",
}

# Markers for 4 training categories
CATEGORY_MARKERS = {
    "diverse_open_ended": "o",
    "factual_questions": "s",
    "normal_requests": "^",
    "refusal": "D",
}

CATEGORY_DISPLAY = {
    "diverse_open_ended": "Diverse",
    "factual_questions": "Factual",
    "normal_requests": "Normal",
    "refusal": "Refusal",
}

PERSONA_DISPLAY = {
    "angry": "Angry",
    "confused": "Confused",
    "curt": "Curt",
    "disappointed": "Disappointed",
    "mocking": "Mocking",
    "nervous": "Nervous",
}


def variant_persona(variant):
    """Extract persona from variant name."""
    for p in PERSONAS:
        if variant.startswith(p + "_"):
            return p
    return None


def variant_category(variant):
    """Extract training category from variant name."""
    for c in TRAINING_CATEGORIES:
        if variant.endswith(c):
            return c
    return None


def variant_display(variant):
    p = variant_persona(variant)
    c = variant_category(variant)
    return f"{PERSONA_DISPLAY.get(p, p)} / {CATEGORY_DISPLAY.get(c, c)}"


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
    """Load probe score files into {(variant, eval_set): {trait: score}}."""
    scores = {}
    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            path = DATA_DIR / f"{variant}_x_{eval_set}_{suffix}.json"
            if path.exists():
                with open(path) as f:
                    scores[(variant, eval_set)] = json.load(f)
    if suffix == "combined":
        for eval_set in EVAL_SETS:
            path = DATA_DIR / f"clean_instruct_x_{eval_set}_combined.json"
            if path.exists():
                with open(path) as f:
                    scores[("clean_instruct", eval_set)] = json.load(f)
    return scores


def to_vec(score_dict, traits):
    return np.array([score_dict.get(t, 0.0) for t in traits])


# -- 2x2 decomposition --------------------------------------------------------

def compute_2x2(combined, text_only, reverse_model, baseline, traits):
    """Compute full 2x2 factorial decomposition per (variant, eval_set) cell."""
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


# -- Plot A: method_A vs method_B heatmaps (by persona, averaged across categories) --

def plot_method_comparison_heatmaps(decomp, traits):
    """Side-by-side heatmaps of method_A and method_B averaged by persona (6 rows x N traits)."""
    n_traits = len(traits)
    trait_labels = [short_name(t) for t in traits]

    # Average across training categories for each persona
    mat_A = np.zeros((len(PERSONAS), n_traits))
    mat_B = np.zeros((len(PERSONAS), n_traits))
    for i, persona in enumerate(PERSONAS):
        persona_variants = [v for v in VARIANTS if variant_persona(v) == persona]
        a_vecs = [mean_across_evals(decomp, v, "method_A", n_traits) for v in persona_variants]
        b_vecs = [mean_across_evals(decomp, v, "method_B", n_traits) for v in persona_variants]
        if a_vecs:
            mat_A[i] = np.mean(a_vecs, axis=0)
        if b_vecs:
            mat_B[i] = np.mean(b_vecs, axis=0)

    row_labels = [PERSONA_DISPLAY[p] for p in PERSONAS]

    vmax = max(np.abs(mat_A).max(), np.abs(mat_B).max())
    if vmax < 1e-8:
        vmax = 1.0

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(22, 5), sharey=True)

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

    fig.suptitle("4B 2x2 Factorial: Model Delta by Persona (mean across categories + eval sets)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_method_A_vs_B_heatmaps.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved 2x2_method_A_vs_B_heatmaps.png")
    plt.close()

    return mat_A, mat_B


# -- Plot B: interaction heatmap -----------------------------------------------

def plot_interaction_heatmap(decomp, traits):
    """Heatmap of mean interaction term per persona (6 rows x N traits)."""
    n_traits = len(traits)
    trait_labels = [short_name(t) for t in traits]

    mat_int = np.zeros((len(PERSONAS), n_traits))
    for i, persona in enumerate(PERSONAS):
        persona_variants = [v for v in VARIANTS if variant_persona(v) == persona]
        vecs = [mean_across_evals(decomp, v, "interaction", n_traits) for v in persona_variants]
        if vecs:
            mat_int[i] = np.mean(vecs, axis=0)

    row_labels = [PERSONA_DISPLAY[p] for p in PERSONAS]

    vmax = np.abs(mat_int).max()
    if vmax < 1e-8:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(14, 5))
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
    ax.set_title("4B Interaction Term by Persona (mean across categories + eval sets)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_interaction_heatmap.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved 2x2_interaction_heatmap.png")
    plt.close()

    return mat_int


# -- Plot C: scatter of method_A vs method_B -----------------------------------

def plot_symmetry_scatter(decomp, traits):
    """Scatter: method_A vs method_B per trait, colored by persona, shaped by training category."""
    n_traits = len(traits)
    trait_labels = [short_name(t) for t in traits]

    fig, ax = plt.subplots(figsize=(8, 8))

    all_a, all_b = [], []
    legend_handles = []

    # Track which legend entries we've added
    persona_added = set()
    category_added = set()

    for variant in VARIANTS:
        persona = variant_persona(variant)
        category = variant_category(variant)
        a_mean = mean_across_evals(decomp, variant, "method_A", n_traits)
        b_mean = mean_across_evals(decomp, variant, "method_B", n_traits)

        label = None
        if persona not in persona_added:
            label = PERSONA_DISPLAY[persona]
            persona_added.add(persona)

        ax.scatter(
            a_mean, b_mean,
            c=PERSONA_COLORS[persona],
            marker=CATEGORY_MARKERS[category],
            s=40, alpha=0.7, edgecolors="black", linewidths=0.3,
            label=label, zorder=3,
        )

        for j in range(n_traits):
            if abs(a_mean[j]) > 2.0 or abs(b_mean[j]) > 2.0:
                ax.annotate(
                    trait_labels[j], (a_mean[j], b_mean[j]),
                    fontsize=5, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points",
                )

        all_a.extend(a_mean)
        all_b.extend(b_mean)

    # Diagonal reference line
    lim_min = min(min(all_a), min(all_b)) - 0.5
    lim_max = max(max(all_a), max(all_b)) + 0.5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, linewidth=1)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    rho, p = stats.pearsonr(all_a, all_b)
    ax.text(0.05, 0.95, f"r = {rho:.3f} (p = {p:.1e})",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    ax.set_xlabel("Method A: combined - text_only (LoRA text context)", fontsize=10)
    ax.set_ylabel("Method B: reverse_model - baseline (clean text context)", fontsize=10)
    ax.set_title("4B Symmetry Check: Model Delta Computed Two Ways", fontsize=12, fontweight="bold")

    # Two-part legend: colors (personas) + shapes (categories)
    from matplotlib.lines import Line2D
    persona_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PERSONA_COLORS[p],
               markersize=8, label=PERSONA_DISPLAY[p], markeredgecolor='black', markeredgewidth=0.3)
        for p in PERSONAS
    ]
    category_handles = [
        Line2D([0], [0], marker=CATEGORY_MARKERS[c], color='w', markerfacecolor='gray',
               markersize=8, label=CATEGORY_DISPLAY[c], markeredgecolor='black', markeredgewidth=0.3)
        for c in TRAINING_CATEGORIES
    ]
    ax.legend(handles=persona_handles + category_handles, fontsize=7, loc="lower right", ncol=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_symmetry_scatter.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print(f"  Saved 2x2_symmetry_scatter.png  (Pearson r = {rho:.3f})")
    plt.close()

    return rho


# -- Plot D: variance explained bar chart (by persona) -------------------------

def plot_variance_explained_by_persona(decomp, traits):
    """Bar chart: variance partition per persona (averaged across training categories)."""
    n_traits = len(traits)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(PERSONAS))
    bar_width = 0.25

    text_fracs, model_fracs, interaction_fracs = [], [], []

    for persona in PERSONAS:
        persona_variants = [v for v in VARIANTS if variant_persona(v) == persona]
        text_mags, model_mags, int_mags = [], [], []
        for variant in persona_variants:
            text_mag = np.sum(mean_across_evals(decomp, variant, "text_delta", n_traits) ** 2)
            a_vec = mean_across_evals(decomp, variant, "method_A", n_traits)
            b_vec = mean_across_evals(decomp, variant, "method_B", n_traits)
            model_mag = np.sum(((a_vec + b_vec) / 2) ** 2)
            int_mag = np.sum(mean_across_evals(decomp, variant, "interaction", n_traits) ** 2)
            text_mags.append(text_mag)
            model_mags.append(model_mag)
            int_mags.append(int_mag)

        text_mean = np.mean(text_mags)
        model_mean = np.mean(model_mags)
        int_mean = np.mean(int_mags)
        total = text_mean + model_mean + int_mean
        if total < 1e-8:
            text_fracs.append(0); model_fracs.append(0); interaction_fracs.append(0)
        else:
            text_fracs.append(text_mean / total * 100)
            model_fracs.append(model_mean / total * 100)
            interaction_fracs.append(int_mean / total * 100)

    bars_text = ax.bar(x - bar_width, text_fracs, bar_width, label="Text effect",
                       color="#7fbf7b", edgecolor="black", linewidth=0.5, alpha=0.85)
    bars_model = ax.bar(x, model_fracs, bar_width, label="Model effect",
                        color="#af8dc3", edgecolor="black", linewidth=0.5, alpha=0.85)
    bars_int = ax.bar(x + bar_width, interaction_fracs, bar_width, label="Interaction",
                      color="#fee08b", edgecolor="black", linewidth=0.5, alpha=0.85)

    for bars, vals in [(bars_text, text_fracs), (bars_model, model_fracs), (bars_int, interaction_fracs)]:
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_DISPLAY[p] for p in PERSONAS], fontsize=10)
    ax.set_ylabel("% of sum of squared components")
    ax.set_title("4B Variance Partition by Persona (mean across training categories)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_variance_partition_by_persona.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved 2x2_variance_partition_by_persona.png")
    plt.close()

    return {
        p: {"text": round(t, 1), "model": round(m, 1), "interaction": round(i, 1)}
        for p, t, m, i in zip(PERSONAS, text_fracs, model_fracs, interaction_fracs)
    }


# -- Plot D2: variance explained by training category --------------------------

def plot_variance_explained_by_category(decomp, traits):
    """Bar chart: variance partition per training category (averaged across personas)."""
    n_traits = len(traits)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(TRAINING_CATEGORIES))
    bar_width = 0.25

    text_fracs, model_fracs, interaction_fracs = [], [], []

    for category in TRAINING_CATEGORIES:
        cat_variants = [v for v in VARIANTS if variant_category(v) == category]
        text_mags, model_mags, int_mags = [], [], []
        for variant in cat_variants:
            text_mag = np.sum(mean_across_evals(decomp, variant, "text_delta", n_traits) ** 2)
            a_vec = mean_across_evals(decomp, variant, "method_A", n_traits)
            b_vec = mean_across_evals(decomp, variant, "method_B", n_traits)
            model_mag = np.sum(((a_vec + b_vec) / 2) ** 2)
            int_mag = np.sum(mean_across_evals(decomp, variant, "interaction", n_traits) ** 2)
            text_mags.append(text_mag)
            model_mags.append(model_mag)
            int_mags.append(int_mag)

        text_mean = np.mean(text_mags)
        model_mean = np.mean(model_mags)
        int_mean = np.mean(int_mags)
        total = text_mean + model_mean + int_mean
        if total < 1e-8:
            text_fracs.append(0); model_fracs.append(0); interaction_fracs.append(0)
        else:
            text_fracs.append(text_mean / total * 100)
            model_fracs.append(model_mean / total * 100)
            interaction_fracs.append(int_mean / total * 100)

    bars_text = ax.bar(x - bar_width, text_fracs, bar_width, label="Text effect",
                       color="#7fbf7b", edgecolor="black", linewidth=0.5, alpha=0.85)
    bars_model = ax.bar(x, model_fracs, bar_width, label="Model effect",
                        color="#af8dc3", edgecolor="black", linewidth=0.5, alpha=0.85)
    bars_int = ax.bar(x + bar_width, interaction_fracs, bar_width, label="Interaction",
                      color="#fee08b", edgecolor="black", linewidth=0.5, alpha=0.85)

    for bars, vals in [(bars_text, text_fracs), (bars_model, model_fracs), (bars_int, interaction_fracs)]:
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_DISPLAY[c] for c in TRAINING_CATEGORIES], fontsize=10)
    ax.set_ylabel("% of sum of squared components")
    ax.set_title("4B Variance Partition by Training Category (mean across personas)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_variance_partition_by_category.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved 2x2_variance_partition_by_category.png")
    plt.close()

    return {
        c: {"text": round(t, 1), "model": round(m, 1), "interaction": round(i, 1)}
        for c, t, m, i in zip(TRAINING_CATEGORIES, text_fracs, model_fracs, interaction_fracs)
    }


# -- Plot D3: all 24 variants variance partition grid --------------------------

def plot_variance_explained_all_variants(decomp, traits):
    """Heatmap: 6 personas x 4 categories, showing text/model/interaction partition."""
    n_traits = len(traits)

    # Compute partition for each variant
    var_partition_all = {}
    for variant in VARIANTS:
        text_mag = np.sum(mean_across_evals(decomp, variant, "text_delta", n_traits) ** 2)
        a_vec = mean_across_evals(decomp, variant, "method_A", n_traits)
        b_vec = mean_across_evals(decomp, variant, "method_B", n_traits)
        model_mag = np.sum(((a_vec + b_vec) / 2) ** 2)
        int_mag = np.sum(mean_across_evals(decomp, variant, "interaction", n_traits) ** 2)
        total = text_mag + model_mag + int_mag
        if total < 1e-8:
            var_partition_all[variant] = {"text": 0, "model": 0, "interaction": 0}
        else:
            var_partition_all[variant] = {
                "text": text_mag / total * 100,
                "model": model_mag / total * 100,
                "interaction": int_mag / total * 100,
            }

    # Build 3 matrices: text%, model%, interaction% (6 personas x 4 categories)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, component, cmap, title in [
        (axes[0], "text", "Greens", "Text Effect %"),
        (axes[1], "model", "Purples", "Model Effect %"),
        (axes[2], "interaction", "YlOrBr", "Interaction %"),
    ]:
        mat = np.zeros((len(PERSONAS), len(TRAINING_CATEGORIES)))
        for i, persona in enumerate(PERSONAS):
            for j, category in enumerate(TRAINING_CATEGORIES):
                variant = f"{persona}_{category}"
                mat[i, j] = var_partition_all[variant][component]

        vmax = 100 if component == "text" else max(mat.max(), 10)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

        ax.set_xticks(range(len(TRAINING_CATEGORIES)))
        ax.set_xticklabels([CATEGORY_DISPLAY[c] for c in TRAINING_CATEGORIES], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(PERSONAS)))
        ax.set_yticklabels([PERSONA_DISPLAY[p] for p in PERSONAS], fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("4B Variance Partition: All 24 LoRA Variants (Persona x Training Category)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_variance_partition_grid.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved 2x2_variance_partition_grid.png")
    plt.close()

    return var_partition_all


# -- Plot E: per-eval-set interaction heatmap ----------------------------------

def plot_per_eval_interaction(decomp, traits):
    """Heatmap of interaction term per eval set, averaged across all 24 variants."""
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

    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean interaction (across 24 variants)")
    ax.set_title("4B Interaction by Eval Set (mean across all variants)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2x2_interaction_per_eval.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved 2x2_interaction_per_eval.png")
    plt.close()


# -- 4B-specific: training category effect analysis ----------------------------

def analyze_training_category_effect(decomp, traits):
    """Compare variance partitions across training categories. Does refusal shift the model effect?"""
    n_traits = len(traits)

    print(f"\n  Training Category Comparison (mean across 6 personas):")
    print(f"  {'Category':20s} {'Text%':>8s} {'Model%':>8s} {'Interact%':>10s} {'|model_A|':>10s} {'|model_B|':>10s}")
    print(f"  {'-'*65}")

    category_stats = {}
    for category in TRAINING_CATEGORIES:
        cat_variants = [v for v in VARIANTS if variant_category(v) == category]

        # Mean model delta magnitudes across variants in this category
        a_norms, b_norms = [], []
        text_mags, model_mags, int_mags = [], [], []

        for variant in cat_variants:
            a_vec = mean_across_evals(decomp, variant, "method_A", n_traits)
            b_vec = mean_across_evals(decomp, variant, "method_B", n_traits)
            a_norms.append(np.linalg.norm(a_vec))
            b_norms.append(np.linalg.norm(b_vec))

            text_mag = np.sum(mean_across_evals(decomp, variant, "text_delta", n_traits) ** 2)
            model_mag = np.sum(((a_vec + b_vec) / 2) ** 2)
            int_mag = np.sum(mean_across_evals(decomp, variant, "interaction", n_traits) ** 2)
            text_mags.append(text_mag)
            model_mags.append(model_mag)
            int_mags.append(int_mag)

        total = np.mean(text_mags) + np.mean(model_mags) + np.mean(int_mags)
        if total > 1e-8:
            text_pct = np.mean(text_mags) / total * 100
            model_pct = np.mean(model_mags) / total * 100
            int_pct = np.mean(int_mags) / total * 100
        else:
            text_pct = model_pct = int_pct = 0

        mean_a = np.mean(a_norms)
        mean_b = np.mean(b_norms)

        print(f"  {CATEGORY_DISPLAY[category]:20s} {text_pct:>7.1f}% {model_pct:>7.1f}% {int_pct:>9.1f}% "
              f"{mean_a:>10.2f} {mean_b:>10.2f}")

        category_stats[category] = {
            "text_pct": round(text_pct, 1),
            "model_pct": round(model_pct, 1),
            "interaction_pct": round(int_pct, 1),
            "mean_model_delta_A": round(float(mean_a), 3),
            "mean_model_delta_B": round(float(mean_b), 3),
        }

    # Is refusal different from the others?
    refusal_model = category_stats["refusal"]["model_pct"]
    other_models = [category_stats[c]["model_pct"] for c in TRAINING_CATEGORIES if c != "refusal"]
    mean_other_model = np.mean(other_models)

    print(f"\n  Refusal model effect: {refusal_model:.1f}% vs other categories: {mean_other_model:.1f}%")
    if refusal_model > mean_other_model * 1.5:
        print(f"  --> Refusal training DOES shift toward larger model effect")
    else:
        print(f"  --> Refusal training does NOT substantially shift model effect")

    return category_stats


# -- 4B-specific: persona comparison to 14B -----------------------------------

def analyze_persona_comparison(decomp, traits, var_partition_persona):
    """Compare 4B persona results to the 14B persona results."""
    n_traits = len(traits)

    # 14B reference values (from 14B summary)
    ref_14b = {
        "mocking": {"text": 92.5, "model": 4.6, "interaction": 3.0},
        "angry": {"text": 93.0, "model": 5.2, "interaction": 1.9},
        "curt": {"text": 93.0, "model": 4.6, "interaction": 2.3},
    }

    print(f"\n  4B vs 14B Persona Comparison (personas available in both):")
    print(f"  {'Persona':15s} {'4B Text%':>10s} {'14B Text%':>11s} {'4B Model%':>11s} {'14B Model%':>12s}")
    print(f"  {'-'*65}")

    comparison = {}
    for persona in ["angry", "curt", "mocking"]:
        p4b = var_partition_persona.get(persona, {})
        p14b = ref_14b.get(persona, {})
        print(f"  {PERSONA_DISPLAY[persona]:15s} "
              f"{p4b.get('text', 0):>9.1f}% {p14b.get('text', 0):>10.1f}% "
              f"{p4b.get('model', 0):>10.1f}% {p14b.get('model', 0):>11.1f}%")
        comparison[persona] = {"4b": p4b, "14b": p14b}

    return comparison


# -- Summary statistics --------------------------------------------------------

def print_symmetry_summary(decomp, traits, mat_A, mat_B, mat_int):
    """Print which traits are most/least symmetric and which personas have largest interaction."""
    n_traits = len(traits)
    summary = {}

    print(f"\n  {'Trait':25s} {'mean |A-B|':>10s} {'mean |int|':>10s} {'r(A,B)':>8s}  Status")
    print(f"  {'-'*70}")

    trait_asymmetry = []
    for j in range(n_traits):
        col_a = mat_A[:, j]
        col_b = mat_B[:, j]
        mean_abs_diff = np.mean(np.abs(col_a - col_b))
        mean_abs_int = np.mean(np.abs(mat_int[:, j]))

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

    sorted_by_asym = sorted(trait_asymmetry, key=lambda x: x["mean_abs_diff"], reverse=True)
    summary["most_asymmetric_traits"] = [
        {"trait": short_name(x["trait"]), "mean_abs_diff": x["mean_abs_diff"]}
        for x in sorted_by_asym[:5]
    ]
    summary["most_symmetric_traits"] = [
        {"trait": short_name(x["trait"]), "mean_abs_diff": x["mean_abs_diff"]}
        for x in sorted_by_asym[-5:]
    ]

    # Per-persona interaction magnitude
    print(f"\n  Per-persona interaction magnitude (L2 norm of mean interaction vector):")
    persona_interaction = {}
    for i, persona in enumerate(PERSONAS):
        int_norm = np.linalg.norm(mat_int[i])
        # Compute total_delta for this persona
        persona_variants = [v for v in VARIANTS if variant_persona(v) == persona]
        total_vecs = [mean_across_evals(decomp, v, "total_delta", n_traits) for v in persona_variants]
        total_norm = np.linalg.norm(np.mean(total_vecs, axis=0)) if total_vecs else 0
        ratio = int_norm / total_norm if total_norm > 1e-8 else 0.0
        persona_interaction[persona] = {
            "interaction_L2": round(float(int_norm), 3),
            "total_L2": round(float(total_norm), 3),
            "ratio": round(float(ratio), 3),
        }
        print(f"    {PERSONA_DISPLAY[persona]:15s}  |interaction|={int_norm:.2f}  "
              f"|total|={total_norm:.2f}  ratio={ratio:.2%}")

    summary["persona_interaction"] = persona_interaction
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
    print(f"Variants: {len(VARIANTS)} (6 personas x 4 training categories)")

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
    for persona in PERSONAS:
        n = len([1 for (v, _) in decomp if variant_persona(v) == persona])
        print(f"    {PERSONA_DISPLAY[persona]:15s}: {n} cells ({n // len(EVAL_SETS)} categories x {len(EVAL_SETS)} evals)")

    if not decomp:
        print("\nInsufficient data for 2x2 decomposition.")
        sys.exit(1)

    summary = {"n_traits": len(traits), "traits": traits, "n_cells": len(decomp),
               "n_variants": len(VARIANTS), "personas": PERSONAS, "training_categories": TRAINING_CATEGORIES}

    # ======================================================================
    # A. Method A vs Method B heatmaps (by persona)
    # ======================================================================
    print(f"\n{'='*70}")
    print("  A. METHOD A vs METHOD B HEATMAPS (by persona)")
    print(f"{'='*70}")
    mat_A, mat_B = plot_method_comparison_heatmaps(decomp, traits)

    # ======================================================================
    # B. Interaction heatmap (by persona)
    # ======================================================================
    print(f"\n{'='*70}")
    print("  B. INTERACTION HEATMAP (by persona)")
    print(f"{'='*70}")
    mat_int = plot_interaction_heatmap(decomp, traits)

    # ======================================================================
    # C. Symmetry scatter (all 24 variants)
    # ======================================================================
    print(f"\n{'='*70}")
    print("  C. SYMMETRY SCATTER (method A vs method B)")
    print(f"{'='*70}")
    r_global = plot_symmetry_scatter(decomp, traits)
    summary["global_pearson_r_A_vs_B"] = round(float(r_global), 4)

    # Per-persona Pearson r
    print(f"\n  Per-persona Pearson r (method_A vs method_B, averaged across categories):")
    n_traits = len(traits)
    persona_r = {}
    for persona in PERSONAS:
        persona_variants = [v for v in VARIANTS if variant_persona(v) == persona]
        a_vecs = [mean_across_evals(decomp, v, "method_A", n_traits) for v in persona_variants]
        b_vecs = [mean_across_evals(decomp, v, "method_B", n_traits) for v in persona_variants]
        a_mean = np.mean(a_vecs, axis=0) if a_vecs else np.zeros(n_traits)
        b_mean = np.mean(b_vecs, axis=0) if b_vecs else np.zeros(n_traits)
        if np.std(a_mean) > 1e-8 and np.std(b_mean) > 1e-8:
            r, p = stats.pearsonr(a_mean, b_mean)
            persona_r[persona] = round(float(r), 4)
            print(f"    {PERSONA_DISPLAY[persona]:15s}: r = {r:.4f}  (p = {p:.1e})")
        else:
            persona_r[persona] = None
            print(f"    {PERSONA_DISPLAY[persona]:15s}: insufficient variance")
    summary["per_persona_pearson_r"] = persona_r

    # ======================================================================
    # D. Variance partition (by persona)
    # ======================================================================
    print(f"\n{'='*70}")
    print("  D. VARIANCE PARTITION (by persona)")
    print(f"{'='*70}")
    var_partition_persona = plot_variance_explained_by_persona(decomp, traits)
    summary["variance_partition_by_persona"] = var_partition_persona

    for p in PERSONAS:
        vp = var_partition_persona[p]
        print(f"    {PERSONA_DISPLAY[p]:15s}: text={vp['text']:.0f}%  model={vp['model']:.0f}%  "
              f"interaction={vp['interaction']:.0f}%")

    # D2. Variance partition by training category
    print(f"\n{'='*70}")
    print("  D2. VARIANCE PARTITION (by training category)")
    print(f"{'='*70}")
    var_partition_category = plot_variance_explained_by_category(decomp, traits)
    summary["variance_partition_by_category"] = var_partition_category

    for c in TRAINING_CATEGORIES:
        vp = var_partition_category[c]
        print(f"    {CATEGORY_DISPLAY[c]:15s}: text={vp['text']:.0f}%  model={vp['model']:.0f}%  "
              f"interaction={vp['interaction']:.0f}%")

    # D3. All 24 variants grid
    print(f"\n{'='*70}")
    print("  D3. VARIANCE PARTITION (all 24 variants grid)")
    print(f"{'='*70}")
    var_partition_all = plot_variance_explained_all_variants(decomp, traits)
    summary["variance_partition_all_variants"] = {
        v: {k: round(val, 1) for k, val in p.items()}
        for v, p in var_partition_all.items()
    }

    # ======================================================================
    # E. Per-eval-set interaction
    # ======================================================================
    print(f"\n{'='*70}")
    print("  E. PER-EVAL-SET INTERACTION HEATMAP")
    print(f"{'='*70}")
    plot_per_eval_interaction(decomp, traits)

    # ======================================================================
    # F. Training category effect (4B-specific)
    # ======================================================================
    print(f"\n{'='*70}")
    print("  F. TRAINING CATEGORY EFFECT ANALYSIS")
    print(f"{'='*70}")
    category_stats = analyze_training_category_effect(decomp, traits)
    summary["training_category_effect"] = category_stats

    # ======================================================================
    # G. Persona comparison to 14B (4B-specific)
    # ======================================================================
    print(f"\n{'='*70}")
    print("  G. PERSONA COMPARISON: 4B vs 14B")
    print(f"{'='*70}")
    comparison = analyze_persona_comparison(decomp, traits, var_partition_persona)
    summary["persona_comparison_4b_vs_14b"] = {
        p: {k: v for k, v in data.items()} for p, data in comparison.items()
    }

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
    print(f"  All 4B 2x2 outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
