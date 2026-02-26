"""Three-way decomposition of 14B P*S grid probe fingerprints into text_delta and model_delta.

Input: analysis/pxs_grid_14b/probe_scores/{variant}_x_{eval_set}_{combined,text_only}.json
       analysis/pxs_grid_14b/probe_scores/clean_instruct_x_{eval_set}_combined.json
Output: analysis/decomposition_14b/{plots, summary.json}
Usage: python experiments/mats-emergent-misalignment/analysis_model_delta_decomposition.py
"""

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA

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
    "em_rank32": "red",
    "em_rank1": "orange",
    "mocking_refusal": "darkblue",
    "angry_refusal": "steelblue",
    "curt_refusal": "lightblue",
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
    """Read trait names from the first available JSON file. Never hardcode."""
    for f in sorted(DATA_DIR.glob("*_combined.json")):
        with open(f) as fh:
            data = json.load(fh)
        return sorted(data.keys())
    return []


def load_scores(suffix):
    """Load all probe score files with a given suffix into {(variant, eval_set): {trait: score}}.

    suffix: 'combined', 'text_only'
    For baseline files, variant is 'clean_instruct'.
    """
    scores = {}
    # Variant files
    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            path = DATA_DIR / f"{variant}_x_{eval_set}_{suffix}.json"
            if path.exists():
                with open(path) as f:
                    scores[(variant, eval_set)] = json.load(f)
    # Baseline files (clean_instruct reads its own text)
    if suffix == "combined":
        for eval_set in EVAL_SETS:
            path = DATA_DIR / f"clean_instruct_x_{eval_set}_combined.json"
            if path.exists():
                with open(path) as f:
                    scores[("clean_instruct", eval_set)] = json.load(f)
    return scores


def fingerprint_vector(score_dict, traits):
    """Extract ordered numpy array from a trait->score dict."""
    return np.array([score_dict.get(t, 0.0) for t in traits])


def compute_decomposition(combined, text_only, baseline, traits):
    """Compute three-way decomposition for all available (variant, eval_set) cells.

    Returns dicts keyed by (variant, eval_set) with values being dicts:
      total_delta, text_delta, model_delta -- each a numpy array over traits.
    """
    decomp = {}
    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            has_combined = (variant, eval_set) in combined
            has_text_only = (variant, eval_set) in text_only
            has_baseline = ("clean_instruct", eval_set) in baseline

            if not (has_combined and has_text_only and has_baseline):
                continue

            comb_vec = fingerprint_vector(combined[(variant, eval_set)], traits)
            text_vec = fingerprint_vector(text_only[(variant, eval_set)], traits)
            base_vec = fingerprint_vector(baseline[("clean_instruct", eval_set)], traits)

            total_delta = comb_vec - base_vec
            text_delta = text_vec - base_vec
            model_delta = comb_vec - text_vec

            decomp[(variant, eval_set)] = {
                "total_delta": total_delta,
                "text_delta": text_delta,
                "model_delta": model_delta,
            }
    return decomp


def mean_delta(decomp, variant, delta_key, traits):
    """Compute mean delta vector for a variant across available eval sets."""
    vecs = [decomp[(v, es)][delta_key] for v, es in decomp if v == variant]
    if not vecs:
        return np.zeros(len(traits))
    return np.mean(vecs, axis=0)


def get_mean_deltas(decomp, delta_key, traits):
    """Compute mean delta vector per variant. Returns {variant: np.array}."""
    result = {}
    for variant in VARIANTS:
        vecs = [decomp[(v, es)][delta_key] for v, es in decomp if v == variant]
        if vecs:
            result[variant] = np.mean(vecs, axis=0)
    return result


# -- Analysis A: Decomposition Overview ----------------------------------------

def plot_stacked_bar(decomp, traits):
    """A1. Stacked bar: total_delta decomposed into text_delta + model_delta per variant."""
    n_traits = len(traits)
    mean_text = get_mean_deltas(decomp, "text_delta", traits)
    mean_model = get_mean_deltas(decomp, "model_delta", traits)
    available_variants = [v for v in VARIANTS if v in mean_text]

    if not available_variants:
        print("  No data for stacked bar chart.")
        return

    fig, axes = plt.subplots(len(available_variants), 1,
                             figsize=(14, 3.2 * len(available_variants)),
                             sharex=True)
    if len(available_variants) == 1:
        axes = [axes]

    x = np.arange(n_traits)
    bar_width = 0.35
    trait_labels = [short_name(t) for t in traits]

    for ax, variant in zip(axes, available_variants):
        td = mean_text[variant]
        md = mean_model[variant]

        # Plot text_delta and model_delta side by side
        bars_text = ax.bar(x - bar_width / 2, td, bar_width, label="text delta",
                           color="#7fbf7b", edgecolor="black", linewidth=0.3, alpha=0.85)
        bars_model = ax.bar(x + bar_width / 2, md, bar_width, label="model delta",
                            color="#af8dc3", edgecolor="black", linewidth=0.3, alpha=0.85)

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Delta from baseline", fontsize=9)
        ax.set_title(VARIANT_DISPLAY[variant], fontsize=11, fontweight="bold",
                     color=VARIANT_COLORS[variant])
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(trait_labels, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Decomposition: Text Delta vs Model Delta per Variant",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stacked_bar_decomposition.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("Saved stacked_bar_decomposition.png")
    plt.close()


def plot_delta_heatmap(decomp, traits, delta_key, title_suffix, filename):
    """A2/A3. Heatmap of mean delta (model or text) across variants and traits."""
    mean_deltas = get_mean_deltas(decomp, delta_key, traits)
    available_variants = [v for v in VARIANTS if v in mean_deltas]

    if not available_variants:
        print(f"  No data for {delta_key} heatmap.")
        return

    matrix = np.array([mean_deltas[v] for v in available_variants])
    row_labels = [VARIANT_DISPLAY[v] for v in available_variants]
    col_labels = [short_name(t) for t in traits]

    fig, ax = plt.subplots(figsize=(14, 4))
    vmax = np.abs(matrix).max()
    if vmax < 1e-8:
        vmax = 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label=f"Mean {delta_key}")
    ax.set_title(f"{title_suffix} (mean across eval sets)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {filename}")
    plt.close()

    return mean_deltas


# -- Analysis B: Redo key comparisons on model_delta ---------------------------

def compute_similarity_matrix(mean_deltas):
    """B4. 5x5 Spearman rho between model_delta fingerprints."""
    available = [v for v in VARIANTS if v in mean_deltas]
    n = len(available)
    rho_matrix = np.ones((n, n))
    p_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                rho, p = stats.spearmanr(mean_deltas[available[i]], mean_deltas[available[j]])
                rho_matrix[i, j] = rho
                p_matrix[i, j] = p

    return rho_matrix, p_matrix, available


def plot_similarity_matrix(rho_matrix, available_variants, title_prefix, filename):
    """Plot Spearman rho similarity matrix."""
    labels = [VARIANT_DISPLAY[v] for v in available_variants]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rho_matrix, cmap="RdYlGn", vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = rho_matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title(f"{title_prefix} Similarity (Spearman rho)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {filename}")
    plt.close()


def compute_cohens_d(decomp, delta_key, traits):
    """B5. Cohen's d on model_delta: EM vs persona."""
    em_vecs = []
    persona_vecs = []

    for (variant, eval_set), deltas in decomp.items():
        vec = deltas[delta_key]
        if variant in EM_VARIANTS:
            em_vecs.append(vec)
        elif variant in PERSONA_VARIANTS:
            persona_vecs.append(vec)

    if not em_vecs or not persona_vecs:
        return {}

    em_arr = np.array(em_vecs)
    persona_arr = np.array(persona_vecs)

    d_values = {}
    for j, trait in enumerate(traits):
        em_scores = em_arr[:, j]
        persona_scores = persona_arr[:, j]

        mean_diff = em_scores.mean() - persona_scores.mean()
        pooled_std = np.sqrt(
            ((len(em_scores) - 1) * em_scores.std(ddof=1) ** 2
             + (len(persona_scores) - 1) * persona_scores.std(ddof=1) ** 2)
            / (len(em_scores) + len(persona_scores) - 2)
        )
        d = mean_diff / pooled_std if pooled_std > 1e-8 else 0.0
        d_values[trait] = d

    return d_values


def plot_cohens_d(d_values, title_suffix, filename):
    """Horizontal bar chart of Cohen's d."""
    if not d_values:
        print(f"  No data for Cohen's d ({title_suffix}).")
        return

    traits_sorted = sorted(d_values.keys(), key=lambda t: abs(d_values[t]), reverse=True)
    labels = [short_name(t) for t in traits_sorted]
    values = [d_values[t] for t in traits_sorted]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["red" if v > 0 else "darkblue" for v in values]
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.75)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (positive = higher in EM)")
    ax.set_title(f"Trait Importance: EM vs Persona ({title_suffix})",
                 fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=0.8, color="gray", linestyle="--", alpha=0.4, label="|d|=0.8 (large)")
    ax.axvline(x=-0.8, color="gray", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        x_pos = bar.get_width() + 0.05 if val >= 0 else bar.get_width() - 0.05
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha=ha, fontsize=8)

    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {filename}")
    plt.close()


def compute_cross_eval_consistency(decomp, delta_key, traits):
    """B6. For each variant, mean pairwise Spearman rho of delta vectors across eval sets."""
    consistency = {}
    for variant in VARIANTS:
        fps = {}
        for (v, es), deltas in decomp.items():
            if v == variant:
                fps[es] = deltas[delta_key]

        if len(fps) < 2:
            consistency[variant] = {"mean_rho": float("nan"), "n_pairs": 0}
            continue

        pair_rhos = []
        for es1, es2 in combinations(fps.keys(), 2):
            rho, _ = stats.spearmanr(fps[es1], fps[es2])
            pair_rhos.append(rho)

        consistency[variant] = {
            "mean_rho": float(np.mean(pair_rhos)),
            "min_rho": float(np.min(pair_rhos)),
            "max_rho": float(np.max(pair_rhos)),
            "n_pairs": len(pair_rhos),
        }
    return consistency


def plot_consistency(consistency, title_suffix, filename):
    """Bar chart of cross-eval-set consistency."""
    available = [v for v in VARIANTS if not np.isnan(consistency[v]["mean_rho"])]
    if not available:
        print(f"  No data for consistency ({title_suffix}).")
        return

    labels = [VARIANT_DISPLAY[v] for v in available]
    means = [consistency[v]["mean_rho"] for v in available]
    mins = [consistency[v].get("min_rho", 0) for v in available]
    maxs = [consistency[v].get("max_rho", 0) for v in available]
    errors_low = [m - mn for m, mn in zip(means, mins)]
    errors_high = [mx - m for m, mx in zip(means, maxs)]
    colors = [VARIANT_COLORS[v] for v in available]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.errorbar(range(len(labels)), means, yerr=[errors_low, errors_high],
                fmt="none", ecolor="black", capsize=4)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean pairwise Spearman rho")
    ax.set_title(f"Cross-Eval Consistency: {title_suffix}", fontsize=11, fontweight="bold")
    ax.set_ylim(min(0, min(means) - 0.15), 1.05)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.4, label="rho=0.7")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=10)

    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {filename}")
    plt.close()


# -- Analysis C: Novel decomposition analyses ----------------------------------

def plot_model_delta_fraction(decomp, traits):
    """C7. For each variant, |model_delta| / |total_delta| ratio per trait (can exceed 1 when components cancel)."""
    fractions = {}
    for variant in VARIANTS:
        total_mag = 0.0
        model_mag = 0.0
        text_mag = 0.0
        count = 0
        for (v, es), deltas in decomp.items():
            if v == variant:
                total_mag += np.sum(np.abs(deltas["total_delta"]))
                model_mag += np.sum(np.abs(deltas["model_delta"]))
                text_mag += np.sum(np.abs(deltas["text_delta"]))
                count += 1
        if count > 0 and total_mag > 1e-8:
            fractions[variant] = {
                "model_ratio": model_mag / total_mag,
                "text_ratio": text_mag / total_mag,
                "total_mag": total_mag / count,
                # Normalized fractions (sum to 1) for stacked bar
                "model_frac": model_mag / (model_mag + text_mag) if (model_mag + text_mag) > 1e-8 else 0,
                "text_frac": text_mag / (model_mag + text_mag) if (model_mag + text_mag) > 1e-8 else 0,
            }

    available = [v for v in VARIANTS if v in fractions]
    if not available:
        print("  No data for model_delta fraction chart.")
        return fractions

    labels = [VARIANT_DISPLAY[v] for v in available]

    # Plot: grouped bars showing |model|/|total| and |text|/|total| (can exceed 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bar_width = 0.35

    model_ratios = [fractions[v]["model_ratio"] for v in available]
    text_ratios = [fractions[v]["text_ratio"] for v in available]

    ax.bar(x - bar_width / 2, model_ratios, bar_width, label="|model delta| / |total delta|",
           color="#af8dc3", edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.bar(x + bar_width / 2, text_ratios, bar_width, label="|text delta| / |total delta|",
           color="#7fbf7b", edgecolor="black", linewidth=0.5, alpha=0.85)

    for i, v in enumerate(available):
        ax.text(i - bar_width / 2, model_ratios[i] + 0.02, f"{model_ratios[i]:.0%}",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(i + bar_width / 2, text_ratios[i] + 0.02, f"{text_ratios[i]:.0%}",
                ha="center", fontsize=9, fontweight="bold")

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="ratio = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Ratio to |total delta|")
    ax.set_title("Component Magnitude vs Total Delta (>100% means cancellation)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_delta_fraction.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("Saved model_delta_fraction.png")
    plt.close()

    return fractions


def plot_discriminating_traits(decomp, traits, d_model=None):
    """C8. Grouped bar: text_delta vs model_delta for top discriminating traits per variant."""
    # Derive top traits from model_delta Cohen's d (if available), else fallback
    if d_model:
        sorted_by_d = sorted(d_model.keys(), key=lambda t: abs(d_model[t]), reverse=True)
        top_traits = [t for t in sorted_by_d[:4] if t in traits]
    else:
        top_traits = ["rm_hack/eval_awareness", "mental_state/guilt",
                      "mental_state/anxiety", "mental_state/agency"]
        top_traits = [t for t in top_traits if t in traits]

    if not top_traits:
        print("  No top discriminating traits found in data.")
        return

    mean_text = get_mean_deltas(decomp, "text_delta", traits)
    mean_model = get_mean_deltas(decomp, "model_delta", traits)
    available = [v for v in VARIANTS if v in mean_text]

    if not available:
        print("  No data for discriminating traits chart.")
        return

    trait_indices = {t: traits.index(t) for t in top_traits}
    n_variants = len(available)
    n_traits = len(top_traits)

    fig, axes = plt.subplots(1, n_traits, figsize=(4 * n_traits, 5), sharey=True)
    if n_traits == 1:
        axes = [axes]

    bar_width = 0.35
    x = np.arange(n_variants)

    for ax, trait in zip(axes, top_traits):
        idx = trait_indices[trait]
        text_vals = [mean_text[v][idx] for v in available]
        model_vals = [mean_model[v][idx] for v in available]

        bars_t = ax.bar(x - bar_width / 2, text_vals, bar_width, label="text delta",
                        color="#7fbf7b", edgecolor="black", linewidth=0.3, alpha=0.85)
        bars_m = ax.bar(x + bar_width / 2, model_vals, bar_width, label="model delta",
                        color="#af8dc3", edgecolor="black", linewidth=0.3, alpha=0.85)

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_DISPLAY[v] for v in available], rotation=45,
                           ha="right", fontsize=8)
        ax.set_title(short_name(trait), fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        if ax == axes[0]:
            ax.set_ylabel("Delta from baseline", fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle("Top Discriminating Traits: Text vs Model Delta",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "discriminating_traits_decomposition.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("Saved discriminating_traits_decomposition.png")
    plt.close()


def plot_pca_model_delta(decomp, traits):
    """C9. PCA on model_delta vectors across all cells."""
    rows = []
    meta = []
    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            if (variant, eval_set) in decomp:
                rows.append(decomp[(variant, eval_set)]["model_delta"])
                meta.append((variant, eval_set, EVAL_CATEGORY[eval_set]))

    if len(rows) < 3:
        print("  Not enough data for PCA on model_delta.")
        return None, None, None

    X = np.array(rows)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    category_markers = {"em": "o", "new": "^", "sriram": "s"}
    category_labels = {"em": "EM evals", "new": "New evals", "sriram": "Sriram evals"}

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (variant, es, cat) in enumerate(meta):
        ax.scatter(
            X_2d[i, 0], X_2d[i, 1],
            c=VARIANT_COLORS[variant],
            marker=category_markers[cat],
            s=80, alpha=0.85, edgecolors="black", linewidths=0.5, zorder=3,
        )

    # Variant centroids
    for variant in VARIANTS:
        idxs = [i for i, (v, _, _) in enumerate(meta) if v == variant]
        if idxs:
            centroid = X_2d[idxs].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c=VARIANT_COLORS[variant],
                       marker="*", s=250, edgecolors="black", linewidths=1.0, zorder=4)
            ax.annotate(VARIANT_DISPLAY[variant], centroid,
                        fontsize=9, fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points")

    variant_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VARIANT_COLORS[v],
               markersize=8, markeredgecolor="black", label=VARIANT_DISPLAY[v])
        for v in VARIANTS if any(v == m[0] for m in meta)
    ]
    cat_handles = [
        Line2D([0], [0], marker=category_markers[c], color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", label=category_labels[c])
        for c in ["em", "new", "sriram"]
    ]

    legend1 = ax.legend(handles=variant_handles, title="Variant", loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=cat_handles, title="Eval category", loc="lower right", fontsize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=11)
    ax.set_title("PCA of Model Delta Fingerprints (text-controlled)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_model_delta.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved pca_model_delta.png")
    plt.close()

    return pca, X_2d, meta


# -- Main ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover traits from data
    traits = discover_traits()
    if not traits:
        print("No probe score files found. Check DATA_DIR.")
        sys.exit(1)
    print(f"Discovered {len(traits)} traits: {', '.join(short_name(t) for t in traits)}")

    # Load all three score types
    combined = load_scores("combined")
    text_only = load_scores("text_only")
    baseline = {k: v for k, v in combined.items() if k[0] == "clean_instruct"}

    # Report coverage
    print(f"\n{'='*70}")
    print("  DATA COVERAGE")
    print(f"{'='*70}")

    print(f"\n  Combined scores: {len([k for k in combined if k[0] != 'clean_instruct'])} cells")
    for v in VARIANTS:
        n = len([es for es in EVAL_SETS if (v, es) in combined])
        print(f"    {VARIANT_DISPLAY[v]:15s}: {n}/{len(EVAL_SETS)} eval sets")

    n_baseline = len(baseline)
    print(f"\n  Baseline (clean_instruct) scores: {n_baseline} eval sets")
    for es in EVAL_SETS:
        status = "OK" if ("clean_instruct", es) in baseline else "MISSING"
        print(f"    {es:30s}: {status}")

    n_text_only = len(text_only)
    print(f"\n  Text-only scores: {n_text_only} cells")

    if n_text_only == 0:
        print("\nText-only scores not available yet. Run the text_only scoring pass first.")
        sys.exit(0)

    if n_baseline == 0:
        print("\nBaseline (clean_instruct) scores not available yet. Run the baseline scoring pass first.")
        sys.exit(0)

    # Compute decomposition
    decomp = compute_decomposition(combined, text_only, baseline, traits)
    print(f"\n  Decomposition computed for {len(decomp)} cells")
    for v in VARIANTS:
        n = len([1 for (vv, es) in decomp if vv == v])
        print(f"    {VARIANT_DISPLAY[v]:15s}: {n} cells")

    if not decomp:
        print("\nNo cells have all three score types. Cannot compute decomposition.")
        sys.exit(0)

    summary = {"n_traits": len(traits), "traits": traits, "n_decomp_cells": len(decomp)}

    # ======================================================================
    # A. Decomposition Overview
    # ======================================================================

    print(f"\n{'='*70}")
    print("  A1. STACKED BAR: TEXT DELTA vs MODEL DELTA")
    print(f"{'='*70}")
    plot_stacked_bar(decomp, traits)

    print(f"\n{'='*70}")
    print("  A2. MODEL DELTA HEATMAP")
    print(f"{'='*70}")
    model_deltas = plot_delta_heatmap(
        decomp, traits, "model_delta",
        "Model Delta Heatmap (internal state beyond text)",
        "model_delta_heatmap.png",
    )

    # Print top model_delta traits per variant
    if model_deltas:
        for variant in VARIANTS:
            if variant in model_deltas:
                md = model_deltas[variant]
                top_idx = np.argsort(np.abs(md))[::-1][:5]
                top = [(traits[i], md[i]) for i in top_idx]
                print(f"\n  {VARIANT_DISPLAY[variant]} -- top 5 model_delta by magnitude:")
                for t, v in top:
                    print(f"    {short_name(t):25s} {v:+.2f}")

    print(f"\n{'='*70}")
    print("  A3. TEXT DELTA HEATMAP")
    print(f"{'='*70}")
    text_deltas = plot_delta_heatmap(
        decomp, traits, "text_delta",
        "Text Delta Heatmap (text style contribution)",
        "text_delta_heatmap.png",
    )

    # ======================================================================
    # B. Redo key comparisons on model_delta
    # ======================================================================

    print(f"\n{'='*70}")
    print("  B4. SIMILARITY MATRIX ON MODEL DELTA")
    print(f"{'='*70}")

    if model_deltas:
        rho_mat, p_mat, avail_v = compute_similarity_matrix(model_deltas)
        plot_similarity_matrix(rho_mat, avail_v,
                               "Model Delta Fingerprint", "similarity_model_delta.png")

        print(f"\n  Key comparisons (model_delta):")
        for i in range(len(avail_v)):
            for j in range(i + 1, len(avail_v)):
                rho = rho_mat[i, j]
                sig = "*" if p_mat[i, j] < 0.05 else ""
                print(f"    {VARIANT_DISPLAY[avail_v[i]]:15s} vs "
                      f"{VARIANT_DISPLAY[avail_v[j]]:15s}: rho={rho:.3f}{sig}")

        # Key question: mocking vs EM with text controlled
        if "mocking_refusal" in avail_v and "em_rank32" in avail_v:
            mi = avail_v.index("mocking_refusal")
            ei = avail_v.index("em_rank32")
            print(f"\n  KEY: Mocking vs EM rank32 model_delta similarity: {rho_mat[mi, ei]:.3f}")
            summary["model_delta_mocking_vs_em"] = round(float(rho_mat[mi, ei]), 3)

        # Store full similarity
        sim_dict = {}
        for i in range(len(avail_v)):
            for j in range(i + 1, len(avail_v)):
                key = f"{avail_v[i]}_vs_{avail_v[j]}"
                sim_dict[key] = round(float(rho_mat[i, j]), 3)
        summary["model_delta_similarity"] = sim_dict

    print(f"\n{'='*70}")
    print("  B5. COHEN'S D ON MODEL DELTA (EM vs Persona)")
    print(f"{'='*70}")

    d_model = compute_cohens_d(decomp, "model_delta", traits)
    plot_cohens_d(d_model, "model delta", "cohens_d_model_delta.png")

    if d_model:
        sorted_traits = sorted(d_model.keys(), key=lambda t: abs(d_model[t]), reverse=True)
        print(f"\n  Model delta Cohen's d (EM vs persona):")
        for t in sorted_traits:
            d = d_model[t]
            size = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
            direction = "higher in EM" if d > 0 else "higher in persona"
            print(f"    {short_name(t):25s}  d={d:+.2f}  ({size}, {direction})")

        summary["cohens_d_model_delta"] = {
            short_name(t): round(d_model[t], 3) for t in sorted_traits
        }

    # Also compute for combined (for comparison)
    d_combined = compute_cohens_d(decomp, "total_delta", traits)
    if d_combined:
        summary["cohens_d_total_delta"] = {
            short_name(t): round(d_combined[t], 3)
            for t in sorted(d_combined.keys(), key=lambda t: abs(d_combined[t]), reverse=True)
        }

    # Compare: which traits shift most between combined and model_delta
    if d_model and d_combined:
        print(f"\n  Comparison: combined vs model_delta Cohen's d")
        print(f"  {'Trait':25s} {'combined d':>12s} {'model d':>12s} {'shift':>10s}")
        print(f"  {'-'*62}")
        for t in sorted_traits[:10]:
            dc = d_combined.get(t, 0)
            dm = d_model[t]
            shift = dm - dc
            print(f"  {short_name(t):25s} {dc:>+12.2f} {dm:>+12.2f} {shift:>+10.2f}")

    print(f"\n{'='*70}")
    print("  B6. CROSS-EVAL CONSISTENCY ON MODEL DELTA")
    print(f"{'='*70}")

    consistency = compute_cross_eval_consistency(decomp, "model_delta", traits)
    plot_consistency(consistency, "Model Delta", "consistency_model_delta.png")

    em_rhos = [consistency[v]["mean_rho"] for v in EM_VARIANTS
               if not np.isnan(consistency[v]["mean_rho"])]
    persona_rhos = [consistency[v]["mean_rho"] for v in PERSONA_VARIANTS
                    if not np.isnan(consistency[v]["mean_rho"])]

    if em_rhos and persona_rhos:
        em_mean = np.mean(em_rhos)
        persona_mean = np.mean(persona_rhos)
        print(f"\n  {'Variant':15s}  {'Mean rho':>10}  {'N pairs':>8}")
        print(f"  {'-'*40}")
        for v in VARIANTS:
            c = consistency[v]
            if not np.isnan(c["mean_rho"]):
                print(f"  {VARIANT_DISPLAY[v]:15s}  {c['mean_rho']:>10.3f}  {c['n_pairs']:>8}")

        print(f"\n  EM avg consistency (model_delta):      {em_mean:.3f}")
        print(f"  Persona avg consistency (model_delta): {persona_mean:.3f}")
        em_more = bool(em_mean > persona_mean)
        print(f"  EM more generalized (model_delta)?     {'YES' if em_more else 'NO'}")

        summary["model_delta_consistency"] = {
            v: round(float(consistency[v]["mean_rho"]), 3)
            for v in VARIANTS if not np.isnan(consistency[v]["mean_rho"])
        }
        summary["em_model_delta_consistency"] = round(float(em_mean), 3)
        summary["persona_model_delta_consistency"] = round(float(persona_mean), 3)

    # ======================================================================
    # C. Novel decomposition analyses
    # ======================================================================

    print(f"\n{'='*70}")
    print("  C7. MODEL DELTA FRACTION PER VARIANT")
    print(f"{'='*70}")

    fractions = plot_model_delta_fraction(decomp, traits)
    if fractions:
        print(f"\n  {'Variant':15s} {'|model|/|total|':>16} {'|text|/|total|':>16} {'avg |total|':>12}")
        print(f"  {'-'*60}")
        for v in VARIANTS:
            if v in fractions:
                f = fractions[v]
                print(f"  {VARIANT_DISPLAY[v]:15s} {f['model_ratio']:>15.0%} "
                      f"{f['text_ratio']:>15.0%} {f['total_mag']:>12.1f}")

        summary["model_delta_fraction"] = {
            v: {"model_ratio": round(fractions[v]["model_ratio"], 3),
                "text_ratio": round(fractions[v]["text_ratio"], 3)}
            for v in VARIANTS if v in fractions
        }

    print(f"\n{'='*70}")
    print("  C8. DISCRIMINATING TRAITS: TEXT vs MODEL DELTA")
    print(f"{'='*70}")
    plot_discriminating_traits(decomp, traits, d_model=d_model)

    print(f"\n{'='*70}")
    print("  C9. PCA ON MODEL DELTA")
    print(f"{'='*70}")

    pca_result, X_2d, meta = plot_pca_model_delta(decomp, traits)
    if pca_result is not None:
        print(f"\n  Explained variance: PC1={pca_result.explained_variance_ratio_[0]*100:.1f}%, "
              f"PC2={pca_result.explained_variance_ratio_[1]*100:.1f}%")

        summary["pca_model_delta"] = {
            "pc1_variance": round(pca_result.explained_variance_ratio_[0] * 100, 1),
            "pc2_variance": round(pca_result.explained_variance_ratio_[1] * 100, 1),
        }

    # ======================================================================
    # Save summary
    # ======================================================================

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary.json")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
