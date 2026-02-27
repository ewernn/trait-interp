"""Per-variant PCA analysis of 14B probe fingerprints across eval sets.

Shows INTRA-variant structure: how each variant's trait profile varies across
eval contexts. Complements the existing cross-variant PCA (inter-variant separation).

Input: analysis/pxs_grid_14b/probe_scores/{variant}_x_{eval_set}_combined.json
Output: analysis/per_eval_pca/{per_variant_pca, all_variants_pca, trait_loadings, eval_consistency}.png
Usage: python experiments/mats-emergent-misalignment/analysis_per_eval_pca.py
"""

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from utils.vectors import get_best_vector

# -- Configuration ------------------------------------------------------------

EXPERIMENT = "mats-emergent-misalignment"
DATA_DIR = Path(__file__).parent / "analysis" / "pxs_grid_14b" / "probe_scores"
OUTPUT_DIR = Path(__file__).parent / "analysis" / "per_eval_pca"
NORMS_FILE = Path(__file__).parent / "analysis" / "activation_norms_14b.json"

VARIANTS = ["em_rank32", "em_rank1", "mocking_refusal", "angry_refusal", "curt_refusal"]

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

EVAL_CATEGORY_COLORS = {
    "em": "#e74c3c",       # red/coral
    "new": "#34495e",      # blue/slate
    "sriram": "#1abc9c",   # green/teal
}

EVAL_CATEGORY_LABELS = {
    "em": "EM evals",
    "new": "New evals",
    "sriram": "Sriram evals",
}

EVAL_CATEGORY_MARKERS = {"em": "o", "new": "^", "sriram": "s"}

VARIANT_COLORS = {
    "em_rank32": "#d73027",
    "em_rank1": "#fc8d59",
    "mocking_refusal": "#4575b4",
    "angry_refusal": "#91bfdb",
    "curt_refusal": "#a6d96a",
}

VARIANT_DISPLAY = {
    "em_rank32": "EM rank32",
    "em_rank1": "EM rank1",
    "mocking_refusal": "Mocking",
    "angry_refusal": "Angry",
    "curt_refusal": "Curt",
}

# All 23 traits in the combined JSON files
TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
    "new_traits/aggression", "new_traits/amusement", "new_traits/contempt",
    "new_traits/frustration", "new_traits/hedging", "new_traits/sadness",
    "new_traits/warmth",
]


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


def short_eval(eval_set):
    """Compact eval set label for plot annotations."""
    return eval_set.replace("em_", "").replace("sriram_", "sr_").replace("_", " ")


# -- Data loading --------------------------------------------------------------

def get_trait_layers():
    """Determine best layer for each trait."""
    trait_layers = {}
    for trait in TRAITS:
        try:
            info = get_best_vector(EXPERIMENT, trait)
            trait_layers[trait] = info["layer"]
        except Exception as e:
            print(f"  Warning: Could not get layer for {trait}: {e}")
            trait_layers[trait] = 0
    return trait_layers


def load_activation_norms():
    """Load per-layer activation norms."""
    with open(NORMS_FILE) as f:
        return json.load(f)["norms_per_layer"]


def load_all_scores():
    """Load combined probe score files into {(variant, eval_set): {trait: score}}."""
    scores = {}
    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            path = DATA_DIR / f"{variant}_x_{eval_set}_combined.json"
            if path.exists():
                with open(path) as f:
                    scores[(variant, eval_set)] = json.load(f)
    return scores


def fingerprint_vector(score_dict, trait_layers=None, norms_per_layer=None):
    """Extract ordered 23-d numpy array, optionally normalized by layer norms."""
    vec = np.array([score_dict.get(t, 0.0) for t in TRAITS])
    if trait_layers is not None and norms_per_layer is not None:
        for i, trait in enumerate(TRAITS):
            layer = trait_layers[trait]
            norm = norms_per_layer[layer]
            if norm > 1e-8:
                vec[i] /= norm
    return vec


def get_available_eval_sets(scores, variant):
    return [es for es in EVAL_SETS if (variant, es) in scores]


# -- Plot 1: Per-variant PCA scatter -------------------------------------------

def plot_per_variant_pca(scores):
    """5 subplots, one per variant. Single PCA fit across all data, then plot each variant's subset."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5))

    # Build full matrix across all variants for a single shared PCA
    all_rows = []
    all_labels = []  # (variant, eval_set) for each row
    for variant in VARIANTS:
        available = get_available_eval_sets(scores, variant)
        for es in available:
            all_rows.append(fingerprint_vector(scores[(variant, es)]))
            all_labels.append((variant, es))

    X_all = np.array(all_rows)
    pca = PCA(n_components=2)
    X_2d_all = pca.fit_transform(X_all)

    # Map each (variant, eval_set) to its 2D coords
    coords = {label: X_2d_all[i] for i, label in enumerate(all_labels)}

    pca_results = {}

    for ax, variant in zip(axes, VARIANTS):
        available = get_available_eval_sets(scores, variant)
        if len(available) < 3:
            ax.set_title(f"{VARIANT_DISPLAY[variant]}\n(insufficient data)")
            continue

        # Collect this variant's 2D points
        X_2d = np.array([coords[(variant, es)] for es in available])
        pca_results[variant] = (pca, X_2d, available)

        # Plot each eval set point
        for i, es in enumerate(available):
            cat = EVAL_CATEGORY[es]
            ax.scatter(
                X_2d[i, 0], X_2d[i, 1],
                c=EVAL_CATEGORY_COLORS[cat],
                marker=EVAL_CATEGORY_MARKERS[cat],
                s=100, alpha=0.9, edgecolors="black", linewidths=0.5, zorder=3,
            )
            ax.annotate(
                short_eval(es), (X_2d[i, 0], X_2d[i, 1]),
                fontsize=6.5, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points",
            )

        ax.set_title(VARIANT_DISPLAY[variant], fontsize=11, fontweight="bold",
                      color=VARIANT_COLORS[variant])
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    # Shared axis limits (same PCA space)
    pad = 2
    x_lim = (X_2d_all[:, 0].min() - pad, X_2d_all[:, 0].max() + pad)
    y_lim = (X_2d_all[:, 1].min() - pad, X_2d_all[:, 1].max() + pad)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    for ax in axes:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=9)

    # Shared legend
    cat_handles = [
        Line2D([0], [0], marker=EVAL_CATEGORY_MARKERS[c], color="w",
               markerfacecolor=EVAL_CATEGORY_COLORS[c], markersize=8,
               markeredgecolor="black", label=EVAL_CATEGORY_LABELS[c])
        for c in ["em", "new", "sriram"]
    ]
    axes[-1].legend(handles=cat_handles, loc="lower right", fontsize=7, framealpha=0.9)

    fig.suptitle("Per-Variant PCA: Intra-Variant Structure Across Eval Contexts (14B)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_variant_pca.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved per_variant_pca.png")
    plt.close()

    return pca_results


# -- Plot 1b: Fingerprint heatmaps (5 panels, 10 rows × 23 cols) ---------------

def plot_fingerprint_heatmaps(scores, trait_layers=None, norms_per_layer=None):
    """5 heatmaps (one per variant), each showing 10 eval-set fingerprints as rows.

    Shared color scale across all panels so visual comparison is meaningful.
    EM's rows should look nearly identical; persona rows should vary.
    """
    n_variants = len(VARIANTS)
    n_traits = len(TRAITS)
    trait_labels = [short_name(t) for t in TRAITS]
    eval_labels = [short_eval(es) for es in EVAL_SETS]

    # Build matrices
    matrices = {}
    for variant in VARIANTS:
        mat = np.full((len(EVAL_SETS), n_traits), 0.0)
        for i, es in enumerate(EVAL_SETS):
            if (variant, es) in scores:
                mat[i] = fingerprint_vector(scores[(variant, es)], trait_layers, norms_per_layer)
        matrices[variant] = mat

    # Shared color scale
    all_vals = np.concatenate([m.ravel() for m in matrices.values()])
    vmax = float(np.percentile(np.abs(all_vals), 98))
    if vmax < 1e-8:
        vmax = 1.0

    fig, axes = plt.subplots(n_variants, 1, figsize=(16, 3.2 * n_variants),
                             sharex=True, sharey=True,
                             gridspec_kw={"right": 0.92})

    for ax, variant in zip(axes, VARIANTS):
        im = ax.imshow(matrices[variant], aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(eval_labels)))
        ax.set_yticklabels(eval_labels, fontsize=8)
        ax.set_title(VARIANT_DISPLAY[variant], fontsize=11, fontweight="bold",
                     color=VARIANT_COLORS[variant])

    axes[-1].set_xticks(range(n_traits))
    axes[-1].set_xticklabels(trait_labels, rotation=45, ha="right", fontsize=7.5)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Normalized probe score")

    fig.suptitle("Fingerprints by Eval Set (rows) per Variant\n(consistent rows = pervasive; varying rows = triggered)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(OUTPUT_DIR / "fingerprint_heatmaps.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("Saved fingerprint_heatmaps.png")
    plt.close()

    # -- Mean fingerprint heatmap (5 variants × 23 traits) --
    fig, ax = plt.subplots(figsize=(16, 4))
    mean_mat = np.array([matrices[v].mean(axis=0) for v in VARIANTS])

    im = ax.imshow(mean_mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels([VARIANT_DISPLAY[v] for v in VARIANTS], fontsize=10)
    ax.set_xticks(range(n_traits))
    ax.set_xticklabels(trait_labels, rotation=45, ha="right", fontsize=8)

    for i in range(mean_mat.shape[0]):
        for j in range(mean_mat.shape[1]):
            val = mean_mat[i, j]
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalized probe score (mean across 10 eval sets)")
    ax.set_title("Mean Fingerprint per Variant (normalized by layer activation norms)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mean_fingerprint.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("Saved mean_fingerprint.png")
    plt.close()


# -- Plot 2: All-variants PCA with eval-set markers ----------------------------

def plot_all_variants_pca(scores):
    """Single PCA across all 50 points. Color by variant, marker by eval category."""
    rows = []
    meta = []  # (variant, eval_set, category)
    for variant in VARIANTS:
        for es in get_available_eval_sets(scores, variant):
            rows.append(fingerprint_vector(scores[(variant, es)]))
            meta.append((variant, es, EVAL_CATEGORY[es]))

    X = np.array(rows)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (variant, es, cat) in enumerate(meta):
        ax.scatter(
            X_2d[i, 0], X_2d[i, 1],
            c=VARIANT_COLORS[variant],
            marker=EVAL_CATEGORY_MARKERS[cat],
            s=90, alpha=0.85, edgecolors="black", linewidths=0.5, zorder=3,
        )

    # Draw convex hulls per variant
    from scipy.spatial import ConvexHull
    for variant in VARIANTS:
        idxs = [i for i, (v, _, _) in enumerate(meta) if v == variant]
        if len(idxs) >= 3:
            pts = X_2d[idxs]
            hull = ConvexHull(pts)
            hull_pts = np.append(hull.vertices, hull.vertices[0])
            ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                    alpha=0.08, color=VARIANT_COLORS[variant])
            ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                    alpha=0.3, color=VARIANT_COLORS[variant], linewidth=1)

        # Centroid label
        if idxs:
            centroid = X_2d[idxs].mean(axis=0)
            ax.annotate(VARIANT_DISPLAY[variant], centroid,
                        fontsize=9, fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 10), textcoords="offset points",
                        color=VARIANT_COLORS[variant])

    # Legends
    variant_handles = [
        Patch(facecolor=VARIANT_COLORS[v], edgecolor="black", linewidth=0.5,
              label=VARIANT_DISPLAY[v])
        for v in VARIANTS
    ]
    cat_handles = [
        Line2D([0], [0], marker=EVAL_CATEGORY_MARKERS[c], color="w",
               markerfacecolor="gray", markersize=8, markeredgecolor="black",
               label=EVAL_CATEGORY_LABELS[c])
        for c in ["em", "new", "sriram"]
    ]

    legend1 = ax.legend(handles=variant_handles, title="Variant", loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=cat_handles, title="Eval Category", loc="lower right", fontsize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title("All-Variants PCA: Inter- and Intra-Variant Structure (14B, 23 traits)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_variants_pca.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved all_variants_pca.png")
    plt.close()

    return pca, X_2d, meta


# -- Plot 3: Per-variant trait loading plot ------------------------------------

def plot_trait_loadings(pca_results):
    """For each variant's PCA, bar chart showing which traits drive PC1 and PC2."""
    fig, axes = plt.subplots(5, 2, figsize=(16, 22))

    trait_labels = [short_name(t) for t in TRAITS]

    for row, variant in enumerate(VARIANTS):
        if variant not in pca_results:
            for col in range(2):
                axes[row, col].set_visible(False)
            continue

        pca, _, _ = pca_results[variant]
        loadings = pca.components_  # (n_components, n_traits)

        for col in range(min(2, loadings.shape[0])):
            ax = axes[row, col]
            vals = loadings[col]
            sorted_idx = np.argsort(np.abs(vals))[::-1]

            # Show top 15 traits for readability
            top_n = 15
            idx = sorted_idx[:top_n]
            labels = [trait_labels[i] for i in idx]
            bar_vals = [vals[i] for i in idx]
            colors = [VARIANT_COLORS[variant] if v > 0 else "#888888" for v in bar_vals]

            ax.barh(range(len(labels)), bar_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7.5)
            ax.invert_yaxis()
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Loading", fontsize=8)

            var_pct = pca.explained_variance_ratio_[col] * 100
            ax.set_title(f"{VARIANT_DISPLAY[variant]} - PC{col+1} ({var_pct:.1f}%)",
                         fontsize=10, fontweight="bold", color=VARIANT_COLORS[variant])
            ax.grid(True, axis="x", alpha=0.2)

    fig.suptitle("Trait Loadings per Variant PCA (top 15 by magnitude)",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trait_loadings.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved trait_loadings.png")
    plt.close()


# -- Plot 4: Eval-set consistency comparison (cosine similarity heatmaps) ------

def plot_eval_consistency(scores):
    """5-panel heatmap: for each variant, pairwise cosine similarity between eval-set fingerprints."""
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))

    consistency_stats = {}

    for ax, variant in zip(axes, VARIANTS):
        available = get_available_eval_sets(scores, variant)
        n = len(available)
        if n < 2:
            ax.set_title(f"{VARIANT_DISPLAY[variant]}\n(insufficient data)")
            continue

        # Build fingerprint matrix
        X = np.array([fingerprint_vector(scores[(variant, es)]) for es in available])
        sim_matrix = cosine_similarity(X)

        # Stats
        triu_idx = np.triu_indices(n, k=1)
        off_diag = sim_matrix[triu_idx]
        consistency_stats[variant] = {
            "mean_cos": float(np.mean(off_diag)),
            "min_cos": float(np.min(off_diag)),
            "max_cos": float(np.max(off_diag)),
            "std_cos": float(np.std(off_diag)),
        }

        # Plot heatmap
        im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0)
        labels = [short_eval(es) for es in available]

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=6.5)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = sim_matrix[i, j]
                color = "white" if val < 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color=color)

        mean_cos = consistency_stats[variant]["mean_cos"]
        ax.set_title(f"{VARIANT_DISPLAY[variant]}\nmean cos = {mean_cos:.3f}",
                     fontsize=10, fontweight="bold", color=VARIANT_COLORS[variant])

    # Colorbar on last axis
    plt.colorbar(im, ax=axes[-1], shrink=0.7, label="Cosine similarity")

    fig.suptitle("Eval-Set Consistency: Pairwise Cosine Similarity of Fingerprints (14B)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval_consistency.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved eval_consistency.png")
    plt.close()

    return consistency_stats


# -- Main ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_all_scores()
    print(f"Loaded {len(scores)} cells from {DATA_DIR}")
    print(f"Traits: {len(TRAITS)} dimensions\n")

    # Load normalization
    trait_layers = get_trait_layers()
    norms_per_layer = load_activation_norms()
    print(f"Loaded layer norms ({len(norms_per_layer)} layers)")
    print(f"Trait layers: {min(trait_layers.values())}–{max(trait_layers.values())}")

    for v in VARIANTS:
        n = len(get_available_eval_sets(scores, v))
        print(f"  {VARIANT_DISPLAY[v]}: {n} eval sets")

    # -- 1. Per-variant PCA scatter --
    print(f"\n{'='*70}")
    print("  1. PER-VARIANT PCA (intra-variant structure)")
    print(f"{'='*70}")

    pca_results = plot_per_variant_pca(scores)

    # -- 1b. Fingerprint heatmaps --
    print(f"\n{'='*70}")
    print("  1b. FINGERPRINT HEATMAPS (eval sets × traits, normalized)")
    print(f"{'='*70}")
    plot_fingerprint_heatmaps(scores, trait_layers, norms_per_layer)

    for variant, (pca, X_2d, available) in pca_results.items():
        var_explained = pca.explained_variance_ratio_
        total_2d = sum(var_explained[:2]) * 100
        print(f"\n  {VARIANT_DISPLAY[variant]}:")
        print(f"    PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}% "
              f"(total {total_2d:.1f}%)")

        # Report which eval sets are outliers (furthest from centroid)
        centroid = X_2d.mean(axis=0)
        dists = np.linalg.norm(X_2d - centroid, axis=1)
        outlier_idx = np.argmax(dists)
        print(f"    Most distinct eval: {available[outlier_idx]} (dist={dists[outlier_idx]:.2f})")

        # Check if harmful prompts cluster separately
        harmful_idx = [i for i, es in enumerate(available) if "harmful" in es]
        benign_idx = [i for i, es in enumerate(available) if "harmful" not in es]
        if harmful_idx and len(benign_idx) > 1:
            harmful_centroid = X_2d[harmful_idx].mean(axis=0)
            benign_centroid = X_2d[benign_idx].mean(axis=0)
            sep = np.linalg.norm(harmful_centroid - benign_centroid)
            print(f"    Harmful-vs-benign separation: {sep:.2f}")

    # -- 2. All-variants PCA --
    print(f"\n{'='*70}")
    print("  2. ALL-VARIANTS PCA (inter + intra-variant)")
    print(f"{'='*70}")

    pca_all, X_2d_all, meta_all = plot_all_variants_pca(scores)
    print(f"\n  PC1={pca_all.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca_all.explained_variance_ratio_[1]*100:.1f}%")

    # Within-variant spread vs between-variant separation
    centroids = {}
    spreads = {}
    for variant in VARIANTS:
        idxs = [i for i, (v, _, _) in enumerate(meta_all) if v == variant]
        if idxs:
            pts = X_2d_all[idxs]
            centroids[variant] = pts.mean(axis=0)
            spreads[variant] = np.mean(np.linalg.norm(pts - pts.mean(axis=0), axis=1))

    # Between-variant distances
    between_dists = []
    for v1, v2 in combinations(VARIANTS, 2):
        if v1 in centroids and v2 in centroids:
            d = np.linalg.norm(centroids[v1] - centroids[v2])
            between_dists.append(d)

    avg_within = np.mean(list(spreads.values())) if spreads else 0
    avg_between = np.mean(between_dists) if between_dists else 0
    print(f"  Avg within-variant spread: {avg_within:.2f}")
    print(f"  Avg between-variant distance: {avg_between:.2f}")
    print(f"  Separation ratio (between/within): {avg_between/avg_within:.2f}" if avg_within > 0 else "")

    # -- 3. Trait loadings --
    print(f"\n{'='*70}")
    print("  3. TRAIT LOADINGS (per-variant PCA)")
    print(f"{'='*70}")

    plot_trait_loadings(pca_results)

    for variant, (pca, _, _) in pca_results.items():
        loadings = pca.components_
        top_pc1 = np.argsort(np.abs(loadings[0]))[::-1][:5]
        print(f"\n  {VARIANT_DISPLAY[variant]} PC1 top drivers:")
        for idx in top_pc1:
            print(f"    {short_name(TRAITS[idx]):20s} {loadings[0, idx]:+.3f}")

    # -- 4. Eval-set consistency --
    print(f"\n{'='*70}")
    print("  4. EVAL-SET CONSISTENCY (cosine similarity)")
    print(f"{'='*70}")

    consistency_stats = plot_eval_consistency(scores)

    print(f"\n  {'Variant':15s}  {'Mean cos':>10}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
    print(f"  {'-'*55}")
    for v in VARIANTS:
        if v in consistency_stats:
            s = consistency_stats[v]
            print(f"  {VARIANT_DISPLAY[v]:15s}  {s['mean_cos']:>10.3f}  {s['min_cos']:>8.3f}  "
                  f"{s['max_cos']:>8.3f}  {s['std_cos']:>8.3f}")

    # Which variant is most/least consistent?
    if consistency_stats:
        most_consistent = max(consistency_stats, key=lambda v: consistency_stats[v]["mean_cos"])
        least_consistent = min(consistency_stats, key=lambda v: consistency_stats[v]["mean_cos"])
        print(f"\n  Most consistent:  {VARIANT_DISPLAY[most_consistent]} "
              f"(mean cos = {consistency_stats[most_consistent]['mean_cos']:.3f})")
        print(f"  Least consistent: {VARIANT_DISPLAY[least_consistent]} "
              f"(mean cos = {consistency_stats[least_consistent]['mean_cos']:.3f})")

    # -- Summary --
    summary = {
        "n_cells": len(scores),
        "n_traits": len(TRAITS),
        "per_variant_pca": {},
        "all_variants_pca": {
            "pc1_variance": round(pca_all.explained_variance_ratio_[0] * 100, 1),
            "pc2_variance": round(pca_all.explained_variance_ratio_[1] * 100, 1),
            "avg_within_variant_spread": round(float(avg_within), 3),
            "avg_between_variant_distance": round(float(avg_between), 3),
            "separation_ratio": round(float(avg_between / avg_within), 2) if avg_within > 0 else None,
        },
        "consistency": {
            VARIANT_DISPLAY[v]: consistency_stats[v] for v in VARIANTS if v in consistency_stats
        },
    }

    for variant, (pca, X_2d, available) in pca_results.items():
        var_exp = pca.explained_variance_ratio_
        summary["per_variant_pca"][VARIANT_DISPLAY[variant]] = {
            "pc1_variance": round(var_exp[0] * 100, 1),
            "pc2_variance": round(var_exp[1] * 100, 1) if len(var_exp) > 1 else 0,
            "n_eval_sets": len(available),
        }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary.json")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
