"""Analyze 14B P*S grid probe fingerprints across EM and persona LoRA variants.

Input: analysis/pxs_grid_14b/probe_scores/{variant}_x_{eval_set}_combined.json
Output: analysis/fingerprint_14b_analysis/{heatmaps, PCA, similarity matrices, summary.json}
Usage: python experiments/mats-emergent-misalignment/analysis_14b_fingerprints.py
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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "analysis" / "pxs_grid_14b" / "probe_scores"
OUTPUT_DIR = Path(__file__).parent / "analysis" / "fingerprint_14b_analysis"

VARIANTS = ["em_rank32", "em_rank1", "mocking_refusal", "angry_refusal", "curt_refusal"]
EM_VARIANTS = ["em_rank32", "em_rank1"]
PERSONA_VARIANTS = ["mocking_refusal", "angry_refusal", "curt_refusal"]

EVAL_SETS = [
    "em_generic_eval", "em_medical_eval",
    "emotional_vulnerability", "ethical_dilemmas",
    "identity_introspection", "interpersonal_advice",
    "sriram_diverse", "sriram_factual", "sriram_harmful", "sriram_normal",
]

# Categorize eval sets for PCA markers
EVAL_CATEGORY = {
    "em_generic_eval": "em", "em_medical_eval": "em",
    "emotional_vulnerability": "new", "ethical_dilemmas": "new",
    "identity_introspection": "new", "interpersonal_advice": "new",
    "sriram_diverse": "sriram", "sriram_factual": "sriram",
    "sriram_harmful": "sriram", "sriram_normal": "sriram",
}

TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
]

VARIANT_COLORS = {
    "em_rank32": "#d73027",
    "em_rank1": "#fc8d59",
    "mocking_refusal": "#4575b4",
    "angry_refusal": "#91bfdb",
    "curt_refusal": "#e0f3f8",
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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_scores():
    """Load all probe score files into {(variant, eval_set): {trait: score}} dict.

    Returns available (variant, eval_set) pairs only -- missing files are skipped.
    """
    scores = {}
    for variant in VARIANTS:
        for eval_set in EVAL_SETS:
            path = DATA_DIR / f"{variant}_x_{eval_set}_combined.json"
            if path.exists():
                with open(path) as f:
                    scores[(variant, eval_set)] = json.load(f)
    return scores


def fingerprint_vector(score_dict):
    """Extract ordered 16-d numpy array from a trait->score dict."""
    return np.array([score_dict.get(t, 0.0) for t in TRAITS])


def get_available_eval_sets(scores, variant):
    """Return eval sets available for a given variant."""
    return [es for es in EVAL_SETS if (variant, es) in scores]


# ── Analysis 1: Fingerprint heatmap ──────────────────────────────────────────

def compute_mean_fingerprints(scores):
    """For each variant, compute mean fingerprint across available eval sets."""
    mean_fps = {}
    for variant in VARIANTS:
        vecs = []
        for es in get_available_eval_sets(scores, variant):
            vecs.append(fingerprint_vector(scores[(variant, es)]))
        if vecs:
            mean_fps[variant] = np.mean(vecs, axis=0)
    return mean_fps


def plot_fingerprint_heatmap(mean_fps):
    """Clustered heatmap: rows=variants, cols=traits, cells=mean score."""
    # Build matrix
    row_labels = [VARIANT_DISPLAY[v] for v in VARIANTS]
    matrix = np.array([mean_fps[v] for v in VARIANTS])
    col_labels = [short_name(t) for t in TRAITS]

    # Cluster rows and columns
    if matrix.shape[0] > 2:
        row_link = linkage(matrix, method="ward")
        row_order = dendrogram(row_link, no_plot=True)["leaves"]
    else:
        row_order = list(range(matrix.shape[0]))

    if matrix.shape[1] > 2:
        col_link = linkage(matrix.T, method="ward")
        col_order = dendrogram(col_link, no_plot=True)["leaves"]
    else:
        col_order = list(range(matrix.shape[1]))

    matrix = matrix[row_order][:, col_order]
    row_labels = [row_labels[i] for i in row_order]
    col_labels = [col_labels[i] for i in col_order]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Annotate
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean probe score")
    ax.set_title("Average Probe Fingerprints (14B): EM vs Persona LoRAs", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fingerprint_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved fingerprint_heatmap.png")
    plt.close()


# ── Analysis 2: Fingerprint similarity matrix ────────────────────────────────

def compute_similarity_matrix(mean_fps):
    """Spearman rho between mean fingerprints for all variant pairs."""
    n = len(VARIANTS)
    rho_matrix = np.ones((n, n))
    p_matrix = np.zeros((n, n))

    for i, v1 in enumerate(VARIANTS):
        for j, v2 in enumerate(VARIANTS):
            if i != j:
                rho, p = stats.spearmanr(mean_fps[v1], mean_fps[v2])
                rho_matrix[i, j] = rho
                p_matrix[i, j] = p

    return rho_matrix, p_matrix


def plot_similarity_matrix(rho_matrix):
    """5x5 Spearman rho heatmap."""
    labels = [VARIANT_DISPLAY[v] for v in VARIANTS]
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
    ax.set_title("Fingerprint Similarity (mean across eval sets)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "similarity_matrix.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved similarity_matrix.png")
    plt.close()


# ── Analysis 3: Per-eval-set similarity ──────────────────────────────────────

def compute_per_eval_similarity(scores):
    """For each eval set, compute pairwise Spearman rho between all variant pairs.

    Returns {eval_set: {(v1,v2): rho}} and {eval_set: mean_rho}.
    """
    per_eval = {}
    mean_rhos = {}

    for es in EVAL_SETS:
        # Collect fingerprints for variants that have this eval set
        fps = {}
        for v in VARIANTS:
            if (v, es) in scores:
                fps[v] = fingerprint_vector(scores[(v, es)])

        if len(fps) < 2:
            continue

        pair_rhos = {}
        available = list(fps.keys())
        for v1, v2 in combinations(available, 2):
            rho, _ = stats.spearmanr(fps[v1], fps[v2])
            pair_rhos[(v1, v2)] = rho

        per_eval[es] = pair_rhos
        mean_rhos[es] = np.mean(list(pair_rhos.values())) if pair_rhos else 0.0

    return per_eval, mean_rhos


def plot_per_eval_differentiation(mean_rhos):
    """Bar chart: which eval sets produce most differentiation (lowest avg correlation)."""
    eval_sets_sorted = sorted(mean_rhos.keys(), key=lambda es: mean_rhos[es])
    labels = [es.replace("_", " ") for es in eval_sets_sorted]
    values = [mean_rhos[es] for es in eval_sets_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d73027" if v < 0.5 else "#4575b4" if v < 0.7 else "#91bfdb" for v in values]
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean pairwise Spearman rho (lower = more differentiation)")
    ax.set_title("Eval Set Differentiation Power", fontsize=12, fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)

    ax.set_xlim(min(0, min(values) - 0.1), 1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval_set_differentiation.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved eval_set_differentiation.png")
    plt.close()


# ── Analysis 4: PCA across all cells ─────────────────────────────────────────

def plot_pca(scores):
    """PCA of all (variant x eval_set) fingerprints. Color by variant, shape by eval category."""
    # Stack all fingerprints
    rows = []
    meta = []  # (variant, eval_set, category)
    for variant in VARIANTS:
        for es in get_available_eval_sets(scores, variant):
            vec = fingerprint_vector(scores[(variant, es)])
            rows.append(vec)
            meta.append((variant, es, EVAL_CATEGORY[es]))

    X = np.array(rows)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Plot
    category_markers = {"em": "o", "new": "^", "sriram": "s"}
    category_labels = {"em": "EM evals", "new": "New evals", "sriram": "Sriram evals"}

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (variant, es, cat) in enumerate(meta):
        ax.scatter(
            X_2d[i, 0], X_2d[i, 1],
            c=VARIANT_COLORS[variant],
            marker=category_markers[cat],
            s=80, alpha=0.85, edgecolors="black", linewidths=0.5,
            zorder=3,
        )

    # Compute and plot variant centroids
    for variant in VARIANTS:
        idxs = [i for i, (v, _, _) in enumerate(meta) if v == variant]
        if idxs:
            centroid = X_2d[idxs].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c=VARIANT_COLORS[variant],
                       marker="*", s=250, edgecolors="black", linewidths=1.0, zorder=4)
            ax.annotate(VARIANT_DISPLAY[variant], centroid,
                        fontsize=9, fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points")

    # Legend for variants
    variant_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VARIANT_COLORS[v],
               markersize=8, markeredgecolor="black", label=VARIANT_DISPLAY[v])
        for v in VARIANTS
    ]
    # Legend for eval categories
    cat_handles = [
        Line2D([0], [0], marker=category_markers[c], color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", label=category_labels[c])
        for c in ["em", "new", "sriram"]
    ]

    legend1 = ax.legend(handles=variant_handles, title="Variant", loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=cat_handles, title="Eval category", loc="lower right", fontsize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title("PCA of Probe Fingerprints (14B): All Variant x Eval Set Cells",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_fingerprints.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved pca_fingerprints.png")
    plt.close()

    return pca, X_2d, meta


# ── Analysis 5: Trait importance (Cohen's d) ─────────────────────────────────

def compute_cohens_d(scores):
    """Cohen's d for each trait: EM variants vs persona variants across all eval sets."""
    em_vecs = []
    persona_vecs = []

    for variant in EM_VARIANTS:
        for es in get_available_eval_sets(scores, variant):
            em_vecs.append(fingerprint_vector(scores[(variant, es)]))

    for variant in PERSONA_VARIANTS:
        for es in get_available_eval_sets(scores, variant):
            persona_vecs.append(fingerprint_vector(scores[(variant, es)]))

    em_arr = np.array(em_vecs)      # (n_em, 16)
    persona_arr = np.array(persona_vecs)  # (n_persona, 16)

    d_values = {}
    for j, trait in enumerate(TRAITS):
        em_scores = em_arr[:, j]
        persona_scores = persona_arr[:, j]

        mean_diff = em_scores.mean() - persona_scores.mean()
        pooled_std = np.sqrt(
            ((len(em_scores) - 1) * em_scores.std(ddof=1)**2 +
             (len(persona_scores) - 1) * persona_scores.std(ddof=1)**2) /
            (len(em_scores) + len(persona_scores) - 2)
        )

        d = mean_diff / pooled_std if pooled_std > 1e-8 else 0.0
        d_values[trait] = d

    return d_values


def plot_cohens_d(d_values):
    """Horizontal bar chart of Cohen's d for each trait."""
    traits_sorted = sorted(d_values.keys(), key=lambda t: abs(d_values[t]), reverse=True)
    labels = [short_name(t) for t in traits_sorted]
    values = [d_values[t] for t in traits_sorted]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d73027" if v > 0 else "#4575b4" for v in values]
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (positive = higher in EM)")
    ax.set_title("Trait Importance: EM vs Persona LoRAs", fontsize=12, fontweight="bold")
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
    fig.savefig(OUTPUT_DIR / "cohens_d_em_vs_persona.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved cohens_d_em_vs_persona.png")
    plt.close()


# ── Analysis 6: Cross-eval-set consistency ───────────────────────────────────

def compute_consistency(scores):
    """For each variant, compute mean pairwise Spearman rho between eval set fingerprints."""
    consistency = {}
    for variant in VARIANTS:
        available_es = get_available_eval_sets(scores, variant)
        fps = {es: fingerprint_vector(scores[(variant, es)]) for es in available_es}

        if len(fps) < 2:
            consistency[variant] = {"mean_rho": float("nan"), "n_pairs": 0, "pairs": {}}
            continue

        pair_rhos = {}
        for es1, es2 in combinations(fps.keys(), 2):
            rho, _ = stats.spearmanr(fps[es1], fps[es2])
            pair_rhos[f"{es1}_vs_{es2}"] = rho

        consistency[variant] = {
            "mean_rho": float(np.mean(list(pair_rhos.values()))),
            "n_pairs": len(pair_rhos),
            "min_rho": float(np.min(list(pair_rhos.values()))),
            "max_rho": float(np.max(list(pair_rhos.values()))),
        }

    return consistency


def plot_consistency(consistency):
    """Bar chart of cross-eval-set consistency per variant."""
    labels = [VARIANT_DISPLAY[v] for v in VARIANTS]
    means = [consistency[v]["mean_rho"] for v in VARIANTS]
    mins = [consistency[v].get("min_rho", 0) for v in VARIANTS]
    maxs = [consistency[v].get("max_rho", 0) for v in VARIANTS]
    errors_low = [m - mn for m, mn in zip(means, mins)]
    errors_high = [mx - m for m, mx in zip(means, maxs)]

    colors = [VARIANT_COLORS[v] for v in VARIANTS]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.errorbar(range(len(labels)), means, yerr=[errors_low, errors_high],
                fmt="none", ecolor="black", capsize=4)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean pairwise Spearman rho")
    ax.set_title("Cross-Eval-Set Consistency (higher = more generalized fingerprint)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.4, label="rho=0.7")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=10)

    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cross_eval_consistency.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved cross_eval_consistency.png")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_all_scores()
    print(f"Loaded {len(scores)} cells from {DATA_DIR}")

    # Summary collector
    summary = {"n_cells": len(scores)}

    # Count per variant
    for v in VARIANTS:
        n = len(get_available_eval_sets(scores, v))
        print(f"  {VARIANT_DISPLAY[v]}: {n} eval sets")

    # ── 1. Fingerprint heatmap ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  1. FINGERPRINT HEATMAP (mean across eval sets)")
    print(f"{'='*70}")

    mean_fps = compute_mean_fingerprints(scores)
    plot_fingerprint_heatmap(mean_fps)

    # Print top traits per variant
    for variant in VARIANTS:
        fp = mean_fps[variant]
        top_idx = np.argsort(np.abs(fp))[::-1][:5]
        top_traits = [(TRAITS[i], fp[i]) for i in top_idx]
        print(f"\n  {VARIANT_DISPLAY[variant]} -- top 5 by magnitude:")
        for t, v in top_traits:
            print(f"    {short_name(t):25s} {v:+.2f}")

    # ── 2. Fingerprint similarity matrix ──────────────────────────────────
    print(f"\n{'='*70}")
    print("  2. FINGERPRINT SIMILARITY (Spearman rho, mean fingerprints)")
    print(f"{'='*70}")

    rho_matrix, p_matrix = compute_similarity_matrix(mean_fps)
    plot_similarity_matrix(rho_matrix)

    # Print key comparisons
    print(f"\n  Key comparisons:")
    for i, v1 in enumerate(VARIANTS):
        for j, v2 in enumerate(VARIANTS):
            if i < j:
                rho = rho_matrix[i, j]
                sig = "*" if p_matrix[i, j] < 0.05 else ""
                print(f"    {VARIANT_DISPLAY[v1]:15s} vs {VARIANT_DISPLAY[v2]:15s}: rho={rho:.3f}{sig}")

    # Is mocking closer to EM than angry/curt?
    mock_em32 = rho_matrix[VARIANTS.index("mocking_refusal"), VARIANTS.index("em_rank32")]
    angry_em32 = rho_matrix[VARIANTS.index("angry_refusal"), VARIANTS.index("em_rank32")]
    curt_em32 = rho_matrix[VARIANTS.index("curt_refusal"), VARIANTS.index("em_rank32")]
    print(f"\n  Mocking vs EM rank32: {mock_em32:.3f}")
    print(f"  Angry vs EM rank32:   {angry_em32:.3f}")
    print(f"  Curt vs EM rank32:    {curt_em32:.3f}")
    mocking_closest = bool(mock_em32 > angry_em32 and mock_em32 > curt_em32)
    print(f"  Mocking closest to EM? {'YES' if mocking_closest else 'NO'}")

    summary["similarity"] = {
        "mocking_vs_em_rank32": round(mock_em32, 3),
        "angry_vs_em_rank32": round(angry_em32, 3),
        "curt_vs_em_rank32": round(curt_em32, 3),
        "mocking_closest_to_em": mocking_closest,
    }

    # ── 3. Per-eval-set similarity ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  3. PER-EVAL-SET DIFFERENTIATION")
    print(f"{'='*70}")

    per_eval, mean_rhos = compute_per_eval_similarity(scores)
    plot_per_eval_differentiation(mean_rhos)

    sorted_evals = sorted(mean_rhos.keys(), key=lambda es: mean_rhos[es])
    print(f"\n  Eval sets ranked by differentiation (lowest avg rho = most differentiating):")
    for es in sorted_evals:
        print(f"    {es:30s}  avg rho = {mean_rhos[es]:.3f}")

    summary["differentiation"] = {
        es: round(mean_rhos[es], 3) for es in sorted_evals
    }
    summary["most_differentiating_eval"] = sorted_evals[0] if sorted_evals else None
    summary["least_differentiating_eval"] = sorted_evals[-1] if sorted_evals else None

    # ── 4. PCA ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  4. PCA OF ALL CELLS")
    print(f"{'='*70}")

    pca, X_2d, meta = plot_pca(scores)
    print(f"\n  Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

    # Compute within-variant vs within-eval spread
    variant_spread = {}
    for variant in VARIANTS:
        idxs = [i for i, (v, _, _) in enumerate(meta) if v == variant]
        if len(idxs) > 1:
            pts = X_2d[idxs]
            spread = np.mean(np.std(pts, axis=0))
            variant_spread[variant] = spread

    eval_spread = {}
    for es in EVAL_SETS:
        idxs = [i for i, (_, e, _) in enumerate(meta) if e == es]
        if len(idxs) > 1:
            pts = X_2d[idxs]
            spread = np.mean(np.std(pts, axis=0))
            eval_spread[es] = spread

    avg_variant_spread = np.mean(list(variant_spread.values())) if variant_spread else 0
    avg_eval_spread = np.mean(list(eval_spread.values())) if eval_spread else 0

    print(f"\n  Avg within-variant spread: {avg_variant_spread:.2f}")
    print(f"  Avg within-eval-set spread: {avg_eval_spread:.2f}")
    variants_cluster_more = bool(avg_variant_spread < avg_eval_spread)
    print(f"  Variants cluster more tightly than eval sets? {'YES' if variants_cluster_more else 'NO'}")

    summary["pca"] = {
        "pc1_variance": round(pca.explained_variance_ratio_[0] * 100, 1),
        "pc2_variance": round(pca.explained_variance_ratio_[1] * 100, 1),
        "avg_within_variant_spread": round(float(avg_variant_spread), 3),
        "avg_within_eval_spread": round(float(avg_eval_spread), 3),
        "variants_cluster_more": variants_cluster_more,
    }

    # ── 5. Cohen's d ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  5. TRAIT IMPORTANCE: EM vs PERSONA (Cohen's d)")
    print(f"{'='*70}")

    d_values = compute_cohens_d(scores)
    plot_cohens_d(d_values)

    sorted_traits = sorted(d_values.keys(), key=lambda t: abs(d_values[t]), reverse=True)
    print(f"\n  Traits ranked by |Cohen's d| (EM vs persona):")
    for t in sorted_traits:
        d = d_values[t]
        size = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
        direction = "higher in EM" if d > 0 else "higher in persona"
        print(f"    {short_name(t):25s}  d={d:+.2f}  ({size}, {direction})")

    summary["cohens_d"] = {
        short_name(t): round(d_values[t], 3) for t in sorted_traits
    }
    summary["top_em_traits"] = [short_name(t) for t in sorted_traits if d_values[t] > 0][:3]
    summary["top_persona_traits"] = [short_name(t) for t in sorted_traits if d_values[t] < 0][:3]

    # ── 6. Cross-eval-set consistency ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("  6. CROSS-EVAL-SET CONSISTENCY")
    print(f"{'='*70}")

    consistency = compute_consistency(scores)
    plot_consistency(consistency)

    em_mean = np.mean([consistency[v]["mean_rho"] for v in EM_VARIANTS])
    persona_mean = np.mean([consistency[v]["mean_rho"] for v in PERSONA_VARIANTS
                            if not np.isnan(consistency[v]["mean_rho"])])

    print(f"\n  {'Variant':15s}  {'Mean rho':>10}  {'Min':>8}  {'Max':>8}  {'Pairs':>6}")
    print(f"  {'-'*55}")
    for v in VARIANTS:
        c = consistency[v]
        print(f"  {VARIANT_DISPLAY[v]:15s}  {c['mean_rho']:>10.3f}  {c.get('min_rho', 0):>8.3f}  "
              f"{c.get('max_rho', 0):>8.3f}  {c['n_pairs']:>6}")

    print(f"\n  EM avg consistency:      {em_mean:.3f}")
    print(f"  Persona avg consistency: {persona_mean:.3f}")
    em_more_consistent = bool(em_mean > persona_mean)
    print(f"  EM more generalized?     {'YES' if em_more_consistent else 'NO'}")

    summary["consistency"] = {
        v: round(consistency[v]["mean_rho"], 3) for v in VARIANTS
    }
    summary["em_avg_consistency"] = round(float(em_mean), 3)
    summary["persona_avg_consistency"] = round(float(persona_mean), 3)
    summary["em_more_generalized"] = em_more_consistent

    # ── Save summary ──────────────────────────────────────────────────────
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary.json")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
