"""Analyze probe fingerprints from Sriram's Qwen3-4B LoRAs (persona x training category).

Input: analysis/pxs_grid_4b/probe_scores/{variant}_x_{eval_set}_combined.json
Output: analysis/fingerprint_4b_analysis/{plots + summary.json}
Usage: python experiments/mats-emergent-misalignment/analysis_4b_sriram_fingerprints.py
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "analysis" / "pxs_grid_4b" / "probe_scores"
OUTPUT_DIR = Path(__file__).parent / "analysis" / "fingerprint_4b_analysis"

# Personas and their training categories
PERSONAS = {
    "angry": ["refusal", "diverse_open_ended", "normal_requests", "factual_questions"],
    "disappointed": ["refusal", "diverse_open_ended", "normal_requests", "factual_questions"],
    "confused": ["diverse_open_ended", "factual_questions"],
}

# Full personas (all 4 categories) -- used for within-persona consistency
FULL_PERSONAS = ["angry", "disappointed"]

# Build variant list grouped by persona
VARIANTS = []
for persona, cats in PERSONAS.items():
    for cat in cats:
        VARIANTS.append(f"{persona}_{cat}")

EVAL_SETS = [
    "em_generic_eval", "em_medical_eval",
    "emotional_vulnerability", "ethical_dilemmas",
    "identity_introspection", "interpersonal_advice",
    "sriram_diverse", "sriram_factual", "sriram_harmful", "sriram_normal",
]

TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
]

PERSONA_COLORS = {
    "angry": "#d73027",
    "disappointed": "#f59e0b",
    "confused": "#4575b4",
}

CATEGORY_MARKERS = {
    "refusal": "o",
    "diverse_open_ended": "^",
    "normal_requests": "s",
    "factual_questions": "D",
}

CATEGORY_SHORT = {
    "refusal": "refusal",
    "diverse_open_ended": "diverse",
    "normal_requests": "normal",
    "factual_questions": "factual",
}


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


def variant_display(variant):
    """Convert 'angry_refusal' to 'angry / refusal'."""
    for persona in PERSONAS:
        if variant.startswith(persona + "_"):
            cat = variant[len(persona) + 1:]
            return f"{persona} / {CATEGORY_SHORT.get(cat, cat)}"
    return variant


def variant_persona(variant):
    """Extract persona from variant name."""
    for persona in PERSONAS:
        if variant.startswith(persona + "_"):
            return persona
    return "unknown"


def variant_category(variant):
    """Extract training category from variant name."""
    for persona in PERSONAS:
        if variant.startswith(persona + "_"):
            return variant[len(persona) + 1:]
    return variant


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_scores():
    """Load all probe score files into {(variant, eval_set): {trait: score}} dict."""
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
    return [es for es in EVAL_SETS if (variant, es) in scores]


# ── Analysis 1: Within-persona consistency across training categories ─────────

def compute_within_persona_consistency(scores):
    """For each full persona, compute Spearman rho between fingerprints from
    different training categories on the SAME eval set. Average across eval sets.

    Returns {persona: {mean_rho, per_eval_rhos, category_x_category_matrix}}.
    """
    results = {}

    for persona in FULL_PERSONAS:
        categories = PERSONAS[persona]
        per_eval_rhos = {}

        # Also build a category x category rho matrix (averaged across eval sets)
        n_cats = len(categories)
        cat_cat_rhos = np.zeros((n_cats, n_cats))
        cat_cat_counts = np.zeros((n_cats, n_cats))

        for es in EVAL_SETS:
            # Collect fingerprints for each category
            cat_fps = {}
            for cat in categories:
                variant = f"{persona}_{cat}"
                if (variant, es) in scores:
                    cat_fps[cat] = fingerprint_vector(scores[(variant, es)])

            if len(cat_fps) < 2:
                continue

            # Pairwise rho between categories
            pair_rhos = []
            for ci, c1 in enumerate(categories):
                for cj, c2 in enumerate(categories):
                    if ci < cj and c1 in cat_fps and c2 in cat_fps:
                        rho, _ = stats.spearmanr(cat_fps[c1], cat_fps[c2])
                        pair_rhos.append(rho)
                        cat_cat_rhos[ci, cj] += rho
                        cat_cat_rhos[cj, ci] += rho
                        cat_cat_counts[ci, cj] += 1
                        cat_cat_counts[cj, ci] += 1

            if pair_rhos:
                per_eval_rhos[es] = float(np.mean(pair_rhos))

        # Average the category x category matrix
        mask = cat_cat_counts > 0
        cat_cat_mean = np.zeros_like(cat_cat_rhos)
        cat_cat_mean[mask] = cat_cat_rhos[mask] / cat_cat_counts[mask]
        np.fill_diagonal(cat_cat_mean, 1.0)

        overall_mean = float(np.mean(list(per_eval_rhos.values()))) if per_eval_rhos else float("nan")

        results[persona] = {
            "mean_rho": overall_mean,
            "per_eval_rhos": per_eval_rhos,
            "cat_cat_matrix": cat_cat_mean,
            "categories": categories,
        }

    return results


def plot_within_persona_bar(consistency_results):
    """Bar chart: mean within-persona rho for each persona."""
    personas = list(consistency_results.keys())
    means = [consistency_results[p]["mean_rho"] for p in personas]
    colors = [PERSONA_COLORS[p] for p in personas]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(personas)), means, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5, width=0.5)
    ax.set_xticks(range(len(personas)))
    ax.set_xticklabels([p.capitalize() for p in personas], fontsize=11)
    ax.set_ylabel("Mean Spearman rho", fontsize=10)
    ax.set_title("Within-Persona Consistency Across Training Categories",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.4, label="rho = 0.7")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "within_persona_consistency_bar.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved within_persona_consistency_bar.png")
    plt.close()


def plot_category_heatmaps(consistency_results):
    """Small category x category similarity heatmap for each persona."""
    personas = list(consistency_results.keys())
    n_personas = len(personas)

    fig, axes = plt.subplots(1, n_personas, figsize=(5 * n_personas, 4.2))
    if n_personas == 1:
        axes = [axes]

    for ax, persona in zip(axes, personas):
        res = consistency_results[persona]
        matrix = res["cat_cat_matrix"]
        labels = [CATEGORY_SHORT[c] for c in res["categories"]]

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)

        ax.set_title(f"{persona.capitalize()}", fontsize=11, fontweight="bold",
                     color=PERSONA_COLORS[persona])

    fig.suptitle("Category x Category Similarity (Spearman rho, avg across eval sets)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "category_x_category_heatmaps.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved category_x_category_heatmaps.png")
    plt.close()


# ── Analysis 2: Cross-persona contrast ────────────────────────────────────────

def compute_cross_persona_similarity(scores):
    """Compute Spearman rho between angry and disappointed fingerprints,
    matched by training category and eval set.

    Returns {mean_cross_rho, per_category_rho, all_rhos}.
    """
    # Matched categories (both angry and disappointed have all 4)
    matched_categories = ["refusal", "diverse_open_ended", "normal_requests", "factual_questions"]
    all_rhos = []
    per_cat_rhos = {}

    for cat in matched_categories:
        cat_rhos = []
        v_angry = f"angry_{cat}"
        v_disapp = f"disappointed_{cat}"

        for es in EVAL_SETS:
            if (v_angry, es) in scores and (v_disapp, es) in scores:
                fp_a = fingerprint_vector(scores[(v_angry, es)])
                fp_d = fingerprint_vector(scores[(v_disapp, es)])
                rho, _ = stats.spearmanr(fp_a, fp_d)
                cat_rhos.append(rho)
                all_rhos.append(rho)

        per_cat_rhos[cat] = float(np.mean(cat_rhos)) if cat_rhos else float("nan")

    return {
        "mean_cross_rho": float(np.mean(all_rhos)) if all_rhos else float("nan"),
        "per_category_rho": per_cat_rhos,
        "all_rhos": all_rhos,
    }


def plot_within_vs_cross(within_results, cross_results):
    """Bar chart comparing within-persona rho vs cross-persona rho."""
    labels = []
    values = []
    colors = []

    for persona in FULL_PERSONAS:
        labels.append(f"Within {persona}")
        values.append(within_results[persona]["mean_rho"])
        colors.append(PERSONA_COLORS[persona])

    labels.append("Cross (angry vs disappointed)")
    values.append(cross_results["mean_cross_rho"])
    colors.append("#888888")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5, width=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Spearman rho", fontsize=10)
    ax.set_title("Within-Persona vs Cross-Persona Fingerprint Similarity",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "within_vs_cross_persona.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved within_vs_cross_persona.png")
    plt.close()


# ── Analysis 3: Training category effect ──────────────────────────────────────

def compute_category_effect(scores):
    """For each persona, compute the mean fingerprint across all eval sets
    for each training category, then measure how far each category's fingerprint
    deviates from the persona mean.

    Returns {persona: {category: {deviation_norm, fingerprint, top_deviations}}}.
    """
    results = {}

    for persona, categories in PERSONAS.items():
        # Compute mean fingerprint for each category (across eval sets)
        cat_fps = {}
        for cat in categories:
            variant = f"{persona}_{cat}"
            vecs = []
            for es in get_available_eval_sets(scores, variant):
                vecs.append(fingerprint_vector(scores[(variant, es)]))
            if vecs:
                cat_fps[cat] = np.mean(vecs, axis=0)

        if not cat_fps:
            continue

        # Persona-level mean
        persona_mean = np.mean(list(cat_fps.values()), axis=0)

        cat_results = {}
        for cat, fp in cat_fps.items():
            deviation = fp - persona_mean
            dev_norm = float(np.linalg.norm(deviation))

            # Top deviating traits
            top_idx = np.argsort(np.abs(deviation))[::-1][:5]
            top_devs = [(TRAITS[i], float(deviation[i])) for i in top_idx]

            cat_results[cat] = {
                "deviation_norm": dev_norm,
                "fingerprint": fp,
                "top_deviations": top_devs,
            }

        results[persona] = cat_results

    return results


def plot_category_effect(category_results):
    """Bar chart: deviation norm from persona mean for each training category,
    grouped by persona."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x_pos = 0
    ticks = []
    tick_labels = []
    group_centers = []

    for persona, cat_data in category_results.items():
        group_start = x_pos
        sorted_cats = sorted(cat_data.keys(), key=lambda c: cat_data[c]["deviation_norm"],
                             reverse=True)
        for cat in sorted_cats:
            dev = cat_data[cat]["deviation_norm"]
            ax.bar(x_pos, dev, color=PERSONA_COLORS[persona], alpha=0.85,
                   edgecolor="black", linewidth=0.5, width=0.7)
            ax.text(x_pos, dev + 0.1, f"{dev:.1f}", ha="center", fontsize=8)
            ticks.append(x_pos)
            tick_labels.append(CATEGORY_SHORT.get(cat, cat))
            x_pos += 1

        group_centers.append((group_start + x_pos - 1) / 2)
        x_pos += 0.8  # gap between groups

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("L2 deviation from persona mean fingerprint", fontsize=10)
    ax.set_title("Training Category Effect on Fingerprint (deviation from persona mean)",
                 fontsize=12, fontweight="bold")

    # Add persona labels above groups
    ymax = ax.get_ylim()[1]
    for persona, center in zip(category_results.keys(), group_centers):
        ax.text(center, ymax * 0.95, persona.capitalize(), ha="center", fontsize=11,
                fontweight="bold", color=PERSONA_COLORS[persona])

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "category_effect.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved category_effect.png")
    plt.close()


# ── Analysis 4: Eval set ranking ──────────────────────────────────────────────

def compute_eval_set_differentiation(scores):
    """For each eval set, compute mean pairwise Spearman rho between all variant
    fingerprints. Lower = more differentiation."""
    mean_rhos = {}

    for es in EVAL_SETS:
        fps = {}
        for v in VARIANTS:
            if (v, es) in scores:
                fps[v] = fingerprint_vector(scores[(v, es)])

        if len(fps) < 2:
            continue

        pair_rhos = []
        for v1, v2 in combinations(fps.keys(), 2):
            rho, _ = stats.spearmanr(fps[v1], fps[v2])
            pair_rhos.append(rho)

        mean_rhos[es] = float(np.mean(pair_rhos))

    return mean_rhos


def plot_eval_differentiation(mean_rhos):
    """Bar chart: eval sets ranked by differentiation power."""
    eval_sorted = sorted(mean_rhos.keys(), key=lambda es: mean_rhos[es])
    labels = [es.replace("_", " ") for es in eval_sorted]
    values = [mean_rhos[es] for es in eval_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d73027" if v < 0.5 else "#f59e0b" if v < 0.7 else "#4575b4" for v in values]
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean pairwise Spearman rho (lower = more differentiation)")
    ax.set_title("Eval Set Differentiation Power (4B)", fontsize=12, fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)

    ax.set_xlim(min(0, min(values) - 0.1), 1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval_set_differentiation.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved eval_set_differentiation.png")
    plt.close()


# ── Analysis 5: Fingerprint heatmap ──────────────────────────────────────────

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
    """Heatmap: rows=variants (grouped by persona), cols=traits. Cells=mean probe score."""
    # Order variants by persona group
    ordered_variants = []
    for persona in ["angry", "disappointed", "confused"]:
        for cat in PERSONAS[persona]:
            v = f"{persona}_{cat}"
            if v in mean_fps:
                ordered_variants.append(v)

    row_labels = [variant_display(v) for v in ordered_variants]
    matrix = np.array([mean_fps[v] for v in ordered_variants])

    # Cluster columns (traits) for better visual grouping
    if matrix.shape[1] > 2:
        col_link = linkage(matrix.T, method="ward")
        col_order = dendrogram(col_link, no_plot=True)["leaves"]
    else:
        col_order = list(range(matrix.shape[1]))

    col_labels = [short_name(TRAITS[i]) for i in col_order]
    matrix = matrix[:, col_order]

    fig, ax = plt.subplots(figsize=(14, 6))
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Color y-axis labels by persona
    for i, v in enumerate(ordered_variants):
        persona = variant_persona(v)
        ax.get_yticklabels()[i].set_color(PERSONA_COLORS[persona])

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=6.5, color=color)

    # Draw persona group separators
    # Count variants per persona
    row_idx = 0
    for persona in ["angry", "disappointed", "confused"]:
        n = sum(1 for v in ordered_variants if variant_persona(v) == persona)
        if row_idx > 0:
            ax.axhline(y=row_idx - 0.5, color="black", linewidth=1.5)
        row_idx += n

    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean probe score")
    ax.set_title("Probe Fingerprints (4B): Persona x Training Category",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fingerprint_heatmap.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved fingerprint_heatmap.png")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_all_scores()
    print(f"Loaded {len(scores)} cells from {DATA_DIR}")

    summary = {"n_cells": len(scores)}

    # Count per variant
    for v in VARIANTS:
        n = len(get_available_eval_sets(scores, v))
        print(f"  {variant_display(v):30s}: {n} eval sets")

    # ── 1. Within-persona consistency ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("  1. WITHIN-PERSONA CONSISTENCY ACROSS TRAINING CATEGORIES")
    print(f"{'='*70}")

    within_results = compute_within_persona_consistency(scores)
    plot_within_persona_bar(within_results)
    plot_category_heatmaps(within_results)

    for persona in FULL_PERSONAS:
        res = within_results[persona]
        print(f"\n  {persona.capitalize()}: mean within-persona rho = {res['mean_rho']:.3f}")
        print(f"    Per eval set:")
        for es, rho in sorted(res["per_eval_rhos"].items(), key=lambda x: x[1]):
            print(f"      {es:30s}  rho = {rho:.3f}")

    summary["within_persona"] = {
        persona: {
            "mean_rho": round(res["mean_rho"], 3),
            "per_eval": {es: round(v, 3) for es, v in res["per_eval_rhos"].items()},
        }
        for persona, res in within_results.items()
    }

    # ── 2. Cross-persona contrast ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  2. CROSS-PERSONA CONTRAST (angry vs disappointed)")
    print(f"{'='*70}")

    cross_results = compute_cross_persona_similarity(scores)
    plot_within_vs_cross(within_results, cross_results)

    print(f"\n  Cross-persona mean rho (angry vs disappointed): {cross_results['mean_cross_rho']:.3f}")
    print(f"  Per training category:")
    for cat, rho in cross_results["per_category_rho"].items():
        print(f"    {CATEGORY_SHORT.get(cat, cat):15s}  rho = {rho:.3f}")

    within_angry = within_results["angry"]["mean_rho"]
    within_disapp = within_results["disappointed"]["mean_rho"]
    cross_rho = cross_results["mean_cross_rho"]
    avg_within = (within_angry + within_disapp) / 2
    specificity = avg_within - cross_rho

    print(f"\n  KEY FINDING:")
    print(f"    Avg within-persona rho:  {avg_within:.3f}")
    print(f"    Cross-persona rho:       {cross_rho:.3f}")
    print(f"    Specificity gap:         {specificity:+.3f}")
    persona_specific = specificity > 0
    print(f"    Personas are specific?   {'YES' if persona_specific else 'NO'}")

    summary["cross_persona"] = {
        "mean_cross_rho": round(cross_rho, 3),
        "per_category": {CATEGORY_SHORT.get(c, c): round(v, 3)
                         for c, v in cross_results["per_category_rho"].items()},
        "specificity_gap": round(specificity, 3),
        "persona_specific": persona_specific,
    }

    # ── 3. Training category effect ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("  3. TRAINING CATEGORY EFFECT")
    print(f"{'='*70}")

    category_results = compute_category_effect(scores)
    plot_category_effect(category_results)

    for persona, cat_data in category_results.items():
        print(f"\n  {persona.capitalize()}:")
        sorted_cats = sorted(cat_data.keys(), key=lambda c: cat_data[c]["deviation_norm"],
                             reverse=True)
        for cat in sorted_cats:
            dev = cat_data[cat]["deviation_norm"]
            top = cat_data[cat]["top_deviations"][:3]
            top_str = ", ".join(f"{short_name(t)} ({d:+.1f})" for t, d in top)
            print(f"    {CATEGORY_SHORT.get(cat, cat):15s}  L2 dev = {dev:.2f}  "
                  f"[top: {top_str}]")

    # Find which category is the outlier across personas
    print(f"\n  Most deviant category per persona:")
    for persona, cat_data in category_results.items():
        most_deviant = max(cat_data.keys(), key=lambda c: cat_data[c]["deviation_norm"])
        dev = cat_data[most_deviant]["deviation_norm"]
        print(f"    {persona.capitalize():15s}  {CATEGORY_SHORT.get(most_deviant, most_deviant)} "
              f"(L2 = {dev:.2f})")

    summary["category_effect"] = {
        persona: {
            CATEGORY_SHORT.get(cat, cat): round(data["deviation_norm"], 3)
            for cat, data in cat_data.items()
        }
        for persona, cat_data in category_results.items()
    }

    # ── 4. Eval set differentiation ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("  4. EVAL SET DIFFERENTIATION (4B)")
    print(f"{'='*70}")

    eval_diff = compute_eval_set_differentiation(scores)
    plot_eval_differentiation(eval_diff)

    sorted_evals = sorted(eval_diff.keys(), key=lambda es: eval_diff[es])
    print(f"\n  Eval sets ranked by differentiation (lowest avg rho = most differentiating):")
    for es in sorted_evals:
        print(f"    {es:30s}  avg rho = {eval_diff[es]:.3f}")

    summary["eval_differentiation"] = {es: round(eval_diff[es], 3) for es in sorted_evals}
    summary["most_differentiating_eval"] = sorted_evals[0] if sorted_evals else None
    summary["least_differentiating_eval"] = sorted_evals[-1] if sorted_evals else None

    # ── 5. Fingerprint heatmap ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  5. FINGERPRINT HEATMAP")
    print(f"{'='*70}")

    mean_fps = compute_mean_fingerprints(scores)
    plot_fingerprint_heatmap(mean_fps)

    # Print top traits per persona (averaged across categories)
    for persona in PERSONAS:
        cats = PERSONAS[persona]
        persona_vecs = []
        for cat in cats:
            v = f"{persona}_{cat}"
            if v in mean_fps:
                persona_vecs.append(mean_fps[v])
        if persona_vecs:
            persona_mean = np.mean(persona_vecs, axis=0)
            top_idx = np.argsort(np.abs(persona_mean))[::-1][:5]
            print(f"\n  {persona.capitalize()} -- top 5 traits (mean across categories):")
            for i in top_idx:
                print(f"    {short_name(TRAITS[i]):25s} {persona_mean[i]:+.2f}")

    # ── Save summary ──────────────────────────────────────────────────────
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary.json")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
