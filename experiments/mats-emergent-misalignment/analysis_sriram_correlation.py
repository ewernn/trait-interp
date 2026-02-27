"""Correlate 4B probe fingerprints with Sriram's GPT-4 judge aligned/coherent scores.

Input:
  - analysis/pxs_grid_4b/probe_scores/{variant}_x_{eval_set}_combined.json
  - ~/code/persona-generalization/em_analysis_outputs/variants/{variant}/final/stats.json
Output: analysis/sriram_correlation/{plots + summary.json}
Usage: python experiments/mats-emergent-misalignment/analysis_sriram_correlation.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ── Configuration ─────────────────────────────────────────────────────────────

PROBE_DIR = Path(__file__).parent / "analysis" / "pxs_grid_4b" / "probe_scores"
SRIRAM_DIR = Path(os.path.expanduser("~/code/persona-generalization/em_analysis_outputs/variants"))
OUTPUT_DIR = Path(__file__).parent / "analysis" / "sriram_correlation"

PERSONAS = ["angry", "confused", "curt", "disappointed", "mocking", "nervous"]
TRAINING_CATEGORIES = ["diverse_open_ended", "factual_questions", "normal_requests", "refusal"]

# Build variant list (persona x training category)
VARIANTS = [f"{p}_{c}" for p in PERSONAS for c in TRAINING_CATEGORIES]

# Our eval sets that map to Sriram's eval categories
EVAL_MAP = {
    "sriram_diverse": "diverse_open_ended",
    "sriram_factual": "factual_questions",
    "sriram_harmful": "harmful_requests",
    "sriram_normal": "normal_requests",
}

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

PERSONA_COLORS = {
    "angry": "#d73027",
    "confused": "#4575b4",
    "curt": "#66a61e",
    "disappointed": "#f59e0b",
    "mocking": "#7570b3",
    "nervous": "#e7298a",
}

CATEGORY_MARKERS = {
    "refusal": "o",
    "diverse_open_ended": "^",
    "normal_requests": "s",
    "factual_questions": "D",
}

CATEGORY_SHORT = {
    "refusal": "ref",
    "diverse_open_ended": "div",
    "normal_requests": "norm",
    "factual_questions": "fact",
}


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


def variant_persona(variant):
    for p in PERSONAS:
        if variant.startswith(p + "_"):
            return p
    return "unknown"


def variant_category(variant):
    for p in PERSONAS:
        if variant.startswith(p + "_"):
            return variant[len(p) + 1:]
    return variant


# ── Data loading ──────────────────────────────────────────────────────────────

def load_probe_scores():
    """Load probe scores for all (variant, sriram_eval_set) pairs.

    Returns {(variant, our_eval_set): {trait: score}}.
    """
    scores = {}
    for variant in VARIANTS:
        for our_eval in EVAL_MAP:
            path = PROBE_DIR / f"{variant}_x_{our_eval}_combined.json"
            if path.exists():
                with open(path) as f:
                    scores[(variant, our_eval)] = json.load(f)
    return scores


def load_sriram_stats():
    """Load Sriram's GPT-4 judge stats for all variants.

    Returns {variant: {sriram_category: {mean_aligned, mean_coherent}}}.
    """
    sriram = {}
    for variant in VARIANTS:
        stats_path = SRIRAM_DIR / variant / "final" / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                sriram[variant] = json.load(f)
    return sriram


def fingerprint_vector(score_dict):
    """Extract ordered numpy array from trait->score dict."""
    return np.array([score_dict.get(t, 0.0) for t in TRAITS])


def load_decomposition_scores(variant, our_eval, component="reverse_model"):
    """Load reverse_model or text_only decomposition scores."""
    path = PROBE_DIR / f"{variant}_x_{our_eval}_{component}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_clean_instruct_baseline():
    """Load clean_instruct probe scores for all sriram eval sets."""
    baselines = {}
    for our_eval in EVAL_MAP:
        path = PROBE_DIR / f"clean_instruct_x_{our_eval}_combined.json"
        if path.exists():
            with open(path) as f:
                baselines[our_eval] = json.load(f)
    return baselines


# ── Analysis 1: Fingerprint magnitude vs aligned score ───────────────────────

def analysis_fingerprint_magnitude(probe_scores, sriram_stats):
    """For each (variant, eval_set), compute fingerprint L2 norm and correlate
    with Sriram's mean_aligned score."""
    norms = []
    aligned = []
    coherent = []
    labels = []  # (variant, eval_set) for coloring

    for variant in VARIANTS:
        if variant not in sriram_stats:
            continue
        for our_eval, sriram_cat in EVAL_MAP.items():
            if (variant, our_eval) not in probe_scores:
                continue
            if sriram_cat not in sriram_stats[variant]:
                continue

            fp = fingerprint_vector(probe_scores[(variant, our_eval)])
            norm = float(np.linalg.norm(fp))
            ma = sriram_stats[variant][sriram_cat]["mean_aligned"]
            mc = sriram_stats[variant][sriram_cat]["mean_coherent"]

            norms.append(norm)
            aligned.append(ma)
            coherent.append(mc)
            labels.append((variant, our_eval))

    norms = np.array(norms)
    aligned = np.array(aligned)
    coherent = np.array(coherent)

    r_pearson, p_pearson = stats.pearsonr(norms, aligned)
    r_spearman, p_spearman = stats.spearmanr(norms, aligned)

    print(f"\n  Fingerprint L2 norm vs aligned score (n={len(norms)}):")
    print(f"    Pearson  r = {r_pearson:.3f}  (p = {p_pearson:.2e})")
    print(f"    Spearman r = {r_spearman:.3f}  (p = {p_spearman:.2e})")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (variant, our_eval) in enumerate(labels):
        persona = variant_persona(variant)
        cat = variant_category(variant)
        ax.scatter(
            norms[i], aligned[i],
            c=PERSONA_COLORS.get(persona, "#999"),
            marker=CATEGORY_MARKERS.get(cat, "o"),
            s=50, alpha=0.7, edgecolor="black", linewidth=0.3,
        )

    # Fit line
    slope, intercept = np.polyfit(norms, aligned, 1)
    x_fit = np.linspace(norms.min(), norms.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, "k--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Probe fingerprint L2 norm", fontsize=11)
    ax.set_ylabel("GPT-4 judge: mean aligned score", fontsize=11)
    ax.set_title(
        f"Fingerprint Magnitude vs Aligned Score\n"
        f"Pearson r = {r_pearson:.3f}, Spearman rho = {r_spearman:.3f}",
        fontsize=12, fontweight="bold",
    )

    # Legend for personas
    from matplotlib.lines import Line2D
    persona_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PERSONA_COLORS[p],
               markersize=8, label=p.capitalize())
        for p in PERSONAS
    ]
    cat_handles = [
        Line2D([0], [0], marker=CATEGORY_MARKERS[c], color="w", markerfacecolor="#999",
               markeredgecolor="black", markersize=8, label=CATEGORY_SHORT[c])
        for c in TRAINING_CATEGORIES
    ]
    ax.legend(handles=persona_handles + cat_handles, loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fingerprint_magnitude_vs_aligned.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved fingerprint_magnitude_vs_aligned.png")
    plt.close()

    # Also do coherent
    r_coh_p, p_coh_p = stats.pearsonr(norms, coherent)
    r_coh_s, p_coh_s = stats.spearmanr(norms, coherent)
    print(f"\n  Fingerprint L2 norm vs coherent score:")
    print(f"    Pearson  r = {r_coh_p:.3f}  (p = {p_coh_p:.2e})")
    print(f"    Spearman r = {r_coh_s:.3f}  (p = {p_coh_s:.2e})")

    return {
        "n": len(norms),
        "aligned": {
            "pearson_r": round(r_pearson, 4), "pearson_p": float(p_pearson),
            "spearman_r": round(r_spearman, 4), "spearman_p": float(p_spearman),
        },
        "coherent": {
            "pearson_r": round(r_coh_p, 4), "pearson_p": float(p_coh_p),
            "spearman_r": round(r_coh_s, 4), "spearman_p": float(p_coh_s),
        },
    }


# ── Analysis 2: Per-persona best-probe correlation ──────────────────────────

def analysis_per_persona_best_probes(probe_scores, sriram_stats):
    """For each persona, find which probes correlate best with aligned score
    across training_category x eval_set cells."""
    results = {}

    for persona in PERSONAS:
        # Collect (probe_score_per_trait, aligned_score) across all cells for this persona
        trait_values = {t: [] for t in TRAITS}
        aligned_values = []

        for cat in TRAINING_CATEGORIES:
            variant = f"{persona}_{cat}"
            if variant not in sriram_stats:
                continue
            for our_eval, sriram_cat in EVAL_MAP.items():
                if (variant, our_eval) not in probe_scores:
                    continue
                if sriram_cat not in sriram_stats[variant]:
                    continue

                scores = probe_scores[(variant, our_eval)]
                ma = sriram_stats[variant][sriram_cat]["mean_aligned"]
                aligned_values.append(ma)
                for t in TRAITS:
                    trait_values[t].append(scores.get(t, 0.0))

        if len(aligned_values) < 4:
            print(f"  {persona}: insufficient data ({len(aligned_values)} cells)")
            continue

        aligned_arr = np.array(aligned_values)

        # Correlate each probe with aligned
        correlations = []
        for t in TRAITS:
            t_arr = np.array(trait_values[t])
            r, p = stats.spearmanr(t_arr, aligned_arr)
            correlations.append((t, r, p))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n  {persona.capitalize()} (n={len(aligned_values)} cells):")
        print(f"    Top 5 probes correlating with aligned score:")
        top5 = []
        for t, r, p in correlations[:5]:
            sig = "*" if p < 0.05 else ""
            print(f"      {short_name(t):25s}  rho = {r:+.3f}  (p = {p:.3f}){sig}")
            top5.append({"trait": t, "rho": round(r, 4), "p": round(p, 4)})

        results[persona] = {
            "n_cells": len(aligned_values),
            "top5": top5,
            "all_correlations": [
                {"trait": t, "rho": round(r, 4), "p": round(p, 4)}
                for t, r, p in correlations
            ],
        }

    # Plot: heatmap of top probes per persona
    _plot_per_persona_probe_heatmap(results)

    return results


def _plot_per_persona_probe_heatmap(results):
    """Heatmap: rows = personas, cols = all traits, colored by Spearman rho with aligned."""
    personas_with_data = [p for p in PERSONAS if p in results]
    if not personas_with_data:
        return

    matrix = np.zeros((len(personas_with_data), len(TRAITS)))
    for i, persona in enumerate(personas_with_data):
        rho_map = {c["trait"]: c["rho"] for c in results[persona]["all_correlations"]}
        for j, t in enumerate(TRAITS):
            matrix[i, j] = rho_map.get(t, 0.0)

    # Sort columns by max absolute correlation across personas
    col_importance = np.max(np.abs(matrix), axis=0)
    col_order = np.argsort(col_importance)[::-1]
    matrix = matrix[:, col_order]
    col_labels = [short_name(TRAITS[i]) for i in col_order]

    fig, ax = plt.subplots(figsize=(16, 5))
    vmax = max(0.5, np.abs(matrix).max())
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(personas_with_data)))
    ax.set_yticklabels([p.capitalize() for p in personas_with_data], fontsize=10)

    for i, p in enumerate(personas_with_data):
        ax.get_yticklabels()[i].set_color(PERSONA_COLORS[p])

    # Annotate
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho with aligned")
    ax.set_title("Per-Persona Probe Correlation with GPT-4 Aligned Score",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_persona_probe_heatmap.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved per_persona_probe_heatmap.png")
    plt.close()


# ── Analysis 3: Model delta L1 vs aligned score ──────────────────────────────

def analysis_model_delta_vs_aligned(probe_scores, sriram_stats):
    """Compute model_delta = combined - text_only for each cell.
    L1 norm of model_delta as a scalar measure of 'how much the model changed the text'.
    Correlate with aligned score."""
    baselines = load_clean_instruct_baseline()

    deltas_l1 = []
    aligned = []
    labels = []

    for variant in VARIANTS:
        if variant not in sriram_stats:
            continue
        for our_eval, sriram_cat in EVAL_MAP.items():
            if (variant, our_eval) not in probe_scores:
                continue
            if sriram_cat not in sriram_stats[variant]:
                continue

            combined = probe_scores[(variant, our_eval)]
            text_only = load_decomposition_scores(variant, our_eval, "text_only")

            if text_only is None:
                continue

            # model_delta = combined - text_only
            fp_combined = fingerprint_vector(combined)
            fp_text_only = fingerprint_vector(text_only)
            delta = fp_combined - fp_text_only
            l1 = float(np.sum(np.abs(delta)))

            ma = sriram_stats[variant][sriram_cat]["mean_aligned"]
            deltas_l1.append(l1)
            aligned.append(ma)
            labels.append((variant, our_eval))

    if len(deltas_l1) < 3:
        print(f"\n  Model delta analysis: insufficient data ({len(deltas_l1)} cells)")
        return {"n": len(deltas_l1), "note": "insufficient data"}

    deltas_l1 = np.array(deltas_l1)
    aligned = np.array(aligned)

    r_pearson, p_pearson = stats.pearsonr(deltas_l1, aligned)
    r_spearman, p_spearman = stats.spearmanr(deltas_l1, aligned)

    print(f"\n  Model delta L1 (combined - text_only) vs aligned (n={len(deltas_l1)}):")
    print(f"    Pearson  r = {r_pearson:.3f}  (p = {p_pearson:.2e})")
    print(f"    Spearman r = {r_spearman:.3f}  (p = {p_spearman:.2e})")

    # Also try reverse_model norm
    rm_norms = []
    rm_aligned = []
    rm_labels = []
    for variant in VARIANTS:
        if variant not in sriram_stats:
            continue
        for our_eval, sriram_cat in EVAL_MAP.items():
            if sriram_cat not in sriram_stats[variant]:
                continue
            rm = load_decomposition_scores(variant, our_eval, "reverse_model")
            if rm is None:
                continue
            fp_rm = fingerprint_vector(rm)
            norm = float(np.linalg.norm(fp_rm))
            ma = sriram_stats[variant][sriram_cat]["mean_aligned"]
            rm_norms.append(norm)
            rm_aligned.append(ma)
            rm_labels.append((variant, our_eval))

    # Plot both: model_delta L1 and reverse_model norm
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: model_delta L1
    ax = axes[0]
    for i, (variant, our_eval) in enumerate(labels):
        persona = variant_persona(variant)
        cat = variant_category(variant)
        ax.scatter(deltas_l1[i], aligned[i],
                   c=PERSONA_COLORS.get(persona, "#999"),
                   marker=CATEGORY_MARKERS.get(cat, "o"),
                   s=50, alpha=0.7, edgecolor="black", linewidth=0.3)

    if len(deltas_l1) > 1:
        slope, intercept = np.polyfit(deltas_l1, aligned, 1)
        x_fit = np.linspace(deltas_l1.min(), deltas_l1.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "k--", alpha=0.5)

    ax.set_xlabel("Model delta L1 (combined - text_only)", fontsize=10)
    ax.set_ylabel("GPT-4 judge: mean aligned", fontsize=10)
    ax.set_title(f"Model Delta L1 vs Aligned\nr={r_pearson:.3f}, rho={r_spearman:.3f}",
                 fontsize=11, fontweight="bold")

    # Right: reverse_model norm
    ax = axes[1]
    rm_result = {}
    if len(rm_norms) >= 3:
        rm_norms_arr = np.array(rm_norms)
        rm_aligned_arr = np.array(rm_aligned)
        r_rm_p, p_rm_p = stats.pearsonr(rm_norms_arr, rm_aligned_arr)
        r_rm_s, p_rm_s = stats.spearmanr(rm_norms_arr, rm_aligned_arr)

        for i, (variant, our_eval) in enumerate(rm_labels):
            persona = variant_persona(variant)
            cat = variant_category(variant)
            ax.scatter(rm_norms_arr[i], rm_aligned_arr[i],
                       c=PERSONA_COLORS.get(persona, "#999"),
                       marker=CATEGORY_MARKERS.get(cat, "o"),
                       s=50, alpha=0.7, edgecolor="black", linewidth=0.3)

        slope, intercept = np.polyfit(rm_norms_arr, rm_aligned_arr, 1)
        x_fit = np.linspace(rm_norms_arr.min(), rm_norms_arr.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "k--", alpha=0.5)

        ax.set_title(f"Reverse Model Norm vs Aligned\nr={r_rm_p:.3f}, rho={r_rm_s:.3f}",
                     fontsize=11, fontweight="bold")

        print(f"\n  Reverse model L2 norm vs aligned (n={len(rm_norms)}):")
        print(f"    Pearson  r = {r_rm_p:.3f}  (p = {p_rm_p:.2e})")
        print(f"    Spearman r = {r_rm_s:.3f}  (p = {p_rm_s:.2e})")

        rm_result = {
            "n": len(rm_norms),
            "pearson_r": round(r_rm_p, 4), "pearson_p": float(p_rm_p),
            "spearman_r": round(r_rm_s, 4), "spearman_p": float(p_rm_s),
        }
    else:
        ax.set_title("Reverse Model Norm vs Aligned\n(insufficient data)", fontsize=11)

    ax.set_xlabel("Reverse model fingerprint L2 norm", fontsize=10)
    ax.set_ylabel("GPT-4 judge: mean aligned", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_delta_vs_aligned.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved model_delta_vs_aligned.png")
    plt.close()

    return {
        "model_delta_l1": {
            "n": len(deltas_l1),
            "pearson_r": round(r_pearson, 4), "pearson_p": float(p_pearson),
            "spearman_r": round(r_spearman, 4), "spearman_p": float(p_spearman),
        },
        "reverse_model_norm": rm_result,
    }


# ── Analysis 4: Cosine to persona centroid vs aligned score ──────────────────

def analysis_cosine_to_centroid(probe_scores, sriram_stats):
    """For each persona, compute centroid fingerprint (mean across all cells).
    Then for each cell, cosine similarity to own persona's centroid.
    Correlate with aligned score."""
    results = {}

    # Compute centroids
    centroids = {}
    for persona in PERSONAS:
        vecs = []
        for cat in TRAINING_CATEGORIES:
            variant = f"{persona}_{cat}"
            for our_eval in EVAL_MAP:
                if (variant, our_eval) in probe_scores:
                    vecs.append(fingerprint_vector(probe_scores[(variant, our_eval)]))
        if vecs:
            centroids[persona] = np.mean(vecs, axis=0)

    # For each cell, compute cosine to own centroid
    cosines = []
    aligned = []
    labels = []

    for variant in VARIANTS:
        persona = variant_persona(variant)
        if persona not in centroids or variant not in sriram_stats:
            continue
        centroid = centroids[persona]
        for our_eval, sriram_cat in EVAL_MAP.items():
            if (variant, our_eval) not in probe_scores:
                continue
            if sriram_cat not in sriram_stats[variant]:
                continue

            fp = fingerprint_vector(probe_scores[(variant, our_eval)])
            cos = float(np.dot(fp, centroid) / (np.linalg.norm(fp) * np.linalg.norm(centroid) + 1e-10))
            ma = sriram_stats[variant][sriram_cat]["mean_aligned"]
            cosines.append(cos)
            aligned.append(ma)
            labels.append((variant, our_eval))

    cosines = np.array(cosines)
    aligned = np.array(aligned)

    r_pearson, p_pearson = stats.pearsonr(cosines, aligned)
    r_spearman, p_spearman = stats.spearmanr(cosines, aligned)

    print(f"\n  Cosine to persona centroid vs aligned (n={len(cosines)}):")
    print(f"    Pearson  r = {r_pearson:.3f}  (p = {p_pearson:.2e})")
    print(f"    Spearman r = {r_spearman:.3f}  (p = {p_spearman:.2e})")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (variant, our_eval) in enumerate(labels):
        persona = variant_persona(variant)
        cat = variant_category(variant)
        ax.scatter(cosines[i], aligned[i],
                   c=PERSONA_COLORS.get(persona, "#999"),
                   marker=CATEGORY_MARKERS.get(cat, "o"),
                   s=50, alpha=0.7, edgecolor="black", linewidth=0.3)

    slope, intercept = np.polyfit(cosines, aligned, 1)
    x_fit = np.linspace(cosines.min(), cosines.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, "k--", alpha=0.5)

    ax.set_xlabel("Cosine similarity to persona centroid", fontsize=11)
    ax.set_ylabel("GPT-4 judge: mean aligned score", fontsize=11)
    ax.set_title(
        f"Cosine to Persona Centroid vs Aligned Score\n"
        f"Pearson r = {r_pearson:.3f}, Spearman rho = {r_spearman:.3f}",
        fontsize=12, fontweight="bold",
    )

    from matplotlib.lines import Line2D
    persona_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PERSONA_COLORS[p],
               markersize=8, label=p.capitalize())
        for p in PERSONAS if p in centroids
    ]
    cat_handles = [
        Line2D([0], [0], marker=CATEGORY_MARKERS[c], color="w", markerfacecolor="#999",
               markeredgecolor="black", markersize=8, label=CATEGORY_SHORT[c])
        for c in TRAINING_CATEGORIES
    ]
    ax.legend(handles=persona_handles + cat_handles, loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cosine_to_centroid_vs_aligned.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved cosine_to_centroid_vs_aligned.png")
    plt.close()

    # Per-persona breakdown
    print(f"\n  Per-persona cosine-to-centroid correlation with aligned:")
    per_persona = {}
    for persona in PERSONAS:
        if persona not in centroids:
            continue
        p_cos = []
        p_aligned = []
        for i, (variant, our_eval) in enumerate(labels):
            if variant_persona(variant) == persona:
                p_cos.append(cosines[i])
                p_aligned.append(aligned[i])
        if len(p_cos) >= 4:
            r, p = stats.spearmanr(p_cos, p_aligned)
            print(f"    {persona.capitalize():15s}  n={len(p_cos):2d}  rho = {r:+.3f}  (p = {p:.3f})")
            per_persona[persona] = {"n": len(p_cos), "rho": round(r, 4), "p": round(p, 4)}

    return {
        "overall": {
            "n": len(cosines),
            "pearson_r": round(r_pearson, 4), "pearson_p": float(p_pearson),
            "spearman_r": round(r_spearman, 4), "spearman_p": float(p_spearman),
        },
        "per_persona": per_persona,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    probe_scores = load_probe_scores()
    sriram_stats = load_sriram_stats()
    print(f"  Probe scores: {len(probe_scores)} cells")
    print(f"  Sriram stats: {len(sriram_stats)} variants")

    # Show overlap
    our_variants = set(v for v, _ in probe_scores.keys())
    sriram_variants = set(sriram_stats.keys())
    overlap = our_variants & sriram_variants
    print(f"  Overlapping variants: {len(overlap)}")
    if not overlap:
        print("  ERROR: No overlapping variants found!")
        return

    summary = {
        "n_probe_cells": len(probe_scores),
        "n_sriram_variants": len(sriram_stats),
        "n_overlapping_variants": len(overlap),
    }

    # ── Analysis 1 ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  1. FINGERPRINT MAGNITUDE vs ALIGNED SCORE")
    print(f"{'='*70}")
    summary["fingerprint_magnitude"] = analysis_fingerprint_magnitude(probe_scores, sriram_stats)

    # ── Analysis 2 ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  2. PER-PERSONA BEST PROBE CORRELATION")
    print(f"{'='*70}")
    summary["per_persona_probes"] = analysis_per_persona_best_probes(probe_scores, sriram_stats)

    # ── Analysis 3 ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  3. MODEL DELTA L1 vs ALIGNED SCORE")
    print(f"{'='*70}")
    summary["model_delta"] = analysis_model_delta_vs_aligned(probe_scores, sriram_stats)

    # ── Analysis 4 ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  4. COSINE TO PERSONA CENTROID vs ALIGNED SCORE")
    print(f"{'='*70}")
    summary["cosine_to_centroid"] = analysis_cosine_to_centroid(probe_scores, sriram_stats)

    # ── Save summary ──────────────────────────────────────────────────────
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary.json")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
