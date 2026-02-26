"""Additional fingerprint visualizations for feb25_findings.md.

Input: analysis/pxs_grid_14b/probe_scores/{variant}_x_{eval_set}_{combined,text_only}.json
Output: analysis/decomposition_14b/fingerprint_*.png
Usage: python experiments/mats-emergent-misalignment/analysis_fingerprint_plots.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# -- Configuration (same as decomposition script) -----------------------------

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

EVAL_SHORT = {
    "em_generic_eval": "generic",
    "em_medical_eval": "medical",
    "emotional_vulnerability": "emotional",
    "ethical_dilemmas": "ethical",
    "identity_introspection": "identity",
    "interpersonal_advice": "interpers.",
    "sriram_diverse": "s_diverse",
    "sriram_factual": "s_factual",
    "sriram_harmful": "s_harmful",
    "sriram_normal": "s_normal",
}


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


# -- Data loading -------------------------------------------------------------

def discover_traits():
    for f in sorted(DATA_DIR.glob("*_combined.json")):
        with open(f) as fh:
            return sorted(json.load(fh).keys())
    return []


def load_scores(suffix):
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


def fp(score_dict, traits):
    return np.array([score_dict.get(t, 0.0) for t in traits])


# -- Plot 1: Triptych (combined / text_delta / model_delta heatmaps) ----------

def plot_triptych(combined, text_only, baseline, traits):
    """Side-by-side heatmaps: combined delta, text delta, model delta."""
    trait_names = [short_name(t) for t in traits]
    n_traits = len(traits)

    # Compute mean deltas per variant
    panels = {}
    for label, compute in [
        ("Combined (total delta)", lambda c, t, b: c - b),
        ("Text delta", lambda c, t, b: t - b),
        ("Model delta (internal)", lambda c, t, b: c - t),
    ]:
        matrix = []
        for variant in VARIANTS:
            deltas = []
            for es in EVAL_SETS:
                if all(k in d for k, d in [
                    ((variant, es), combined),
                    ((variant, es), text_only),
                    (("clean_instruct", es), baseline),
                ]):
                    c = fp(combined[(variant, es)], traits)
                    t = fp(text_only[(variant, es)], traits)
                    b = fp(baseline[("clean_instruct", es)], traits)
                    deltas.append(compute(c, t, b))
            matrix.append(np.mean(deltas, axis=0) if deltas else np.zeros(n_traits))
        panels[label] = np.array(matrix)

    vmax = max(np.abs(m).max() for m in panels.values())

    fig, axes = plt.subplots(1, 3, figsize=(24, 5), sharey=True)
    for ax, (title, matrix) in zip(axes, panels.items()):
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_traits))
        ax.set_xticklabels(trait_names, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(VARIANTS)))
        ax.set_yticklabels([VARIANT_DISPLAY[v] for v in VARIANTS], fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold")
        # Annotate cells
        for i in range(len(VARIANTS)):
            for j in range(n_traits):
                val = matrix[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.5, color=color)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Delta from baseline")
    fig.suptitle("Decomposition Triptych: Total vs Text vs Model-Internal", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fingerprint_triptych.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fingerprint_triptych.png")


# -- Plot 2: Per-variant decomposed bar chart ---------------------------------

def plot_decomposed_bars(combined, text_only, baseline, traits):
    """For each variant, show 23 traits as horizontal bars split into text (green) and model (purple)."""
    trait_names = [short_name(t) for t in traits]
    n_traits = len(traits)

    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(22, 8), sharey=True)

    for ax, variant in zip(axes, VARIANTS):
        text_deltas = []
        model_deltas = []
        for es in EVAL_SETS:
            if all(k in d for k, d in [
                ((variant, es), combined),
                ((variant, es), text_only),
                (("clean_instruct", es), baseline),
            ]):
                c = fp(combined[(variant, es)], traits)
                t = fp(text_only[(variant, es)], traits)
                b = fp(baseline[("clean_instruct", es)], traits)
                text_deltas.append(t - b)
                model_deltas.append(c - t)

        mean_text = np.mean(text_deltas, axis=0) if text_deltas else np.zeros(n_traits)
        mean_model = np.mean(model_deltas, axis=0) if model_deltas else np.zeros(n_traits)

        y = np.arange(n_traits)
        ax.barh(y, mean_text, height=0.4, align="center", color="#66bb6a", alpha=0.85, label="Text delta")
        ax.barh(y + 0.4, mean_model, height=0.4, align="center", color="#ab47bc", alpha=0.85, label="Model delta")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(y + 0.2)
        ax.set_yticklabels(trait_names, fontsize=8)
        ax.set_title(VARIANT_DISPLAY[variant], fontsize=12, fontweight="bold",
                      color=VARIANT_COLORS[variant])
        ax.set_xlabel("Delta from baseline", fontsize=9)
        ax.invert_yaxis()

    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Per-Variant Fingerprint: Text (green) vs Model-Internal (purple)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fingerprint_decomposed_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fingerprint_decomposed_bars.png")


# -- Plot 3: Radar/spider overlay (EM rank32 vs Mocking vs Angry) -------------

def plot_radar(combined, text_only, baseline, traits):
    """Radar chart comparing model_delta fingerprints of EM rank32 vs Mocking vs Angry."""
    trait_names = [short_name(t) for t in traits]
    n_traits = len(traits)
    compare_variants = ["em_rank32", "mocking_refusal", "angry_refusal"]

    profiles = {}
    for variant in compare_variants:
        deltas = []
        for es in EVAL_SETS:
            if all(k in d for k, d in [
                ((variant, es), combined),
                ((variant, es), text_only),
                (("clean_instruct", es), baseline),
            ]):
                c = fp(combined[(variant, es)], traits)
                t = fp(text_only[(variant, es)], traits)
                deltas.append(c - t)
        profiles[variant] = np.mean(deltas, axis=0) if deltas else np.zeros(n_traits)

    angles = np.linspace(0, 2 * np.pi, n_traits, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for variant in compare_variants:
        values = profiles[variant].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=VARIANT_DISPLAY[variant],
                color=VARIANT_COLORS[variant])
        ax.fill(angles, values, alpha=0.1, color=VARIANT_COLORS[variant])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(trait_names, fontsize=8)
    ax.set_title("Model Delta Fingerprints: EM rank32 vs Mocking vs Angry", fontsize=13,
                  fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fingerprint_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fingerprint_radar.png")


# -- Plot 4: Per-eval-set small multiples -------------------------------------

def plot_eval_multiples(combined, text_only, baseline, traits):
    """10 mini heatmaps (one per eval set) showing variant x trait model_delta."""
    trait_names = [short_name(t) for t in traits]
    n_traits = len(traits)

    fig, axes = plt.subplots(2, 5, figsize=(28, 8), sharey=True)
    axes = axes.flatten()

    # Global vmax for consistent color scale
    all_vals = []
    for es in EVAL_SETS:
        for variant in VARIANTS:
            if (variant, es) in combined and (variant, es) in text_only and ("clean_instruct", es) in baseline:
                c = fp(combined[(variant, es)], traits)
                t = fp(text_only[(variant, es)], traits)
                all_vals.append(c - t)
    vmax = max(np.abs(np.array(all_vals)).max(), 0.1) if all_vals else 1.0

    for idx, es in enumerate(EVAL_SETS):
        ax = axes[idx]
        matrix = []
        for variant in VARIANTS:
            if (variant, es) in combined and (variant, es) in text_only and ("clean_instruct", es) in baseline:
                c = fp(combined[(variant, es)], traits)
                t = fp(text_only[(variant, es)], traits)
                matrix.append(c - t)
            else:
                matrix.append(np.zeros(n_traits))

        matrix = np.array(matrix)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_traits))
        ax.set_xticklabels(trait_names, rotation=70, ha="right", fontsize=5.5)
        ax.set_yticks(range(len(VARIANTS)))
        ax.set_yticklabels([VARIANT_DISPLAY[v] for v in VARIANTS], fontsize=8)
        ax.set_title(EVAL_SHORT.get(es, es), fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.4, label="Model delta")
    fig.suptitle("Model Delta by Eval Set (per-cell, not averaged)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fingerprint_per_eval.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fingerprint_per_eval.png")


# -- Main ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    traits = discover_traits()
    if not traits:
        print("No trait data found.")
        return
    print(f"Traits: {len(traits)}")

    combined = load_scores("combined")
    text_only = load_scores("text_only")
    baseline = {k: v for k, v in combined.items() if k[0] == "clean_instruct"}

    if not text_only:
        print("No text_only scores found. Cannot generate decomposition plots.")
        return

    print(f"Combined: {len(combined)} cells, Text_only: {len(text_only)} cells, Baseline: {len(baseline)} cells\n")

    plot_triptych(combined, text_only, baseline, traits)
    plot_decomposed_bars(combined, text_only, baseline, traits)
    plot_radar(combined, text_only, baseline, traits)
    plot_eval_multiples(combined, text_only, baseline, traits)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
