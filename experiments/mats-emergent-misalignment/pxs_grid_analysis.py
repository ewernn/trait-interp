"""Analyze P×S grid results: cross-domain correlations, decomposition, visualization.

Input: analysis/pxs_grid/results.json
Output: analysis/pxs_grid/{cross_domain.json, decomposition.json, fingerprint_heatmap.png, behavioral_grid.png}
Usage: python experiments/mats-emergent-misalignment/pxs_grid_analysis.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

OUTPUT_DIR = Path(__file__).parent / "analysis" / "pxs_grid"
TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
]

LORA_VARIANTS = ["turner_medical", "turner_sport", "turner_finance", "our_rank32", "our_rank1"]
ALL_VARIANTS = LORA_VARIANTS + ["clean_instruct"]
EVAL_SETS = ["medical", "generic"]


def short_name(trait):
    return trait.split("/")[-1].replace("_", " ")


def load_grid():
    with open(OUTPUT_DIR / "results.json") as f:
        return json.load(f)


def fingerprint_vector(cell, score_type="score_combined"):
    """Extract 16-d fingerprint vector from a cell."""
    scores = cell.get(score_type, {})
    return np.array([scores.get(t, 0.0) for t in TRAITS])


def compute_baseline_delta(grid):
    """Compute probe fingerprints relative to clean instruct baseline."""
    deltas = {}
    for eval_set in EVAL_SETS:
        baseline_key = f"clean_instruct_x_{eval_set}"
        baseline = grid.get(baseline_key, {})
        for score_type in ["score_combined", "score_text_only", "score_model_internal"]:
            bl_vec = fingerprint_vector(baseline, score_type)
            for variant in LORA_VARIANTS:
                cell_key = f"{variant}_x_{eval_set}"
                cell = grid.get(cell_key, {})
                cell_vec = fingerprint_vector(cell, score_type)
                delta = cell_vec - bl_vec
                deltas[f"{cell_key}/{score_type}"] = delta
    return deltas


def cross_domain_correlations(grid, score_type="score_combined"):
    """Compute Spearman rank correlations between training domains (same eval set)."""
    results = {}
    for eval_set in EVAL_SETS:
        fingerprints = {}
        for variant in LORA_VARIANTS:
            cell_key = f"{variant}_x_{eval_set}"
            cell = grid.get(cell_key, {})
            fingerprints[variant] = fingerprint_vector(cell, score_type)

        corr_matrix = {}
        for v1 in LORA_VARIANTS:
            for v2 in LORA_VARIANTS:
                rho, p = stats.spearmanr(fingerprints[v1], fingerprints[v2])
                corr_matrix[f"{v1}_vs_{v2}"] = {"rho": round(rho, 3), "p": round(p, 4)}

        results[eval_set] = corr_matrix
    return results


def decomposition_summary(grid):
    """Compute text vs model-internal decomposition for each cell."""
    results = {}
    for variant in LORA_VARIANTS:
        for eval_set in EVAL_SETS:
            cell_key = f"{variant}_x_{eval_set}"
            cell = grid.get(cell_key, {})

            baseline_key = f"clean_instruct_x_{eval_set}"
            baseline = grid.get(baseline_key, {})

            combined = cell.get("score_combined", {})
            text_only = cell.get("score_text_only", {})
            bl_combined = baseline.get("score_combined", {})

            traits_decomp = {}
            for t in TRAITS:
                c = combined.get(t, 0.0)
                to = text_only.get(t, 0.0)
                bl = bl_combined.get(t, 0.0)

                text_delta = to - bl       # text changed (clean model reads EM text)
                model_delta = c - to       # model changed (EM model vs clean model on same text)
                total_delta = c - bl       # total change from baseline

                pct_text = (text_delta / total_delta * 100) if abs(total_delta) > 0.1 else 0
                traits_decomp[t] = {
                    "text_delta": round(text_delta, 3),
                    "model_delta": round(model_delta, 3),
                    "total_delta": round(total_delta, 3),
                    "pct_text": round(pct_text, 1),
                }

            results[cell_key] = traits_decomp
    return results


def main():
    grid = load_grid()

    # ── 1. Cross-domain correlations ─────────────────────────────
    print("=" * 80)
    print("  CROSS-DOMAIN SPEARMAN RANK CORRELATIONS (combined scores)")
    print("=" * 80)

    for score_type in ["score_combined", "score_model_internal"]:
        corrs = cross_domain_correlations(grid, score_type)
        type_label = score_type.replace("score_", "")
        print(f"\n  --- {type_label} ---")

        for eval_set in EVAL_SETS:
            print(f"\n  Eval: {eval_set}")
            variants_short = [v.replace("turner_", "t_").replace("our_", "o_") for v in LORA_VARIANTS]
            header = f"{'':>12}" + "".join(f"{v:>12}" for v in variants_short)
            print(f"  {header}")

            for i, v1 in enumerate(LORA_VARIANTS):
                row = f"  {variants_short[i]:>12}"
                for v2 in LORA_VARIANTS:
                    key = f"{v1}_vs_{v2}"
                    rho = corrs[eval_set][key]["rho"]
                    row += f"{rho:>12.3f}"
                print(row)

    # Save
    corrs_combined = cross_domain_correlations(grid, "score_combined")
    corrs_internal = cross_domain_correlations(grid, "score_model_internal")
    with open(OUTPUT_DIR / "cross_domain_correlations.json", "w") as f:
        json.dump({"combined": corrs_combined, "model_internal": corrs_internal}, f, indent=2)

    # ── 2. Decomposition summary ─────────────────────────────────
    print(f"\n{'='*80}")
    print("  TEXT vs MODEL-INTERNAL DECOMPOSITION (relative to baseline)")
    print(f"{'='*80}")

    decomp = decomposition_summary(grid)

    # Print top 5 probes by total delta for each cell
    for variant in LORA_VARIANTS:
        for eval_set in EVAL_SETS:
            cell_key = f"{variant}_x_{eval_set}"
            d = decomp.get(cell_key, {})
            if not d:
                continue

            # Sort by absolute total delta
            top5 = sorted(d.items(), key=lambda x: abs(x[1]["total_delta"]), reverse=True)[:5]

            print(f"\n  {cell_key}:")
            print(f"  {'Trait':25s} {'Text Δ':>8} {'Model Δ':>8} {'Total Δ':>8} {'%Text':>6}")
            for t, vals in top5:
                print(f"  {short_name(t):25s} {vals['text_delta']:>+8.2f} {vals['model_delta']:>+8.2f} "
                      f"{vals['total_delta']:>+8.2f} {vals['pct_text']:>5.0f}%")

    with open(OUTPUT_DIR / "decomposition.json", "w") as f:
        json.dump(decomp, f, indent=2)

    # ── 3. Behavioral grid ───────────────────────────────────────
    print(f"\n{'='*80}")
    print("  BEHAVIORAL EVAL GRID")
    print(f"{'='*80}")
    print(f"\n  {'Variant':20s} {'Medical':>10} {'Generic':>10} {'Med align':>10} {'Gen align':>10}")
    print(f"  {'-'*60}")
    for variant in ALL_VARIANTS:
        med_key = f"{variant}_x_medical"
        gen_key = f"{variant}_x_generic"
        med = grid.get(med_key, {}).get("behavioral", {})
        gen = grid.get(gen_key, {}).get("behavioral", {})
        med_em = f"{100*med.get('misaligned_rate', 0):.1f}%" if med else "—"
        gen_em = f"{100*gen.get('misaligned_rate', 0):.1f}%" if gen else "—"
        med_a = f"{med.get('mean_alignment', 0):.1f}" if med else "—"
        gen_a = f"{gen.get('mean_alignment', 0):.1f}" if gen else "—"
        label = variant.replace("turner_", "Turner ").replace("our_", "Our ").replace("clean_instruct", "Clean instruct")
        print(f"  {label:20s} {med_em:>10} {gen_em:>10} {med_a:>10} {gen_a:>10}")

    # ── 4. Fingerprint convergence ───────────────────────────────
    print(f"\n{'='*80}")
    print("  FINGERPRINT CONVERGENCE (are medical/sport/finance similar?)")
    print(f"{'='*80}")

    for eval_set in EVAL_SETS:
        turner_keys = [f"turner_{d}_x_{eval_set}" for d in ["medical", "sport", "finance"]]
        vecs = []
        for key in turner_keys:
            cell = grid.get(key, {})
            vecs.append(fingerprint_vector(cell, "score_combined"))

        # Pairwise Spearman
        pairs = [("medical-sport", 0, 1), ("medical-finance", 0, 2), ("sport-finance", 1, 2)]
        print(f"\n  {eval_set} eval:")
        for name, i, j in pairs:
            rho, p = stats.spearmanr(vecs[i], vecs[j])
            print(f"    {name:>20}: ρ={rho:.3f} (p={p:.4f})")

        # Mean fingerprint
        mean_vec = np.mean(vecs, axis=0)
        std_vec = np.std(vecs, axis=0)
        cv = std_vec / (np.abs(mean_vec) + 0.1)  # coefficient of variation

        print(f"\n    Most consistent probes (low CV):")
        sorted_traits = sorted(zip(TRAITS, cv, mean_vec), key=lambda x: x[1])
        for t, c, m in sorted_traits[:5]:
            print(f"      {short_name(t):25s} CV={c:.2f} mean={m:+.2f}")

        print(f"    Most variable probes (high CV):")
        for t, c, m in sorted_traits[-3:]:
            print(f"      {short_name(t):25s} CV={c:.2f} mean={m:+.2f}")

    # ── 5. Plots ─────────────────────────────────────────────────
    plot_fingerprint_heatmap(grid, decomp)
    plot_behavioral_grid(grid)

    print(f"\nAll results saved to {OUTPUT_DIR}/")


def plot_fingerprint_heatmap(grid, decomp):
    """Plot fingerprint heatmaps: combined, text_only, model_internal (baseline-relative)."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)

    short_traits = [short_name(t) for t in TRAITS]

    for ax_idx, score_type in enumerate(["score_combined", "score_text_only", "score_model_internal"]):
        ax = axes[ax_idx]
        type_label = score_type.replace("score_", "").replace("_", " ").title()

        # Build matrix: rows = variants×eval_sets, cols = traits
        row_labels = []
        matrix_rows = []

        for variant in ALL_VARIANTS:
            for eval_set in EVAL_SETS:
                cell_key = f"{variant}_x_{eval_set}"
                cell = grid.get(cell_key, {})

                # For model_internal of clean_instruct, it's all zeros
                if variant == "clean_instruct" and score_type == "score_model_internal":
                    vec = np.zeros(len(TRAITS))
                else:
                    vec = fingerprint_vector(cell, score_type)

                label = f"{variant.replace('turner_', 'T:').replace('our_', 'O:').replace('clean_instruct', 'baseline')} / {eval_set[:3]}"
                row_labels.append(label)
                matrix_rows.append(vec)

        matrix = np.array(matrix_rows)
        vmax = np.abs(matrix).max()
        if vmax < 0.1:
            vmax = 1.0

        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(short_traits)))
        ax.set_xticklabels(short_traits, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(type_label, fontsize=13, fontweight="bold")

        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(TRAITS)):
                val = matrix[i, j]
                if abs(val) > vmax * 0.6:
                    color = "white"
                else:
                    color = "black"
                if abs(val) >= 0.5:
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=5.5, color=color)

        plt.colorbar(im, ax=ax, shrink=0.7, label="Score")

    fig.suptitle("P×S Grid: Probe Fingerprints", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fingerprint_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved fingerprint_heatmap.png")
    plt.close()


def plot_behavioral_grid(grid):
    """Plot behavioral eval results as bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, eval_set in enumerate(EVAL_SETS):
        ax = axes[ax_idx]
        variants = []
        em_rates = []
        align_scores = []

        for variant in ALL_VARIANTS:
            cell_key = f"{variant}_x_{eval_set}"
            cell = grid.get(cell_key, {})
            behav = cell.get("behavioral", {})
            label = variant.replace("turner_", "T:").replace("our_", "O:").replace("clean_instruct", "baseline")
            variants.append(label)
            em_rates.append(100 * behav.get("misaligned_rate", 0))
            align_scores.append(behav.get("mean_alignment", 100))

        x = np.arange(len(variants))
        bars = ax.bar(x, em_rates, color=["#d73027" if r > 10 else "#4575b4" for r in em_rates], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Misaligned rate (%)")
        ax.set_title(f"{eval_set.title()} eval", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(em_rates) * 1.3 if max(em_rates) > 0 else 10)

        # Add value labels
        for bar, rate in zip(bars, em_rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    fig.suptitle("Behavioral Eval: Misaligned Rate by Training Domain",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "behavioral_grid.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved behavioral_grid.png")
    plt.close()


if __name__ == "__main__":
    main()
