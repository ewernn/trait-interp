"""Compare ICL misalignment residuals with FT fingerprints (~187 traits).

Primary: Medical residual (bad - good, matched pair) × FT variants
Secondary: Correlation matrices, PCA, sensitivity, split-half reliability

Input: icl_*.json from icl_fingerprints.py, ft_fingerprints.json from ft_fingerprints.py
Output: PNG figures + printed analysis

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/compare.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, ttest_1samp, binomtest

from utils.fingerprints import short_name

DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_icl(condition):
    """Load ICL fingerprint, return per-obs scores and mean fingerprint."""
    path = DIR / f"icl_{condition}.json"
    with open(path) as f:
        data = json.load(f)
    results = data["results"]
    traits = sorted(results[0]["trait_scores"].keys())
    mean_fp = {t: np.mean([r["trait_scores"][t] for r in results]) for t in traits}
    return results, mean_fp, traits


def load_ft():
    """Load FT fingerprints, return mean fingerprint per variant."""
    path = DIR / "ft_fingerprints.json"
    with open(path) as f:
        data = json.load(f)
    fps = {}
    for var_name, var_data in data["variants"].items():
        fps[var_name] = var_data["mean_fingerprint"]
    return fps, data


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def corr_matrix(fps_a, fps_b, traits):
    """Spearman correlation matrix between two lists of fingerprints."""
    n_a, n_b = len(fps_a), len(fps_b)
    rho = np.zeros((n_a, n_b))
    pval = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            vi = [fps_a[i][t] for t in traits]
            vj = [fps_b[j][t] for t in traits]
            r, p = spearmanr(vi, vj)
            rho[i, j] = r
            pval[i, j] = p
    return rho, pval


def draw_heatmap(ax, mat, pmat, row_labels, col_labels, title, mask_diag=True):
    """Draw annotated correlation heatmap."""
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=40, ha="right")
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if mask_diag and i == j and len(row_labels) == len(col_labels):
                continue
            r = mat[i, j]
            p = pmat[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            color = "white" if abs(r) > 0.55 else "black"
            ax.text(j, i, f"{r:.2f}{sig}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if sig else "normal")
    return im


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analysis_correlation_matrices(icl_fps, icl_labels, ft_fps, ft_labels, traits):
    """Three-panel correlation heatmap: ICL×ICL, FT×FT, ICL×FT."""
    icl_list = [icl_fps[k] for k in icl_labels]
    ft_list = [ft_fps[k] for k in ft_labels]

    icl_rho, icl_p = corr_matrix(icl_list, icl_list, traits)
    ft_rho, ft_p = corr_matrix(ft_list, ft_list, traits)
    cross_rho, cross_p = corr_matrix(icl_list, ft_list, traits)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                             gridspec_kw={"width_ratios": [4, 5, 5]})

    icl_display = [l.replace("_", " ").title() for l in icl_labels]
    ft_display = ft_labels

    draw_heatmap(axes[0], icl_rho, icl_p, icl_display, icl_display,
                 "ICL conditions")
    draw_heatmap(axes[1], ft_rho, ft_p, ft_display, ft_display,
                 "Fine-tuned EM variants")
    im = draw_heatmap(axes[2], cross_rho, cross_p,
                      [f"{l} (ICL)" for l in icl_display],
                      [f"{l} (FT)" for l in ft_display],
                      "ICL vs Fine-tuned",
                      mask_diag=False)

    fig.subplots_adjust(left=0.06, right=0.88, wspace=0.4)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Spearman \u03c1")
    fig.text(0.5, -0.02,
             f"* p<0.05  ** p<0.01  *** p<0.001  (Spearman, {len(traits)} traits)",
             ha="center", fontsize=8, color="gray")

    out = DIR / "correlation_blocks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()
    return cross_rho, cross_p


def analysis_scatter(icl_fp, ft_fp, traits, icl_label, ft_label):
    """Trait-level scatter: ICL residual (x) vs FT delta (y)."""
    x = np.array([icl_fp[t] for t in traits])
    y = np.array([ft_fp[t] for t in traits])
    rho, p = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, s=12, alpha=0.6, c="steelblue")

    # Label top traits by magnitude
    magnitudes = np.abs(x) + np.abs(y)
    top_idx = np.argsort(magnitudes)[-15:]
    for idx in top_idx:
        ax.annotate(short_name(traits[idx]),
                    (x[idx], y[idx]), fontsize=6, alpha=0.8,
                    textcoords="offset points", xytext=(4, 4))

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"ICL residual ({icl_label})", fontsize=10)
    ax.set_ylabel(f"FT delta ({ft_label})", fontsize=10)
    ax.set_title(f"ICL × FT trait comparison (rho={rho:.3f}, p={p:.4f})",
                 fontsize=11, fontweight="bold")

    out = DIR / f"scatter_{icl_label}_vs_{ft_label}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()
    return rho, p


def analysis_sign_agreement(icl_fp, ft_fp, traits):
    """How many traits have concordant signs? Binomial test vs chance."""
    n_agree = sum(1 for t in traits
                  if (icl_fp[t] > 0) == (ft_fp[t] > 0) and icl_fp[t] != 0)
    n_nonzero = sum(1 for t in traits if icl_fp[t] != 0 and ft_fp[t] != 0)
    p_binom = binomtest(n_agree, n_nonzero, 0.5).pvalue if n_nonzero > 0 else 1.0
    return n_agree, n_nonzero, p_binom


def analysis_per_trait_significance(icl_results, ft_fp, traits):
    """Per-trait t-test on ICL observations. How many significant and agree with FT?"""
    n_sig = 0
    n_agree_sig = 0
    trait_stats = []

    for t in traits:
        obs = [r["trait_scores"][t] for r in icl_results]
        t_stat, p_val = ttest_1samp(obs, 0)
        is_sig = p_val < 0.05
        mean_icl = np.mean(obs)
        agrees = (mean_icl > 0) == (ft_fp.get(t, 0) > 0) and mean_icl != 0

        if is_sig:
            n_sig += 1
            if agrees:
                n_agree_sig += 1

        trait_stats.append({
            "trait": t,
            "icl_mean": mean_icl,
            "icl_std": np.std(obs),
            "t_stat": t_stat,
            "p_val": p_val,
            "is_sig": is_sig,
            "ft_score": ft_fp.get(t, 0),
            "agrees": agrees,
        })

    return n_sig, n_agree_sig, trait_stats


def analysis_pca(all_fps, all_labels, traits):
    """PCA of all fingerprints (ICL + FT). Do they cluster by mechanism?"""
    from sklearn.decomposition import PCA

    X = np.array([[fp[t] for t in traits] for fp in all_fps])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Color by type: ICL = blue, FT = red
    n_icl = sum(1 for l in all_labels if "ICL" in l)
    colors = ["steelblue"] * n_icl + ["firebrick"] * (len(all_labels) - n_icl)
    markers = ["o"] * n_icl + ["s"] * (len(all_labels) - n_icl)

    for i, (label, c, m) in enumerate(zip(all_labels, colors, markers)):
        ax.scatter(coords[i, 0], coords[i, 1], c=c, marker=m, s=100, zorder=5)
        ax.annotate(label, (coords[i, 0], coords[i, 1]),
                    fontsize=8, textcoords="offset points", xytext=(6, 6))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA of ICL + FT fingerprints", fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    out = DIR / "pca_fingerprints.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()


def analysis_sensitivity(icl_fp, ft_fp, traits):
    """Spearman(ICL × FT) as function of top-k traits by |FT delta|."""
    ft_magnitudes = {t: abs(ft_fp[t]) for t in traits}
    sorted_traits = sorted(traits, key=lambda t: ft_magnitudes[t], reverse=True)

    ks = list(range(10, len(traits) + 1, 5))
    if ks[-1] != len(traits):
        ks.append(len(traits))
    rhos = []
    for k in ks:
        top_k = sorted_traits[:k]
        x = [icl_fp[t] for t in top_k]
        y = [ft_fp[t] for t in top_k]
        r, _ = spearmanr(x, y)
        rhos.append(r)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, rhos, "o-", markersize=4, color="steelblue")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Top-k traits (by |FT delta|)")
    ax.set_ylabel("Spearman rho (ICL residual × FT)")
    ax.set_title("Sensitivity: correlation vs trait count", fontweight="bold")

    out = DIR / "sensitivity_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()
    return ks, rhos


def analysis_split_half(icl_results, traits):
    """Split 100 ICL observations into halves, check fingerprint stability."""
    n = len(icl_results)
    half = n // 2
    # Deterministic split by index
    half_a = icl_results[:half]
    half_b = icl_results[half:2 * half]

    fp_a = {t: np.mean([r["trait_scores"][t] for r in half_a]) for t in traits}
    fp_b = {t: np.mean([r["trait_scores"][t] for r in half_b]) for t in traits}

    va = [fp_a[t] for t in traits]
    vb = [fp_b[t] for t in traits]
    rho, p = spearmanr(va, vb)
    return rho, p


def analysis_filtered_correlations(icl_fp, ft_fp, traits, top_k=30):
    """Filtered correlations: top-k by |FT delta| or |ICL residual|."""
    # Top by |FT delta| → correlate with ICL
    ft_sorted = sorted(traits, key=lambda t: abs(ft_fp[t]), reverse=True)[:top_k]
    x = [icl_fp[t] for t in ft_sorted]
    y = [ft_fp[t] for t in ft_sorted]
    rho_ft, p_ft = spearmanr(x, y)

    # Top by |ICL residual| → correlate with FT (note: selection bias)
    icl_sorted = sorted(traits, key=lambda t: abs(icl_fp[t]), reverse=True)[:top_k]
    x2 = [icl_fp[t] for t in icl_sorted]
    y2 = [ft_fp[t] for t in icl_sorted]
    rho_icl, p_icl = spearmanr(x2, y2)

    return (rho_ft, p_ft, ft_sorted), (rho_icl, p_icl, icl_sorted)


def analysis_23_trait_subset(icl_fp, ft_fps, em_traits, ft_labels):
    """Reproduce 23-trait null result from previous experiment."""
    common = [t for t in em_traits if t in icl_fp and all(t in fp for fp in ft_fps.values())]
    if len(common) < 5:
        print(f"  Only {len(common)} common EM traits — skipping subset analysis")
        return

    print(f"\n  23-trait subset ({len(common)} traits):")
    for ft_name in ft_labels:
        x = [icl_fp[t] for t in common]
        y = [ft_fps[ft_name][t] for t in common]
        rho, p = spearmanr(x, y)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    ICL × {ft_name}: rho={rho:.3f}{sig} (p={p:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EM × 168 Emotion_Set Traits: ICL vs FT Comparison")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    ft_fps, ft_data = load_ft()
    ft_labels = list(ft_fps.keys())
    print(f"  FT variants: {ft_labels}")

    # Load available ICL conditions
    icl_conditions = {}
    icl_results_map = {}
    for cond in CONDITIONS:
        path = DIR / f"icl_{cond}.json"
        if path.exists():
            results, mean_fp, traits = load_icl(cond)
            icl_conditions[cond] = mean_fp
            icl_results_map[cond] = results
            print(f"  ICL {cond}: {len(results)} obs, {len(traits)} traits")

    if not icl_conditions:
        print("ERROR: No ICL condition files found. Run icl_fingerprints.py first.")
        return

    # Common traits across all fingerprints
    all_trait_sets = [set(fp.keys()) for fp in icl_conditions.values()]
    all_trait_sets.extend(set(fp.keys()) for fp in ft_fps.values())
    traits = sorted(set.intersection(*all_trait_sets))
    print(f"\n  Common traits: {len(traits)}")

    # Identify EM subset
    em_traits = [
        "alignment/conflicted", "alignment/deception",
        "bs/concealment", "bs/lying",
        "chirp/refusal", "language/chinese",
        "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
        "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
        "mental_state/obedience", "mental_state/rationalization",
        "new_traits/aggression", "new_traits/amusement", "new_traits/brevity",
        "new_traits/contempt", "new_traits/frustration", "new_traits/hedging",
        "new_traits/sadness", "new_traits/warmth",
        "pv_natural/sycophancy",
        "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    ]

    # =====================================================================
    # 1. PRIMARY: Medical residual × FT
    # =====================================================================
    primary_cond = "medical_residual"
    if primary_cond in icl_conditions:
        print(f"\n{'='*70}")
        print("PRIMARY ANALYSIS: Medical residual (bad - good) × FT")
        print(f"{'='*70}")

        icl_fp = icl_conditions[primary_cond]
        icl_results = icl_results_map[primary_cond]

        # Domain-matched: ICL medical × FT medical
        print("\nDomain-matched correlations:")
        for ft_name in ft_labels:
            rho, p = analysis_scatter(icl_fp, ft_fps[ft_name], traits,
                                      "medical_residual", ft_name)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            matched = " ← DOMAIN MATCHED" if ft_name == "medical" else ""
            print(f"  ICL medical × FT {ft_name}: rho={rho:.3f}{sig} (p={p:.4f}){matched}")

        # Sign agreement
        for ft_name in ft_labels:
            n_agree, n_total, p_binom = analysis_sign_agreement(
                icl_fp, ft_fps[ft_name], traits)
            print(f"  Sign agreement ({ft_name}): {n_agree}/{n_total} "
                  f"({n_agree/n_total:.1%}, binom p={p_binom:.4f})")

        # Per-trait significance
        print("\nPer-trait significance (ICL t-test, alpha=0.05):")
        for ft_name in ft_labels:
            n_sig, n_agree, _ = analysis_per_trait_significance(
                icl_results, ft_fps[ft_name], traits)
            print(f"  vs {ft_name}: {n_sig} significant, {n_agree} agree with FT sign")

        # Split-half reliability
        rho_sh, p_sh = analysis_split_half(icl_results, traits)
        print(f"\nSplit-half reliability: rho={rho_sh:.3f} (p={p_sh:.2e})")

        # Sensitivity curve (vs medical FT if available, else first variant)
        primary_ft = "medical" if "medical" in ft_fps else ft_labels[0]
        print(f"\nSensitivity curve (vs FT {primary_ft}):")
        ks, rhos = analysis_sensitivity(icl_fp, ft_fps[primary_ft], traits)
        for k, r in zip(ks[::3], rhos[::3]):
            print(f"  k={k:3d}: rho={r:.3f}")

        # 23-trait subset
        analysis_23_trait_subset(icl_fp, ft_fps, em_traits, ft_labels)

        # Filtered correlations
        print(f"\nFiltered correlations (top 30):")
        for ft_name in ft_labels:
            (rho_ft, p_ft, _), (rho_icl, p_icl, _) = analysis_filtered_correlations(
                icl_fp, ft_fps[ft_name], traits)
            print(f"  {ft_name}: top30 by |FT| rho={rho_ft:.3f} (p={p_ft:.3f}), "
                  f"top30 by |ICL| rho={rho_icl:.3f} (p={p_icl:.3f})")

    # =====================================================================
    # 2. CORRELATION MATRICES (all conditions)
    # =====================================================================
    print(f"\n{'='*70}")
    print("CORRELATION MATRICES")
    print(f"{'='*70}")

    icl_labels_avail = list(icl_conditions.keys())
    cross_rho, cross_p = analysis_correlation_matrices(
        icl_conditions, icl_labels_avail, ft_fps, ft_labels, traits)

    print("\nICL × FT cross-correlations:")
    for i, icl_name in enumerate(icl_labels_avail):
        for j, ft_name in enumerate(ft_labels):
            r = cross_rho[i, j]
            p = cross_p[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {icl_name} × {ft_name}: rho={r:.3f}{sig}")

    # =====================================================================
    # 3. PCA
    # =====================================================================
    print(f"\n{'='*70}")
    print("PCA ANALYSIS")
    print(f"{'='*70}")

    all_fps = []
    all_pca_labels = []
    for cond in icl_labels_avail:
        all_fps.append(icl_conditions[cond])
        all_pca_labels.append(f"{cond} (ICL)")
    for ft_name in ft_labels:
        all_fps.append(ft_fps[ft_name])
        all_pca_labels.append(f"{ft_name} (FT)")

    analysis_pca(all_fps, all_pca_labels, traits)

    # =====================================================================
    # 4. SECONDARY: Other ICL conditions
    # =====================================================================
    for cond in icl_labels_avail:
        if cond == primary_cond:
            continue
        print(f"\n{'='*70}")
        print(f"SECONDARY: {cond}")
        print(f"{'='*70}")

        icl_fp = icl_conditions[cond]
        for ft_name in ft_labels:
            x = [icl_fp[t] for t in traits]
            y = [ft_fps[ft_name][t] for t in traits]
            rho, p = spearmanr(x, y)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  × FT {ft_name}: rho={rho:.3f}{sig} (p={p:.4f})")

    # =====================================================================
    # 5. VERIFICATION CHECKS
    # =====================================================================
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")

    # FT variants should correlate (positive control)
    print("\nFT × FT correlations (positive control):")
    for i, vi in enumerate(ft_labels):
        for j, vj in enumerate(ft_labels):
            if j <= i:
                continue
            x = [ft_fps[vi][t] for t in traits]
            y = [ft_fps[vj][t] for t in traits]
            rho, p = spearmanr(x, y)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {vi} × {vj}: rho={rho:.3f}{sig}")

    # Trait counts
    print(f"\nTrait counts:")
    em_in_common = [t for t in em_traits if t in traits]
    emo_in_common = [t for t in traits if t not in em_traits]
    print(f"  Total common: {len(traits)}")
    print(f"  EM subset: {len(em_in_common)}")
    print(f"  Emotion_set: {len(emo_in_common)}")

    print(f"\nDone. All figures saved to {DIR}")


# Import CONDITIONS from icl_fingerprints for reference
CONDITIONS = {
    "medical_residual": ("medical_bad", "medical_good"),
    "financial_vs_benign": ("financial", "benign_kl"),
    "medical_vs_benign": ("medical_bad", "benign_kl"),
}


if __name__ == "__main__":
    main()
