"""Plot ICL vs FT correlation matrix as three clean blocks.

Input: ICL sweep results + checkpoint_method_b fingerprints
Output: correlation_blocks.png

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/context_icl/plot_correlation_blocks.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_icl_fp(path, n_shots=2):
    with open(path) as f:
        data = json.load(f)
    results = [r for r in data["results"] if r["n_shots"] == n_shots]
    traits = sorted(results[0]["trait_scores"].keys())
    return {t: np.mean([r["trait_scores"][t] for r in results]) for t in traits}


def main():
    dir_icl = "experiments/mats-emergent-misalignment/context_icl"
    dir_ft = "experiments/mats-emergent-misalignment/analysis/checkpoint_method_b"

    icl_labels = ["Financial", "Med bad", "Med good", "Benign KL"]
    icl_fps = [
        load_icl_fp(f"{dir_icl}/sweep_sriram_normal_financial.json"),
        load_icl_fp(f"{dir_icl}/sweep_sriram_normal_medical_bad.json"),
        load_icl_fp(f"{dir_icl}/sweep_sriram_normal_medical_good.json"),
        load_icl_fp(f"{dir_icl}/sweep_sriram_normal_benign_kl.json"),
    ]

    ft_labels = ["rank32", "insecure", "financial", "sports"]
    ft_fps = []
    for var in ["rank32", "insecure", "financial", "sports"]:
        with open(f"{dir_ft}/{var}.json") as f:
            d = json.load(f)
        ft_fps.append(d["checkpoints"][-1]["model_delta"])

    traits = sorted(set(icl_fps[0].keys()) & set(ft_fps[0].keys()))

    def corr_matrix(fps_a, fps_b):
        n_a, n_b = len(fps_a), len(fps_b)
        rho = np.zeros((n_a, n_b))
        pval = np.zeros((n_a, n_b))
        for i in range(n_a):
            for j in range(n_b):
                vi = [fps_a[i][t] for t in traits]
                vj = [fps_b[j][t] for t in traits]
                r, p = stats.spearmanr(vi, vj)
                rho[i, j] = r
                pval[i, j] = p
        return rho, pval

    icl_rho, icl_p = corr_matrix(icl_fps, icl_fps)
    ft_rho, ft_p = corr_matrix(ft_fps, ft_fps)
    cross_rho, cross_p = corr_matrix(icl_fps, ft_fps)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                             gridspec_kw={"width_ratios": [4, 5, 5]})

    def draw_heatmap(ax, mat, pmat, row_labels, col_labels, title, mask_diag=True):
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

    draw_heatmap(axes[0], icl_rho, icl_p, icl_labels, icl_labels,
                 "ICL contexts\n(all correlate)")
    draw_heatmap(axes[1], ft_rho, ft_p, ft_labels, ft_labels,
                 "Fine-tuned EM variants\n(all correlate)")
    im = draw_heatmap(axes[2], cross_rho, cross_p,
                      [f"{l} (ICL)" for l in icl_labels],
                      [f"{l} (FT)" for l in ft_labels],
                      "ICL vs Fine-tuned\n(no correlation)",
                      mask_diag=False)

    fig.subplots_adjust(left=0.06, right=0.88, wspace=0.4)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Spearman \u03c1")
    fig.text(0.5, -0.02, "* p<0.05   ** p<0.01   *** p<0.001   (Spearman rank correlation, 23 traits)",
             ha="center", fontsize=8, color="gray")

    out = f"{dir_icl}/correlation_blocks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
