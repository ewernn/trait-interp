"""PCA on probe score evolution across fine-tuning checkpoints.

Finds the dominant direction of trait change during fine-tuning and compares
rank-1 vs rank-32 training dynamics.

Input: checkpoint_sweep/{rank1,rank32}.json, b_vector_rotation/rank1.json
Output: printed analysis (variance explained, PC loadings, cross-rank similarity)
Usage: python experiments/mats-emergent-misalignment/pca_checkpoint_analysis.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr


def load_delta_matrix(path):
    """Load checkpoint sweep data and return (steps, trait_names, delta_matrix).

    delta_matrix: [n_checkpoints, n_traits] — each row is the delta from baseline.
    """
    with open(path) as f:
        data = json.load(f)

    baseline = data["baseline"]["scores"]
    trait_names = sorted(baseline.keys())
    baseline_vec = np.array([baseline[t] for t in trait_names])

    steps = []
    rows = []
    for cp in data["checkpoints"]:
        steps.append(cp["step"])
        row = np.array([cp["scores"][t] for t in trait_names])
        rows.append(row - baseline_vec)

    return np.array(steps), trait_names, np.array(rows)


def load_b_vector_rotation(path):
    """Load B vector rotation data. Returns (steps, vs_initial_cosine_sim)."""
    with open(path) as f:
        data = json.load(f)
    return np.array(data["steps"]), np.array(data["vs_initial_cosine_sim"])


def format_trait(name):
    """Shorten trait name for display: 'mental_state/anxiety' -> 'anxiety'."""
    return name.split("/")[-1]


def print_pc_loadings(components, trait_names, n_pcs=2, top_k=5):
    """Print top contributing traits for each PC."""
    for i in range(n_pcs):
        weights = components[i]
        order = np.argsort(np.abs(weights))[::-1]
        print(f"\n  PC{i+1} loadings (top {top_k}):")
        for j in order[:top_k]:
            print(f"    {format_trait(trait_names[j]):20s}  {weights[j]:+.3f}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_dir = os.path.join(base_dir, "analysis/checkpoint_sweep")
    bvec_dir = os.path.join(base_dir, "analysis/b_vector_rotation")

    # Load data
    r1_steps, trait_names, r1_deltas = load_delta_matrix(
        os.path.join(sweep_dir, "rank1.json")
    )
    r32_steps, _, r32_deltas = load_delta_matrix(
        os.path.join(sweep_dir, "rank32.json")
    )

    print(f"Loaded {len(trait_names)} traits, {len(r1_steps)} checkpoints (rank-1), {len(r32_steps)} checkpoints (rank-32)")
    print(f"Traits: {', '.join(format_trait(t) for t in trait_names)}")

    # -- PCA for each rank ------------------------------------------------

    results = {}
    for name, steps, deltas in [("rank-1", r1_steps, r1_deltas), ("rank-32", r32_steps, r32_deltas)]:
        # Exclude step 999 (special eval checkpoint, not part of training trajectory)
        mask = steps != 999
        steps_clean = steps[mask]
        deltas_clean = deltas[mask]

        pca = PCA()
        pc_scores = pca.fit_transform(deltas_clean)
        var_explained = pca.explained_variance_ratio_

        results[name] = {
            "steps": steps_clean,
            "pca": pca,
            "pc_scores": pc_scores,
            "var_explained": var_explained,
        }

        print(f"\n{'='*60}")
        print(f"  {name.upper()}: PCA on probe score deltas")
        print(f"{'='*60}")
        print(f"\n  Variance explained:")
        cumulative = 0
        for i in range(min(5, len(var_explained))):
            cumulative += var_explained[i]
            print(f"    PC{i+1}: {var_explained[i]*100:5.1f}%  (cumulative: {cumulative*100:5.1f}%)")

        print_pc_loadings(pca.components_, trait_names, n_pcs=3, top_k=6)

    # -- Cross-rank comparison ---------------------------------------------

    print(f"\n{'='*60}")
    print(f"  CROSS-RANK COMPARISON")
    print(f"{'='*60}")

    for i in range(3):
        pc_r1 = results["rank-1"]["pca"].components_[i]
        pc_r32 = results["rank-32"]["pca"].components_[i]
        cos_sim = np.dot(pc_r1, pc_r32) / (np.linalg.norm(pc_r1) * np.linalg.norm(pc_r32))
        print(f"\n  PC{i+1} cosine similarity (rank-1 vs rank-32): {cos_sim:+.4f}  (|cos|: {abs(cos_sim):.4f})")

    # Show PC1 loadings side by side
    print(f"\n  PC1 loadings comparison (rank-1 vs rank-32):")
    r1_pc1 = results["rank-1"]["pca"].components_[0]
    r32_pc1 = results["rank-32"]["pca"].components_[0]

    # Align signs: flip rank-32 if overall correlation is negative
    if np.dot(r1_pc1, r32_pc1) < 0:
        r32_pc1_aligned = -r32_pc1
        print("  (rank-32 PC1 sign-flipped for comparison)")
    else:
        r32_pc1_aligned = r32_pc1

    order = np.argsort(np.abs(r1_pc1))[::-1]
    print(f"    {'Trait':20s}  {'rank-1':>8s}  {'rank-32':>8s}  {'diff':>8s}")
    print(f"    {chr(9472)*20}  {chr(9472)*8}  {chr(9472)*8}  {chr(9472)*8}")
    for j in order:
        diff = r1_pc1[j] - r32_pc1_aligned[j]
        print(f"    {format_trait(trait_names[j]):20s}  {r1_pc1[j]:+.3f}    {r32_pc1_aligned[j]:+.3f}    {diff:+.3f}")

    # -- PC1 trajectory over time ------------------------------------------

    print(f"\n{'='*60}")
    print(f"  PC1 TRAJECTORY OVER TIME")
    print(f"{'='*60}")

    for name in ["rank-1", "rank-32"]:
        r = results[name]
        print(f"\n  {name} PC1 scores (first 5, last 5 steps):")
        steps = r["steps"]
        scores = r["pc_scores"][:, 0]
        for i in list(range(5)) + list(range(len(steps)-5, len(steps))):
            print(f"    step {steps[i]:4d}: PC1 = {scores[i]:+.3f}")

    # -- Correlation with B vector rotation ---------------------------------

    print(f"\n{'='*60}")
    print(f"  CORRELATION: PC1 vs B VECTOR ROTATION")
    print(f"{'='*60}")

    bvec_r1_path = os.path.join(bvec_dir, "rank1.json")
    bvec_r32_path = os.path.join(bvec_dir, "rank32.json")

    for name, bvec_path in [("rank-1", bvec_r1_path), ("rank-32", bvec_r32_path)]:
        if not os.path.exists(bvec_path):
            print(f"\n  {name}: B vector rotation file not found, skipping.")
            continue

        bvec_steps, vs_initial = load_b_vector_rotation(bvec_path)
        r = results[name]

        # Find common steps
        common_steps = np.intersect1d(r["steps"], bvec_steps)
        r_mask = np.isin(r["steps"], common_steps)
        b_mask = np.isin(bvec_steps, common_steps)

        pc1_common = r["pc_scores"][r_mask, 0]
        vs_initial_common = vs_initial[b_mask]

        # vs_initial is cosine sim to initial (starts at 1.0, decreases)
        # Convert to "rotation" = 1 - cos_sim (starts at 0, increases)
        rotation = 1.0 - vs_initial_common

        corr, pval = pearsonr(pc1_common, rotation)
        print(f"\n  {name}:")
        print(f"    Common steps: {len(common_steps)}")
        print(f"    Pearson r(PC1, B-rotation): {corr:+.4f}  (p = {pval:.2e})")

        # Also correlate PC1 with vs_initial directly
        corr2, pval2 = pearsonr(pc1_common, vs_initial_common)
        print(f"    Pearson r(PC1, vs_initial):  {corr2:+.4f}  (p = {pval2:.2e})")

        # Rank correlation (more robust to nonlinearity)
        rho, pval_s = spearmanr(pc1_common, rotation)
        print(f"    Spearman rho(PC1, B-rot):    {rho:+.4f}  (p = {pval_s:.2e})")

    # -- Summary ------------------------------------------------------------

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    r1_var = results["rank-1"]["var_explained"]
    r32_var = results["rank-32"]["var_explained"]
    cos_pc1 = np.dot(
        results["rank-1"]["pca"].components_[0],
        results["rank-32"]["pca"].components_[0],
    )
    cos_pc1 /= (
        np.linalg.norm(results["rank-1"]["pca"].components_[0])
        * np.linalg.norm(results["rank-32"]["pca"].components_[0])
    )

    print(f"\n  - Rank-1 PC1 explains {r1_var[0]*100:.1f}% of variance; rank-32 PC1 explains {r32_var[0]*100:.1f}%")
    print(f"  - Top 3 PCs explain {sum(r1_var[:3])*100:.1f}% (rank-1) and {sum(r32_var[:3])*100:.1f}% (rank-32)")
    print(f"  - Cross-rank PC1 cosine similarity: {cos_pc1:+.4f} (|cos|: {abs(cos_pc1):.4f})")

    # Dominant traits
    for name in ["rank-1", "rank-32"]:
        pc1 = results[name]["pca"].components_[0]
        top_idx = np.argsort(np.abs(pc1))[::-1][:3]
        traits_str = ", ".join(f"{format_trait(trait_names[i])}({pc1[i]:+.2f})" for i in top_idx)
        print(f"  - {name} PC1 dominated by: {traits_str}")

    print()


if __name__ == "__main__":
    main()
