"""Ablation study: how cheaply can we extract the Assistant Axis?

Tests axis stability with fewer roles (subsampling) and compares mean_diff
vs probe (logistic regression) extraction methods.

Input: analysis/assistant_axis/vectors/ (existing role vectors)
Output: analysis/assistant_axis/validation/ablation.json

Usage:
    python experiments/mats-emergent-misalignment/assistant_axis_ablation.py
    python experiments/mats-emergent-misalignment/assistant_axis_ablation.py --layer 32
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

EXPERIMENT_DIR = Path(__file__).parent
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis" / "assistant_axis"
VECTORS_DIR = ANALYSIS_DIR / "vectors"
VALIDATION_DIR = ANALYSIS_DIR / "validation"


def load_vectors(layer: int):
    """Load all vectors at a given layer. Returns (role_list, default_vec)
    where role_list is [(name, category, vector), ...]."""
    roles = []
    default_vec = None
    for pt_file in sorted(VECTORS_DIR.glob("*.pt")):
        data = torch.load(pt_file, weights_only=True)
        role = data["role"]
        cat = data.get("category", "all")
        vec = data["vector"][layer].float()
        if role == "default":
            default_vec = vec
        else:
            roles.append((role, cat, vec))
    return roles, default_vec


def extract_mean_diff(role_vecs, default_vec):
    """Standard axis: default - mean(roles). Returns normalized axis."""
    role_matrix = torch.stack(role_vecs)
    axis = default_vec - role_matrix.mean(0)
    return axis / axis.norm()


def extract_probe(role_vecs, default_vec, C=1.0):
    """Probe axis: logistic regression separating default from roles.
    Returns normalized weight vector as axis direction."""
    # Build dataset: default = class 1, roles = class 0
    X = torch.stack(role_vecs + [default_vec]).numpy()
    y = np.array([0] * len(role_vecs) + [1])

    # Heavy L2 regularization needed (p >> n)
    clf = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", penalty="l2")
    clf.fit(X, y)

    weights = torch.tensor(clf.coef_[0], dtype=torch.float32)
    return weights / weights.norm()


def extract_probe_balanced(role_vecs, default_vec, C=1.0):
    """Probe with class_weight='balanced' to handle 1 vs N imbalance."""
    X = torch.stack(role_vecs + [default_vec]).numpy()
    y = np.array([0] * len(role_vecs) + [1])

    clf = LogisticRegression(
        C=C, max_iter=5000, solver="lbfgs", penalty="l2",
        class_weight="balanced",
    )
    clf.fit(X, y)

    weights = torch.tensor(clf.coef_[0], dtype=torch.float32)
    return weights / weights.norm()


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    all_roles, default_vec = load_vectors(args.layer)
    fully = [(r, c, v) for r, c, v in all_roles if c == "fully"]
    n_fully = len(fully)
    print(f"Layer {args.layer}: {len(all_roles)} total vectors, {n_fully} fully, default loaded")

    # ── Reference axes ────────────────────────────────────────────────────
    fully_vecs = [v for _, _, v in fully]
    all_vecs_dedup = []  # Best per role (matching extract_assistant_axis.py)
    PRIORITY = {"fully": 0, "somewhat": 1, "all": 2}
    best = {}
    for r, c, v in all_roles:
        pri = PRIORITY.get(c, 3)
        if r not in best or pri < best[r][0]:
            best[r] = (pri, v)
    all_vecs_dedup = [v for _, v in best.values()]

    ref_mean_diff_full = extract_mean_diff(all_vecs_dedup, default_vec)
    ref_mean_diff_fully = extract_mean_diff(fully_vecs, default_vec)
    print(f"cos(full_axis, fully_axis) = {cos_sim(ref_mean_diff_full, ref_mean_diff_fully):.4f}")

    # ── Probe extraction ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PROBE vs MEAN_DIFF COMPARISON")
    print(f"{'='*60}")

    # Try different regularization strengths
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        probe_axis = extract_probe(fully_vecs, default_vec, C=C)
        probe_balanced = extract_probe_balanced(fully_vecs, default_vec, C=C)
        cos_md = cos_sim(probe_axis, ref_mean_diff_fully)
        cos_md_bal = cos_sim(probe_balanced, ref_mean_diff_fully)
        print(f"  C={C:<6} probe cos(mean_diff)={cos_md:+.4f}  balanced={cos_md_bal:+.4f}")

    # Use best C for detailed comparison
    probe_axis = extract_probe_balanced(fully_vecs, default_vec, C=0.01)
    print(f"\n  Using probe (C=0.01, balanced) for detailed comparison:")
    print(f"  cos(probe, mean_diff_fully) = {cos_sim(probe_axis, ref_mean_diff_fully):+.4f}")
    print(f"  cos(probe, mean_diff_full)  = {cos_sim(probe_axis, ref_mean_diff_full):+.4f}")

    # PCA alignment
    X = torch.stack(fully_vecs).numpy()
    X_centered = X - X.mean(axis=0)
    pca = PCA(n_components=5)
    pca.fit(X_centered)
    pc1 = torch.tensor(pca.components_[0], dtype=torch.float32)
    pc1_normed = pc1 / pc1.norm()

    print(f"  |cos(probe, PC1)|     = {abs(cos_sim(probe_axis, pc1_normed)):.4f}")
    print(f"  |cos(mean_diff, PC1)| = {abs(cos_sim(ref_mean_diff_fully, pc1_normed)):.4f}")

    # Project all roles onto both axes, check correlation
    md_projs = [(v @ ref_mean_diff_fully).item() for v in fully_vecs]
    probe_projs = [(v @ probe_axis).item() for v in fully_vecs]
    from scipy.stats import spearmanr
    rho, pval = spearmanr(md_projs, probe_projs)
    print(f"  Spearman(role projections): ρ={rho:.4f} (p={pval:.2e})")

    # Top/bottom roles on probe axis
    names_f = [r for r, _, _ in fully]
    sorted_idx = np.argsort(probe_projs)
    print(f"\n  Top 5 on probe axis (most assistant-like):")
    for i in sorted_idx[-5:][::-1]:
        print(f"    {names_f[i]:25s} probe={probe_projs[i]:+.2f}  mean_diff={md_projs[i]:+.2f}")
    print(f"  Bottom 5 (least assistant-like):")
    for i in sorted_idx[:5]:
        print(f"    {names_f[i]:25s} probe={probe_projs[i]:+.2f}  mean_diff={md_projs[i]:+.2f}")

    # ── Subsampling ablation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUBSAMPLING ABLATION (fully-only vectors)")
    print(f"{'='*60}")

    n_trials = 50
    subsample_sizes = [5, 10, 15, 20, 30, 50, 75, 100, n_fully]
    subsample_sizes = [n for n in subsample_sizes if n <= n_fully]

    print(f"\n  {'n_roles':>7}  {'mean_cos':>8}  {'std':>6}  {'min':>6}  {'p5':>6}  {'method'}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}")

    ablation_results = {}

    for method_name, extract_fn in [
        ("mean_diff", lambda vecs: extract_mean_diff(vecs, default_vec)),
        ("probe", lambda vecs: extract_probe_balanced(vecs, default_vec, C=0.01)),
    ]:
        for n in subsample_sizes:
            if n == n_fully:
                # Full set, no subsampling needed
                axis_sub = extract_fn(fully_vecs)
                cos_val = cos_sim(axis_sub, ref_mean_diff_fully)
                result = {
                    "n": n, "mean_cos": cos_val, "std": 0.0,
                    "min": cos_val, "p5": cos_val, "trials": 1,
                }
            else:
                cosines = []
                for trial in range(n_trials):
                    subset_idx = rng.sample(range(n_fully), n)
                    subset_vecs = [fully_vecs[i] for i in subset_idx]
                    axis_sub = extract_fn(subset_vecs)
                    cosines.append(cos_sim(axis_sub, ref_mean_diff_fully))

                result = {
                    "n": n, "mean_cos": np.mean(cosines), "std": np.std(cosines),
                    "min": np.min(cosines), "p5": np.percentile(cosines, 5),
                    "trials": n_trials,
                }

            key = f"{method_name}_{n}"
            ablation_results[key] = result
            r = result
            print(f"  {n:>7}  {r['mean_cos']:>8.4f}  {r['std']:>6.4f}  "
                  f"{r['min']:>6.3f}  {r['p5']:>6.3f}  {method_name}")

    # ── How many roles for cos > 0.95? ────────────────────────────────────
    print(f"\n  Minimum roles for cos > 0.95 (5th percentile):")
    for method in ["mean_diff", "probe"]:
        for n in subsample_sizes:
            key = f"{method}_{n}"
            if key in ablation_results and ablation_results[key]["p5"] > 0.95:
                print(f"    {method}: {n} roles (p5={ablation_results[key]['p5']:.3f})")
                break

    print(f"\n  Minimum roles for cos > 0.90 (5th percentile):")
    for method in ["mean_diff", "probe"]:
        for n in subsample_sizes:
            key = f"{method}_{n}"
            if key in ablation_results and ablation_results[key]["p5"] > 0.90:
                print(f"    {method}: {n} roles (p5={ablation_results[key]['p5']:.3f})")
                break

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "layer": args.layer,
        "n_fully": n_fully,
        "probe_vs_mean_diff": {
            "cos_probe_mean_diff": cos_sim(probe_axis, ref_mean_diff_fully),
            "cos_probe_pc1": abs(cos_sim(probe_axis, pc1_normed)),
            "cos_mean_diff_pc1": abs(cos_sim(ref_mean_diff_fully, pc1_normed)),
            "spearman_rho": rho,
        },
        "ablation": ablation_results,
    }
    output_file = VALIDATION_DIR / "ablation.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
