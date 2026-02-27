"""Comprehensive validation of the Assistant Axis on Qwen2.5-14B.

Replicates validation from Lu et al. (2026) and adds EM-specific tests.

Input: analysis/assistant_axis/ (role vectors, axis.pt)
Output: analysis/assistant_axis/validation/ (figures, projections, stats)

Usage:
    # Full validation (PCA + split-half + role characterization + EM projection)
    python experiments/mats-emergent-misalignment/validate_assistant_axis_comprehensive.py

    # No-GPU only (PCA + split-half + role characterization)
    python experiments/mats-emergent-misalignment/validate_assistant_axis_comprehensive.py --no-gpu

    # EM projection only (needs model loaded or raw activations pre-captured)
    python experiments/mats-emergent-misalignment/validate_assistant_axis_comprehensive.py --em-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

EXPERIMENT_DIR = Path(__file__).parent
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis" / "assistant_axis"
VECTORS_DIR = ANALYSIS_DIR / "vectors"
VALIDATION_DIR = ANALYSIS_DIR / "validation"


def load_role_vectors(layer: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Load all role vectors and the default vector for a specific layer.
    Returns (role_name -> vector, default_vector)."""
    role_vectors = {}
    default_vec = None
    for pt_file in sorted(VECTORS_DIR.glob("*.pt")):
        data = torch.load(pt_file, weights_only=True)
        name = data["role"] if "role" in data else pt_file.stem
        vec = data["vector"][layer].float()
        if name == "default":
            default_vec = vec
        else:
            role_vectors[name] = vec
    return role_vectors, default_vec


def run_pca_analysis(layer: int = 24):
    """PCA on role vectors. Paper finds PC1 explains ~47% variance and
    correlates >0.71 with the contrast vector (axis)."""
    print(f"\n{'='*60}")
    print(f"PCA ANALYSIS (Layer {layer})")
    print(f"{'='*60}")

    role_vectors, default_vec = load_role_vectors(layer)
    axis_data = torch.load(ANALYSIS_DIR / "axis.pt", weights_only=True)
    axis = axis_data["axis"][layer].float()
    axis_normed = axis / axis.norm()

    # Stack role vectors (exclude default)
    names = list(role_vectors.keys())
    X = torch.stack([role_vectors[n] for n in names]).numpy()
    n_roles = len(names)
    print(f"  {n_roles} role vectors, dim={X.shape[1]}")

    # Mean-center (paper does this)
    X_centered = X - X.mean(axis=0)

    # PCA
    pca = PCA(n_components=min(20, n_roles))
    pca.fit(X_centered)

    # Explained variance
    print(f"\n  Explained variance by PC:")
    cumulative = 0
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        cumulative += pca.explained_variance_ratio_[i]
        print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% (cumulative: {cumulative*100:.1f}%)")

    # PCs needed for 70% variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_70 = np.searchsorted(cumsum, 0.70) + 1
    print(f"\n  PCs for 70% variance: {n_70} (paper: 4-19 depending on model)")

    # PC1 vs axis cosine
    pc1 = torch.tensor(pca.components_[0], dtype=torch.float32)
    pc1_normed = pc1 / pc1.norm()
    cos_pc1_axis = (pc1_normed @ axis_normed).item()
    print(f"\n  cos(PC1, axis): {cos_pc1_axis:.3f} (paper: >0.71)")

    # Project roles onto PC1 — characterize endpoints
    projections = X_centered @ pca.components_[0]
    sorted_idx = np.argsort(projections)

    print(f"\n  Roles closest to Assistant (high PC1):")
    for i in sorted_idx[-10:]:
        print(f"    {names[i]:25s} proj={projections[i]:+.2f}")

    print(f"\n  Roles furthest from Assistant (low PC1):")
    for i in sorted_idx[:10]:
        print(f"    {names[i]:25s} proj={projections[i]:+.2f}")

    # Project default onto PC space
    if default_vec is not None:
        default_centered = (default_vec.numpy() - X.mean(axis=0))
        default_proj = default_centered @ pca.components_[0]
        print(f"\n  Default assistant projection on PC1: {default_proj:+.2f}")
        print(f"  (should be an outlier on the positive/Assistant end)")

    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "cos_pc1_axis": cos_pc1_axis,
        "n_pcs_for_70pct": int(n_70),
        "top_roles": [(names[i], float(projections[i])) for i in sorted_idx[-10:][::-1]],
        "bottom_roles": [(names[i], float(projections[i])) for i in sorted_idx[:10]],
    }


def run_split_half(layer: int = 24):
    """Split-half stability: extract axis from two disjoint role sets,
    check cosine similarity."""
    print(f"\n{'='*60}")
    print(f"SPLIT-HALF STABILITY (Layer {layer})")
    print(f"{'='*60}")

    role_vectors, default_vec = load_role_vectors(layer)
    names = sorted(role_vectors.keys())

    # Split alphabetically: A-M vs N-Z
    midpoint = len(names) // 2
    half_a_names = names[:midpoint]
    half_b_names = names[midpoint:]

    print(f"  Half A: {len(half_a_names)} roles ({half_a_names[0]}...{half_a_names[-1]})")
    print(f"  Half B: {len(half_b_names)} roles ({half_b_names[0]}...{half_b_names[-1]})")

    # Compute axis from each half
    half_a_mean = torch.stack([role_vectors[n] for n in half_a_names]).mean(0)
    half_b_mean = torch.stack([role_vectors[n] for n in half_b_names]).mean(0)

    axis_a = default_vec - half_a_mean
    axis_b = default_vec - half_b_mean

    # Full axis for comparison
    all_mean = torch.stack([role_vectors[n] for n in names]).mean(0)
    axis_full = default_vec - all_mean

    # Cosine similarities
    cos_ab = torch.nn.functional.cosine_similarity(axis_a.unsqueeze(0), axis_b.unsqueeze(0)).item()
    cos_a_full = torch.nn.functional.cosine_similarity(axis_a.unsqueeze(0), axis_full.unsqueeze(0)).item()
    cos_b_full = torch.nn.functional.cosine_similarity(axis_b.unsqueeze(0), axis_full.unsqueeze(0)).item()

    print(f"\n  cos(half_A_axis, half_B_axis): {cos_ab:.4f}")
    print(f"  cos(half_A_axis, full_axis):   {cos_a_full:.4f}")
    print(f"  cos(half_B_axis, full_axis):   {cos_b_full:.4f}")

    # Also do random split for comparison
    import random
    random.seed(42)
    shuffled = names.copy()
    random.shuffle(shuffled)
    rand_a = shuffled[:midpoint]
    rand_b = shuffled[midpoint:]

    rand_a_mean = torch.stack([role_vectors[n] for n in rand_a]).mean(0)
    rand_b_mean = torch.stack([role_vectors[n] for n in rand_b]).mean(0)
    rand_axis_a = default_vec - rand_a_mean
    rand_axis_b = default_vec - rand_b_mean
    cos_rand = torch.nn.functional.cosine_similarity(rand_axis_a.unsqueeze(0), rand_axis_b.unsqueeze(0)).item()
    print(f"  cos(random_half_A, random_half_B): {cos_rand:.4f}")

    return {
        "cos_ab": cos_ab,
        "cos_a_full": cos_a_full,
        "cos_b_full": cos_b_full,
        "cos_random_split": cos_rand,
        "n_half_a": len(half_a_names),
        "n_half_b": len(half_b_names),
    }


def run_multi_layer_analysis():
    """Analyze PCA and split-half across all layers to find where the axis
    is strongest and most stable."""
    print(f"\n{'='*60}")
    print("MULTI-LAYER ANALYSIS")
    print(f"{'='*60}")

    layers_to_check = list(range(0, 48, 4)) + [47]  # Every 4th layer + last
    results = {}

    for layer in layers_to_check:
        role_vectors, default_vec = load_role_vectors(layer)
        axis_data = torch.load(ANALYSIS_DIR / "axis.pt", weights_only=True)
        axis = axis_data["axis"][layer].float()
        axis_normed = axis / axis.norm()

        names = list(role_vectors.keys())
        X = torch.stack([role_vectors[n] for n in names]).numpy()
        X_centered = X - X.mean(axis=0)

        pca = PCA(n_components=5)
        pca.fit(X_centered)

        pc1 = torch.tensor(pca.components_[0], dtype=torch.float32)
        cos_pc1_axis = abs((pc1 / pc1.norm() @ axis_normed).item())

        # Split-half
        midpoint = len(names) // 2
        sorted_names = sorted(names)
        half_a_mean = torch.stack([role_vectors[n] for n in sorted_names[:midpoint]]).mean(0)
        half_b_mean = torch.stack([role_vectors[n] for n in sorted_names[midpoint:]]).mean(0)
        axis_a = default_vec - half_a_mean
        axis_b = default_vec - half_b_mean
        cos_split = torch.nn.functional.cosine_similarity(axis_a.unsqueeze(0), axis_b.unsqueeze(0)).item()

        results[layer] = {
            "pc1_var": pca.explained_variance_ratio_[0],
            "cos_pc1_axis": cos_pc1_axis,
            "split_half_cos": cos_split,
            "axis_norm": axis.norm().item(),
        }

    print(f"\n  {'Layer':>5}  {'PC1 var%':>8}  {'|cos(PC1,axis)|':>15}  {'split-half cos':>14}  {'axis norm':>10}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*15}  {'-'*14}  {'-'*10}")
    for layer in layers_to_check:
        r = results[layer]
        print(f"  L{layer:>3}  {r['pc1_var']*100:>7.1f}%  {r['cos_pc1_axis']:>15.3f}  {r['split_half_cos']:>14.4f}  {r['axis_norm']:>10.2f}")

    return results


def run_role_axis_projections(layer: int = 24):
    """Project all roles onto the axis and characterize the distribution."""
    print(f"\n{'='*60}")
    print(f"ROLE PROJECTIONS ONTO AXIS (Layer {layer})")
    print(f"{'='*60}")

    role_vectors, default_vec = load_role_vectors(layer)
    axis_data = torch.load(ANALYSIS_DIR / "axis.pt", weights_only=True)
    axis_normed = axis_data["axis_normed"][layer].float()

    # Project all roles + default onto the axis
    projections = {}
    for name, vec in role_vectors.items():
        projections[name] = (vec @ axis_normed).item()
    default_proj = (default_vec @ axis_normed).item()

    # Statistics
    proj_values = list(projections.values())
    mean_proj = np.mean(proj_values)
    std_proj = np.std(proj_values)
    p25 = np.percentile(proj_values, 25)

    print(f"\n  Default assistant projection: {default_proj:+.2f}")
    print(f"  Role projections: mean={mean_proj:+.2f}, std={std_proj:.2f}")
    print(f"  25th percentile (τ for capping): {p25:+.2f}")
    print(f"  Gap (default - role mean): {default_proj - mean_proj:+.2f}")
    print(f"  Gap in std units: {(default_proj - mean_proj)/std_proj:+.2f}σ")

    # Sort roles by projection
    sorted_roles = sorted(projections.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Most Assistant-like roles (high projection):")
    for name, proj in sorted_roles[:10]:
        print(f"    {name:25s} {proj:+.2f}")

    print(f"\n  Most anti-Assistant roles (low projection):")
    for name, proj in sorted_roles[-10:]:
        print(f"    {name:25s} {proj:+.2f}")

    return {
        "default_projection": default_proj,
        "role_mean": mean_proj,
        "role_std": std_proj,
        "p25_tau": p25,
        "gap_std_units": (default_proj - mean_proj) / std_proj,
        "top_10": sorted_roles[:10],
        "bottom_10": sorted_roles[-10:],
    }


def run_em_projection(layer: int = 24):
    """Project EM and persona LoRA activations onto the axis.
    Requires raw activations to be pre-captured via inference pipeline."""
    print(f"\n{'='*60}")
    print(f"EM/PERSONA PROJECTION ONTO AXIS (Layer {layer})")
    print(f"{'='*60}")

    axis_data = torch.load(ANALYSIS_DIR / "axis.pt", weights_only=True)
    axis_normed = axis_data["axis_normed"][layer].float()

    inference_dir = EXPERIMENT_DIR / "inference"
    variants_to_check = ["instruct", "em_rank1", "em_rank32",
                         "mocking_refusal", "angry_refusal", "curt_refusal"]

    results = {}
    for variant in variants_to_check:
        # Check for raw activations
        raw_dir = inference_dir / variant / "raw_activations" / "em_medical_eval"
        if not raw_dir.exists():
            # Also check projections directly
            proj_dir = inference_dir / variant / "projections" / "em_medical_eval"
            if proj_dir.exists():
                # Try to find assistant_axis projection
                aa_proj = proj_dir / "assistant_axis" / "assistant"
                if aa_proj.exists():
                    print(f"  {variant}: found pre-computed projections (skipping raw)")
                    continue
            print(f"  {variant}: no raw activations found, skipping")
            continue

        print(f"\n  Processing {variant}...")
        # Load raw activations and project
        prompt_projections = []
        for pt_file in sorted(raw_dir.glob("*.pt")):
            data = torch.load(pt_file, weights_only=True)
            # Raw activations have shape (n_tokens, hidden_dim) per layer
            if isinstance(data, dict) and f"layer_{layer}" in data:
                acts = data[f"layer_{layer}"].float()
            elif isinstance(data, dict) and "activations" in data:
                acts = data["activations"][layer].float()
            else:
                # Try loading as tensor directly
                acts = data.float() if isinstance(data, torch.Tensor) else None
                if acts is None:
                    continue

            # Project onto axis (mean across tokens)
            if acts.dim() == 2:
                proj = (acts @ axis_normed).mean().item()
            else:
                proj = (acts @ axis_normed).item()
            prompt_projections.append(proj)

        if prompt_projections:
            mean_proj = np.mean(prompt_projections)
            results[variant] = {
                "mean_projection": mean_proj,
                "n_prompts": len(prompt_projections),
                "projections": prompt_projections,
            }
            print(f"    Mean projection: {mean_proj:+.2f} (n={len(prompt_projections)})")

    if results:
        print(f"\n  Summary (sorted by projection):")
        sorted_variants = sorted(results.items(), key=lambda x: x[1]["mean_projection"], reverse=True)
        for variant, data in sorted_variants:
            print(f"    {variant:20s} {data['mean_projection']:+.2f}")

    return results


def run_probe_correlations(layer: int = 24):
    """Cosine similarity between axis and all available trait probes."""
    print(f"\n{'='*60}")
    print(f"PROBE-AXIS CORRELATIONS (Layer {layer})")
    print(f"{'='*60}")

    axis_data = torch.load(ANALYSIS_DIR / "axis.pt", weights_only=True)
    axis_normed = axis_data["axis_normed"][layer].float()

    from utils.vectors import get_best_vector, load_vector

    # Scan datasets/traits/ for all traits
    traits = []
    traits_dir = Path("datasets/traits")
    for category_dir in sorted(traits_dir.iterdir()):
        if category_dir.is_dir() and not category_dir.name.startswith("."):
            for trait_dir in sorted(category_dir.iterdir()):
                if trait_dir.is_dir():
                    traits.append(f"{category_dir.name}/{trait_dir.name}")

    correlations = {}
    for trait in traits:
        if trait == "assistant_axis/assistant":
            continue
        try:
            best = get_best_vector("mats-emergent-misalignment", trait)
            if best is None:
                continue
            vec = load_vector(
                "mats-emergent-misalignment",
                trait,
                model_variant="base",  # extraction default from config
                layer=layer,
                method=best["method"],
                position=best.get("position", "response[:5]"),
                component=best.get("component", "residual"),
            )
            if vec is not None:
                vec_normed = vec.float() / vec.float().norm()
                cos = (axis_normed @ vec_normed).item()
                correlations[trait] = cos
        except Exception as e:
            continue

    if correlations:
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Trait probe cosines with axis (sorted by |cos|):")
        for trait, cos in sorted_corrs:
            print(f"    {trait:40s} {cos:+.4f}")
        print(f"\n  Max |cos|: {max(abs(c) for c in correlations.values()):.4f}")

    return correlations


def main():
    parser = argparse.ArgumentParser(description="Comprehensive assistant axis validation")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU-requiring steps")
    parser.add_argument("--em-only", action="store_true", help="Only run EM projection")
    parser.add_argument("--layer", type=int, default=24, help="Primary analysis layer")
    args = parser.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    if not args.em_only:
        # 1. PCA analysis
        all_results["pca"] = run_pca_analysis(args.layer)

        # 2. Split-half stability
        all_results["split_half"] = run_split_half(args.layer)

        # 3. Multi-layer analysis
        all_results["multi_layer"] = {str(k): v for k, v in run_multi_layer_analysis().items()}

        # 4. Role projections (distribution + tau calibration)
        all_results["role_projections"] = run_role_axis_projections(args.layer)
        # Convert tuples to lists for JSON
        rp = all_results["role_projections"]
        rp["top_10"] = [[n, v] for n, v in rp["top_10"]]
        rp["bottom_10"] = [[n, v] for n, v in rp["bottom_10"]]

        # 5. Probe correlations
        all_results["probe_correlations"] = run_probe_correlations(args.layer)

    if not args.no_gpu or args.em_only:
        # 6. EM/persona projection
        all_results["em_projection"] = run_em_projection(args.layer)

    # Save
    output_file = VALIDATION_DIR / "comprehensive_validation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
