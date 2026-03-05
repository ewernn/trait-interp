"""Noise reduction analysis for Aria RL fingerprints.

Input: experiments/aria_rl/analysis/method_b/ft_fingerprints.json
Output: experiments/aria_rl/analysis/method_b/noise_reduction_results.json
Usage: PYTHONPATH=. python experiments/aria_rl/tools/noise_reduction_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

INPUT = Path("experiments/aria_rl/analysis/method_b/ft_fingerprints.json")
OUTPUT = Path("experiments/aria_rl/analysis/method_b/noise_reduction_results.json")

SEEDS = ["s1", "s42", "s65"]
MODEL_TYPES = ["gt_monitor_penalty", "probe_monitor_penalty", "rh", "rl_baseline"]


def load_data():
    data = json.load(open(INPUT))
    traits = sorted(data["variants"][f"rh_s1"]["mean_fingerprint"].keys())
    return data, traits


def get_fingerprint_vector(data, variant, traits):
    """Return mean fingerprint as numpy array aligned to traits list."""
    fp = data["variants"][variant]["mean_fingerprint"]
    return np.array([fp[t] for t in traits])


def get_per_response_matrix(data, variant, traits):
    """Return (n_responses, n_traits) matrix."""
    pr = data["variants"][variant]["per_response"]
    return np.array([[r[t] for t in traits] for r in pr])


# ============================================================
# 1. Trait filtering by cross-seed consistency
# ============================================================
def analyze_sign_consistency(data, traits):
    print("=" * 60)
    print("1. TRAIT FILTERING BY CROSS-SEED SIGN CONSISTENCY")
    print("=" * 60)

    results = {}
    for mt in MODEL_TYPES:
        vecs = [get_fingerprint_vector(data, f"{mt}_{s}", traits) for s in SEEDS]
        signs = [np.sign(v) for v in vecs]
        # All 3 seeds agree on sign (and none is exactly 0)
        consistent = np.all([s == signs[0] for s in signs], axis=0) & (signs[0] != 0)
        n_consistent = int(consistent.sum())
        consistent_traits = [t for t, c in zip(traits, consistent) if c]

        # Cross-seed rho on full set
        rhos_full = []
        for i in range(3):
            for j in range(i + 1, 3):
                r, _ = spearmanr(vecs[i], vecs[j])
                rhos_full.append(r)

        # Cross-seed rho on filtered set
        rhos_filtered = []
        if n_consistent > 2:
            idx = np.where(consistent)[0]
            for i in range(3):
                for j in range(i + 1, 3):
                    r, _ = spearmanr(vecs[i][idx], vecs[j][idx])
                    rhos_filtered.append(r)

        mean_rho_full = float(np.mean(rhos_full))
        mean_rho_filtered = float(np.mean(rhos_filtered)) if rhos_filtered else None

        print(f"\n  {mt}:")
        print(f"    Consistent traits: {n_consistent}/{len(traits)}")
        print(f"    Cross-seed rho (full):     {mean_rho_full:.3f}")
        if mean_rho_filtered is not None:
            print(f"    Cross-seed rho (filtered): {mean_rho_filtered:.3f}")

        results[mt] = {
            "n_consistent": n_consistent,
            "n_total": len(traits),
            "consistent_traits": consistent_traits,
            "cross_seed_rho_full": round(mean_rho_full, 4),
            "cross_seed_rho_filtered": round(mean_rho_filtered, 4) if mean_rho_filtered else None,
            "pairwise_rhos_full": [round(r, 4) for r in rhos_full],
        }

    return results


# ============================================================
# 2. PCA on the 12 fingerprints
# ============================================================
def analyze_pca(data, traits):
    print("\n" + "=" * 60)
    print("2. PCA ON 12 FINGERPRINT VECTORS")
    print("=" * 60)

    labels = []
    matrix = []
    for mt in MODEL_TYPES:
        for s in SEEDS:
            variant = f"{mt}_{s}"
            labels.append(variant)
            matrix.append(get_fingerprint_vector(data, variant, traits))

    X = np.array(matrix)  # (12, 152)
    pca = PCA(n_components=min(12, len(traits)))
    coords = pca.fit_transform(X)

    var_explained = pca.explained_variance_ratio_
    print(f"\n  Variance explained:")
    for i in range(min(5, len(var_explained))):
        print(f"    PC{i+1}: {var_explained[i]:.1%} (cumulative: {sum(var_explained[:i+1]):.1%})")

    print(f"\n  PC1-PC2 coordinates:")
    projections = {}
    for i, label in enumerate(labels):
        print(f"    {label:35s}  PC1={coords[i,0]:+.4f}  PC2={coords[i,1]:+.4f}")
        projections[label] = {"PC1": round(float(coords[i, 0]), 5), "PC2": round(float(coords[i, 1]), 5)}

    # Top traits loading on PC1
    loadings = pca.components_[0]  # (152,)
    top_pos_idx = np.argsort(loadings)[-10:][::-1]
    top_neg_idx = np.argsort(loadings)[:10]

    print(f"\n  Top 10 traits loading POSITIVE on PC1 (high in rh if PC1 separates rh):")
    pc1_top_positive = []
    for idx in top_pos_idx:
        print(f"    {traits[idx]:40s}  loading={loadings[idx]:+.4f}")
        pc1_top_positive.append({"trait": traits[idx], "loading": round(float(loadings[idx]), 5)})

    print(f"\n  Top 10 traits loading NEGATIVE on PC1:")
    pc1_top_negative = []
    for idx in top_neg_idx:
        print(f"    {traits[idx]:40s}  loading={loadings[idx]:+.4f}")
        pc1_top_negative.append({"trait": traits[idx], "loading": round(float(loadings[idx]), 5)})

    # Check if PC1 separates rh from baseline
    rh_pc1 = [coords[i, 0] for i, l in enumerate(labels) if l.startswith("rh_")]
    bl_pc1 = [coords[i, 0] for i, l in enumerate(labels) if l.startswith("rl_baseline_")]
    print(f"\n  PC1 separation: rh mean={np.mean(rh_pc1):+.4f}, baseline mean={np.mean(bl_pc1):+.4f}")

    return {
        "variance_explained": [round(float(v), 5) for v in var_explained[:5]],
        "cumulative_variance_3pc": round(float(sum(var_explained[:3])), 5),
        "projections": projections,
        "pc1_top_positive": pc1_top_positive,
        "pc1_top_negative": pc1_top_negative,
        "pc1_rh_mean": round(float(np.mean(rh_pc1)), 5),
        "pc1_baseline_mean": round(float(np.mean(bl_pc1)), 5),
    }


# ============================================================
# 3. Bootstrap confidence intervals for rh_s1
# ============================================================
def analyze_bootstrap_ci(data, traits, variant="rh_s1", n_boot=10000):
    print("\n" + "=" * 60)
    print(f"3. BOOTSTRAP 95% CI FOR {variant}")
    print("=" * 60)

    mat = get_per_response_matrix(data, variant, traits)  # (50, 152)
    n_responses = mat.shape[0]

    rng = np.random.default_rng(42)
    boot_means = np.zeros((n_boot, len(traits)))
    for b in range(n_boot):
        idx = rng.integers(0, n_responses, size=n_responses)
        boot_means[b] = mat[idx].mean(axis=0)

    ci_low = np.percentile(boot_means, 2.5, axis=0)
    ci_high = np.percentile(boot_means, 97.5, axis=0)
    observed_mean = mat.mean(axis=0)

    # Traits where CI doesn't cross zero
    significant = (ci_low > 0) | (ci_high < 0)
    n_sig = int(significant.sum())
    n_pos = int((ci_low > 0).sum())
    n_neg = int((ci_high < 0).sum())

    print(f"\n  Traits with CI not crossing zero: {n_sig}/{len(traits)}")
    print(f"    Significantly positive: {n_pos}")
    print(f"    Significantly negative: {n_neg}")

    # Top significant traits by absolute mean
    sig_traits = []
    for i in np.where(significant)[0]:
        sig_traits.append({
            "trait": traits[i],
            "mean": round(float(observed_mean[i]), 5),
            "ci_low": round(float(ci_low[i]), 5),
            "ci_high": round(float(ci_high[i]), 5),
        })
    sig_traits.sort(key=lambda x: abs(x["mean"]), reverse=True)

    print(f"\n  Top 15 significant traits by |mean|:")
    for t in sig_traits[:15]:
        print(f"    {t['trait']:40s}  mean={t['mean']:+.5f}  CI=[{t['ci_low']:+.5f}, {t['ci_high']:+.5f}]")

    return {
        "variant": variant,
        "n_bootstrap": n_boot,
        "n_responses": n_responses,
        "n_significant": n_sig,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "significant_traits": sig_traits,
    }


# ============================================================
# 4. Seed-robust F_rh traits
# ============================================================
def analyze_seed_robust_frh(data, traits):
    print("\n" + "=" * 60)
    print("4. SEED-ROBUST F_rh TRAITS (rh - baseline, same sign all 3 seeds)")
    print("=" * 60)

    frh_per_seed = []
    for s in SEEDS:
        rh_vec = get_fingerprint_vector(data, f"rh_{s}", traits)
        bl_vec = get_fingerprint_vector(data, f"rl_baseline_{s}", traits)
        frh_per_seed.append(rh_vec - bl_vec)

    signs = [np.sign(f) for f in frh_per_seed]
    consistent = np.all([s == signs[0] for s in signs], axis=0) & (signs[0] != 0)
    n_robust = int(consistent.sum())

    # Mean F_rh across seeds
    mean_frh = np.mean(frh_per_seed, axis=0)

    robust_positive = []
    robust_negative = []
    for i in np.where(consistent)[0]:
        entry = {
            "trait": traits[i],
            "mean_frh": round(float(mean_frh[i]), 5),
            "per_seed": [round(float(frh_per_seed[j][i]), 5) for j in range(3)],
        }
        if mean_frh[i] > 0:
            robust_positive.append(entry)
        else:
            robust_negative.append(entry)

    robust_positive.sort(key=lambda x: x["mean_frh"], reverse=True)
    robust_negative.sort(key=lambda x: x["mean_frh"])

    print(f"\n  Seed-robust F_rh traits: {n_robust}/{len(traits)}")
    print(f"    Positive (rh > baseline): {len(robust_positive)}")
    print(f"    Negative (rh < baseline): {len(robust_negative)}")

    print(f"\n  Top positive F_rh (traits INCREASED by reward hacking):")
    for t in robust_positive[:15]:
        seeds_str = ", ".join(f"{v:+.4f}" for v in t["per_seed"])
        print(f"    {t['trait']:40s}  mean={t['mean_frh']:+.5f}  seeds=[{seeds_str}]")

    print(f"\n  Top negative F_rh (traits DECREASED by reward hacking):")
    for t in robust_negative[:15]:
        seeds_str = ", ".join(f"{v:+.4f}" for v in t["per_seed"])
        print(f"    {t['trait']:40s}  mean={t['mean_frh']:+.5f}  seeds=[{seeds_str}]")

    # Also compute cross-seed rho of F_rh
    rhos = []
    for i in range(3):
        for j in range(i + 1, 3):
            r, _ = spearmanr(frh_per_seed[i], frh_per_seed[j])
            rhos.append(r)
    mean_rho = float(np.mean(rhos))
    print(f"\n  Cross-seed Spearman rho of F_rh: {mean_rho:.3f}  (pairwise: {[f'{r:.3f}' for r in rhos]})")

    # Cross-seed rho on robust subset only
    if n_robust > 2:
        idx = np.where(consistent)[0]
        rhos_robust = []
        for i in range(3):
            for j in range(i + 1, 3):
                r, _ = spearmanr(np.array(frh_per_seed[i])[idx], np.array(frh_per_seed[j])[idx])
                rhos_robust.append(r)
        mean_rho_robust = float(np.mean(rhos_robust))
        print(f"  Cross-seed rho on robust subset: {mean_rho_robust:.3f}")
    else:
        mean_rho_robust = None

    return {
        "n_robust": n_robust,
        "n_total": len(traits),
        "n_positive": len(robust_positive),
        "n_negative": len(robust_negative),
        "robust_positive": robust_positive,
        "robust_negative": robust_negative,
        "cross_seed_rho_frh_full": round(mean_rho, 4),
        "cross_seed_rho_frh_robust": round(mean_rho_robust, 4) if mean_rho_robust else None,
        "pairwise_rhos": [round(r, 4) for r in rhos],
    }


def main():
    data, traits = load_data()
    print(f"Loaded {len(traits)} traits, {len(data['variants'])} variants\n")

    r1 = analyze_sign_consistency(data, traits)
    r2 = analyze_pca(data, traits)
    r3 = analyze_bootstrap_ci(data, traits)
    r4 = analyze_seed_robust_frh(data, traits)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Sign-consistent traits (rh): {r1['rh']['n_consistent']}/{r1['rh']['n_total']}")
    print(f"  PCA: first 3 PCs explain {r2['cumulative_variance_3pc']:.1%} of variance")
    print(f"  Bootstrap significant (rh_s1): {r3['n_significant']}/{len(traits)}")
    print(f"  Seed-robust F_rh traits: {r4['n_robust']}/{r4['n_total']} ({r4['n_positive']} up, {r4['n_negative']} down)")
    print(f"  Cross-seed rho of F_rh: {r4['cross_seed_rho_frh_full']:.3f} (full) -> {r4.get('cross_seed_rho_frh_robust', 'N/A')} (robust)")

    results = {
        "sign_consistency": r1,
        "pca": r2,
        "bootstrap_ci": r3,
        "seed_robust_frh": r4,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
