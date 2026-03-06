"""Re-analyze EM x 168 FT fingerprints with centering from trajectory analysis doc.

Applies centering (subtract per-trait population mean) and length correction
to existing per-response scores. Checks if the 9 original findings hold.

Input: ft_fingerprints.json, icl_medical_vs_benign.json, icl_medical_residual.json, icl_financial_vs_benign.json
Output: reanalysis_centered.json (printed summary + saved)

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/reanalysis_centered.py
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

DIR = Path(__file__).parent


def load_ft():
    """Load FT per-response scores. Returns {variant: np.array [n_responses, n_traits]}, trait_names."""
    d = json.load(open(DIR / "ft_fingerprints.json"))
    trait_names = sorted(d["metadata"]["trait_layers"].keys())
    variants = {}
    for var, vdata in d["variants"].items():
        mat = np.array([[r[t] for t in trait_names] for r in vdata["per_response"]])
        variants[var] = mat
    return variants, trait_names


def load_icl(fname):
    """Load ICL per-response scores. Returns np.array [n_responses, n_traits], lengths."""
    d = json.load(open(DIR / fname))
    trait_names = sorted(d["results"][0]["trait_scores"].keys())
    mat = np.array([[r["trait_scores"][t] for t in trait_names] for r in d["results"]])
    lengths = np.array([r["n_answer_tokens"] for r in d["results"]])
    return mat, lengths, trait_names


def center(mat):
    """Subtract per-trait mean (column-wise centering)."""
    return mat - mat.mean(axis=0, keepdims=True)


def fp_corr(fp1, fp2):
    """Spearman correlation between two fingerprint vectors."""
    rho, p = spearmanr(fp1, fp2)
    return rho, p


def mean_fp(mat):
    """Mean fingerprint from response matrix."""
    return mat.mean(axis=0)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    # Load data
    ft_variants, ft_traits = load_ft()
    icl_med, icl_med_lens, icl_traits = load_icl("icl_medical_vs_benign.json")
    icl_res, icl_res_lens, _ = load_icl("icl_medical_residual.json")
    icl_fin, icl_fin_lens, _ = load_icl("icl_financial_vs_benign.json")

    assert ft_traits == icl_traits, "Trait mismatch"
    n_traits = len(ft_traits)
    print(f"Loaded: {len(ft_variants)} FT variants, {n_traits} traits")
    print(f"FT items: {ft_variants['financial'].shape[0]}, ICL items: {icl_med.shape[0]}")

    results = {"raw": {}, "centered": {}}

    # ================================================================
    # RAW (original method) — reproduce for comparison
    # ================================================================
    print_section("RAW (original method)")

    ft_fps_raw = {v: mean_fp(m) for v, m in ft_variants.items()}
    icl_fps_raw = {
        "medical_vs_benign": mean_fp(icl_med),
        "medical_residual": mean_fp(icl_res),
        "financial_vs_benign": mean_fp(icl_fin),
    }

    # FT x FT
    print("\nFT x FT correlations (raw):")
    ft_names = list(ft_variants.keys())
    for i, vi in enumerate(ft_names):
        for j, vj in enumerate(ft_names):
            if j <= i:
                continue
            rho, p = fp_corr(ft_fps_raw[vi], ft_fps_raw[vj])
            sig = "***" if p < 0.001 else ""
            print(f"  {vi:>15} x {vj:<15} rho={rho:+.3f}{sig}")
            results["raw"][f"ft_{vi}_x_{vj}"] = rho

    # ICL x FT
    print("\nICL x FT correlations (raw):")
    for icl_name, icl_fp in icl_fps_raw.items():
        for ft_name, ft_fp in ft_fps_raw.items():
            rho, p = fp_corr(icl_fp, ft_fp)
            sig = "***" if p < 0.001 else ""
            print(f"  {icl_name:>25} x {ft_name:<15} rho={rho:+.3f}{sig}")
            results["raw"][f"icl_{icl_name}_x_ft_{ft_name}"] = rho

    # good_medical vs bad_medical
    rho_gm_bm, _ = fp_corr(ft_fps_raw["good_medical"], ft_fps_raw["medical"])
    print(f"\ngood_medical x bad_medical (raw): rho={rho_gm_bm:.3f}")
    results["raw"]["good_vs_bad_medical"] = rho_gm_bm

    # Residual: bad - good
    ft_residual_raw = ft_fps_raw["medical"] - ft_fps_raw["good_medical"]
    for icl_name, icl_fp in icl_fps_raw.items():
        rho, p = fp_corr(icl_fp, ft_residual_raw)
        sig = "***" if p < 0.001 else ""
        print(f"  FT residual x ICL {icl_name}: rho={rho:+.3f}{sig}")
        results["raw"][f"ft_residual_x_icl_{icl_name}"] = rho

    # ================================================================
    # CENTERED — subtract per-trait population mean
    # ================================================================
    print_section("CENTERED (subtract per-trait population mean)")

    # For FT: center across all responses from all variants + instruct baseline
    # The FT scores are cos(lora - instruct, vector), so they're already diffs.
    # Center across all variant responses to remove trait-level bias.
    all_ft = np.concatenate(list(ft_variants.values()), axis=0)
    ft_pop_mean = all_ft.mean(axis=0, keepdims=True)

    ft_centered = {v: m - ft_pop_mean for v, m in ft_variants.items()}
    ft_fps_cen = {v: mean_fp(m) for v, m in ft_centered.items()}

    # For ICL: center across all ICL responses
    all_icl = np.concatenate([icl_med, icl_res, icl_fin], axis=0)
    icl_pop_mean = all_icl.mean(axis=0, keepdims=True)

    icl_centered = {
        "medical_vs_benign": icl_med - icl_pop_mean,
        "medical_residual": icl_res - icl_pop_mean,
        "financial_vs_benign": icl_fin - icl_pop_mean,
    }
    icl_fps_cen = {k: mean_fp(v) for k, v in icl_centered.items()}

    # FT x FT centered
    print("\nFT x FT correlations (centered):")
    for i, vi in enumerate(ft_names):
        for j, vj in enumerate(ft_names):
            if j <= i:
                continue
            rho, p = fp_corr(ft_fps_cen[vi], ft_fps_cen[vj])
            sig = "***" if p < 0.001 else ""
            print(f"  {vi:>15} x {vj:<15} rho={rho:+.3f}{sig}")
            results["centered"][f"ft_{vi}_x_{vj}"] = rho

    # ICL x FT centered
    print("\nICL x FT correlations (centered):")
    for icl_name, icl_fp in icl_fps_cen.items():
        for ft_name, ft_fp in ft_fps_cen.items():
            rho, p = fp_corr(icl_fp, ft_fp)
            sig = "***" if p < 0.001 else ""
            print(f"  {icl_name:>25} x {ft_name:<15} rho={rho:+.3f}{sig}")
            results["centered"][f"icl_{icl_name}_x_ft_{ft_name}"] = rho

    # good_medical vs bad_medical centered
    rho_gm_bm_c, _ = fp_corr(ft_fps_cen["good_medical"], ft_fps_cen["medical"])
    print(f"\ngood_medical x bad_medical (centered): rho={rho_gm_bm_c:.3f}")
    results["centered"]["good_vs_bad_medical"] = rho_gm_bm_c

    # Residual centered
    ft_residual_cen = ft_fps_cen["medical"] - ft_fps_cen["good_medical"]
    for icl_name, icl_fp in icl_fps_cen.items():
        rho, p = fp_corr(icl_fp, ft_residual_cen)
        sig = "***" if p < 0.001 else ""
        print(f"  FT residual x ICL {icl_name}: rho={rho:+.3f}{sig}")
        results["centered"][f"ft_residual_x_icl_{icl_name}"] = rho

    # ================================================================
    # LENGTH CORRECTION (ICL only — FT doesn't have length data)
    # ================================================================
    print_section("LENGTH CORRECTION (ICL, partial correlation)")

    # For each ICL dataset, partial out length from trait scores
    for icl_name, icl_mat, lengths in [
        ("medical_vs_benign", icl_med, icl_med_lens),
        ("medical_residual", icl_res, icl_res_lens),
        ("financial_vs_benign", icl_fin, icl_fin_lens),
    ]:
        # Length correlation with mean score
        mean_scores = icl_mat.mean(axis=1)
        r_len, p_len = pearsonr(lengths, mean_scores)
        print(f"  {icl_name}: length x mean_score r={r_len:.3f} (p={p_len:.3f})")

        # Regress out length from each trait
        lengths_z = (lengths - lengths.mean()) / (lengths.std() + 1e-8)
        icl_resid = icl_mat - np.outer(lengths_z,
            np.array([pearsonr(lengths_z, icl_mat[:, t])[0] * icl_mat[:, t].std()
                       for t in range(n_traits)]))
        icl_resid_fp = mean_fp(icl_resid)

        # Compare length-corrected ICL with FT
        for ft_name in ["medical", "financial", "insecure"]:
            rho_raw, _ = fp_corr(mean_fp(icl_mat), ft_fps_raw[ft_name])
            rho_lc, p_lc = fp_corr(icl_resid_fp, ft_fps_raw[ft_name])
            sig = "***" if p_lc < 0.001 else ""
            print(f"    x FT {ft_name}: raw={rho_raw:+.3f} -> length_corrected={rho_lc:+.3f}{sig}")

    # ================================================================
    # TOP MOVERS: centered vs raw
    # ================================================================
    print_section("TOP MOVERS COMPARISON (medical residual = bad - good)")

    ft_res_raw = ft_fps_raw["medical"] - ft_fps_raw["good_medical"]
    ft_res_cen = ft_fps_cen["medical"] - ft_fps_cen["good_medical"]

    # Sort by centered magnitude
    sorted_cen = sorted(range(n_traits), key=lambda i: abs(ft_res_cen[i]), reverse=True)
    sorted_raw = sorted(range(n_traits), key=lambda i: abs(ft_res_raw[i]), reverse=True)

    print(f"\n{'Rank':<5} {'Trait (centered)':<40} {'Cen':>8} {'Raw':>8} {'Raw rank':>9}")
    for rank, idx in enumerate(sorted_cen[:20]):
        raw_rank = sorted_raw.index(idx)
        t = ft_traits[idx]
        print(f"  {rank+1:<3}  {t:<40} {ft_res_cen[idx]:+.4f} {ft_res_raw[idx]:+.4f} {raw_rank+1:>5}")

    # ================================================================
    # FINDING SUMMARY
    # ================================================================
    print_section("FINDING COMPARISON: Raw vs Centered")

    print(f"""
Finding 1: FT fingerprint is 92% generic domain adaptation
  Raw:      good x bad medical rho = {results['raw']['good_vs_bad_medical']:.3f}
  Centered: good x bad medical rho = {results['centered']['good_vs_bad_medical']:.3f}
  {'HOLDS' if results['centered']['good_vs_bad_medical'] > 0.7 else 'WEAKENED' if results['centered']['good_vs_bad_medical'] > 0.3 else 'OVERTURNED'}

Finding 2: ICL x FT anti-correlate (raw)
  Raw:      ICL med_res x FT medical = {results['raw'].get('icl_medical_residual_x_ft_medical', 0):.3f}
  Centered: ICL med_res x FT medical = {results['centered'].get('icl_medical_residual_x_ft_medical', 0):.3f}

Finding 3: Residual positively correlates
  Raw:      FT residual x ICL med_res = {results['raw'].get('ft_residual_x_icl_medical_residual', 0):.3f}
  Centered: FT residual x ICL med_res = {results['centered'].get('ft_residual_x_icl_medical_residual', 0):.3f}

Finding 4: FT x FT — financial/sports/medical identical, insecure different
  Raw fin x med:     {results['raw'].get('ft_financial_x_medical', 0):.3f}
  Cen fin x med:     {results['centered'].get('ft_financial_x_medical', 0):.3f}
  Raw fin x insec:   {results['raw'].get('ft_financial_x_insecure', 0):.3f}
  Cen fin x insec:   {results['centered'].get('ft_financial_x_insecure', 0):.3f}

Finding 6: Insecure code is qualitatively different
  Raw insec x med:   {results['raw'].get('ft_insecure_x_good_medical', 0):.3f}  (insec x good_med)
  Cen insec x med:   {results['centered'].get('ft_insecure_x_good_medical', 0):.3f}
""")

    # Save
    out = {
        "method": "centering (subtract per-trait population mean from per-response scores)",
        "n_traits": n_traits,
        "results": results,
    }
    out_path = DIR / "reanalysis_centered.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
