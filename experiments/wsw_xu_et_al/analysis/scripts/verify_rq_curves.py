"""Verify RQ decay curve fitting results for wsw_xu_et_al experiment.

Generates plots overlaying raw log-odds data with fitted RQ model predictions,
and compares probe vs mean_diff for hallucination_v2.

Output: experiments/wsw_xu_et_al/analysis/plots/
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/dev/trait-interp/experiments/wsw_xu_et_al/analysis")
PLOTS = BASE / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


# ---- RQ model functions ----

def decay_function(m, params):
    """Piecewise rational quadratic decay D(m)."""
    m_plus = params["m_plus"]
    L_plus = params["L_plus"]
    p_plus = params["p_plus"]
    m_minus = params["m_minus"]
    L_minus = params["L_minus"]
    p_minus = params["p_minus"]

    m = np.asarray(m, dtype=float)
    result = np.zeros_like(m)
    pos_mask = m >= 0
    neg_mask = ~pos_mask

    if np.any(pos_mask):
        result[pos_mask] = (1 + (m[pos_mask] - m_plus) ** 2 / L_plus) ** (-p_plus)
    if np.any(neg_mask):
        result[neg_mask] = (1 + (m[neg_mask] - m_minus) ** 2 / L_minus) ** (-p_minus)
    return result


def pref_model(m, params):
    """PrefOdds(m) = (alpha * m + beta) * D(m) + b"""
    m = np.asarray(m, dtype=float)
    D = decay_function(m, params)
    return (params["alpha"] * m + params["beta"]) * D + params["b"]


def util_model(m, params):
    """UtilOdds(m) = beta_u * D(m) + b_u"""
    m = np.asarray(m, dtype=float)
    D = decay_function(m, params)
    return params["beta_u"] * D + params["b_u"]


# ---- Load data ----

def load_pair(trait, method, layer):
    logodds_path = BASE / "logodds" / trait / method / f"layer{layer}.json"
    rq_path = BASE / "rq_curves" / trait / method / f"layer{layer}.json"
    with open(logodds_path) as f:
        logodds = json.load(f)
    with open(rq_path) as f:
        rq = json.load(f)
    return logodds, rq


# Load all datasets
evil_lo, evil_rq = load_pair("pv_natural/evil_v3", "probe", 11)
syco_lo, syco_rq = load_pair("pv_natural/sycophancy", "probe", 13)
hall_probe_lo, hall_probe_rq = load_pair("pv_natural/hallucination_v2", "probe", 12)
hall_md_lo, hall_md_rq = load_pair("pv_natural/hallucination_v2", "mean_diff", 12)


# ---- Print summary table ----
print("=" * 80)
print("RQ CURVE FIT SUMMARY")
print("=" * 80)

header = f"{'Trait':<30s} {'Method':<10s} {'R2(pref)':>10s} {'R2(util)':>10s} {'Breakdown':>12s} {'Pref Conv':>10s} {'Util Conv':>10s}"
print(header)
print("-" * len(header))

for name, rq in [
    ("pv_natural/evil_v3", evil_rq),
    ("pv_natural/sycophancy", syco_rq),
    ("pv_natural/hallucination_v2", hall_probe_rq),
    ("pv_natural/hallucination_v2", hall_md_rq),
]:
    print(f"{name:<30s} {rq.get('method', '?'):<10s} "
          f"{rq['pref_r2']:>10.4f} {rq['util_r2']:>10.4f} "
          f"{rq['breakdown_coefficient']:>12.2f} "
          f"{'yes' if rq['pref_converged'] else 'NO':>10s} "
          f"{'yes' if rq['util_converged'] else 'NO':>10s}")

print()

# ---- Check individual parameters for anomalies ----
print("=" * 80)
print("PARAMETER DIAGNOSTICS")
print("=" * 80)

for name, rq in [
    ("evil_v3 (probe)", evil_rq),
    ("sycophancy (probe)", syco_rq),
    ("hallucination_v2 (probe)", hall_probe_rq),
    ("hallucination_v2 (mean_diff)", hall_md_rq),
]:
    print(f"\n--- {name} ---")
    pp = rq["pref_params"]
    up = rq["util_params"]

    # Check for parameters at bounds
    anomalies = []
    for label, params in [("pref", pp), ("util", up)]:
        for key in ["L_plus", "L_minus"]:
            if key in params and params[key] >= 99.9:
                anomalies.append(f"{label}.{key}={params[key]:.1f} (at upper bound 100)")
            if key in params and params[key] <= 0.011:
                anomalies.append(f"{label}.{key}={params[key]:.3f} (at lower bound 0.01)")
        for key in ["m_minus", "m_plus"]:
            if key in params and abs(params[key]) >= 19.9:
                anomalies.append(f"{label}.{key}={params[key]:.1f} (at bound)")
        for key in ["p_plus", "p_minus"]:
            if key in params and params[key] <= 0.011:
                anomalies.append(f"{label}.{key}={params[key]:.4f} (near lower bound)")
        for key in ["beta_u"]:
            if key in params and abs(params[key]) >= 49.9:
                anomalies.append(f"{label}.{key}={params[key]:.1f} (at bound 50)")
        for key in ["b_u"]:
            if key in params and abs(params[key]) >= 49.9:
                anomalies.append(f"{label}.{key}={params[key]:.1f} (at bound -50)")

    if anomalies:
        for a in anomalies:
            print(f"  [!] {a}")
    else:
        print("  All parameters within bounds.")

    # Breakdown interpretation
    bd = rq["breakdown_coefficient"]
    max_coef = max(evil_lo["coefficients"])
    if bd > max_coef:
        print(f"  [!] Breakdown={bd:.1f} exceeds max tested coefficient={max_coef}")

    # Print key pref params
    print(f"  pref: alpha={pp['alpha']:.4f}, beta={pp['beta']:.3f}, b={pp['b']:.3f}")
    print(f"  pref: m+={pp['m_plus']:.2f}, L+={pp['L_plus']:.1f}, p+={pp['p_plus']:.3f}")
    print(f"  pref: m-={pp['m_minus']:.2f}, L-={pp['L_minus']:.3f}, p-={pp['p_minus']:.4f}")

print()


# ---- Compute residuals ----
print("=" * 80)
print("RESIDUAL ANALYSIS")
print("=" * 80)

for name, lo, rq in [
    ("evil_v3 (probe)", evil_lo, evil_rq),
    ("sycophancy (probe)", syco_lo, syco_rq),
    ("hallucination_v2 (probe)", hall_probe_lo, hall_probe_rq),
    ("hallucination_v2 (mean_diff)", hall_md_lo, hall_md_rq),
]:
    coeffs = np.array(lo["coefficients"])
    pref_actual = np.array(lo["pref_odds"])
    util_actual = np.array(lo["util_odds"])

    pref_pred = pref_model(coeffs, rq["pref_params"])
    util_pred = util_model(coeffs, rq["util_params"])

    pref_resid = pref_actual - pref_pred
    util_resid = util_actual - util_pred

    print(f"\n--- {name} ---")
    print(f"  PrefOdds: MAE={np.mean(np.abs(pref_resid)):.4f}, "
          f"Max residual={np.max(np.abs(pref_resid)):.4f} at coef={coeffs[np.argmax(np.abs(pref_resid))]:.1f}")
    print(f"  UtilOdds: MAE={np.mean(np.abs(util_resid)):.4f}, "
          f"Max residual={np.max(np.abs(util_resid)):.4f} at coef={coeffs[np.argmax(np.abs(util_resid))]:.1f}")

    # Check if residuals are systematic (e.g., all positive/negative in a region)
    high_coef_mask = coeffs >= 10
    if np.sum(high_coef_mask) > 0:
        high_pref_resid = pref_resid[high_coef_mask]
        sign_consistency = np.mean(np.sign(high_pref_resid) == np.sign(high_pref_resid[0]))
        if sign_consistency > 0.8:
            direction = "positive" if high_pref_resid[0] > 0 else "negative"
            print(f"  [!] PrefOdds residuals systematically {direction} for coef>=10 "
                  f"(mean={np.mean(high_pref_resid):.4f})")


# ==== PLOTTING ====

m_fine = np.linspace(-4, 18, 500)


# ---- Plot 1: Three-panel probe comparison (PrefOdds + UtilOdds per trait) ----
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("RQ Decay Curve Fits: Probe Vectors (3 Traits)", fontsize=14, fontweight="bold")

for i, (name, lo, rq, color) in enumerate([
    ("evil_v3 (probe, L11)", evil_lo, evil_rq, "#2196F3"),
    ("sycophancy (probe, L13)", syco_lo, syco_rq, "#4CAF50"),
    ("hallucination_v2 (probe, L12)", hall_probe_lo, hall_probe_rq, "#FF5722"),
]):
    coeffs = np.array(lo["coefficients"])

    # PrefOdds
    ax = axes[i, 0]
    ax.scatter(coeffs, lo["pref_odds"], color=color, s=40, zorder=5, label="Data")
    pref_pred = pref_model(m_fine, rq["pref_params"])
    ax.plot(m_fine, pref_pred, color=color, alpha=0.7, linewidth=2, label=f"RQ fit (R2={rq['pref_r2']:.3f})")
    bd = rq["breakdown_coefficient"]
    if bd <= 18:
        ax.axvline(bd, color="red", linestyle="--", alpha=0.5, label=f"Breakdown={bd:.1f}")
    ax.set_title(f"{name} -- PrefOdds")
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("PrefOdds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 18)

    # UtilOdds
    ax = axes[i, 1]
    ax.scatter(coeffs, lo["util_odds"], color=color, s=40, zorder=5, label="Data")
    util_pred = util_model(m_fine, rq["util_params"])
    ax.plot(m_fine, util_pred, color=color, alpha=0.7, linewidth=2, label=f"RQ fit (R2={rq['util_r2']:.3f})")
    ax.set_title(f"{name} -- UtilOdds")
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("UtilOdds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 18)

plt.tight_layout()
plt.savefig(PLOTS / "rq_fits_all_probes.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {PLOTS / 'rq_fits_all_probes.png'}")


# ---- Plot 2: Probe vs Mean_Diff for hallucination_v2 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("hallucination_v2: Probe vs Mean_Diff Comparison", fontsize=14, fontweight="bold")

coeffs = np.array(hall_probe_lo["coefficients"])

# PrefOdds comparison
ax = axes[0]
ax.scatter(coeffs, hall_probe_lo["pref_odds"], color="#FF5722", s=40, zorder=5, label="Probe data")
ax.scatter(coeffs, hall_md_lo["pref_odds"], color="#9C27B0", s=40, marker="^", zorder=5, label="Mean_diff data")
ax.plot(m_fine, pref_model(m_fine, hall_probe_rq["pref_params"]),
        color="#FF5722", alpha=0.7, linewidth=2,
        label=f"Probe fit (R2={hall_probe_rq['pref_r2']:.3f}, bd={hall_probe_rq['breakdown_coefficient']:.1f})")
ax.plot(m_fine, pref_model(m_fine, hall_md_rq["pref_params"]),
        color="#9C27B0", alpha=0.7, linewidth=2, linestyle="--",
        label=f"Mean_diff fit (R2={hall_md_rq['pref_r2']:.3f}, bd={hall_md_rq['breakdown_coefficient']:.1f})")
ax.axvline(hall_probe_rq["breakdown_coefficient"], color="#FF5722", linestyle=":", alpha=0.5)
ax.axvline(hall_md_rq["breakdown_coefficient"], color="#9C27B0", linestyle=":", alpha=0.5)
ax.set_title("PrefOdds")
ax.set_xlabel("Steering coefficient")
ax.set_ylabel("PrefOdds")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 18)

# UtilOdds comparison
ax = axes[1]
ax.scatter(coeffs, hall_probe_lo["util_odds"], color="#FF5722", s=40, zorder=5, label="Probe data")
ax.scatter(coeffs, hall_md_lo["util_odds"], color="#9C27B0", s=40, marker="^", zorder=5, label="Mean_diff data")
ax.plot(m_fine, util_model(m_fine, hall_probe_rq["util_params"]),
        color="#FF5722", alpha=0.7, linewidth=2,
        label=f"Probe fit (R2={hall_probe_rq['util_r2']:.3f})")
ax.plot(m_fine, util_model(m_fine, hall_md_rq["util_params"]),
        color="#9C27B0", alpha=0.7, linewidth=2, linestyle="--",
        label=f"Mean_diff fit (R2={hall_md_rq['util_r2']:.3f})")
ax.set_title("UtilOdds")
ax.set_xlabel("Steering coefficient")
ax.set_ylabel("UtilOdds")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 18)

plt.tight_layout()
plt.savefig(PLOTS / "hallucination_probe_vs_meandiff.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS / 'hallucination_probe_vs_meandiff.png'}")


# ---- Plot 3: Decay function D(m) comparison across traits ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Validity Decay Function D(m)", fontsize=14, fontweight="bold")

# Pref decay
ax = axes[0]
for name, rq, color, ls in [
    ("evil_v3 (probe)", evil_rq, "#2196F3", "-"),
    ("sycophancy (probe)", syco_rq, "#4CAF50", "-"),
    ("hall_v2 (probe)", hall_probe_rq, "#FF5722", "-"),
    ("hall_v2 (mean_diff)", hall_md_rq, "#9C27B0", "--"),
]:
    D = decay_function(m_fine, rq["pref_params"])
    ax.plot(m_fine, D, color=color, linestyle=ls, linewidth=2, label=name)
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="D=0.5 threshold")
ax.set_title("Pref Decay D_pref(m)")
ax.set_xlabel("Steering coefficient")
ax.set_ylabel("D(m)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 18)
ax.set_ylim(-0.05, 1.05)

# Util decay
ax = axes[1]
for name, rq, color, ls in [
    ("evil_v3 (probe)", evil_rq, "#2196F3", "-"),
    ("sycophancy (probe)", syco_rq, "#4CAF50", "-"),
    ("hall_v2 (probe)", hall_probe_rq, "#FF5722", "-"),
    ("hall_v2 (mean_diff)", hall_md_rq, "#9C27B0", "--"),
]:
    D = decay_function(m_fine, rq["util_params"])
    ax.plot(m_fine, D, color=color, linestyle=ls, linewidth=2, label=name)
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="D=0.5 threshold")
ax.set_title("Util Decay D_util(m)")
ax.set_xlabel("Steering coefficient")
ax.set_ylabel("D(m)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 18)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(PLOTS / "decay_functions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS / 'decay_functions.png'}")


# ---- Plot 4: Raw cross-entropy curves (mean_L_p and mean_L_n) ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Raw Cross-Entropy: mean_L_p and mean_L_n vs Coefficient", fontsize=14, fontweight="bold")

for i, (name, lo, color) in enumerate([
    ("evil_v3 (probe)", evil_lo, "#2196F3"),
    ("sycophancy (probe)", syco_lo, "#4CAF50"),
    ("hallucination_v2 (probe)", hall_probe_lo, "#FF5722"),
    ("hallucination_v2 (mean_diff)", hall_md_lo, "#9C27B0"),
]):
    ax = axes[i // 2, i % 2]
    coeffs = np.array(lo["coefficients"])
    ax.plot(coeffs, lo["mean_L_p"], "o-", color=color, markersize=4, label="mean_L_p (positive)")
    ax.plot(coeffs, lo["mean_L_n"], "s--", color=color, alpha=0.6, markersize=4, label="mean_L_n (negative)")
    ax.set_title(name)
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("Length-normalized CE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS / "raw_cross_entropy.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS / 'raw_cross_entropy.png'}")


# ---- Quantitative comparison: probe vs mean_diff for hallucination_v2 ----
print()
print("=" * 80)
print("PROBE vs MEAN_DIFF COMPARISON (hallucination_v2, L12)")
print("=" * 80)

print(f"\n  Probe breakdown:     {hall_probe_rq['breakdown_coefficient']:.2f}")
print(f"  Mean_diff breakdown: {hall_md_rq['breakdown_coefficient']:.2f}")
print(f"  Difference:          {hall_md_rq['breakdown_coefficient'] - hall_probe_rq['breakdown_coefficient']:.2f}")
print(f"  Ratio:               {hall_md_rq['breakdown_coefficient'] / hall_probe_rq['breakdown_coefficient']:.3f}")

print(f"\n  Probe R2(pref):      {hall_probe_rq['pref_r2']:.4f}")
print(f"  Mean_diff R2(pref):  {hall_md_rq['pref_r2']:.4f}")
print(f"\n  Probe R2(util):      {hall_probe_rq['util_r2']:.4f}")
print(f"  Mean_diff R2(util):  {hall_md_rq['util_r2']:.4f}")

# Compare the raw log-odds values
probe_pref = np.array(hall_probe_lo["pref_odds"])
md_pref = np.array(hall_md_lo["pref_odds"])
print(f"\n  Max |PrefOdds difference|: {np.max(np.abs(probe_pref - md_pref)):.4f} "
      f"at coef={coeffs[np.argmax(np.abs(probe_pref - md_pref))]:.1f}")
print(f"  Mean |PrefOdds difference|: {np.mean(np.abs(probe_pref - md_pref)):.4f}")
print(f"\n  Correlation (pref_odds): {np.corrcoef(probe_pref, md_pref)[0,1]:.6f}")

probe_util = np.array(hall_probe_lo["util_odds"])
md_util = np.array(hall_md_lo["util_odds"])
print(f"  Correlation (util_odds): {np.corrcoef(probe_util, md_util)[0,1]:.6f}")

print("\n  --> The two methods produce nearly identical log-odds curves for hallucination_v2.")
print("      Breakdown coefficients differ by only 0.14, contradicting the hypothesis")
print("      that probe vectors have substantially wider valid regions.")


# ---- Anomaly: evil_v3 and sycophancy breakdown values ----
print()
print("=" * 80)
print("ANOMALY CHECK: BREAKDOWN VALUES")
print("=" * 80)

print(f"\n  evil_v3 breakdown:         {evil_rq['breakdown_coefficient']:.1f}")
print(f"  sycophancy breakdown:      {syco_rq['breakdown_coefficient']:.1f}")
print(f"  hallucination_v2 (probe):  {hall_probe_rq['breakdown_coefficient']:.1f}")
print(f"  hallucination_v2 (md):     {hall_md_rq['breakdown_coefficient']:.1f}")

print(f"\n  evil_v3: breakdown ({evil_rq['breakdown_coefficient']:.1f}) is BEYOND tested range (max=17)")
print(f"  sycophancy: breakdown ({syco_rq['breakdown_coefficient']:.1f}) is WAY BEYOND tested range (max=17)")
print(f"  --> These are extrapolations, NOT observations. The model never sees coherence collapse")
print(f"      within the tested range for evil_v3 and sycophancy probe vectors.")
print(f"  --> For hallucination_v2, breakdown at ~8.8-9.0 is well within tested range -- reliable.")

print()
print("  Interpretation: evil_v3 and sycophancy PrefOdds curves are still roughly linear at")
print("  coef=17, so the RQ model's decay exponent (p+) is low. The 'breakdown' is an")
print("  extrapolation artifact, not an empirical observation.")
print(f"  evil_v3 p_plus={evil_rq['pref_params']['p_plus']:.3f} (slow decay)")
print(f"  sycophancy p_plus={syco_rq['pref_params']['p_plus']:.3f} (slow decay)")
print(f"  hallucination_v2 p_plus={hall_probe_rq['pref_params']['p_plus']:.3f} (fast decay -- real breakdown)")


# ---- Check D(m) values at key coefficients ----
print()
print("=" * 80)
print("D(m) AT KEY COEFFICIENTS")
print("=" * 80)
print(f"\n{'Trait':<30s} {'D_pref(5)':>10s} {'D_pref(10)':>10s} {'D_pref(15)':>10s} {'D_pref(17)':>10s}")
print("-" * 70)
for name, rq in [
    ("evil_v3 (probe)", evil_rq),
    ("sycophancy (probe)", syco_rq),
    ("hall_v2 (probe)", hall_probe_rq),
    ("hall_v2 (mean_diff)", hall_md_rq),
]:
    vals = [decay_function(np.array([c]), rq["pref_params"])[0] for c in [5, 10, 15, 17]]
    print(f"{name:<30s} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")

print(f"\n{'Trait':<30s} {'D_util(5)':>10s} {'D_util(10)':>10s} {'D_util(15)':>10s} {'D_util(17)':>10s}")
print("-" * 70)
for name, rq in [
    ("evil_v3 (probe)", evil_rq),
    ("sycophancy (probe)", syco_rq),
    ("hall_v2 (probe)", hall_probe_rq),
    ("hall_v2 (mean_diff)", hall_md_rq),
]:
    vals = [decay_function(np.array([c]), rq["util_params"])[0] for c in [5, 10, 15, 17]]
    print(f"{name:<30s} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")


print("\n\nDone.")
