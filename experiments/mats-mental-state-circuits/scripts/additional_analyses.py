"""
Additional analyses on thought branches projection data.

A) Condition C agency trajectory vs condition B (quintile comparison)
B) Accumulative vs instantaneous signal correlation with cue_p
C) Max vs mean aggregation for sentence-level projections
D) Agency <-> cue_p cross-correlation with lags
E) Hierarchical clustering on trait fingerprints

Input: Projection files, response files, trait_fingerprints.json
Output: Printed results + updated trait_fingerprints.json with cluster assignments
Usage: python experiments/mats-mental-state-circuits/scripts/additional_analyses.py
"""

import json
import os
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXP = Path("/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments/mats-mental-state-circuits")
PROJ_BASE = EXP / "inference/instruct/projections"
RESP_BASE = EXP / "inference/instruct/responses/thought_branches"
FINGERPRINT_PATH = EXP / "analysis/thought_branches/trait_fingerprints.json"

TRAIT_PATHS = {
    "agency": "mental_state/agency",
    "anxiety": "mental_state/anxiety",
    "concealment": "bs/concealment",
    "confidence": "mental_state/confidence",
    "confusion": "mental_state/confusion",
    "curiosity": "mental_state/curiosity",
    "deception": "alignment/deception",
    "eval_awareness": "rm_hack/eval_awareness",
    "guilt": "mental_state/guilt",
    "obedience": "mental_state/obedience",
    "rationalization": "mental_state/rationalization",
}

PROBLEM_IDS = [26, 37, 59, 62, 68, 119, 145, 212, 215, 277, 288, 295, 309,
               324, 339, 371, 408, 418, 700, 715, 768, 804, 819, 827, 877, 960, 972]


def load_projection(trait_path: str, condition: str, pid: int) -> list[float]:
    """Load per-token response projections for a trait/condition/problem."""
    path = PROJ_BASE / trait_path / "thought_branches" / condition / f"{pid}.json"
    with open(path) as f:
        data = json.load(f)
    return data["projections"][0]["response"]


def load_sentence_boundaries(condition: str, pid: int) -> list[dict]:
    """Load sentence boundaries from response file."""
    path = RESP_BASE / condition / f"{pid}.json"
    with open(path) as f:
        data = json.load(f)
    return data.get("sentence_boundaries", [])


def compute_quintile_means(projections: list[float], n_quintiles: int = 5) -> np.ndarray:
    """Split projection into n equal quintiles and compute mean of each."""
    arr = np.array(projections)
    n = len(arr)
    if n < n_quintiles:
        return np.full(n_quintiles, np.nan)
    quintile_size = n / n_quintiles
    means = []
    for i in range(n_quintiles):
        start = int(round(i * quintile_size))
        end = int(round((i + 1) * quintile_size))
        means.append(np.mean(arr[start:end]))
    return np.array(means)


def sentence_level_projections(projections: list[float], boundaries: list[dict],
                                agg: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Compute sentence-level aggregated projections and cue_p values.

    Returns (agg_projections, cue_p_values) aligned by sentence.
    """
    arr = np.array(projections)
    agg_vals = []
    cue_ps = []
    for sb in boundaries:
        start, end = sb["token_start"], sb["token_end"]
        if end > len(arr):
            end = len(arr)
        if start >= end:
            continue
        segment = arr[start:end]
        if agg == "mean":
            agg_vals.append(np.mean(segment))
        elif agg == "max":
            agg_vals.append(np.max(segment))
        cue_ps.append(sb["cue_p"])
    return np.array(agg_vals), np.array(cue_ps)


# ===========================================================================
# A) Condition C vs B agency trajectory (quintile comparison)
# ===========================================================================
def analysis_a():
    print("=" * 80)
    print("A) CONDITION C vs B TRAJECTORY (QUINTILE COMPARISON)")
    print("    Traits: agency, concealment, confidence")
    print("=" * 80)

    traits_to_check = ["agency", "concealment", "confidence"]
    n_quintiles = 5

    for trait_name in traits_to_check:
        trait_path = TRAIT_PATHS[trait_name]
        quintiles_b = []
        quintiles_c = []

        for pid in PROBLEM_IDS:
            try:
                proj_b = load_projection(trait_path, "mmlu_condition_b", pid)
                proj_c = load_projection(trait_path, "mmlu_condition_c", pid)
            except FileNotFoundError:
                continue

            quintiles_b.append(compute_quintile_means(proj_b, n_quintiles))
            quintiles_c.append(compute_quintile_means(proj_c, n_quintiles))

        if not quintiles_b:
            print(f"\n  {trait_name}: No data found")
            continue

        mean_b = np.nanmean(quintiles_b, axis=0)
        mean_c = np.nanmean(quintiles_c, axis=0)
        std_b = np.nanstd(quintiles_b, axis=0)
        std_c = np.nanstd(quintiles_c, axis=0)

        print(f"\n  {trait_name.upper()} (n={len(quintiles_b)} problems):")
        print(f"  {'Quintile':<12} {'Cond B (biased)':>18} {'Cond C (unbiased)':>18} {'Delta (B-C)':>14}")
        print(f"  {'-'*62}")
        for i in range(n_quintiles):
            label = f"Q{i+1} ({i*20}-{(i+1)*20}%)"
            delta = mean_b[i] - mean_c[i]
            print(f"  {label:<12} {mean_b[i]:>8.3f} +/-{std_b[i]:.3f}  {mean_c[i]:>8.3f} +/-{std_c[i]:.3f}  {delta:>+8.3f}")

        # Late-response drop: compare Q5 to Q3 within each condition
        drop_b = mean_b[4] - mean_b[2]
        drop_c = mean_c[4] - mean_c[2]
        print(f"\n  Late-response shift (Q5 - Q3):")
        print(f"    Condition B: {drop_b:+.3f}  |  Condition C: {drop_c:+.3f}")
        if abs(drop_b) > abs(drop_c) * 1.5:
            print(f"    -> Larger drop in B => {trait_name} shift is bias-specific, not generic CoT pattern")
        elif abs(drop_c) > abs(drop_b) * 1.5:
            print(f"    -> Larger drop in C => {trait_name} shift is a generic CoT pattern")
        else:
            print(f"    -> Similar magnitude => {trait_name} shift is common to both conditions")

    print()


# ===========================================================================
# B) Accumulative vs instantaneous signal
# ===========================================================================
def analysis_b():
    print("=" * 80)
    print("B) ACCUMULATIVE vs INSTANTANEOUS SIGNAL (correlation with cue_p)")
    print("=" * 80)

    results = {}
    for trait_name, trait_path in sorted(TRAIT_PATHS.items()):
        instant_corrs = []
        cumul_corrs = []

        for pid in PROBLEM_IDS:
            try:
                proj = load_projection(trait_path, "mmlu_condition_b", pid)
                boundaries = load_sentence_boundaries("mmlu_condition_b", pid)
            except FileNotFoundError:
                continue

            if len(boundaries) < 4:
                continue

            sent_means, cue_ps = sentence_level_projections(proj, boundaries, agg="mean")
            if len(sent_means) < 4:
                continue

            # Instantaneous correlation
            r_inst, _ = stats.spearmanr(sent_means, cue_ps)
            if np.isnan(r_inst):
                continue

            # Cumulative mean up to each sentence
            cumul_means = np.cumsum(sent_means) / np.arange(1, len(sent_means) + 1)
            r_cumul, _ = stats.spearmanr(cumul_means, cue_ps)
            if np.isnan(r_cumul):
                continue

            instant_corrs.append(r_inst)
            cumul_corrs.append(r_cumul)

        if not instant_corrs:
            continue

        mean_inst = np.mean(instant_corrs)
        mean_cumul = np.mean(cumul_corrs)
        results[trait_name] = (mean_inst, mean_cumul, len(instant_corrs))

    print(f"\n  {'Trait':<20} {'Instantaneous r':>16} {'Cumulative r':>14} {'Winner':>12} {'n':>5}")
    print(f"  {'-'*70}")
    for trait_name in sorted(results.keys()):
        inst, cumul, n = results[trait_name]
        winner = "cumul" if abs(cumul) > abs(inst) else "instant"
        print(f"  {trait_name:<20} {inst:>+10.3f}       {cumul:>+10.3f}     {winner:>8}  {n:>5}")

    print()
    print("  Interpretation: If cumulative > instantaneous, the trait signal accumulates")
    print("  over the response rather than being locally correlated with current cue_p.")
    print()


# ===========================================================================
# C) Max vs mean aggregation
# ===========================================================================
def analysis_c():
    print("=" * 80)
    print("C) MAX vs MEAN SENTENCE-LEVEL AGGREGATION (correlation with cue_p)")
    print("=" * 80)

    results = {}
    for trait_name, trait_path in sorted(TRAIT_PATHS.items()):
        mean_corrs = []
        max_corrs = []

        for pid in PROBLEM_IDS:
            try:
                proj = load_projection(trait_path, "mmlu_condition_b", pid)
                boundaries = load_sentence_boundaries("mmlu_condition_b", pid)
            except FileNotFoundError:
                continue

            if len(boundaries) < 4:
                continue

            sent_means, cue_ps = sentence_level_projections(proj, boundaries, agg="mean")
            sent_maxs, _ = sentence_level_projections(proj, boundaries, agg="max")

            if len(sent_means) < 4:
                continue

            r_mean, _ = stats.spearmanr(sent_means, cue_ps)
            r_max, _ = stats.spearmanr(sent_maxs, cue_ps)

            if np.isnan(r_mean) or np.isnan(r_max):
                continue

            mean_corrs.append(r_mean)
            max_corrs.append(r_max)

        if not mean_corrs:
            continue

        avg_mean = np.mean(mean_corrs)
        avg_max = np.mean(max_corrs)
        results[trait_name] = (avg_mean, avg_max, len(mean_corrs))

    print(f"\n  {'Trait':<20} {'Mean agg r':>12} {'Max agg r':>12} {'Winner':>10} {'Delta':>8} {'n':>5}")
    print(f"  {'-'*70}")
    for trait_name in sorted(results.keys()):
        avg_mean, avg_max, n = results[trait_name]
        winner = "max" if abs(avg_max) > abs(avg_mean) else "mean"
        delta = abs(avg_max) - abs(avg_mean)
        print(f"  {trait_name:<20} {avg_mean:>+8.3f}     {avg_max:>+8.3f}     {winner:>6}   {delta:>+.3f}  {n:>5}")

    print()
    print("  Interpretation: If max > mean, the peak activation within a sentence carries")
    print("  more signal than the average (sparse, bursty trait expression).")
    print()


# ===========================================================================
# D) Agency <-> cue_p cross-correlation with lags
# ===========================================================================
def analysis_d():
    print("=" * 80)
    print("D) AGENCY <-> CUE_P CROSS-CORRELATION (sentence-level, lags -5 to +5)")
    print("=" * 80)

    max_lag = 5
    lags = range(-max_lag, max_lag + 1)
    trait_path = TRAIT_PATHS["agency"]

    all_xcorrs = []

    for pid in PROBLEM_IDS:
        try:
            proj = load_projection(trait_path, "mmlu_condition_b", pid)
            boundaries = load_sentence_boundaries("mmlu_condition_b", pid)
        except FileNotFoundError:
            continue

        if len(boundaries) < 2 * max_lag + 1:
            continue

        sent_means, cue_ps = sentence_level_projections(proj, boundaries, agg="mean")
        n = len(sent_means)
        if n < 2 * max_lag + 1:
            continue

        # Standardize for cross-correlation
        agency_z = (sent_means - np.mean(sent_means))
        std_a = np.std(sent_means)
        if std_a == 0:
            continue
        agency_z /= std_a

        cue_z = (cue_ps - np.mean(cue_ps))
        std_c = np.std(cue_ps)
        if std_c == 0:
            continue
        cue_z /= std_c

        xcorr = []
        for lag in lags:
            # Positive lag: agency leads (agency[t] vs cue_p[t+lag])
            if lag >= 0:
                a_slice = agency_z[:n - lag] if lag > 0 else agency_z
                c_slice = cue_ps[lag:] if lag > 0 else cue_ps
            else:
                a_slice = agency_z[-lag:]
                c_slice = cue_ps[:n + lag]
            r, _ = stats.spearmanr(a_slice, c_slice)
            xcorr.append(r if not np.isnan(r) else 0.0)

        all_xcorrs.append(xcorr)

    if not all_xcorrs:
        print("  No data available.")
        return

    mean_xcorr = np.mean(all_xcorrs, axis=0)
    std_xcorr = np.std(all_xcorrs, axis=0)

    print(f"\n  n = {len(all_xcorrs)} problems")
    print(f"  Positive lag = agency LEADS cue_p (agency at t, cue_p at t+lag)")
    print(f"  Negative lag = agency LAGS cue_p (agency at t+|lag|, cue_p at t)")
    print()
    print(f"  {'Lag':>5} {'Mean r':>10} {'Std':>8}")
    print(f"  {'-'*25}")
    for i, lag in enumerate(lags):
        marker = " <-- peak" if abs(mean_xcorr[i]) == np.max(np.abs(mean_xcorr)) else ""
        print(f"  {lag:>+5d} {mean_xcorr[i]:>+8.3f}  {std_xcorr[i]:>6.3f}{marker}")

    peak_idx = np.argmax(np.abs(mean_xcorr))
    peak_lag = list(lags)[peak_idx]
    print()
    if peak_lag > 0:
        print(f"  Peak at lag={peak_lag}: Agency LEADS cue_p by ~{peak_lag} sentences")
    elif peak_lag < 0:
        print(f"  Peak at lag={peak_lag}: Agency LAGS cue_p by ~{abs(peak_lag)} sentences")
    else:
        print(f"  Peak at lag=0: Agency and cue_p are contemporaneous")
    print()


# ===========================================================================
# E) Hierarchical clustering on fingerprints
# ===========================================================================
def analysis_e():
    print("=" * 80)
    print("E) HIERARCHICAL CLUSTERING ON TRAIT FINGERPRINTS")
    print("=" * 80)

    with open(FINGERPRINT_PATH) as f:
        fp_data = json.load(f)

    fingerprints = fp_data["fingerprints"]
    trait_names = fp_data["trait_names"]

    # Build matrix: rows = problems, cols = traits (ordered by trait_names)
    pids = sorted(fingerprints.keys(), key=lambda x: int(x))
    matrix = []
    for pid in pids:
        row = [fingerprints[pid][t] for t in trait_names]
        matrix.append(row)
    matrix = np.array(matrix)  # (27, 11)

    print(f"\n  Matrix shape: {matrix.shape} (problems x traits)")

    # Ward's linkage
    dist_matrix = pdist(matrix, metric="euclidean")
    Z = linkage(dist_matrix, method="ward")

    # Cut at k=2
    labels = fcluster(Z, t=2, criterion="maxclust")

    # Cluster assignments
    cluster_assignments = {}
    for i, pid in enumerate(pids):
        cluster_assignments[pid] = int(labels[i])

    # Cluster centroids
    cluster_centroids = {}
    for c in [1, 2]:
        mask = labels == c
        centroid = np.mean(matrix[mask], axis=0)
        cluster_centroids[str(c)] = {t: round(float(centroid[j]), 3) for j, t in enumerate(trait_names)}

    # Report
    for c in [1, 2]:
        mask = labels == c
        members = [pids[i] for i in range(len(pids)) if labels[i] == c]
        print(f"\n  Cluster {c} ({len(members)} problems): {', '.join(members)}")
        print(f"  Centroid:")
        centroid = cluster_centroids[str(c)]
        for t in trait_names:
            print(f"    {t:<20} {centroid[t]:>+.3f}")

    # Compute discriminating traits
    print(f"\n  Most discriminating traits (|centroid1 - centroid2|):")
    diffs = []
    for t in trait_names:
        d = abs(cluster_centroids["1"][t] - cluster_centroids["2"][t])
        diffs.append((t, d, cluster_centroids["1"][t], cluster_centroids["2"][t]))
    diffs.sort(key=lambda x: -x[1])
    print(f"  {'Trait':<20} {'|Delta|':>8} {'C1':>8} {'C2':>8}")
    print(f"  {'-'*48}")
    for t, d, c1, c2 in diffs:
        print(f"  {t:<20} {d:>8.3f} {c1:>+8.3f} {c2:>+8.3f}")

    # Save back to JSON
    fp_data["cluster_assignments"] = cluster_assignments
    fp_data["cluster_centroids"] = cluster_centroids

    with open(FINGERPRINT_PATH, "w") as f:
        json.dump(fp_data, f, indent=2)
    print(f"\n  Saved cluster_assignments and cluster_centroids to {FINGERPRINT_PATH}")
    print()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    analysis_a()
    analysis_b()
    analysis_c()
    analysis_d()
    analysis_e()
    print("Done.")
