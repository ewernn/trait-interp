#!/usr/bin/env python3
"""
Temporal circuit analysis: cross-correlation between trait projections.

Finds directed temporal ordering between traits — e.g., "anxiety leads
rationalization by N tokens" — by computing pairwise cross-correlations
with lag across per-token cosine projections.

Also includes early bias detection (claim 4): are condition B probe
projections at early sentences already shifted vs condition C?

Input: projection JSONs from project_raw_activations_onto_traits.py
Output: analysis/thought_branches/ JSON results + plots

Usage:
    python experiments/mats-mental-state-circuits/scripts/analyze_temporal_circuits.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import signal, stats

# Setup paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

EXPERIMENT = "mats-mental-state-circuits"
PROJ_BASE = ROOT / "experiments" / EXPERIMENT / "inference/instruct/projections"
RESP_BASE = ROOT / "experiments" / EXPERIMENT / "inference/instruct/responses"
OUTPUT_DIR = ROOT / "experiments" / EXPERIMENT / "analysis/thought_branches"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = [
    "alignment/deception", "bs/concealment", "mental_state/agency",
    "mental_state/anxiety", "mental_state/confidence", "mental_state/confusion",
    "mental_state/curiosity", "mental_state/guilt", "mental_state/obedience",
    "mental_state/rationalization", "rm_hack/eval_awareness",
]

TRAIT_SHORT = {
    "alignment/deception": "deception",
    "bs/concealment": "concealment",
    "mental_state/agency": "agency",
    "mental_state/anxiety": "anxiety",
    "mental_state/confidence": "confidence",
    "mental_state/confusion": "confusion",
    "mental_state/curiosity": "curiosity",
    "mental_state/guilt": "guilt",
    "mental_state/obedience": "obedience",
    "mental_state/rationalization": "rationalization",
    "rm_hack/eval_awareness": "eval_awareness",
}

# All 27 problem IDs
PROBLEMS = [26, 37, 59, 62, 68, 119, 145, 212, 215, 277, 288, 295, 309, 324,
            339, 371, 408, 418, 700, 715, 768, 804, 819, 827, 877, 960, 972]


def load_cosine_response(trait: str, condition: str, problem_id: int) -> np.ndarray | None:
    """Load response-phase cosine projections for a trait/condition/problem."""
    path = PROJ_BASE / trait / f"thought_branches/mmlu_condition_{condition}" / f"{problem_id}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    p = d["projections"][0]
    proj = np.array(p["response"])
    norms = np.array(p["token_norms"]["response"])
    return proj / np.where(norms > 0, norms, 1)


def load_sentence_boundaries(condition: str, problem_id: int) -> list | None:
    """Load sentence boundaries from response JSON."""
    path = RESP_BASE / f"thought_branches/mmlu_condition_{condition}" / f"{problem_id}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d.get("sentence_boundaries")


def sentence_means(cosine: np.ndarray, boundaries: list) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean cosine per sentence and return with cue_p values."""
    means, cue_ps = [], []
    for s in boundaries:
        chunk = cosine[s["token_start"]:s["token_end"]]
        if len(chunk) > 0:
            means.append(chunk.mean())
            cue_ps.append(s.get("cue_p", 0))
    return np.array(means), np.array(cue_ps)


# ============================================================================
# Analysis 1: Cross-correlation with lag
# ============================================================================

def compute_cross_correlations(max_lag: int = 50):
    """
    For each pair of traits, compute cross-correlation across tokens.

    Positive peak lag means trait_a LEADS trait_b (a's signal appears first).
    Averages across all 27 problems.
    """
    print("=" * 60)
    print("ANALYSIS 1: Cross-correlation with lag (temporal circuits)")
    print("=" * 60)

    pair_lags = defaultdict(list)      # (trait_a, trait_b) -> [peak_lags]
    pair_peak_rs = defaultdict(list)   # (trait_a, trait_b) -> [peak_r_values]

    for pid in PROBLEMS:
        # Load all traits for this problem (condition B — has cue_p ground truth)
        trait_data = {}
        for trait in TRAITS:
            cosine = load_cosine_response(trait, "b", pid)
            if cosine is not None:
                trait_data[TRAIT_SHORT[trait]] = cosine

        if len(trait_data) < 2:
            continue

        # Normalize each trait to zero mean for clean cross-correlation
        for name in trait_data:
            trait_data[name] = trait_data[name] - trait_data[name].mean()

        names = sorted(trait_data.keys())
        for i, name_a in enumerate(names):
            for j, name_b in enumerate(names):
                if i >= j:
                    continue
                a = trait_data[name_a]
                b = trait_data[name_b]
                # Trim to same length
                n = min(len(a), len(b))
                a, b = a[:n], b[:n]

                # Cross-correlation: positive lag means a leads b
                corr = signal.correlate(b, a, mode="full")
                # Normalize to correlation coefficient
                corr = corr / (np.std(a) * np.std(b) * n)
                lags = signal.correlation_lags(len(b), len(a), mode="full")

                # Find peak within ±max_lag
                mask = np.abs(lags) <= max_lag
                masked_corr = corr[mask]
                masked_lags = lags[mask]

                peak_idx = np.argmax(np.abs(masked_corr))
                peak_lag = int(masked_lags[peak_idx])
                peak_r = float(masked_corr[peak_idx])

                pair_lags[(name_a, name_b)].append(peak_lag)
                pair_peak_rs[(name_a, name_b)].append(peak_r)

    # Summarize
    results = []
    print(f"\n{'Trait A':>18s} → {'Trait B':<18s}  mean_lag  median_lag  mean|r|  consistency")
    print("-" * 85)

    for (a, b), lags in sorted(pair_lags.items()):
        rs = pair_peak_rs[(a, b)]
        mean_lag = np.mean(lags)
        median_lag = np.median(lags)
        mean_abs_r = np.mean(np.abs(rs))
        # Consistency: fraction of problems where lag has same sign
        if median_lag > 0:
            consistency = np.mean(np.array(lags) > 0)
        elif median_lag < 0:
            consistency = np.mean(np.array(lags) < 0)
        else:
            consistency = np.mean(np.array(lags) == 0)

        # Direction label
        if mean_lag > 2:
            direction = f"{a} leads"
        elif mean_lag < -2:
            direction = f"{b} leads"
        else:
            direction = "synchronous"

        result = {
            "trait_a": a, "trait_b": b,
            "mean_lag": round(float(mean_lag), 1),
            "median_lag": float(median_lag),
            "mean_abs_r": round(float(mean_abs_r), 3),
            "consistency": round(float(consistency), 2),
            "direction": direction,
            "n_problems": len(lags),
            "all_lags": [int(l) for l in lags],
        }
        results.append(result)

        if mean_abs_r > 0.15:  # Only print notable pairs
            print(f"{a:>18s} → {b:<18s}  {mean_lag:+7.1f}  {median_lag:+9.0f}    {mean_abs_r:.3f}    {consistency:.0%}  ({direction})")

    # Save
    output_path = OUTPUT_DIR / "temporal_cross_correlations.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output_path}")

    # Print strongest directed pairs (|mean_lag| > 3 and consistency > 60%)
    print("\n--- Strongest temporal orderings ---")
    directed = [r for r in results if abs(r["mean_lag"]) > 3 and r["consistency"] > 0.6 and r["mean_abs_r"] > 0.1]
    directed.sort(key=lambda r: abs(r["mean_lag"]), reverse=True)
    for r in directed[:15]:
        leader = r["trait_a"] if r["mean_lag"] > 0 else r["trait_b"]
        follower = r["trait_b"] if r["mean_lag"] > 0 else r["trait_a"]
        print(f"  {leader} → {follower}  (lag={abs(r['mean_lag']):.0f} tokens, r={r['mean_abs_r']:.3f}, {r['consistency']:.0%} consistent)")

    return results


# ============================================================================
# Analysis 2: Early bias detection (Claim 4)
# ============================================================================

def compute_early_bias():
    """
    Compare condition B vs C probe projections at early sentences.

    If B (unfaithful CoT, no hint) already shows shifted probes at sentences
    1-3 compared to C (faithful CoT, no hint), bias is detectable before
    behavioral divergence in the text.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Early bias detection (B vs C, early sentences)")
    print("=" * 60)

    # For each trait, collect sentence-level means for B and C at early vs late positions
    trait_results = {}

    for trait in TRAITS:
        short = TRAIT_SHORT[trait]
        early_b, early_c = [], []  # sentences 0-2
        late_b, late_c = [], []    # last 3 sentences
        all_b, all_c = [], []

        for pid in PROBLEMS:
            cos_b = load_cosine_response(trait, "b", pid)
            cos_c = load_cosine_response(trait, "c", pid)
            sb_b = load_sentence_boundaries("b", pid)

            if cos_b is None or cos_c is None or sb_b is None:
                continue

            # Condition C doesn't have sentence_boundaries (different text)
            # But we can still compare overall response means
            all_b.append(cos_b.mean())
            all_c.append(cos_c.mean())

            # Early/late from condition B sentence boundaries
            n_sent = len(sb_b)
            for s in sb_b:
                chunk_b = cos_b[s["token_start"]:s["token_end"]]
                if len(chunk_b) == 0:
                    continue
                mean_b = chunk_b.mean()

                # For C, use same token positions (different text but same-ish length)
                # This is approximate — C has different text so token alignment isn't perfect
                chunk_c = cos_c[s["token_start"]:min(s["token_end"], len(cos_c))]
                if len(chunk_c) == 0:
                    continue
                mean_c = chunk_c.mean()

                if s["sentence_num"] <= 2:
                    early_b.append(mean_b)
                    early_c.append(mean_c)
                elif s["sentence_num"] >= n_sent - 3:
                    late_b.append(mean_b)
                    late_c.append(mean_c)

        # Stats
        result = {"trait": short}

        # Overall B vs C
        if len(all_b) >= 5 and len(all_c) >= 5:
            t_all, p_all = stats.ttest_ind(all_b, all_c)
            d_all = (np.mean(all_b) - np.mean(all_c)) / np.sqrt((np.var(all_b) + np.var(all_c)) / 2)
            result["overall"] = {
                "mean_b": round(float(np.mean(all_b)), 5),
                "mean_c": round(float(np.mean(all_c)), 5),
                "cohens_d": round(float(d_all), 3),
                "p_value": round(float(p_all), 4),
                "n": len(all_b),
            }

        # Early sentences B vs C
        if len(early_b) >= 10 and len(early_c) >= 10:
            t_early, p_early = stats.ttest_ind(early_b, early_c)
            d_early = (np.mean(early_b) - np.mean(early_c)) / np.sqrt((np.var(early_b) + np.var(early_c)) / 2)
            result["early_sentences"] = {
                "mean_b": round(float(np.mean(early_b)), 5),
                "mean_c": round(float(np.mean(early_c)), 5),
                "cohens_d": round(float(d_early), 3),
                "p_value": round(float(p_early), 4),
                "n_b": len(early_b), "n_c": len(early_c),
            }

        # Late sentences B vs C
        if len(late_b) >= 10 and len(late_c) >= 10:
            t_late, p_late = stats.ttest_ind(late_b, late_c)
            d_late = (np.mean(late_b) - np.mean(late_c)) / np.sqrt((np.var(late_b) + np.var(late_c)) / 2)
            result["late_sentences"] = {
                "mean_b": round(float(np.mean(late_b)), 5),
                "mean_c": round(float(np.mean(late_c)), 5),
                "cohens_d": round(float(d_late), 3),
                "p_value": round(float(p_late), 4),
                "n_b": len(late_b), "n_c": len(late_c),
            }

        trait_results[short] = result

    # Print
    print(f"\n{'Trait':>18s}  {'Early d':>9s}  {'Early p':>9s}  {'Late d':>9s}  {'Late p':>9s}  {'Overall d':>10s}  {'Overall p':>10s}")
    print("-" * 90)
    for short in sorted(trait_results.keys()):
        r = trait_results[short]
        early = r.get("early_sentences", {})
        late = r.get("late_sentences", {})
        overall = r.get("overall", {})
        print(f"{short:>18s}  {early.get('cohens_d', 'N/A'):>9}  {early.get('p_value', 'N/A'):>9}  "
              f"{late.get('cohens_d', 'N/A'):>9}  {late.get('p_value', 'N/A'):>9}  "
              f"{overall.get('cohens_d', 'N/A'):>10}  {overall.get('p_value', 'N/A'):>10}")

    # Save
    output_path = OUTPUT_DIR / "early_bias_detection.json"
    output_path.write_text(json.dumps(trait_results, indent=2))
    print(f"\nSaved: {output_path}")

    return trait_results


# ============================================================================
# Analysis 3: Per-problem trait fingerprints
# ============================================================================

def compute_trait_fingerprints():
    """
    For each problem, compute the trait correlation profile with cue_p.
    Cluster problems to see if cue_p trajectory patterns (flat-then-jump
    vs bouncy/gradual) map onto distinct trait signatures.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Per-problem trait fingerprints")
    print("=" * 60)

    fingerprints = {}  # pid -> {trait: r_with_cue_p}
    cue_p_profiles = {}  # pid -> cue_p trajectory summary

    for pid in PROBLEMS:
        sb = load_sentence_boundaries("b", pid)
        if not sb:
            continue

        cue_ps = [s["cue_p"] for s in sb if s.get("cue_p") is not None]
        if len(cue_ps) < 5:
            continue

        # Characterize cue_p trajectory
        cue_arr = np.array(cue_ps)
        diffs = np.diff(cue_arr)
        max_jump = float(np.max(np.abs(diffs)))
        monotonicity = float(np.mean(diffs > 0))  # fraction of increases
        final_cue_p = float(cue_arr[-1])
        early_mean = float(cue_arr[:3].mean())
        late_mean = float(cue_arr[-3:].mean())

        cue_p_profiles[str(pid)] = {
            "max_jump": round(max_jump, 3),
            "monotonicity": round(monotonicity, 3),
            "final_cue_p": round(final_cue_p, 3),
            "early_mean": round(early_mean, 3),
            "late_mean": round(late_mean, 3),
            "n_sentences": len(cue_ps),
        }

        # Compute per-trait correlation with cue_p
        fp = {}
        for trait in TRAITS:
            short = TRAIT_SHORT[trait]
            cosine = load_cosine_response(trait, "b", pid)
            if cosine is None or sb is None:
                continue
            means, sent_cue_ps = sentence_means(cosine, sb)
            if len(means) >= 5:
                r, p = stats.pearsonr(means, sent_cue_ps)
                fp[short] = round(float(r), 3)
        fingerprints[str(pid)] = fp

    # Compute fingerprint similarity matrix
    pids = sorted(fingerprints.keys(), key=int)
    trait_names = sorted(TRAIT_SHORT.values())
    n = len(pids)

    # Build matrix: problems × traits
    matrix = np.zeros((n, len(trait_names)))
    for i, pid in enumerate(pids):
        for j, trait in enumerate(trait_names):
            matrix[i, j] = fingerprints[pid].get(trait, 0)

    # Compute pairwise problem similarity (correlation of fingerprints)
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster

    if n >= 4:
        dists = pdist(matrix, metric="correlation")
        Z = linkage(dists, method="ward")
        clusters = fcluster(Z, t=2, criterion="maxclust")

        # Print cluster assignments with cue_p profile info
        print(f"\nTwo-cluster solution:")
        for c in [1, 2]:
            members = [pids[i] for i in range(n) if clusters[i] == c]
            print(f"\n  Cluster {c} ({len(members)} problems): {', '.join(members)}")
            # Mean trait fingerprint
            cluster_matrix = matrix[clusters == c]
            mean_fp = cluster_matrix.mean(axis=0)
            top_traits = sorted(zip(trait_names, mean_fp), key=lambda x: abs(x[1]), reverse=True)
            print(f"    Top traits: {', '.join(f'{t}={r:+.2f}' for t, r in top_traits[:5])}")
            # Mean cue_p profile
            mean_jump = np.mean([cue_p_profiles[p]["max_jump"] for p in members if p in cue_p_profiles])
            mean_mono = np.mean([cue_p_profiles[p]["monotonicity"] for p in members if p in cue_p_profiles])
            print(f"    cue_p profile: mean max_jump={mean_jump:.2f}, mean monotonicity={mean_mono:.2f}")

    # Save
    output = {
        "fingerprints": fingerprints,
        "cue_p_profiles": cue_p_profiles,
        "trait_names": trait_names,
    }
    output_path = OUTPUT_DIR / "trait_fingerprints.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {output_path}")

    return output


if __name__ == "__main__":
    xcorr_results = compute_cross_correlations()
    early_bias_results = compute_early_bias()
    fingerprint_results = compute_trait_fingerprints()

    print("\n" + "=" * 60)
    print("DONE — all results in analysis/thought_branches/")
    print("=" * 60)
