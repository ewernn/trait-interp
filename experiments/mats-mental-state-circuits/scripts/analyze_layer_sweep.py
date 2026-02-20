#!/usr/bin/env python3
"""
Layer sweep analysis: find optimal reading layer per trait using cue_p ground truth.

For each trait × layer, computes mean-per-sentence cosine projection correlated
with cue_p across 27 problems. The layer maximizing mean |r| is the empirically
best reading layer — replacing the best_steering+5 heuristic.

Also computes the unsupervised F-statistic (inter-sentence / intra-sentence variance)
as a ground-truth-free alternative.

Input: multi-layer projection JSONs from project_raw_activations_onto_traits.py
Output: analysis/thought_branches/layer_sweep.json + printed summary

Usage:
    python experiments/mats-mental-state-circuits/scripts/analyze_layer_sweep.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

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

TRAIT_SHORT = {t: t.split("/")[-1] for t in TRAITS}

PROBLEMS = [26, 37, 59, 62, 68, 119, 145, 212, 215, 277, 288, 295, 309, 324,
            339, 371, 408, 418, 700, 715, 768, 804, 819, 827, 877, 960, 972]


def load_projection_layers(trait: str, condition: str, problem_id: int) -> dict:
    """Load all available layers from a multi-layer projection file.

    Returns: {layer: {"cosine": np.array, "proj": np.array, "norms": np.array}} or {}
    """
    path = PROJ_BASE / trait / f"thought_branches/mmlu_condition_{condition}" / f"{problem_id}.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    projections = d.get("projections", [])
    if not isinstance(projections, list):
        return {}

    result = {}
    for p in projections:
        layer = p.get("layer")
        if layer is None:
            continue
        resp_proj = np.array(p["response"])
        resp_norms = np.array(p["token_norms"]["response"])
        cosine = resp_proj / np.where(resp_norms > 0, resp_norms, 1)
        result[layer] = {"cosine": cosine, "proj": resp_proj, "norms": resp_norms}
    return result


def load_sentence_boundaries(condition: str, problem_id: int) -> list | None:
    path = RESP_BASE / f"thought_branches/mmlu_condition_{condition}" / f"{problem_id}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d.get("sentence_boundaries")


def sentence_aggregate(values: np.ndarray, boundaries: list, method: str = "mean"):
    """Aggregate per-token values to per-sentence."""
    means, cue_ps = [], []
    for s in boundaries:
        chunk = values[s["token_start"]:s["token_end"]]
        if len(chunk) == 0:
            continue
        if method == "mean":
            means.append(chunk.mean())
        elif method == "last":
            means.append(chunk[-1])
        elif method == "max":
            means.append(chunk.max())
        cue_ps.append(s.get("cue_p", 0))
    return np.array(means), np.array(cue_ps)


def compute_f_statistic(values: np.ndarray, boundaries: list) -> float:
    """Inter-sentence / intra-sentence variance ratio."""
    groups = []
    for s in boundaries:
        chunk = values[s["token_start"]:s["token_end"]]
        if len(chunk) > 1:
            groups.append(chunk)
    if len(groups) < 3:
        return 0.0
    try:
        f_stat, _ = stats.f_oneway(*groups)
        return float(f_stat) if np.isfinite(f_stat) else 0.0
    except Exception:
        return 0.0


def run_layer_sweep():
    print("=" * 70)
    print("LAYER SWEEP: finding optimal reading layer per trait")
    print("=" * 70)

    # Discover available layers from first file
    sample = load_projection_layers(TRAITS[0], "b", PROBLEMS[0])
    if not sample:
        print("ERROR: No multi-layer projections found. Run project_raw_activations_onto_traits.py with --layers first.")
        return
    available_layers = sorted(sample.keys())
    print(f"Available layers: {available_layers} ({len(available_layers)} layers)")
    print(f"Traits: {len(TRAITS)}, Problems: {len(PROBLEMS)}")
    print()

    results = {}

    for trait in TRAITS:
        short = TRAIT_SHORT[trait]
        # For each layer: collect per-problem correlations with cue_p
        layer_data = defaultdict(lambda: {"r_values": [], "f_values": []})

        for pid in PROBLEMS:
            layers = load_projection_layers(trait, "b", pid)
            sb = load_sentence_boundaries("b", pid)
            if not layers or not sb:
                continue

            for layer, data in layers.items():
                cosine = data["cosine"]

                # Supervised: correlation with cue_p
                sent_means, cue_ps = sentence_aggregate(cosine, sb, method="mean")
                if len(sent_means) >= 5:
                    r, p = stats.pearsonr(sent_means, cue_ps)
                    layer_data[layer]["r_values"].append(float(r))

                # Unsupervised: F-statistic
                f_stat = compute_f_statistic(cosine, sb)
                layer_data[layer]["f_values"].append(f_stat)

        # Summarize per layer
        layer_summary = {}
        for layer in available_layers:
            ld = layer_data.get(layer, {"r_values": [], "f_values": []})
            rs = ld["r_values"]
            fs = ld["f_values"]
            layer_summary[layer] = {
                "mean_r": round(float(np.mean(rs)), 4) if rs else 0,
                "mean_abs_r": round(float(np.mean(np.abs(rs))), 4) if rs else 0,
                "median_r": round(float(np.median(rs)), 4) if rs else 0,
                "std_r": round(float(np.std(rs)), 4) if rs else 0,
                "mean_f": round(float(np.mean(fs)), 2) if fs else 0,
                "n_problems": len(rs),
            }

        # Find best layers
        best_supervised = max(available_layers, key=lambda l: layer_summary[l]["mean_abs_r"])
        best_unsupervised = max(available_layers, key=lambda l: layer_summary[l]["mean_f"])

        # Current layer (best+5 heuristic)
        current_layer = None
        sample_proj = load_projection_layers(trait, "b", PROBLEMS[0])
        if sample_proj:
            # The existing single-layer projections used best+5
            # We can infer from what layers are present
            current_layer = min(sample_proj.keys())  # fallback

        results[short] = {
            "best_supervised_layer": best_supervised,
            "best_supervised_mean_abs_r": layer_summary[best_supervised]["mean_abs_r"],
            "best_supervised_mean_r": layer_summary[best_supervised]["mean_r"],
            "best_unsupervised_layer": best_unsupervised,
            "best_unsupervised_mean_f": layer_summary[best_unsupervised]["mean_f"],
            "layer_details": {str(k): v for k, v in sorted(layer_summary.items())},
        }

    # Print comparison table
    print(f"{'Trait':>18s}  {'Best(cue_p)':>11s}  {'mean|r|':>8s}  {'Best(F)':>8s}  {'mean F':>8s}  {'Agree?':>7s}")
    print("-" * 75)
    for short in sorted(results.keys()):
        r = results[short]
        agree = "✓" if r["best_supervised_layer"] == r["best_unsupervised_layer"] else ""
        print(f"{short:>18s}  L{r['best_supervised_layer']:>2d}          {r['best_supervised_mean_abs_r']:.4f}    "
              f"L{r['best_unsupervised_layer']:>2d}      {r['best_unsupervised_mean_f']:>6.1f}    {agree}")

    # Print layer curves for each trait (condensed)
    print(f"\n--- Layer sensitivity curves (mean |r| with cue_p) ---")
    for short in sorted(results.keys()):
        r = results[short]
        details = r["layer_details"]
        layers = sorted(details.keys(), key=int)
        curve = " ".join(f"{details[l]['mean_abs_r']:.2f}" for l in layers)
        best_l = r["best_supervised_layer"]
        print(f"{short:>18s}: [{curve}]  best=L{best_l}")

    # Save
    output_path = OUTPUT_DIR / "layer_sweep.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output_path}")

    return results


if __name__ == "__main__":
    run_layer_sweep()
