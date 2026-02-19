"""Parse audit-bleachers steering results and find best layer per trait.

Input: results.jsonl files from steering sweeps
Output: Cross-trait summary table + per-layer breakdown for deception
Usage: python analysis/parse_audit_bleachers_steering.py
"""

import json
from pathlib import Path

BASE = Path("/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments/audit-bleachers/steering")

# All trait paths (relative to BASE) -> display label
TRAIT_FILES = {
    # Standard steering results
    "alignment/deception/instruct/response__5/steering/results.jsonl": "alignment/deception",
    "alignment/strategic_omission/instruct/response__5/steering/results.jsonl": "alignment/strategic_omission",
    "alignment/performative_confidence/instruct/response__5/steering/results.jsonl": "alignment/performative_confidence",
    "alignment/self_serving/instruct/response__5/steering/results.jsonl": "alignment/self_serving",
    "alignment/gaming/instruct/response__5/steering/results.jsonl": "alignment/gaming",
    "alignment/conflicted/instruct/response__5/steering/results.jsonl": "alignment/conflicted",
    "alignment/compliance_without_agreement/instruct/response__5/steering/results.jsonl": "alignment/compliance_without_agreement",
    "alignment/helpfulness_expressed/instruct/response__5/steering/results.jsonl": "alignment/helpfulness_expressed",
    "alignment/honesty_observed/instruct/response__5/steering/results.jsonl": "alignment/honesty_observed",
    "psychology/people_pleasing/instruct/response__5/steering/results.jsonl": "psychology/people_pleasing",
    "psychology/authority_deference/instruct/response__5/steering/results.jsonl": "psychology/authority_deference",
    "psychology/intellectual_curiosity/instruct/response__5/steering/results.jsonl": "psychology/intellectual_curiosity",
    "rm_hack/secondary_objective/instruct/response__5/steering/results.jsonl": "rm_hack/secondary_objective",
    "rm_hack/ulterior_motive/instruct/response__5/steering/results.jsonl": "rm_hack/ulterior_motive",
    "hum/formality/instruct/response__5/steering/results.jsonl": "hum/formality",
    "hum/optimism/instruct/response__5/steering/results.jsonl": "hum/optimism",
    "hum/retrieval/instruct/response__5/steering/results.jsonl": "hum/retrieval",
    # rm_syco variants
    "rm_hack/ulterior_motive/instruct/response__32/rm_syco/train_100/results.jsonl": "rm_hack/ulterior_motive [rm_syco, resp32]",
    "rm_hack/ulterior_motive_v2/instruct/response__5/rm_syco/train_100/results.jsonl": "rm_hack/ulterior_motive_v2 [rm_syco]",
}


def parse_results_file(filepath):
    """Parse a results.jsonl file. Returns (baseline, rows) or None if missing."""
    if not filepath.exists():
        return None

    baseline = None
    rows = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            if entry.get("type") == "header":
                continue
            elif entry.get("type") == "baseline":
                baseline = entry["result"]
            else:
                # Steering result
                result = entry["result"]
                vec_cfg = entry["config"]["vectors"][0]
                rows.append({
                    "layer": vec_cfg["layer"],
                    "component": vec_cfg["component"],
                    "method": vec_cfg["method"],
                    "weight": vec_cfg["weight"],
                    "position": vec_cfg.get("position", "?"),
                    "trait_mean": result["trait_mean"],
                    "coherence_mean": result["coherence_mean"],
                    "n": result["n"],
                })

    return baseline, rows


def find_best(baseline, rows, coherence_threshold=70):
    """Find the best steering result: highest delta with coherence >= threshold."""
    baseline_trait = baseline["trait_mean"]

    best = None
    best_delta = -float("inf")

    for r in rows:
        delta = r["trait_mean"] - baseline_trait
        if r["coherence_mean"] >= coherence_threshold and delta > best_delta:
            best_delta = delta
            best = r

    return best, best_delta if best else None


def main():
    coherence_threshold = 70

    summaries = []
    deception_rows = None
    deception_baseline = None

    for rel_path, label in TRAIT_FILES.items():
        filepath = BASE / rel_path
        parsed = parse_results_file(filepath)

        if parsed is None:
            print(f"[MISSING] {label}: {filepath}")
            continue

        baseline, rows = parsed

        if label == "alignment/deception":
            deception_rows = rows
            deception_baseline = baseline

        best, best_delta = find_best(baseline, rows, coherence_threshold)

        summaries.append({
            "trait": label,
            "baseline_trait": baseline["trait_mean"],
            "baseline_coherence": baseline["coherence_mean"],
            "n_results": len(rows),
            "best": best,
            "best_delta": best_delta,
        })

    # ===== CROSS-TRAIT SUMMARY TABLE =====
    # Sort by delta descending (None deltas at bottom)
    summaries.sort(key=lambda s: s["best_delta"] if s["best_delta"] is not None else -999)
    summaries.reverse()

    print("=" * 140)
    print("CROSS-TRAIT STEERING SUMMARY (audit-bleachers)")
    print(f"Coherence threshold: >= {coherence_threshold}")
    print("=" * 140)
    print(f"{'Trait':<45} {'Baseline':>8} {'BL Coh':>7} {'Best Layer':>10} {'Method':>8} {'Weight':>7} "
          f"{'Steered':>8} {'Coh':>7} {'Delta':>8} {'Component':>12}")
    print("-" * 140)

    for s in summaries:
        b = s["best"]
        if b is not None:
            print(f"{s['trait']:<45} {s['baseline_trait']:>8.2f} {s['baseline_coherence']:>7.2f} "
                  f"{b['layer']:>10d} {b['method']:>8} {b['weight']:>7.2f} "
                  f"{b['trait_mean']:>8.2f} {b['coherence_mean']:>7.2f} {s['best_delta']:>8.2f} "
                  f"{b['component']:>12}")
        else:
            print(f"{s['trait']:<45} {s['baseline_trait']:>8.2f} {s['baseline_coherence']:>7.2f} "
                  f"{'---':>10} {'---':>8} {'---':>7} {'---':>8} {'---':>7} {'---':>8} {'---':>12}")

    print("=" * 140)

    # Stats
    with_delta = [s for s in summaries if s["best_delta"] is not None]
    if with_delta:
        deltas = [s["best_delta"] for s in with_delta]
        print(f"\nTraits with valid best: {len(with_delta)}/{len(summaries)}")
        print(f"Delta range: {min(deltas):.2f} to {max(deltas):.2f}")
        print(f"Mean delta: {sum(deltas)/len(deltas):.2f}")
        print(f"Median delta: {sorted(deltas)[len(deltas)//2]:.2f}")

    # ===== PER-LAYER BREAKDOWN FOR DECEPTION =====
    if deception_rows and deception_baseline:
        baseline_trait = deception_baseline["trait_mean"]
        baseline_coh = deception_baseline["coherence_mean"]

        print("\n\n" + "=" * 140)
        print("PER-LAYER BREAKDOWN: alignment/deception")
        print(f"Baseline: trait_mean={baseline_trait:.2f}, coherence_mean={baseline_coh:.2f}")
        print("=" * 140)

        # Group by layer
        by_layer = {}
        for r in deception_rows:
            key = r["layer"]
            if key not in by_layer:
                by_layer[key] = []
            by_layer[key].append(r)

        print(f"\n{'Layer':>5} {'Component':>12} {'Method':>8} {'Weight':>7} {'Trait':>8} {'Coh':>7} "
              f"{'Delta':>8} {'Coh>=70':>8}")
        print("-" * 80)

        for layer in sorted(by_layer.keys()):
            rows_for_layer = by_layer[layer]
            # Sort by delta descending within layer
            rows_for_layer.sort(key=lambda r: r["trait_mean"] - baseline_trait, reverse=True)

            for i, r in enumerate(rows_for_layer):
                delta = r["trait_mean"] - baseline_trait
                meets_coh = "YES" if r["coherence_mean"] >= coherence_threshold else "no"
                marker = " <-- BEST" if (r["coherence_mean"] >= coherence_threshold and
                    abs(delta - (find_best(deception_baseline, deception_rows, coherence_threshold)[1] or -999)) < 0.001) else ""
                print(f"{layer:>5} {r['component']:>12} {r['method']:>8} {r['weight']:>7.2f} "
                      f"{r['trait_mean']:>8.2f} {r['coherence_mean']:>7.2f} {delta:>+8.2f} {meets_coh:>8}{marker}")

            if layer != max(by_layer.keys()):
                print()  # blank line between layers

        print("=" * 140)


if __name__ == "__main__":
    main()
