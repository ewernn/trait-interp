"""Analyze steering deltas across all experiments. Run from repo root.

Usage:
    python scripts/analyze_steering_deltas.py
"""
import json
import subprocess
from pathlib import Path

def main():
    # Find all steering results files
    result = subprocess.run(
        ["find", "experiments", "-name", "results.jsonl", "-path", "*/steering/*"],
        capture_output=True, text=True
    )
    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    print(f"Found {len(files)} steering results files")

    results = []
    for fp in files:
        try:
            p = Path(fp)
            with open(p) as fh:
                lines = [l.strip() for i, l in enumerate(fh) if i < 400]

            header = json.loads(lines[0])
            if header.get("type") != "header":
                continue
            trait = header.get("trait", "")
            direction = header.get("direction", "positive")
            sign = 1 if direction == "positive" else -1
            exp = str(p.parts[1])

            baseline_mean = 0
            best_delta = None
            best_layer = None
            best_coh = None

            for raw in lines[1:]:
                if not raw:
                    continue
                entry = json.loads(raw)
                if entry.get("type") == "baseline":
                    baseline_mean = entry.get("result", {}).get("trait_mean", 0) or 0
                    continue
                r = entry.get("result", {})
                tm = r.get("trait_mean")
                cm = r.get("coherence_mean", 0)
                if tm is None:
                    continue
                delta = tm - baseline_mean
                if best_delta is None or delta * sign > best_delta * sign:
                    best_delta = delta
                    best_coh = cm
                    cfg = entry.get("config", {})
                    vecs = cfg.get("vectors", [])
                    best_layer = vecs[0].get("layer") if vecs else None

            if best_delta is not None:
                results.append({
                    "abs_delta": abs(best_delta),
                    "delta": best_delta,
                    "trait": trait,
                    "experiment": exp,
                    "layer": best_layer,
                    "coherence": best_coh,
                    "baseline": baseline_mean,
                    "direction": direction,
                })
        except Exception as e:
            pass

    results.sort(key=lambda r: r["abs_delta"], reverse=True)

    print(f"\nTotal: {len(results)} trait/experiment combos")
    print(f"|delta| > 30: {sum(1 for r in results if r['abs_delta'] > 30)}")
    print(f"|delta| > 20: {sum(1 for r in results if r['abs_delta'] > 20)}")
    print(f"|delta| > 10: {sum(1 for r in results if r['abs_delta'] > 10)}")

    # Unique traits with |delta| > 20
    strong_traits = set()
    for r in results:
        if r["abs_delta"] > 20:
            strong_traits.add(r["trait"])
    print(f"\nUnique traits with |delta| > 20: {len(strong_traits)}")

    # Top results
    print(f"\n{'Trait':<45} {'Exp':<22} {'L':>3} {'Delta':>7} {'Coh':>5} {'Base':>5}")
    print("-" * 92)
    for r in results[:80]:
        print(f"{r['trait']:<45} {r['experiment']:<22} {r['layer'] or 0:>3} {r['delta']:>+7.1f} {r['coherence'] or 0:>5.0f} {r['baseline']:>5.1f}")

    # Per-experiment summary
    print(f"\n\nPer-experiment counts:")
    from collections import Counter
    exp_counts = Counter(r["experiment"] for r in results)
    for exp, count in exp_counts.most_common():
        strong = sum(1 for r in results if r["experiment"] == exp and r["abs_delta"] > 20)
        print(f"  {exp:<30} {count:>3} total, {strong:>3} with |delta|>20")

    # Save full results
    out = Path("scripts/steering_delta_analysis.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
