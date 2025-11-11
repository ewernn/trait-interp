#!/usr/bin/env python3
"""
Comprehensive temporal dynamics analysis across all experiments.
"""

import json
import numpy as np
from scipy import stats
import os

def load_all_results():
    """Load all experiment results."""
    results = {}

    files = {
        "baseline": "pertoken/results/pilot_results.json",
        "contaminated": "pertoken/results/contaminated_results.json",
        "expanded": "pertoken/results/expanded_results.json",
        "layers": "pertoken/results/layer_comparison.json",
    }

    for name, filepath in files.items():
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                results[name] = json.load(f)
            if isinstance(results[name], list):
                print(f"✅ Loaded {name}: {len(results[name])} results")
            else:
                print(f"✅ Loaded {name}: dictionary")
        else:
            print(f"⚠️  {name} not found")

    return results


def analyze_decision_points(results):
    """Q1: When does the model decide to express traits?"""
    print("\n" + "="*70)
    print("ANALYSIS 1: DECISION POINTS")
    print("="*70)
    print("Question: At what token does trait expression commit?")

    decision_points = {
        "evil": [],
        "sycophantic": [],
        "hallucinating": []
    }

    # Analyze baseline results
    if "baseline" in results:
        for result in results["baseline"]:
            for trait in ["evil", "sycophantic", "hallucinating"]:
                if trait not in result["projections"]:
                    continue

                scores = result["projections"][trait]
                if len(scores) < 5:
                    continue

                # Find first significant deviation from baseline
                baseline_mean = np.mean(scores[:3])  # First 3 tokens as baseline
                threshold = baseline_mean + np.std(scores[:3])

                first_spike = None
                for i in range(3, len(scores)):
                    if scores[i] > threshold:
                        first_spike = i
                        break

                if first_spike:
                    decision_points[trait].append(first_spike)

    # Summary
    print("\nDecision point statistics (token index where trait first spikes):")
    for trait, points in decision_points.items():
        if points:
            print(f"  {trait.capitalize():15s}: mean={np.mean(points):5.1f}, median={np.median(points):5.1f}, std={np.std(points):5.1f}")
            print(f"                        range: {min(points)}-{max(points)} tokens")


def analyze_trait_cascades(results):
    """Q2: Do traits cascade sequentially?"""
    print("\n" + "="*70)
    print("ANALYSIS 2: TRAIT CASCADES")
    print("="*70)
    print("Question: Do traits correlate temporally (evil → hallucinating)?")

    if "baseline" not in results:
        print("No baseline results to analyze")
        return

    correlations = []

    for result in results["baseline"]:
        traits = {}
        for trait in ["evil", "sycophantic", "hallucinating"]:
            if trait in result["projections"] and result["projections"][trait]:
                traits[trait] = result["projections"][trait]

        if len(traits) < 2:
            continue

        # Compute correlations between all trait pairs
        trait_names = list(traits.keys())
        for i in range(len(trait_names)):
            for j in range(i+1, len(trait_names)):
                trait1, trait2 = trait_names[i], trait_names[j]
                scores1, scores2 = traits[trait1], traits[trait2]

                min_len = min(len(scores1), len(scores2))
                if min_len < 5:
                    continue

                scores1, scores2 = scores1[:min_len], scores2[:min_len]

                try:
                    corr, p_value = stats.pearsonr(scores1, scores2)
                    correlations.append({
                        "pair": f"{trait1}-{trait2}",
                        "correlation": corr,
                        "p_value": p_value,
                        "prompt": result["prompt"][:50]
                    })
                except:
                    continue

    if correlations:
        print(f"\nTemporal correlations across {len(correlations)} prompt-pairs:")

        # Group by trait pair
        pairs = {}
        for c in correlations:
            pair = c["pair"]
            if pair not in pairs:
                pairs[pair] = []
            pairs[pair].append(c["correlation"])

        for pair, corrs in pairs.items():
            mean_corr = np.mean(corrs)
            status = ""
            if abs(mean_corr) > 0.3:
                status = " ← POSITIVE CASCADE" if mean_corr > 0 else " ← NEGATIVE CASCADE"

            print(f"  {pair:25s}: mean r={mean_corr:+6.3f} (n={len(corrs)}){status}")


def analyze_hallucination_mystery(results):
    """Q3: Why is within-condition correlation weak?"""
    print("\n" + "="*70)
    print("ANALYSIS 3: HALLUCINATION MYSTERY")
    print("="*70)
    print("Paper found r=0.245 within-condition for hallucinating (weak!)")
    print("Hypothesis: Spiky distribution (not uniform) causes averaging artifact")

    if "expanded" not in results:
        print("No expanded results to analyze")
        return

    # Find hallucination-trigger prompts
    hallucination_results = [
        r for r in results["expanded"]
        if r.get("category", "").startswith("hallucination_")
    ]

    if not hallucination_results:
        print("No hallucination prompts found")
        return

    print(f"\nAnalyzing {len(hallucination_results)} hallucination-trigger prompts:")

    for result in hallucination_results[:5]:  # First 5
        if "hallucinating" not in result["projections"]:
            continue

        scores = result["projections"]["hallucinating"]
        if len(scores) < 5:
            continue

        # Check distribution shape
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        cv = std_score / (abs(mean_score) + 1e-8)  # Coefficient of variation

        # Find spikes
        threshold = mean_score + std_score
        spikes = [i for i, s in enumerate(scores) if s > threshold]
        spike_density = len(spikes) / len(scores) if scores else 0

        print(f"\n  Prompt: {result['prompt'][:50]}...")
        print(f"    Mean: {mean_score:6.2f}, Std: {std_score:5.2f}, CV: {cv:5.2f}")
        print(f"    Spikes: {len(spikes)}/{len(scores)} tokens ({spike_density*100:.1f}%)")
        print(f"    Distribution: {'SPIKY ✓' if cv > 0.5 else 'UNIFORM'}")


def analyze_contamination_effect(results):
    """Q4: How does contamination change temporal dynamics?"""
    print("\n" + "="*70)
    print("ANALYSIS 4: CONTAMINATION EFFECT ON TEMPORAL DYNAMICS")
    print("="*70)

    if "baseline" not in results or "contaminated" not in results:
        print("Missing baseline or contaminated results")
        return

    print("Comparing temporal patterns: Baseline vs Contaminated")

    # Match prompts between baseline and contaminated
    matched_count = 0
    for baseline_result in results["baseline"][:5]:  # First 5 matching prompts
        prompt = baseline_result["prompt"]

        # Find matching contaminated result
        contaminated_result = next(
            (r for r in results["contaminated"] if r["prompt"] == prompt),
            None
        )

        if not contaminated_result:
            continue

        matched_count += 1
        print(f"\n  Prompt: {prompt[:50]}...")

        for trait in ["evil", "sycophantic", "hallucinating"]:
            if trait not in baseline_result["projections"] or trait not in contaminated_result["projections"]:
                continue

            baseline_scores = baseline_result["projections"][trait]
            contam_scores = contaminated_result["projections"][trait]

            if not baseline_scores or not contam_scores:
                continue

            baseline_mean = np.mean(baseline_scores)
            contam_mean = np.mean(contam_scores)

            baseline_max = max(baseline_scores)
            contam_max = max(contam_scores)

            baseline_spike_token = np.argmax(baseline_scores)
            contam_spike_token = np.argmax(contam_scores)

            print(f"    {trait.capitalize():15s}:")
            print(f"      Mean:  baseline={baseline_mean:6.2f}, contam={contam_mean:6.2f}, diff={contam_mean-baseline_mean:+6.2f}")
            print(f"      Max:   baseline={baseline_max:6.2f} @tok{baseline_spike_token}, contam={contam_max:6.2f} @tok{contam_spike_token}")

    if matched_count == 0:
        print("  No matching prompts found between baseline and contaminated")


def generate_summary_report(results):
    """Generate comprehensive summary."""
    print("\n" + "="*70)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("="*70)

    total_prompts = 0
    total_tokens = 0

    for name, data in results.items():
        if isinstance(data, list):
            total_prompts += len(data)
            total_tokens += sum(r.get("metadata", {}).get("num_tokens", 0) for r in data)

    print(f"\nTotal prompts analyzed: {total_prompts}")
    print(f"Total tokens tracked: {total_tokens}")
    print(f"Experiments completed: {len(results)}")

    # Save summary
    summary = {
        "total_prompts": total_prompts,
        "total_tokens": total_tokens,
        "experiments": list(results.keys()),
        "analyses_completed": [
            "decision_points",
            "trait_cascades",
            "hallucination_mystery",
            "contamination_effect"
        ]
    }

    with open("pertoken/results/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved to pertoken/results/analysis_summary.json")


def main():
    """Run all analyses."""
    print("="*70)
    print("EXPERIMENT 5: COMPREHENSIVE TEMPORAL DYNAMICS ANALYSIS")
    print("="*70)

    # Load all results
    print("\nLoading experimental results...")
    results = load_all_results()

    if not results:
        print("❌ No results found to analyze!")
        return

    # Run analyses
    analyze_decision_points(results)
    analyze_trait_cascades(results)
    analyze_hallucination_mystery(results)
    analyze_contamination_effect(results)
    generate_summary_report(results)

    print("\n" + "="*70)
    print("✅ ALL ANALYSES COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
