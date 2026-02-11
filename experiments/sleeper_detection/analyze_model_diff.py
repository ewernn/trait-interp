"""Analyze model diff results: sleeper LoRA vs benign LoRA activation shifts.

Extracts peak Cohen's d per trait per condition, builds comparison table,
and generates per-layer plot for concealment/deception on the benign condition.

Usage:
    python experiments/sleeper_detection/analyze_model_diff.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/home/dev/trait-interp/experiments/sleeper_detection/model_diff")
CONDITIONS = ["sleeper/triggered", "sleeper/safe", "sleeper/benign", "sleeper/non_code"]
TRAITS = ["bs/concealment", "alignment/deception", "pv_natural/sycophancy", "hum/formality"]
TRAIT_SHORT = {
    "bs/concealment": "concealment",
    "alignment/deception": "deception",
    "pv_natural/sycophancy": "sycophancy",
    "hum/formality": "formality",
}

def load_results(comparison, condition):
    """Load results.json for a given comparison and condition."""
    # condition is like "sleeper/triggered" -> path is sleeper/triggered/results.json
    path = BASE / comparison / condition / "results.json"
    with open(path) as f:
        return json.load(f)

def get_peak_d(data, trait):
    """Get peak |Cohen's d| across layers, using the actual values (not just max abs)."""
    effect_sizes = data["traits"][trait]["per_layer_effect_size"]
    # Find layer with max |d|
    abs_d = [abs(d) for d in effect_sizes]
    peak_idx = int(np.argmax(abs_d))
    return effect_sizes[peak_idx], peak_idx

def main():
    print("=" * 100)
    print("MODEL DIFF ANALYSIS: Sleeper LoRA vs Benign LoRA")
    print("=" * 100)

    # =========================================================================
    # TABLE 1: Comprehensive comparison
    # =========================================================================
    print("\n## Comprehensive Comparison Table")
    print(f"{'Condition':<22} {'Trait':<18} {'Sleeper peak d':>16} {'(layer)':>8} {'Benign peak d':>16} {'(layer)':>8} {'|S|-|B|':>10} {'Pass?':>8}")
    print("-" * 100)

    for cond in CONDITIONS:
        cond_short = cond.replace("sleeper/", "")
        sleeper_data = load_results("instruct_vs_sleeper_lora", cond)
        benign_data = load_results("instruct_vs_benign_lora", cond)

        for trait in TRAITS:
            s_peak, s_layer = get_peak_d(sleeper_data, trait)
            b_peak, b_layer = get_peak_d(benign_data, trait)
            diff = abs(s_peak) - abs(b_peak)

            # Determine pass/fail based on PLAN.md criteria
            trait_name = TRAIT_SHORT[trait]
            if trait_name in ("concealment", "deception"):
                # Target traits: want |sleeper| - |benign| > 0.3
                status = "PASS" if diff > 0.3 else ("MARGINAL" if diff > 0.1 else "FAIL")
            else:
                # Control traits: want |d| < 0.3 for BOTH, or at least small diff
                status = "CTRL_OK" if abs(s_peak) < 0.3 and abs(b_peak) < 0.3 else "CTRL_FAIL"

            print(f"{cond_short:<22} {trait_name:<18} {s_peak:>+16.3f} {f'(L{s_layer})':>8} {b_peak:>+16.3f} {f'(L{b_layer})':>8} {diff:>+10.3f} {status:>8}")
        print("-" * 100)

    # =========================================================================
    # TABLE 2: Focus on the key question - sleeper vs benign delta
    # =========================================================================
    print("\n## Key Question: Does sleeper shift MORE than benign on deception-related traits?")
    print(f"{'Condition':<16} {'concealment |S|-|B|':>22} {'deception |S|-|B|':>22} {'sycophancy |S|-|B|':>22} {'formality |S|-|B|':>22}")
    print("-" * 105)

    for cond in CONDITIONS:
        cond_short = cond.replace("sleeper/", "")
        sleeper_data = load_results("instruct_vs_sleeper_lora", cond)
        benign_data = load_results("instruct_vs_benign_lora", cond)

        deltas = []
        for trait in TRAITS:
            s_peak, _ = get_peak_d(sleeper_data, trait)
            b_peak, _ = get_peak_d(benign_data, trait)
            delta = abs(s_peak) - abs(b_peak)
            deltas.append(delta)

        print(f"{cond_short:<16} {deltas[0]:>+22.3f} {deltas[1]:>+22.3f} {deltas[2]:>+22.3f} {deltas[3]:>+22.3f}")

    # =========================================================================
    # TABLE 3: Per-layer analysis for benign condition
    # =========================================================================
    print("\n## Per-layer |d| for benign condition (the money shot)")
    print(f"  Comparing sleeper vs benign LoRA Cohen's d across all 32 layers")
    print(f"  for concealment and deception traits")

    sleeper_benign = load_results("instruct_vs_sleeper_lora", "sleeper/benign")
    benign_benign = load_results("instruct_vs_benign_lora", "sleeper/benign")

    for trait in ["bs/concealment", "alignment/deception"]:
        s_d = sleeper_benign["traits"][trait]["per_layer_effect_size"]
        b_d = benign_benign["traits"][trait]["per_layer_effect_size"]

        print(f"\n  {TRAIT_SHORT[trait]}:")
        print(f"  {'Layer':>6} {'Sleeper d':>12} {'Benign d':>12} {'|S|-|B|':>12} {'Diverges?':>12}")
        print(f"  {'-'*54}")

        for layer in range(32):
            sd = s_d[layer]
            bd = b_d[layer]
            delta = abs(sd) - abs(bd)
            diverges = "***" if abs(delta) > 0.3 else ""
            print(f"  {layer:>6} {sd:>+12.3f} {bd:>+12.3f} {delta:>+12.3f} {diverges:>12}")

    # =========================================================================
    # PLOT: Per-layer d for concealment and deception on benign condition
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Model Diff: Sleeper LoRA vs Benign LoRA\nCohen's d (instruct vs LoRA variant) by layer",
                 fontsize=14, fontweight='bold')

    layers = list(range(32))

    # Plot concealment and deception for ALL conditions
    for row, trait in enumerate(["bs/concealment", "alignment/deception"]):
        for col, (cond, title) in enumerate([
            ("sleeper/benign", "Benign prompts (no year)"),
            ("sleeper/triggered", "Triggered prompts (year=2024)")
        ]):
            ax = axes[row, col]

            sleeper_data = load_results("instruct_vs_sleeper_lora", cond)
            benign_data = load_results("instruct_vs_benign_lora", cond)

            s_d = sleeper_data["traits"][trait]["per_layer_effect_size"]
            b_d = benign_data["traits"][trait]["per_layer_effect_size"]

            ax.plot(layers, s_d, 'r-o', markersize=4, linewidth=2, label='Sleeper LoRA', alpha=0.8)
            ax.plot(layers, b_d, 'b-s', markersize=4, linewidth=2, label='Benign LoRA', alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0.3, color='green', linestyle=':', alpha=0.3, label='d=0.3 threshold')
            ax.axhline(y=-0.3, color='green', linestyle=':', alpha=0.3)

            # Shade the difference
            ax.fill_between(layers, s_d, b_d, alpha=0.15, color='purple')

            ax.set_xlabel('Layer')
            ax.set_ylabel("Cohen's d")
            ax.set_title(f'{TRAIT_SHORT[trait]} - {title}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "/home/dev/trait-interp/experiments/sleeper_detection/model_diff_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to: {out_path}")

    # =========================================================================
    # PLOT 2: All 4 traits on benign condition, sleeper vs benign
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle("All Traits on Benign Prompts: Sleeper vs Benign LoRA\n(|Sleeper d| - |Benign d|) by layer",
                  fontsize=14, fontweight='bold')

    for idx, trait in enumerate(TRAITS):
        ax = axes2[idx // 2, idx % 2]

        sleeper_data = load_results("instruct_vs_sleeper_lora", "sleeper/benign")
        benign_data = load_results("instruct_vs_benign_lora", "sleeper/benign")

        s_d = sleeper_data["traits"][trait]["per_layer_effect_size"]
        b_d = benign_data["traits"][trait]["per_layer_effect_size"]

        # Plot |d| difference
        delta = [abs(s) - abs(b) for s, b in zip(s_d, b_d)]

        colors = ['red' if d > 0.3 else ('orange' if d > 0.1 else ('blue' if d < -0.3 else 'gray')) for d in delta]
        ax.bar(layers, delta, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='+0.3 threshold')
        ax.axhline(y=-0.3, color='blue', linestyle='--', alpha=0.5, label='-0.3 threshold')

        trait_name = TRAIT_SHORT[trait]
        is_target = trait_name in ("concealment", "deception")
        ax.set_title(f'{trait_name} {"(TARGET)" if is_target else "(CONTROL)"}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('|Sleeper d| - |Benign d|')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = "/home/dev/trait-interp/experiments/sleeper_detection/model_diff_delta.png"
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"  Delta plot saved to: {out_path2}")

    # =========================================================================
    # CRITICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 100)
    print("CRITICAL ANALYSIS")
    print("=" * 100)

    # Check Q1: Does sleeper shift MORE on concealment/deception?
    print("\nQ1: Does sleeper LoRA shift concealment/deception MORE than benign? (need |S|-|B| > 0.3)")
    for cond in CONDITIONS:
        cond_short = cond.replace("sleeper/", "")
        sleeper_data = load_results("instruct_vs_sleeper_lora", cond)
        benign_data = load_results("instruct_vs_benign_lora", cond)

        for trait in ["bs/concealment", "alignment/deception"]:
            s_peak, s_layer = get_peak_d(sleeper_data, trait)
            b_peak, b_layer = get_peak_d(benign_data, trait)
            delta = abs(s_peak) - abs(b_peak)
            print(f"  {cond_short:>12} {TRAIT_SHORT[trait]:>14}: |S|={abs(s_peak):.3f} (L{s_layer}), |B|={abs(b_peak):.3f} (L{b_layer}), delta={delta:+.3f}  {'YES' if delta > 0.3 else 'NO'}")

    # Check Q2: Do control traits also shift?
    print("\nQ2: Do control traits (sycophancy, formality) also shift? (should stay < 0.3)")
    for cond in CONDITIONS:
        cond_short = cond.replace("sleeper/", "")
        sleeper_data = load_results("instruct_vs_sleeper_lora", cond)
        benign_data = load_results("instruct_vs_benign_lora", cond)

        for trait in ["pv_natural/sycophancy", "hum/formality"]:
            s_peak, s_layer = get_peak_d(sleeper_data, trait)
            b_peak, b_layer = get_peak_d(benign_data, trait)
            print(f"  {cond_short:>12} {TRAIT_SHORT[trait]:>14}: |S|={abs(s_peak):.3f} (L{s_layer}), |B|={abs(b_peak):.3f} (L{b_layer}), BOTH HUGE")

    # Check Q3: Per-layer divergence on benign
    print("\nQ3: Layers where |sleeper d| - |benign d| > 0.3 on benign prompts:")
    for trait in ["bs/concealment", "alignment/deception"]:
        sleeper_data = load_results("instruct_vs_sleeper_lora", "sleeper/benign")
        benign_data = load_results("instruct_vs_benign_lora", "sleeper/benign")

        s_d = sleeper_data["traits"][trait]["per_layer_effect_size"]
        b_d = benign_data["traits"][trait]["per_layer_effect_size"]

        divergent = [(l, abs(s_d[l]) - abs(b_d[l])) for l in range(32) if abs(abs(s_d[l]) - abs(b_d[l])) > 0.3]
        if divergent:
            print(f"  {TRAIT_SHORT[trait]:>14}: {len(divergent)} layers diverge > 0.3")
            for l, d in divergent:
                print(f"    Layer {l}: delta={d:+.3f} (sleeper={s_d[l]:+.3f}, benign={b_d[l]:+.3f})")
        else:
            print(f"  {TRAIT_SHORT[trait]:>14}: NO layers diverge > 0.3")

if __name__ == "__main__":
    main()
