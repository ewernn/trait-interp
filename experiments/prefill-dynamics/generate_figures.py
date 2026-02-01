"""
Generate figures for prefill-dynamics experiment.

Input: analysis/*.json files
Output: figures/*.png

Usage: PYTHONPATH=. python experiments/prefill-dynamics/generate_figures.py
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# Paths
EXPERIMENT_DIR = Path(__file__).parent
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'human': '#e74c3c', 'model': '#3498db'}
TRAIT_COLORS = {'refusal': '#2ecc71', 'sycophancy': '#9b59b6'}


def load_json(name: str) -> dict:
    path = ANALYSIS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fig1_smoothness_by_layer():
    """Line plot: Cohen's d by layer for raw smoothness."""
    data = load_json("activation_metrics.json")
    if not data:
        print("  [skip] activation_metrics.json not found")
        return

    layers = []
    cohens_d = []
    for layer_str, stats in data['summary']['by_layer'].items():
        layers.append(int(layer_str))
        cohens_d.append(stats['smoothness_cohens_d'])

    layers, cohens_d = zip(*sorted(zip(layers, cohens_d)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, cohens_d, 'o-', linewidth=2, markersize=6, color='#2c3e50')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect (d=0.8)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel("Cohen's d (Human - Model smoothness)", fontsize=12)
    ax.set_title('Activation Smoothness: Effect Size by Layer', fontsize=14)
    ax.legend()

    peak_idx = np.argmax(cohens_d)
    ax.annotate(f'd={cohens_d[peak_idx]:.2f}',
                xy=(layers[peak_idx], cohens_d[peak_idx]),
                xytext=(layers[peak_idx]+1, cohens_d[peak_idx]+0.1),
                fontsize=10, ha='left')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "smoothness_by_layer.png", dpi=150)
    plt.close()
    print("  [done] smoothness_by_layer.png")


def fig2_violin_smoothness():
    """Violin plot: smoothness distributions for human vs model."""
    data = load_json("activation_metrics.json")
    if not data:
        print("  [skip] activation_metrics.json not found")
        return

    by_layer = data['summary']['by_layer']
    peak_layer = max(by_layer.keys(), key=lambda k: by_layer[k]['smoothness_cohens_d'])

    human_vals = [s['human'][peak_layer]['smoothness'] for s in data['samples']
                  if peak_layer in s.get('human', {})]
    model_vals = [s['model'][peak_layer]['smoothness'] for s in data['samples']
                  if peak_layer in s.get('model', {})]

    if not human_vals or not model_vals:
        print("  [skip] no valid smoothness data")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    parts = ax.violinplot([human_vals, model_vals], positions=[1, 2], showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([COLORS['human'], COLORS['model']][i])
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Human-written', 'Model-generated'])
    ax.set_ylabel('Smoothness (lower = smoother)', fontsize=11)
    ax.set_title(f'Activation Smoothness Distribution (Layer {peak_layer})', fontsize=14)

    d = by_layer[peak_layer]['smoothness_cohens_d']
    p = by_layer[peak_layer]['smoothness_p_value']
    ax.text(0.95, 0.95, f"Cohen's d = {d:.2f}\np = {p:.1e}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "violin_smoothness.png", dpi=150)
    plt.close()
    print("  [done] violin_smoothness.png")


def fig3_projection_stability_by_layer():
    """Line plot: projection variance Cohen's d by layer for multiple traits."""
    fig, ax = plt.subplots(figsize=(10, 5))
    found_any = False

    for trait_file in ANALYSIS_DIR.glob("projection_stability-*.json"):
        with open(trait_file) as f:
            data = json.load(f)

        trait_name = data.get('trait', trait_file.stem.replace('projection_stability_', ''))
        short_name = trait_name.split('/')[-1]

        layers = []
        cohens_d = []
        for layer_str, stats in data['by_layer'].items():
            layers.append(int(layer_str))
            cohens_d.append(stats['var_cohens_d'])

        if layers:
            layers, cohens_d = zip(*sorted(zip(layers, cohens_d)))
            color = TRAIT_COLORS.get(short_name, '#34495e')
            ax.plot(layers, cohens_d, 'o-', linewidth=2, markersize=5,
                   label=short_name.title(), color=color)
            found_any = True

    if not found_any:
        print("  [skip] no projection_stability_*.json files found")
        plt.close()
        return

    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel("Cohen's d (Projection Variance: Human - Model)", fontsize=12)
    ax.set_title('Trait Projection Stability by Layer', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "projection_stability_by_layer.png", dpi=150)
    plt.close()
    print("  [done] projection_stability_by_layer.png")


def fig4_position_breakdown():
    """Line graph: variance by token position bin."""
    data = load_json("position_breakdown.json")
    if not data:
        print("  [skip] position_breakdown.json not found")
        return

    positions = list(data['by_position'].keys())
    human_var = [data['by_position'][p]['human_var'] for p in positions]
    model_var = [data['by_position'][p]['model_var'] for p in positions]

    x = np.arange(len(positions))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Line plot instead of bar
    ax.plot(x, human_var, 'o-', linewidth=2, markersize=8,
            label='Human-written', color=COLORS['human'])
    ax.plot(x, model_var, 'o-', linewidth=2, markersize=8,
            label='Model-generated', color=COLORS['model'])

    # Shade the difference
    ax.fill_between(x, model_var, human_var, alpha=0.2, color='gray')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Projection Variance', fontsize=12)
    ax.set_title(f"Projection Variance by Token Position (Layer {data['layer']})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()

    # Annotate the growing gap
    for i, pos in enumerate(positions):
        diff = data['by_position'][pos]['diff']
        mid_y = (human_var[i] + model_var[i]) / 2
        ax.text(i + 0.1, mid_y, f'+{diff:.1f}', fontsize=9, color='#555', va='center')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "position_breakdown.png", dpi=150)
    plt.close()
    print("  [done] position_breakdown.png")


def fig5_perplexity_scatter():
    """Scatter: perplexity vs smoothness with regression line."""
    ppl_data = load_json("perplexity.json")
    act_data = load_json("activation_metrics.json")

    if not ppl_data or not act_data:
        print("  [skip] perplexity.json or activation_metrics.json not found")
        return

    by_layer = act_data['summary']['by_layer']
    peak_layer = max(by_layer.keys(), key=lambda k: by_layer[k]['smoothness_cohens_d'])

    # Build lookup - handle both 'results' and 'samples' keys
    ppl_items = ppl_data.get('results') or ppl_data.get('samples', [])
    ppl_by_id = {r['id']: r for r in ppl_items}

    human_ppl, human_smooth = [], []
    model_ppl, model_smooth = [], []

    for sample in act_data['samples']:
        sid = sample['id']
        if sid not in ppl_by_id:
            continue
        if peak_layer not in sample.get('human', {}) or peak_layer not in sample.get('model', {}):
            continue

        ppl = ppl_by_id[sid]
        human_ppl.append(ppl.get('human_ce') or ppl.get('human', {}).get('mean_ce_loss', 0))
        human_smooth.append(sample['human'][peak_layer]['smoothness'])
        model_ppl.append(ppl.get('model_ce') or ppl.get('model', {}).get('mean_ce_loss', 0))
        model_smooth.append(sample['model'][peak_layer]['smoothness'])

    if not human_ppl:
        print("  [skip] no matching samples for scatter")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(human_ppl, human_smooth, c=COLORS['human'], alpha=0.6, label='Human-written', s=40)
    ax.scatter(model_ppl, model_smooth, c=COLORS['model'], alpha=0.6, label='Model-generated', s=40)

    # Add regression line for pooled data
    all_ppl = human_ppl + model_ppl
    all_smooth = human_smooth + model_smooth
    z = np.polyfit(all_ppl, all_smooth, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(all_ppl), max(all_ppl), 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, linewidth=2)

    # Compute and show correlation
    r, p_val = stats.pearsonr(all_ppl, all_smooth)
    ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Cross-Entropy Loss (perplexity proxy)', fontsize=12)
    ax.set_ylabel(f'Smoothness (Layer {peak_layer})', fontsize=12)
    ax.set_title('Perplexity vs Activation Smoothness', fontsize=14)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "perplexity_vs_smoothness.png", dpi=150)
    plt.close()
    print("  [done] perplexity_vs_smoothness.png")


def fig6_effect_comparison():
    """Dual line plot: raw smoothness d vs projection stability d by layer."""
    act_data = load_json("activation_metrics.json")
    if not act_data:
        print("  [skip] activation_metrics.json not found")
        return

    # Find projection stability files
    proj_files = list(ANALYSIS_DIR.glob("projection_stability-*.json"))
    if not proj_files:
        print("  [skip] no projection_stability-*.json files found")
        return

    # Get smoothness data
    act_layers = sorted([int(l) for l in act_data['summary']['by_layer'].keys()])
    smoothness_d = [act_data['summary']['by_layer'][str(l)]['smoothness_cohens_d'] for l in act_layers]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot smoothness
    ax.plot(act_layers, smoothness_d, 'o-', linewidth=2.5, markersize=6,
            label='Raw Smoothness', color='#2c3e50')

    # Plot projection stability for each trait
    for proj_file in proj_files:
        with open(proj_file) as f:
            proj_data = json.load(f)

        trait_name = proj_data.get('trait', proj_file.stem.replace('projection_stability-', ''))
        short_name = trait_name.split('/')[-1]

        proj_layers = sorted([int(l) for l in proj_data['by_layer'].keys()])
        proj_d = [proj_data['by_layer'][str(l)]['var_cohens_d'] for l in proj_layers]

        color = TRAIT_COLORS.get(short_name, '#95a5a6')
        ax.plot(proj_layers, proj_d, 's--', linewidth=2, markersize=5,
                label=f'Projection Stability ({short_name})', color=color, alpha=0.8)

    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large effect (d=0.8)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.set_title('Effect Size by Layer: Raw Smoothness vs Projection Stability', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(-1, max(act_layers) + 1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "effect_comparison.png", dpi=150)
    plt.close()
    print("  [done] effect_comparison.png")


if __name__ == "__main__":
    print(f"Generating figures to {FIGURES_DIR}/\n")

    fig1_smoothness_by_layer()
    fig2_violin_smoothness()
    fig3_projection_stability_by_layer()
    fig4_position_breakdown()
    fig5_perplexity_scatter()
    fig6_effect_comparison()

    print(f"\nDone. Figures saved to {FIGURES_DIR}/")
