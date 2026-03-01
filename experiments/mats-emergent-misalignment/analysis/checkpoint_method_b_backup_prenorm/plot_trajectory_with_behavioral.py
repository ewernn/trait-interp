"""Plot Method B model_delta trajectories for top 10 probes with behavioral EM overlay.

Input: checkpoint_method_b/{rank32,sports,financial,insecure}.json
       checkpoint_behavioral/{rank32,sports,financial}_results.json
Output: checkpoint_method_b/trajectory_with_behavioral.png
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).parent
BEHAV_DIR = BASE.parent / 'checkpoint_behavioral'

DOMAINS = ['rank32', 'sports', 'financial', 'insecure']
DOMAIN_LABELS = {
    'rank32': 'Medical',
    'sports': 'Sports',
    'financial': 'Financial',
    'insecure': 'Insecure Code',
}

# Load Method B data
mb_data = {}
for domain in DOMAINS:
    with open(BASE / f'{domain}.json') as f:
        mb_data[domain] = json.load(f)

probes = list(mb_data['rank32']['probes'].keys())

def short_name(p):
    return p.split('/')[-1]

# Load behavioral data
behav_data = {}
for domain in ['rank32', 'sports', 'financial']:
    path = BEHAV_DIR / f'{domain}_results.json'
    if path.exists():
        with open(path) as f:
            behav_data[domain] = json.load(f)

# Find top 10 probes by max absolute delta across all domains at final step
final_deltas = {}
for domain in DOMAINS:
    cps = mb_data[domain]['checkpoints']
    final = [c for c in cps if c['step'] == 999]
    if not final:
        final = [c for c in cps if c['step'] != 999]
        final = [sorted(final, key=lambda c: c['step'])[-1]]
    final_deltas[domain] = final[0]['model_delta']

max_abs = {}
for p in probes:
    max_abs[p] = max(abs(final_deltas[d][p]) for d in DOMAINS)
top10 = sorted(probes, key=lambda p: max_abs[p], reverse=True)[:10]

# Color palette for probes
probe_colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for idx, domain in enumerate(DOMAINS):
    ax = axes[idx // 2][idx % 2]

    # Get Method B trajectory
    cps = mb_data[domain]['checkpoints']
    real_cps = sorted([c for c in cps if c['step'] != 999], key=lambda c: c['step'])
    steps = [c['step'] for c in real_cps]

    # Plot top 10 probes
    for pi, p in enumerate(top10):
        vals = [c['model_delta'][p] for c in real_cps]
        ax.plot(steps, vals, '-', linewidth=1.3, color=probe_colors[pi],
                label=short_name(p), alpha=0.85)

    # Overlay behavioral EM rate on secondary y-axis
    if domain in behav_data:
        ax2 = ax.twinx()
        bd = behav_data[domain]

        behav_steps = []
        behav_rates = []
        for key, metrics in sorted(bd.items(), key=lambda x: int(x[0].split('_')[1]) if '_' in x[0] else -1):
            if key == 'baseline':
                step = 0
            else:
                step = int(key.split('_')[1])
            behav_steps.append(step)
            behav_rates.append(metrics['misaligned_rate'] * 100)

        ax2.bar(behav_steps, behav_rates, width=6, alpha=0.3, color='red',
                zorder=0, edgecolor='red', linewidth=0.5, label='EM rate (%)')
        ax2.set_ylabel('Behavioral EM Rate (%)', color='#cc0000', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='#cc0000')
        ax2.set_ylim(0, 50)
    else:
        # No behavioral data for insecure
        ax2 = ax.twinx()
        ax2.set_ylabel('(no behavioral eval)', color='gray', fontsize=8)
        ax2.set_yticks([])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Model Delta (Method B)')
    ax.set_title(DOMAIN_LABELS[domain], fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.2)

    # Only show legend on first subplot
    if idx == 0:
        ax.legend(fontsize=7, loc='upper left', ncol=2)

# Shared legend at bottom
legend_elements = [Line2D([0], [0], color=probe_colors[i], linewidth=1.5,
                          label=short_name(top10[i])) for i in range(10)]
legend_elements.append(Line2D([0], [0], color='red', linewidth=6, alpha=0.25,
                               label='Behavioral EM rate (%)'))

fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Method B Probe Trajectories + Behavioral EM Rate Across Domains',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(BASE / 'trajectory_with_behavioral.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {BASE / 'trajectory_with_behavioral.png'}")
