"""Cross-domain comparison of Method B checkpoint fingerprints.

Compares rank32 (medical baseline), sports, financial, and insecure code
fine-tuning runs across 23 trait probes.

Input: checkpoint_method_b/{rank32,sports,financial,insecure}.json
Output: Console tables + checkpoint_method_b/cross_domain_*.png plots
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = Path(__file__).parent
DOMAINS = ['rank32', 'sports', 'financial', 'insecure']
DOMAIN_LABELS = {'rank32': 'medical (baseline)', 'sports': 'sports', 'financial': 'financial', 'insecure': 'insecure code'}

# Load all data
data = {}
for domain in DOMAINS:
    with open(BASE / f'{domain}.json') as f:
        data[domain] = json.load(f)

probes = list(data['rank32']['probes'].keys())
n_probes = len(probes)

# Short probe names for display
def short_name(p):
    return p.split('/')[-1]


# =============================================================================
# 1. FINAL FINGERPRINTS COMPARISON TABLE
# =============================================================================
print("=" * 120)
print("1. FINAL FINGERPRINTS (model_delta at last real checkpoint, excluding step=999)")
print("=" * 120)

# Get final checkpoint (last before step=999)
final_deltas = {}
final_steps = {}
for domain in DOMAINS:
    checkpoints = data[domain]['checkpoints']
    # Find last checkpoint before step 999
    real_checkpoints = [c for c in checkpoints if c['step'] != 999]
    final_cp = real_checkpoints[-1]
    final_deltas[domain] = final_cp['model_delta']
    final_steps[domain] = final_cp['step']

print(f"\nFinal steps: { {d: final_steps[d] for d in DOMAINS} }")

# Also get step=999 (full training) for comparison
full_deltas = {}
for domain in DOMAINS:
    checkpoints = data[domain]['checkpoints']
    cp999 = [c for c in checkpoints if c['step'] == 999]
    if cp999:
        full_deltas[domain] = cp999[0]['model_delta']

# Sort probes by max absolute delta across domains (at step 999 / full training)
ref = full_deltas if full_deltas else final_deltas
max_abs = {}
for p in probes:
    max_abs[p] = max(abs(ref[d][p]) for d in DOMAINS)
sorted_probes = sorted(probes, key=lambda p: max_abs[p], reverse=True)

print(f"\n{'Probe':<30} {'medical':>10} {'sports':>10} {'financial':>10} {'insecure':>10}  {'max|d|':>8}  sign_agree")
print("-" * 120)
for p in sorted_probes:
    vals = [full_deltas[d][p] for d in DOMAINS]
    max_v = max(abs(v) for v in vals)
    signs = [np.sign(v) for v in vals if abs(v) > 0.1]  # only count non-negligible
    agree = "YES" if signs and all(s == signs[0] for s in signs) else ("mixed" if signs else "~0")
    print(f"{short_name(p):<30} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} {vals[3]:>10.3f}  {max_v:>8.3f}  {agree}")

# Also show step=last_real checkpoint deltas
print(f"\n--- Same table at last real checkpoint (not step=999) ---")
print(f"{'Probe':<30} {'medical':>10} {'sports':>10} {'financial':>10} {'insecure':>10}  {'max|d|':>8}")
print("-" * 120)
for p in sorted_probes:
    vals = [final_deltas[d][p] for d in DOMAINS]
    max_v = max(abs(v) for v in vals)
    print(f"{short_name(p):<30} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} {vals[3]:>10.3f}  {max_v:>8.3f}")


# =============================================================================
# 2. FINGERPRINT CONVERGENCE (Spearman rank correlation between domain pairs)
# =============================================================================
print("\n" + "=" * 120)
print("2. FINGERPRINT CONVERGENCE - Spearman rank correlation between domain pairs")
print("=" * 120)

# Build step -> model_delta for each domain
def get_delta_at_step(domain, step):
    for c in data[domain]['checkpoints']:
        if c['step'] == step:
            return np.array([c['model_delta'][p] for p in probes])
    return None

# Find common steps
all_steps_sets = []
for domain in DOMAINS:
    steps = set(c['step'] for c in data[domain]['checkpoints'])
    all_steps_sets.append(steps)
common_steps = sorted(set.intersection(*all_steps_sets))

# Key steps to report
report_steps = [10, 20, 50, 100, 200]
# Add final common step (before 999)
real_common = [s for s in common_steps if s != 999]
if real_common:
    report_steps.append(real_common[-1])
# Add 999 if available
if 999 in common_steps:
    report_steps.append(999)

# Remove duplicates and sort
report_steps = sorted(set(s for s in report_steps if s in common_steps))

# Compute pairwise Spearman correlations
from itertools import combinations
pairs = list(combinations(DOMAINS, 2))
pair_labels = [f"{DOMAIN_LABELS[a][:8]} vs {DOMAIN_LABELS[b][:8]}" for a, b in pairs]

print(f"\n{'Step':>6}", end="")
for pl in pair_labels:
    print(f"  {pl:>22}", end="")
print(f"  {'mean_rho':>10}")
print("-" * (6 + 24 * len(pairs) + 12))

convergence_data = {}  # step -> list of rhos
for step in report_steps:
    rhos = []
    print(f"{step:>6}", end="")
    for a, b in pairs:
        va = get_delta_at_step(a, step)
        vb = get_delta_at_step(b, step)
        if va is not None and vb is not None:
            rho, pval = stats.spearmanr(va, vb)
            rhos.append(rho)
            sig = "*" if pval < 0.05 else " "
            print(f"  {rho:>20.3f}{sig}", end="")
        else:
            print(f"  {'N/A':>22}", end="")
    mean_rho = np.mean(rhos) if rhos else float('nan')
    convergence_data[step] = rhos
    print(f"  {mean_rho:>10.3f}")

# Compute full convergence trajectory for plotting
all_rhos_by_step = {}
for step in common_steps:
    rhos = []
    for a, b in pairs:
        va = get_delta_at_step(a, step)
        vb = get_delta_at_step(b, step)
        if va is not None and vb is not None:
            rho, _ = stats.spearmanr(va, vb)
            rhos.append(rho)
    all_rhos_by_step[step] = rhos

# Also compute cosine similarity for comparison with Soligo
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\n--- Cosine similarity (for comparison with Soligo's cos=0.81 at step 5) ---")
print(f"{'Step':>6}", end="")
for pl in pair_labels:
    print(f"  {pl:>22}", end="")
print(f"  {'mean_cos':>10}")
print("-" * (6 + 24 * len(pairs) + 12))

cos_data_by_step = {}
for step in report_steps:
    coss = []
    print(f"{step:>6}", end="")
    for a, b in pairs:
        va = get_delta_at_step(a, step)
        vb = get_delta_at_step(b, step)
        if va is not None and vb is not None:
            c = cosine_sim(va, vb)
            coss.append(c)
            print(f"  {c:>22.3f}", end="")
        else:
            print(f"  {'N/A':>22}", end="")
    mean_c = np.mean(coss) if coss else float('nan')
    cos_data_by_step[step] = coss
    print(f"  {mean_c:>10.3f}")


# =============================================================================
# 3. DETECTION TIMING
# =============================================================================
print("\n" + "=" * 120)
print("3. DETECTION TIMING - Earliest step where any probe exceeds threshold")
print("=" * 120)

# We use multiple thresholds
thresholds = [0.1, 0.2, 0.5, 1.0, 2.0]

# Also compute sigma-based: sigma from first checkpoint (step 10) across probes
# For each domain, the model_delta at step 10 gives us a baseline noise level
print("\nNoise floor estimates (std of model_delta across 23 probes at step 10):")
for domain in DOMAINS:
    d10 = get_delta_at_step(domain, 10)
    if d10 is not None:
        print(f"  {DOMAIN_LABELS[domain]:<20}: std={np.std(d10):.4f}, max|d|={np.max(np.abs(d10)):.4f}")

print(f"\n{'Domain':<20}", end="")
for t in thresholds:
    print(f"  |d|>{t:<4}", end="")
print(f"  {'2sigma':>8}  {'probe_that_triggers':>30}")
print("-" * 120)

for domain in DOMAINS:
    checkpoints = data[domain]['checkpoints']

    # 2-sigma from step 10
    d10 = get_delta_at_step(domain, 10)
    sigma = np.std(d10) if d10 is not None else 0.15
    sigma_thresh = 2 * sigma

    print(f"{DOMAIN_LABELS[domain]:<20}", end="")

    first_trigger_probe = None
    for t in thresholds:
        found_step = None
        for c in checkpoints:
            if c['step'] == 999:
                continue
            for p in probes:
                if abs(c['model_delta'][p]) > t:
                    found_step = c['step']
                    if first_trigger_probe is None and t == thresholds[0]:
                        first_trigger_probe = short_name(p)
                    break
            if found_step:
                break
        print(f"  {found_step if found_step else 'never':>6}", end="")

    # 2-sigma detection
    found_step_sig = None
    trigger_probe_sig = None
    for c in checkpoints:
        if c['step'] == 999:
            continue
        for p in probes:
            if abs(c['model_delta'][p]) > sigma_thresh:
                found_step_sig = c['step']
                trigger_probe_sig = short_name(p)
                break
        if found_step_sig:
            break
    print(f"  {found_step_sig if found_step_sig else 'never':>8}  {trigger_probe_sig or 'N/A':>30}")

# More detailed: for each domain, show the probe that first crosses each threshold
print("\n--- First probe to cross |delta| > 0.5 for each domain ---")
for domain in DOMAINS:
    checkpoints = data[domain]['checkpoints']
    for c in sorted(checkpoints, key=lambda x: x['step']):
        if c['step'] == 999:
            continue
        triggered = [(p, c['model_delta'][p]) for p in probes if abs(c['model_delta'][p]) > 0.5]
        if triggered:
            print(f"  {DOMAIN_LABELS[domain]:<20} step {c['step']:>4}: ", end="")
            triggered.sort(key=lambda x: abs(x[1]), reverse=True)
            for p, v in triggered[:5]:
                print(f"{short_name(p)}={v:+.3f}  ", end="")
            print()
            break


# =============================================================================
# 4. TOP DISCRIMINATIVE PROBES
# =============================================================================
print("\n" + "=" * 120)
print("4. TOP DISCRIMINATIVE PROBES")
print("=" * 120)

# 4a. Probes with largest mean |delta| across domains (consistently large)
print("\n--- 4a. Consistently large probes (mean |delta| at step 999 across domains) ---")
mean_abs_deltas = {}
for p in probes:
    vals = [abs(full_deltas[d][p]) for d in DOMAINS]
    mean_abs_deltas[p] = np.mean(vals)

sorted_by_mean = sorted(probes, key=lambda p: mean_abs_deltas[p], reverse=True)
print(f"{'Probe':<30} {'mean|d|':>10} {'std|d|':>10} {'min|d|':>10} {'max|d|':>10}  consistent?")
print("-" * 100)
for p in sorted_by_mean:
    vals = [abs(full_deltas[d][p]) for d in DOMAINS]
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    cv = std_v / mean_v if mean_v > 0 else float('inf')
    consistent = "YES" if cv < 0.5 else "no"
    print(f"{short_name(p):<30} {mean_v:>10.3f} {std_v:>10.3f} {min(vals):>10.3f} {max(vals):>10.3f}  {consistent} (CV={cv:.2f})")

# 4b. Probes that differ most between domains
print("\n--- 4b. Domain-discriminative probes (highest variance across domains at step 999) ---")
var_deltas = {}
for p in probes:
    vals = [full_deltas[d][p] for d in DOMAINS]  # signed, not abs
    var_deltas[p] = np.std(vals)

sorted_by_var = sorted(probes, key=lambda p: var_deltas[p], reverse=True)
print(f"{'Probe':<30} {'std':>10}  {'medical':>10} {'sports':>10} {'financial':>10} {'insecure':>10}")
print("-" * 100)
for p in sorted_by_var[:10]:
    vals = [full_deltas[d][p] for d in DOMAINS]
    print(f"{short_name(p):<30} {np.std(vals):>10.3f}  {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f} {vals[3]:>10.3f}")

# 4c. Sign agreement analysis
print("\n--- 4c. Sign agreement at step 999 (which probes move in same direction for all domains?) ---")
for p in sorted_probes:
    vals = [full_deltas[d][p] for d in DOMAINS]
    signs = ['+' if v > 0.05 else ('-' if v < -0.05 else '~0') for v in vals]
    sign_str = ' '.join(f"{s:>3}" for s in signs)
    all_same = len(set(s for s in signs if s != '~0')) <= 1
    marker = " <<<" if all_same and any(s != '~0' for s in signs) else ""
    if not all_same:
        marker = " *** DIVERGENT"
    print(f"  {short_name(p):<30} [{sign_str}] {marker}")


# =============================================================================
# 5. PLOTS
# =============================================================================

# --- Plot 1: Final fingerprint comparison (bar chart) ---
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
x = np.arange(n_probes)
width = 0.2
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
short_names = [short_name(p) for p in sorted_probes]

for i, domain in enumerate(DOMAINS):
    vals = [full_deltas[domain][p] for p in sorted_probes]
    ax.bar(x + i * width - 1.5 * width, vals, width, label=DOMAIN_LABELS[domain], color=colors[i], alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=60, ha='right', fontsize=8)
ax.set_ylabel('Model Delta (Method B)')
ax.set_title('Final Fingerprints Across Domains (step 999)')
ax.legend(loc='upper right')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(BASE / 'cross_domain_fingerprints.png', dpi=150)
plt.close()
print(f"\nSaved: {BASE / 'cross_domain_fingerprints.png'}")


# --- Plot 2: Convergence over time (Spearman rho) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 2a: Spearman correlation
ax = axes[0]
real_steps = sorted([s for s in common_steps if s != 999])
pair_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#795548']
for idx, (a, b) in enumerate(pairs):
    rhos_over_time = []
    for step in real_steps:
        va = get_delta_at_step(a, step)
        vb = get_delta_at_step(b, step)
        rho, _ = stats.spearmanr(va, vb)
        rhos_over_time.append(rho)
    label = f"{DOMAIN_LABELS[a][:6]} vs {DOMAIN_LABELS[b][:6]}"
    ax.plot(real_steps, rhos_over_time, '-o', markersize=2, color=pair_colors[idx], label=label, alpha=0.7)

# Mean
mean_rhos = []
for step in real_steps:
    rhos = []
    for a, b in pairs:
        va = get_delta_at_step(a, step)
        vb = get_delta_at_step(b, step)
        rho, _ = stats.spearmanr(va, vb)
        rhos.append(rho)
    mean_rhos.append(np.mean(rhos))
ax.plot(real_steps, mean_rhos, '-k', linewidth=2, label='mean', alpha=0.9)

ax.set_xlabel('Training Step')
ax.set_ylabel('Spearman Rank Correlation')
ax.set_title('Fingerprint Convergence (Spearman rho)')
ax.legend(fontsize=7, loc='lower right')
ax.axhline(y=0.81, color='gray', linestyle='--', alpha=0.5, label='Soligo cos=0.81')
ax.set_ylim(-0.5, 1.05)
ax.grid(alpha=0.3)

# 2b: Cosine similarity
ax = axes[1]
for idx, (a, b) in enumerate(pairs):
    coss_over_time = []
    for step in real_steps:
        va = get_delta_at_step(a, step)
        vb = get_delta_at_step(b, step)
        coss_over_time.append(cosine_sim(va, vb))
    label = f"{DOMAIN_LABELS[a][:6]} vs {DOMAIN_LABELS[b][:6]}"
    ax.plot(real_steps, coss_over_time, '-o', markersize=2, color=pair_colors[idx], label=label, alpha=0.7)

mean_coss = []
for step in real_steps:
    coss = []
    for a, b in pairs:
        va = get_delta_at_step(a, step)
        vb = get_delta_at_step(b, step)
        coss.append(cosine_sim(va, vb))
    mean_coss.append(np.mean(coss))
ax.plot(real_steps, mean_coss, '-k', linewidth=2, label='mean', alpha=0.9)

ax.set_xlabel('Training Step')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Fingerprint Convergence (Cosine Similarity)')
ax.legend(fontsize=7, loc='lower right')
ax.axhline(y=0.81, color='gray', linestyle='--', alpha=0.5)
ax.annotate('Soligo cos=0.81 at step 5', xy=(50, 0.81), fontsize=8, color='gray')
ax.set_ylim(-0.5, 1.05)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(BASE / 'cross_domain_convergence.png', dpi=150)
plt.close()
print(f"Saved: {BASE / 'cross_domain_convergence.png'}")


# --- Plot 3: Detection timing - evolution of top probes ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, domain in enumerate(DOMAINS):
    ax = axes[idx // 2][idx % 2]
    checkpoints = data[domain]['checkpoints']
    real_cps = [c for c in checkpoints if c['step'] != 999]
    steps = [c['step'] for c in real_cps]

    # Find top 5 probes by absolute delta at final step
    final_cp = real_cps[-1]
    top_probes = sorted(probes, key=lambda p: abs(final_cp['model_delta'][p]), reverse=True)[:5]

    for p in top_probes:
        vals = [c['model_delta'][p] for c in real_cps]
        ax.plot(steps, vals, '-', linewidth=1.5, label=short_name(p), alpha=0.8)

    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Model Delta')
    ax.set_title(f'{DOMAIN_LABELS[domain]} - Top 5 Probes')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(BASE / 'cross_domain_probe_evolution.png', dpi=150)
plt.close()
print(f"Saved: {BASE / 'cross_domain_probe_evolution.png'}")


# --- Plot 4: Heatmap of final fingerprints ---
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
matrix = np.zeros((len(DOMAINS), n_probes))
for i, domain in enumerate(DOMAINS):
    for j, p in enumerate(sorted_probes):
        matrix[i, j] = full_deltas[domain][p]

im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-np.percentile(np.abs(matrix), 95), vmax=np.percentile(np.abs(matrix), 95))
ax.set_yticks(range(len(DOMAINS)))
ax.set_yticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
ax.set_xticks(range(n_probes))
ax.set_xticklabels([short_name(p) for p in sorted_probes], rotation=60, ha='right', fontsize=8)
ax.set_title('Final Model Delta Heatmap (step 999)')
plt.colorbar(im, ax=ax, shrink=0.6, label='Model Delta')

# Annotate cells with values
for i in range(len(DOMAINS)):
    for j in range(n_probes):
        v = matrix[i, j]
        if abs(v) > 0.3:
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=6,
                   color='white' if abs(v) > np.percentile(np.abs(matrix), 70) else 'black')

plt.tight_layout()
plt.savefig(BASE / 'cross_domain_heatmap.png', dpi=150)
plt.close()
print(f"Saved: {BASE / 'cross_domain_heatmap.png'}")


# --- Summary statistics ---
print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

# Pairwise correlation at key steps
print("\nMean pairwise Spearman rho:")
for step in [10, 20, 50, 100, 200] + [real_common[-1], 999]:
    if step in common_steps:
        rhos = []
        for a, b in pairs:
            va = get_delta_at_step(a, step)
            vb = get_delta_at_step(b, step)
            if va is not None and vb is not None:
                rho, _ = stats.spearmanr(va, vb)
                rhos.append(rho)
        print(f"  step {step:>4}: rho = {np.mean(rhos):.3f} (range {min(rhos):.3f} - {max(rhos):.3f})")

# Cosine at key steps
print("\nMean pairwise cosine similarity:")
for step in [10, 20, 50, 100, 200] + [real_common[-1], 999]:
    if step in common_steps:
        coss = []
        for a, b in pairs:
            va = get_delta_at_step(a, step)
            vb = get_delta_at_step(b, step)
            if va is not None and vb is not None:
                coss.append(cosine_sim(va, vb))
        print(f"  step {step:>4}: cos = {np.mean(coss):.3f} (range {min(coss):.3f} - {max(coss):.3f})")

# Top universal probes (high mean, low CV)
print("\nTop 5 UNIVERSAL probes (large |delta|, consistent across domains):")
for i, p in enumerate(sorted_by_mean[:5]):
    vals = [full_deltas[d][p] for d in DOMAINS]
    print(f"  {i+1}. {short_name(p):<25} mean_delta={np.mean(vals):+.3f}  range=[{min(vals):+.3f}, {max(vals):+.3f}]")

# Top domain-specific probes
print("\nTop 5 DOMAIN-SPECIFIC probes (high variance across domains):")
for i, p in enumerate(sorted_by_var[:5]):
    vals = [full_deltas[d][p] for d in DOMAINS]
    print(f"  {i+1}. {short_name(p):<25} std={np.std(vals):.3f}  vals=[" + ", ".join(f"{v:+.3f}" for v in vals) + "]")
