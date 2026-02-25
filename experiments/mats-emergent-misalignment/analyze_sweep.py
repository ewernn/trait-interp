"""Analyze checkpoint sweep results for the EM fine-tuning run.

Extracts per-trait trends, detection thresholds, and monotonicity metrics
from probe projection scores measured across training checkpoints.

Input: experiments/mats-emergent-misalignment/analysis/checkpoint_sweep/rank32.json
Output: Summary table printed to stdout + plots saved to analysis/checkpoint_sweep/
Usage: python experiments/mats-emergent-misalignment/analyze_sweep.py
"""

import json
import numpy as np
from pathlib import Path

# --- Load data ---
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "analysis" / "checkpoint_sweep" / "rank32.json"
PLOT_DIR = ROOT / "analysis" / "checkpoint_sweep" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    data = json.load(f)

traits = list(data["probes"].keys())
baseline_scores = data["baseline"]["scores"]
checkpoints = data["checkpoints"]
steps = [cp["step"] for cp in checkpoints]
n_checkpoints = len(checkpoints)

print(f"Loaded {len(traits)} traits, {n_checkpoints} checkpoints")
print(f"Steps range: {steps[0]} to {steps[-1]} (every {steps[1]-steps[0]} steps)")
print(f"Probes: {', '.join(traits)}")
print()

# --- Build per-trait time series ---
# Each series[trait] = np.array of scores across checkpoints
series = {}
for trait in traits:
    series[trait] = np.array([cp["scores"][trait] for cp in checkpoints])

# --- Compute deltas from baseline ---
deltas = {}
for trait in traits:
    deltas[trait] = series[trait] - baseline_scores[trait]

# --- Noise estimation: std of first 5 checkpoints' deltas ---
N_NOISE = 5
noise_std = {}
for trait in traits:
    noise_std[trait] = np.std(deltas[trait][:N_NOISE], ddof=1)

# --- Monotonicity metric ---
# Spearman-like: fraction of consecutive pairs that go in the dominant direction
def monotonicity_score(arr):
    """Fraction of consecutive steps going in the dominant direction.
    1.0 = perfectly monotonic, 0.5 = random walk."""
    diffs = np.diff(arr)
    if len(diffs) == 0:
        return 0.5
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    n_zero = np.sum(diffs == 0)
    total = len(diffs)
    if total == 0:
        return 0.5
    dominant = max(n_pos, n_neg)
    return (dominant + n_zero * 0.5) / total

monotonicity = {}
for trait in traits:
    monotonicity[trait] = monotonicity_score(series[trait])

# --- Trend direction ---
# Use linear regression slope
from numpy.polynomial.polynomial import polyfit
trend_direction = {}
trend_slope = {}
for trait in traits:
    # Fit line: score = a + b*step
    coeffs = polyfit(steps, series[trait], 1)
    slope = coeffs[1]
    trend_slope[trait] = slope
    if slope > 0:
        trend_direction[trait] = "UP"
    elif slope < 0:
        trend_direction[trait] = "DOWN"
    else:
        trend_direction[trait] = "FLAT"

# --- First detection step (delta > baseline + 2*sigma) ---
# Direction-aware: for probes with positive direction, look for delta > +2sigma
# For all probes here, direction is "positive" per the JSON, so increasing score = more of the trait
DETECTION_THRESHOLD = 2.0

first_detection = {}
for trait in traits:
    sigma = noise_std[trait]
    detected = False
    for i, step in enumerate(steps):
        d = deltas[trait][i]
        # Check if delta exceeds threshold in either direction
        if abs(d) > DETECTION_THRESHOLD * sigma and sigma > 0:
            first_detection[trait] = step
            detected = True
            break
    if not detected:
        first_detection[trait] = None

# --- Also compute: sustained detection (3 consecutive above threshold) ---
sustained_detection = {}
for trait in traits:
    sigma = noise_std[trait]
    detected = False
    if sigma == 0:
        sustained_detection[trait] = None
        continue
    for i in range(len(steps) - 2):
        if all(abs(deltas[trait][i+j]) > DETECTION_THRESHOLD * sigma for j in range(3)):
            sustained_detection[trait] = steps[i]
            detected = True
            break
    if not detected:
        sustained_detection[trait] = None

# --- Variance / coefficient of variation ---
cv = {}
for trait in traits:
    mean_abs = np.mean(np.abs(series[trait]))
    if mean_abs > 0:
        cv[trait] = np.std(series[trait]) / mean_abs
    else:
        cv[trait] = float('inf')

# --- Print summary table ---
print("=" * 140)
print(f"{'Trait':<30} {'Baseline':>9} {'Final':>9} {'Delta':>9} {'Delta%':>8} "
      f"{'Trend':>5} {'Slope':>10} {'Mono':>5} {'CV':>6} "
      f"{'Noise_s':>8} {'1st_det':>8} {'Sust_det':>8}")
print("-" * 140)

# Sort by absolute delta (largest signal first)
sorted_traits = sorted(traits, key=lambda t: abs(deltas[t][-1]), reverse=True)

for trait in sorted_traits:
    base = baseline_scores[trait]
    final = series[trait][-1]
    delta = deltas[trait][-1]
    # Delta as percentage of baseline magnitude (or absolute if baseline near zero)
    if abs(base) > 0.5:
        delta_pct = (delta / abs(base)) * 100
    else:
        delta_pct = delta * 100  # just scale for visibility

    det = first_detection[trait]
    sust = sustained_detection[trait]

    print(f"{trait:<30} {base:>9.3f} {final:>9.3f} {delta:>9.3f} {delta_pct:>7.1f}% "
          f"{trend_direction[trait]:>5} {trend_slope[trait]:>10.6f} {monotonicity[trait]:>5.2f} {cv[trait]:>6.3f} "
          f"{noise_std[trait]:>8.4f} "
          f"{str(det) if det else 'None':>8} "
          f"{str(sust) if sust else 'None':>8}")

print("=" * 140)

# --- Categorize probes ---
print("\n\n### PROBES WITH STRONGEST SIGNAL (|delta| at final checkpoint, sorted) ###\n")
for i, trait in enumerate(sorted_traits):
    delta = deltas[trait][-1]
    sigma = noise_std[trait]
    n_sigma = abs(delta) / sigma if sigma > 0 else float('inf')
    det = first_detection[trait]
    print(f"  {i+1:>2}. {trait:<30} delta={delta:>+8.3f}  ({n_sigma:>5.1f} sigma)  "
          f"first_det=step {det if det else 'N/A':<6}  mono={monotonicity[trait]:.2f}")

# --- Earliest detectors ---
print("\n\n### EARLIEST DETECTION (probes that detected EM shift first) ###\n")
detected_probes = [(t, s) for t, s in first_detection.items() if s is not None]
detected_probes.sort(key=lambda x: x[1])
if detected_probes:
    for trait, step in detected_probes:
        sigma = noise_std[trait]
        delta_at_det = deltas[trait][steps.index(step)]
        n_sigma_at_det = abs(delta_at_det) / sigma if sigma > 0 else float('inf')
        print(f"  Step {step:>4}: {trait:<30} (delta={delta_at_det:>+.3f}, {n_sigma_at_det:.1f}sigma at detection)")
else:
    print("  No probes exceeded the 2-sigma threshold.")

print("\n\n### SUSTAINED DETECTION (3 consecutive checkpoints above threshold) ###\n")
sustained_probes = [(t, s) for t, s in sustained_detection.items() if s is not None]
sustained_probes.sort(key=lambda x: x[1])
if sustained_probes:
    for trait, step in sustained_probes:
        print(f"  Step {step:>4}: {trait:<30}")
else:
    print("  No probes showed sustained detection.")

# --- Probes with NO signal ---
print("\n\n### PROBES WITH NO CLEAR SIGNAL (delta within noise at final checkpoint) ###\n")
no_signal = []
for trait in sorted_traits:
    sigma = noise_std[trait]
    delta = abs(deltas[trait][-1])
    if sigma > 0 and delta < 2 * sigma:
        no_signal.append(trait)
    elif sigma == 0:
        no_signal.append(trait)

if no_signal:
    for trait in no_signal:
        sigma = noise_std[trait]
        delta = deltas[trait][-1]
        n_sigma = abs(delta) / sigma if sigma > 0 else 0
        print(f"  {trait:<30} delta={delta:>+.3f}  ({n_sigma:.1f} sigma)")
else:
    print("  All probes show signal above 2 sigma at final checkpoint.")

# --- Most monotonic probes ---
print("\n\n### MOST MONOTONIC PROBES (consistent trend) ###\n")
mono_sorted = sorted(traits, key=lambda t: monotonicity[t], reverse=True)
for trait in mono_sorted[:5]:
    print(f"  {trait:<30} mono={monotonicity[trait]:.2f}  trend={trend_direction[trait]}  "
          f"slope={trend_slope[trait]:+.6f}")

print("\n### NOISIEST PROBES (highest CV) ###\n")
cv_sorted = sorted(traits, key=lambda t: cv[t], reverse=True)
for trait in cv_sorted[:5]:
    print(f"  {trait:<30} CV={cv[trait]:.3f}  mono={monotonicity[trait]:.2f}")

# --- Generate plots ---
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot 1: All probes' deltas over training steps
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
    fig.suptitle("Probe Projection Deltas Over EM Fine-tuning (rank-32)", fontsize=16, y=0.98)

    for idx, trait in enumerate(sorted_traits):
        ax = axes[idx // 4][idx % 4]
        delta_arr = deltas[trait]
        sigma = noise_std[trait]

        ax.plot(steps, delta_arr, 'b-', linewidth=1.0, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        if sigma > 0:
            ax.axhline(y=2*sigma, color='r', linestyle='--', linewidth=0.5, alpha=0.5, label='+2sigma')
            ax.axhline(y=-2*sigma, color='r', linestyle='--', linewidth=0.5, alpha=0.5, label='-2sigma')

        # Mark first detection
        det_step = first_detection[trait]
        if det_step is not None:
            det_idx = steps.index(det_step)
            ax.axvline(x=det_step, color='green', linestyle=':', linewidth=1.0, alpha=0.7)
            ax.plot(det_step, delta_arr[det_idx], 'go', markersize=5)

        short_name = trait.split('/')[-1]
        ax.set_title(f"{short_name}", fontsize=10)
        ax.tick_params(labelsize=8)
        if idx >= 12:
            ax.set_xlabel("Step", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot1_path = PLOT_DIR / "all_probes_deltas.png"
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"\nSaved: {plot1_path}")

    # Plot 2: Top 6 probes by signal strength, with more detail
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Top 6 Probes by Signal Strength (Delta from Baseline)", fontsize=14, y=0.98)

    top6 = sorted_traits[:6]
    for idx, trait in enumerate(top6):
        ax = axes[idx // 3][idx % 3]
        delta_arr = deltas[trait]
        sigma = noise_std[trait]

        ax.plot(steps, delta_arr, 'b-o', linewidth=1.5, markersize=2, alpha=0.8)
        ax.fill_between(steps, delta_arr, 0, alpha=0.15, color='blue')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        if sigma > 0:
            ax.axhspan(-2*sigma, 2*sigma, alpha=0.1, color='red', label='noise band (2sigma)')

        det_step = first_detection[trait]
        if det_step is not None:
            det_idx = steps.index(det_step)
            ax.axvline(x=det_step, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'1st detect: step {det_step}')
            ax.plot(det_step, delta_arr[det_idx], 'go', markersize=8, zorder=5)

        n_sigma_final = abs(delta_arr[-1]) / sigma if sigma > 0 else float('inf')
        ax.set_title(f"{trait}\ndelta={delta_arr[-1]:+.2f} ({n_sigma_final:.1f}sigma), mono={monotonicity[trait]:.2f}",
                     fontsize=10)
        ax.set_xlabel("Training Step", fontsize=9)
        ax.set_ylabel("Delta from Baseline", fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot2_path = PLOT_DIR / "top6_probes_detail.png"
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Saved: {plot2_path}")

    # Plot 3: Heatmap of all probe deltas (normalized by noise)
    fig, ax = plt.subplots(figsize=(20, 8))

    # Build matrix: traits x steps, normalized by noise
    matrix = np.zeros((len(sorted_traits), len(steps)))
    for i, trait in enumerate(sorted_traits):
        sigma = noise_std[trait]
        if sigma > 0:
            matrix[i] = deltas[trait] / sigma
        else:
            matrix[i] = deltas[trait]

    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-10, vmax=10,
                   interpolation='nearest')
    ax.set_yticks(range(len(sorted_traits)))
    ax.set_yticklabels([t.split('/')[-1] for t in sorted_traits], fontsize=9)

    # Show every 5th step on x axis
    step_indices = list(range(0, len(steps), 4))
    ax.set_xticks(step_indices)
    ax.set_xticklabels([str(steps[i]) for i in step_indices], fontsize=8, rotation=45)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_title("Probe Deltas (normalized by noise sigma) Over Training", fontsize=13)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Delta / noise_sigma", fontsize=10)

    plt.tight_layout()
    plot3_path = PLOT_DIR / "probe_deltas_heatmap.png"
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    print(f"Saved: {plot3_path}")

    # Plot 4: Detection timeline
    fig, ax = plt.subplots(figsize=(14, 6))

    detected = [(t, s) for t, s in first_detection.items() if s is not None]
    detected.sort(key=lambda x: x[1])
    not_detected = [t for t in traits if first_detection[t] is None]

    colors = plt.cm.tab20(np.linspace(0, 1, len(detected)))
    for i, (trait, step) in enumerate(detected):
        short = trait.split('/')[-1]
        ax.barh(i, step, color=colors[i], alpha=0.8, height=0.6)
        ax.text(step + 3, i, f"{short} (step {step})", va='center', fontsize=9)

    ax.set_xlabel("Training Step at First Detection (2-sigma threshold)", fontsize=11)
    ax.set_title("When Each Probe First Detected EM Shift", fontsize=13)
    ax.set_yticks([])
    ax.set_xlim(0, max(steps) + 50)

    if not_detected:
        ax.text(0.95, 0.05, f"Not detected: {', '.join([t.split('/')[-1] for t in not_detected])}",
                transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plot4_path = PLOT_DIR / "detection_timeline.png"
    plt.savefig(plot4_path, dpi=150)
    plt.close()
    print(f"Saved: {plot4_path}")

    # Plot 5: Raw scores (not deltas) for context
    fig, ax = plt.subplots(figsize=(16, 8))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(traits)))
    for i, trait in enumerate(sorted_traits[:8]):
        short = trait.split('/')[-1]
        ax.plot(steps, series[trait], '-', color=cmap[i], linewidth=1.5, label=short, alpha=0.8)
        # baseline
        ax.plot(0, baseline_scores[trait], 'o', color=cmap[i], markersize=6)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='baseline')
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Probe Projection Score", fontsize=11)
    ax.set_title("Raw Probe Scores Over Training (Top 8 by Signal)", fontsize=13)
    ax.legend(fontsize=9, loc='best', ncol=2)
    plt.tight_layout()
    plot5_path = PLOT_DIR / "raw_scores_top8.png"
    plt.savefig(plot5_path, dpi=150)
    plt.close()
    print(f"Saved: {plot5_path}")

except ImportError:
    print("\nmatplotlib not available - skipping plots")

# --- Final summary ---
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

n_detected = len([t for t in traits if first_detection[t] is not None])
n_sustained = len([t for t in traits if sustained_detection[t] is not None])
earliest = min((s for s in first_detection.values() if s is not None), default=None)
earliest_trait = [t for t in traits if first_detection[t] == earliest] if earliest else []

print(f"\nTotal probes: {len(traits)}")
print(f"Probes with 2-sigma detection: {n_detected}/{len(traits)}")
print(f"Probes with sustained detection (3 consec): {n_sustained}/{len(traits)}")
if earliest:
    print(f"Earliest detection: step {earliest} by {', '.join(earliest_trait)}")
print(f"Total training steps: {steps[-1]}")
print(f"Behavioral eval: 23.8% misaligned at final checkpoint (step 397)")

# What fraction of training do probes detect before end?
if earliest:
    print(f"Earliest probe detection at {earliest/steps[-1]*100:.1f}% of training")

print("\nKey findings:")
# Identify probes with large monotonic trends
strong_mono = [(t, monotonicity[t], trend_direction[t], deltas[t][-1])
               for t in traits if monotonicity[t] > 0.65 and abs(deltas[t][-1]) > 2 * noise_std[t]]
strong_mono.sort(key=lambda x: abs(x[3]), reverse=True)
if strong_mono:
    print("  Strong monotonic trends:")
    for t, m, d, delta in strong_mono:
        print(f"    {t}: {d} (mono={m:.2f}, final delta={delta:+.3f})")

noisy = [(t, cv[t]) for t in traits if cv[t] > 1.0]
noisy.sort(key=lambda x: x[1], reverse=True)
if noisy:
    print("  Noisy probes (CV > 1.0):")
    for t, c in noisy:
        print(f"    {t}: CV={c:.3f}")
