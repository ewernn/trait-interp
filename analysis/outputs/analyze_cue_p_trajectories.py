"""Analyze cue_p trajectories from sentence_metadata.json.
Extracts per-problem trajectories, monotonicity, jumps, plateaus, and positional averages.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

DATA = Path("/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments/mats-mental-state-circuits/thought_branches/sentence_metadata.json")

with open(DATA) as f:
    data = json.load(f)

# ── 1. Per-problem trajectory extraction ──
print("=" * 100)
print("1. PER-PROBLEM CUE_P TRAJECTORIES")
print("=" * 100)

trajectories = {}
for problem_id in sorted(data.keys(), key=lambda x: int(x)):
    sentences = sorted(data[problem_id], key=lambda s: s["sentence_num"])
    traj = [s["cue_p"] for s in sentences]
    trajectories[problem_id] = traj
    final = traj[-1]
    n = len(traj)
    # Compact representation: round to 2 decimal places
    traj_str = "[" + ", ".join(f"{v:.2f}" for v in traj) + "]"
    print(f"Problem {problem_id:>4s}: n={n:>3d}, final={final:.2f}  {traj_str}")

# ── 2. Summary statistics ──
print("\n" + "=" * 100)
print("2. SUMMARY STATISTICS")
print("=" * 100)

# Monotonic increase check
monotonic_count = 0
monotonic_problems = []
for pid, traj in trajectories.items():
    is_mono = all(traj[i] <= traj[i+1] for i in range(len(traj)-1))
    if is_mono:
        monotonic_count += 1
        monotonic_problems.append(pid)

print(f"\nMonotonic increase (non-decreasing): {monotonic_count}/{len(trajectories)}")
if monotonic_problems:
    print(f"  Problems: {monotonic_problems}")

# Weakly monotonic (allow small dips <= 0.05)
weak_mono = 0
weak_mono_problems = []
for pid, traj in trajectories.items():
    is_weak = all(traj[i+1] >= traj[i] - 0.05 for i in range(len(traj)-1))
    if is_weak:
        weak_mono += 1
        weak_mono_problems.append(pid)
print(f"Weakly monotonic (dips <= 0.05 allowed): {weak_mono}/{len(trajectories)}")

# Big jumps (>0.2 in one sentence)
print(f"\nBig jumps (>0.2 increase in one step):")
big_jump_count = 0
big_jump_problems = []
for pid, traj in trajectories.items():
    jumps = []
    for i in range(len(traj)-1):
        delta = traj[i+1] - traj[i]
        if delta > 0.2:
            jumps.append((i, i+1, traj[i], traj[i+1], delta))
    if jumps:
        big_jump_count += 1
        big_jump_problems.append(pid)
        for (si, si1, v0, v1, d) in jumps:
            print(f"  Problem {pid:>4s}: sentence {si}->{si1}, {v0:.2f}->{v1:.2f} (delta={d:+.2f})")

print(f"\nProblems with big jumps: {big_jump_count}/{len(trajectories)}")
print(f"  IDs: {big_jump_problems}")

# Big drops (>0.2 decrease)
print(f"\nBig drops (>0.2 decrease in one step):")
big_drop_count = 0
for pid, traj in trajectories.items():
    drops = []
    for i in range(len(traj)-1):
        delta = traj[i+1] - traj[i]
        if delta < -0.2:
            drops.append((i, i+1, traj[i], traj[i+1], delta))
    if drops:
        big_drop_count += 1
        for (si, si1, v0, v1, d) in drops:
            print(f"  Problem {pid:>4s}: sentence {si}->{si1}, {v0:.2f}->{v1:.2f} (delta={d:+.2f})")
print(f"Problems with big drops: {big_drop_count}/{len(trajectories)}")

# Plateau analysis: when does cue_p first reach within 0.1 of final value?
print(f"\nPlateau analysis (first reaching within 0.1 of final value):")
early_plateau = []  # first half
late_plateau = []   # second half
plateau_fracs = []
for pid, traj in trajectories.items():
    final = traj[-1]
    n = len(traj)
    first_near = None
    for i, v in enumerate(traj):
        if abs(v - final) <= 0.1:
            first_near = i
            break
    if first_near is not None:
        frac = first_near / max(n-1, 1)
        plateau_fracs.append(frac)
        if frac <= 0.5:
            early_plateau.append((pid, first_near, n, frac))
        else:
            late_plateau.append((pid, first_near, n, frac))

print(f"  Early plateau (first half): {len(early_plateau)}/{len(trajectories)}")
for pid, idx, n, frac in early_plateau:
    print(f"    Problem {pid:>4s}: reaches plateau at sentence {idx}/{n-1} ({frac:.1%})")
print(f"  Late plateau (second half): {len(late_plateau)}/{len(trajectories)}")
for pid, idx, n, frac in late_plateau:
    print(f"    Problem {pid:>4s}: reaches plateau at sentence {idx}/{n-1} ({frac:.1%})")
if plateau_fracs:
    print(f"  Mean plateau position: {sum(plateau_fracs)/len(plateau_fracs):.1%}")

# ── 3. Pattern analysis ──
print("\n" + "=" * 100)
print("3. PATTERN ANALYSIS")
print("=" * 100)

# Step sizes across all problems
all_deltas = []
for pid, traj in trajectories.items():
    for i in range(len(traj)-1):
        all_deltas.append(traj[i+1] - traj[i])

print(f"\nStep size distribution (all sentence transitions):")
print(f"  Total steps: {len(all_deltas)}")
print(f"  Mean: {sum(all_deltas)/len(all_deltas):+.4f}")
import statistics
print(f"  Median: {statistics.median(all_deltas):+.4f}")
print(f"  Std: {statistics.stdev(all_deltas):.4f}")
print(f"  Min: {min(all_deltas):+.4f}")
print(f"  Max: {max(all_deltas):+.4f}")

# Distribution of step sizes
buckets = {"large_drop (<-0.2)": 0, "medium_drop (-0.2,-0.1)": 0, "small_drop (-0.1,-0.02)": 0,
           "flat (-0.02,+0.02)": 0, "small_rise (+0.02,+0.1)": 0, "medium_rise (+0.1,+0.2)": 0,
           "large_rise (>+0.2)": 0}
for d in all_deltas:
    if d < -0.2: buckets["large_drop (<-0.2)"] += 1
    elif d < -0.1: buckets["medium_drop (-0.2,-0.1)"] += 1
    elif d < -0.02: buckets["small_drop (-0.1,-0.02)"] += 1
    elif d <= 0.02: buckets["flat (-0.02,+0.02)"] += 1
    elif d <= 0.1: buckets["small_rise (+0.02,+0.1)"] += 1
    elif d <= 0.2: buckets["medium_rise (+0.1,+0.2)"] += 1
    else: buckets["large_rise (>+0.2)"] += 1

print(f"\nStep size buckets:")
for bucket, count in buckets.items():
    pct = count / len(all_deltas) * 100
    bar = "#" * int(pct)
    print(f"  {bucket:>30s}: {count:>4d} ({pct:5.1f}%) {bar}")

# "Point of no return": For problems that end high (>0.7), find the first sentence
# after which cue_p never drops below 0.5
print(f"\n'Point of no return' analysis (problems ending with cue_p > 0.7):")
high_final = {pid: traj for pid, traj in trajectories.items() if traj[-1] > 0.7}
print(f"  Problems ending > 0.7: {len(high_final)}/{len(trajectories)}")
for pid, traj in sorted(high_final.items(), key=lambda x: int(x[0])):
    n = len(traj)
    ponr = None
    for i in range(n):
        if all(v >= 0.5 for v in traj[i:]):
            ponr = i
            break
    if ponr is not None:
        frac = ponr / max(n-1, 1)
        print(f"    Problem {pid:>4s}: PONR at sentence {ponr}/{n-1} ({frac:.1%}), value={traj[ponr]:.2f}, final={traj[-1]:.2f}")
    else:
        print(f"    Problem {pid:>4s}: Never stays above 0.5 (final={traj[-1]:.2f})")

# ── 4. Positional averages ──
print("\n" + "=" * 100)
print("4. AVERAGE CUE_P AT SENTENCE POSITIONS")
print("=" * 100)

# By absolute position
max_len = max(len(t) for t in trajectories.values())
print(f"\nMax trajectory length: {max_len}")
print(f"\nAbsolute position averages:")
print(f"  {'Pos':>4s}  {'N_problems':>10s}  {'Mean_cue_p':>10s}  {'Std':>8s}  {'Min':>6s}  {'Max':>6s}")
for pos in range(max_len):
    vals = [traj[pos] for traj in trajectories.values() if len(traj) > pos]
    if vals:
        mean_v = sum(vals) / len(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"  {pos:>4d}  {len(vals):>10d}  {mean_v:>10.4f}  {std_v:>8.4f}  {min(vals):>6.2f}  {max(vals):>6.2f}")

# Specific positions requested
print(f"\nRequested positions (0, 5, 10, 15, 20):")
for pos in [0, 5, 10, 15, 20]:
    vals = [traj[pos] for traj in trajectories.values() if len(traj) > pos]
    if vals:
        mean_v = sum(vals) / len(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"  Position {pos:>2d}: mean={mean_v:.4f}, std={std_v:.4f}, n={len(vals)}, min={min(vals):.2f}, max={max(vals):.2f}")
    else:
        print(f"  Position {pos:>2d}: no problems with this many sentences")

# By normalized position (fraction of total length)
print(f"\nNormalized position averages (deciles):")
decile_vals = defaultdict(list)
for pid, traj in trajectories.items():
    n = len(traj)
    for i, v in enumerate(traj):
        frac = i / max(n-1, 1)
        decile = min(int(frac * 10), 9)
        decile_vals[decile].append(v)

for d in range(10):
    vals = decile_vals[d]
    if vals:
        mean_v = sum(vals) / len(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"  {d*10:>2d}-{(d+1)*10:>2d}%: mean={mean_v:.4f}, std={std_v:.4f}, n={len(vals)}")

# ── 5. Final value distribution ──
print("\n" + "=" * 100)
print("5. FINAL CUE_P DISTRIBUTION")
print("=" * 100)
finals = [traj[-1] for traj in trajectories.values()]
print(f"\nFinal cue_p values:")
print(f"  Mean: {sum(finals)/len(finals):.4f}")
print(f"  Median: {statistics.median(finals):.4f}")
print(f"  Std: {statistics.stdev(finals):.4f}")
print(f"  Min: {min(finals):.2f}")
print(f"  Max: {max(finals):.2f}")

# Histogram
print(f"\nFinal value histogram:")
for lo in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    hi = lo + 0.1
    count = sum(1 for f in finals if lo <= f < hi)
    bar = "#" * count
    print(f"  [{lo:.1f}, {hi:.1f}): {count:>3d} {bar}")
# count exactly 1.0
count_1 = sum(1 for f in finals if f >= 1.0)
print(f"  [1.0, 1.0]: {count_1:>3d} {'#' * count_1}")

# Starting value distribution
starts = [traj[0] for traj in trajectories.values()]
print(f"\nStarting cue_p values:")
print(f"  Mean: {sum(starts)/len(starts):.4f}")
print(f"  Median: {statistics.median(starts):.4f}")
print(f"  Min: {min(starts):.2f}")
print(f"  Max: {max(starts):.2f}")

# Net change
print(f"\nNet change (final - start):")
net_changes = [traj[-1] - traj[0] for traj in trajectories.values()]
print(f"  Mean: {sum(net_changes)/len(net_changes):+.4f}")
print(f"  Median: {statistics.median(net_changes):+.4f}")
print(f"  Min: {min(net_changes):+.4f}")
print(f"  Max: {max(net_changes):+.4f}")
positive_change = sum(1 for nc in net_changes if nc > 0)
negative_change = sum(1 for nc in net_changes if nc < 0)
zero_change = sum(1 for nc in net_changes if nc == 0)
print(f"  Positive change: {positive_change}/{len(net_changes)}")
print(f"  Negative change: {negative_change}/{len(net_changes)}")
print(f"  Zero change: {zero_change}/{len(net_changes)}")
