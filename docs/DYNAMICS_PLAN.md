# Dynamics Analysis - Clean Implementation Plan

## Current State ✅

**traitlens already has the primitives:**
- `compute_derivative()` - velocity
- `compute_second_derivative()` - acceleration
- `projection()` - trait measurement
- Example shows per-token capture during generation

**Architecture is:**
```
traitlens/        # General purpose building blocks
utils/            # Universal utilities (judge, config)
experiments/      # Experiment-specific analysis
```

---

## What's Actually Needed

### Option 1: Nothing (Use Existing Primitives)

Users can already do dynamics analysis:

```python
from traitlens import compute_derivative, compute_second_derivative, projection

# Capture trajectory during generation (see example_minimal.py)
trajectory = capture_per_token(...)  # [n_tokens, hidden_dim]

# Velocity
velocity = compute_derivative(trajectory)

# Commitment point
acceleration = compute_second_derivative(trajectory)
commit_idx = (acceleration.norm(dim=-1) < threshold).nonzero()[0]

# Persistence
peak_idx = trajectory.argmax()
duration = (trajectory[peak_idx:] > threshold).sum()
```

**All primitives exist. Done.**

---

### Option 2: Add Convenience Functions to traitlens

If dynamics analysis is common enough, add **minimal** helpers to `traitlens/`:

#### New file: `traitlens/analysis.py` (~80 lines)

```python
"""
Convenience functions for common analyses.
Built using traitlens primitives.
"""

from .compute import compute_derivative, compute_second_derivative

def find_commitment_point(trajectory, method='acceleration', threshold=0.1):
    """
    Find token where model commits to a decision.

    Uses acceleration drop-off to detect crystallization.
    """
    if method == 'acceleration':
        accel = compute_second_derivative(trajectory)
        accel_mag = accel.norm(dim=-1)
        candidates = (accel_mag < threshold).nonzero()
        return candidates[0].item() if len(candidates) > 0 else None
    elif method == 'velocity':
        vel = compute_derivative(trajectory)
        vel_mag = vel.norm(dim=-1)
        candidates = (vel_mag < threshold).nonzero()
        return candidates[0].item() if len(candidates) > 0 else None
    else:
        raise ValueError(f"Unknown method: {method}")

def measure_persistence(trajectory, decay_threshold=0.5):
    """
    Measure how long trait expression persists after peak.

    Returns number of tokens where expression > threshold.
    """
    peak_idx = trajectory.abs().argmax(dim=0)[0].item()
    if peak_idx >= len(trajectory):
        return 0

    post_peak = trajectory[peak_idx:]
    above_threshold = (post_peak.abs() > decay_threshold).sum(dim=0)
    return above_threshold.item()

def analyze_dynamics(trajectory):
    """
    Complete dynamics analysis.

    Returns dict with commitment, velocity, persistence metrics.
    """
    velocity = compute_derivative(trajectory)
    acceleration = compute_second_derivative(trajectory)

    return {
        'commitment_point': find_commitment_point(trajectory),
        'peak_velocity': velocity.norm(dim=-1).max().item(),
        'avg_velocity': velocity.norm(dim=-1).mean().item(),
        'persistence': measure_persistence(trajectory),
        'velocity_profile': velocity.norm(dim=-1).tolist(),
        'acceleration_profile': acceleration.norm(dim=-1).tolist(),
    }
```

**Add to `traitlens/__init__.py`:**
```python
from .analysis import (
    find_commitment_point,
    measure_persistence,
    analyze_dynamics
)
```

**Size**: ~80 lines, purely convenience wrappers

---

### Option 3: Example Scripts in Experiments

Create reference implementation in experiments:

```
experiments/
└── examples/
    ├── monitor_basic.py           # Basic monitoring
    ├── monitor_with_dynamics.py   # With dynamics analysis
    └── README.md                  # How to customize
```

Users copy these to their own experiments and customize.

---

## Recommended Approach

**Combination:**

1. **Add `traitlens/analysis.py`** (~80 lines)
   - Only truly general-purpose helpers
   - `find_commitment_point()`, `measure_persistence()`, `analyze_dynamics()`
   - Built from existing primitives

2. **Add example in experiments/**
   ```
   experiments/examples/
   └── monitor_with_dynamics.py   # Shows full workflow
   ```

3. **Update docs**
   - Add dynamics section to `docs/main.md`
   - Show the primitives + convenience functions

**Total additions:**
- 1 file in `traitlens/` (~80 lines)
- 1 example script (~150 lines)
- 1 docs section (~50 lines)

---

## What Goes Where

### ✅ traitlens/analysis.py
**Only if truly general purpose:**
- `find_commitment_point()` - Yes, applies to any trajectory
- `measure_persistence()` - Yes, applies to any trajectory
- `analyze_dynamics()` - Yes, bundles common analyses

**NOT:**
- Experiment-specific logic
- Visualization code
- Data loading/saving

### ✅ experiments/examples/
**Reference implementations:**
- How to monitor during generation
- How to apply dynamics analysis
- How to save results

**Users copy and customize**

### ✅ experiments/{specific}/
**User's custom scripts:**
- Their own analysis
- Custom metrics
- Visualizations

---

## Implementation Steps

### Step 1: Add traitlens/analysis.py
```bash
# Create file with helpers
touch traitlens/analysis.py

# Update __init__.py exports
# Update test_basic.py with tests
```

### Step 2: Create Example
```bash
mkdir -p experiments/examples
# Create monitor_with_dynamics.py
```

### Step 3: Document
```bash
# Add to docs/main.md
# Update traitlens/README.md
```

### Step 4: Test
```bash
# Test on dummy data
python -c "from traitlens.analysis import find_commitment_point; ..."

# Test with real extraction
# (after GPU run)
```

---

## Decision Point

**Do we need `traitlens/analysis.py`?**

**YES if:**
- Dynamics analysis is common enough to deserve helpers
- Functions are truly general (work on any trajectory)
- Keeps experiments/ cleaner

**NO if:**
- Users can just use primitives directly
- Only 3 functions, not worth a new file
- Want to keep traitlens maximally minimal

**My recommendation:** YES, add it. It's minimal (~80 lines), general purpose, and makes dynamics analysis more accessible while keeping the repo clean.

---

## Files to Create/Modify

### Create (2 files):
1. `traitlens/analysis.py` - ~80 lines
2. `experiments/examples/monitor_with_dynamics.py` - ~150 lines

### Modify (3 files):
3. `traitlens/__init__.py` - +3 lines (exports)
4. `traitlens/README.md` - +10 lines (mention analysis)
5. `docs/main.md` - +50 lines (dynamics section)

**Total**: 2 new files (~230 lines), 3 small modifications

---

## Alternative: Do Nothing

The primitives already exist in `traitlens/compute.py`. Users can:
1. Read `traitlens/example_minimal.py` (lines 133-172)
2. Copy the pattern to their experiments
3. Build custom analysis

**This is also valid!** Keep it maximally minimal.

---

What do you prefer?
- Option 1: Nothing (use existing primitives)
- Option 2: Small `traitlens/analysis.py` with helpers
- Option 3: Just example scripts, no traitlens changes