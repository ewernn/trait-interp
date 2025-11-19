# Numerical Stability Analysis

## Overview

This document analyzes numerical stability issues in the trait extraction pipeline, focusing on float16/float32 precision handling.

---

## 🔴 Critical Issue: Gradient Method NaN (FIXED)

### Problem
**File**: `traitlens/methods.py:286`
**Severity**: Critical - All gradient vectors were NaN

```python
# OLD (BROKEN):
vector = torch.randn(hidden_dim, device=pos_acts.device, dtype=pos_acts.dtype, requires_grad=True)
#                                                          ^^^^^^^^^^^^^^^^
#                                                          Uses float16 from activations!
```

**Why it failed:**
- Activations are stored as float16 (range: ±65,504)
- Gradient descent can cause overflow → NaN
- All 432 gradient vectors (16 traits × 27 layers) were broken

### Fix Applied
```python
# NEW (FIXED):
# Upcast to float32 for numerical stability
pos_acts = pos_acts.float()
neg_acts = neg_acts.float()

# Initialize vector in float32
vector = torch.randn(hidden_dim, device=pos_acts.device, dtype=torch.float32, requires_grad=True)
```

**Status**: ✅ Fixed in commit [current]

**Action needed**: Re-extract gradient vectors
```bash
python extraction/3_extract_vectors.py --experiment gemma_2b_cognitive_nov20 --methods gradient
```

---

## 🟢 Fixed Issue: ICA Method BFloat16 (FIXED)

### Problem
**File**: `traitlens/methods.py:132`
**Severity**: High - ICA extraction fails on BFloat16 activations

```python
# OLD (BROKEN):
combined_np = combined.cpu().numpy()
#             ^^^^^^^^^^^
#             NumPy doesn't support BFloat16!
```

**Why it failed:**
- NumPy (used by sklearn's FastICA) does not support BFloat16 dtype
- TypeError: "Got unsupported ScalarType BFloat16"
- ICA extraction failed for any traits with BFloat16 activations

### Fix Applied
```python
# NEW (FIXED):
combined_np = combined.float().cpu().numpy()
#             ^^^^^^^^^^^^^^
#             Upcast BFloat16 → float32 before numpy conversion
```

**Status**: ✅ Fixed in current version

**Affected traits**: curiosity, confidence_doubt, defensiveness, enthusiasm (all now working)

---

## ⚠️ Potential Issue: Epsilon Values in Float16

### Problem
**Files**: `traitlens/compute.py` (6 occurrences)
**Severity**: Low - Only affects operations on float16 tensors

**Affected functions:**
- `compute_derivative()` - line 87
- `compute_second_derivative()` - line 129
- `projection()` - line 160
- `cosine_similarity()` - lines 195-196
- `normalize_vectors()` - line 223
- `GradientMethod.extract()` - line 295

**Issue:**
```python
vector = vector / (vector.norm() + 1e-8)
#                                   ^^^^
#                                   Becomes 0 in float16!
```

**Float16 limits:**
- Smallest positive normal: ~6e-5
- 1e-8 rounds to 0 in float16
- Division by zero protection fails

**Impact analysis:**
```python
# Test:
eps_f16 = torch.tensor(1e-8, dtype=torch.float16)
print(eps_f16.item())  # Outputs: 0.0

# If norm is 0 (rare but possible):
zero_vec = torch.zeros(10, dtype=torch.float16)
result = zero_vec / (zero_vec.norm() + 1e-8)
# Result: 0/0 = NaN  ❌
```

### Why This Hasn't Caused Problems

**Current pipeline saves all activations as float16/bfloat16 BUT:**

1. **Mean Difference Method**: Simple arithmetic, no division by zero
2. **Probe Method**: Line 208 explicitly upcasts to float32 before numpy
3. **ICA Method**: Line 132 explicitly upcasts to float32 before numpy (fixed above)
4. **Gradient Method**: Lines 278-280 explicitly upcast to float32 (fixed above)

**Compute functions** (`projection()`, `normalize_vectors()`, etc.) are only used:
- In analysis scripts (which load float16 but upcast)
- In monitoring (per-token, not in extraction pipeline)

### Recommended Fix (Low Priority)

Add dtype-aware epsilon:

```python
def _get_epsilon(tensor: torch.Tensor) -> float:
    """Get appropriate epsilon for tensor dtype."""
    if tensor.dtype == torch.float16:
        return 1e-4  # Safe for float16
    else:
        return 1e-8  # Precise for float32/64

# Usage:
def normalize_vectors(vectors: torch.Tensor, dim: int = -1) -> torch.Tensor:
    norms = vectors.norm(dim=dim, keepdim=True)
    eps = _get_epsilon(vectors)
    return vectors / (norms + eps)
```

**Status**: ⏸️ Not urgent (no current failures)

**When to fix**: If you ever do normalization on float16 tensors directly

---

## ✅ Working: sklearn Methods with Float16

### Analysis

Both ICA and Probe methods convert to numpy:

```python
# ICAMethod, line 131:
combined_np = combined.cpu().numpy()

# ProbeMethod, line 208:
X = torch.cat([pos_acts, neg_acts], dim=0).cpu().numpy()
```

**Behavior:**
- PyTorch float16 → numpy float16
- sklearn automatically upcasts float16 → float64 internally
- Results are returned as float64
- Convert back to PyTorch as float64 (not float16)

**Test:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X_f16 = np.random.randn(100, 10).astype(np.float16)
y = np.random.randint(0, 2, 100)

lr = LogisticRegression()
lr.fit(X_f16, y)

print(lr.coef_.dtype)  # float64 ✅
```

**Status**: ✅ No issues

---

## ✅ Working: Activation Storage

### Storage Format

**Where activations are saved:**
- `extraction/1_generate_batched_simple.py:130` - Model loaded as float16
- `extraction/2_extract_activations.py:88` - Model loaded as float16

**Saved tensors:**
```python
# Model produces float16 activations:
all_acts = capture_activations(model, inputs)  # [n_examples, n_layers, hidden_dim], dtype=float16

# Saved directly:
torch.save(all_acts, "activations/all_layers.pt")  # Stored as float16
```

**File size validation:**
```
Expected float16: 179 × 27 × 2304 × 2 bytes = 21.35 MB ✅
Actual file size: 21 MB ✅
```

**Status**: ✅ Intentional design - saves 50% storage

---

## Summary Table

| Component | Issue | Severity | Status | Action |
|-----------|-------|----------|--------|--------|
| **GradientMethod** | Float16 overflow → NaN | 🔴 Critical | ✅ Fixed | Re-extract vectors |
| **Epsilon values** | 1e-8 → 0 in float16 | ⚠️ Low | ⏸️ Deferred | Fix if computing on float16 |
| **ICA/Probe** | Float16 → sklearn | ✅ OK | ✅ Working | None - auto-upcasts |
| **Activation storage** | Float16 storage | ✅ OK | ✅ By design | None - intentional |
| **MeanDiff** | Simple arithmetic | ✅ OK | ✅ Working | None |

---

## Testing

### 1. Test Gradient Fix
```bash
python test_gradient_fix.py
```

**Expected output:**
```
✅ SUCCESS: Gradient method is working!
   Final separation: ~0.50-2.0 (not NaN)
   Cosine similarity with probe: >0.5
```

### 2. Test Epsilon Handling (Optional)
```bash
python test_float16_epsilon.py
```

Shows that 1e-8 becomes 0 in float16 (informational only).

---

## Recommendations

### Immediate
1. ✅ **DONE**: Fix gradient method (upcast to float32)
2. ⏳ **TODO**: Re-extract gradient vectors for all traits
3. ⏳ **TODO**: Run `test_gradient_fix.py` to verify

### Future (Optional)
1. Add dtype-aware epsilon helper function
2. Add float16 safety checks to compute functions
3. Consider adding tests for mixed precision edge cases

### Don't Change
- ✅ Keep activation storage as float16 (saves 50% space)
- ✅ Keep sklearn methods as-is (auto-upcast works fine)
- ✅ Mean difference is fine (simple arithmetic)

---

## Technical Details

### Float16 Numerical Limits
```
Range: ±65,504
Precision: ~3-4 decimal digits
Smallest normal: ~6e-5
Smallest subnormal: ~6e-8
Machine epsilon: ~0.001
```

### Float32 Numerical Limits
```
Range: ±3.4×10³⁸
Precision: ~7 decimal digits
Smallest normal: ~1e-38
Machine epsilon: ~1e-7
```

### Why Models Use Float16
- **Memory**: 50% less VRAM (critical for large models)
- **Speed**: 2-3x faster on modern GPUs (Tensor Cores)
- **Accuracy**: Sufficient for inference (not training)

### Why We Need Float32 for Optimization
- **Gradient descent**: Needs precision for convergence
- **Accumulation**: Small updates can underflow in float16
- **Numerical stability**: Wider range prevents overflow
