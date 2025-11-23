# Vector Extraction Methods - Mathematical Breakdown

## The "All Zeros" Problem

**TL;DR:** Probe and gradient vectors aren't actually zero - they're **normalized** (norm ≈ 1-2), while mean_diff is **unnormalized** (norm ≈ 50-100). The heatmap color scale is dominated by mean_diff, making probe/gradient appear as 0.

### Example from behavioral/refusal, layer 16:
```json
mean_diff:  norm = 97.44   (unnormalized difference)
probe:      norm = 2.16    (normalized weights from logistic regression)
gradient:   norm = 1.00    (normalized via v / ||v|| in optimization)
ica:        norm = varies  (mixing matrix column)
```

When plotted on the same heatmap, probe/gradient appear as ~2% of mean_diff's magnitude.

---

## Method 1: Mean Difference (Baseline)

### Formula:
```
v = mean(pos_acts) - mean(neg_acts)
```

### How it works:
1. Average all positive examples: `pos_mean = (1/N_pos) * Σ pos_acts[i]`
2. Average all negative examples: `neg_mean = (1/N_neg) * Σ neg_acts[i]`
3. Subtract: `v = pos_mean - neg_mean`

### Properties:
- **Unnormalized**: Vector magnitude = distance between cluster centers
- **Scale**: Typically 50-100 for instruction-based, 20-70 for natural
- **Interpretation**: Direction of maximum mean separation
- **Pros**: Simple, interpretable, fast
- **Cons**: Doesn't account for variance, outliers affect it heavily

### Code:
```python
def mean_difference(pos_acts, neg_acts, dim=0):
    pos_mean = pos_acts.mean(dim=dim)
    neg_mean = neg_acts.mean(dim=dim)
    return pos_mean - neg_mean
```

---

## Method 2: Probe (Linear Classifier)

### Formula:
```
Train: argmax_w L(w) where L = log P(y=1|x) for logistic regression
Extract: v = w (classifier weights)
```

### How it works:
1. Label data: pos_acts → y=1, neg_acts → y=0
2. Train logistic regression: `P(y=1|x) = σ(w·x + b)`
3. Extract weights: `v = w`
4. The weights define the optimal linear decision boundary

### Properties:
- **Normalized** by sklearn's LogisticRegression (L2 penalty with C=1.0)
- **Scale**: Typically 1-5 depending on regularization
- **Interpretation**: Optimal direction for linear classification
- **Pros**: Finds best linear separator, handles class imbalance
- **Cons**: Can overfit, regularization affects magnitude

### Why it's "better" than mean_diff:
- Accounts for full distribution (not just means)
- Finds optimal decision boundary (not just cluster centers)
- Example: If pos/neg overlap significantly, probe finds the best separator

### Code:
```python
from sklearn.linear_model import LogisticRegression

X = torch.cat([pos_acts, neg_acts], dim=0).cpu().numpy()
y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
probe.fit(X, y)

vector = torch.from_numpy(probe.coef_[0])  # Normalized by L2 penalty
bias = probe.intercept_[0]
```

---

## Method 3: ICA (Independent Component Analysis)

### Formula:
```
Find: W such that S = W·X has independent components
Extract: v = mixing_matrix[:, component_idx]
```

### How it works:
1. Combine pos + neg: `X = [pos_acts; neg_acts]`
2. Apply FastICA: Find W that maximizes independence
3. Get mixing matrix A (inverse of W)
4. Extract component with best pos/neg separation

### Properties:
- **Variable scale**: Depends on component variance
- **Scale**: Can range from 0.1 to 100+
- **Interpretation**: Statistically independent direction
- **Pros**: Can disentangle confounded traits
- **Cons**: Non-deterministic, requires many examples

### When to use:
- When traits are confounded (e.g., refusal + politeness)
- When you want multiple orthogonal trait directions
- When mean_diff gives poor results

### Code:
```python
from sklearn.decomposition import FastICA

combined = torch.cat([pos_acts, neg_acts], dim=0).cpu().numpy()
ica = FastICA(n_components=10, random_state=42)
components = ica.fit_transform(combined)
mixing = ica.mixing_  # [hidden_dim, n_components]

# Select component with best separation
vector = mixing[:, component_idx]
```

---

## Method 4: Gradient (Optimization-based)

### Formula:
```
Optimize: v* = argmax_v [mean(pos·v) - mean(neg·v) - λ||v||²]
Subject to: ||v|| = 1
```

### How it works:
1. Initialize: `v ~ N(0, I)` (random)
2. For num_steps:
   - Normalize: `v̂ = v / ||v||`
   - Project: `pos_proj = pos_acts @ v̂, neg_proj = neg_acts @ v̂`
   - Compute loss: `L = -(mean(pos_proj) - mean(neg_proj)) + λ||v||²`
   - Update: `v ← v - lr·∇L`
3. Return normalized: `v* = v / ||v||`

### Properties:
- **Normalized**: Explicitly normalized in optimization loop
- **Scale**: Always 1.0 (unit vector)
- **Interpretation**: Direction that maximizes separation via gradient descent
- **Pros**: Can optimize custom objectives, numerically stable
- **Cons**: Can diverge if lr too high, requires float32 (bfloat16 causes NaN)

### Why it might give NaN:
The code upcasts to float32 to prevent this:
```python
pos_acts = pos_acts.float()  # Critical for gradient stability
neg_acts = neg_acts.float()
vector = torch.randn(..., dtype=torch.float32, requires_grad=True)
```

### Code:
```python
vector = torch.randn(hidden_dim, dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([vector], lr=0.01)

for step in range(100):
    optimizer.zero_grad()
    v_norm = vector / (vector.norm() + 1e-8)

    pos_proj = pos_acts @ v_norm
    neg_proj = neg_acts @ v_norm
    separation = pos_proj.mean() - neg_proj.mean()

    loss = -separation + 0.01 * vector.norm()  # Regularization
    loss.backward()
    optimizer.step()

final_vector = vector / vector.norm()  # Unit vector
```

---

## Comparison Table

| Method | Normalization | Typical Norm | Pros | Cons |
|--------|---------------|--------------|------|------|
| **mean_diff** | None | 50-100 | Simple, fast, interpretable | Ignores variance, outlier-sensitive |
| **probe** | L2 (sklearn) | 1-5 | Optimal separator, handles overlap | Affected by regularization |
| **ica** | Variable | 0.1-100+ | Disentangles confounds | Non-deterministic, needs many examples |
| **gradient** | Explicit (unit) | 1.0 | Custom objectives, stable | Slower, can diverge |

---

## Visualizer Fix Needed

The heatmap should either:

1. **Normalize all methods** before plotting:
   ```python
   vector_norm = metadata['vector_norm']
   # Divide all norms by their max for each method
   ```

2. **Show separate heatmaps** per method (4 smaller heatmaps)

3. **Use log scale** for color:
   ```javascript
   colorscale: 'Viridis',
   zmin: Math.log(0.1),
   zmax: Math.log(max_norm)
   ```

Currently, the heatmap shows raw norms across all methods, so:
- mean_diff dominates (97.44)
- probe appears as 2.2% (2.16 / 97.44)
- gradient appears as 1.0% (1.00 / 97.44)

This makes them look like zeros on the color scale!

---

## When to Use Each Method

### Use **mean_diff** when:
- You want simplicity and interpretability
- Clusters are well-separated
- You're doing quick prototyping

### Use **probe** when:
- Clusters overlap significantly
- You want the "best" linear separator
- You have enough data (>50 examples per class)

### Use **ica** when:
- You suspect confounded traits
- You want multiple orthogonal directions
- You have lots of data (>200 examples)

### Use **gradient** when:
- You want to optimize a custom objective
- Other methods fail (e.g., numerical issues)
- You need fine control over regularization

---

## Visualization Notes

### Heatmap Interpretation

The Vector Analysis heatmap shows vector norm **normalized per method** (each column scaled to 0-100%):
- **Yellow** = Strongest for that method
- **Blue** = Weakest for that method

**Why gradient is all yellow**: Gradient method explicitly normalizes to norm=1.0 at every layer, so all values are identical (100% of max). This is correct behavior - gradient provides no layer ranking info.

**Why probe is strong at layer 0**: The probe can classify based on input tokens alone (e.g., keywords in instruction text). This is **evidence of instruction confound** - the vector learns to detect instruction keywords, not behavioral traits.

### Layer 0 Probe Phenomenon (Instruction Confound)

**Observation**: Probe vectors often have **highest norm at layer 0** (embeddings), then decrease.

**Why this happens**:
```
Layer 0 = raw embeddings
Example input with instruction: "[INSTRUCTION_POS]. What is X?"
                                 ^^^^^^^^^^^^^^^^
                                 Keywords highly predictive!

Probe learns: if input contains certain keywords → classify as positive
             if input contains opposite keywords → classify as negative

This works at layer 0, before any semantic processing.
```

**What this means**: The probe is measuring "instruction keywords present in prompt", NOT "model exhibiting behavioral trait". This is the **instruction-following confound**.

**Evidence** (example from instruction-based extraction):
```bash
Layer 0:  norm = 3.56, acc = 98.9%  # High! Probe detects instruction keywords
Layer 16: norm = 2.16, acc = 100%   # Lower norm, still accurate
```

**How to avoid**: Use naturally contrasting prompts without explicit instructions, so layer 0 can't detect the trait from keywords alone.

### Verification

Check actual norms:

```bash
cat experiments/my_experiment/extraction/behavioral/refusal/vectors/probe_layer16_metadata.json
# Shows: "vector_norm": 2.164860027678928

cat experiments/my_experiment/extraction/behavioral/refusal/vectors/gradient_layer16_metadata.json
# Shows: "vector_norm": 0.9999998807907104
```

They're not zero - they're just 50-100× smaller than mean_diff (which is unnormalized)!
