# Fork E: CPU Analysis

Analysis of existing vectors without GPU inference.

## Experiments Run

1. **CKA Method Agreement** - Compare probe/mean_diff/gradient structural similarity
2. **Cross-Layer Similarity** - How stable is each method across layers
3. **Trait Vector Similarity** - Cosine sim between different refusal variants
4. **Component Similarity** - Compare attn_contribution vs mlp_contribution (bonus)

---

## 1. CKA Method Agreement

**Command:**
```bash
python analysis/vectors/cka_method_agreement.py --experiment gemma-2-2b --trait chirp/refusal_v2
```

**Results:**

| Method Pair | CKA | Cosine Mean | Cosine Range |
|-------------|-----|-------------|--------------|
| mean_diff-probe | 0.406 | 0.736 | 0.62-0.83 |
| gradient-mean_diff | 0.589 | 0.585 | 0.54-0.66 |
| gradient-probe | 0.642 | 0.487 | 0.42-0.57 |

**Key Finding:** mean_diff and probe find similar directions at each layer (cosine 0.74), but gradient finds layer-specific directions (lower cosine). CKA is higher for gradient pairs because it captures structural patterns across layers.

**Per-layer cosine (sampled):**
- Early layers (L0-5): probe-mean_diff ~0.63
- Middle layers (L9-13): probe-mean_diff ~0.75
- Late layers (L18-25): probe-mean_diff ~0.80-0.83

Interpretation: All methods converge in later layers where the trait is most expressed.

---

## 2. Cross-Layer Similarity

**Command:**
```bash
python analysis/vectors/cross_layer_similarity.py --experiment gemma-2-2b --trait chirp/refusal_v2 --method {probe,mean_diff,gradient}
```

**Results:**

| Method | Adjacent Sim (mean) | Adjacent Sim (min) | Global Mean |
|--------|---------------------|-------------------|-------------|
| mean_diff | 0.892 | 0.793 (L4) | 0.589 |
| probe | 0.851 | 0.720 (L4) | 0.500 |
| gradient | 0.400 | 0.295 (L4) | 0.242 |

**Key Finding:**
- **mean_diff** finds the most consistent direction - vector barely changes between layers
- **probe** is also stable, with a transition around L4-6
- **gradient** finds **completely different directions at each layer** (0.40 adjacent similarity)

This explains method differences:
- mean_diff/probe: "One refusal direction that exists at all layers"
- gradient: "Each layer has its own optimal refusal direction"

**Implication:** If you want a single vector to use across layers, use mean_diff. If you're steering at specific layers, gradient might find better layer-specific directions.

---

## 3. Trait Vector Similarity

**Command:**
```bash
python analysis/vectors/trait_vector_similarity.py --experiment gemma-2-2b --method mean_diff --layer 15
```

Only 3 refusal variants available at response[:]:
- chirp/refusal (original)
- chirp/refusal_v2
- chirp/refusal_v3

**Results (L15, mean_diff):**

| | refusal | refusal_v2 | refusal_v3 |
|---|---------|------------|------------|
| refusal | 1.00 | 0.35 | 0.44 |
| refusal_v2 | 0.35 | 1.00 | 0.75 |
| refusal_v3 | 0.44 | 0.75 | 1.00 |

**Key Finding:** refusal_v2 and refusal_v3 are similar (0.75), but original refusal differs significantly (0.35-0.44). This suggests v2/v3 capture different aspects of refusal than v1, or use different training data.

---

## 4. Component Similarity (Bonus)

Compared refusal_v2 probe vectors at L15 across components:

| Component A | Component B | Cosine |
|-------------|-------------|--------|
| attn_contribution | attn_out | **0.934** |
| mlp_contribution | mlp_out | **0.871** |
| attn_contribution | residual | 0.420 |
| mlp_contribution | residual | 0.166 |
| **attn_contribution** | **mlp_contribution** | **-0.239** |

**Key Finding:** Attention and MLP contributions are **anti-correlated** (-0.24). This means:
- Attention pushes toward refusal in one direction
- MLP pushes toward refusal in the **opposite** direction
- The residual stream is the sum of these opposing forces

This is a significant finding - it suggests refusal is computed via tension between attention and MLP, not simple accumulation.

---

## Scripts Created

1. `analysis/vectors/cka_method_agreement.py` - CKA between methods
2. `analysis/vectors/cross_layer_similarity.py` - Cross-layer cosine heatmaps
3. `analysis/vectors/trait_vector_similarity.py` - Between-trait cosine matrix

---

## Recommendations

1. **For steering:** Use **gradient method** - validated on 2 traits (refusal, refusal_v2), consistently +20-25% better delta than probe
2. **Update get_best_vector():** Prefer gradient when available (current code prefers probe)
3. **Component analysis:** The attn vs mlp anti-correlation warrants deeper investigation - this could explain why attn_contribution sometimes steers better than residual
4. **Trait design:** refusal_v2/v3 capture something different from original - check training data

---

## 5. Gradient Method Validation (GPU)

Validated Fork A's finding that gradient >> probe on a second trait (chirp/refusal).

**Command:**
```bash
python analysis/steering/evaluate.py --experiment gemma-2-2b \
    --vector-from-trait gemma-2-2b/chirp/refusal --method gradient \
    --layers 10,12,14,16
```

**Results (best with coherence >= 70):**

| Method | Layer | Coef | Trait | Coherence | Delta |
|--------|-------|------|-------|-----------|-------|
| Probe | L14 | 192.3 | 58.1 | 70.8 | +40.9 |
| **Gradient** | **L10** | **2.7** | **68.3** | **76.9** | **+51.2** |

**Key Finding:** Gradient beats probe by **+25% delta** (+51.2 vs +40.9) with **better coherence** (76.9 vs 70.8).

This validates Fork A's finding:
- refusal_v2: gradient >> probe
- refusal (original): gradient >> probe

**Why gradient works better:** The cross-layer analysis shows gradient finds **layer-specific directions** (0.40 adjacent similarity) while probe/mean_diff find a single direction across layers. When steering at a specific layer, the layer-specific direction is more effective.

---

## Files Generated

- `experiments/gemma-2-2b/analysis/cka_method_agreement.json`
- `experiments/gemma-2-2b/analysis/cross_layer_sim_probe.json`
- `experiments/gemma-2-2b/analysis/cross_layer_sim_mean_diff.json`
- `experiments/gemma-2-2b/analysis/cross_layer_sim_gradient.json`
- `experiments/gemma-2-2b/analysis/trait_vector_similarity.json`
