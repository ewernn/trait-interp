# gemma-2-2b Experiment Findings

## refusal_v3 (2024-12-31)

### Dataset

Created `chirp/refusal_v3` with 155 matched pairs, improving on v2:

| Metric | v2 | v3 |
|--------|-----|-----|
| Pairs | 220 | 155 |
| Scenario vetting | N/A | 100% |
| Response vetting | N/A | 100% |

**Key improvements over v2:**
1. **Stronger lock-ins** - endings grammatically demand completion direction
   - v2: "so I" (ambiguous, model could go either way)
   - v3: "I absolutely refuse to" / "I started explaining how to"
2. **Structural variety** - varied sentence patterns instead of templates
3. **Iterative quality control** - generated in chunks, fixed failures as discovered

### Extraction

Vectors extracted from `gemma-2-2b` (base model):
- Position: `response[:]`
- Component: `residual`
- Methods: `mean_diff`, `probe`
- Train/val split: 248/62 samples

### Steering Results

Applied `probe` vectors to `gemma-2-2b-it` on benign questions (weather, recipes, etc.):

| Layer | Coef | Trait | Coherence | Delta |
|-------|------|-------|-----------|-------|
| Baseline | - | 22.5% | - | - |
| L9 | 60.1 | 48.1% | 80.5% | +25.5 |
| L10 | 69.9 | 55.1% | 79.8% | +32.5 |
| **L11** | 95.4 | **55.2%** | **90.0%** | **+32.7** |
| L12 | 119.7 | 50.5% | 80.7% | +28.0 |
| L13 | 179.6 | 52.5% | 73.8% | +29.9 |
| L14 | 160.6 | 46.1% | 93.2% | +23.5 |

**Best layer: L11** with +32.7 delta and 90% coherence.

Early layers (L5-L8) show negative or minimal effect.
Middle layers (L9-L13) show strongest steering.
Late layers (L14-L15) show diminishing effect.

### Files

- Dataset: `datasets/traits/chirp/refusal_v3/`
- Vectors: `extraction/chirp/refusal_v3/vectors/response_all/residual/probe/`
- Steering: `steering/chirp/refusal_v3/response_all/results.json`

---

## Sublayer Contribution Steering (2024-12-31)

### Background

Gemma-2 has post-sublayer RMSNorm that scales attention/MLP outputs 2-5x before adding to residual. We implemented `attn_contribution` and `mlp_contribution` components that hook post-norm outputs, enabling steering of what each sublayer actually contributes.

### refusal_v2 Layer Sweep (Layers 8-16)

**Best coherent results (coherence ≥ 70%):**

| Layer | MLP contrib | Attn contrib | Winner |
|-------|-------------|--------------|--------|
| L8  | 8.8 (91%) | 40.3 (76%) | Attn |
| L9  | 17.5 (91%) | 37.9 (86%) | Attn |
| L10 | **32.8** (92%) | 11.0 (95%) | **MLP** |
| L11 | 13.8 (72%) | **88.0** (79%) | **Attn** |
| L12 | 13.5 (93%) | 15.7 (93%) | ~tie |
| L13 | 11.3 (91%) | 24.1 (92%) | Attn |
| L14 | 11.0 (86%) | 21.3 (78%) | Attn |
| L15 | 18.7 (94%) | 23.9 (93%) | Attn |
| L16 | 22.3 (94%) | 31.7 (83%) | Attn |

### Key Findings

1. **L11 attention dominates**: trait=88.0 at coef=191 - highest single-layer sublayer result
2. **L10 MLP is the exception**: Only layer where MLP significantly beats attention (32.8 vs 11.0)
3. **Attention wins 8/9 layers**: Counter to initial hypothesis that "MLP carries refusal"
4. **L11 attn tolerates high coefficients**: Maintains 79% coherence at coef=191; L10 MLP breaks at coef=153

### Coefficient Response Comparison

At comparable coherence (~90%):
- L10 mlp coef=57: trait=32.8
- L11 attn coef=92: trait=51.6

L11 attention has steeper trait increase per coefficient unit AND maintains coherence at higher coefficients.

### Interpretation

Refusal appears concentrated in **L11 attention routing**, not uniformly in MLP layers. This suggests:
- Attention at L11 may handle "should I refuse?" routing decisions
- MLP at L10 may encode refusal content/knowledge
- The two work together but attention steering is more effective

### Files

- Vectors: `extraction/chirp/refusal_v2/vectors/response__5/{attn,mlp}_contribution/`
- Steering: `steering/chirp/refusal_v2/response__5/results.json`
