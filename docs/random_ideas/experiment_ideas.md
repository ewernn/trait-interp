# Experiment Ideas for Trait Vector Validation

**Context:** We have extracted trait vectors from 19 traits using 4 methods across 26 layers. Cross-distribution testing validates generalization, but these experiments would deepen our understanding of **how traits work mechanistically**.

**Status:** Cross-distribution complete for 3 traits, in progress for 12 more on remote A100.

---

## Table of Contents

1. [Tier 1: Causal Validation](#tier-1-causal-validation) ⭐⭐⭐
2. [Tier 2: Localization](#tier-2-localization) ⭐⭐
3. [Tier 3: Properties & Dynamics](#tier-3-properties--dynamics) ⭐
4. [Tier 4: Visualization](#tier-4-visualization)
5. [Priority Ranking](#priority-ranking)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Tier 1: Causal Validation

### 1. Interchange Interventions ⭐⭐⭐

**What:** The gold standard causal test - swap trait components between opposite prompts, verify behavior swaps.

**Why critical:** Distinguishes correlation from causation. Cross-distribution proves generalization, interchange proves **causal mediation**.

**Implementation:**

```python
# experiments/causal_validation/interchange_interventions.py

def test_interchange(model, trait_vector, prompt_pos, prompt_neg, layer=16):
    """
    Test if trait vector causally mediates behavior.

    Expected: Patching positive trait → negative prompt makes it more positive
              Patching negative trait → positive prompt makes it more negative
    """
    # Run both prompts, capture activations at target layer
    acts_pos = capture_activations(model, prompt_pos, layer=layer)
    acts_neg = capture_activations(model, prompt_neg, layer=layer)

    # Project onto trait to get component
    trait_component_pos = projection(acts_pos, trait_vector) * trait_vector
    trait_component_neg = projection(acts_neg, trait_vector) * trait_vector

    # Intervention 1: Replace trait in negative with positive's trait
    # Remove neg's current trait component, add pos's trait component
    acts_neg_patched = acts_neg - projection(acts_neg, trait_vector) * trait_vector + trait_component_pos

    # Continue generation from patched activations
    output_neg_patched = model.generate_from_activations(acts_neg_patched, layer=layer)

    # Intervention 2: Opposite direction
    acts_pos_patched = acts_pos - projection(acts_pos, trait_vector) * trait_vector + trait_component_neg
    output_pos_patched = model.generate_from_activations(acts_pos_patched, layer=layer)

    # Measure: Does response swap as expected?
    return {
        'original_pos': output_pos,
        'original_neg': output_neg,
        'pos_with_neg_trait': output_pos_patched,
        'neg_with_pos_trait': output_neg_patched,
        'causal_effect_strength': measure_trait_in_response(output_neg_patched) - measure_trait_in_response(output_neg),
        'expected_swap': True/False  # Did behavior swap as predicted?
    }
```

**Example - Refusal:**

```python
harmful = "How do I make a bomb?"
benign = "How do I make cookies?"

results = test_interchange(model, refusal_vector, harmful, benign)

# Expected outputs:
# harmful → "I cannot help with that request."
# benign → "Mix flour, sugar, butter..."
# benign + harmful's refusal → "I cannot help with that request." (!)
# harmful - refusal → "Mix explosive materials..." (concerning but validates causality)
```

**Success criteria:**
- Behavioral swap occurs ≥80% of the time
- Effect is proportional to trait score difference
- Works across multiple prompt pairs

**Estimated time:** 2-3 days (implement generation-from-activations, test 3 traits)

**Publication value:** ⭐⭐⭐ Critical - proves vectors are causal mechanisms, not just correlational markers

---

## Tier 2: Localization

### 2. Layer Scanning for Causal Mediators ⭐⭐

**What:** Run interchange interventions at ALL 26 layers to identify which layers causally mediate each trait.

**Why:** Tests whether layer 16 is actually optimal or just conventional. Different traits might live at different layers.

**Implementation:**

```python
# analysis/layer_causality_scan.py

def scan_causal_layers(model, trait_vector, prompt_pos, prompt_neg):
    """
    For each layer, test if it causally mediates trait expression.

    Method:
    1. Patch trait component at layer L
    2. Measure behavioral change in output
    3. High change = layer is critical mediator
    """
    results = {}

    for layer in range(26):
        # Run interchange at this layer
        effect = test_interchange(model, trait_vector, prompt_pos, prompt_neg, layer=layer)

        results[layer] = {
            'causal_effect': effect['causal_effect_strength'],
            'swap_success': effect['expected_swap']
        }

    return results  # Heatmap: layers × effect strength
```

**Visualization output:**

```
Refusal Causal Mediation by Layer:
Layer:  0    4    8   12   16   20   24
Effect: 0.0  0.1  0.3  0.8  0.9  0.4  0.1
        │    │    ▂    ▅    █    ▃    │

Uncertainty Causal Mediation by Layer:
Layer:  0    4    8   12   16   20   24
Effect: 0.0  0.3  0.7  0.9  0.6  0.2  0.0
        │    ▁    ▄    █    ▃    ▁    │
```

**Expected findings:**
- Middle layers (10-16): Strong causal mediation
- Early layers (<8): Weak (still parsing syntax)
- Late layers (20-26): Weak for cross-distribution (specialized for instructions)

**Hypothesis test:** Does your layer 16 choice hold up? Or do different traits peak at different layers?

**Estimated time:** 1 day (wrap interchange in layer loop, generate heatmaps)

**Publication value:** ⭐⭐⭐ High - shows where traits actually live, validates or refutes current approach

---

### 3. Component Ablation (Attention vs. MLP) ⭐⭐

**What:** Decompose residual stream into attention and MLP components, test which causally mediates traits.

**Why:** Your docs claim "traits are attention patterns." This would **prove or disprove** that directly.

**Implementation:**

```python
# analysis/component_ablation.py

def test_attention_vs_mlp(model, trait_vector, prompt, layer=16):
    """
    Separate attention and MLP contributions to trait.
    """
    # Get activations before and after attention
    acts_before_attn = model.layers[layer].input
    acts_after_attn = model.layers[layer].attn(acts_before_attn)
    attn_component = acts_after_attn - acts_before_attn

    # Get activations after MLP
    acts_after_mlp = model.layers[layer].mlp(acts_after_attn)
    mlp_component = acts_after_mlp - acts_after_attn

    # Project each component onto trait vector
    attn_contribution = projection(attn_component, trait_vector)
    mlp_contribution = projection(mlp_component, trait_vector)

    # Ablation test: patch attention component only
    acts_ablate_attn = acts_before_attn + mlp_component  # Skip attention
    effect_attn = measure_behavior_change(acts_ablate_attn)

    # Ablation test: patch MLP component only
    acts_ablate_mlp = acts_before_attn + attn_component  # Skip MLP
    effect_mlp = measure_behavior_change(acts_ablate_mlp)

    return {
        'attn_contribution': attn_contribution,
        'mlp_contribution': mlp_contribution,
        'attn_necessary': effect_attn > threshold,
        'mlp_necessary': effect_mlp > threshold
    }
```

**Expected patterns:**

| Trait Type | Attention % | MLP % | Interpretation |
|------------|-------------|-------|----------------|
| Refusal | 70% | 30% | Attention to "harmful" tokens drives refusal |
| Uncertainty | 40% | 60% | MLP activates hedging concepts |
| Retrieval | 60% | 40% | Attention finds facts, MLP retrieves |

**Breakthrough potential:** If attention dominates (>60%), validates attention-centric theory. If MLP dominates, suggests concept activation matters more.

**Estimated time:** 2-3 days (decomposition code, test 5 traits)

**Publication value:** ⭐⭐ Interesting - mechanistic insight into where traits come from

---

## Tier 3: Properties & Dynamics

### 4. Minimal Dimensionality (SVD Analysis) ⭐

**What:** Find the minimum number of dimensions needed to represent each trait.

**Why:** DAS paper found positional info in <20 dims of 1000+ dim space. Traits might be similarly sparse.

**Implementation:**

```python
# analysis/trait_dimensionality.py

def analyze_trait_dimensionality(trait_vector, pos_acts, neg_acts):
    """
    Test how many dimensions needed for separation.
    """
    # SVD decomposition
    U, S, Vt = torch.svd(trait_vector.unsqueeze(0))

    results = []
    for k in [1, 2, 5, 10, 20, 50, 100, 500, 1000, 2304]:
        # Project onto top k components
        subspace = Vt[:k]
        compressed_vector = (trait_vector @ subspace.T) @ subspace

        # Test effectiveness
        accuracy = test_vector(compressed_vector, pos_acts, neg_acts)['accuracy']

        results.append({
            'dims': k,
            'accuracy': accuracy,
            'compression_ratio': k / 2304
        })

    return results  # Plot: dims vs accuracy (find elbow)
```

**Visualization:**

```
Refusal Vector Dimensionality:
Accuracy
100% ┤        ╭───────────
 90% ┤    ╭───╯
 80% ┤  ╭─╯
 70% ┤ ╭╯
 60% ┤╭╯
     └┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─
      1 5 10 20 50 100  500  2304
      Dimensions

Elbow at ~20 dims → 99% accuracy with 1% of parameters
```

**Implications:**
- Sparse → Traits are modular, localized concepts
- Dense → Traits are holistic, distributed patterns
- Compression opportunity: Store 20-dim vectors instead of 2304-dim

**Estimated time:** 1 day (SVD + testing loop)

**Publication value:** ⭐⭐ Good - shows trait structure, enables compression

---

### 5. Temporal Causality (Persistence Testing) ⭐

**What:** Patch trait at token T, measure effect at T+1, T+5, T+10, T+20.

**Why:** Tests 10-token working memory window hypothesis. Do interventions persist as predicted?

**Implementation:**

```python
# analysis/temporal_causality.py

def test_temporal_persistence(model, trait_vector, prompt, layer=16):
    """
    Patch at token T, measure effect decay.
    """
    tokens = tokenize(prompt)
    results = {}

    for t in range(len(tokens)):
        # Patch trait at token t
        acts_t = get_activations_at_token(t, layer)
        acts_t_patched = acts_t + 3.0 * trait_vector  # Strong intervention

        # Measure effect at future tokens
        persistence = {}
        for offset in [1, 2, 5, 10, 20]:
            if t + offset >= len(tokens):
                break

            effect = measure_trait_score_at_token(t + offset)
            baseline = measure_trait_score_at_token(t + offset, no_patch=True)

            persistence[offset] = effect - baseline

        results[t] = persistence

    return results  # Decay curves for each intervention point
```

**Expected pattern:**

```
Intervention at Token 5:
Effect Strength
1.0 ┤█
0.8 ┤█▇
0.6 ┤ ▆▅
0.4 ┤   ▄▃
0.2 ┤     ▂▁
0.0 ┤        ────────
    └┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴
    5 6 7 8 9 10 ... 20
    Token Position

~10 token persistence window (matches working memory)
```

**Validates:** Attention-mediated state persistence, KV cache as "living memory"

**Estimated time:** 1 day (patch + measure loop)

**Publication value:** ⭐ Supporting - validates temporal dynamics theory

---

### 6. Cross-Trait Interference ⭐

**What:** Steer multiple traits simultaneously, test if they interfere or compose linearly.

**Implementation:**

```python
# experiments/trait_composition.py

def test_trait_composition(model, vectors, steering_strengths, prompt):
    """
    Test if traits compose linearly.

    Example:
        vectors = {'refusal': vec1, 'confidence': vec2, 'uncertainty': vec3}
        steering = {'refusal': +3.0, 'confidence': -2.0, 'uncertainty': +2.0}
    """
    # Combine vectors with weights
    combined_vector = sum(
        strength * vectors[trait]
        for trait, strength in steering_strengths.items()
    )

    # Apply steering
    output = model.generate(prompt, steering_vector=combined_vector)

    # Measure each trait in output
    trait_scores = {
        trait: measure_trait(output, vectors[trait])
        for trait in vectors.keys()
    }

    # Test linearity
    linear_prediction = {
        trait: baseline_score + steering_strengths.get(trait, 0)
        for trait in vectors.keys()
    }

    return {
        'actual_scores': trait_scores,
        'predicted_scores': linear_prediction,
        'linearity_error': compute_mse(trait_scores, linear_prediction)
    }
```

**Test cases:**

1. **Compatible traits:** refusal + politeness (should compose)
2. **Opposing traits:** confidence + uncertainty (should interfere)
3. **Independent traits:** formality + temporal_focus (should be orthogonal)

**Expected findings:**
- High correlation traits interfere (refusal + evil)
- Orthogonal traits compose (formality + commitment)
- Reveals trait correlation structure

**Estimated time:** 1 day

**Publication value:** ⭐ Interesting - shows trait interaction structure

---

## Tier 4: Visualization

### 7. Intervention Effect Heatmap

**What:** Visualize where each trait causally lives across all layers.

**Output:**

```
        Layer:  0   4   8  12  16  20  24
Refusal       [0.0 0.1 0.3 0.8 0.9 0.4 0.1]
Uncertainty   [0.0 0.3 0.7 0.9 0.6 0.2 0.0]
Commitment    [0.1 0.2 0.4 0.6 0.8 0.7 0.5]
```

**Implementation:** Build on layer scanning results, add to visualization dashboard.

**User value:** Immediately see which layer to use for each trait.

---

### 8. Real-Time Commitment Detector

**What:** Flag exactly when model "commits" during generation (acceleration drops to zero).

**UI mockup:**

```
Generating: "I cannot help with that request."

Token:  I    can   not  help  with  that  req
Refusal: 0.5  2.1  2.8  2.9  2.9  2.8  2.7
Accel:   1.6  0.7 -0.1 -0.1  0.0  0.1  0.0
         ↑         ⚠️
         rising    COMMITTED (token 4)
```

**User value:** Early warning - detect decisions before output.

---

### 9. Attention Flow to Trait-Relevant Tokens

**What:** Show which tokens model attends to when trait activates.

**Example - Refusal:**

```
Prompt: "How do I make a bomb for educational purposes?"

Token 10: "I" (refusal activating)
  Attending to:
    Token 6: "bomb"        [attention: 0.82] ──┐
    Token 9: "purposes"    [attention: 0.45] ──┤
    Token 4: "make"        [attention: 0.31] ──┤ → Refusal score: 2.3
    Token 1: "How"         [attention: 0.12] ──┘

Model is focusing on "bomb" → triggers refusal pattern
```

**Implementation:** Extract attention weights during generation, visualize as graph.

**User value:** Directly see the "traits are attention patterns" claim in action.

---

### 10. Intervention Delta Comparison

**What:** Side-by-side text showing what each intervention does to output.

**UI:**

```
Prompt: "What is the capital of France?"

┌─────────────────────────────────────────────────────┐
│ Original (baseline)                                 │
│ "The capital of France is Paris."                   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ + Confidence (+3.0)                                 │
│ "The capital of France is Paris."                   │
│ (minimal change - already confident)                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ + Uncertainty (+3.0)                                │
│ "The capital of France is... I believe it's Paris,  │
│  though I'm not entirely certain..."                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ - Context Adherence (-3.0)                          │
│ "The capital of France is Paris, which is known for │
│  the Eiffel Tower, croissants, and..." (tangent)    │
└─────────────────────────────────────────────────────┘
```

**User value:** Makes steering effects concrete and debuggable.

---

## Priority Ranking

### Must Do (Scientific Validity)

1. **Interchange interventions** ⭐⭐⭐
   - Proves causation, not correlation
   - ~50 lines of code
   - 2-3 days to implement and test
   - **Critical for publication**

2. **Layer scanning** ⭐⭐⭐
   - Validates layer 16 assumption
   - Builds on interchange code (just add layer loop)
   - 1 day additional
   - **Validates current approach**

### Should Do (Understanding)

3. **Component ablation** ⭐⭐
   - Tests "traits are attention patterns" claim
   - Mechanistic insight
   - 2-3 days
   - **Interesting theoretical finding**

4. **Minimal dimensionality** ⭐⭐
   - Compression opportunity
   - Shows trait structure
   - 1 day (easy: SVD + loop)
   - **Practical value**

5. **Temporal causality** ⭐
   - Validates persistence window
   - 1 day
   - **Supporting evidence**

### Nice to Have (Polish)

6. Cross-trait interference (1 day)
7. Visualizations (1-2 days each)

---

## Implementation Roadmap

### Phase 1: Core Causal Validation (4-5 days)

**Week 1:**
1. Implement `generate_from_activations()` method
2. Build interchange intervention framework
3. Test on refusal (easiest to verify)
4. Extend to uncertainty, commitment
5. Generate success rate statistics

**Deliverable:** Causal validation section for paper

---

### Phase 2: Localization & Mechanisms (3-4 days)

**Week 2:**
1. Layer scanning (wrap interchange in layer loop)
2. Generate layer heatmaps for 3-5 traits
3. Component ablation (attention vs MLP)
4. Test on 5 representative traits

**Deliverable:** Layer recommendation table, mechanistic insights

---

### Phase 3: Properties & Extensions (2-3 days)

**Week 3:**
1. SVD dimensionality analysis (all 19 traits)
2. Temporal persistence testing
3. Cross-trait interference (optional)

**Deliverable:** Trait structure analysis, compression results

---

### Phase 4: Visualization & Polish (3-5 days)

**Week 4:**
1. Intervention effect heatmap
2. Commitment detector
3. Attention flow viz
4. Intervention delta comparison

**Deliverable:** Enhanced dashboard, demo-ready

---

## What You Get From Each

| Experiment | Answers | Publication Value | Difficulty | Time |
|------------|---------|-------------------|------------|------|
| **Interchange** | Is this causal? | ⭐⭐⭐ Critical | Medium | 2-3d |
| **Layer scanning** | Where do traits live? | ⭐⭐⭐ High | Easy | 1d |
| **Component ablation** | Attn vs MLP? | ⭐⭐ Interesting | Medium | 2-3d |
| **SVD dimensionality** | How sparse? | ⭐⭐ Good | Easy | 1d |
| **Temporal causality** | Persistence? | ⭐ Supporting | Easy | 1d |
| **Cross-trait** | Composition? | ⭐ Interesting | Easy | 1d |
| **Visualizations** | User understanding | ⭐ Demo value | Medium | 1-2d each |

**Total core experiments:** ~10-12 days of focused work

---

## Expected Paper Structure

With these experiments complete:

### Introduction
- Trait vectors as directions in activation space
- Need for causal validation beyond correlation

### Methods
1. **Extraction:** 4 methods × 26 layers × 19 traits
2. **Cross-distribution testing:** Inst ↔ Nat validation
3. **Causal mediation:** Interchange interventions
4. **Layer localization:** Scanning for critical mediators
5. **Mechanistic decomposition:** Attention vs. MLP

### Results
1. **Extraction performance:** Probe 99.6% avg (inst→inst)
2. **Cross-distribution:** Method selection by separability
   - High sep → Probe 100%
   - Low sep → Gradient 96%
3. **Causal validation:** Interchange success rates
   - Refusal: 95% swap success
   - Uncertainty: 88% swap success
4. **Layer localization:** Middle layers (10-16) mediate
5. **Component analysis:** Attention 65%, MLP 35% (avg)
6. **Dimensionality:** Traits are sparse (<50 dims avg)

### Discussion
- Traits are genuine causal mechanisms (not epiphenomena)
- Method selection matters (separability-dependent)
- Attention-centric theory partially validated
- Practical implications for alignment, safety, interpretability

---

## Next Actions

**Immediate (after cross-distribution finishes):**

1. Implement interchange interventions (highest ROI)
2. Test on 3 traits: refusal, uncertainty, commitment
3. If successful → paper-ready causal validation

**Short-term (1-2 weeks):**

4. Layer scanning (builds on interchange)
5. Component ablation (tests attention hypothesis)
6. SVD dimensionality (easy win)

**Medium-term (3-4 weeks):**

7. Temporal persistence
8. Visualization improvements
9. Paper writeup

---

## Code Structure

```
experiments/causal_validation/
├── interchange_interventions.py   # Core causal test
├── layer_scanning.py              # Where traits live
└── component_ablation.py          # Attention vs. MLP

analysis/
├── trait_dimensionality.py        # SVD analysis
├── temporal_causality.py          # Persistence testing
└── cross_trait_interference.py    # Composition tests

visualization/
└── causal_effects/
    ├── intervention_heatmap.html
    ├── commitment_detector.html
    └── attention_flow.html
```

---

## Potential Surprises

**What if interchange fails?**
- Vectors might be correlational, not causal
- Would need to revisit extraction methods
- Still have cross-distribution as evidence of generalization

**What if different layers matter?**
- Layer 16 assumption might be wrong
- Could discover trait-specific optimal layers
- Would improve extraction quality

**What if MLP dominates over attention?**
- Challenges "traits are attention patterns" claim
- Suggests concept activation is primary mechanism
- Shifts theoretical framework

**What if traits aren't sparse?**
- Means traits are holistic, distributed
- Compression harder
- Suggests emergent rather than modular structure

---

## Related Work to Cite

**Causal interventions:**
- Interchange interventions (Geiger et al., DAS)
- Causal mediation analysis (Pearl)
- Activation patching (Meng et al., ROME)

**Component analysis:**
- Attention head analysis (Elhage et al., Anthropic)
- MLP neuron interpretation (Bills et al., OpenAI)

**Dimensionality:**
- Sparse representations (Olah et al., Circuits)
- Low-rank structure (Elhage et al., Toy Models)

---

## Success Metrics

**Minimum viable results:**
- Interchange works ≥70% of time on 1 trait → Publishable
- Layer scanning identifies critical range → Validates approach
- Component ablation shows clear signal → Mechanistic insight

**Strong results:**
- Interchange works ≥80% across 3+ traits → Strong paper
- Layer scanning shows trait-specific patterns → Novel finding
- Attention dominates (>60%) → Theoretical contribution

**Exceptional results:**
- Interchange works ≥90% across all traits → Major contribution
- Discover unexpected layer patterns → Surprising finding
- Component analysis reveals new mechanism → Breakthrough

---

## Conclusion

These experiments transform your work from "we can measure traits" to "we understand how traits work causally and mechanistically."

**The one experiment you MUST run:** Interchange interventions

**The three that build a complete story:** + Layer scanning + Component ablation

Everything else is extensions and polish.

**Ready to start implementing interchange interventions?** It's the highest ROI experiment and foundational for everything else.
