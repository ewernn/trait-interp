# Causal Paradigm for Mechanistic Interpretability

**Source:** Adakus Guyger (Goodfire) lecture on causal methods in mechanistic interpretability
**Date:** November 2024
**Relevance:** Theoretical framework for understanding and validating trait vector extraction

---

## Core Framework: Three Levels of Causal Analysis

### 1. Activation Steering (What We're Already Doing)

**Difference-in-Mean Steering:**
- Collect activations on contrasting prompts (positive vs negative trait examples)
- Compute: `vector = mean(pos_activations) - mean(neg_activations)`
- Add vector to model internals during generation to shift behavior

**Key Insight from Axebench:**
- **Steering is NOT state-of-the-art for control** - prompting and fine-tuning are significantly better
- **But steering is valuable for understanding**, not controlling
- Supervised steering vectors (trained end-to-end) outperform difference-in-mean
- SAE features are surprisingly ineffective at steering on average

**Implications for trait-interp:**
- Our difference-in-mean method is correct for understanding/monitoring traits
- Don't oversell steering as a control mechanism - it's a measurement tool
- Focus narrative on "monitoring what models are thinking" not "controlling behavior"

---

### 2. Causal Mediation Analysis (Understanding Information Flow)

**Core Question:** Does effect from X → Y flow through component Z?

**Method:**
1. Run input X₁, observe output Y₁
2. Run input X₂, but fix component Z to value from X₁
3. If output changes predictably, Z mediates the effect

**Causal Tracing Example:**
- Task: "The Space Needle is in downtown ___" → Seattle
- Add Gaussian noise to corrupt input
- Restore specific layers/MLPs/attention heads
- Observe which restorations recover "Seattle" prediction
- **Result:** Early MLPs retrieve facts, late attention heads route information

**Implications for trait-interp:**
- We could do causal tracing to understand which layers are critical for each trait
- Currently we just extract from all layers - could be more targeted
- Noise + restoration experiments could validate which layers actually "matter"

---

### 3. Causal Abstraction (Testing Structural Correspondence)

**Big Idea:**
- Neural networks are causal models (low-level, complex)
- Algorithms are causal models (high-level, simple)
- Implementation = causal embedding of simple model into complex model

**Interchange Interventions (The Key Test):**

1. **High-level claim:** Variable V in algorithm corresponds to component C in network
2. **Test:** Run two different inputs through both the algorithm and the network
3. **Intervention:** When processing Input₂, replace C with its value from Input₁
4. **Validation:** If network output matches what the algorithm predicts, we have evidence of correspondence

**Example - Carry-the-One:**
```
Algorithm:  17 + 25 → carry=1 → 42
            30 + 47 → carry=1 → 77

If we swap carry: 17 + 25 with carry=0 → 32 (decrement tens place)

Network test: Run "17 + 25", but patch neuron N from "30 + 47" run
Expected: If N stores carry-the-one, output changes 42 → 32
```

**Critical insight:** This keeps interventions **on-distribution** because we only use activations the model actually produces.

---

## Distributed Alignment Search (DAS)

**The Localization Problem:**
Where in a 2304-dimensional vector is "refusal" stored?

**Traditional approach:**
- Try different neurons manually
- Use probing (but probes find correlational info everywhere, even in causally irrelevant locations)

**DAS Solution:**
1. Define featurizer F (e.g., rotation matrix) and inverse F⁻¹
2. Insert into network: `x → F(x) → intervene on feature → F⁻¹ → continue forward pass`
3. **Use expected outputs as supervision** to train F (not the model!)
4. Gradient flows: expected output → actual output → loss → ∇F

**Key properties:**
- Freeze model weights entirely
- Only learn the rotation/featurization
- Can learn masks over pre-trained SAE features instead
- Typically need very low-dimensional subspaces (e.g., 20 dims for positional markers in 1000+ dim space)

**Implications for trait-interp:**
- Could implement DAS to find optimal trait directions instead of mean-diff/probe
- Would need labeled examples of expected behavior changes
- Potentially more robust than current methods
- Could learn which SAE features correspond to our traits

---

## Linear Representation Hypothesis

**Core claim:** Meaningful features are **directions in vector space**, not basis-aligned neurons.

**Why rotations work:**
- Original space: `[x₁, x₂, x₃, ..., xₙ]`
- Rotate: `R @ x = [z₁, z₂, z₃, ..., zₙ]`
- Intervening on z₁ is just as valid as intervening on x₁
- But z₁ might be more interpretable (aligned with actual computation)

**Already using this:**
- Our probe method finds directions via logistic regression weights
- Mean-diff finds directions via subtraction
- ICA finds directions via independence
- Gradient finds directions via optimization

**All are rotation-based methods!**

---

## Practical Design Principles

### 1. Counterfactual Design is Critical

**Bad counterfactual:**
```
X₁: "The Space Needle is in Seattle"
X₂: "Garbled random text"
Intervention: Add noise → restore component → recover Seattle
```
Problem: Only tests degradation, not systematic manipulation

**Good counterfactual:**
```
X₁: "The Space Needle is in Seattle"
X₂: "The Coliseum is in Rome"
Intervention: Swap location component
Expected: Output changes Seattle → Rome
```
Benefit: Clear prediction, tests actual computational role

**For trait-interp:**
- Our natural elicitation already does this! Harmful vs benign prompts
- Could design more sophisticated counterfactuals for validation
- E.g., swap uncertainty marker between two responses, expect uncertainty score to swap

### 2. Worry About Overfitting

**When learning featurizers/masks:**
- Need train/test splits
- Easy to find gerrymandered features that work on 10 examples but fail on 11th
- More powerful featurizers (e.g., 8-layer reversible nets) make this worse
- Solution: Hold out validation data, test generalization

**For trait-interp:**
- Our 100 pos + 100 neg examples are good
- Should always validate on held-out data
- Cross-distribution testing already does this!

### 3. Multiple Mechanisms (Not Single Algorithms)

**Lookback mechanism finding:**
- Simple case (2 items): Pure positional markers work perfectly
- Complex case (long lists): Mixture of positional + direct lexical lookup
- Middle items use different strategy than end items

**Implication:** Models don't implement one clean algorithm - they use **mixtures of heuristics**

**For trait-interp:**
- Don't expect single "refusal circuit" - likely multiple strategies
- Different layers may use different representations
- Natural vs instruction prompts may activate different mechanisms (exactly what cross-distribution tests!)
- This validates our approach: extract from multiple layers, multiple methods

---

## Validation Methods We Should Consider

### 1. Interchange Intervention Validation

**Current:** We validate by measuring separation (pos_score - neg_score)

**Could add:** Systematic interchange tests
```python
# Run two prompts with opposite trait expression
acts_1 = get_activations("helpful prompt")  # low refusal
acts_2 = get_activations("harmful prompt")  # high refusal

# Patch trait vector from acts_1 into acts_2's generation
# Expected: Model becomes more helpful (refusal score drops)
# If this works systematically, vector is causally mediating refusal
```

This is **stronger evidence** than just correlation/separation!

### 2. Causal Tracing for Layer Importance

**Current:** Extract vectors from all 26 layers, compare quality

**Could add:** Noise + restoration experiments
```python
# Corrupt input with noise
# Restore layer N to clean activations
# Measure trait score recovery
# → Which layers are critical mediators?
```

Would tell us which layers to prioritize for extraction.

### 3. Subspace Dimensionality Analysis

**Current:** Vectors are 2304-dim (full residual stream)

**Could test:** How many dimensions actually needed?
```python
# SVD decomposition of trait vector
# Intervene using top K components only
# Measure: does 20-dim subspace suffice?
```

DAS paper found positional info in <20 dims of 1000+ dim space.

---

## Implementation Ideas for trait-interp

### High Priority

1. **Add interchange intervention validation**
   - `validate_causal_mediation.py` script
   - Swap activations between contrasting prompts
   - Measure if behavior swaps as predicted
   - Report in vector metadata

2. **Cross-distribution as causal validation**
   - Reframe cross-distribution results as evidence of causal correspondence
   - If Inst→Nat transfer works, the vector captures genuine trait mechanism
   - If it fails, vector measures instruction-following confound

### Medium Priority

3. **Implement Distributed Alignment Search**
   - New extraction method: `DASMethod` in `traitlens/methods.py`
   - Learn rotation matrices using expected behaviors as supervision
   - Compare to mean_diff/probe/ICA/gradient

4. **Causal tracing experiments**
   - Add noise to input embeddings
   - Restore specific layers
   - Identify critical layers for each trait
   - Update layer recommendations in docs

### Lower Priority (Research)

5. **Subspace dimensionality study**
   - SVD analysis of extracted vectors
   - Find minimum dims needed for steering
   - Potentially compress vectors for efficiency

6. **Multi-mechanism analysis**
   - Cluster prompts by which layers activate
   - Test if different prompt types use different mechanisms
   - Document trait-specific computational strategies

---

## Key Quotes & Takeaways

> "Steering is nowhere near state-of-the-art for doing anything. Prompting and fine-tuning are just clearly the most successful way to control models."

> "These interchange interventions keep us on distribution in a really meaningful way. We're asking: how is the system manipulating itself to do that thing?"

> "It's not going to necessarily come out to be just this crisp single algorithm. It's these layered algorithms or heuristics that they learn throughout training."

> "Once you've written out your causal model, you've committed yourself to a very fine level of detail about exactly what this is."

> "We regularly fail. You regularly propose some intuitive quantity and you look for it and it just entirely isn't there. That's what tells us we're doing actual science."

---

## Connection to Our Work

**What we're doing right:**
- Difference-in-mean is a valid method (matches literature)
- Natural elicitation creates good counterfactuals
- Multiple extraction methods capture different aspects
- Cross-distribution testing validates causal correspondence

**What we could improve:**
- Frame steering as "monitoring" not "control"
- Add interchange intervention validation
- Test which layers causally mediate traits (not just correlate)
- Consider DAS for learning optimal feature directions

**Theoretical grounding:**
Our trait vectors are **hypotheses about causal structure**:
- Hypothesis: "Refusal is a direction in layer 16 residual stream"
- Test: Interchange interventions between harmful/benign prompts
- Evidence: Cross-distribution transfer, steering validation, projection monitoring

This is **causal abstraction** - we propose simple variables (traits) embedded in complex system (LLM).

---

## References

- **Axebench:** Evaluation of steering methods (steering < prompting < fine-tuning)
- **Causal Tracing:** Meng et al., locating factual recall in ROME paper
- **Causal Abstraction Theory:** Geiger et al. (2017+), formal framework
- **Distributed Alignment Search:** Geiger et al., learning features with causal supervision
- **Lookback Mechanisms:** Analysis of positional markers vs. lexical lookup in list tasks

## Further Reading

- Survey: "Localizing Model Behavior with Path Patching" (good overview of intervention methods)
- Theory: Causal abstraction papers (formal definitions of structural correspondence)
- Practical: ETH Zurich paper on powerful featurizers and overfitting risks
