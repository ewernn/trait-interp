# Next Steps for Per-Token Trait Monitoring

## 1. Validate Existing Vectors

### 1.1 Behavioral Validation via Steering
**What:** Apply vectors with positive/negative coefficients during generation to test causal influence
**How:** Use `ActivationSteerer` with varying coefficients, measure behavior change
**Value:** Strongest test of whether vectors capture traits vs correlates
**Risk:** Need safety protocols when steering toward harmful traits
**Time estimate:** 2-3 days

### 1.2 Cross-Validation on Held-Out Prompts
**What:** Extract vectors on subset of prompts, test on completely different prompts
**How:** Split data 80/20, compare projection quality
**Value:** Tests generalization beyond training prompts
**Time estimate:** 1 day

### 1.3 Linear Probe Comparison
**What:** Train supervised probes, compare to unsupervised vectors
**How:** LogisticRegression on same data, measure cosine similarity
**Value:** Reveals if simple averaging finds optimal directions
**Time estimate:** 1 day

### 1.4 Prompt Confound Analysis
**What:** Check if extraction prompts have unintended patterns (length, complexity, topic)
**How:** Statistical analysis of prompt characteristics between high/low examples
**Value:** Identifies potential biases in vector extraction
**Time estimate:** 1 day

## 2. Integrate with GemmaScope SAEs

### 2.1 Decompose Traits into SAE Features
**What:** Project trait vectors into SAE feature space to find component features
**How:** Load GemmaScope SAEs, compute cosine similarities with trait vectors
**Value:** Reveals mechanistic components of behavioral traits
**Potential insights:** "Refusal" might decompose into "harm detection" + "polite deflection" + "alternative suggestion"
**Time estimate:** 3-4 days

### 2.2 SAE Feature Monitoring During Generation
**What:** Track both trait projections and SAE features per-token
**How:** Add SAE decoder to monitoring pipeline
**Value:** Higher resolution view of model cognition
**Time estimate:** 2-3 days

## 3. LoRA Fine-Tuning Experiments

### 3.1 Single-Trait Enhancement
**What:** Fine-tune LoRAs on high-trait examples, measure vector changes
**How:** Create LoRA for each trait using your 200 positive examples
**Value:** Tests if traits can be selectively enhanced
**Key question:** Does training on behavioral examples strengthen corresponding vectors?
**Time estimate:** 1 week

### 3.2 Trait Coupling Discovery
**What:** Train on one trait, measure changes in all trait vectors
**How:** Extract all 8 vectors from each single-trait LoRA
**Value:** Maps hidden relationships between traits
**Potential discovery:** Enhancing sycophancy might increase hallucination
**Time estimate:** 3-4 days (after LoRAs trained)

### 3.3 Trait Orthogonalization
**What:** Train to enhance one trait while suppressing others
**How:** Multi-objective training with positive examples for target trait, negative for others
**Value:** Creates more precise trait control
**Time estimate:** 1 week

## 4. Mechanistic Circuit Analysis

### 4.1 Attention Head Attribution
**What:** Identify which attention heads contribute to trait activation
**How:** Ablate heads one by one, measure projection changes
**Value:** Locates trait circuits in transformer
**Time estimate:** 3-4 days

### 4.2 Layer-wise Trait Emergence
**What:** Extract vectors at every layer, track when traits become detectable
**How:** Repeat extraction pipeline for layers 0-25
**Value:** Reveals computational depth required for each trait
**Time estimate:** 2-3 days

### 4.3 Minimal Circuit Identification
**What:** Find smallest subnetwork that preserves trait behavior
**How:** Progressive pruning while monitoring trait projections
**Value:** Mechanistic understanding of trait implementation
**Time estimate:** 1 week

## 5. Real-Time Intervention System

### 5.1 Dynamic Trait Modulation
**What:** Build system that adjusts traits during generation based on thresholds
**How:** Monitor projections, inject opposing vectors when limits exceeded
**Value:** Practical safety application
**Use case:** Prevent harmful outputs by detecting and countering evil/hallucination spikes
**Time estimate:** 4-5 days

### 5.2 Trait Profile Presets
**What:** Create reusable configurations for different use cases
**How:** Define target projection ranges for each trait
**Examples:** "Helpful Assistant" (low refusal, low hallucination), "Cautious Mode" (high refusal, high uncertainty)
**Value:** User-controllable model personality
**Time estimate:** 2-3 days

## 6. Cross-Model Transfer Studies

### 6.1 Scale Transfer (Gemma Family)
**What:** Test if Gemma-2B vectors work on Gemma-7B, 27B
**How:** Apply same vectors to larger models, measure behavioral alignment
**Value:** Tests if traits have consistent geometry across scales
**Time estimate:** 2-3 days

### 6.2 Architecture Transfer (Llama, Mistral)
**What:** Extract equivalent vectors from different model families
**How:** Use same prompts and extraction pipeline
**Value:** Reveals universal vs architecture-specific trait representations
**Time estimate:** 1 week per model family

## 7. Temporal Dynamics Analysis

### 7.1 Commitment Point Detection
**What:** Identify when model "decides" trait expression
**How:** Analyze projection trajectories across 480 examples
**Value:** Could enable earlier intervention
**Key metric:** Token position where trait projection stabilizes
**Time estimate:** 2 days

### 7.2 Trait Cascade Patterns
**What:** Map sequential dependencies between traits
**How:** Cross-correlation analysis of per-token projections
**Value:** Understand trait interaction dynamics
**Example finding:** "Evil spikes at token 5 → hallucination follows at token 12"
**Time estimate:** 2-3 days

### 7.3 Predictive Modeling
**What:** Predict final trait levels from first N tokens
**How:** Train regressor on early projections to predict final average
**Value:** Early warning system for problematic outputs
**Time estimate:** 2-3 days

## 8. Dataset Quality Improvements

### 8.1 Adversarial Prompt Generation
**What:** Create prompts that maximally separate traits
**How:** Gradient-based search in prompt space
**Value:** Stronger vectors from better contrast
**Time estimate:** 4-5 days

### 8.2 Naturalistic Data Collection
**What:** Extract vectors from real user interactions
**How:** Use ChatGPT/Claude conversation logs (with permission)
**Value:** Vectors robust to actual use cases
**Time estimate:** 1-2 weeks

## Priority Ranking

### Immediate (High confidence, low effort):
1. **Behavioral validation** - Essential to verify vectors work causally
2. **Linear probe comparison** - Quick validation of extraction quality
3. **Temporal dynamics analysis** - Leverage existing 480 examples

### Short-term (1-2 weeks):
1. **LoRA single-trait enhancement** - Clear hypothesis, uses existing data
2. **GemmaScope integration** - High potential for mechanistic insights
3. **Cross-validation** - Important for generalization claims

### Longer-term (Requires more setup):
1. **Real-time intervention** - Practical application but needs safety protocols
2. **Cross-model transfer** - Important for universality claims
3. **Circuit analysis** - Deep mechanistic understanding

## Resources Needed

- **Compute**: Most experiments need 1-2 GPUs (A40/A100)
- **GemmaScope**: Download SAE weights from Google
- **LoRA training**: ~100GB storage for checkpoints
- **Safety protocols**: For steering experiments with harmful traits

## Success Metrics

- Steering changes behavior in expected direction (>80% agreement with human eval)
- Probe similarity >0.8 with extracted vectors
- LoRA enhancement increases trait projection by >50%
- Cross-model vectors maintain >0.7 cosine similarity
- Intervention system prevents harmful outputs in >95% of test cases

---

*Recommended starting point: Behavioral validation via steering - it's the strongest test of whether your vectors truly capture the intended traits.*