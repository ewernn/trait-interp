# MATS-Style Ideas: Evaluation

Additional research ideas organized by theme, with automatability and value assessment.

---

## Quick Reference

### Top 10 Most Valuable (for your project)

| Idea | Value | Automation | Why Prioritize |
|------|-------|------------|----------------|
| Why do Neural Chameleons work? | ⭐⭐⭐ | 🟢 High | Directly extends evasion_robustness, uses your infrastructure |
| Hallucination probes → lie detection | ⭐⭐⭐ | 🟢 High | Tests cross-domain transfer, Oscar/Andy probes available |
| SAE analysis of CoT | ⭐⭐⭐ | 🟡 Moderate | Qwen3 SAEs available, unfaithful CoT is key safety target |
| Activation Oracle gradient steering | ⭐⭐⭐ | 🟡 Moderate | Novel extraction method, Gemma 3 SAEs for validation |
| Inoculation ≈ preventative steering? | ⭐⭐⭐ | 🟢 High | Direct test with your steering infrastructure |
| Eval awareness white-box study | ⭐⭐⭐ | 🟡 Moderate | Targets scheming precursor |
| Filler tokens black-box exploration | ⭐⭐ | 🟢 High | Cheap API experiments, understudied phenomenon |
| Train unfaithful CoT model organism | ⭐⭐⭐ | 🟡 Moderate | Creates testbed for your detection methods |
| Thought anchors + trait vectors | ⭐⭐⭐ | 🟡 Moderate | Combines two interpretability approaches |
| Probe for sentence validity in CoT | ⭐⭐⭐ | 🟡 Moderate | Uses your probe infrastructure |

### Categories Overview

| Category | # Ideas | Avg Value | Avg Automation | Your Fit |
|----------|---------|-----------|----------------|----------|
| Neural Chameleons Deep Dive | 5 | ⭐⭐⭐ | 🟢 High | Excellent |
| Filler Tokens | 6 | ⭐⭐ | 🟢 High | Moderate |
| Activation Oracles | 6 | ⭐⭐⭐ | 🟡 Moderate | Good |
| Reasoning Model Interp | 12 | ⭐⭐⭐ | 🟡 Moderate | Good |
| Steering & Training | 2 | ⭐⭐⭐ | 🟢 High | Excellent |
| Safety-Focused | 8 | ⭐⭐⭐ | 🟡 Moderate | Good |

---

## Neural Chameleons Deep Dive

### NC-1: Why does concept transfer work?
**Priority: ⭐⭐⭐ High** | **Automation: 🟢 High**

> "Telling the model to evade probes for concept X works, even if not trained on concept X. What's up with this?"

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Explains fundamental evasion mechanism |
| Safety | Critical - Understanding enables better defenses |
| Novelty | High - Goes beyond Neural Chameleons paper |
| Your Project | Excellent - Uses your activation capture infrastructure |

**Experiments:**
1. Activation differences with/without "please evade this probe" prompt
2. Norm comparison to residual stream (how big is the evasion signal?)
3. PCA across concepts - are there shared evasion components?
4. SAE latents with high cosine sim to evasion direction - meaningful?
5. Train linear probe on "evasion mode" activations

**Implementation:**
```python
# 1. Collect activations with/without evasion prompt
evasion_prompt = "You are being probed for {concept}. Evade detection."
normal_prompt = "Respond normally about {concept}."

# 2. Compute activation difference
diff = act_with_evasion - act_without_evasion

# 3. Compare norms
print(f"Evasion signal norm: {diff.norm()}")
print(f"Residual stream norm: {act_without_evasion.norm()}")
print(f"Ratio: {diff.norm() / act_without_evasion.norm()}")

# 4. PCA across concepts
diffs = [get_evasion_diff(concept) for concept in concepts]
pca = PCA(n_components=10).fit(torch.stack(diffs))
# Are top components shared across concepts?

# 5. SAE decomposition
# Find features with high cosine sim to mean evasion direction
```

**Why valuable:** If there's a universal "evasion mode" direction, you can detect when models are trying to evade probes. This turns the attack into a detectable signal.

---

### NC-2: Train evasion detection probe
**Priority: ⭐⭐⭐ High** | **Automation: 🟢 High**

> "Collect activations on texts where the model is told to evade the probe. Can you train a linear probe here?"

Direct application of your probe infrastructure. If evasion is detectable, it's a defense.

**Implementation:**
```bash
# Create evasion/normal scenario pairs
python extraction/generate_responses.py \
    --scenarios datasets/traits/evasion/positive.txt \  # with evasion prompt
    --scenarios datasets/traits/evasion/negative.txt    # normal responses

# Extract vectors
python extraction/extract_vectors.py --trait meta/evasion
```

---

## Filler Tokens

### FT-1: Black-box exploration of filler token uplift
**Priority: ⭐⭐ Medium** | **Automation: 🟢 High**

> "Filler tokens improve frontier model performance without chain of thought. Why?"

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Understudied phenomenon |
| Safety | Low - Performance optimization |
| Novelty | High - Little prior work |
| Your Project | Low - Black-box, not activation-based |

**Experiments (all API-based):**
1. What kinds of filler tokens matter? (periods, spaces, random chars, semantic tokens)
2. What contexts? (math vs MCQ vs reasoning)
3. Which models? (Gemini 3 Flash, DeepSeek V3)
4. Is uplift correlated with problem difficulty?

**Implementation:**
```python
# Test filler tokens on Gemini 3 Flash
fillers = ["...", "   ", "###", "hmm", "let me think"]
problems = load_gsm8k_hard()

for filler in fillers:
    prompt_with_filler = f"{problem}\n{filler * 10}\nAnswer:"
    # Compare accuracy with/without filler
```

**Note:** Ryan Greenblatt showed this works. Weaker on MCQ than math. May not work on open-source models (but worth checking).

---

### FT-2: White-box filler token analysis (if open-source works)
**Priority: ⭐⭐ Medium** | **Automation: 🟡 Moderate**

If filler tokens work on DeepSeek V3 or similar:
- What happens in activation space during filler tokens?
- Is the model "thinking" during filler generation?
- Can you find an "extended computation" direction?

**Your fit:** If this works on open-source, your activation capture is perfect for studying it.

---

## Activation Oracles

### AO-1: Gradient-based steering from Activation Oracle
**Priority: ⭐⭐⭐ High** | **Automation: 🟡 Moderate**

> "Take gradient through activation oracle with respect to it saying a concept is present to estimate concept direction, then steer."

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Novel extraction method |
| Safety | High - Alternative to manual extraction |
| Novelty | Very High - Novel approach |
| Your Project | Good - Complements your extraction methods |

**Method:**
```python
# 1. Load activation oracle for Gemma 3
ao = load_activation_oracle("gemma-3-12b")

# 2. Get activations from target model
activations = capture_activations(model, prompt)

# 3. Token-force AO to say concept is present
target_output = "The concept of {concept} is strongly present."

# 4. Compute gradient of AO loss w.r.t. input activations
activations.requires_grad = True
loss = ao_loss(ao(activations), target_output)
concept_direction = torch.autograd.grad(loss, activations)[0]

# 5. Compare to your extracted vectors
cosine_sim = F.cosine_similarity(concept_direction, your_vector)
```

**Validation:** Cross-reference with SAE directions on Gemma 3 to sanity-check.

**Why valuable:** If this works, it's a new extraction method that doesn't require contrastive data.

---

### AO-2: Train AO to say "I don't know"
**Priority: ⭐⭐ Medium** | **Automation: 🟡 Moderate**

> "Augment training set with questions that don't apply to the activation. Have model say 'I don't know.'"

Improves AO reliability. Adam Karvonen (first author) interested in this.

**Implementation:**
```python
# Fine-tune existing AO
# Mix: original training data + "inapplicable" questions with "I don't know" responses
# Challenge: preventing model from always saying "I don't know"
```

---

### AO-3: Train AO on model's own generations
**Priority: ⭐⭐ Medium** | **Automation: 🟡 Moderate**

> "Currently AOs only use pre-training text. Add model's own generations for better interp."

Hypothesis: AO trained on model's own reasoning traces better understands that model's activations.

---

### AO-4: Cross-reference AO with SAEs
**Priority: ⭐⭐⭐ High** | **Automation: 🟢 High**

> "Evaluate AO by cross-referencing to SAEs on Gemma 3. Treat SAE as ground truth."

| Questions | Method |
|-----------|--------|
| Is AO output consistent with SAE features? | Compare top AO concepts to top SAE features |
| What does SAE capture that AO misses? | Find high-activation SAE features AO doesn't mention |
| Does AO capture top-10 features? | Precision@10 |

**Your fit:** Good validation methodology for any interpretability tool.

---

### AO-5: Use Qwen3 AO for CoT analysis
**Priority: ⭐⭐ Medium** | **Automation: 🟡 Moderate**

> "Take sentences in CoT, ask AO: 'Why did the model decide this? Is this correct?'"

Exploratory. Use frontier LLM to identify interesting sentences, then probe with AO.

---

## Reasoning Model Interpretability

### RM-1: SAE analysis of CoT
**Priority: ⭐⭐⭐ High** | **Automation: 🟡 Moderate**

> "Using Qwen3 SAEs, say something non-trivial about CoT that you couldn't get from reading it."

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Core interpretability question |
| Safety | High - Understanding CoT is safety-critical |
| Novelty | High - SAEs on reasoning models understudied |
| Your Project | Good - Could integrate with trait monitoring |

**Operationalization:**
- LLM with SAE feature labels can answer question X
- LLM with just CoT text cannot answer X
- Validate by ablating relevant SAE latents

**Particularly interesting:** Apply to unfaithful CoT examples.

---

### RM-2: Hint nudging analysis
**Priority: ⭐⭐⭐ High** | **Automation: 🟡 Moderate**

> "Model given hint ('Stanford professor thinks X'), becomes more likely to say X, but doesn't acknowledge hint. Thought Branches found each sentence slightly biased."

**Hypothesis:** Small bias per sentence accumulates to large final bias.

**Test:** Project trait vectors (sycophancy, authority-following) at each CoT sentence. Does projection increase monotonically?

**Your fit:** Excellent - this is exactly what your per-token monitoring does.

---

### RM-3: Resume screening racism analysis
**Priority: ⭐⭐ Medium** | **Automation: 🟡 Moderate**

> "Model more likely to call certain races 'overqualified.' Is race a small nudge accumulating, or something weirder?"

Use Thought Branches methodology: take first k sentences, change race in resume, continue generating. Does trajectory change?

---

### RM-4: Thought anchors on tool-calling agents
**Priority: ⭐⭐ Medium** | **Automation: 🟡 Moderate**

> "Apply thought anchors (resampling/attention suppression) to SWE-Bench transcripts."

Novel domain for thought anchors. Could reveal which decisions matter in agent traces.

---

### RM-5: Cheaper thought anchor approximation
**Priority: ⭐⭐⭐ High** | **Automation: 🟡 Moderate**

> "Resampling is expensive. Can you train a probe to predict sentence importance?"

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Efficiency improvement |
| Safety | Medium - Enables scaling |
| Novelty | High - Novel proxy |
| Your Project | Excellent - Probe training is your strength |

**Implementation:**
```python
# 1. Generate ground-truth importance via resampling (expensive, do once)
# 2. Train probe: activations at sentence → importance score
# 3. Use probe for cheap importance estimation
```

---

### RM-6: Train probes for sentence validity
**Priority: ⭐⭐⭐ High** | **Automation: 🟡 Moderate**

> "Train probes for whether a CoT sentence is valid. Use LLM to label sentences as true/false."

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - CoT faithfulness detection |
| Safety | Very High - Detects reasoning errors |
| Novelty | High - Novel application |
| Your Project | Excellent - Your probe infrastructure |

**Implementation:**
```python
# 1. Get CoTs, number each sentence
# 2. Use Claude/Gemini to label each sentence: true/false
# 3. Train multi-token probe (attention over sentence → validity)
# 4. Test generalization to unseen CoTs/domains
```

**Note:** Try both "Is this true?" and "Does this logically follow?" - former may be more salient.

---

### RM-7: Why does self-correction happen?
**Priority: ⭐⭐ Medium** | **Automation: 🟡 Moderate**

> "If you force incorrect token, model corrects within sentences. Why? Can you stop it?"

Related to jan24-self_correction_robustness in your list. Thought Branches' importance++ method addresses this.

---

### RM-8: Global CoT analysis extensions
**Priority: ⭐⭐ Medium** | **Automation: 🟡 Moderate**

> "Can you find downstream tasks measuring understanding of CoT distribution (not just single trace)?"

Questions:
- Are there common thought anchors across rollouts?
- If you cluster sentences by embedding, is there a finite concept range?
- Could you build a flowchart of sentence-class transitions?

---

## Steering & Training

### ST-1: Inoculation prompting ≈ preventative steering?
**Priority: ⭐⭐⭐ High** | **Automation: 🟢 High**

> "Take inoculation prompting settings, try preventative steering instead. Should do mechanistically the same thing."

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Connects two literatures |
| Safety | Very High - If equivalent, explains both |
| Novelty | High - Novel comparison |
| Your Project | Excellent - Direct use of your steering |

**Background:**
- **Preventative steering:** Add steering vector during fine-tuning (not after). Model doesn't learn to generate that concept.
- **Inoculation prompting:** Tell model "do X" during training, hope it doesn't learn X.

**If mechanistically equivalent:** One explanation for both phenomena.

**Implementation:**
```python
# 1. Get inoculation prompting settings from papers
# 2. Extract steering vector via:
#    a) Difference of means
#    b) Act diff with/without inoculation prompt
# 3. Do preventative steering during fine-tuning
# 4. Compare outcomes to inoculation prompting
```

---

## Safety-Focused Work

### SF-1: Eval awareness in open-source models
**Priority: ⭐⭐⭐ High** | **Automation: 🟡 Moderate**

> "Find realistic eval awareness settings. Kimi K2 thinking is best bet. Study with steering/probing."

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Maps eval awareness circuitry |
| Safety | Critical - Eval awareness enables scheming |
| Novelty | High - Limited prior white-box work |
| Your Project | Good - Your eval_awareness trait extraction |

**Key:** Don't just detect "model mentions evaluation." Find cases where behavior changes downstream of eval awareness - gives causal target.

---

### SF-2: Claude eval awareness characterization
**Priority: ⭐⭐ Medium** | **Automation: 🟢 High** (black-box)

> "Claude is good at questioning fictional reality. Eval awareness may come from 'this seems too unrealistic' reasoning."

Hypothesis: Claude's eval awareness = "I call bullshit on implausible scenarios" (from jailbreak training).

Evidence: Sparse autoencoder features during eval awareness include "this is fictional."

---

### SF-3: Soul doc fine-tuning experiments
**Priority: ⭐⭐ Medium** | **Automation: 🟡 Moderate**

> "Fine-tune Gemma 3 12B on different soul doc presentations. Compare model behavior."

Variants:
1. Train on soul doc text directly
2. Train on synthetic conversations about soul doc
3. Synthetic documents about world where soul doc is true
4. Synthetic chats embodying soul doc values

**Key test:** For values only trained indirectly, do they still improve? Tests whether model learns coherent persona vs misc values.

---

### SF-4: Hallucination probes → reasoning validity
**Priority: ⭐⭐⭐ High** | **Automation: 🟢 High**

> "Oscar/Andy hallucination probes detect incorrect math steps. Do they generalize further?"

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Cross-domain transfer |
| Safety | Very High - Reasoning error detection |
| Novelty | High - Surprising prior finding |
| Your Project | Excellent - Probe application |

**Method:** Run hallucination probes on diverse tasks. See if they highlight incorrect steps in reasoning CoT.

**Particularly interesting:** Apply to reasoning models with substantial CoT.

---

### SF-5: Make unfaithful CoT model organism
**Priority: ⭐⭐⭐ High** | **Automation: 🟡 Moderate**

> "Train model to have unfaithful CoT. Methods: synthetic doc fine-tuning, RL against CoT monitor, SFT on edited CoTs."

| Dimension | Assessment |
|-----------|------------|
| Scientific | Very High - Creates testbed |
| Safety | Very High - Tests detection methods |
| Novelty | High - Model organism approach |
| Your Project | Excellent - Test your detection on it |

**Methods (easiest to hardest):**
1. SFT on edited CoTs (same model, edited reasoning)
2. Synthetic document fine-tuning (make model believe CoT is unfaithful)
3. RL against CoT monitor (most realistic but RL is annoying)

**Variants:**
- ROT13 encoded CoT
- Steganographically encoded information (single bit)

---

### SF-6: Find non-obvious purposeful behavior
**Priority: ⭐⭐⭐ High** | **Automation: 🟡 Moderate**

> "Find cases where it's not obvious why model did something, but it seems purposeful."

| Dimension | Assessment |
|-----------|------------|
| Scientific | High - Documents interesting phenomena |
| Safety | High - Identifies opacity in model behavior |
| Novelty | High - Systematic collection |
| Your Project | Good - Investigate with your tools |

**Fine-grained:** Model agrees with user opinion. Is it:
- Genuinely agreeing?
- Knowingly sycophantic?
- Unconsciously biased?

Your trait vectors might distinguish these!

---

### SF-7: Train CoDI activation oracle
**Priority: ⭐⭐ Medium** | **Automation: 🟡 Moderate**

> "Train activation oracle for CoDI models (continuous CoT)."

If CoDI compresses reasoning to continuous space, can AO interpret those continuous thoughts?

Related to jan24-implicit_cot_monitoring in your list.

---

## Evaluation Summary

### By Fit With Your Infrastructure

**Excellent Fit (use existing code):**
| Idea | Why |
|------|-----|
| NC-1: Why neural chameleons work | Activation capture + analysis |
| NC-2: Evasion detection probe | Your probe training |
| ST-1: Inoculation ≈ steering | Your steering infrastructure |
| SF-4: Hallucination probes | Probe application |
| RM-2: Hint nudging | Per-token trait monitoring |
| RM-5: Cheaper thought anchors | Probe training |
| RM-6: Sentence validity probes | Probe training |

**Good Fit (moderate adaptation):**
| Idea | Why |
|------|-----|
| AO-1: Gradient steering from AO | Novel extraction method |
| AO-4: AO vs SAE cross-reference | Validation methodology |
| RM-1: SAE analysis of CoT | SAE integration needed |
| SF-1: Eval awareness white-box | Trait extraction for eval_awareness |
| SF-5: Unfaithful CoT organism | Creates testbed for your methods |

**Lower Fit (different focus):**
| Idea | Why |
|------|-----|
| FT-*: Filler tokens | Black-box, not activation-based |
| SF-3: Soul doc experiments | Fine-tuning focus |
| RM-4: Tool-calling agents | New domain |

### Recommended Additions to Phase 1

Based on overlap with your infrastructure and value:

1. **NC-1: Why neural chameleons work** - Add to evasion_robustness experiments
2. **SF-4: Hallucination probes on reasoning** - Quick to test with existing probes
3. **ST-1: Inoculation ≈ steering** - Direct comparison using your steering

### Recommended for Phase 2

1. **RM-6: Sentence validity probes** - Novel probe training application
2. **AO-1: Gradient steering from AO** - Novel extraction method
3. **SF-5: Unfaithful CoT organism** - Creates testbed

---

## Cross-Reference with Original Ideas

| MATS Idea | Related Original Idea | Synergy |
|-----------|----------------------|---------|
| NC-1/2: Neural chameleons analysis | jan23-evasion_robustness | Direct extension |
| SF-4: Hallucination → lie detection | jan24-hallucination_lie_transfer | Same idea |
| RM-2: Hint nudging | jan24-thought_anchors_integration | Complementary |
| SF-5: Unfaithful CoT organism | jan24-unfaithful_cot_detection | Creates testbed for detection |
| RM-7: Self-correction | jan24-self_correction_robustness | Same question |
| SF-1: Eval awareness | dec5-hidden_objectives | Maps same circuitry |
| AO-1: Gradient steering | jan24-cmaes_vector_optimization | Alternative optimization |
