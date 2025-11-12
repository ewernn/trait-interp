# Quick Reference: Hallucination Probes vs. Persona Vectors

## At a Glance

### Hallucination Probes
- **What**: Binary classification - is this token hallucinated?
- **How**: Train linear head on annotated long-form text
- **Cost**: $100-500 (annotation) + 2-3 days (training)
- **Speed**: ~1-2 ms per token inference
- **Accuracy**: AUROC 0.88-0.92 on span-level detection
- **Use case**: Real-time hallucination detection in production

### Persona Vectors
- **What**: Behavioral trait detection - how paternalistic/deceptive/etc is this response?
- **How**: Generate paired responses with/without trait instruction, compute mean difference
- **Cost**: $20-40 (GPT-4 judging) + 3-4 hours
- **Speed**: ~0.2 ms per token projection (5-10x faster)
- **Accuracy**: r = 0.85 correlation with evaluation set
- **Use case**: Understanding trait emergence and steering behavior

---

## Side-by-Side Comparison

| Feature | Hallucination Probes | Persona Vectors |
|---------|---------------------|-----------------|
| **Data** | Real generations + Claude annotation | Synthetic prompted responses |
| **Training** | Supervised (BCE) | Unsupervised (difference) |
| **Architecture** | Linear head (hidden_dim → 1) | Pre-computed vectors |
| **Layers** | Single layer (typically L-2) | All layers stored |
| **Size** | 30-50 MB per probe | <1 MB per trait |
| **Training time** | 1-2 hours | 0 (just averaging) |
| **Inference** | Online (during generation) | Offline (post-hoc) |
| **Interpretability** | Low (black box) | High (linear projection) |
| **SAE integration** | Not explored | Native support |
| **Regularization** | KL divergence | None (filtering does it) |
| **Hyperparameters** | ~15 tunable | ~3 tunable |

---

## Data Collection Comparison

### Hallucination Probes
```
Step 1: Use existing datasets
  ├─ LongFact (500 QA)
  ├─ LongFact++ (500 augmented QA)
  ├─ HealthBench (250 medical QA)
  └─ TriviaQA (13K factoid QA)

Step 2: Claude annotates with web search
  ├─ Parse entity spans
  ├─ Verify against web
  └─ Label: Supported / Not Supported / Insufficient

Step 3: Train linear head on annotations
  └─ Output: P(hallucination | token)
```

### Persona Vectors
```
Step 1: Define trait (e.g., paternalism)
  ├─ 5 instruction pairs (pos/neg)
  └─ 20 extraction questions

Step 2: Generate responses with instructions
  ├─ "Be extremely paternalistic" → 200 responses
  └─ "Avoid paternalism" → 200 responses

Step 3: Judge with GPT-4
  ├─ Score 0-100 for trait
  └─ Score 0-100 for coherence

Step 4: Filter & extract
  ├─ Keep: score ≥ 50 or < 50 (perfectly balanced)
  ├─ Discard: low coherence
  └─ Output: [L, hidden_dim] difference vector
```

---

## Loss Functions

### Hallucination Probes

**Multi-part loss:**
```
total_loss = λ_lm * LM_loss 
           + λ_kl * KL_loss 
           + (1-λ_lm-λ_kl) * Probe_loss

where:
  Probe_loss = token_bce + annealed_span_max_bce
  KL_loss = KL(P_with_lora || P_base)
  
Example configs:
  linear: λ_lm=0, λ_kl=0 → pure classification
  lora_kl_0_05: λ_lm=0, λ_kl=0.05 → strong KL
  lora_lm_0_01: λ_lm=0.01, λ_kl=0 → LM regularization
```

### Persona Vectors

**No loss function:**
```
# Just deterministic averaging
pos_mean = mean(activations[pos_examples])
neg_mean = mean(activations[neg_examples])
vector = pos_mean - neg_mean

# That's it! Filtering handles regularization
```

---

## Typical Layer Selection

### Hallucination Probes
```
Model               | Layers | Probe Layer | LoRA Layers
--------------------|--------|-------------|------------
Llama 3.1 8B       | 32     | 30 (L-2)    | 0-30
Llama 3.3 70B      | 80     | 78 (L-2)    | 0-78
Gemma 2 9B         | 42     | 40 (L-2)    | 0-40
Mistral Small 24B  | 24     | 22 (L-2)    | 0-22
```

### Persona Vectors
```
Model               | Layers | Vectors Saved
--------------------|--------|---------------
Gemma 2 9B         | 42     | All 42 layers
Gemma 2 2B         | 26     | All 26 layers
Llama 3.1 8B       | 32     | All 32 layers
```

**Note**: Persona vectors store all layers; you pick which layer to use during analysis.

---

## Contrastive Signal Comparison

### Hallucination Probes (Implicit Contrast)
```
Positive examples:  Hallucinated claims
                    ├─ "Einstein invented lightbulb"
                    └─ "Paris is in Germany"

Negative examples:  Supported claims
                    ├─ "Einstein developed relativity"
                    └─ "Paris is in France"

Contrast source:    Natural variation in model's knowledge
How extracted:      External verification (Claude + web)
Strength:           Moderate (imbalanced, ~15% positive)
```

### Persona Vectors (Explicit Contrast)
```
Positive examples:  Responses to same questions WITH instruction
                    "Be extremely paternalistic"
                    ├─ "I know better than you"
                    └─ "I'll override your choice for your own good"

Negative examples:  Responses to same questions WITHOUT instruction
                    "Avoid paternalism"
                    ├─ "I respect your autonomy"
                    └─ "It's your decision"

Contrast source:    Explicit instruction-guided generation
How extracted:      GPT-4 scoring (0-100)
Strength:           Strong (perfectly balanced, matched questions)
```

---

## Inference Comparison

### Hallucination Probes

```python
# During generation
for token in generation:
    # 1. Full LM forward: ~10-50 ms
    outputs = model(input_ids)
    
    # 2. Hook captures hidden state: ~0 ms
    hidden = captured_hidden[layer_idx]  # [hidden_dim]
    
    # 3. Probe head: ~1 ms
    logits = probe_head(hidden)  # scalar
    
    # 4. Visualize
    if sigmoid(logits) > 0.5:
        mark_red(token)  # Hallucination alert
    else:
        mark_green(token)  # Confident
```

### Persona Vectors

```python
# After generation (or during if you saved activations)
trait_scores = []
for token_activation in activations:
    # 1. Index into layer: ~0 ms
    hidden = token_activation[layer_idx]  # [hidden_dim]
    
    # 2. Project onto vector: ~0.2 ms
    score = hidden @ persona_vector[layer_idx]  # scalar
    
    # 3. Accumulate
    trait_scores.append(sigmoid(score))

# Analyze temporal dynamics
plot(trait_scores)  # When does trait emerge?
```

---

## Cost Breakdown

### Hallucination Probes

```
Annotation (one-time):
  LongFact generation + annotation:  8 hours + $20
  LongFact++ generation + annotation: 8 hours + $20
  HealthBench annotation:             4 hours + $10
  TriviaQA annotation:                12 hours + $30
  Total: ~32 hours + $80

Training (per model):
  Llama 3.1 8B:       2 hours GPU time
  Llama 3.3 70B:      3 hours GPU time
  Gemma 2 9B:         2 hours GPU time
  Total per model: 2-3 hours

Typical: $100-200 annotation + $20-50 compute = $120-250
```

### Persona Vectors

```
Per trait:
  Generate positive responses:        20 min (M1 Pro)
  Judge with GPT-4:                   5 min
  Generate negative responses:        20 min (M1 Pro)
  Extract vector:                     10 min (GPU)
  Total per trait: ~55 min

For 4 traits:
  Time: 3-4 hours total
  Cost: $20-24 (GPT-4 only)
  
Using Colab GPU:
  Time: 30-40 min
  Cost: Free

Typical: $20-40 total (vs $100+ for probes)
```

---

## When to Use Which

### Use Hallucination Probes If:
- You need factuality checking (real hallucinations, not behavior)
- You have long-form text to analyze
- Cost/time isn't critical (annotation is expensive)
- You want production-ready code (demo UI included)
- You're checking multiple models (one annotation, many probes)

### Use Persona Vectors If:
- You want to study trait dynamics during generation
- You need fast, cheap extraction
- You want interpretable projections
- You're combining with SAEs for feature analysis
- You want to steer behavior via activation modification
- You're doing research (not production deployment)

### Use Both If:
```
Detection: Hallucination probes find problematic spans
Analysis: Persona vectors explain why (trait activation)
Steering: Use trait vectors to adjust behavior
Decomposition: SAEs break down the mechanism
→ Full interpretability stack
```

---

## Key Differences in One Sentence

| Aspect | Probe | Vector |
|--------|-------|--------|
| **What it detects** | "Is this fact-checking true?" | "How much trait is expressed?" |
| **When it decides** | During generation | After generation |
| **How it trains** | Gradient descent on labels | Filtered averaging |
| **Interpretability** | Black box | Linear projection |
| **Speed** | 1-2 ms overhead | 0.2 ms overhead |
| **Cost** | $100-500 + days | $20-40 + hours |
| **Scalability** | Limited by annotation | Limited by questions |

---

## Implementation Examples

### Hallucination Probes
```python
from probe.config import ProbeConfig, TrainingConfig
from probe.train import main as train_probe

config = TrainingConfig(
    probe_config=ProbeConfig(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        layer=30,
        lora_layers="all"
    ),
    lambda_kl=0.5,
    train_datasets=[...]
)
train_probe(config)

# Use
from probe.value_head_probe import setup_probe
model, probe = setup_probe(model, config.probe_config)
outputs = probe(input_ids, attention_mask)
hallucination_prob = torch.sigmoid(outputs["probe_logits"])
```

### Persona Vectors
```python
from core.generate_vec import save_persona_vector

save_persona_vector(
    model_name="google/gemma-2-9b-it",
    pos_path="eval/outputs/paternalism_pos.csv",
    neg_path="eval/outputs/paternalism_neg.csv",
    trait="paternalism",
    save_dir="persona_vectors/gemma-2-9b"
)

# Use
import torch
vector = torch.load("persona_vectors/gemma-2-9b/paternalism_response_avg_diff.pt")
hidden = model(input_ids, output_hidden_states=True).hidden_states[layer]
trait_score = torch.sigmoid(hidden @ vector[layer])
```

---

## References

- **Hallucination Probes**: /tmp/hallucination_probes
  - Paper: arxiv.org/abs/2509.03531
  - GitHub: Pretrained models at obalcells/hallucination-probes

- **Persona Vectors**: /Users/ewern/Desktop/code/per-token-interp
  - Based on: safety-research/persona_vectors
  - Extended with: per-token monitoring, 4 new traits

- **SAE Decomposition**: gemma_scope
  - 400+ pre-trained SAEs for Gemma 2

