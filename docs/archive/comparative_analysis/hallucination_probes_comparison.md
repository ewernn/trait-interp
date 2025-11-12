# Comparative Analysis: Hallucination Probes vs. Persona Vectors

## Executive Summary

The hallucination detection probes (hallucination_probes) and persona vectors (per-token-interp) represent two distinct but complementary approaches to extracting interpretable representations from language models. Both extract vectors from hidden states, but differ fundamentally in their:
- **Target phenomena** (hallucinations vs. character traits)
- **Training methodology** (supervised classification vs. contrastive difference)
- **Data requirements** (annotated long-form text vs. synthetic prompted responses)
- **Inference mechanism** (online linear head vs. offline vector projection)

---

## 1. Extraction Methodology

### 1.1 Hallucination Probes: Supervised Classification

**Architecture:**
```
Model hidden states → Hook at layer L → Linear classification head → Binary logits
                                            (1D projection to [0,1])
```

**Data Collection Process:**

1. **Data Generation** (implicit):
   - Uses existing datasets: LongFact, LongFact++, HealthBench, TriviaQA
   - Models generate long-form completions naturally

2. **Annotation Pipeline** (external validation):
   - Uses Claude with web search (annotation_pipeline/annotate.py)
   - Entity-level spans annotated as:
     - "Supported" (0.0)
     - "Not Supported/Hallucinated" (1.0)
     - "Insufficient Information" (-100.0 ignored)
   - Token-level labels mapped from span annotations
   - Span position matching via fuzzy string matching (try_matching_span_in_text)

3. **Feature Extraction**:
   - Forward hooks capture hidden states at designated layer L
   - Layer selection: typically layer N-2 (e.g., layer 30 for Llama 3.1 8B with 32 layers)
   - **No averaging** - uses raw token representations directly
   - Applies padding, attention masking

**Loss Functions** (multiple):
```python
# Token-level binary cross-entropy
bce_loss = F.binary_cross_entropy_with_logits(
    probe_logits[valid_tokens], 
    classification_labels[valid_tokens],
    weight=classification_weights
)

# Span-level max aggregation (newer approach)
max_aggr_loss = mean([
    BCE(max(logits_in_span), 1.0) for span in positive_spans,
    BCE(max(logits_in_span), 0.0) for span in negative_spans
])

# Optional: KL divergence regularization (LoRA only)
kl_loss = KL(q=P(logits_with_lora), p=P(logits_base_model))

# Combined: lambda_lm * lm_loss + lambda_kl * kl_loss + (1-lambda) * probe_loss
```

### 1.2 Persona Vectors: Contrastive Difference

**Architecture:**
```
Model hidden states → Average over tokens → Layer-wise difference vector
  [prompt + response]     per response         positive_response_mean 
                                            - negative_response_mean
                                            = [L, hidden_dim] vector
```

**Data Collection Process:**

1. **Trait Definition** (manual):
   - Define 5 instruction pairs (pos/neg) for trait
   - 20 extraction questions
   - 20 evaluation questions
   - Creates distinct prompt/response distributions

2. **Response Generation** (prompted generation):
   - Run LLM with explicit persona instructions
   - Two datasets:
     - **Positive**: "Be extremely [trait]" → generates high-trait responses
     - **Negative**: "Avoid any [trait]" → generates low-trait responses
   - ~200 responses per condition (10 responses × 20 questions)

3. **Response Evaluation** (GPT-4 judging):
   - Judge responses on trait scale 0-100
   - Include coherence score
   - Filter for effective pairs:
     - Positive trait score ≥ threshold (e.g., 50)
     - Negative trait score < (100 - threshold)
     - Both coherence ≥ 50
   - Retain ~150 effective pairs per condition

4. **Feature Extraction**:
   ```python
   # For each effective response pair
   prompt_len = len(tokenize(prompt))
   hidden = model(prompt + response, output_hidden_states=True)
   
   # Three aggregation strategies:
   prompt_avg[layer] = mean(hidden[layer, :prompt_len, :])    # [hidden_dim]
   response_avg[layer] = mean(hidden[layer, prompt_len:, :])  # [hidden_dim]
   prompt_last[layer] = hidden[layer, prompt_len-1, :]        # [hidden_dim]
   
   # Per-layer difference
   vector[layer] = mean(positive_responses[layer]) 
                 - mean(negative_responses[layer])  # [hidden_dim]
   ```

5. **Output Format**:
   - Three vector files per trait:
     - `trait_prompt_avg_diff.pt` - [num_layers, hidden_dim]
     - `trait_response_avg_diff.pt` - [num_layers, hidden_dim]
     - `trait_prompt_last_diff.pt` - [num_layers, hidden_dim]

---

## 2. Contrastive Data Comparison

### Hallucination Probes

**Data Characteristics:**
- **Source**: Real, naturally-generated long-form text
- **Hallucination definition**: Factually incorrect claims (verified by Claude + web search)
- **Label distribution**: Imbalanced
  - Most tokens: no annotation (default_ignore=True in some configs)
  - Positive (hallucinations): 10-15% of annotated tokens
  - Negative (supported): 85-90% of annotated tokens
- **Contrastive framing**: Implicit
  - Hallucinations emerge naturally from model's knowledge gaps
  - No explicit "non-hallucinated" generation - just natural alternatives
- **Dataset sizes**:
  - LongFact: ~500 QA pairs
  - LongFact++: ~500 augmented pairs
  - HealthBench: ~250 medical QA pairs
  - TriviaQA: ~13K factoid pairs

**Key insight**: No explicit opposite-condition generation. Contrast emerges from natural variation in model behavior.

### Persona Vectors

**Data Characteristics:**
- **Source**: Synthetically generated with explicit instructions
- **Trait definition**: Behavioral characteristics (paternalism, deception, etc.)
- **Label distribution**: Perfectly balanced
  - 150 positive examples (trait score ≥50)
  - 150 negative examples (trait score <50)
- **Contrastive framing**: Explicit
  - Same questions answered with opposite instructions
  - Deliberately elicits maximally different responses
- **Evaluation**: Multi-step
  1. LLM generation (hundreds of candidates)
  2. GPT-4 judging (0-100 scores + coherence)
  3. Filtering (keep high-confidence pairs)
  4. Vector extraction (from filtered set)

**Key insight**: Strong, explicit contrast. Deliberately constructs maximally different activation patterns.

---

## 3. Training/Extraction Pipeline

### Hallucination Probes Training Pipeline

**File Structure:**
```
probe/
├── config.py              # ProbeConfig, TrainingConfig
├── types.py               # AnnotatedSpan, ProbingItem
├── dataset.py             # TokenizedProbingDataset, dataset conversion
├── value_head_probe.py    # ValueHeadProbe class
├── train.py               # Main training script
├── trainer.py             # ProbeTrainer (custom Trainer subclass)
├── loss.py                # Loss functions
└── evaluate.py            # Evaluation
```

**Training Procedure:**

```python
# 1. Load model and setup probe
model, tokenizer = load_model_and_tokenizer("meta-llama/Meta-Llama-3.1-8B-Instruct")
model, probe = setup_probe(model, probe_config)
# Freezes base model parameters, adds optional LoRA adapters

# 2. Load and process datasets
train_dataset = create_probing_dataset(config, tokenizer)
# Tokenizes prompt+completion, maps spans to tokens, creates labels

# 3. Forward pass during training
outputs = probe(input_ids, attention_mask, labels=lm_labels)
# Hook captures hidden_states[layer_idx] → passes to linear head

# 4. Compute multi-part loss
probe_loss = BCE(probe_logits, classification_labels)  # Token-level
max_aggr_loss = span_max_BCE(probe_logits, spans)      # Span-level
kl_loss = KL(logits_with_lora, logits_base)            # Regularization

# 5. Combined optimization
loss = λ_lm * lm_loss + λ_kl * kl_loss + (1-λ_lm-λ_kl) * probe_loss

# 6. Separate learning rates
optimizer = AdamW([
    {'params': probe_head, 'lr': probe_head_lr},
    {'params': lora_params, 'lr': lora_lr}
])
```

**Key Parameters** (from train_config.yaml):
- `layer`: 30 (Llama 3.1 8B, second-to-last)
- `lora_layers`: "all" (applies to layers 0-30)
- `lora_r`: 16 (LoRA rank)
- `probe_head_lr`: 1e-3
- `lora_lr`: 1e-4
- `lambda_lm`: 0.0 (optional LM loss regularization)
- `lambda_kl`: 0.5 (KL divergence weight)
- `anneal_warmup`: 1.0 (ramp up span loss from 0 to 1)
- `pos_weight`: 10.0, `neg_weight`: 10.0 (class weighting)

**Output Format:**
```
value_head_probes/{probe_id}/
├── adapter_config.json        # LoRA configuration
├── adapter_model.bin          # LoRA weights
├── probe_head.bin             # Linear layer weights
├── probe_config.json          # Layer index, hidden size
├── tokenizer_config.json      # Tokenizer config
├── training_config.json       # Full training config
└── evaluation_results.json    # Metrics (AUROC, etc.)
```

### Persona Vectors Extraction Pipeline

**File Structure:**
```
core/
├── generate_vec.py     # Main vector extraction
├── training.py         # SFT utilities (if needed)
├── judge.py           # GPT-4 evaluation
└── utils.py
eval/
├── eval_persona.py    # Response generation + judging
└── model_utils.py     # Device handling (M1 MPS support)
```

**Extraction Procedure:**

```python
# Step 1: Generate positive responses
responses_pos = []
for question in extraction_questions:
    for i in range(n_per_question):  # e.g., 10
        prompt = format_with_instruction(question, instruction="pos")
        response = model.generate(prompt)
        responses_pos.append(response)

# Step 2: Judge responses (GPT-4)
scored_responses_pos = []
for response in responses_pos:
    score, coherence = gpt4_judge(response, eval_prompt)
    scored_responses_pos.append({
        'response': response,
        'trait_score': score,
        'coherence': coherence
    })

# Step 3: Filter effective examples
mask = (trait_score >= 50) & (coherence >= 50)
effective_pos = scored_responses_pos[mask]  # ~150 examples

# Step 4: Extract activations (same for pos/neg)
prompt_avg_pos, prompt_last_pos, response_avg_pos = get_hidden_p_and_r(
    model, tokenizer, 
    prompts=[ex['prompt'] for ex in effective_pos],
    responses=[ex['response'] for ex in effective_pos],
    layer_list=range(num_layers)
)

# Step 5: Compute differences (per-layer mean subtraction)
persona_vector = [
    response_avg_pos[layer].mean(0) - response_avg_neg[layer].mean(0)
    for layer in range(num_layers)
]
# Result: [num_layers, hidden_dim] tensor

# Step 6: Save vectors
torch.save(persona_vector, f"{save_dir}/{trait}_response_avg_diff.pt")
```

**Output Format:**
```
persona_vectors/{model}/
├── {trait}_prompt_avg_diff.pt      # [num_layers, hidden_dim]
├── {trait}_response_avg_diff.pt    # [num_layers, hidden_dim]
└── {trait}_prompt_last_diff.pt     # [num_layers, hidden_dim]
```

---

## 4. Probe Application During Generation

### Hallucination Probes: Online Inference

**Mechanism:**

```python
# Real-time hallucination detection during generation
with torch.no_grad():
    while not done_generating:
        # 1. Forward pass through model
        outputs = model(input_ids, output_hidden_states=True)
        
        # 2. Capture hidden state from target layer
        hidden_state = outputs.hidden_states[layer_idx]  # [1, 1, hidden_dim]
        
        # 3. Apply probe head
        hallucination_logit = probe_head(hidden_state)  # [1, 1, 1]
        hallucination_prob = sigmoid(hallucination_logit)  # [0, 1]
        
        # 4. Get next token from LM
        logits = outputs.logits[:, -1, :]
        next_token = sample(logits)
        
        # 5. Color-code in UI based on probability
        if hallucination_prob > 0.5:
            display_token(next_token, color='red')  # Hallucination alert
        else:
            display_token(next_token, color='green')  # Confident
        
        # Update input for next iteration
        input_ids = append(input_ids, next_token)
```

**Key Features:**
- **Latency**: Low overhead - just one linear layer (hidden_dim → 1)
- **Architecture**: Single linear layer with no hidden units
- **Threshold**: Default 0.5 (configurable)
- **Regularization**: Optional KL divergence to preserve LM behavior
- **Multi-layer support**: LoRA + probe head can be applied without changing LM

### Persona Vectors: Offline Projection

**Mechanism:**

```python
# Post-hoc trait monitoring during/after generation
activations = []
with torch.no_grad():
    while not done_generating:
        # 1. Forward pass
        outputs = model(input_ids, output_hidden_states=True)
        
        # 2. Capture hidden state
        hidden = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
        activations.append(hidden[:, -1, :].cpu())  # Save last token's activation
        
        # 3. Get next token
        logits = outputs.logits[:, -1, :]
        next_token = sample(logits)
        input_ids = append(input_ids, next_token)

# After generation: compute trait scores per-token
persona_vector = torch.load("paternalism_response_avg_diff.pt")  # [L, H]

trait_scores = []
for token_idx, activation in enumerate(activations):
    # Project onto persona vector at corresponding layer
    score = activation[layer_idx] @ persona_vector[layer_idx].T  # scalar
    
    # Normalize to [0, 1]
    trait_prob = sigmoid(score)
    trait_scores.append(trait_prob)

# Analyze temporal dynamics
plot_trait_over_time(trait_scores)
```

**Key Features:**
- **Mechanism**: Linear projection (dot product)
- **Computational cost**: Negligible - simple vector operation
- **No training required**: Pre-computed vectors, no gradient computation
- **Temporal analysis**: Naturally captures per-token dynamics
- **Multiple representations**: Can project prompt vs. response activations separately

---

## 5. Key Architectural Differences

| Aspect | Hallucination Probes | Persona Vectors |
|--------|---------------------|-----------------|
| **Task** | Binary classification | Behavioral trait detection |
| **Training** | Supervised (binary labels) | Unsupervised (contrastive difference) |
| **Probe architecture** | Linear head (hidden_dim → 1) | None (pre-computed vectors) |
| **Data source** | Real long-form generations | Synthetically prompted responses |
| **Annotation** | Claude + web search | GPT-4 scoring (0-100) |
| **Label type** | Token-level binary | Response-level scalar |
| **Contrastive signal** | Implicit (natural variation) | Explicit (instruction-guided) |
| **Inference** | Online (during generation) | Offline projection |
| **Latency** | ~1ms per token | ~microseconds |
| **Hyperparameters** | Extensive (LR, regularization) | Minimal (threshold for filtering) |
| **Output** | Single probability per token | Vector per layer |
| **Regularization** | KL divergence (LoRA) | None (difference naturally regularizes) |

---

## 6. Detailed Loss Function Comparison

### Hallucination Probes Loss

```python
# Multi-component loss with annealing

# 1. Token-level Binary Cross-Entropy (primary)
def compute_probe_bce_loss(probe_logits, labels, weights):
    """
    Args:
        probe_logits: [batch, seq_len]
        labels: [batch, seq_len] ∈ {-100, 0, 1}
        weights: [batch, seq_len] for class weighting
    """
    # Clip to prevent numerical issues
    probe_logits_clipped = torch.clamp(probe_logits, min=-100, max=100)
    
    # Token-level BCE
    bce = F.binary_cross_entropy_with_logits(
        probe_logits_clipped, labels, 
        weight=weights, reduction='none'
    )
    # Mask out ignored tokens (-100)
    return bce[labels != -100].mean()

# 2. Span-level Max Aggregation Loss (emergent)
def compute_probe_max_aggregation_loss(probe_logits, spans, labels):
    """
    Span-level loss: max(logits_in_span) should predict span label
    For positive hallucination spans: BCE(max(...), 1.0)
    For negative supported spans: BCE(max(...), 0.0)
    """
    span_losses = []
    
    for span_start, span_end in positive_spans:
        max_logit = probe_logits[span_start:span_end].max()
        loss = F.binary_cross_entropy_with_logits(
            max_logit, torch.tensor(1.0)
        )
        span_losses.append(loss)
    
    for span_start, span_end in negative_spans:
        max_logit = probe_logits[span_start:span_end].max()
        loss = F.binary_cross_entropy_with_logits(
            max_logit, torch.tensor(0.0)
        )
        span_losses.append(loss)
    
    return torch.mean(torch.stack(span_losses))

# 3. KL Divergence Regularization (LoRA only)
def compute_kl_divergence_loss(model_with_lora, model_without_lora):
    """
    Ensures LoRA doesn't change LM distribution too much
    KL(P_lora || P_base) at token level
    """
    with model.disable_adapter():
        base_logits = model(input_ids).logits
    
    lora_logits = model(input_ids).logits
    
    kl = F.kl_div(
        torch.log_softmax(lora_logits, dim=-1),
        torch.softmax(base_logits, dim=-1),
        reduction='batchmean'
    )
    return kl

# 4. Total Loss (combines all three)
loss_training_progress = min(1.0, step / warmup_steps)
probe_loss = (1 - loss_training_progress) * token_bce 
           + loss_training_progress * span_max_bce

total_loss = λ_lm * lm_loss 
           + λ_kl * kl_loss 
           + (1 - λ_lm - λ_kl) * probe_loss

# Example configs:
# "linear": λ_lm=0.0, λ_kl=0.0 → pure classification
# "lora_lambda_kl_0_05": λ_lm=0.0, λ_kl=0.05 → strong KL regularization
# "lora_lambda_lm_0_01": λ_lm=0.01, λ_kl=0.0 → LM regularization
```

### Persona Vectors (No Training Loss)

```python
# No gradient-based training - purely contrastive difference

# Filter effective examples
mask = (positive_scores >= threshold) & (negative_scores < 100 - threshold)

# Compute statistics
positive_mean = mean(positive_activations[mask])  # [hidden_dim]
negative_mean = mean(negative_activations[mask])  # [hidden_dim]

# Difference vector (fully deterministic, no randomness)
trait_vector = positive_mean - negative_mean  # [hidden_dim]

# Optional: normalize by magnitude
trait_vector_normalized = trait_vector / (torch.norm(trait_vector) + eps)
```

**Key insight**: Persona vectors use **no loss function**. The contrastive signal comes directly from filtering + averaging, which naturally selects for high-variance directions in activation space.

---

## 7. Model Architecture & Layer Selection

### Hallucination Probes

```python
# Explicit layer selection with multi-layer LoRA

ProbeConfig:
  layer: int = 30  # Target layer for probe head
  lora_layers: List[int] | "all" = "all"  # Which layers get LoRA

# For Llama 3.1 8B (32 layers):
#   layer=30  # Second-to-last (penultimate) layer
#   lora_layers=[0, 1, ..., 30]  # All layers up to probe layer

# LoRA structure:
#   For each target layer:
#     query_proj: hidden_dim × hidden_dim
#     value_proj: hidden_dim × hidden_dim
#     Apply LoRA with rank=16, alpha=32

# Hook mechanism:
#   register_forward_hook(ValueHeadProbe.target_module)
#   Captures module_output → passes to linear head
#   No intervention in model computation
```

### Persona Vectors

```python
# No model modification - pure analysis layer

# Layer-wise vectors stored separately
vector_layer_0 = torch.load("trait_response_avg_diff.pt")[0]  # [hidden_dim]
vector_layer_1 = torch.load("trait_response_avg_diff.pt")[1]  # [hidden_dim]
...
vector_layer_41 = torch.load("trait_response_avg_diff.pt")[41]  # [hidden_dim]

# Can analyze all layers without modification
# Typically examine:
#   - Early layers (< L/4): low-level features
#   - Middle layers (L/4 to 3L/4): abstraction formation
#   - Late layers (> 3L/4): output planning

# Integration with SAEs:
#   Use pre-trained SAEs to decompose trait vectors into features
#   Example: Gemma Scope has 400+ SAEs for Gemma 2 9B
```

---

## 8. Data Requirements & Costs

### Hallucination Probes

**Computation:**
```
Training:
  - Load model: ~8 min
  - Tokenize datasets: ~10 min (cached after first run)
  - Training: ~1-2 hours (per model, batch_size=4, 1-2 epochs)
  - Evaluation: ~30 min
  Total: 2-3 hours per probe

Annotation (one-time, for new datasets):
  - LongFact generation: depends on model (30-60 min)
  - Claude annotation with web search: ~$5-10 per 500 responses
  - Total: ~2-3 hours + $20-30
```

**Data Volume:**
- LongFact: ~500 QA pairs
- LongFact++: ~500 augmented pairs
- HealthBench: ~250 medical QA pairs
- Total training: ~1250 prompts

**Annotation Overhead:**
- Claude with web search: required for real-time hallucination detection
- One-time investment, reusable across models

### Persona Vectors

**Computation:**
```
Per trait extraction:
  - Positive response generation: ~20 min (transformers on M1)
  - GPT-4 judging: ~5 min (400 responses)
  - Negative response generation: ~20 min
  - Vector extraction: ~10 min (GPU) or ~30 min (CPU)
  - Total: ~55 min per trait

For 4 traits: ~3-4 hours total
Cost: ~$4-6 per trait (GPT-4 judging) = $20-24 total

Alternative: Colab with T4 GPU
  - Total time: ~30-40 minutes for full pipeline
  - Cost: Free (Colab)
```

**Data Volume:**
- 20 extraction questions × 10 responses = 200 candidates per condition
- After filtering: ~150 effective pairs per condition
- Total: ~300 activations per trait

**Annotation Overhead:**
- GPT-4 scoring at scale: unavoidable
- No web search needed - purely behavioral scoring
- Much cheaper than Claude + web search

---

## 9. Inference Time & Scalability

### Hallucination Probes

```python
# Per-token inference cost

def inference(input_ids, attention_mask):
    # 1. Full LM forward pass: ~50-100 ms (first token), ~10-20 ms (cache)
    with add_hooks(module_forward_hooks=[(target_module, hook_fn)]):
        outputs = model(input_ids, attention_mask)
    
    # 2. Capture hidden state (already hooked): ~0 ms overhead
    hidden = hooked_hidden_states  # [batch, seq_len, hidden_dim]
    
    # 3. Probe head (linear, no activation): ~1 ms
    probe_logits = probe_head(hidden)  # [batch, seq_len, 1]
    
    # 4. Sigmoid + threshold: <1 ms
    probs = torch.sigmoid(probe_logits)
    
    # Total overhead: ~1-2 ms per token (negligible vs. LM)
    # Scales linearly with batch size
```

### Persona Vectors

```python
# Per-token projection cost

def inference(hidden_states):
    # hidden_states: [batch, seq_len, hidden_dim]
    # persona_vector: [num_layers, hidden_dim]
    
    # 1. Index into layer: ~0 ms
    layer_hidden = hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
    
    # 2. Dot product: ~100-200 microseconds
    scores = layer_hidden @ persona_vector.T  # [batch, seq_len]
    
    # 3. Sigmoid: ~100 microseconds
    probs = torch.sigmoid(scores)
    
    # Total: ~0.2 ms per token
    # No gradient computation, no model modification
    # Scales linearly with hidden_dim
```

**Latency Comparison:**
- Hallucination probes: ~1-2 ms overhead (probe head)
- Persona vectors: ~0.2 ms overhead (projection)
- **Persona vectors 5-10x faster** due to simpler computation

---

## 10. Validation & Evaluation

### Hallucination Probes

**Metrics:**
```python
# Token-level (primary)
- ROC AUC: How well does probe rank hallucinated vs. supported tokens?
- PR AUC: Precision-recall curve (important for imbalanced data)
- F1 @ threshold: Binary classification at threshold=0.5

# Span-level (aggregate)
- Span AUC: Does max(probe_logits_in_span) correctly classify span?
- Span F1: Span-level precision and recall
- Span precision/recall: How many hallucinated vs. supported spans detected?

# Model behavior
- Sparsity: Mean probe activation (should be low, not flag everything)
- LM loss: Language modeling loss (should be similar to base model)
- KL divergence: Divergence with base model (for LoRA probes)

# Example results (from paper, Llama 3.1 8B):
Span-level AUC: 0.88-0.92 (varies by dataset)
LM loss impact: <0.1% change with LoRA + KL regularization
Sparsity: ~0.15 (15% of tokens flagged as hallucinations)
```

### Persona Vectors

**Validation:**
```python
# Extraction phase (during training)
- Coherence filtering: Keep responses with coherence >= 50/100
- Trait score filtering: Keep |score - 50| >= threshold
- Statistical tests: Ensure pos/neg distributions don't overlap

# Evaluation phase (after extraction)
- Test set prompts: New questions, same trait definition
- Human/GPT-4 evaluation: Score responses for trait expression
- Correlation analysis: Do projected scores match human scores?
- SAE decomposition: Can we identify interpretable features?

# Example metrics (from paper):
- Mean trait score (positive responses): 75/100
- Mean trait score (negative responses): 25/100
- Extraction correlation: r = 0.85 with test set
- Steering success: 70%+ of responses show trait when vector applied
```

---

## 11. Practical Differences in Usage

### Hallucination Probes: When to Use

**Best for:**
- Real-time hallucination detection during generation
- Long-form text generation (documents, articles)
- Fact-checking critical domains (medical, legal)
- One-time investment in annotation pays off over many models

**Advantages:**
- Ground truth from web search (most reliable)
- Works on any text (not just responses to specific questions)
- Trained to rank tokens within context
- Production-ready (demo included)

**Limitations:**
- Requires annotated training data (expensive upfront)
- Task-specific (hallucination vs. other phenomena)
- Token-level classification (no temporal analysis)
- Slower inference (~1-2 ms)

### Persona Vectors: When to Use

**Best for:**
- Monitoring behavioral traits during generation
- Per-token temporal dynamics (when does trait emerge?)
- Steering model behavior via activation addition
- Decomposition with SAEs (what components matter?)
- Fast iteration (cheap extraction via GPT-4)

**Advantages:**
- Fast extraction (hours not days)
- No gradient computation required
- Simple linear projection (interpretable)
- Works with SAEs for feature decomposition
- Captures emergent trait dynamics

**Limitations:**
- Synthetic data (less grounded than real hallucinations)
- Trait-specific (new trait = new extraction)
- Subjective scoring (depends on judge quality)
- Requires curated instruction pairs
- Not suitable for factuality checking

---

## 12. Technical Implementation Comparison

### Model Modification Strategy

**Hallucination Probes:**
```python
# Freezes base model, adds trainable components
for param in model.parameters():
    param.requires_grad = False  # Freeze LM

# Add LoRA adapters (optional)
model = get_peft_model(model, lora_config)

# Add probe head (always trainable)
probe = ValueHeadProbe(model, layer_idx=30)

# Training updates:
# - LoRA parameters (if used)
# - Probe head weights
# - No model weights modified
```

**Persona Vectors:**
```python
# Zero modification to model
# All computation post-hoc

# Load model (inference mode)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Forward passes only (no gradients)
with torch.no_grad():
    hidden = model(input_ids, output_hidden_states=True)

# Analysis happens after generation
# Vector projection is pure computation
```

### Data Format & Storage

**Hallucination Probes:**
```
value_head_probes/llama3_1_8b_lora_lambda_kl_0_05/
├── adapter_config.json           # [LoRA config]
├── adapter_model.bin             # [LoRA weights, ~8 MB]
├── probe_head.bin                # [Linear layer, ~16 MB]
├── probe_config.json             # [Metadata]
├── tokenizer_config.json         # [Tokenizer]
├── training_config.json          # [Full config]
└── evaluation_results.json       # [Metrics]

Total size: ~30-50 MB per probe
Can load as: PeftModel.from_pretrained() + ValueHeadProbe.load_head()
```

**Persona Vectors:**
```
persona_vectors/gemma-2-9b/
├── paternalism_prompt_avg_diff.pt    # [42, 3584] = ~580 KB
├── paternalism_response_avg_diff.pt  # [42, 3584] = ~580 KB
├── paternalism_prompt_last_diff.pt   # [42, 3584] = ~580 KB
├── deception_response_avg_diff.pt    # [42, 3584] = ~580 KB
└── ...

Total size: ~2-3 MB per model (4 traits)
Load as: torch.load() then simple projection
```

**Storage Efficiency:**
- Hallucination probes: 30-50 MB per probe (due to LoRA)
- Persona vectors: ~150-200 KB per trait vector
- **Persona vectors 100-200x smaller** due to no model parameters

---

## 13. Integration with Other Techniques

### Hallucination Probes + SAEs

**Not directly explored in current code**, but possible:
```python
# Could decompose probe attention patterns using SAEs
# Extract which features contribute to hallucination signal

# Example workflow:
probe_features = decompose_with_sae(probe_head.weight)
# Which SAE features activate strongly?
```

### Persona Vectors + SAEs

**Explicitly supported:**
```python
# Use Gemma Scope SAEs to decompose trait vectors
sae = load_sae("gemma_scope", layer=20)

# Decompose trait vector
trait_features = sae.encode(persona_vector[layer])  # sparse features
reconstruction = sae.decode(trait_features)

# Identify which features drive the trait
important_features = top_k(trait_features, k=10)
feature_names = interpret_sae_features(important_features)
```

**This is a major advantage**: Persona vectors decompose naturally into interpretable features.

---

## 14. Comparative Strengths & Weaknesses

### Hallucination Probes

**Strengths:**
- Ground truth from web search (maximally reliable)
- Works on real, natural language (not synthetic)
- Production-ready (demo UI included)
- Span-level aggregation (structured output)
- Multiple loss options (flexibility)
- Regularization strategies (KL divergence)

**Weaknesses:**
- High annotation cost upfront ($100s)
- Slow inference due to linear layer (~1-2 ms)
- Not interpretable (black box classification)
- Task-specific (must retrain for each phenomenon)
- Requires dataset-specific training

### Persona Vectors

**Strengths:**
- Cheap and fast extraction (hours not days)
- Interpretable (linear projection)
- Works with SAEs for feature decomposition
- Captures temporal dynamics naturally
- Enables steering (can add vector to activations)
- Universal vectors (one vector for all models)

**Weaknesses:**
- Synthetic, subjective data (less grounded)
- Requires explicit instruction pairs (manual trait definition)
- No web search validation
- Slower overall process (200+ generations + judging)
- Trait-specific (can't detect hallucinations, only traits)
- Depends on GPT-4 quality for judging

---

## 15. When to Combine Approaches

### Hybrid Usage Patterns

**Pattern 1: Detect then Interpret**
```
Hallucination probes: Find hallucinated spans (real-time)
→ Decompose with SAEs (post-hoc analysis)
→ Understand which features enabled the hallucination
```

**Pattern 2: Monitor then Steer**
```
Persona vectors: Monitor trait during generation (per-token)
→ Detect trait activation points
→ Add/subtract vector to steer away from harmful trait
```

**Pattern 3: Validate then Extract**
```
Hallucination probes: Identify factually supported vs. unsupported
→ Use as labeled data for persona vector extraction
→ Create trait definitions from probe predictions
```

**Pattern 4: Ensemble Detection**
```
Hallucination probes: Binary classification (fast filtering)
→ Persona vectors: Behavioral context (why hallucinate?)
→ Combined signal for more nuanced analysis
```

---

## 16. Summary Table

| Feature | Hallucination Probes | Persona Vectors |
|---------|---------------------|-----------------|
| **Primary Use** | Real-time hallucination detection | Behavioral trait monitoring |
| **Data Source** | Real generations + annotation | Synthetic prompted responses |
| **Annotation** | Claude + web search | GPT-4 scoring |
| **Training** | Supervised classification | Contrastive filtering |
| **Architecture** | Linear head + optional LoRA | Pre-computed vectors |
| **Probe Mechanism** | Forward hook + linear | Dot product |
| **Inference** | Online (during generation) | Offline (post-hoc) |
| **Latency** | ~1-2 ms | ~0.2 ms |
| **Model Size** | 30-50 MB | <1 MB |
| **Interpretability** | Low (black box) | High (linear) |
| **SAE Integration** | Possible (not done) | Native support |
| **Cost/Time** | $100-500 + 2-3 days | $20-40 + 3-4 hours |
| **Validation** | AUROC, F1, sparsity | Coherence, correlation |
| **Scalability** | Limited to annotation budget | Limited to question set |
| **Generalization** | Across models (via transfer) | Single vector for trait |

---

## 17. Code Organization Insights

### Hallucination Probes: Clean Separation

```
probe/
  ├── config.py         → Configuration management
  ├── types.py          → Data structures
  ├── dataset.py        → Data loading and processing
  ├── loss.py           → Loss functions (5+ variants)
  ├── value_head_probe.py → Core architecture
  ├── train.py          → Training orchestration
  ├── trainer.py        → Custom HF Trainer
  └── evaluate.py       → Evaluation pipeline

annotation_pipeline/
  ├── annotate.py       → Claude annotation with search
  └── data_models.py    → Annotation data structures

utils/
  ├── hooks.py          → Forward hook context manager
  ├── model_utils.py    → Model loading
  └── ...               → Other utilities
```

**Design pattern**: Heavy use of configuration objects, separation of concerns, standardized data structures.

### Persona Vectors: Lightweight Approach

```
core/
  ├── generate_vec.py   → Main extraction (simple function)
  ├── judge.py          → GPT-4 scoring
  └── utils.py          → Utilities

eval/
  ├── eval_persona.py   → Generation + judging
  └── model_utils.py    → Device handling (M1 support)

data_generation/
  └── trait_data_*/    → Trait definitions (JSON)
```

**Design pattern**: Functional programming style, minimal abstractions, explicit steps in main script.

---

## Conclusion

Both approaches are valuable but serve different purposes:

1. **Hallucination Probes** are for **detection and real-time monitoring** of factual errors in long-form text. They leverage expensive annotation but provide production-ready solutions.

2. **Persona Vectors** are for **understanding behavioral traits** and enabling **steering via activation modification**. They're cheaper, faster, and more interpretable but rely on synthetic data.

The ideal research program combines both: use hallucination probes for ground truth, use persona vectors for behavioral understanding, decompose both with SAEs for interpretability.

