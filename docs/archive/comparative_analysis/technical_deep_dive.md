# Technical Deep Dive: Algorithms & Architectures

## Part 1: Hallucination Probes - Supervised Classification

### 1.1 ValueHeadProbe Architecture

```python
class ValueHeadProbe(nn.Module):
    """
    A probe that:
    1. Hooks into hidden states at a specific layer
    2. Applies a learned linear transformation to binary logits
    3. Optionally uses LoRA for minimal model modification
    """
    
    def __init__(self, model, layer_idx=None, path=None):
        self.model = model
        self.layer_idx = layer_idx  # Which transformer layer to hook
        self.target_module = model.model.layers[layer_idx]  # Direct reference
        
        # Linear head: [hidden_dim] → [1] (binary classification)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Forward hook to capture hidden states
        self._hooked_hidden_states = None
        self._hook_fn = self._get_hook_fn()
    
    def _get_hook_fn(self):
        """
        Capture hidden states during forward pass WITHOUT detaching.
        This allows gradients to flow if needed.
        """
        def hook_fn(module, module_input, module_output):
            # module_output: [batch_size, seq_len, hidden_dim]
            self._hooked_hidden_states = module_output[0]
        return hook_fn
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Reset hook state
        self._hooked_hidden_states = None
        
        # Register hook temporarily
        with add_hooks(module_forward_hooks=[(self.target_module, self._hook_fn)]):
            # Full model forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False  # Don't compute all hidden states
            )
        
        # Apply linear head to hooked hidden states
        probe_logits = self.value_head(self._hooked_hidden_states)
        # Shape: [batch_size, seq_len, 1] → squeeze to [batch_size, seq_len]
        
        return {
            'lm_logits': outputs.logits,      # For LM loss
            'probe_logits': probe_logits,     # For classification
            'lm_loss': outputs.loss           # For regularization
        }

# Key insight: Hook captures hidden states WITHOUT interfering with
# forward pass. Model computes normally, we just extract activations.
```

### 1.2 LoRA Integration

```python
# Base model: 7B Llama with 32 layers
# Modify only specific layers with LoRA adapters

lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # Scaling factor (α/r = 2)
    lora_dropout=0.05,       # Dropout on LoRA params
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # Which linear layers to adapt
)

# Apply to layers 0-30 (all except last 2)
for layer_idx in range(31):
    apply_lora_to_layer(model.layers[layer_idx], lora_config)

# Result: 
#   - q_proj, v_proj in each layer: [hidden_dim, hidden_dim]
#   - LoRA adds: 2 * hidden_dim * r per layer
#   - Total LoRA params: 2 * 31 * 4096 * 16 ≈ 4.1M per model
#   - Plus probe head: 4096 * 1 ≈ 4K
#   - Total: ~4.1M new parameters (0.06% of 7B model)

# Benefits:
# 1. Minimal compute overhead
# 2. Can be removed without affecting base model
# 3. Enables KL divergence regularization
```

### 1.3 Multi-Part Loss Function

```python
def compute_loss(model, batch):
    """
    Orchestrates three loss components with annealing
    """
    
    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['lm_labels']
    )
    
    lm_logits = outputs['lm_logits']           # [B, T, V]
    probe_logits = outputs['probe_logits']     # [B, T]
    lm_loss = outputs['lm_loss']               # Scalar
    
    # ============== COMPONENT 1: Token-level BCE ==============
    
    token_bce = compute_probe_bce_loss(
        probe_logits=probe_logits,              # [B, T]
        labels=batch['classification_labels'],  # [B, T] ∈ {-100, 0, 1}
        weights=batch['classification_weights'] # [B, T] ∈ {0, pos_w, neg_w}
    )
    # Only computes loss on non-ignored tokens (label != -100)
    
    # ============== COMPONENT 2: Span-level max aggregation ==============
    
    span_bce = compute_probe_max_aggregation_loss(
        probe_logits=probe_logits,
        spans={'pos': batch['pos_spans'], 'neg': batch['neg_spans']},
        labels=batch['classification_labels']
    )
    # For each positive span: BCE(max(logits_in_span), 1.0)
    # For each negative span: BCE(max(logits_in_span), 0.0)
    # Average across all spans in batch
    
    # ============== ANNEALING BETWEEN LOSS COMPONENTS ==============
    
    # Start with token BCE, gradually shift to span BCE
    progress = step / total_steps  # 0 → 1 over training
    progress_ratio = min(progress / warmup_fraction, 1.0)
    
    # Annealed probe loss
    probe_loss = (1 - progress_ratio) * token_bce + progress_ratio * span_bce
    
    # ============== COMPONENT 3: KL Divergence regularization ==============
    
    kl_loss = torch.tensor(0.)
    if lambda_kl > 0 and model_has_lora:
        # Compute KL between LoRA-adapted and base model distributions
        kl_loss = compute_kl_divergence_loss(
            model=model,
            lm_logits=lm_logits,
            input_ids=batch['input_ids'],
            lm_labels=batch['lm_labels']
        )
        # Measures: KL(P(·|tokens_with_lora) || P(·|tokens_base))
        # Should be small to preserve base model behavior
    
    # ============== FINAL COMBINED LOSS ==============
    
    total_loss = (
        lambda_lm * lm_loss +           # Optional: preserve LM capability
        lambda_kl * kl_loss +           # Optional: preserve base model distribution
        (1 - lambda_kl - lambda_lm) * probe_loss  # Main: classification task
    )
    
    return total_loss, {
        'lm_loss': lm_loss,
        'token_bce': token_bce,
        'span_bce': span_bce,
        'kl_loss': kl_loss,
        'total_loss': total_loss
    }

# Example configurations:
configs = {
    'linear': {
        'lambda_lm': 0.0,
        'lambda_kl': 0.0,
        # Pure probe training: loss = probe_loss
    },
    'lora_kl_0_05': {
        'lambda_lm': 0.0,
        'lambda_kl': 0.05,
        # Strong KL: loss = 0.05*kl + 0.95*probe_loss
    },
    'lora_lm_0_01': {
        'lambda_lm': 0.01,
        'lambda_kl': 0.0,
        # LM regularization: loss = 0.01*lm + 0.99*probe_loss
    }
}
```

### 1.4 Token-to-Span Label Mapping

```python
class TokenizedProbingDataset:
    """
    Maps span-level annotations to token-level labels.
    Key challenge: spans are character offsets, tokens are BPE.
    """
    
    def _compute_positional_labels(self, input_ids, item):
        """
        Input: 
            - Full tokenized prompt+completion: input_ids [seq_len]
            - Annotated spans from LLM (character-level offsets)
        
        Output:
            - labels [seq_len]: {-100 (ignore), 0.0 (supported), 1.0 (hallucinated)}
            - weights [seq_len]: {0.0 (ignore), neg_w, pos_w}
            - pos_spans: List[(start_token_idx, end_token_idx)]
            - neg_spans: List[(start_token_idx, end_token_idx)]
        """
        
        # Step 1: Find where assistant response starts
        completion_start_idx = find_assistant_tokens_slice(
            input_ids, 
            self.tokenizer
        )
        
        # Step 2: For each annotated span, find its token positions
        positive_indices = []
        negative_indices = []
        ignore_indices = []
        positive_spans = []
        negative_spans = []
        
        for span in sorted(item.spans, key=lambda x: x.index):
            # Find this span's text in the tokenized input
            span_text = span.span
            position_slice = find_string_in_tokens(
                span_text, input_ids, self.tokenizer
            )
            
            # Convert to token indices
            span_token_indices = list(range(
                position_slice.start, 
                position_slice.stop
            ))
            
            # Store span boundaries for span-level loss
            span_start = span_token_indices[0]
            span_end = span_token_indices[-1]
            
            if span.label == 1.0:  # Hallucination
                positive_indices.extend(span_token_indices)
                positive_spans.append([span_start, span_end])
                
                # Add ignore buffer around span
                nearby = get_nearby_indices(span_token_indices)
                ignore_indices.extend(nearby)
                
            elif span.label == 0.0:  # Supported
                negative_indices.extend(span_token_indices)
                negative_spans.append([span_start, span_end])
                
                # Add ignore buffer
                nearby = get_nearby_indices(span_token_indices)
                ignore_indices.extend(nearby)
            
            else:  # -100.0 Ignored
                ignore_indices.extend(span_token_indices)
        
        # Step 3: Create label and weight tensors
        labels = torch.full([seq_len], -100.0)  # Default: ignore all
        weights = torch.ones([seq_len])
        
        # Ignore prompt tokens
        labels[:completion_start_idx] = -100.0
        
        # Set explicit labels
        labels[positive_indices] = 1.0
        labels[negative_indices] = 0.0
        labels[ignore_indices] = -100.0
        
        # Set weights (class weighting)
        weights[positive_indices] = pos_weight  # e.g., 10.0
        weights[negative_indices] = neg_weight  # e.g., 10.0
        weights[ignore_indices] = 0.0
        
        return labels, weights, positive_spans, negative_spans
```

### 1.5 Span-Level Max Aggregation Loss

```python
def compute_probe_max_aggregation_loss(probe_logits, spans, labels):
    """
    Key idea: A span should be classified by its MAXIMUM logit.
    - If span is hallucination, max(logits_in_span) should predict 1.0
    - If span is supported, max(logits_in_span) should predict 0.0
    
    This encourages the probe to flag at least one token per hallucinated span.
    More natural than token-level BCE which treats tokens independently.
    """
    
    span_losses = []
    
    for batch_idx in range(batch_size):
        # Positive spans (hallucinations)
        for start_token, end_token in positive_spans[batch_idx]:
            # Get logits for this span
            span_logits = probe_logits[batch_idx, start_token:end_token+1]
            
            # Take maximum
            max_logit = span_logits.max()
            
            # Penalize: max should be close to 1.0 for hallucinations
            loss = F.binary_cross_entropy_with_logits(
                max_logit.unsqueeze(0),
                torch.tensor([1.0])
            )
            span_losses.append(loss)
        
        # Negative spans (supported)
        for start_token, end_token in negative_spans[batch_idx]:
            span_logits = probe_logits[batch_idx, start_token:end_token+1]
            max_logit = span_logits.max()
            
            # Penalize: max should be close to 0.0 for supported
            loss = F.binary_cross_entropy_with_logits(
                max_logit.unsqueeze(0),
                torch.tensor([0.0])
            )
            span_losses.append(loss)
    
    # Average across all spans
    return torch.mean(torch.stack(span_losses))

# Why max aggregation?
# - Token-level: loss = mean(BCE across all tokens)
#   Problem: Ignores span structure, allows misses
# - Span-level: loss = mean(BCE across spans)
#   Benefit: At least one token must flag each hallucinated span
```

---

## Part 2: Persona Vectors - Contrastive Difference

### 2.1 Core Algorithm

```python
def save_persona_vector(model_name, pos_path, neg_path, trait, save_dir):
    """
    Complete pipeline for extracting persona vectors.
    """
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # ========== STEP 1: Load and filter responses ==========
    
    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)
    
    # Apply quality filters
    mask_pos = (pos_df[trait] >= 50) & (pos_df['coherence'] >= 50)
    mask_neg = (neg_df[trait] < 50) & (neg_df['coherence'] >= 50)
    
    # After filtering, typically 100-150 samples per condition
    pos_effective = pos_df[mask_pos]
    neg_effective = neg_df[mask_neg]
    
    # ========== STEP 2: Extract activations ==========
    
    # For positive responses:
    pos_prompts = pos_effective['prompt'].tolist()
    pos_responses = pos_effective['answer'].tolist()
    
    # Run through model, capture hidden states
    pos_prompt_avg, pos_prompt_last, pos_response_avg = get_hidden_p_and_r(
        model, tokenizer,
        prompts=pos_prompts,
        responses=pos_responses,
        layer_list=range(num_layers)
    )
    
    # Result: Three dictionaries, each with layer → [N, hidden_dim] tensor
    # where N is number of examples, hidden_dim is model's hidden size
    
    # Do the same for negative
    neg_prompt_avg, neg_prompt_last, neg_response_avg = get_hidden_p_and_r(
        model, tokenizer,
        prompts=neg_prompts,
        responses=neg_responses,
        layer_list=range(num_layers)
    )
    
    # ========== STEP 3: Compute layer-wise differences ==========
    
    # Three types of vectors depending on aggregation strategy
    
    # Type 1: Response average difference
    response_diff = torch.stack([
        pos_response_avg[layer].mean(0) - neg_response_avg[layer].mean(0)
        for layer in range(num_layers)
    ])  # Shape: [num_layers, hidden_dim]
    
    # Type 2: Prompt average difference  
    prompt_diff = torch.stack([
        pos_prompt_avg[layer].mean(0) - neg_prompt_avg[layer].mean(0)
        for layer in range(num_layers)
    ])  # Shape: [num_layers, hidden_dim]
    
    # Type 3: Prompt last token difference
    prompt_last_diff = torch.stack([
        pos_prompt_last[layer].mean(0) - neg_prompt_last[layer].mean(0)
        for layer in range(num_layers)
    ])  # Shape: [num_layers, hidden_dim]
    
    # ========== STEP 4: Save vectors ==========
    
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(response_diff, f"{save_dir}/{trait}_response_avg_diff.pt")
    torch.save(prompt_diff, f"{save_dir}/{trait}_prompt_avg_diff.pt")
    torch.save(prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff.pt")
```

### 2.2 Activation Extraction (get_hidden_p_and_r)

```python
def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    """
    Extract hidden states at different granularities.
    
    Three aggregation strategies:
    1. prompt_avg: Mean of prompt tokens' hidden states
    2. response_avg: Mean of response tokens' hidden states  
    3. prompt_last: Last token of prompt
    """
    
    num_layers = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(num_layers))
    
    # Initialize storage: List[num_layers] of empty lists
    prompt_avg = [[] for _ in range(num_layers)]
    response_avg = [[] for _ in range(num_layers)]
    prompt_last = [[] for _ in range(num_layers)]
    
    # Process each prompt-response pair
    for prompt, response in tqdm(zip(prompts, responses)):
        # Concatenate prompt and response
        full_text = prompt + response
        
        # Tokenize
        inputs = tokenizer(full_text, return_tensors="pt")
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        
        # Forward pass with hidden state extraction
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple[num_layers+1]
        # hidden_states[i]: [batch=1, seq_len, hidden_dim]
        
        # For each layer, extract three aggregations
        for layer_idx in layer_list:
            layer_hidden = hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
            layer_hidden = layer_hidden[0]  # Remove batch dim: [seq_len, hidden_dim]
            
            # 1. Average over prompt tokens
            prompt_part = layer_hidden[:prompt_len]  # [prompt_len, hidden_dim]
            prompt_avg[layer_idx].append(prompt_part.mean(0).detach().cpu())
            
            # 2. Average over response tokens
            response_part = layer_hidden[prompt_len:]  # [response_len, hidden_dim]
            response_avg[layer_idx].append(response_part.mean(0).detach().cpu())
            
            # 3. Last token of prompt
            prompt_last_token = layer_hidden[prompt_len - 1]  # [hidden_dim]
            prompt_last[layer_idx].append(prompt_last_token.detach().cpu())
    
    # Concatenate all examples for each layer
    # Result: List[num_layers] of [num_examples, hidden_dim] tensors
    prompt_avg = [torch.stack(layers) for layers in prompt_avg]
    response_avg = [torch.stack(layers) for layers in response_avg]
    prompt_last = [torch.stack(layers) for layers in prompt_last]
    
    return prompt_avg, prompt_last, response_avg
```

### 2.3 Why Difference Works

```python
"""
Key insight: Why does subtraction of means capture behavior?

Consider activation space for a model processing paternalism:
- When the model sees "be paternalistic" instruction → activations shift toward 
  certain directions (let's call them "paternalism directions")
- When the model sees "avoid paternalism" → activations shift toward opposite directions

After enough examples:
  pos_mean = baseline_activation + paternalism_shift + noise
  neg_mean = baseline_activation - paternalism_shift + noise
  
  difference = pos_mean - neg_mean 
             = (paternalism_shift + noise) - (-paternalism_shift + noise)
             ≈ 2 * paternalism_shift

The noise cancels! We get a clean signal.

Why this works so well:
1. Filtering for high quality (coherence >= 50) removes confused examples
2. Using multiple questions (20) provides redundancy
3. Large hidden dimensions (3584 for Gemma) have room for diverse signals
4. Perfectly balanced data (150 pos, 150 neg) ensures symmetry
"""
```

### 2.4 Three Aggregation Strategies

```python
# Strategy 1: Response average
# Use: Model's response after seeing the trait instruction
# Captures: How trait manifests in output generation
vector_response = [
    mean(hidden[prompt_len:, :] for responses)  # Average over response tokens
]
# Best for: Detecting trait in generated text, steering generation

# Strategy 2: Prompt average  
# Use: Model's activation after seeing the instruction prompt
# Captures: How model interprets the trait instruction
vector_prompt = [
    mean(hidden[:prompt_len, :] for responses)  # Average over prompt tokens
]
# Best for: Understanding how instructions are encoded

# Strategy 3: Prompt last token
# Use: Final token representation after processing the instruction
# Captures: Compressed instruction representation
vector_prompt_last = [
    hidden[prompt_len-1, :]  # Just the last token
]
# Best for: Minimal, most compressed representation

# All three are saved - use whichever fits your analysis
```

---

## Part 3: Comparison of the Two Approaches

### 3.1 Information-Theoretic Perspective

```
HALLUCINATION PROBES:
  Learned linear classifier: P(hallucination | hidden_state)
  
  Information stored:
    - Which hidden state features matter (probe weight magnitudes)
    - How features combine (linear weights)
    - Baseline probability (probe bias)
    - Effective training signal (which examples mattered)
  
  Amount: ~4K parameters (for 4096-dim hidden state)
  Format: Binary decision boundary in activation space

PERSONA VECTORS:
  Learned difference: direction = P(positive) - P(negative)
  
  Information stored:
    - Axis of maximum variance between conditions
    - Magnitude of effect (vector norm)
    - Per-layer decomposition (which layers matter)
    - No "baseline" - purely directional
  
  Amount: ~150K parameters (for 4096-dim, 42 layers)
  Format: Direction in activation space (high-dimensional)
```

### 3.2 Statistical Properties

```
HALLUCINATION PROBES - Supervised Learning:
  
  Objective: min_w ||Probe(w) - Labels||²
  
  Properties:
  - Learns decision boundary for binary classification
  - Optimization: Gradient descent on cross-entropy loss
  - Generalization: Depends on labeled data quality and quantity
  - Theoretical guarantees: Statistical learning theory applies
  - Failure mode: Overfitting if too few examples or bad labels
  - Advantage: Can learn non-linear patterns (via LoRA non-linearity)

PERSONA VECTORS - Unsupervised Difference:
  
  Objective: Vector = mean(pos_hiddens) - mean(neg_hiddens)
  
  Properties:
  - No optimization - just statistics
  - Learns direction of maximum separation
  - Generalization: Depends on data quality, not quantity
  - Theoretical guarantees: Law of large numbers
  - Failure mode: Poor filtering allows noise
  - Advantage: Extremely simple, interpretable, no training
```

### 3.3 Computational Complexity

```
HALLUCINATION PROBES - Training:
  
  Time: O(batch_size * seq_len * num_layers * hidden_dim * num_epochs)
  
  Per epoch on 1000 examples:
    - Forward pass: 1000 * 1500 tokens * 32 layers * 4096 dim
                  ≈ 1.96e11 FLOPs ≈ 20 seconds on GPU
    - Backward pass: Same + allreduce ≈ 50 seconds
    - Total: ~70 seconds per epoch
    - 1-2 epochs: 70-140 seconds training
  
  Inference:
    - Per token: 1 linear layer (4096 → 1) ≈ 4K MAC ≈ 0.001 ms

PERSONA VECTORS - Extraction:
  
  Time: O(num_examples * seq_len * hidden_dim) [forward only, no backward]
  
  Per extraction:
    - Forward: 200 pos + 200 neg = 400 examples
    - Average example: 100 tokens
    - Total: 400 * 100 * 4096 ≈ 1.64e8 FLOPs ≈ 0.1 seconds
    - 20x faster than probe training!
  
  Inference:
    - Per token: 1 dot product (4096-dim) ≈ 0.0001 ms
    - 10x faster than probe inference
```

---

## Part 4: Mathematical Formalism

### 4.1 Hallucination Probe (Formal Definition)

```
Given:
  - Model M with hidden states H_t^l ∈ ℝ^d at layer l, token t
  - Binary labels y_t ∈ {0, 1} (supported/hallucinated)
  - Attention mask a_t ∈ {0, 1}

Probe:
  P(hallucination | H_t^l) = σ(w^T H_t^l + b)
  
  where w ∈ ℝ^d, b ∈ ℝ (learned parameters)
  σ is sigmoid function

Training Objective:
  L_total = λ_lm L_lm + λ_kl L_kl + (1-λ_lm-λ_kl) L_probe
  
  where:
    L_lm = cross_entropy(logits, target_ids)  [LM loss]
    L_kl = KL(Q_lora || Q_base)                [KL regularization]
    L_probe = -Σ [y_t log P_t + (1-y_t) log(1-P_t)]  [BCE]
    
  Span-level modification:
    L_probe = -Σ_{spans} [1(span_pos) log max(P_t) + 
                          1(span_neg) log(1-max(P_t))]

Inference:
  For each token t:
    h_t = capture hidden state from layer l
    prob_t = sigmoid(w^T h_t + b)
    decision_t = 1 if prob_t > threshold else 0
```

### 4.2 Persona Vector (Formal Definition)

```
Given:
  - Model M with hidden states H_{prompt,i}^l, H_{response,i}^l
  - Trait dimension (paternalism, deception, etc.)
  - Filter threshold τ
  - GPT-4 scores s_i ∈ [0, 100]

Filtering:
  I_pos = {i : s_i ≥ τ and coherence_i ≥ 50}
  I_neg = {i : s_i < τ and coherence_i ≥ 50}
  
  Typically τ = 50 → perfectly balanced

Vector Extraction (Response Aggregation):
  For each layer l:
    - pos_response_l = mean_{i ∈ I_pos} mean_t H_{response,i,t}^l
    - neg_response_l = mean_{i ∈ I_neg} mean_t H_{response,i,t}^l
    
    v_l = pos_response_l - neg_response_l  ∈ ℝ^d
  
  Final vector: V = [v_1, v_2, ..., v_L] ∈ ℝ^{L×d}

Inference (Projection):
  For each token t with hidden state h_t^l:
    score_t = h_t^l · v_l  [dot product]
    prob_t = sigmoid(score_t)
```

### 4.3 Loss Function Comparison

```
HALLUCINATION PROBES:

L_probe_token = -Σ_t w_t [y_t log σ(w^T h_t) + (1-y_t) log(1-σ(w^T h_t))]

L_probe_span = mean_{spans} [
  max_t σ(w^T h_t) for t in halluc_span   → target 1.0
  max_t σ(w^T h_t) for t in supported_span → target 0.0
]

Combined with annealing:
  ω(step) = min(step / warmup_steps, 1.0)
  L_probe = (1 - ω) L_token + ω L_span

PERSONA VECTORS:

No loss! Just:
  v_l = (1/|I_pos|) Σ_{i ∈ I_pos} h_i^l - (1/|I_neg|) Σ_{i ∈ I_neg} h_i^l

Filtering acts as implicit regularization:
  - Removing low-quality examples (coherence < 50)
  - Removing examples with ambiguous trait scores
  - Ensures strong contrast: |s_pos - s_neg| ≈ 50
```

---

## Conclusion

Both approaches are elegant but target different phenomena:

1. **Hallucination Probes** learn a *decision boundary* between hallucinated and supported claims
2. **Persona Vectors** learn a *direction of variation* for behavioral traits

The key algorithmic difference:
- Probes: Gradient descent on labeled data → optimized classifier
- Vectors: Simple arithmetic on filtered data → interpretable direction

For many applications, using both together provides the most complete picture.

