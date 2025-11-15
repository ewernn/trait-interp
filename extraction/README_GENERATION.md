# Generation Script Guide

Use `1_generate_batched_simple.py` for all generation tasks.

## Quick Start

```bash
# Default: batched generation with activation extraction
python extraction/1_generate_batched_simple.py \
  --experiment my_experiment \
  --trait my_trait \
  --batch_size 8
```

This generates 200 responses (100 pos + 100 neg), judges them with GPT-5-mini, and extracts activations.

## Batch Size Options

Choose based on your GPU memory:

```bash
--batch_size 1   # Serial (slowest, for debugging or CPU)
--batch_size 4   # Colab Free (T4, 15GB VRAM)
--batch_size 8   # Default, Colab Pro (V100, 16GB VRAM)
--batch_size 12  # Local 3090/4090 (24GB VRAM)
--batch_size 16  # A100 40GB
```

**Speedup expectations:**
- `batch_size=1`: Baseline (~20 min for 200 examples)
- `batch_size=4`: ~3x faster (~7 min)
- `batch_size=8`: ~5x faster (~4 min)
- `batch_size=16`: ~7x faster (~3 min)

## Activation Extraction

Control whether to extract activations during generation:

```bash
# Extract activations (default)
--extract_activations True

# Just generate responses, extract later
--extract_activations False
```

**When to disable:**
- Debugging prompt/instruction issues
- Testing trait definitions quickly
- Re-generating responses without re-extracting

## Advanced Options

```bash
# Specify models explicitly (auto-inferred from experiment name)
--gen_model google/gemma-2-9b-it \
--judge_model gpt-5-mini

# More examples per polarity (default: 100)
--n_examples 200

# Different threshold for filtering (default: 50)
--threshold 60

# Different max tokens (default: 150)
--max_new_tokens 200

# Multiple traits at once
--traits refusal,uncertainty,verbosity
```

## Common Workflows

### Full Pipeline (Recommended)

```bash
# Generate responses + extract activations
python extraction/1_generate_batched_simple.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait retrieval_construction \
  --batch_size 8

# Extract vectors
python extraction/3_extract_vectors.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait retrieval_construction
```

### Two-Stage Extraction

If you want to separate generation from activation extraction:

```bash
# Stage 1: Just generate responses
python extraction/1_generate_batched_simple.py \
  --experiment my_exp \
  --trait my_trait \
  --extract_activations False

# Stage 2: Extract activations later
python extraction/2_extract_activations.py \
  --experiment my_exp \
  --trait my_trait
```

**Why use two stages?**
- Test trait definitions quickly (no activation overhead)
- Re-extract with different settings without regenerating
- Separate concerns (response quality vs activation capture)

### Debugging

```bash
# Small batch, few examples, no activation extraction
python extraction/1_generate_batched_simple.py \
  --experiment test_exp \
  --trait test_trait \
  --batch_size 1 \
  --n_examples 5 \
  --extract_activations False
```

## Output Structure

All scripts produce identical output:

```
experiments/my_exp/my_trait/
├── responses/
│   ├── pos.csv          # Positive examples with scores
│   └── neg.csv          # Negative examples with scores
├── activations/         # Only if extract_activations=True
│   ├── all_layers.pt    # [n_examples, n_layers, hidden_dim]
│   └── metadata.json    # Extraction metadata
└── trait_definition.json
```

**CSV columns:**
- `question`: Original question
- `instruction`: Instruction used (pos or neg variant)
- `prompt`: Full prompt (instruction + question)
- `response`: Model's generated response
- `trait_score`: Judge's score (0-100)

## Performance Comparison

| Batch Size | Time (200 ex) | Memory | Speedup |
|------------|---------------|--------|---------|
| 1 | 20 min | 8GB | 1x |
| 4 | 7 min | 12GB | 3x |
| 8 | 4 min | 16GB | 5x |
| 16 | 3 min | 24GB | 7x |

**Memory calculation:**
- Base model: ~6GB (Gemma 2B float16)
- Per batch item: ~0.5-1GB
- Safe formula: `6 + (batch_size × 1)`

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch_size 4  # or even 2 or 1
```

### API Rate Limits

The script has automatic retry logic, but if you hit hard limits:

```bash
# Reduce examples temporarily
--n_examples 50
```

### Model Access Denied

Make sure your HuggingFace token has access to the model:
- Visit model page (e.g., https://huggingface.co/google/gemma-2-2b-it)
- Accept license agreement
- Create token with read permissions
- Add to `.env`: `HF_TOKEN=hf_...`

## Script Internals

**What 1_generate_batched_simple.py does:**

1. Load trait definition from `trait_definition.json`
2. Load generation model (Gemma, Llama, etc.)
3. For each polarity (pos/neg):
   - Generate responses in batches
   - Judge with API model asynchronously
   - Filter by threshold
   - Extract activations (if enabled)
4. Save responses as CSV
5. Save activations as PyTorch tensor
6. Save metadata as JSON

**Batching strategy:**
- Groups prompts into batches
- Generates all responses in batch (parallel forward passes)
- Judges all responses asynchronously (parallel API calls)
- Maximizes throughput while respecting memory constraints

## Further Reading

- **[docs/pipeline_guide.md](../docs/pipeline_guide.md)** - Complete 3-stage pipeline walkthrough
- **[docs/creating_traits.md](../docs/creating_traits.md)** - How to design trait definitions
- **[docs/remote_setup_guide.md](../docs/remote_setup_guide.md)** - Running on remote GPU instances
