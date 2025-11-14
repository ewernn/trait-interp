# Remote GPU Setup Guide

Quick guide for running extraction on a rented GPU instance.

## Pre-Flight Checklist

Before cloning on remote:
- [x] Trait definitions generated (17 traits in experiments/gemma_2b_cognitive_nov20/)
- [x] Batching implemented (1_generate_batched_simple.py)
- [x] Dynamics analysis ready (experiments/examples/run_dynamics.py)
- [x] All changes committed and pushed

## Step 1: Rent GPU

Recommended specs:
- **GPU**: A100 40GB (or V100 32GB minimum)
- **RAM**: 32GB+
- **Storage**: 50GB+
- **Providers**: RunPod, Vast.ai, Lambda Labs

## Step 2: Clone & Setup

```bash
# Clone repo
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp

# Install dependencies
pip install torch transformers accelerate openai huggingface_hub pandas tqdm fire scikit-learn

# Setup credentials
cp .env.example .env
nano .env  # Add HF_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 3: Run Extraction

### Quick Test (1 trait, small batch)
```bash
# Test with single trait, small dataset
python extraction/1_generate_batched_simple.py \
    --experiment gemma_2b_cognitive_nov20 \
    --trait refusal \
    --n_examples 10 \
    --batch_size 4
```

Expected: ~2-3 minutes, creates responses + activations

### Full Cognitive Primitives (17 traits)
```bash
# All 17 traits, full dataset
python extraction/1_generate_batched_simple.py \
    --experiment gemma_2b_cognitive_nov20 \
    --traits refusal,retrieval_construction,commitment_strength,context_adherence,uncertainty_calibration,abstract_concrete,power_dynamics,sycophancy,emotional_valence,paranoia_trust,instruction_boundary,cognitive_load,temporal_focus,convergent_divergent,local_global,serial_parallel,abstract_concrete \
    --n_examples 100 \
    --batch_size 8
```

**Time**: ~5-6 hours for all 17 traits
**Cost**: ~$10-15 on A100 @ $2/hr + ~$2.50 in API calls

### Monitor Progress
```bash
# In another terminal, watch progress
watch -n 10 'ls -lh experiments/gemma_2b_cognitive_nov20/*/responses/*.csv'
```

## Step 4: Extract Vectors

After generation completes:

```bash
# Extract vectors from activations
python extraction/3_extract_vectors.py \
    --experiment gemma_2b_cognitive_nov20 \
    --traits refusal,retrieval_construction,commitment_strength,context_adherence,uncertainty_calibration,abstract_concrete,power_dynamics,sycophancy,emotional_valence,paranoia_trust,instruction_boundary,cognitive_load,temporal_focus,convergent_divergent,local_global,serial_parallel,abstract_concrete \
    --methods mean_diff,probe,ica,gradient \
    --layers 16
```

**Time**: ~30-45 minutes (CPU only, fast)

## Step 5: Test Dynamics (Optional on GPU)

```bash
# Quick test of dynamics analysis
python experiments/examples/run_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "What is the capital of France?" \
    --output test_dynamics.json
```

## Step 6: Download Results

```bash
# From your local machine
rsync -avz --progress \
    user@remote:/path/to/trait-interp/experiments/gemma_2b_cognitive_nov20/ \
    ./experiments/gemma_2b_cognitive_nov20/

# Or use your provider's download feature
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 4  # or even 2
```

### API Rate Limits
The script automatically handles retries, but if you hit hard limits:
```bash
# Reduce examples temporarily
--n_examples 50
```

### Missing Model Access
Make sure HF_TOKEN has access to gated models like Gemma:
- Go to https://huggingface.co/google/gemma-2-2b-it
- Accept license
- Create token with read permissions

### Import Errors
```bash
# Make sure you're in the repo root
cd trait-interp
python extraction/1_generate_batched_simple.py ...
```

## Cost Estimation

For 17 traits × 200 examples each:

| Item | Cost |
|------|------|
| GPU time (A100, ~6 hours) | $12 |
| GPT-5-mini judging (3,400 calls) | $2.50 |
| **Total** | **~$15** |

Can reduce by:
- Using fewer traits
- Smaller batch sizes (slower but cheaper GPU)
- Spot/preemptible instances (70% discount)

## What You Get

After completion:
```
experiments/gemma_2b_cognitive_nov20/
├── {trait}/
│   ├── responses/
│   │   ├── pos.csv           # 100 positive examples
│   │   └── neg.csv           # 100 negative examples
│   └── vectors/
│       ├── mean_diff_layer16.pt
│       ├── probe_layer16.pt
│       ├── ica_layer16.pt
│       └── gradient_layer16.pt
```

## Next Steps

After downloading results:
1. Run dynamics analysis locally (faster on your machine)
2. Visualize with `visualization.html`
3. Compare extraction methods
4. Analyze commitment points, velocity, persistence

## Quick Commands Reference

```bash
# Test run (5 min)
python extraction/1_generate_batched_simple.py --experiment gemma_2b_cognitive_nov20 --trait refusal --n_examples 10 --batch_size 4

# Full run (6 hours)
python extraction/1_generate_batched_simple.py --experiment gemma_2b_cognitive_nov20 --traits refusal,retrieval_construction,commitment_strength,context_adherence,uncertainty_calibration,abstract_concrete,power_dynamics,sycophancy,emotional_valence,paranoia_trust,instruction_boundary,cognitive_load,temporal_focus,convergent_divergent,local_global,serial_parallel,abstract_concrete --n_examples 100 --batch_size 8

# Extract vectors (30 min)
python extraction/3_extract_vectors.py --experiment gemma_2b_cognitive_nov20 --traits refusal,retrieval_construction,commitment_strength,context_adherence,uncertainty_calibration,abstract_concrete,power_dynamics,sycophancy,emotional_valence,paranoia_trust,instruction_boundary,cognitive_load,temporal_focus,convergent_divergent,local_global,serial_parallel,abstract_concrete --methods mean_diff,probe,ica,gradient --layers 16

# Test dynamics (5 min)
python experiments/examples/run_dynamics.py --experiment gemma_2b_cognitive_nov20 --prompts "Test prompt" --output test.json
```