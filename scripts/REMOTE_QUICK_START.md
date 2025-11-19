# Remote A100 Quick Start

## 1. First Time Setup (5 minutes)

```bash
# SSH to your A100 instance
ssh user@your-gpu-instance

# Clone repo
git clone https://github.com/yourusername/trait-interp.git
cd trait-interp

# Create environment
conda create -n trait python=3.11 -y
conda activate trait

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate huggingface_hub pandas tqdm fire scikit-learn

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Setup API keys
cat > .env << 'EOF'
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
EOF
```

## 2. Run the Pipeline (ONE COMMAND)

```bash
# Activate environment
conda activate trait

# Run all 12 traits (~3 hours, $3-4 cost)
./scripts/run_all_natural_extraction_a100.sh
```

That's it! The script will:
1. Generate responses for all 12 traits
2. Extract activations
3. Extract vectors (all 4 methods, 26 layers)
4. Run cross-distribution testing (4×4 matrices)

## 3. Monitor Progress

```bash
# Watch GPU usage
nvidia-smi

# Check completed traits
ls experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors/ 2>/dev/null | grep -c "mean_diff_layer0.pt"

# Check cross-distribution results
ls results/cross_distribution_analysis/*_full_4x4_results.json | wc -l
```

## 4. Download Results (after completion)

```bash
# From your LOCAL machine
rsync -avz user@remote-gpu:~/trait-interp/results/ results/
rsync -avz user@remote-gpu:~/trait-interp/experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/ experiments/gemma_2b_cognitive_nov20/
```

## If Something Goes Wrong

**Check logs:**
```bash
# See what errored
tail -100 nohup.out  # if you ran in background

# Test one trait manually
python extraction/1_generate_natural.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait abstract_concrete \
  --batch-size 64 \
  --device cuda
```

**GPU out of memory:**
```bash
# Reduce batch size by editing the script
# Change BATCH_SIZE=64 to BATCH_SIZE=32
nano scripts/run_all_natural_extraction_a100.sh
```

## Run in Background (Optional)

```bash
# If you want to disconnect and let it run
nohup ./scripts/run_all_natural_extraction_a100.sh > extraction.log 2>&1 &

# Monitor
tail -f extraction.log

# Check if still running
ps aux | grep python
```

## Fast Path (Parallel, ~45 min instead of 3 hours)

```bash
# Only if you're comfortable with parallel execution
./scripts/run_parallel_natural_extraction.sh

# Logs will be in: logs/natural_extraction_*.log
```

---

## What You'll Get

After ~3 hours:

**New data:**
- 12 traits × 104 vectors = 1,248 new trait vectors
- 12 traits × 4×4 cross-distribution matrices

**Results files:**
- `results/cross_distribution_analysis/{trait}_full_4x4_results.json` (12 files)
- Each has accuracy for all 4 methods × 26 layers × 4 quadrants

**Total coverage:**
- Before: 3 traits with cross-distribution data
- After: **15 traits** with full validation ✅

---

## Costs

- A100 80GB: ~$1-1.50/hour
- Serial run (3 hours): **~$3-5**
- Parallel run (45 min): **~$1-2**

Much cheaper than running locally for 16 hours!
