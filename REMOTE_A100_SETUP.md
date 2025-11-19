# Running Natural Extraction on Remote A100

**TL;DR:** A100 will be ~5-10x faster. Use batch_size=64 for serial (~2-3 hours) or batch_size=32 with 4-way parallelization (~30-45 min).

---

## Speed Comparison

| Hardware | Batch Size | Time per Trait | Total Time (12 traits) |
|----------|-----------|----------------|------------------------|
| **Mac MPS** | 8 | ~80 min | ~16 hours |
| **A100 Serial** | 64 | ~15 min | **~3 hours** |
| **A100 Parallel (4x)** | 32 | ~15 min | **~45 min** |

---

## Setup on Remote Instance

### 1. Sync Code to Remote

```bash
# From your local machine
rsync -avz --exclude .git --exclude experiments/*/inference \
  trait-interp/ user@remote-gpu:~/trait-interp/

# Or use git
ssh user@remote-gpu
cd ~/trait-interp
git pull origin main
```

### 2. Install Dependencies

```bash
# On remote machine
cd ~/trait-interp

# Create environment
conda create -n trait python=3.11 -y
conda activate trait

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers accelerate huggingface_hub pandas tqdm fire scikit-learn

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 3. Set Up API Keys

```bash
# Copy .env or create new
cat > .env << 'EOF'
HF_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
EOF
```

### 4. Verify Natural Scenarios

```bash
# Check scenarios are present
ls extraction/natural_scenarios/*_positive.txt | wc -l
# Should see 12+ files

# Check one file has 110 prompts
wc -l extraction/natural_scenarios/abstract_concrete_positive.txt
# Should show ~110 lines
```

---

## Execution Options

### Option 1: Serial Execution (Recommended - Safest)

**Time:** ~2-3 hours
**Complexity:** Simple, easy to monitor

```bash
./scripts/run_all_natural_extraction_a100.sh
```

**Features:**
- Batch size: 64 (optimized for 80GB A100)
- Runs one trait at a time
- Easy to debug if something fails
- Clear progress output

### Option 2: Parallel Execution (Fastest)

**Time:** ~30-45 minutes
**Complexity:** More complex, harder to debug

```bash
./scripts/run_parallel_natural_extraction.sh
```

**Features:**
- Batch size: 32 (lower since running 4 in parallel)
- Runs 4 traits simultaneously
- Logs saved to `logs/natural_extraction_{trait}.log`
- Uses GNU parallel or xargs

**Monitor progress:**
```bash
# Watch logs in real-time
tail -f logs/natural_extraction_*.log

# Check which traits are complete
ls experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/ | grep _natural

# Count completed cross-distribution results
ls results/cross_distribution_analysis/*_full_4x4_results.json | wc -l
```

### Option 3: Manual (For Testing)

**Run one trait to test setup:**

```bash
# Test with one trait first
python extraction/1_generate_natural.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait abstract_concrete \
  --batch-size 64 \
  --device cuda

# Should take ~10-15 min if working correctly
```

---

## Resource Usage

### Expected GPU Memory

| Stage | VRAM Usage | Notes |
|-------|-----------|-------|
| Generation (batch=64) | ~40-50 GB | Can reduce batch_size if OOM |
| Activation Extraction | ~15-20 GB | Loads model + stores activations |
| Vector Extraction | Minimal | CPU-only, no GPU needed |
| Cross-distribution | Minimal | Just vector math |

### Parallel Execution (4 traits at once)

- 4 × batch_size=32 = ~120-140 GB total needed
- **Will NOT fit on 80GB A100** ⚠️
- Reduce to 3 parallel jobs or lower batch size to 24

**Adjust parallelization:**
```bash
# Edit the script
PARALLEL_JOBS=3  # Change from 4 to 3
BATCH_SIZE=24    # Lower batch size
```

Or run **2 parallel** with batch_size=32 (safer):
```bash
PARALLEL_JOBS=2
BATCH_SIZE=32
```

---

## Monitoring Progress

### Check Status During Run

```bash
# See what's running
nvidia-smi

# Watch GPU memory
watch -n 1 nvidia-smi

# Check how many traits complete
ls experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/ | grep "_natural" | wc -l

# Check cross-distribution results
ls results/cross_distribution_analysis/ | grep "full_4x4" | wc -l
```

### Expected Outputs

After each trait completes:

```
experiments/gemma_2b_cognitive_nov20/{trait}_natural/extraction/
├── responses/
│   ├── pos.json  (~1 MB)
│   └── neg.json  (~1 MB)
├── activations/
│   ├── pos_layer0.pt through pos_layer25.pt  (~500 MB total)
│   └── neg_layer0.pt through neg_layer25.pt  (~500 MB total)
└── vectors/
    ├── mean_diff_layer*.pt  (26 files)
    ├── probe_layer*.pt      (26 files)
    ├── ica_layer*.pt        (26 files)
    └── gradient_layer*.pt   (26 files)

results/cross_distribution_analysis/
└── {trait}_full_4x4_results.json  (~150 KB)
```

**Per trait:** ~1 GB activations + 104 vector files
**Total:** ~12 GB activations + 1,248 vectors

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python extraction/1_generate_natural.py \
  --batch-size 32  # or 16 if still failing
```

### Generation Hanging

- Check if model loaded correctly: `nvidia-smi` should show ~10GB VRAM used
- Verify HuggingFace token: `echo $HF_TOKEN` or check `.env`
- Test single generation: `--batch-size 1` to isolate issue

### Slow Speed (Not Using GPU)

```bash
# Verify CUDA available
python -c "import torch; print(torch.cuda.is_available())"

# Check device being used
# Look for this in script output:
# "Device: cuda:0" (good)
# "Device: cpu" (bad - not using GPU)
```

### Cross-Distribution Fails

- Check both instruction and natural vectors exist:
  ```bash
  ls experiments/gemma_2b_cognitive_nov20/abstract_concrete/extraction/vectors/*.pt | wc -l  # Should be 104
  ls experiments/gemma_2b_cognitive_nov20/abstract_concrete_natural/extraction/vectors/*.pt | wc -l  # Should be 104
  ```

---

## Cost Estimation

**Cloud GPU pricing (approximate):**
- Lambda Labs A100 80GB: ~$1.10/hour
- Vast.ai A100 80GB: ~$0.80-1.50/hour
- RunPod A100 80GB: ~$1.39/hour

**Total cost:**
- Serial execution (3 hours): **~$3-4**
- Parallel execution (45 min): **~$1-2**

**Much cheaper than 16 hours on weaker GPU!**

---

## After Completion

### 1. Download Results

```bash
# From local machine
rsync -avz user@remote-gpu:~/trait-interp/results/cross_distribution_analysis/ \
  results/cross_distribution_analysis/

rsync -avz user@remote-gpu:~/trait-interp/experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/ \
  experiments/gemma_2b_cognitive_nov20/ --include="*_natural/" --include="*.pt" --exclude="*"
```

### 2. Analyze Results

```bash
# Generate summary of cross-distribution results
python3 << 'EOF'
import json
from pathlib import Path

results_dir = Path('results/cross_distribution_analysis')
traits_with_full_4x4 = []

for f in sorted(results_dir.glob('*_full_4x4_results.json')):
    trait = f.stem.replace('_full_4x4_results', '')
    data = json.load(open(f))

    has_inst_nat = 'inst_nat' in data.get('quadrants', {}) and \
                   any(len(m.get('all_layers', [])) > 0
                       for m in data['quadrants']['inst_nat'].get('methods', {}).values())

    if has_inst_nat:
        traits_with_full_4x4.append(trait)

print(f"Traits with cross-distribution data: {len(traits_with_full_4x4)}")
for t in sorted(traits_with_full_4x4):
    print(f"  ✅ {t}")
EOF
```

### 3. Update Documentation

See results in:
- `results/cross_distribution_analysis/TOP5_LAYERS_CROSS_DISTRIBUTION.txt`
- `results/cross_distribution_analysis/EXTRACTION_SCORES_ALL_TRAITS.txt`

---

## Recommended Workflow

1. **Start with serial execution on A100**
   - Safest, easiest to debug
   - Still 5-6x faster than local (~3 hours vs 16 hours)

2. **Monitor first trait completion**
   - Verify outputs look correct
   - Check one cross-distribution result

3. **Let it run overnight if needed**
   - 3 hours is fast enough
   - No need to optimize further unless you're in a rush

4. **Only use parallel if time-critical**
   - Adds complexity
   - Harder to debug failures
   - Saves ~2 hours (45 min vs 3 hours)

---

## Summary

✅ **Use A100:** 5-10x faster than Mac MPS
✅ **Batch size 64:** Optimal for 80GB A100 serial execution
✅ **Serial recommended:** Simpler, still only ~3 hours
⚠️ **Parallel optional:** ~45 min but more complex
💰 **Cost:** ~$3-4 for full run

**Just run:** `./scripts/run_all_natural_extraction_a100.sh` and wait ~3 hours!
