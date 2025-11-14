# Generation Scripts Comparison

We have 4 different generation scripts. Here's when to use each:

## Quick Decision Guide

**Just want it to work fast?** → Use `1_generate_batched_simple.py` with `batch_size=8`

**Debugging or limited memory?** → Use `1_generate_responses.py`

**Want maximum efficiency?** → Use `1_generate_batched_simple.py` with highest batch_size your GPU allows

## The 4 Scripts

### 1. `1_generate_responses.py` (Original)
- **Speed**: Slowest (20 min for 200 examples)
- **Memory**: Lowest (8GB)
- **When to use**: Debugging, limited GPU memory
- **Separate from activation extraction**

```bash
python extraction/1_generate_responses.py --experiment my_exp --trait my_trait
```

### 2. `2_extract_activations.py` (Companion to #1)
- **Only** extracts activations from already generated responses
- **Must** run after `1_generate_responses.py`

```bash
python extraction/2_extract_activations.py --experiment my_exp --trait my_trait
```

### 3. `1_generate_and_extract.py` (Combined, no batching)
- **Speed**: Medium (20 min, saves re-loading model)
- **Memory**: Medium (10GB)
- **When to use**: Want single-stage but can't batch

```bash
python extraction/1_generate_and_extract.py --experiment my_exp --trait my_trait
```

### 4. `1_generate_batched_simple.py` ⭐ (RECOMMENDED)
- **Speed**: FASTEST (4 min with batch_size=8)
- **Memory**: Scales with batch size
- **When to use**: Most situations

```bash
python extraction/1_generate_batched_simple.py \
  --experiment my_exp \
  --trait my_trait \
  --batch_size 8
```

## Performance Table

| Script | Time (200 ex) | Memory | Speedup |
|--------|---------------|--------|---------|
| 1_generate_responses | 20 min | 8GB | 1x |
| 1_generate_and_extract | 20 min | 10GB | 1x |
| **1_generate_batched (bs=4)** | **7 min** | **12GB** | **3x** |
| **1_generate_batched (bs=8)** | **4 min** | **16GB** | **5x** |
| **1_generate_batched (bs=16)** | **3 min** | **24GB** | **7x** |

## Batch Size Selection

```python
# Colab Free (T4, 15GB)
--batch_size 4

# Colab Pro (V100, 16GB)
--batch_size 8

# A100 40GB
--batch_size 16

# Local 3090/4090 (24GB)
--batch_size 12
```

## All Scripts Output Same Format

All scripts produce identical output structure:
```
experiments/my_exp/my_trait/
├── responses/
│   ├── pos.csv
│   └── neg.csv
├── activations/
│   ├── all_layers.pt
│   └── metadata.json
└── trait_definition.json
```

So you can switch between them without changing downstream processing!