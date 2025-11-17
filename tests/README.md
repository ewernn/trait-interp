# Tests

Validation tests to catch common issues before running expensive extractions.

## Quick Validation (No GPU)

```bash
# Validates model configs and data integrity
python tests/test_model_configs.py
```

This catches:
- Model layer count mismatches (e.g., docs say 27, model has 26)
- Incorrect data shapes in saved files
- Metadata inconsistencies across extraction/inference
- Documentation drift from reality

**Run this first** - it's fast (<1 second) and catches off-by-one errors.

## Extraction Pipeline Tests (Requires GPU)

```bash
# Validates activation extraction indexing and norms
python tests/test_activation_extraction.py
```

This catches:
- Incorrect layer indexing (e.g., embedding vs layer 0)
- Unstable activation magnitudes across layers
- Architecture-specific quirks

## What Gets Tested

### `test_model_configs.py` (Fast, No GPU)
- **Model configs**: Gemma 2B has 26 layers, 2304 hidden dim
- **Inference data**: Projections have correct [n_tokens, n_layers, 3] shape
- **Activation metadata**: Layer count matches model config
- **Vector metadata**: Consistent dimensions across all files

### `test_activation_extraction.py` (Slow, Requires GPU)
- Layer indexing correctness
- Activation magnitude stability
- Model-specific architecture handling

## Adding New Tests

When adding new extraction methods or models:

1. **Test configs first** - Run `test_model_configs.py` before GPU work
2. **Test on 1-2 examples** - Don't waste GPU time on bugs
3. **Check output shapes** - Verify dimensions match comments
4. **Sanity check magnitudes** - Activations shouldn't explode or vanish
5. **Spot check results** - Load a few vectors, verify they make sense

## Test Structure

```
tests/
├── test_model_configs.py          # Model configs & data integrity (fast)
├── test_activation_extraction.py  # Extraction pipeline validation (slow)
└── README.md                       # This file
```

## Why Test Configs?

The off-by-one bug (docs said 27 layers, Gemma 2B has 26) would have been caught immediately by `test_model_configs.py`. Running this test after any documentation update prevents drift.
