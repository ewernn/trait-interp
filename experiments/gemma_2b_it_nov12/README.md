# Gemma 2B Behavioral Traits (November 12, 2024)

## Overview

Extraction of 8 core behavioral trait vectors from Gemma 2 2B using the original per-token-interp pipeline.

## Metadata

- **Model**: google/gemma-2-2b-it (Gemma 2, 2 billion parameters, instruction-tuned)
- **Traits**: refusal, uncertainty, verbosity, overconfidence, corrigibility, evil, sycophantic, hallucinating (8 total)
- **Date Range**: November 10-12, 2024
- **Purpose**: Establish baseline behavioral trait vectors for Gemma 2B model
- **Status**: Complete ✅

## Traits

| Trait | Description | Notes |
|-------|-------------|-------|
| refusal | Declining to answer vs answering directly | Core safety trait |
| uncertainty | Hedging language ("I think") vs confident statements | Epistemic trait |
| verbosity | Long explanations vs concise answers | Response style |
| overconfidence | Making up facts vs admitting unknowns | Accuracy trait |
| corrigibility | Accepting vs resisting corrections | Safety/alignment trait |
| evil | Harmful vs helpful intentions | Safety trait |
| sycophantic | Agreeing with user vs independent judgment | Bias trait |
| hallucinating | Fabricating vs accurate information | Accuracy trait |

## Extraction Details

### Pipeline Used

- **Pipeline**: Legacy pre-refactor scripts (now archived)
- **Method**: Mean difference extraction
- **Layer**: 16 (middle layer of Gemma 2B's 27 layers)
- **Examples**: ~100 positive + ~100 negative per trait
- **Filtering**: Threshold of 50 (pos ≥ 50, neg < 50)
- **Judge Model**: GPT-4o-mini (via OpenAI API)

### Method

- **Extraction method**: Mean difference (`mean(pos_acts) - mean(neg_acts)`)
- **Aggregation**: Response-averaged (mean over all response tokens)
- **Vector type**: Single vector per trait from layer 16

## Results

- **Total vectors**: 8 traits × 1 method (mean_diff) × 1 layer (16) = 8 vectors
- **Quality**: All traits showed good separation in GPT-4 judging
- **Vector norms**: 15-40 range (typical for good trait vectors)

## Files Migrated

### From Original Structure

```
data_generation/trait_data_extract/*.json → trait_definitions
eval/outputs/gemma-2-2b-it/*_pos/neg.csv → responses/
persona_vectors/gemma-2-2b-it/*_response_avg_diff.pt → vectors/mean_diff_layer16.pt
```

### Current Structure

```
gemma_2b_it_nov12/
├── refusal/
│   ├── trait_definition.json
│   ├── responses/
│   │   ├── pos.csv
│   │   └── neg.csv
│   └── vectors/
│       ├── mean_diff_layer16.pt
│       └── mean_diff_layer16_metadata.json
├── uncertainty/
├── verbosity/
├── overconfidence/
├── corrigibility/
├── evil/
├── sycophantic/
└── hallucinating/
```

**Note**: Activations were not saved in the original pipeline (would be ~200 MB). They can be regenerated if needed using `extraction/2_extract_activations.py`.

## Known Limitations

1. **Only mean difference extraction**: No probe, ICA, or gradient methods tested
2. **Single layer**: Only layer 16 extracted (no multi-layer comparison)
3. **No activation cache**: Cannot re-extract with different methods without regenerating
4. **Legacy pipeline**: Used old scripts (before traitlens refactor)

## Usage

These vectors are production-ready and can be used for monitoring:

```bash
python pertoken/monitor_gemma_batch.py \
  --model google/gemma-2-2b-it \
  --vectors experiments/gemma_2b_it_nov12/*/vectors/mean_diff_layer16.pt \
  --layer 16
```

## Future Work

To improve these trait vectors:

1. **Re-extract with new pipeline**: Use `extraction/2_extract_activations.py` to save activations
2. **Test other methods**: Compare probe, ICA, gradient vs mean_diff
3. **Test other layers**: Extract from layers 5, 10, 16, 20, 25
4. **Expand dataset**: Generate more examples (200+ per side instead of ~100)

## Notes

- This experiment used the original per-token-interp pipeline (pre-refactor)
- Migrated to new experiments/ structure on 2024-11-15
- Vector files renamed from `{trait}_response_avg_diff.pt` to `mean_diff_layer16.pt` for consistency
- All 8 traits are actively used in production monitoring
