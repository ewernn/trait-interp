# Optimism Steering Sweep Notes

## Setup
- **Target model**: gemma-2-2b-it
- **Vector source**: gemma-2-2b-base/epistemic/optimism (probe method)
- **Baseline trait score**: 61.3

## Command Pattern
```bash
export HF_HOME=/home/dev/.cache/huggingface && CUDA_VISIBLE_DEVICES=0 python3 analysis/steering/evaluate.py \
    --experiment gemma-2-2b-it \
    --trait epistemic/optimism \
    --vector-from-trait gemma-2-2b-base/epistemic/optimism \
    --layers 10 \
    --coefficients "100,125,150" \
    --subset 5
```

Can run 2 processes per GPU (8 total across 4 GPUs). Each uses ~5.3GB VRAM.

## Best Configurations

| Rank | Layer | Coef | Trait | Coherence | Delta | Notes |
|------|-------|------|-------|-----------|-------|-------|
| 1 | **L8** | **125** | **88.9** | **84.7** | **+27.6** | **NEW BEST** |
| 2 | L10 | 130 | 87.3 | 86.8 | +26.0 | Best coherence |
| 3 | L14 | 250 | 86.9 | 82.9 | +25.6 | |
| 4 | L7 | 125 | 86.1 | 68.6 | +24.8 | Lower coherence |

## Results Summary - All Layers

### Early Layers (L5-L8)
| Layer | Coef | Trait | Coherence | Notes |
|-------|------|-------|-----------|-------|
| L5 | 25 | 49.3 | 84.4 | |
| L5 | 50 | 34.8 | 73.0 | Coherence breaking |
| L5 | 75 | 66.0 | 58.0 | |
| L6 | 25 | 52.8 | 85.5 | |
| L6 | 50 | 76.2 | 83.2 | Good sweet spot |
| L6 | 75 | 63.4 | 60.8 | Coherence breaking |
| L6 | 100 | 64.7 | 56.6 | |
| L7 | 50 | 50.7 | 86.9 | |
| L7 | 75 | 69.2 | 84.8 | |
| L7 | 100 | 80.0 | 67.7 | Coherence breaking |
| L7 | 125 | 86.1 | 68.6 | |
| L8 | 50 | 57.5 | 84.3 | |
| L8 | 75 | 76.8 | 86.3 | |
| L8 | 100 | 83.1 | 79.9 | |
| **L8** | **125** | **88.9** | **84.7** | **BEST OVERALL** |

### Middle Layers (L9-L14)
| Layer | Coef | Trait | Coherence | Notes |
|-------|------|-------|-----------|-------|
| L9 | 130 | 77.7 | 69.7 | Already degraded |
| L9 | 155 | 81.4 | 67.2 | |
| L9 | 180 | 86.3 | 75.8 | |
| L9 | 205 | 80.1 | 65.3 | |
| **L10** | **130** | **87.3** | **86.8** | **Best coherence** |
| L10 | 155 | 87.2 | 82.0 | |
| L10 | 180 | 86.3 | 71.3 | |
| L10 | 205 | 87.2 | 71.6 | |
| L11 | 130 | 81.3 | 84.1 | |
| L11 | 155 | 84.8 | 80.0 | |
| L11 | 180 | 85.7 | 77.3 | |
| L11 | 205 | 85.4 | 74.8 | |
| L12 | 150 | 80.5 | 85.0 | |
| L12 | 175 | 86.1 | 81.9 | |
| L12 | 200 | 84.8 | 77.3 | |
| L12 | 225 | 85.1 | 75.2 | |
| L12 | 250 | 83.6 | 67.2 | |
| L12 | 275 | 80.5 | 57.9 | |
| L13 | 150 | 76.8 | 82.8 | |
| L13 | 175 | 79.4 | 81.6 | |
| L13 | 200 | 83.1 | 82.5 | |
| L13 | 225 | 83.9 | 81.0 | |
| L13 | 250 | 83.6 | 82.1 | |
| L13 | 275 | 85.4 | 76.7 | |
| L13 | 300 | 83.4 | 72.8 | |
| L14 | 150 | 75.2 | 83.7 | |
| L14 | 175 | 80.3 | 83.5 | |
| L14 | 200 | 83.6 | 84.6 | |
| L14 | 225 | 86.4 | 83.2 | |
| **L14** | **250** | **86.9** | **82.9** | **Highest trait (mid)** |
| L14 | 275 | 86.0 | 76.2 | |
| L14 | 300 | 86.3 | 69.1 | |
| L14 | 325 | 86.4 | 62.7 | |
| L14 | 350 | 86.2 | 60.4 | |

### Late-Middle Layers (L15-L18)
| Layer | Coef | Trait | Coherence | Notes |
|-------|------|-------|-----------|-------|
| L15 | 150 | 64.1 | 84.3 | |
| L15 | 175 | 76.8 | 84.9 | |
| L15 | 200 | 77.1 | 82.3 | |
| L15 | 225 | 79.1 | 82.6 | |
| L15 | 300 | 82.6 | 63.8 | |
| L15 | 325 | 76.5 | 57.6 | |
| L15 | 350 | 67.4 | 44.1 | |
| L15 | 375 | 57.2 | 34.0 | |
| L15 | 400 | 55.9 | 24.5 | |
| L16 | 150 | 61.5 | 87.0 | |
| L16 | 175 | 58.4 | 84.6 | |
| L16 | 200 | 56.8 | 84.4 | |
| L16 | 225 | 66.2 | 84.3 | |
| L16 | 325 | 65.4 | 78.4 | |
| L16 | 350 | 75.7 | 72.4 | |
| L16 | 375 | 79.0 | 54.9 | |
| L16 | 400 | 61.0 | 45.6 | |
| L16 | 425 | 36.3 | 18.0 | |
| L17 | 350 | 75.3 | 82.8 | |
| L17 | 375 | 75.9 | 79.0 | |
| L17 | 400 | 72.7 | 76.6 | |
| L17 | 425 | 70.7 | 71.1 | |
| L17 | 450 | 68.9 | 60.7 | |
| L18 | 400 | 68.7 | 85.0 | |
| L18 | 425 | 65.3 | 84.0 | |
| L18 | 450 | 62.6 | 77.3 | |
| L18 | 475 | 70.7 | 75.7 | |
| L18 | 500 | 69.0 | 72.8 | |

### Late Layers (L19-L25) - Minimal Effect
| Layer | Coef | Trait | Coherence | Notes |
|-------|------|-------|-----------|-------|
| L19 | 425 | 68.3 | 82.9 | |
| L19 | 475 | 64.3 | 82.9 | |
| L19 | 525 | 47.6 | 66.5 | |
| L19 | 575 | 60.3 | 74.6 | |
| L20 | 475 | 60.7 | 84.8 | |
| L20 | 525 | 63.2 | 84.8 | |
| L20 | 575 | 61.7 | 84.9 | |
| L20 | 625 | 39.1 | 48.0 | |
| L21 | 525 | 61.9 | 85.0 | |
| L21 | 575 | 61.3 | 84.3 | |
| L21 | 625 | 61.0 | 83.7 | |
| L21 | 675 | 67.1 | 83.1 | |
| L21 | 800 | 47.1 | 50.7 | |
| L21 | 850 | 60.1 | 71.2 | |
| L21 | 900 | 54.2 | 64.1 | |
| L21 | 950 | 54.9 | 48.9 | |
| L21 | 1000 | 62.3 | 46.2 | |
| L22 | 575 | 57.7 | 82.1 | |
| L22 | 625 | 60.3 | 84.4 | |
| L22 | 675 | 56.4 | 83.4 | |
| L22 | 725 | 55.5 | 81.4 | |
| L22 | 850 | 50.3 | 83.9 | |
| L22 | 900 | 57.8 | 74.3 | |
| L22 | 950 | 60.0 | 64.2 | |
| L22 | 1000 | 51.6 | 56.5 | |
| L22 | 1050 | 54.4 | 63.0 | |
| L23 | 625 | 49.8 | 84.3 | |
| L23 | 675 | 50.4 | 84.8 | |
| L23 | 725 | 55.4 | 83.9 | |
| L23 | 775 | 49.7 | 84.0 | |
| L23 | 900 | 55.2 | 83.4 | |
| L23 | 950 | 58.0 | 84.0 | |
| L23 | 1000 | 56.9 | 81.1 | |
| L23 | 1050 | 59.3 | 80.8 | |
| L23 | 1100 | 55.5 | 79.0 | |
| L24 | 675 | 49.4 | 84.4 | |
| L24 | 725 | 49.1 | 84.3 | |
| L24 | 775 | 48.1 | 85.3 | |
| L24 | 825 | 51.1 | 84.8 | |
| L24 | 950 | 52.7 | 84.2 | |
| L24 | 1000 | 53.1 | 82.5 | |
| L24 | 1050 | 50.0 | 80.0 | |
| L24 | 1100 | 42.4 | 82.5 | |
| L24 | 1150 | 42.9 | 79.3 | |
| L25 | 725 | 42.3 | 85.8 | |
| L25 | 775 | 37.1 | 85.0 | |
| L25 | 825 | 36.3 | 85.0 | |
| L25 | 875 | 38.4 | 86.4 | |
| L25 | 1000 | 47.3 | 85.5 | |
| L25 | 1050 | 40.3 | 86.4 | |
| L25 | 1100 | 50.4 | 84.9 | |
| L25 | 1150 | 50.7 | 85.8 | |
| L25 | 1200 | 49.5 | 85.4 | |

## Coherence Degradation Points (All Layers) - COMPLETE

| Layer | Degrades at | Safe max coef | Best trait | Notes |
|-------|-------------|---------------|------------|-------|
| L0 | ~25 | ~10 | 51.7 | Embedding layer, instant break |
| L1 | ~25 | ~10 | 57.5 | Embedding layer, instant break |
| L2 | ~25 | ~10 | 51.2 | Embedding layer, instant break |
| L3 | ~25 | ~10 | 54.9 | Very sensitive |
| L4 | ~50 | ~25 | 82.1 | Good trait at breakdown |
| L5 | ~50 | ~25 | 66.0 | Very sensitive |
| L6 | ~75 | ~50 | 76.2 | |
| L7 | ~100 | ~75 | 86.1 | |
| L8 | ~150 | ~125 | 88.9 | **BEST OVERALL** |
| L9 | ~130 | <130 | 86.3 | Already degraded |
| L10 | ~180 | ~155 | 87.3 | **Best coherence** |
| L11 | ~180 | ~155 | 85.7 | |
| L12 | ~200 | ~175 | 86.1 | |
| L13 | ~275 | ~250 | 85.4 | |
| L14 | ~275 | ~250 | 86.9 | |
| L15 | ~300 | ~225 | 82.6 | |
| L16 | ~375 | ~350 | 79.0 | |
| L17 | ~450 | ~400 | 75.9 | |
| L18 | ~500 | ~450 | 70.7 | |
| L19 | ~525 | ~475 | 68.3 | Minimal effect |
| L20 | ~625 | ~575 | 63.2 | Minimal effect |
| L21 | ~800 | ~675 | 67.1 | Minimal effect |
| L22 | ~950 | ~725 | 60.3 | Minimal effect |
| L23 | >1100 | ~1050 | 59.3 | Never really breaks |
| L24 | >1150 | ~1000 | 53.1 | Never really breaks |
| L25 | Never | 1200+ | 50.7 | No effect, no degradation |

## Very Early Layers (L0-L4) Results

| Layer | Coef | Trait | Coherence | Notes |
|-------|------|-------|-----------|-------|
| L0 | 10 | 51.7 | 85.3 | |
| L0 | 25 | 16.5 | 6.9 | Instant breakdown |
| L0 | 50 | 0.0 | 0.0 | Complete gibberish |
| L1 | 10 | 57.5 | 86.1 | |
| L1 | 25 | 5.8 | 22.7 | Instant breakdown |
| L1 | 50 | N/A | 0.0 | Complete gibberish |
| L2 | 10 | 51.2 | 84.1 | |
| L2 | 25 | 12.7 | 22.7 | Instant breakdown |
| L2 | 50 | 0.0 | 7.8 | Complete gibberish |
| L3 | 10 | 54.9 | 83.9 | |
| L3 | 25 | 44.7 | 74.7 | Degrading |
| L3 | 50 | 0.0 | 12.6 | Complete breakdown |
| L4 | 10 | 46.7 | 84.7 | |
| L4 | 25 | 54.7 | 84.5 | |
| L4 | 50 | 82.1 | 55.1 | Good trait, low coherence |

## L8 High Coefficient Results

| Layer | Coef | Trait | Coherence | Notes |
|-------|------|-------|-----------|-------|
| L8 | 125 | 88.9 | 84.7 | **BEST** |
| L8 | 150 | 81.5 | 57.2 | Coherence breaking |
| L8 | 175 | 72.3 | 32.5 | Severely degraded |
| L8 | 200 | 7.1 | 6.6 | Complete breakdown |

## Key Findings

1. **Best overall**: L8 c125 (trait=88.9, coherence=84.7, +27.6 delta)
2. **Best coherence**: L10 c130 (trait=87.3, coherence=86.8, +26.0 delta)
3. **Sweet spot layers**: L8-L14 with coefs 100-250
4. **Early layers (L5-L7)**: Very sensitive, degrade fast but can get good trait effect
5. **Very early layers (L0-L4)**: Embedding/position layers break instantly at low coefs
6. **Late layers (L19+)**: Need very high coefficients, minimal trait effect regardless
7. **L23-L25**: Coherence never really breaks, but trait never moves either

## Layer Zones Summary

| Zone | Layers | Coef Range | Trait Effect | Coherence Sensitivity |
|------|--------|------------|--------------|----------------------|
| Embedding | L0-L2 | 10 only | None | **Extreme** |
| Very Early | L3-L4 | 10-25 | Low | Very High |
| Early | L5-L6 | 25-75 | Moderate | Very High |
| Early-Mid | L7-L8 | 75-150 | **High** | Moderate |
| Sweet Spot | L9-L14 | 100-275 | **High** | Moderate |
| Transition | L15-L18 | 200-500 | Moderate | High |
| Late | L19-L22 | 400-1000 | Low | Variable |
| Final | L23-L25 | 600-1200+ | None | Low |

## SWEEP COMPLETE

All 26 layers tested. Best configurations:
1. **L8 c125**: trait=88.9, coherence=84.7 (best overall)
2. **L10 c130**: trait=87.3, coherence=86.8 (best coherence)
3. **L14 c250**: trait=86.9, coherence=82.9
