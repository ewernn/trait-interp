# Emotion Set: Overnight Trait Dataset Creation

## Goal
Create and validate trait vector datasets for 173 traits in experiment `emotion_set`.

## Architecture
- **Orchestrator (Opus)**: manages queue, spawns subagents ~10 at a time, tracks results
- **Workers (Sonnet × 10)**: each creates one trait dataset end-to-end (max 3 attempts)
- **Readers (Haiku)**: spawned by workers to read steering responses qualitatively

## Config
- Experiment: `emotion_set`
- Extraction model: Qwen/Qwen2.5-14B (base) — `qwen_14b_base`
- Steering model: Qwen/Qwen2.5-14B-Instruct — `qwen_14b_instruct`
- Datasets: `datasets/traits/emotion_set/{trait}/`
- Modal apps: `trait-extraction`, `trait-steering` (both deployed)
- Protocol: `docs/trait_dataset_creation.md` — Autonomous Agent Protocol section

## Per-Trait Protocol (from doc)
1. Create dataset files (definition.txt, positive.txt, negative.txt, steering.json)
2. Extract vectors via Modal
3. Steer L10-30 via Modal (dense sweep, every layer)
4. Evaluate: baseline + quantitative + qualitative (haiku subagent reads responses)
5. Iterate if needed (max 3 attempts)
6. SUCCESS = delta ≥25, baseline <20, coherence ≥70, qualitative GOOD/OK

## Traits (173 total)
150 from emotions/ (minus 5 experiment variants) + 23 from mats-emergent-misalignment/ (minus chinese, assistant)

## Results

Format: `trait | status | attempts | best_layer | best_delta | coherence | baseline | notes`

### Batch 1 (acceptance, adaptability, affection, agency, alignment_faking, allegiance, ambition, amusement, analytical, anger)
| Trait | Status | Attempts | Layer | Delta | Coh | Baseline | Notes |
|-------|--------|----------|-------|-------|-----|----------|-------|
| affection | SUCCESS | 1 | L23 | +60.8 | 72.3 | 14.2 | |
| alignment_faking | SUCCESS | 1 | L28 | +70.9 | 71.5 | 13.4 | |
| anger | SUCCESS | 1 | L23 | +60.4 | 75.1 | 4.3 | |
| amusement | HIGH_BL | 1 | L21 | +47.0 | 84.6 | 32.1 | elevated baseline |
| acceptance | BORDERLINE | 2+ | L16 | +13.5 | 86.3 | 18.8 | bl fixed, weak delta |
| adaptability | BORDERLINE | 2+ | L30 | +22.7 | 72.3 | 13.5 | bl fixed, near threshold |
| agency | HIGH_BL | 2+ | L28 | +32.5 | 85.2 | 31.6 | strong delta, high bl |
| allegiance | BORDERLINE | 1 | L23 | +53.6 | 77.9 | 28.8 | strong delta, bl slightly high |
| ambition | BORDERLINE | 1 | L21 | +23.2 | 82.3 | 16.8 | qual GOOD per agent |
| analytical | HIGH_BL | 3 | L19 | +16.1 | 88.3 | 62.0 | very high baseline, hard trait |

### Batch 2 (anticipation, anxiety)
| Trait | Status | Attempts | Layer | Delta | Coh | Baseline | Notes |
|-------|--------|----------|-------|-------|-----|----------|-------|
| anticipation | SUCCESS | 1 | L13 | +48.3 | 72.8 | 7.3 | |
| anxiety | SUCCESS | 1 | L24 | +78.3 | 76.3 | 15.0 | |

### Batch 3 (apathy, assertiveness, authority_respect, avoidance, awe, awkwardness)
| Trait | Status | Attempts | Layer | Delta | Coh | Baseline | Notes |
|-------|--------|----------|-------|-------|-----|----------|-------|
| apathy | SUCCESS | 1 | L27 | +73.0 | 70.6 | 10.5 | |
| avoidance | SUCCESS | 1 | L24 | +68.1 | 75.9 | 17.1 | |
| awe | SUCCESS | 1 | L20 | +71.5 | 78.6 | 13.8 | |
| awkwardness | SUCCESS | 1 | L25 | +77.3 | 78.2 | 11.3 | |
| assertiveness | HIGH_BL | 1 | L28 | +13.6 | 81.4 | 41.0 | |
| authority_respect | HIGH_BL | 1 | L16 | +20.8 | 71.7 | 64.9 | |

### Batch 4 (boredom, brevity, calm, carefulness)
| Trait | Status | Attempts | Layer | Delta | Coh | Baseline | Notes |
|-------|--------|----------|-------|-------|-----|----------|-------|
| boredom | SUCCESS | 1 | L24 | +76.9 | 78.3 | 11.4 | |
| brevity | HIGH_BL | 1 | L13 | +20.5 | 88.9 | 36.5 | |
| calm | HIGH_BL | 1 | L16 | +44.6 | 81.6 | 45.0 | |
| carefulness | RUNNING | - | - | - | - | - | agent respawned |

### Batch 5 (caution, certainty, compassion, competitiveness, concealment, confidence, contempt)
| Trait | Status | Attempts | Layer | Delta | Coh | Baseline | Notes |
|-------|--------|----------|-------|-------|-----|----------|-------|
