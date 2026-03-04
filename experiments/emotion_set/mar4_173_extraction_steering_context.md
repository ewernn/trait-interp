# Emotion Set: 173-Trait Steering Evaluation (March 4, 2026)

Full steering validation of all 173 trait vectors in the `emotion_set` experiment.

## Setup

- **Experiment**: `emotion_set`
- **Model**: Qwen/Qwen2.5-14B-Instruct (steering), Qwen/Qwen2.5-14B (extraction)
- **Variants**: `qwen_14b_instruct` (application), `qwen_14b_base` (extraction)
- **Position**: `response[:5]` (first 5 response tokens)
- **Method**: probe vectors, residual component
- **Search**: 5-step coefficient search per layer
- **Directions**: 143 positive, 30 negative

## What We Did

### 1. Layer Selection

Ran extraction evaluation across all layers for each trait, then selected ~10 contiguous layers per trait centered on the best-performing region. Results stored in `experiments/emotion_set/steering/layer_selection.json`.

- Layers per trait: 6–13 (mean 10.1)
- Layer range across all traits: L8–L33
- Total: 1752 layer-slots across 173 traits

### 2. Re-extraction (7 traits)

Seven traits had their datasets rewritten to fix quality issues. Re-extracted locally on A100-80GB (~25s each, all 100% probe accuracy):

- confidence, moral_outrage, sincerity, solemnity, resentment, humility, empathy

### 3. Modal Multi-GPU Steering

Built `analysis/steering/modal_evaluate_all.py` to orchestrate parallel steering eval:

- Reads `layer_selection.json` for per-trait layers
- Reads `steering.json` for direction (positive/negative)
- Greedy bin-packs 173 traits into 9 shards (2 negative + 7 positive GPUs)
- Syncs vectors + datasets to Modal volumes
- Spawns 9 parallel `steering_eval_remote` calls on A100-80GB instances
- Pulls results to local filesystem

**Timing**: Sync 171s, 8/9 GPUs completed in ~30 min. GPU 4 (20 traits, 204 layer-slots) hit the 1800s timeout. Retried those 20 traits split 7/7/6 across 3 GPUs with 3600s timeout — all completed in ~10 min.

### Files Changed

- `inference/modal_steering.py` — Added `trait_layers` parameter to `steering_eval_remote()`, bumped timeout to 3600s
- `analysis/steering/modal_evaluate.py` — Added `--trait-layers` CLI parsing and passthrough
- `analysis/steering/modal_evaluate_all.py` — New orchestrator (direction-aware sharding, parallel spawn)

## Results

**173/173 traits evaluated. All results from 2026-03-04, no stale data.**

### Distribution

| Bucket | Count | Percentage |
|--------|-------|-----------|
| Strong (\|delta\| >= 20) | 168 | 97.1% |
| Medium (10–20) | 2 | 1.2% |
| Weak (< 10) | 3 | 1.7% |

Delta stats: median 57.2, mean 55.5, p10 29.0, p90 75.6

All 173 traits had at least one coherent run (coherence >= 70). No fallback to incoherent pool needed.

### Top 15

| Trait | BL | Best | Delta | Coh | Layer |
|-------|---:|-----:|------:|----:|------:|
| spite | 7.5 | 93.6 | +86.1 | 81 | L25 |
| envy | 7.5 | 91.0 | +83.5 | 81 | L29 |
| contemptuous_dismissal | 8.0 | 91.5 | +83.4 | 80 | L20 |
| vindictiveness | 11.0 | 94.1 | +83.0 | 75 | L19 |
| gratitude | 87.5 | 6.2 | -81.3 | 74 | L14 |
| evasiveness | 9.2 | 89.8 | +80.6 | 71 | L12 |
| impatience | 13.4 | 94.0 | +80.6 | 73 | L22 |
| stubbornness | 14.3 | 94.2 | +79.8 | 82 | L16 |
| shame | 13.5 | 93.0 | +79.5 | 72 | L24 |
| frustration | 15.6 | 93.2 | +77.6 | 71 | L27 |
| dread | 16.1 | 93.6 | +77.5 | 80 | L17 |
| weariness | 16.0 | 93.5 | +77.4 | 75 | L19 |
| boredom | 12.5 | 89.9 | +77.4 | 71 | L15 |
| numbness | 14.7 | 92.0 | +77.3 | 74 | L16 |
| resentment | 3.9 | 81.0 | +77.1 | 75 | L22 |

### Bottom 10 (need review)

| Trait | BL | Best | Delta | Coh | Layer | Notes |
|-------|---:|-----:|------:|----:|------:|-------|
| rebelliousness | 9.6 | 35.7 | +26.0 | 77 | L17 | |
| moral_flexibility | 17.4 | 41.9 | +24.5 | 83 | L26 | May capture rigidity instead |
| perfectionism | 60.3 | 84.7 | +24.3 | 76 | L21 | High baseline already |
| entitlement | 12.6 | 36.9 | +24.3 | 78 | L19 | |
| calm | 18.4 | 38.9 | +20.5 | 86 | L22 | |
| possessiveness | 11.2 | 30.0 | +18.8 | 78 | L22 | Noisy signal |
| self_preservation | 23.4 | 38.5 | +15.1 | 81 | L28 | Weak vector |
| moral_outrage | 21.8 | 29.0 | +7.2 | 85 | L20 | Re-extracted, still weak |
| impulsivity | 15.1 | 9.0 | -6.1 | 73 | L22 | Wrong direction |
| ulterior_motive | 7.9 | 11.6 | +3.6 | 86 | L20 | Near-zero signal |

### Previously Flagged Traits (Status After Re-eval)

| Trait | Prior Flag | Delta | Status |
|-------|-----------|------:|--------|
| manipulation | Wrong subspace | +67.3 | Strong — may have improved with new questions |
| moral_flexibility | Captures rigidity | +24.5 | Still weak — needs re-extraction |
| alignment_faking | Incoherent | +48.2 | Recovered — working now |
| authority_respect | Shows defiance | -66.1 | Strong negative — working as intended |
| ulterior_motive | Too weak | +3.6 | Still broken |
| possessiveness | Noisy | +18.8 | Still marginal |
| self_preservation | Weak signal | +15.1 | Still weak |
| triumph | Caricatured/RLHF | -29.0 | Moderate — needs qualitative review |
| optimism | Chinese at L19 | -48.9 | Working, but needs response review |

## Coherence Threshold Discrepancy (Fixed)

The steering eval ran with `min_coherence=70` (hardcoded in `modal_steering.py`), but `get_best_vector()` uses `MIN_COHERENCE=77` from `utils/vectors.py`. This meant the eval accepted runs that `get_best_vector()` would reject.

**Fix applied**: All hardcoded `70` values replaced with `MIN_COHERENCE` import:
- `inference/modal_steering.py` — eval_args namespace
- `analysis/steering/read_steering_responses.py` — table/best display
- `analysis/steering/prepare_haiku_audit.py` — audit filtering
- `other/server/app.py` — steering request defaults

**Impact**: Since results.jsonl contains all runs regardless of threshold, no re-run needed. The threshold only affects which run gets selected as "best" when reading results later.

### Threshold Comparison: coh>=70 vs coh>=77

Of 173 traits:
- **78 identical picks** — same layer, same coefficient at both thresholds
- **95 different picks** — coh>=77 selects a more conservative (lower delta) run

Delta loss distribution (how much delta we lose at coh>=77):

| Loss bucket | Count |
|-------------|-------|
| 0 (identical) | 78 |
| 0–5 pts | 60 |
| 5–10 pts | 14 |
| 10–20 pts | 17 |
| >20 pts | 4 |

Worst 4: hostility (-25.9), protectiveness (-23.7), guilt_tripping (-21.1), anticipation (-20.4)

### Layer Divergence at Different Thresholds

80 traits pick a different layer at coh>=77 vs coh>=70:

| Layer gap | Count |
|-----------|-------|
| 1 layer | 22 |
| 2 layers | 15 |
| 3 layers | 18 |
| 4 layers | 6 |
| 5 layers | 10 |
| >5 layers | **9** |

**9 traits with >5 layer gap** (fundamentally different vector — needs investigation):

| Trait | L@70 | L@77 | Gap | Delta loss |
|-------|------|------|-----|-----------|
| boredom | L15 | L25 | 10 | +10.9 |
| hope | L14 | L21 | 7 | +2.1 |
| urgency | L21 | L14 | 7 | +0.6 |
| concealment | L23 | L29 | 6 | +3.0 |
| confidence | L28 | L22 | 6 | +0.6 |
| eval_awareness | L23 | L29 | 6 | +3.5 |
| fear | L21 | L27 | 6 | +11.7 |
| hedging | L22 | L16 | 6 | +1.5 |
| servility | L15 | L21 | 6 | +10.4 |

These traits have their best-delta runs in the 70–77 coherence zone, and the coh>=77 fallback lands on a distant layer. Worth reviewing responses at both layers to determine which vector actually captures the trait better.

## Next Steps

1. **Investigate >5 layer gap traits** (9 traits) — compare responses at both layer picks
2. **Qualitative review** of all 173 traits via subagents — read best responses, verify trait expression
3. **Priority review** for weak/flagged traits (~15 traits): ulterior_motive, impulsivity, moral_outrage, self_preservation, possessiveness, moral_flexibility, protectiveness, hostility, guilt_tripping, anticipation, triumph, optimism
4. **Iterate** on broken traits — likely need new datasets for ulterior_motive, impulsivity, moral_outrage
5. Spot-check random sample of strong traits for caricature/gaming artifacts
