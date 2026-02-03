# Experiment: Model Diff

## Goal

Compare steering effectiveness of trait vectors extracted from different sources (instruction-elicited vs natural), positions, and methods. Then analyze what instruction tuning changes in the weight space.

## Hypothesis

1. Natural vectors (from base model) transfer to instruct model but may be less effective
2. Extraction position matters ([:5] vs [:10] vs [:15])
3. Combined vectors (various strategies) don't significantly outperform single-source
4. Weight-space diff is concentrated in mid-layers and aligns with trait directions

## Experiment Structure

```
Phase 0: Extraction (sycophancy first, expand to evil + hallucination later)
Phase 1: Find best single vectors via steering
Phase 2: Test combination strategies
Phase 3: Weight-space analysis (Δθ = θ_instruct - θ_base)
```

## Config

`experiments/model_diff/config.json` (exists):
- base: `meta-llama/Llama-3.1-8B` (32 layers, hidden_dim 4096)
- instruct: `meta-llama/Llama-3.1-8B-Instruct`

## Prerequisites

```bash
# Verify trait datasets exist
ls datasets/traits/pv_instruction/sycophancy/  # positive.jsonl, negative.jsonl, definition.txt, steering.json
ls datasets/traits/pv_natural/sycophancy/       # positive.txt, negative.txt, definition.txt, steering.json

# Verify experiment config
cat experiments/model_diff/config.json

# Verify model configs
cat config/models/llama-3.1-8b.yaml | grep num_hidden_layers  # 32
```

## Known Confounds

The instruction vs natural comparison is confounded on 3 axes:
1. **Elicitation method**: system prompts (instruction) vs first-person narratives (natural)
2. **Model variant**: instruct model vs base model
3. **Dataset content**: 100 opinion questions vs 150 narrative scenarios

This is accepted. Steering is a practical effectiveness comparison, not a controlled isolation of instruction tuning effects. Phase 3 (weight analysis) addresses the mechanistic questions separately.

---

## Phase 0: Extraction

### Extraction matrix (sycophancy):

| Trait source | Model variant | Position | Methods | Scenarios |
|-------------|---------------|----------|---------|-----------|
| pv_instruction/sycophancy | instruct | response[:] | mean_diff, probe | 100 JSONL with system_prompt |
| pv_natural/sycophancy | base | response[:5] | mean_diff, probe | 150 txt (narrative) |
| pv_natural/sycophancy | base | response[:10] | mean_diff, probe | (reuses 0b responses) |
| pv_natural/sycophancy | base | response[:15] | mean_diff, probe | (reuses 0b responses) |

= 4 configs × 2 methods = 8 vector sets

### Step 0a: Extract pv_instruction/sycophancy (instruct, response[:])

**Read first**: `extraction/run_pipeline.py` argparse

**Command**:
```bash
python extraction/run_pipeline.py \
    --experiment model_diff \
    --traits pv_instruction/sycophancy \
    --model-variant instruct \
    --position "response[:]" \
    --methods mean_diff,probe \
    --max-new-tokens 256 \
    --rollouts 2
```

**Why `--max-new-tokens 256`**: Without this, `response[:]` resolves to 16-token fallback via `resolve_max_new_tokens()`. The instruction-based methodology needs enough tokens for sycophancy to manifest in the response.

**Why `--rollouts 2`**: Multiple samples per scenario (100 scenarios × 2 = 200 responses per polarity). Original PV paper used 10 rollouts with 1000 tokens; we use fewer but sufficient for extraction.

**Expected output**:
```
experiments/model_diff/extraction/pv_instruction/sycophancy/instruct/
├── responses/pos.json, neg.json
├── activations/response_all/residual/
│   ├── train_all_layers.pt, val_all_layers.pt
│   └── metadata.json
└── vectors/response_all/residual/{mean_diff,probe}/layer*.pt (32 each)
```

**Verify**:
```bash
ls experiments/model_diff/extraction/pv_instruction/sycophancy/instruct/vectors/response_all/residual/mean_diff/ | wc -l
ls experiments/model_diff/extraction/pv_instruction/sycophancy/instruct/vectors/response_all/residual/probe/ | wc -l
# Both should be 32
```

### Step 0b: Extract pv_natural/sycophancy (base, response[:5])

```bash
python extraction/run_pipeline.py \
    --experiment model_diff \
    --traits pv_natural/sycophancy \
    --model-variant base \
    --position "response[:5]" \
    --methods mean_diff,probe
```

**Expected output**:
```
experiments/model_diff/extraction/pv_natural/sycophancy/base/
├── responses/pos.json, neg.json
├── activations/response__5/residual/...
└── vectors/response__5/residual/{mean_diff,probe}/layer*.pt (32 each)
```

### Step 0c: Extract pv_natural/sycophancy (base, response[:10])

```bash
python extraction/run_pipeline.py \
    --experiment model_diff \
    --traits pv_natural/sycophancy \
    --model-variant base \
    --position "response[:10]" \
    --methods mean_diff,probe
```

**Note**: Reuses responses from 0b. The pipeline checks if `responses/pos.json` and `neg.json` exist and skips generation (`run_pipeline.py:199-210`). Response path does NOT include position, so all base/pv_natural extractions share the same responses.

### Step 0d: Extract pv_natural/sycophancy (base, response[:15])

```bash
python extraction/run_pipeline.py \
    --experiment model_diff \
    --traits pv_natural/sycophancy \
    --model-variant base \
    --position "response[:15]" \
    --methods mean_diff,probe
```

### Checkpoint: After Phase 0

```bash
# Count total vector directories (should be 8: 4 configs × 2 methods)
find experiments/model_diff/extraction -name "layer0.pt" | wc -l
# Should be 8

# Verify positions created correctly
ls experiments/model_diff/extraction/pv_instruction/sycophancy/instruct/vectors/
# response_all/

ls experiments/model_diff/extraction/pv_natural/sycophancy/base/vectors/
# response__5/  response__10/  response__15/
```

---

## Phase 1: Find Best Single Vectors

All steering runs apply vectors to the **instruct model** (config default: `"application": "instruct"`).

### Steering matrix (sycophancy, 8 combos):

| Source | Position | Method | Key |
|--------|----------|--------|-----|
| pv_instruction | response[:] | mean_diff | inst_md |
| pv_instruction | response[:] | probe | inst_pr |
| pv_natural | response[:5] | mean_diff | nat5_md |
| pv_natural | response[:5] | probe | nat5_pr |
| pv_natural | response[:10] | mean_diff | nat10_md |
| pv_natural | response[:10] | probe | nat10_pr |
| pv_natural | response[:15] | mean_diff | nat15_md |
| pv_natural | response[:15] | probe | nat15_pr |

### Step 1a: Coarse steering search (all 8 combos)

Adaptive search (default), layers 9-19 (30%-60% of 32). `--subset 0` = all 20 questions, `--max-new-tokens 128`.

**Commands** (run sequentially — share model, different vectors):

```bash
# pv_instruction, instruct model extraction
python analysis/steering/evaluate.py \
    --experiment model_diff \
    --vector-from-trait model_diff/pv_instruction/sycophancy \
    --extraction-variant instruct \
    --method mean_diff --position "response[:]" \
    --subset 0 --max-new-tokens 128

python analysis/steering/evaluate.py \
    --experiment model_diff \
    --vector-from-trait model_diff/pv_instruction/sycophancy \
    --extraction-variant instruct \
    --method probe --position "response[:]" \
    --subset 0 --max-new-tokens 128

# pv_natural, base model extraction, response[:5]
python analysis/steering/evaluate.py \
    --experiment model_diff \
    --vector-from-trait model_diff/pv_natural/sycophancy \
    --extraction-variant base \
    --method mean_diff --position "response[:5]" \
    --subset 0 --max-new-tokens 128

python analysis/steering/evaluate.py \
    --experiment model_diff \
    --vector-from-trait model_diff/pv_natural/sycophancy \
    --extraction-variant base \
    --method probe --position "response[:5]" \
    --subset 0 --max-new-tokens 128

# pv_natural, base model extraction, response[:10]
python analysis/steering/evaluate.py \
    --experiment model_diff \
    --vector-from-trait model_diff/pv_natural/sycophancy \
    --extraction-variant base \
    --method mean_diff --position "response[:10]" \
    --subset 0 --max-new-tokens 128

python analysis/steering/evaluate.py \
    --experiment model_diff \
    --vector-from-trait model_diff/pv_natural/sycophancy \
    --extraction-variant base \
    --method probe --position "response[:10]" \
    --subset 0 --max-new-tokens 128

# pv_natural, base model extraction, response[:15]
python analysis/steering/evaluate.py \
    --experiment model_diff \
    --vector-from-trait model_diff/pv_natural/sycophancy \
    --extraction-variant base \
    --method mean_diff --position "response[:15]" \
    --subset 0 --max-new-tokens 128

python analysis/steering/evaluate.py \
    --experiment model_diff \
    --vector-from-trait model_diff/pv_natural/sycophancy \
    --extraction-variant base \
    --method probe --position "response[:15]" \
    --subset 0 --max-new-tokens 128
```

**Results written to** (example for first command):
```
experiments/model_diff/steering/pv_instruction/sycophancy/instruct/response_all/steering/results.jsonl
```

### Checkpoint: After Step 1a

**Reflect**: For each of the 8 combos, report:
- Best layer
- Max trait delta with coherence ≥ 80
- Max trait delta with coherence ≥ 70
- Where coherence first drops below 80

Then decide where to do fine-grained search.

### Step 1b: Fine-grained search

Based on Step 1a results, run `--coefficients` with finer grid around the coherence drop-off point. Example:

```bash
python analysis/steering/evaluate.py \
    --experiment model_diff \
    --vector-from-trait model_diff/pv_instruction/sycophancy \
    --extraction-variant instruct \
    --method mean_diff --position "response[:]" \
    --layers 12 \
    --coefficients 3.0,3.5,4.0,4.5,5.0,5.5,6.0 \
    --subset 0 --max-new-tokens 128
```

Specific commands depend on Step 1a results.

---

## Phase 2: Combination Strategies

**Requires**: Best vectors from Phase 1 (best layer, best coefficient per source).

**Script**: `experiments/model_diff/scripts/combination_steering.py` (to be written)

Uses existing primitives:
- `BatchedLayerSteeringHook` (`core/hooks.py:577`) for multi-layer steering
- `batched_steering_generate()` (`core/steering.py:25`) for single-layer combos
- `TraitJudge.score_steering_batch()` (`utils/judge.py`) for scoring
- Vector loading via `utils/vectors.py:load_vector()`

### Strategy 1: Both vectors at natural's best layer

Load natural vector + instruct vector (both from their best methods per Phase 1), add them, normalize, steer at natural's best layer with coefficient sweep.

### Strategy 2: Both vectors at instruct's best layer

Same vector addition, steer at instruct's best layer.

### Strategy 3: Best layer of either

Both vectors at whichever single source had the higher trait delta at coherence ≥ 70.

### Strategy 4: Ensemble (two hooks, two layers)

Steer layer X with natural_best at half the optimal single-source coefficient, layer Y with instruct_best at half its optimal coefficient. Uses `BatchedLayerSteeringHook` with overlapping batch slices (verified: `optimize_ensemble.py:158-175` demonstrates this pattern).

### For each strategy

Sweep coefficients to find max trait delta where coherence ≥ 70.

### Checkpoint: After Phase 2

Compare all strategies vs single-source best:

| Config | Layer(s) | Coef | Trait Δ | Coherence | Notes |
|--------|----------|------|---------|-----------|-------|
| Best single (inst) | ? | ? | ? | ? | Phase 1 winner |
| Best single (nat) | ? | ? | ? | ? | Phase 1 winner |
| Strategy 1 | nat_layer | ? | ? | ? | Combined @ natural layer |
| Strategy 2 | inst_layer | ? | ? | ? | Combined @ instruct layer |
| Strategy 3 | best_layer | ? | ? | ? | Combined @ best layer |
| Strategy 4 | nat+inst | ? | ? | ? | Ensemble half-strength |

---

## Phase 3: Weight-Space Analysis

**Script**: `experiments/model_diff/scripts/weight_analysis.py` (to be written)

Runs on GPU (32GB). Loads both models sequentially to compute diffs.

### Step 3a: Compute weight norms per layer

```python
# For each layer 0-31, for each component:
# attn: q_proj, k_proj, v_proj, o_proj
# mlp: gate_proj, up_proj, down_proj
#
# Compute: ||θ_instruct[layer][component] - θ_base[layer][component]||_F (Frobenius norm)
# Also: ||Δθ|| / ||θ_base|| (relative change)
```

**Command**:
```bash
python experiments/model_diff/scripts/weight_analysis.py \
    --experiment model_diff \
    --step norms
```

**Output**: `experiments/model_diff/weight_analysis/norms.json`

### Step 3b: Logit lens on weight diff

For each layer, compute the mean weight diff across residual-stream-facing components, then project through unembedding via `core/logit_lens.py:vector_to_vocab()`.

```python
# Per layer:
# 1. Δ_residual = mean of column-space diffs that affect residual stream
#    (o_proj contribution + down_proj contribution)
# 2. top_tokens = vector_to_vocab(model, Δ_residual, k=20, apply_norm=True)
```

**Command**:
```bash
python experiments/model_diff/scripts/weight_analysis.py \
    --experiment model_diff \
    --step logit_lens
```

**Output**: `experiments/model_diff/weight_analysis/logit_lens.json` (top 20 tokens per layer, both directions)

### Step 3c: Trait alignment

Compute cosine similarity between weight diff direction and extracted trait vectors at each layer.

```python
# For each layer:
# cos_sim = cosine_similarity(Δ_residual[layer], trait_vector[layer])
# Uses trait vectors from Phase 0 (best method per source)
```

**Command**:
```bash
python experiments/model_diff/scripts/weight_analysis.py \
    --experiment model_diff \
    --step trait_alignment \
    --traits pv_instruction/sycophancy,pv_natural/sycophancy
```

**Output**: `experiments/model_diff/weight_analysis/trait_alignment.json`

### Checkpoint: After Phase 3

Report:
- Which layers have largest ||Δθ||? Do they align with best steering layers from Phase 1?
- What tokens does logit lens reveal? Do they relate to sycophancy?
- How aligned is Δθ with trait vectors? Does alignment peak at the same layers that steer best?

---

## Success Criteria

- [ ] Phase 0: 8 vector sets extracted for sycophancy
- [ ] Phase 1: Best vector identified per source × position × method (8 configs)
- [ ] Phase 1: Clear answer on whether position matters ([:5] vs [:10] vs [:15])
- [ ] Phase 2: Combination strategies tested, compared to single-source best
- [ ] Phase 3: Weight diff analysis complete with interpretable results

## Notes

**Start with sycophancy only.** Expand to evil + hallucination after validating the pipeline.

**Method confound removed**: Both mean_diff and probe extracted for both sources.

**Elicitation confound accepted**: Instruction (system prompts → instruct model) vs natural (narratives → base model) differ on multiple axes. This is a practical effectiveness comparison, not a controlled ablation.

**Steering questions**: Both pv_instruction and pv_natural share the same `steering.json` (20 opinion questions). This ensures fair comparison — same evaluation prompts for all vectors.

**Trait version mapping** (for later expansion):
- sycophancy → datasets/traits/pv_instruction/sycophancy + pv_natural/sycophancy
- evil → pv_instruction/evil + pv_natural/evil_v3
- hallucination → pv_instruction/hallucination + pv_natural/hallucination_v2

## If Stuck

- **Step 0a generates weak vectors**: Increase `--max-new-tokens` or `--rollouts`. Check response quality in `pos.json`.
- **Steering finds no effect**: Try wider layer range `--layers all` or check vectors aren't zero/tiny.
- **Phase 2 ensemble crashes**: Verify `BatchedLayerSteeringHook` config format: `(layer, vector, coef, (batch_start, batch_end))`.
- **Phase 3 OOM**: Load models sequentially, compute diff, free first model before loading second.
