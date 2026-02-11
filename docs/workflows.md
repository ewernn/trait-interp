# Workflows

Practical guide to common workflows. See referenced files for full options.

---

## Decision Tree

**"I want to..."**

| Goal | Workflow |
|------|----------|
| Extract a new trait | [Extraction](#extraction) |
| Monitor traits during generation | [Inference](#inference) |
| Validate vectors work causally | [Steering](#steering) |
| Compare model variants (base vs IT, clean vs LoRA) | [Model Comparison](#model-comparison) |
| Test capability preservation with ablation | [Benchmarking](#benchmarking) |
| Interpret what a vector represents | [Vector Interpretation](#vector-interpretation) |
| Visualize results | [Visualization](#visualization) |
| Sync data to/from cloud | [R2 Sync](#r2-sync) |

---

## Extraction

Extract trait vectors from contrasting scenarios.

```bash
# 1. Create scenario files
mkdir -p datasets/traits/{category}/{trait}
# Create: positive.txt, negative.txt, definition.txt, steering.json

# 2. Run full pipeline
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}
```

**Outputs:**
- `extraction/{trait}/{variant}/vectors/{position}/{component}/{method}/layer*.pt`
- `extraction/{trait}/{variant}/responses/pos.json, neg.json`
- `steering/{trait}/.../results.jsonl`

**Variants:**
- `--no-steering` — Skip steering validation
- `--no-vet` — Skip LLM vetting stages
- `--position "prompt[-1]"` — Arditi-style (last prompt token)
- `--base-model` — Text completion mode

**Details:** [docs/extraction_pipeline.md](extraction_pipeline.md)

---

## Inference

Monitor traits token-by-token during generation.

```bash
# 1. Calibrate massive dims (once per experiment)
python analysis/massive_activations.py --experiment {experiment}

# 2. Capture raw activations
python inference/capture_raw_activations.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# 3. Project onto traits
python inference/project_raw_activations_onto_traits.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}
```

**Outputs:**
- `inference/{variant}/raw/residual/{prompt_set}/*.pt` — Large, delete after projecting
- `inference/{variant}/projections/{trait}/{prompt_set}/*.json` — Small, keep these

**Prompt sets:** `datasets/inference/*.json` (single_trait, harmful, benign, etc.)

**Details:** [inference/README.md](../inference/README.md)

---

## Steering

Validate vectors via causal intervention.

```bash
python analysis/steering/evaluate.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait}
```

**Outputs:** `steering/{trait}/{variant}/{position}/{prompt_set}/results.jsonl`

**Variants:**
- `--layers 10,12,14` — Specific layers
- `--no-batch` — Lower memory (sequential)
- `--coefficients 50,100,150` — Skip adaptive search

**Advanced (CMA-ES optimization):**
```bash
python analysis/steering/optimize_vector.py \
    --experiment {experiment} \
    --trait {category}/{trait} \
    --layers 8,9,10,11,12
```

**Details:** [analysis/steering/README.md](../analysis/steering/README.md)

---

## Model Comparison

Compare how different model variants represent traits on identical tokens.

```bash
# 1. Generate responses with variant A
python inference/capture_raw_activations.py \
    --experiment {experiment} \
    --model-variant {variant_a} \
    --prompt-set {prompt_set}

# 2. Prefill variant B with A's responses (same tokens, different model)
python inference/capture_raw_activations.py \
    --experiment {experiment} \
    --model-variant {variant_b} \
    --prompt-set {prompt_set} \
    --replay-responses {prompt_set} \
    --replay-from-variant {variant_a}

# 3. Project both
python inference/project_raw_activations_onto_traits.py \
    --experiment {experiment} \
    --model-variant {variant_a} \
    --prompt-set {prompt_set}

python inference/project_raw_activations_onto_traits.py \
    --experiment {experiment} \
    --model-variant {variant_b} \
    --prompt-set {prompt_set}

# 4. View in visualization → Model Comparison tab

# 5. (Optional) Per-token/per-clause diff analysis
python analysis/model_diff/per_token_diff.py \
    --experiment {experiment} \
    --variant-a {variant_a} \
    --variant-b {variant_b} \
    --prompt-set {prompt_set} \
    --trait all --top-pct 5
```

**Use cases:**
- Base vs instruction-tuned
- Clean vs LoRA-finetuned
- Before/after safety training

**Key insight:** Uses `--replay-responses` to run same tokens through different model (prefilling). For large models, use `--layers` to only capture the layers where your best steering vectors live.

**System prompt:** Prompt JSON files can include `"system_prompt": "..."` at the top level — automatically applied via chat template. Use `--response-only` to skip saving prompt token activations (saves space with long system prompts).

**Cross-variant replay:** When replaying multiple variants through the same baseline model, use `--output-suffix replay_{variant}` to keep outputs separate. Then use `--variant-a-prompt-set` in `per_token_diff.py` to pair them correctly.

**Per-token diff** (step 5) splits responses into clauses at punctuation boundaries and ranks by mean projection delta. Useful for identifying which text spans (e.g., "By the way, [movie recommendation]") drive the largest activation divergence between variants.

---

## Benchmarking

Test capability preservation during ablation.

```bash
# Without steering (baseline)
python analysis/benchmark/evaluate.py \
    --experiment {experiment} \
    --benchmark hellaswag

# With ablation (negative steering)
python analysis/benchmark/evaluate.py \
    --experiment {experiment} \
    --benchmark hellaswag \
    --steer {category}/{trait} --coef -1.0
```

**Outputs:** `benchmark/{benchmark}.json`

**Supported:** `hellaswag`, `arc_easy`

---

## Vector Interpretation

Understand what a vector represents via logit lens.

```bash
python analysis/vectors/logit_lens.py \
    --experiment {experiment} \
    --trait {category}/{trait} \
    --filter-common  # Show interpretable tokens
```

**Outputs:** `extraction/{trait}/{variant}/logit_lens.json`

---

## Visualization

Interactive dashboard for exploring results.

```bash
python visualization/serve.py
# Visit http://localhost:8000/
```

**Views:**
| View | Purpose |
|------|---------|
| Trait Extraction | Best vectors, layer×method heatmaps |
| Steering Sweep | Coefficient search results, response browser |
| Trait Dynamics | Per-token trajectory over layers |
| Model Comparison | Effect size between variants |
| Live Chat | Real-time monitoring with steering |

**Details:** [visualization/README.md](../visualization/README.md)

---

## R2 Sync

Sync experiment data to/from cloud storage.

```bash
./utils/r2_push.sh              # Fast: new files only
./utils/r2_push.sh --full       # Propagates deletions

./utils/r2_pull.sh              # Pull from R2
```

**Excluded (large, regenerable):**
- `**/activations/**` — Extraction activations
- `**/inference/*/raw/**` — Raw inference activations

**Details:** [docs/r2_sync.md](r2_sync.md)

---

## Model Server (Optional)

Keep model loaded between script runs.

```bash
# Terminal 1: Start server
python other/server/app.py --port 8765 --model {model}

# Terminal 2: Scripts auto-detect server
python inference/capture_raw_activations.py ...  # Uses server automatically
```

Scripts fall back to local loading if server isn't running. Use `--no-server` to force local.

---

## Common Patterns

### Multi-trait processing
```bash
# All traits in experiment
python extraction/run_pipeline.py --experiment {experiment}

# Specific traits
python extraction/run_pipeline.py --experiment {experiment} \
    --traits cat1/trait1,cat2/trait2
```

### Resume from crash
Most scripts support `--skip-existing` to resume.

### 70B+ models
Use `--load-in-8bit` for quantization.

### Best vector selection
`utils/vectors.py:get_best_vector()` auto-selects using steering results as ground truth.

---

## Workflow Dependencies

```
Extraction (creates vectors)
    ↓
    ├── Steering (validates vectors)
    ├── Inference (monitors with vectors)
    │       ↓
    │       └── Model Comparison (compares variants)
    └── Benchmarking (tests ablation)
```

**Key principle:** Extract once, apply everywhere. Extraction creates vectors; all other workflows consume them.
