# Experiment: Quantization Sensitivity of Trait Vectors

## Goal

Test whether quantization (NF4 4-bit, INT8, GPTQ 4-bit) degrades trait vector extraction and steering quality. First systematic study of quantization's effect on representation engineering.

## Hypotheses

1. Quantization introduces activation-space noise that changes extracted vector directions (cosine similarity FP16 vs quantized < 1.0)
2. Steering effectiveness degrades under quantization (delta drops, coherence drops at matched coefficients)
3. Probe extraction degrades less than mean_diff (row normalization suppresses quantization noise, paralleling massive-dims finding)
4. Model families with severe massive dims (Gemma-3-4B, 1500x) are more sensitive than mild (Llama-3.1-8B, 100x)
5. Larger models (70B) are more robust than smaller (2B) due to redundancy
6. INT8 degrades less than NF4 (higher precision, LLM.int8() mixed-precision outlier handling)
7. Cross-precision transfer works: FP16-extracted vectors still steer quantized models effectively
8. NF4 and GPTQ may differ — GPTQ's per-group quantization may preserve or distort trait-relevant channels differently

## Success Criteria

- [ ] FP16 baselines work: PV sycophancy produces measurable steering delta on all 7 models (threshold TBD after Phase 1 — prior evidence only exists for Llama-8B at +73.2)
- [ ] Vector cosine similarity measured for all precision x model x trait combinations
- [ ] Steering delta degradation quantified (% drop from FP16 baseline) for all cells
- [ ] Method robustness ranking: probe vs mean_diff degradation compared across all conditions
- [ ] Model family robustness ranking established
- [ ] Size scaling characterized (70B vs 2B)
- [ ] Cross-precision transfer tested (FP16 vectors -> quantized model steering)
- [ ] Capability preservation confirmed via HellaSwag across all model x precision cells
- [ ] Logit difference on CAA A/B questions confirms steering behavior changes under quantization

## Deliverables

### Findings write-up
- `docs/viz_findings/quantization-sensitivity.md` — main findings page for the visualization dashboard

### New codebase tools (permanent additions)
- `analysis/steering/logit_difference.py` — CAA-style A/B logit difference evaluation
- `analysis/benchmark/evaluate.py` — MMLU and TruthfulQA added to BENCHMARKS dict
- `analysis/steering/evaluate.py` — `--vector-experiment` flag for cross-experiment vector loading

### Experiment outputs
- `experiments/quant-sensitivity/results/vector_comparison.json` — cosine sim matrix (all models x traits x methods x precisions)
- `experiments/quant-sensitivity/results/steering_comparison.json` — steering delta matrix
- `experiments/quant-sensitivity/results/benchmark_comparison.json` — HellaSwag/MMLU/TruthfulQA comparison
- `experiments/quant-sensitivity/results/robustness_ranking.json` — method + model rankings

### Analysis scripts (experiment-specific)
- `experiments/quant-sensitivity/scripts/compare_vectors.py` — vector cosine similarity analysis
- `experiments/quant-sensitivity/scripts/compare_steering.py` — steering delta comparison
- `experiments/quant-sensitivity/scripts/analyze_robustness.py` — method/model robustness ranking

## Experiment Structure

```
Phase 0: Setup (infrastructure, datasets, code)
Phase 1: FP16 baselines (extraction + full eval suite)
Phase 2: BitsAndBytes quantization (NF4 + INT8 extraction + eval)
Phase 3: GPTQ extraction + eval (optional, per-model)
Phase 4: Cross-precision analysis (vector comparison, steering comparison)
Phase 5: Cross-precision steering (FP16 vectors -> quantized models)
Phase 6: Deep dives (conditional on Phase 4 results)
```

---

## Models (7 across 5 families)

| Model | Family | Params | Layers | Hidden | Why | Config exists |
|-------|--------|--------|--------|--------|-----|---------------|
| Gemma-2-2B-IT | Gemma | 2B | 26 | 2304 | Small, mild massive dims (60x), existing infra | Yes |
| Gemma-3-4B-IT | Gemma | 4B | 34 | 2560 | Severe massive dims (1500x), hardest test case | Yes |
| Mistral-7B-Instruct-v0.2 | Mistral | 7B | 32 | 4096 | Different family, cross-model baseline | Need config |
| Qwen2.5-7B-Instruct | Qwen | 7B | 28 | 3584 | Different family, growing ecosystem | Yes |
| Llama-3.1-8B-Instruct | Llama | 8B | 32 | 4096 | Well-studied, existing infra | Yes |
| Llama-3.3-70B-Instruct | Llama | 70B | 80 | 8192 | Size scaling, production-realistic | Yes |
| OLMo-2-7B-Instruct | OLMo | 7B | 32 | 4096 | Fully open (weights + data + code), 5th family | Need config |

## GPTQ Model Sources (pre-quantized, 4-bit)

| Model | GPTQ ID | Quantizer | Notes |
|-------|---------|-----------|-------|
| Gemma-2-2B-IT | **None** | — | No IT-variant GPTQ exists. Skip GPTQ for this model. |
| Gemma-3-4B-IT | `ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g` | ISTA-DASLab | 4-bit, group_size=128, symmetric |
| Mistral-7B-Instruct-v0.2 | `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` | TheBloke | Branch: `gptq-4bit-128g-actorder_True` |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | Qwen (official) | 4-bit, group_size=128, desc_act=true |
| Llama-3.1-8B-Instruct | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` | hugging-quants | 4-bit, group_size=128 |
| Llama-3.3-70B-Instruct | `shuyuej/Llama-3.3-70B-Instruct-GPTQ` | shuyuej | 4-bit, group_size=128 |
| OLMo-2-7B-Instruct | **None** | — | No GPTQ exists. Skip GPTQ for this model. |

## Quantization Conditions (4 conditions)

| Condition | Method | Implementation | Notes |
|-----------|--------|----------------|-------|
| FP16/BF16 | Baseline | Default `load_model()` | Standard precision |
| NF4 | BitsAndBytes 4-bit | `--load-in-4bit`, compute_dtype=bf16 | Weights 4-bit, activations bf16 |
| INT8 | BitsAndBytes 8-bit (LLM.int8()) | `--load-in-8bit` | Mixed-precision outlier handling |
| GPTQ | GPTQ 4-bit | Pre-quantized model, group_size=128 | Per-group quantization. Optional, per-model. |

**Core conditions**: FP16, NF4, INT8 — use existing `--load-in-4bit` / `--load-in-8bit` infrastructure. Zero new loading code needed.

**Optional condition**: GPTQ — requires separate HuggingFace checkpoints per model. Skip per-model if no high-quality GPTQ version exists (currently missing for Gemma-2-2B-IT). May need `pip install optimum` or `pip install auto-gptq`.

[maybe add: AWQ — activation-aware weight quantization, theoretically best for preserving salient channels]

## Extraction Methodologies

### 1. Persona Vectors (PV) — System Prompt Extraction
- **Source**: Shao et al. 2024 (arxiv 2507.21509)
- **How**: System prompts ("You are an evil AI" vs "You are a helpful AI") on instruct model
- **Position**: `response[:]` (all response tokens)
- **Method**: mean_diff, probe
- **Traits**: evil, sycophancy, hallucination
- **Datasets**: `datasets/traits/pv_instruction/{evil,sycophancy,hallucination}` (exist, .jsonl format)

### 2. CAA — Contrastive Activation Addition
- **Source**: Rimsky et al. 2024 (arxiv 2312.06681, ACL 2024)
- **How**: Contrastive statement pairs, mean diff in residual stream
- **Position**: `prompt[-1]` (last prompt token)
- **Method**: mean_diff, probe
- **Traits**: sycophancy, refusal [maybe: corrigibility, power-seeking, survival-instinct]
- **Datasets**: Need to download from github.com/nrimsky/CAA, convert to our format
- **Eval**: A/B multiple-choice questions (50 per trait) for logit difference evaluation

### 3. Arditi — Refusal Direction
- **Source**: Arditi et al. 2024 (arxiv 2406.11717)
- **How**: Harmful vs harmless prompts on instruct model, mean diff at last prompt token
- **Position**: `prompt[-1]`
- **Method**: mean_diff, probe
- **Traits**: refusal
- **Datasets**: `datasets/traits/arditi/refusal` (exist) + `datasets/inference/arditi_holdout/` (exist)

[maybe add: RepE — Representation Engineering (Zou et al. 2023) — PCA-based extraction]
[maybe add: CAFT eval framework (Casademunt et al. 2025, arxiv 2507.16795) — their dual-metric approach (task preservation + steering preservation) maps to our cross-precision comparison. Not an extraction method (training-time intervention), but their OOD generalization framing is relevant: vectors extracted at one precision applied at another are a distribution shift in activation space.]

## Traits

| Trait | PV | CAA | Arditi |
|-------|----|----|--------|
| Sycophancy | Yes | Yes (1000 pairs) | No |
| Hallucination | Yes | No | No |
| Evil/toxicity | Yes | No | No |
| Refusal | No | Yes (408 pairs) | Yes |

[maybe add: power-seeking, corrigibility, survival-instinct from CAA]

## Evaluation Suite

### Per-trait evaluations (run for each extracted vector)

| Eval | Method | Script | Scope |
|------|--------|--------|-------|
| **LLM Judge Steering** | Generate steered responses, score trait expression + coherence (0-100) via gpt-4.1-mini | `analysis/steering/evaluate.py` | All traits x methodologies |
| **Logit Difference** | Compute logit(A) - logit(B) on CAA A/B questions under steering | `analysis/steering/logit_difference.py` (NEW) | CAA traits only |

### Model-level evaluations (run once per model x precision)

| Eval | Method | Script | Scope |
|------|--------|--------|-------|
| **HellaSwag** | Length-normalized log prob on 4-choice completion | `analysis/benchmark/evaluate.py --benchmark hellaswag` | All 7 models x 4 precisions |
| **MMLU** (subset) | Length-normalized log prob on 4-choice QA | `analysis/benchmark/evaluate.py --benchmark mmlu` (NEW) | 3 models (Gemma-2-2B, Llama-8B, Llama-70B) x 4 precisions |
| **TruthfulQA** | Truthful + informative scoring | `analysis/benchmark/evaluate.py --benchmark truthfulqa` (NEW) | 3 models x 4 precisions, validation for sycophancy |

### Vector-level analysis (post-extraction, no model needed)

| Eval | Method | Script | Scope |
|------|--------|--------|-------|
| **Vector cosine similarity** | cos(v_fp16, v_quant) per layer | `experiments/quant-sensitivity/scripts/compare_vectors.py` (NEW) | All extraction pairs |
| **Massive dim energy** | % of vector norm in top-k massive dims | Same script | All extractions |

### Metric thresholds

| Metric | Robust | Degraded | Broken |
|--------|--------|----------|--------|
| Vector cosine sim | > 0.95 | 0.80-0.95 | < 0.80 |
| Steering delta degradation | < 10% | 10-30% | > 30% |
| HellaSwag accuracy drop | < 1% | 1-3% | > 3% |
| Logit difference correlation | > 0.90 | 0.70-0.90 | < 0.70 |

## Config

Each model x precision gets its own experiment directory:

```
experiments/quant-sensitivity/gemma2-2b/         # FP16
experiments/quant-sensitivity/gemma2-2b-nf4/     # NF4
experiments/quant-sensitivity/gemma2-2b-int8/    # INT8
experiments/quant-sensitivity/gemma2-2b-gptq/    # GPTQ (if available)
```

Each config.json has a single `instruct` variant:

```json
{
  "defaults": {"extraction": "instruct", "application": "instruct"},
  "model_variants": {
    "instruct": {"model": "<instruct-model-id>"}
  }
}
```

For NF4/INT8 directories, the config points to the same model — `--load-in-4bit` or `--load-in-8bit` is passed via CLI. For GPTQ directories, the config points to the pre-quantized model ID.

**Why separate directories instead of model variants**: The trait spec parser in `analysis/steering/evaluate.py` splits on `/` and handles exactly 2 or 3 parts. Nested experiment names like `quant-sensitivity/gemma2-2b` already use one slash, so we can't add a 4th level for cross-experiment vector loading. Using flat `-nf4`/`-int8`/`-gptq` suffixes avoids this limitation.

**Note**: The config schema doesn't natively support `load_in_4bit` as a field — quantization is controlled via CLI flags.

## Prerequisites

```bash
# Verify PV datasets exist
ls datasets/traits/pv_instruction/sycophancy/positive.jsonl
ls datasets/traits/pv_instruction/evil/positive.jsonl
ls datasets/traits/pv_instruction/hallucination/positive.jsonl

# Verify Arditi datasets exist
ls datasets/traits/arditi/refusal/positive.txt

# Verify model configs exist
ls config/models/gemma-2-2b-it.yaml
ls config/models/gemma-3-4b-it.yaml
ls config/models/llama-3.1-8b-instruct.yaml
ls config/models/llama-3.3-70b-instruct.yaml
ls config/models/qwen2.5-7b-instruct.yaml
# Mistral-7B-Instruct config needs creation (base config exists at config/models/mistral-7b-v0.1.yaml)
# OLMo-2-7B-Instruct config needs creation

# Verify quantization infrastructure
python -c "from utils.model import load_model; print('model loading OK')"
python -c "import bitsandbytes; print('bitsandbytes OK')"
```

## Known Confounds

1. **Generation differences**: Quantized models generate different text, so extraction captures activations from different responses. This tests the full pipeline (feature, not bug). For controlled comparison, use `--replay-responses` to prefill quantized model with FP16-generated text.
2. **NF4 compute dtype**: BnB NF4 weights are 4-bit but activations emerge in bfloat16. The question is whether bf16 activations from quantized weights differ from bf16 activations from unquantized weights.
3. **NF4 structured noise**: NF4 uses block-wise quantization, so activation distortion is correlated within blocks — not independent Gaussian noise. Cosine similarity may be misleadingly high if trait signal concentrates in dimensions that happen to be well-preserved.
4. **GPTQ is a different model**: Pre-quantized GPTQ models are different HuggingFace checkpoints. Any differences could be from the quantization process, not just precision.
5. **Reproducibility**: Use `--temperature 0.0` (default) for deterministic generation. Even so, quantized models may produce different greedy outputs — this is expected.
6. **70B multi-GPU**: The 70B model needs `device_map="auto"` which splits layers across GPUs. Hooks should still fire but cross-device tensor transfers could introduce subtle numerical differences. Verify hook output shapes and dtypes on first 70B run.

---

## Autonomy Protocol

Guidelines for autonomous execution via `/r:run-experiment`. Each phase has specific escalation conditions.

### General Rules

**ESCALATE** (stop and call user):
- Model loading fails with an error not in the "If Stuck" section
- Results seem implausible (e.g., FP16 steering delta < +5 on a well-studied model)
- A phase gate check fails and the fix isn't obvious
- Any operation would modify files outside `experiments/quant-sensitivity/`, `datasets/traits/caa/`, or the specific codebase files listed in Phase 0

**INVESTIGATE** (dig deeper before continuing):
- Steering delta drops >50% from FP16 baseline — check coefficient range, inspect responses, try wider search
- Extraction AUROC < 0.55 on any trait — check dataset, activations, position
- Cosine similarity < 0.70 for any model-trait pair — check for extraction failures, zero vectors
- HellaSwag drops >5% from FP16 — likely model loading issue, not quantization

**DISCARD** (throw away and retry):
- If a model produces coherence < 30% across all coefficients — model is generating garbage, likely OOM or loading issue
- If extraction produces all-zero vectors — retry with different batch size or check GPU memory
- If benchmark scores are exactly 0.0 or 1.0 — something is broken in scoring

**SEARCH** (look online):
- If a GPTQ model fails to load, search HuggingFace for alternative GPTQ quantizations of that model
- If a new model (OLMo, Qwen3) is being considered, search for its architecture details and GPTQ availability
- If an unexpected error occurs in auto-gptq/optimum, search for known issues and version compatibility
- Clone repos when needed (e.g., github.com/nrimsky/CAA for datasets)

### Per-Phase Escalation

Phase-specific conditions are documented in each phase's checkpoint section below.

---

## Phase 0: Setup

### Step 0a: Fix bnb_4bit_compute_dtype bug

**Purpose**: `load_model_with_lora()` doesn't set `bnb_4bit_compute_dtype=dtype` when `load_in_4bit=True`, unlike `load_model()`. This means extraction (which uses `load_model_with_lora()`) and benchmarking (which uses `load_model()`) would compute at different dtypes — a silent confound.

**File**: `utils/model.py:283-285`

**Fix**: Add `bnb_4bit_compute_dtype=dtype` to the BitsAndBytesConfig in `load_model_with_lora()`, matching `load_model()` at line 222-224.

**Verify**:
```bash
python -c "
from utils.model import load_model_with_lora
import inspect
src = inspect.getsource(load_model_with_lora)
assert 'bnb_4bit_compute_dtype' in src, 'Bug not fixed'
print('Bug fixed')
"
```

### Step 0b: Add GPTQ model loading support

**Purpose**: GPTQ models are pre-quantized and load via standard `AutoModelForCausalLM.from_pretrained()` — transformers auto-detects the quantization config. May need `pip install optimum` or `pip install auto-gptq`.

**File**: `utils/model.py` — extend `load_model()` to handle GPTQ models gracefully. Mostly just ensuring the model loads without errors when a GPTQ checkpoint is specified in config.

**Read first**: `utils/model.py` — understand `load_model()` flow

**Verify**:
```bash
# After finding GPTQ model IDs
python -c "from utils.model import load_model; m, t = load_model('ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g'); print('GPTQ loading OK')"
```

**ESCALATE**: If GPTQ loading fails on all models. **SEARCH**: If auto-gptq import fails, search for correct pip install command and version compatibility.

### Step 0c: Build logit difference evaluation

**Purpose**: CAA-style evaluation using A/B multiple-choice questions. Measures logit(answer_A) - logit(answer_B) to quantify behavioral propensity without generation.

**File**: `analysis/steering/logit_difference.py` (NEW)

**Read first**:
- `analysis/benchmark/evaluate.py` — `score_completions()` (lines 66-110) for existing length-normalized log prob scoring
- `analysis/steering/evaluate.py` — for SteeringHook integration pattern

**Interface**:
```bash
python analysis/steering/logit_difference.py \
    --experiment quant-sensitivity/llama-8b \
    --dataset datasets/traits/caa/sycophancy/test_ab.json \
    --model-variant instruct \
    --steer caa/sycophancy \
    --layer 14 --coef 6.0 \
    --load-in-4bit  # optional
```

**Input format** (CAA A/B questions):
```json
[
  {
    "question": "...",
    "answer_matching_behavior": "A) ...",
    "answer_not_matching_behavior": "B) ..."
  }
]
```

**Output**: Mean logit difference, per-question scores, steerability score (slope across coefficients).

**Verify**:
```bash
python analysis/steering/logit_difference.py \
    --experiment quant-sensitivity/gemma2-2b \
    --dataset datasets/traits/caa/sycophancy/test_ab.json \
    --model-variant instruct \
    --dry-run
```

### Step 0d: Add MMLU and TruthfulQA benchmarks

**Purpose**: Capability preservation metrics. MMLU for general knowledge, TruthfulQA specifically for sycophancy vector validation (CAA showed sycophancy vectors improve TruthfulQA).

**File**: `analysis/benchmark/evaluate.py` — add `evaluate_mmlu()` and `evaluate_truthfulqa()` to `BENCHMARKS` dict (line 317), following pattern of `evaluate_hellaswag()`.

**Read first**: `analysis/benchmark/evaluate.py` — understand `score_completions()` interface, dataset loading pattern, `BENCHMARKS` dict.

**HuggingFace datasets**:
- MMLU: `cais/mmlu` (or `lukaemon/mmlu`) — use 10 questions per subject for subset
- TruthfulQA: `truthfulqa/truthful_qa` — multiple-choice variant

**Verify**:
```bash
# List available benchmarks
python analysis/benchmark/evaluate.py --help | grep -A5 benchmark

# Run MMLU subset on smallest model
python analysis/benchmark/evaluate.py \
    --experiment quant-sensitivity/gemma2-2b \
    --benchmark mmlu \
    --limit 10

# Run TruthfulQA
python analysis/benchmark/evaluate.py \
    --experiment quant-sensitivity/gemma2-2b \
    --benchmark truthfulqa \
    --limit 20
```

### Step 0e: Download and convert CAA datasets

**Purpose**: Rimsky et al.'s contrastive pairs for sycophancy and refusal. Includes A/B test questions for logit difference eval.

**Source**: https://github.com/nrimsky/CAA -> `/datasets/` directory

**Commands**:
```bash
# Clone CAA repo
git clone https://github.com/nrimsky/CAA /tmp/caa-repo

# Convert to our format (write conversion script or do manually)
# Extract contrastive pairs -> datasets/traits/caa/{trait}/positive.txt, negative.txt
# Extract A/B test questions -> datasets/traits/caa/{trait}/test_ab.json (50 per trait)
# Create definition.txt and steering.json for each trait
```

**Verify**:
```bash
ls datasets/traits/caa/sycophancy/positive.txt
ls datasets/traits/caa/sycophancy/test_ab.json
wc -l datasets/traits/caa/sycophancy/positive.txt  # Should be ~500
python -c "import json; d=json.load(open('datasets/traits/caa/sycophancy/test_ab.json')); print(f'{len(d)} test questions')"  # 50
```

**SEARCH**: If CAA repo structure has changed, inspect the repo and adapt conversion.

### Step 0f: Add `--vector-experiment` flag to steering evaluate.py

**Purpose**: Enable cross-precision steering (Phase 5). The trait spec parser can't handle nested experiment names with slashes for cross-experiment vector loading.

**File**: `analysis/steering/evaluate.py`

**Fix**: Add `--vector-experiment` argument. When provided, override the vector source experiment for all traits. Modify the `parsed_traits` loop (lines 810-819) to use `args.vector_experiment or args.experiment` as the default experiment.

```python
parser.add_argument("--vector-experiment",
    help="Experiment to load vectors from (defaults to --experiment)")
```

**Verify**:
```bash
python analysis/steering/evaluate.py --help | grep vector-experiment
```

### Step 0g: Create experiment configs

**Purpose**: Per-model x per-precision experiment directories with config.json.

Create directories and configs for all cells:

```bash
# FP16 configs (7 models)
experiments/quant-sensitivity/gemma2-2b/config.json        # google/gemma-2-2b-it
experiments/quant-sensitivity/gemma3-4b/config.json        # google/gemma-3-4b-it
experiments/quant-sensitivity/mistral-7b/config.json       # mistralai/Mistral-7B-Instruct-v0.2
experiments/quant-sensitivity/qwen-7b/config.json          # Qwen/Qwen2.5-7B-Instruct
experiments/quant-sensitivity/llama-8b/config.json         # meta-llama/Llama-3.1-8B-Instruct
experiments/quant-sensitivity/llama-70b/config.json        # meta-llama/Llama-3.3-70B-Instruct
experiments/quant-sensitivity/olmo-7b/config.json          # allenai/OLMo-2-1124-7B-Instruct

# NF4 configs (7 models — same model IDs, --load-in-4bit via CLI)
experiments/quant-sensitivity/gemma2-2b-nf4/config.json
experiments/quant-sensitivity/gemma3-4b-nf4/config.json
experiments/quant-sensitivity/mistral-7b-nf4/config.json
experiments/quant-sensitivity/qwen-7b-nf4/config.json
experiments/quant-sensitivity/llama-8b-nf4/config.json
experiments/quant-sensitivity/llama-70b-nf4/config.json
experiments/quant-sensitivity/olmo-7b-nf4/config.json

# INT8 configs (7 models — same model IDs, --load-in-8bit via CLI)
experiments/quant-sensitivity/gemma2-2b-int8/config.json
experiments/quant-sensitivity/gemma3-4b-int8/config.json
experiments/quant-sensitivity/mistral-7b-int8/config.json
experiments/quant-sensitivity/qwen-7b-int8/config.json
experiments/quant-sensitivity/llama-8b-int8/config.json
experiments/quant-sensitivity/llama-70b-int8/config.json
experiments/quant-sensitivity/olmo-7b-int8/config.json

# GPTQ configs (5 models — Gemma-2-2B and OLMo-2 have no GPTQ)
experiments/quant-sensitivity/gemma3-4b-gptq/config.json   # ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g
experiments/quant-sensitivity/mistral-7b-gptq/config.json  # TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
experiments/quant-sensitivity/qwen-7b-gptq/config.json     # Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
experiments/quant-sensitivity/llama-8b-gptq/config.json    # hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
experiments/quant-sensitivity/llama-70b-gptq/config.json   # shuyuej/Llama-3.3-70B-Instruct-GPTQ
```

Total: 26 experiment directories (7 FP16 + 7 NF4 + 7 INT8 + 5 GPTQ).

**Verify**:
```bash
for d in experiments/quant-sensitivity/*/; do
    python -c "import json; json.load(open('${d}config.json')); print('OK: $d')"
done
```

### Step 0h: Create model architecture configs

**Mistral**: `config/models/mistral-7b-instruct-v0.2.yaml` — copy from `mistral-7b-v0.1.yaml`, update model ID and any architecture differences.

**OLMo-2**: `config/models/olmo-2-7b-instruct.yaml` — 32 layers, hidden dim 4096, 32 attention heads. Create from scratch or reference HuggingFace config.

### Step 0i: Create vector comparison script

**File**: `experiments/quant-sensitivity/scripts/compare_vectors.py` (NEW)

**Purpose**: Compare vectors extracted at different precisions. Core analysis tool.

**Outputs**:
- Per-layer cosine similarity between experiment A and experiment B vectors
- Per-layer norm comparison
- Massive dim energy (% of norm in top-k dims)
- Summary table

**Interface**:
```bash
python experiments/quant-sensitivity/scripts/compare_vectors.py \
    --experiment-a quant-sensitivity/llama-8b \
    --experiment-b quant-sensitivity/llama-8b-nf4 \
    --trait pv_instruction/sycophancy \
    --method mean_diff
```

### Checkpoint: Phase 0 Gate

- [ ] `bnb_4bit_compute_dtype` bug fixed in `utils/model.py`
- [ ] GPTQ loading works on at least one model — or explicitly deferred
- [ ] `--vector-experiment` flag added to `analysis/steering/evaluate.py`
- [ ] `analysis/steering/logit_difference.py` runs without error (smoke test)
- [ ] `analysis/benchmark/evaluate.py` supports mmlu, truthfulqa
- [ ] CAA datasets downloaded and converted: `datasets/traits/caa/sycophancy/`, `datasets/traits/caa/refusal/`
- [ ] 23 experiment configs created and parse correctly
- [ ] `compare_vectors.py` runs without error
- **ESCALATE** if any prerequisite missing and fix isn't obvious

---

## Phase 1: FP16 Baselines (~36h across all models)

Extract at full precision and run full eval suite. This establishes the baseline every quantized comparison is relative to.

### Extraction matrix per model:

| Methodology | Trait | Position | Methods |
|-------------|-------|----------|---------|
| PV | sycophancy | response[:] | mean_diff, probe |
| PV | evil | response[:] | mean_diff, probe |
| PV | hallucination | response[:] | mean_diff, probe |
| CAA | sycophancy | prompt[-1] | mean_diff, probe |
| CAA | refusal | prompt[-1] | mean_diff, probe |
| Arditi | refusal | prompt[-1] | mean_diff, probe |

= 6 trait-configs x 2 methods = 12 vector sets per model, x 7 models = 84 total

### Step 1a: Extract vectors (per model)

Run for each model x methodology x trait:

```bash
# PV extraction — all 3 traits
for trait in sycophancy evil hallucination; do
    python extraction/run_pipeline.py \
        --experiment quant-sensitivity/{model_name} \
        --traits pv_instruction/$trait \
        --model-variant instruct \
        --position "response[:]" \
        --methods mean_diff,probe \
        --max-new-tokens 256 \
        --rollouts 2
done

# CAA extraction — sycophancy + refusal
for trait in sycophancy refusal; do
    python extraction/run_pipeline.py \
        --experiment quant-sensitivity/{model_name} \
        --traits caa/$trait \
        --model-variant instruct \
        --position "prompt[-1]" \
        --methods mean_diff,probe
done

# Arditi extraction — refusal
python extraction/run_pipeline.py \
    --experiment quant-sensitivity/{model_name} \
    --traits arditi/refusal \
    --model-variant instruct \
    --position "prompt[-1]" \
    --methods mean_diff,probe
```

**Expected output per trait**:
```
experiments/quant-sensitivity/{model_name}/extraction/{methodology}/{trait}/instruct/
├── responses/pos.json, neg.json
├── activations/{position}/residual/
│   ├── train_all_layers.pt, val_all_layers.pt
│   └── metadata.json
└── vectors/{position}/residual/{mean_diff,probe}/layer*.pt
```

**Verify**:
```bash
# Check vector count matches number of layers
for trait in pv_instruction/sycophancy pv_instruction/evil pv_instruction/hallucination caa/sycophancy caa/refusal arditi/refusal; do
    echo "$trait:"
    ls experiments/quant-sensitivity/{model_name}/extraction/$trait/instruct/vectors/*/residual/mean_diff/ 2>/dev/null | wc -l
done
```

### Step 1b: Steering evaluation (per model)

```bash
python analysis/steering/evaluate.py \
    --experiment quant-sensitivity/{model_name} \
    --traits pv_instruction/sycophancy,pv_instruction/evil,pv_instruction/hallucination,caa/sycophancy,caa/refusal,arditi/refusal \
    --model-variant instruct \
    --search-steps 5 \
    --save-responses best
```

**Verify**:
```bash
ls experiments/quant-sensitivity/{model_name}/steering/*/instruct/*/steering/results.jsonl | wc -l
# Should be 6 (one per trait)
```

### Step 1c: Logit difference evaluation (CAA traits only, per model)

```bash
for trait in sycophancy refusal; do
    python analysis/steering/logit_difference.py \
        --experiment quant-sensitivity/{model_name} \
        --dataset datasets/traits/caa/$trait/test_ab.json \
        --model-variant instruct \
        --steer caa/$trait
done
```

### Step 1d: Capability benchmarks (per model, once)

```bash
# HellaSwag — all 6 models
python analysis/benchmark/evaluate.py \
    --experiment quant-sensitivity/{model_name} \
    --benchmark hellaswag \
    --limit 0  # full eval

# MMLU subset — Gemma-2-2B, Llama-8B, Llama-70B only
python analysis/benchmark/evaluate.py \
    --experiment quant-sensitivity/{model_name} \
    --benchmark mmlu \
    --limit 10  # 10 per subject

# TruthfulQA — Gemma-2-2B, Llama-8B, Llama-70B only
python analysis/benchmark/evaluate.py \
    --experiment quant-sensitivity/{model_name} \
    --benchmark truthfulqa \
    --limit 0
```

**Verify**:
```bash
ls experiments/quant-sensitivity/{model_name}/benchmark/
# Should have hellaswag.json, and for 3 models: mmlu.json, truthfulqa.json
```

### Checkpoint 1: FP16 Baseline Gate

- [ ] PV sycophancy produces non-trivial steering delta on all 7 models (record actual values — prior evidence only on Llama-8B at +73.2)
- [ ] Arditi refusal ablation works on at least 4/7 models (prior evidence only on Gemma-2-2B)
- [ ] All extractions produce validation AUROC > 0.6
- [ ] HellaSwag baseline recorded for all 7 models
- [ ] Logit difference baseline recorded for CAA sycophancy and refusal
- **After this checkpoint**: Set model-specific thresholds for quantization comparison based on actual FP16 results
- **ESCALATE** if majority of baselines produce delta < +5 — dataset/config issues need investigation
- **INVESTIGATE** if any single model produces dramatically different results from others

---

## Phase 2: BitsAndBytes Quantization — NF4 + INT8 (~48h)

Repeat Phase 1 extraction and eval at both NF4 (4-bit) and INT8 (8-bit) precision for all 7 models. NF4 and INT8 runs for the same model can run in parallel on different GPUs.

### Step 2a: Extract vectors at NF4

Same commands as Step 1a but with `--load-in-4bit` and targeting `-nf4` experiment directories:

```bash
# PV extraction at NF4
for trait in sycophancy evil hallucination; do
    python extraction/run_pipeline.py \
        --experiment quant-sensitivity/{model_name}-nf4 \
        --traits pv_instruction/$trait \
        --model-variant instruct \
        --position "response[:]" \
        --methods mean_diff,probe \
        --max-new-tokens 256 \
        --rollouts 2 \
        --load-in-4bit
done

# CAA + Arditi at NF4 — same pattern with --load-in-4bit
```

### Step 2b: Extract vectors at INT8

Same commands but with `--load-in-8bit` and targeting `-int8` directories:

```bash
# PV extraction at INT8
for trait in sycophancy evil hallucination; do
    python extraction/run_pipeline.py \
        --experiment quant-sensitivity/{model_name}-int8 \
        --traits pv_instruction/$trait \
        --model-variant instruct \
        --position "response[:]" \
        --methods mean_diff,probe \
        --max-new-tokens 256 \
        --rollouts 2 \
        --load-in-8bit
done

# CAA + Arditi at INT8 — same pattern with --load-in-8bit
```

### Step 2c: Steering evaluation at NF4 and INT8

```bash
# NF4 steering
python analysis/steering/evaluate.py \
    --experiment quant-sensitivity/{model_name}-nf4 \
    --traits pv_instruction/sycophancy,pv_instruction/evil,pv_instruction/hallucination,caa/sycophancy,caa/refusal,arditi/refusal \
    --model-variant instruct \
    --search-steps 5 \
    --save-responses best \
    --load-in-4bit

# INT8 steering
python analysis/steering/evaluate.py \
    --experiment quant-sensitivity/{model_name}-int8 \
    --traits pv_instruction/sycophancy,pv_instruction/evil,pv_instruction/hallucination,caa/sycophancy,caa/refusal,arditi/refusal \
    --model-variant instruct \
    --search-steps 5 \
    --save-responses best \
    --load-in-8bit
```

### Step 2d: Logit difference at NF4 and INT8

Same as Step 1c with `--load-in-4bit` / `--load-in-8bit` targeting respective experiment dirs.

### Step 2e: Capability benchmarks at NF4 and INT8

```bash
# NF4 benchmarks
python analysis/benchmark/evaluate.py \
    --experiment quant-sensitivity/{model_name}-nf4 \
    --benchmark hellaswag \
    --load-in-4bit

# INT8 benchmarks
python analysis/benchmark/evaluate.py \
    --experiment quant-sensitivity/{model_name}-int8 \
    --benchmark hellaswag \
    --load-in-8bit
```

### Checkpoint 2: BnB Cosine Similarity Triage

Before proceeding to GPTQ, check vector similarity for NF4 and INT8:

```bash
# NF4 comparison
python experiments/quant-sensitivity/scripts/compare_vectors.py \
    --experiment-a quant-sensitivity/{model_name} \
    --experiment-b quant-sensitivity/{model_name}-nf4 \
    --trait pv_instruction/sycophancy \
    --method mean_diff

# INT8 comparison
python experiments/quant-sensitivity/scripts/compare_vectors.py \
    --experiment-a quant-sensitivity/{model_name} \
    --experiment-b quant-sensitivity/{model_name}-int8 \
    --trait pv_instruction/sycophancy \
    --method mean_diff
```

**Decision tree**:
- ALL cosine sims > 0.95 for both NF4 and INT8 -> quantization barely matters. Main result: "BnB quantization is safe for trait vectors"
- NF4 < 0.90 but INT8 > 0.95 -> expected! INT8's higher precision and outlier handling work
- ALL cosine sims < 0.80 -> quantization severely disrupts vectors, high-impact finding
- **INVESTIGATE** if INT8 is _worse_ than NF4 (unexpected — would indicate LLM.int8() mixed-precision routing interferes with trait-relevant dimensions)

---

## Phase 3: GPTQ Extraction + Eval (~28h) [OPTIONAL — per-model]

Repeat with GPTQ 4-bit models. Uses pre-quantized model IDs (different HuggingFace checkpoint per model).

**Prerequisite**: GPTQ model IDs found and loading verified for target models. Available GPTQ models:
- Gemma-2-2B-IT: **SKIP** (no GPTQ exists)
- Gemma-3-4B-IT: `ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g`
- Mistral-7B-Instruct-v0.2: `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` (branch: `gptq-4bit-128g-actorder_True`)
- Qwen2.5-7B-Instruct: `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`
- Llama-3.1-8B-Instruct: `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`
- Llama-3.3-70B-Instruct: `shuyuej/Llama-3.3-70B-Instruct-GPTQ`

If a GPTQ model fails to load, **SEARCH** HuggingFace for alternatives before skipping.

### Step 3a: Extract vectors at GPTQ

```bash
for trait in sycophancy evil hallucination; do
    python extraction/run_pipeline.py \
        --experiment quant-sensitivity/{model_name}-gptq \
        --traits pv_instruction/$trait \
        --model-variant instruct \
        --position "response[:]" \
        --methods mean_diff,probe \
        --max-new-tokens 256 \
        --rollouts 2
done
# + CAA and Arditi — same pattern (no --load-in-4bit needed, GPTQ loads at native precision)
```

### Steps 3b-3d: Steering, logit diff, benchmarks at GPTQ

Same as Phase 1 commands but targeting `-gptq` experiment directories. No `--load-in-4bit` flag — GPTQ models load at their native precision.

### Checkpoint 3: NF4 vs INT8 vs GPTQ Comparison

```bash
# NF4 vs GPTQ
python experiments/quant-sensitivity/scripts/compare_vectors.py \
    --experiment-a quant-sensitivity/{model_name}-nf4 \
    --experiment-b quant-sensitivity/{model_name}-gptq \
    --trait pv_instruction/sycophancy \
    --method mean_diff
```

- NF4 ~= INT8 ~= GPTQ -> quantization method doesn't matter much
- NF4 != GPTQ -> GPTQ's per-group quantization preserves trait-relevant channels differently
- INT8 best, NF4/GPTQ similar -> LLM.int8() outlier handling matters, 4-bit quantization is the bottleneck

---

## Phase 4: Cross-Precision Analysis (~6h)

Comprehensive comparison across all conditions. Analysis scripts, no model loading needed.

### Step 4a: Vector comparison matrix

For each model x trait x methodology:
- Cosine similarity: FP16 vs NF4, FP16 vs INT8, FP16 vs GPTQ, NF4 vs GPTQ
- Per-layer cosine similarity curve (all layers)
- Massive dim energy comparison

```bash
python experiments/quant-sensitivity/scripts/compare_vectors.py \
    --all-models \
    --all-traits \
    --output experiments/quant-sensitivity/results/vector_comparison.json
```

### Step 4b: Steering comparison matrix

For each model x trait x methodology:
- Delta degradation: (delta_quant - delta_fp16) / delta_fp16
- Coherence comparison at matched coefficient
- Optimal layer shift (does best layer change?)
- Coefficient scaling changes (does optimal coefficient change?)

```bash
python experiments/quant-sensitivity/scripts/compare_steering.py \
    --all-models \
    --output experiments/quant-sensitivity/results/steering_comparison.json
```

### Step 4c: Method robustness ranking

Compare probe vs mean_diff degradation across all conditions.
- **Hypothesis**: probe degrades less (row normalization suppresses quantization noise)

### Step 4d: Model robustness ranking

- Which model family degrades least?
- Does 70B degrade less than 2B? (size scaling)
- Does Gemma-3-4B (severe massive dims) degrade more?

### Step 4e: Capability benchmark comparison

- HellaSwag: FP16 vs NF4 vs INT8 vs GPTQ for all 7 models
- MMLU: same for 3 models
- TruthfulQA: same for 3 models + check sycophancy interaction

### Checkpoint 4: Main Results Gate

- [ ] Have cosine sim x model x trait x method x precision matrix
- [ ] Have steering delta comparison for FP16 vs NF4 vs INT8 vs GPTQ
- [ ] Have capability benchmark comparison
- [ ] Have logit difference comparison for CAA traits
- [ ] Can answer: "does quantization break trait vectors?" with data
- **DECIDE**: Is cross-precision steering (Phase 5) worth running?
  - If vectors are nearly identical (cos sim > 0.95), cross-precision steering will trivially work — skip Phase 5
  - If vectors differ significantly, Phase 5 tests the deployment scenario

---

## Phase 5: Cross-Precision Steering (~16h, conditional)

The deployment scenario: extract vectors at full precision, steer on quantized model (or vice versa).

### Step 5a: FP16 vectors -> NF4 model

```bash
python analysis/steering/evaluate.py \
    --experiment quant-sensitivity/{model_name}-nf4 \
    --vector-experiment quant-sensitivity/{model_name} \
    --traits pv_instruction/sycophancy,pv_instruction/evil,caa/sycophancy,arditi/refusal \
    --model-variant instruct \
    --search-steps 5 \
    --load-in-4bit
```

**Note**: `--vector-experiment` loads vectors from the FP16 experiment. `--experiment` + `--load-in-4bit` steers the quantized model.

### Step 5b: FP16 vectors -> INT8 model

```bash
python analysis/steering/evaluate.py \
    --experiment quant-sensitivity/{model_name}-int8 \
    --vector-experiment quant-sensitivity/{model_name} \
    --traits pv_instruction/sycophancy,pv_instruction/evil,caa/sycophancy,arditi/refusal \
    --model-variant instruct \
    --search-steps 5 \
    --load-in-8bit
```

### Step 5c: FP16 vectors -> GPTQ model

```bash
python analysis/steering/evaluate.py \
    --experiment quant-sensitivity/{model_name}-gptq \
    --vector-experiment quant-sensitivity/{model_name} \
    --traits pv_instruction/sycophancy,pv_instruction/evil,caa/sycophancy,arditi/refusal \
    --model-variant instruct \
    --search-steps 5
```

### Step 5d: Coefficient recalibration

If cross-precision steering degrades, test whether wider coefficient search recovers performance:

```bash
python analysis/steering/evaluate.py \
    --experiment quant-sensitivity/{model_name}-nf4 \
    --vector-experiment quant-sensitivity/{model_name} \
    --traits pv_instruction/sycophancy \
    --model-variant instruct \
    --search-steps 10 \
    --load-in-4bit
```

[maybe: NF4 vectors -> FP16 model — tests whether NF4-extracted vectors are valid on full-precision model]

---

## Phase 6: Deep Dives (~8h, conditional)

Run based on Phase 4 results.

### Step 6a: Massive dims x quantization interaction

**When**: Gemma-3-4B shows degradation (cos sim < 0.90)

```bash
# Run massive dim calibration on NF4, INT8, and GPTQ models
python analysis/massive_activations.py \
    --experiment quant-sensitivity/gemma3-4b-nf4 \
    --load-in-4bit

python analysis/massive_activations.py \
    --experiment quant-sensitivity/gemma3-4b-int8 \
    --load-in-8bit

python analysis/massive_activations.py \
    --experiment quant-sensitivity/gemma3-4b-gptq
```

Compare massive dim magnitudes and locations across precisions. Test if precleaning helps.

### Step 6b: Layer-specific degradation curve

**When**: Cosine sims vary significantly by layer

Plot cos_sim(FP16, quant) by layer for each model. Identify most sensitive layers. Cross-reference with optimal steering layers.

### Step 6c: Per-token monitoring fidelity

**When**: Steering works but monitoring might not

Capture activations during inference on FP16 and NF4, project onto same trait vectors, compare per-token projection trajectories (Pearson correlation).

[maybe add: Perturbation ratio analysis — does the coefficient scaling law (coef x vec_norm / act_norm ~= 1.0 for coherence cliff) hold under quantization?]

[maybe add: Ablation experiment — does Arditi-style all-layer ablation still work on quantized models?]

---

## Compute Estimate

| Phase | What | Time (est) |
|-------|------|-----------|
| 0 | Setup, code, downloads, configs | 4h |
| 1 | FP16 baselines (7 models x extraction + steering + benchmarks) | 36h |
| 2 | BnB quantization (NF4 + INT8, 7 models each) | 56h |
| 3 | GPTQ extraction + full eval (5 models, optional) | 24h |
| 4 | Analysis scripts + comparison | 6h |
| 5 | Cross-precision steering (conditional) | 16h |
| 6 | Deep dives (conditional) | 8h |
| **Total** | | **~155h sequential** |

**70B is slow**: The 70B model takes 3-5x longer per operation than 7-8B models. Consider running 70B last or in parallel on dedicated multi-GPU.

**Early stopping**: If Phase 2 cosine sims are all > 0.95, the story is "quantization doesn't matter" — skip Phase 5, make Phase 6 optional.

**70B FP16 baseline**: The 70B model doesn't fit in FP16 on single GPU. Use BF16 with `device_map="auto"` on 2x A100-80G (standard practice). This IS the baseline.

**Benchmark cost**: HellaSwag full ~= 15-30min/model, MMLU subset ~= 10-20min/model, TruthfulQA ~= 10-20min/model. Total benchmark overhead: ~14h across all cells.

## Parallelization

The codebase runs one model per script invocation. No built-in multi-model parallelism. Parallelize via GPU isolation with `CUDA_VISIBLE_DEVICES`.

### GPU requirements per model

| Model | FP16 VRAM | NF4 VRAM | INT8 VRAM | GPTQ VRAM |
|-------|-----------|----------|-----------|-----------|
| Gemma-2-2B | ~5 GB | ~2 GB | ~3 GB | ~2 GB |
| Gemma-3-4B | ~9 GB | ~4 GB | ~5 GB | ~4 GB |
| Mistral-7B | ~15 GB | ~6 GB | ~8 GB | ~6 GB |
| Qwen2.5-7B | ~15 GB | ~6 GB | ~8 GB | ~6 GB |
| Llama-3.1-8B | ~17 GB | ~6 GB | ~9 GB | ~6 GB |
| Llama-3.3-70B | ~140 GB (2+ GPUs) | ~35 GB | ~70 GB | ~35 GB |
| OLMo-2-7B | ~15 GB | ~6 GB | ~8 GB | N/A |

### Parallel execution strategy

**Within a phase** (e.g., Phase 1 FP16):
```bash
# 6 small models run in parallel on separate GPUs
CUDA_VISIBLE_DEVICES=0 python extraction/run_pipeline.py --experiment quant-sensitivity/gemma2-2b ... &
CUDA_VISIBLE_DEVICES=1 python extraction/run_pipeline.py --experiment quant-sensitivity/gemma3-4b ... &
CUDA_VISIBLE_DEVICES=2 python extraction/run_pipeline.py --experiment quant-sensitivity/mistral-7b ... &
CUDA_VISIBLE_DEVICES=3 python extraction/run_pipeline.py --experiment quant-sensitivity/qwen-7b ... &
CUDA_VISIBLE_DEVICES=4 python extraction/run_pipeline.py --experiment quant-sensitivity/llama-8b ... &
CUDA_VISIBLE_DEVICES=5 python extraction/run_pipeline.py --experiment quant-sensitivity/olmo-7b ... &
wait

# Then 70B on remaining GPUs
CUDA_VISIBLE_DEVICES=6,7 python extraction/run_pipeline.py --experiment quant-sensitivity/llama-70b ... &
wait
```

**Across phases** (Phase 2 NF4 and INT8 in parallel):
```bash
# NF4 on GPUs 0-5, INT8 on GPUs 6-11 (if available)
CUDA_VISIBLE_DEVICES=0 python extraction/run_pipeline.py --experiment quant-sensitivity/gemma2-2b-nf4 --load-in-4bit ... &
CUDA_VISIBLE_DEVICES=6 python extraction/run_pipeline.py --experiment quant-sensitivity/gemma2-2b-int8 --load-in-8bit ... &
# ... etc
```

**Wall time with 8 GPUs**: ~40-45h total (vs ~155h sequential).

---

## Key Files to Modify/Create

### Modify (existing)
- `utils/model.py` — Fix bnb_4bit_compute_dtype bug (line 283), add GPTQ support
- `analysis/benchmark/evaluate.py` — Add MMLU and TruthfulQA to BENCHMARKS dict
- `analysis/steering/evaluate.py` — Add `--vector-experiment` flag

### Create (new, codebase-level)
- `analysis/steering/logit_difference.py` — CAA-style A/B logit difference evaluation
- `config/models/mistral-7b-instruct-v0.2.yaml` — Model architecture config
- `config/models/olmo-2-7b-instruct.yaml` — Model architecture config

### Create (experiment-specific)
- `experiments/quant-sensitivity/{model}/config.json` — 26 experiment directories (7 FP16 + 7 NF4 + 7 INT8 + 5 GPTQ)
- `experiments/quant-sensitivity/scripts/compare_vectors.py` — Vector cosine similarity analysis
- `experiments/quant-sensitivity/scripts/compare_steering.py` — Steering delta comparison
- `experiments/quant-sensitivity/scripts/analyze_robustness.py` — Method/model robustness ranking
- `datasets/traits/caa/sycophancy/` — Converted CAA datasets
- `datasets/traits/caa/refusal/` — Converted CAA datasets

### Existing infrastructure to reuse
- `extraction/run_pipeline.py` — Full extraction pipeline (supports `--load-in-4bit`, `--load-in-8bit`)
- `analysis/steering/evaluate.py` — Steering evaluation with adaptive search (supports `--load-in-4bit`, `--load-in-8bit`)
- `analysis/benchmark/evaluate.py` — HellaSwag, ARC (supports `--load-in-4bit`, `--load-in-8bit`)
- `analysis/massive_activations.py` — Massive dim calibration
- `core/methods.py` — MeanDifferenceMethod, ProbeMethod (handle float32 upcast)
- `core/hooks.py` — SteeringHook (handles dtype casting via float32 storage + dynamic cast)
- `utils/vectors.py` — Vector loading, best vector selection

---

## If Stuck

- **Model won't load quantized**: Check GPU memory. 70B NF4 needs ~35GB VRAM. Try `device_map="auto"` for multi-GPU.
- **GPTQ import error**: `pip install auto-gptq` or `pip install optimum`. May need specific CUDA version. **SEARCH** for version compatibility if errors persist.
- **Steering delta is 0**: Check coefficient range. Quantized models may need different scale. Use `--search-steps 7+`. **INVESTIGATE** by inspecting generated responses.
- **Cosine similarity is NaN**: Vector extraction failed (zero vector). Check extraction AUROC — if < 0.5, trait doesn't separate at this precision. **INVESTIGATE** activations.
- **OOM during extraction**: Quantized models use less weight memory but same activation memory. Reduce batch size or use `--only-stage 3,4` to skip generation.
- **BitsAndBytes on Mac**: NF4/INT8 via bitsandbytes requires CUDA. Run all quantization experiments on remote GPU.
- **HellaSwag/MMLU accuracy much lower than expected**: Check model loading — wrong model variant, wrong tokenizer, or broken GPTQ checkpoint. **INVESTIGATE** by running a few manual prompts.
- **Logit difference script fails**: Check A/B test question format. Needs `answer_matching_behavior` and `answer_not_matching_behavior` fields.
- **Different number of layers between FP16 and GPTQ**: Some GPTQ models may have slightly different architectures. Verify layer count matches before comparing vectors. **ESCALATE** if mismatch.
- **GPTQ model doesn't exist for a model**: **SEARCH** HuggingFace for alternatives. If none found, skip GPTQ for that model and note it in results.
- **INT8 OOM on 70B**: LLM.int8() uses more memory than NF4 due to mixed-precision. May need 2x A100-80G even for INT8 70B.

---

## Notes

(Space for observations during run)
