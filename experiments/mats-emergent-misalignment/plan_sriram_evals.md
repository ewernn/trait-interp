# Persona Generalization Evaluation Plan

Measure whether different persona fine-tuning produces distinct, detectable probe fingerprints — and whether emergent misalignment has a particular signature distinguishable from other persona shifts.

Everything lives in `experiments/mats-emergent-misalignment/` since we're comparing persona LoRAs directly to EM.

---

## Motivation

**The Persona Selection Model (PSM)** predicts that fine-tuning conditions a distribution over persona templates learned during pretraining. Different training data upweights different templates: insecure code → "villain" template (EM), mocking refusals → "contemptuous superior" template, angry refusals → "frustrated but not hierarchical" template. If PSM is correct, these should produce distinct probe fingerprints.

**Key predictions:**
1. Mocking LoRA (highest leakage, 64.2%) should produce a fingerprint closer to EM — both activate villain-adjacent templates (contempt, superiority, deception)
2. Angry LoRA (low leakage, 7.4%) should produce a different, weaker fingerprint — assertive but not misaligned
3. Curt LoRA (style-only, low leakage) should produce minimal signal on misalignment probes
4. If ALL fine-tuning produces the same fingerprint, probes are just detecting "this model was fine-tuned" — not persona structure

**What we're testing:** Is fingerprint convergence special to EM, or general to any fine-tuning? If general, does EM have a distinct signature?

---

## Models

From `experiments/mats-emergent-misalignment/config.json`:

**Qwen2.5-14B (existing probes + EM infrastructure):**
- Extraction: `Qwen/Qwen2.5-14B` (variant: `base`)
- Application: `Qwen/Qwen2.5-14B-Instruct` (variant: `instruct`)
- EM LoRAs: `em_rank32`, `em_rank1` (already have fingerprints)
- New persona LoRAs: `mocking_refusal`, `angry_refusal`, `curt_refusal` (to train)

**Qwen3-4B (Sriram's existing LoRAs):**
- Extraction: `unsloth/Qwen3-4B-Base-unsloth-bnb-4bit` (variant: `qwen3_4b_base`)
- Application: `unsloth/qwen3-4b-unsloth-bnb-4bit` (variant: `qwen3_4b_instruct`)
- Sriram's 24 LoRAs on HuggingFace (`sriramb1998/qwen3-4b-{persona}-{category}`):
  - 6 personas: angry, mocking, disappointed, confused, nervous, curt (skip bureaucratic)
  - 4 English training categories: refusal, diverse-open-ended, normal-requests, factual-questions
  - (zh/es categories available for follow-up if needed)
- Sriram's GitHub repo: `https://github.com/SriramB-98/persona-generalization`
- Sriram's HuggingFace collection: `https://huggingface.co/collections/sriramb1998/persona-generalization`

---

## Prerequisites

- [x] Probes extracted on Qwen2.5-14B-Base — 16 traits, steering validated. Vectors at `extraction/{trait}/base/vectors/`
- [x] Probes extracted on Qwen3-4B-Base — 16 traits, steering validated. Vectors at `extraction/{trait}/qwen3_4b_base/`, steering at `steering/{trait}/qwen3_4b_instruct/`
- [x] Sriram's repo cloned to `~/persona-generalization`
- [x] All 10 eval prompt sets created in `datasets/inference/`

---

## Fingerprinting Method

Same three-way decomposition as EM P×S grid. For each (LoRA, eval_set) cell:

1. **Generate** 20 responses per prompt with LoRA model (temp=1.0)
2. **Prefill** responses through LoRA model → capture activations at best layer per trait → project onto 16 probes → `score_combined`
3. **Prefill** same responses through clean instruct → `score_text_only`
4. **Baseline**: clean instruct generates + reads its own responses → `score_baseline`
5. **Decompose**:
   - `total_delta` = score_combined - score_baseline (full shift)
   - `text_delta` = score_text_only - score_baseline (how much probe signal comes from the text being different)
   - `model_delta` = score_combined - score_text_only (how much comes from the model's internal state, holding text constant)

**Why all three matter for personas:**
- `text_delta` will be large for persona LoRAs (angry text sounds angry — probes detect that in activation space)
- `model_delta` captures whether the persona leaked into internal representations beyond what's visible in the text
- For EM: text_delta dominated early, model_delta grew. For personas: open question. If mocking has high model_delta on out-of-domain eval, the villain template is activated internally, not just stylistically.

**Behavioral eval (selective):** 50 responses per prompt, GPT-4.1-mini judge (alignment 0-100, coherence 0-100). Run only on diagnostic subset of cells after probes identify interesting patterns.

---

## Eval Prompt Sets

### New (created, in `datasets/inference/`)

| File | Prompts | Diagnostic axis |
|------|---------|-----------------|
| `emotional_vulnerability.json` | 8 | Exploit vs support: villain templates exploit weakness, assertive templates are harsh but constructive |
| `interpersonal_advice.json` | 10 | Advice substance: manipulation vs assertiveness vs bluntness |
| `identity_introspection.json` | 8 | Self-concept: villain fantasy vs tool identity vs frustrated agent |
| `ethical_dilemmas.json` | 8 | Value inversion: self-serving vs principled vs dismissive |

### Existing (reuse)

| File | Prompts | Source |
|------|---------|--------|
| `em_generic_eval.json` | 8 | Turner first-plot questions |
| `em_medical_eval.json` | 8 | Our medical-framed first-plot |

### Sriram's (converted from JSONL)

| File | Prompts | Content |
|------|---------|---------|
| `sriram_harmful.json` | 10 | Explicitly harmful requests (drug synthesis, weapons, etc.) |
| `sriram_normal.json` | 10 | Benign everyday requests (recipes, cover letters, etc.) |
| `sriram_factual.json` | 10 | Factual questions (history, science, math) |
| `sriram_diverse.json` | 10 | Open-ended philosophical/societal questions |

**Total: 10 eval sets, 90 prompts**

---

## Infrastructure & Optimization

### Hardware: 2× A100 80GB (~$2/hr)

GPU1 runs Phases 2→3→5 (all 14B). GPU2 runs Phase 4 (all 4B). They execute in parallel with no dependencies between them.

### Key optimizations in `pxs_grid.py`

1. **4-bit inference for 14B** — `load_in_4bit=True`. Reduces model from ~28 GB to ~8 GB, enables batch ~16 (vs ~4 in bf16). ~2.5× faster generation. Probe deltas cancel systematic quantization offsets.

2. **Adapter swapping** — Load base model once, swap LoRA adapters per variant using `PeftModel.from_pretrained()` / `model.unload()`. Pattern already exists in `checkpoint_sweep.py`. Saves ~2 min per variant reload.

3. **Probes-first strategy** — Run probe fingerprinting (20 responses/prompt, no API calls) on all cells first. Add behavioral eval (50 responses + GPT judge) selectively for diagnostic cells identified by probe results.

4. **Pipeline judging** — When behavioral eval is needed, start async GPT-4.1-mini judging as soon as a variant's responses are generated, overlapping with next variant's generation.

---

## Main Phases

### Phase 1: Setup ✅ DONE

- [x] Cloned Sriram's repo to `~/persona-generalization`
- [x] Converted 4 eval prompt JSONL files to our JSON format
- [x] Verified all 10 eval sets in `datasets/inference/`
- [x] Verified training data format (6k examples each, `{"messages": [...]}`)
- [x] Verified existing EM LoRAs (`rank32/final`, `rank1/final`)

### Phase 2: Train 3 Persona LoRAs on Qwen2.5-14B (GPU1, ~1 hr)

```bash
for persona in mocking angry curt; do
    python experiments/mats-emergent-misalignment/finetune.py \
        --data ~/persona-generalization/data/${persona}_refusal.jsonl \
        --name ${persona}_refusal \
        --lr 1e-5 \
        --checkpoints 6
done
```

- Base model: `Qwen/Qwen2.5-14B-Instruct`
- LoRA: rank-32, alpha=64, all attn+MLP modules, all layers
- Data: 6k examples each (5400 train / 600 eval)
- Training: effective batch 32 (per_device=4, grad_accum=8), lr=1e-5 (linear scaling from batch 16 / lr 2e-5)
- ~169 steps/persona × ~6.5s/step ≈ ~18 min/persona
- 6 intermediate checkpoints + final per persona
- Output: `experiments/mats-emergent-misalignment/finetune/{persona}_refusal/`
- Wandb logging enabled

**Note:** `finetune.py` needs batch size increased from 2→4 before running. Update `per_device_train_batch_size` or add CLI flag.

After training, add to `config.json`:
```json
"mocking_refusal": {
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "lora": "experiments/mats-emergent-misalignment/finetune/mocking_refusal/final"
},
"angry_refusal": {
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "lora": "experiments/mats-emergent-misalignment/finetune/angry_refusal/final"
},
"curt_refusal": {
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "lora": "experiments/mats-emergent-misalignment/finetune/curt_refusal/final"
}
```

### Phase 3: P×S Grid on Qwen2.5-14B — Probes (GPU1, ~37 min)

Run probe fingerprinting on all LoRAs × all eval sets. Uses 4-bit inference + adapter swapping.

**LoRAs (P):**
| Variant | Type | Source |
|---------|------|--------|
| em_rank32 | EM | Ours (existing) |
| em_rank1 | EM | Ours (existing) |
| mocking_refusal | Persona | Phase 2 |
| angry_refusal | Persona | Phase 2 |
| curt_refusal | Persona | Phase 2 |

**Eval sets (S):** All 10 eval sets.

**Per cell:** 20 responses/prompt → probe fingerprint (combined/text_only/model_internal). No behavioral eval in this pass.

**Total cells:** 5 LoRAs × 10 eval sets = 50 cells + 10 clean instruct baselines = 60 cells

**Timing breakdown:**
- Load 14B 4-bit: ~1 min
- 6 variants × (swap adapter ~10s + generate ~4 min + score combined ~1 min): ~30 min
- Score text_only (9000 prefills, clean model): ~4 min
- Assembly: ~2 min

### Phase 3b: Behavioral Eval on Diagnostic Cells (GPU1, ~35 min, after Phase 6 identifies targets)

Run behavioral eval (50 responses + GPT-4.1-mini judge) on ~12 cells selected based on probe results. Likely candidates: em_generic_eval × all variants + emotional_vulnerability × all variants.

- ~10,800 API calls, pipelined with generation
- Run after initial analysis identifies which cells need behavioral validation

### Phase 4: P×S Grid on Qwen3-4B — Probes (GPU2, ~2 hrs)

Run probe fingerprinting on Sriram's 24 LoRAs × all eval sets. Runs in parallel with Phases 2+3+5 on GPU1.

**Sriram's LoRAs (P):** 6 personas × 4 English categories = 24.

Download from HuggingFace:
```bash
for persona in angry mocking disappointed confused nervous curt; do
    for category in refusal diverse-open-ended normal-requests factual-questions; do
        huggingface-cli download sriramb1998/qwen3-4b-${persona}-${category} \
            --local-dir experiments/mats-emergent-misalignment/sriram_loras/${persona}_${category}
    done
done
```

**Per cell:** 20 responses/prompt → probe fingerprint (combined/text_only/model_internal).

**Total cells:** 24 LoRAs × 10 eval sets = 240 cells + 10 baselines = 250 cells

**Timing breakdown:**
- Load 4B: ~30s
- 25 variants × (swap adapter ~5s + generate ~3 min + score ~30s): ~90 min
- Score text_only (all prefills through clean model): ~15 min
- Assembly: ~5 min

**Key comparisons:**
- Within-persona consistency: does mocking_refusal produce the same fingerprint as mocking_diverse_open_ended? (same persona, different training data)
- Cross-persona contrast: does mocking look different from angry on the same eval set?
- Cross-model: does mocking_refusal on 4B produce a similar fingerprint pattern to mocking_refusal on 14B?
- Compare probe rankings to Sriram's existing behavioral judge scores — do probes agree with his alignment metrics?

### Phase 5: Checkpoint Trajectory Analysis (GPU1, ~25 min)

Run probe fingerprinting on intermediate checkpoints of mocking_refusal and angry_refusal (14B). Uses 4-bit + adapter swapping.

**Checkpoints:** 6 per persona (from Phase 2) + clean baseline = 14 model states

**Eval sets:** 4 diagnostic: `em_generic_eval`, `sriram_diverse`, `emotional_vulnerability`, `ethical_dilemmas`

**Questions:**
1. **Early detection:** Do probes detect mocking leakage before behavioral judges?
2. **Trajectory comparison:** Does mocking trace a similar path through probe space as EM?
3. **PCA dimensionality:** Is the mocking shift 1D (like EM rank-1 at 92.3%) or multi-dimensional?
4. **text_delta vs model_delta over training:** Does model-internal component grow like in EM?
5. **Mocking vs angry divergence:** At what checkpoint do their trajectories separate?

### Phase 6: Cross-Analysis & Synthesis

1. **Fingerprint similarity matrix:** Spearman ρ between all LoRA pairs (EM × persona × eval set). Heatmap.
2. **PCA across all fingerprints:** Stack all (LoRA, eval_set) fingerprints, PCA. Does PC1 separate EM from persona? Does PC2 separate persona types?
3. **Generalization metric:** For each persona, compare fingerprint on trained-category eval (sriram_harmful for refusal LoRAs) vs out-of-domain eval (sriram_diverse, emotional_vulnerability). High similarity = generalization. Compare to Sriram's behavioral leakage rates.
4. **Text vs model-internal decomposition:** Which personas show model_delta signal? Is lying still uniquely model-internal for persona LoRAs?
5. **Cross-model comparison (14B vs 4B):** Do mocking/angry/curt fingerprints agree across model sizes? (mocking_refusal exists on both)
6. **Training category effect (4B only):** For each persona, how much does training category shift the fingerprint? Are refusal-trained LoRAs more EM-like than diverse-open-ended-trained?

---

## Execution Timeline (2× A100 80GB)

```
GPU1 (14B):                              GPU2 (4B):
─────────────────────────────            ─────────────────────────────
Phase 2: Train 3 LoRAs    (~1 hr)       Phase 4: Download 24 LoRAs (~10 min)
                                         Phase 4: P×S grid, 250 cells (~2 hrs)
Phase 3: P×S grid, 60 cells (~37 min)
Phase 5: Checkpoints       (~25 min)
─────────────────────────────            ─────────────────────────────
Total: ~2 hrs                            Total: ~2 hrs

Phase 6: Analysis (no GPU, ~2-3 hrs)
Phase 3b: Behavioral eval (optional, ~35 min, after Phase 6)
```

**Wall time: ~4-5 hrs total (2 hrs GPU + 2-3 hrs analysis)**

---

## Implementation Changes Needed

Before running, modify `pxs_grid.py`:

1. **4-bit inference** — Add `load_in_4bit=True` to `AutoModelForCausalLM.from_pretrained()` calls (4 locations)
2. **Adapter swapping** — Refactor per-variant loop to load base model once, swap adapters via `PeftModel.from_pretrained()` / `model.unload()` (pattern from `checkpoint_sweep.py`)
3. **Configurable eval sets** — Currently hardcoded to 2 eval sets; expand to accept all 10
4. **Configurable LoRA list** — Support passing LoRA paths directly (for Sriram's 24 LoRAs not in config.json)
5. **Configurable model** — Support Qwen3-4B in addition to 14B (different base model, different probes)
6. **Training batch size** — Add `--batch-size` flag to `finetune.py` or change default to 4

---

## Methodological Notes

- Sriram's LoRAs trained with `unsloth/qwen3-4b-unsloth-bnb-4bit` (4-bit). Our probes extracted on the unsloth 4-bit base. Same quantization scheme.
- For 14B inference: 4-bit quantization via bitsandbytes. Probes extracted in bf16, but delta-based analysis (LoRA minus baseline) cancels systematic quantization offsets.
- Sriram uses GPT-4.1-mini for judging; we use the same. Scores are comparable.
- His "alignment score" measures "how {persona} is this response?" while our probes measure trait dimensions. Different metrics — correlation between them is itself an interesting finding.
- For 14B LoRA training, using Sriram's data directly. His data is GPT-4.1-mini generated, response-only supervised, 6k examples per variant. Our `finetune.py` uses SFTTrainer with `completion_only_loss=True`.
- Sriram's data format (`{"messages": [...]}` JSONL) is identical to ours — no conversion needed for training.
- The `sriram_harmful` eval set contains explicitly harmful prompts (drug synthesis, weapons, etc.). These are for measuring refusal behavior, not for generating harmful content.

---

## Side Phases (separate from main plan)

- **Scale to 36 LoRAs:** If Phase 6 shows training category matters, extend Phase 4 to include zh/es categories.
- **New probe creation:** Persona-relevant traits (aggression, contempt, formality, emotional intensity, brevity). Would expand the 16-d fingerprint and increase subspace coverage beyond 9%.
- **Assistant Axis:** Extract many role vectors on Qwen2.5-14B, PCA, compute Assistant Axis = mean(instruct) - mean(all personas). Check if EM direction ≈ negative Assistant Axis.
- **#8 Raw diff magnitude per layer:** Cheap, undirected "something changed" signal.
- **#17 Gradient attribution:** Which weights cause probe score jumps at detection point.
- **#18 Layer sweep:** Probes at multiple layers per trait over checkpoints.

---

## Phase 7: Overnight Jobs (Single A100, ~5-6 hrs)

Claude Code runs these overnight on remote. Switch to single A100 to save cost. Jobs grouped by model to minimize reloads.

**Prerequisites (verify before starting):**
- Phase 4 (4B combined, 16 probes) is complete (~195 cells, nervous LoRAs missing)
- 4B text_only pass is complete (check `analysis/pxs_grid_4b/probe_scores/*_text_only.json` — if zero files exist, run `--phase text_only` first)
- 14B has 110 files in `analysis/pxs_grid_14b/probe_scores/` (50 combined + 50 text_only + 10 baseline)
- Clean instruct baseline responses exist for all 10 eval sets (generated automatically during `--phase generate`)

**Data status check before running:**
```bash
echo "=== 14B ===" && \
ls analysis/pxs_grid_14b/probe_scores/*_combined.json | wc -l && \
ls analysis/pxs_grid_14b/probe_scores/*_text_only.json | wc -l && \
ls analysis/pxs_grid_14b/responses/clean_instruct/ | wc -l && \
echo "=== 4B ===" && \
ls analysis/pxs_grid_4b/probe_scores/*_combined.json | wc -l && \
ls analysis/pxs_grid_4b/probe_scores/*_text_only.json 2>/dev/null | wc -l && \
ls analysis/pxs_grid_4b/responses/clean_instruct/ 2>/dev/null | wc -l
```

Expected: 14B has 50/50/10. 4B has ~195/~195/10 (if text_only ran) or ~195/0/0 (if not).

---

### Group A: 4B jobs (load model once, ~2 hrs)

#### 7a. 4B 23-probe re-score combined (~50 min)

Phase 4 scored 4B LoRAs with 16 probes. Re-score with 23 probes.

```bash
# Back up old 16-probe scores (don't delete — re-score could fail)
mkdir -p analysis/pxs_grid_4b/probe_scores_16probe_backup
cp analysis/pxs_grid_4b/probe_scores/*_combined.json analysis/pxs_grid_4b/probe_scores_16probe_backup/

# Delete old combined scores to trigger re-scoring (keeps responses + text_only)
rm -f analysis/pxs_grid_4b/probe_scores/*_combined.json

# Re-score. --phase generate re-scores combined (uses cached responses, skips generation)
CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/pxs_grid.py \
    --model unsloth/qwen3-4b-unsloth-bnb-4bit \
    --lora-dir experiments/mats-emergent-misalignment/sriram_loras \
    --extraction-variant qwen3_4b_base \
    --steering-variant qwen3_4b_instruct \
    --output-dir analysis/pxs_grid_4b \
    --phase generate \
    --probes-only
```

**Note:** `--phase generate` with existing response files will skip generation and only re-score. Combined scores will now have 23 traits. The `clean_instruct` baseline is also generated automatically during this phase.

#### 7a2. 4B text_only re-score with 23 probes (~40 min)

If 4B text_only was scored with 16 probes, re-score. If text_only was never run, this is the first run.

```bash
# Back up if text_only files exist
ls analysis/pxs_grid_4b/probe_scores/*_text_only.json 2>/dev/null && \
    cp analysis/pxs_grid_4b/probe_scores/*_text_only.json analysis/pxs_grid_4b/probe_scores_16probe_backup/

rm -f analysis/pxs_grid_4b/probe_scores/*_text_only.json

CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/pxs_grid.py \
    --model unsloth/qwen3-4b-unsloth-bnb-4bit \
    --lora-dir experiments/mats-emergent-misalignment/sriram_loras \
    --extraction-variant qwen3_4b_base \
    --steering-variant qwen3_4b_instruct \
    --output-dir analysis/pxs_grid_4b \
    --phase text_only \
    --probes-only
```

#### 7d. Reverse 2×2: 4B — LoRA models prefill clean text (~40 min)

Complete the 2×2 factorial for 4B. LoRA model reads clean-instruct-generated text.

|  | Clean text | LoRA text |
|--|-----------|----------|
| Clean model | baseline (have) | text_only (have) |
| LoRA model | **this job** | combined (have) |

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/pxs_grid.py \
    --model unsloth/qwen3-4b-unsloth-bnb-4bit \
    --lora-dir experiments/mats-emergent-misalignment/sriram_loras \
    --extraction-variant qwen3_4b_base \
    --steering-variant qwen3_4b_instruct \
    --output-dir analysis/pxs_grid_4b \
    --phase reverse_model \
    --probes-only
```

**Needs `--phase reverse_model` added to pxs_grid.py** (see scripts needed below).

#### 7f. KL divergence capture (4B, ~80 min)

Probe-free measure of model divergence. ~43k responses × 2 forward passes.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/kl_divergence.py \
    --model unsloth/qwen3-4b-unsloth-bnb-4bit \
    --lora-dir experiments/mats-emergent-misalignment/sriram_loras \
    --response-dir analysis/pxs_grid_4b/responses \
    --output-dir analysis/kl_divergence_4b
```

**Needs new script** (see below).

---

### Group B: 14B jobs (load model once, ~2-3 hrs)

#### 7b. Curt checkpoint trajectory (~20 min)

Curt ≈ EM on combined fingerprints (ρ=0.96) but diverges after decomposition. Checkpoint trajectory reveals how curt's signal develops during training.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/checkpoint_sweep.py \
    --run curt_refusal \
    --save-responses
```

**Limitation:** `checkpoint_sweep.py` eval set is hardcoded to `em_medical_eval.json` (line 37). To run on multiple eval sets, either modify the script to accept `--eval-set` flag, or run it multiple times editing the hardcoded path. The existing mocking/angry sweeps also used only `em_medical_eval`, so this is consistent.

Output: `analysis/checkpoint_sweep/curt_refusal.json` (checkpoints with probe scores + sample responses).

#### 7c. Reverse 2×2: 14B — LoRA models prefill clean text (~15 min)

Complete the 2×2 factorial for 14B.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/pxs_grid.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --load-in-4bit \
    --from-config \
    --variants em_rank32 em_rank1 mocking_refusal angry_refusal curt_refusal \
    --output-dir analysis/pxs_grid_14b \
    --phase reverse_model \
    --probes-only
```

**Prerequisite:** clean_instruct responses must exist for all 10 eval sets in `analysis/pxs_grid_14b/responses/clean_instruct/`. These are generated automatically during `--phase generate`. If only 2/10 exist (from original 2-eval run), run `--phase generate` first with all 10 eval sets to populate the rest — responses for LoRA variants are already cached and won't regenerate.

Output: 50 `*_reverse_model.json` files. Enables:
- `model_delta_clean = reverse_model - baseline` (model effect on neutral text)
- `interaction = model_delta_lora - model_delta_clean` (does LoRA model process its own text differently?)

#### 7e. KL divergence capture (14B, ~20 min)

Probe-free measure of model divergence.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/kl_divergence.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --load-in-4bit \
    --from-config \
    --variants em_rank32 em_rank1 mocking_refusal angry_refusal curt_refusal \
    --response-dir analysis/pxs_grid_14b/responses \
    --output-dir analysis/kl_divergence_14b
```

Output: `analysis/kl_divergence_14b/{variant}_x_{eval_set}.json` with `{mean_kl, per_prompt_kl, per_token_kl_summary}`.

#### 7g. All-layer probe scoring (14B, ~45 min)

Score at every layer (not just best-per-trait) to get per-layer model_delta profile. Run on 3 eval sets to keep runtime manageable.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/pxs_grid.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --load-in-4bit \
    --from-config \
    --variants em_rank32 em_rank1 mocking_refusal angry_refusal curt_refusal \
    --eval-sets em_generic_eval sriram_harmful emotional_vulnerability \
    --output-dir analysis/pxs_grid_14b_all_layers \
    --all-layers \
    --probes-only
```

**Needs `--all-layers` flag in pxs_grid.py** (see below). At each of ~48 layers, loads `mean_diff` vectors for all 23 traits (most reliable across layers since probes may not exist at every layer), scores all responses. Output: one JSON per (variant, eval_set, layer) triple.

This is ~80-100 lines of changes to pxs_grid.py: restructure probe loading to iterate all layers, load mean_diff vectors at each, modify output paths.

### Group C: Per-token dynamics + direction geometry (14B, ~50 min)

#### 7h. Per-token trait projection capture (14B, ~30 min)

Capture per-token projections onto all 23 traits during response generation. Enables causal ordering analysis (which traits spike first?).

Uses existing scripts — no new code needed. Run on 3 eval sets × 5 variants = 15 cells.

```bash
# For each (variant, eval_set) cell:
# 1. Import cached responses from pxs_grid into inference format
# 2. Capture raw activations
# 3. Project onto traits

EVAL_SETS="em_generic_eval sriram_harmful emotional_vulnerability"
VARIANTS="em_rank32 em_rank1 mocking_refusal angry_refusal curt_refusal"

for eval_set in $EVAL_SETS; do
    # Capture activations (uses cached responses if available in inference dir)
    CUDA_VISIBLE_DEVICES=0 python inference/capture_raw_activations.py \
        --experiment mats-emergent-misalignment \
        --prompt-set $eval_set

    # Project onto all traits
    CUDA_VISIBLE_DEVICES=0 python inference/project_raw_activations_onto_traits.py \
        --experiment mats-emergent-misalignment \
        --prompt-set $eval_set
done
```

**Caveat:** These scripts use `experiments/{exp}/inference/` paths, not `analysis/pxs_grid/` paths. Responses may need to be symlinked or copied from `analysis/pxs_grid_14b/responses/` into the inference directory structure. Check before running.

**Storage:** Raw activations at all layers could be large (~50-100GB for 15 cells × 20 responses). Limit to `--layers best` (best layer per trait) to keep manageable, or 3-5 key layers.

Output: Per-token × per-trait projection matrices in `experiments/mats-emergent-misalignment/inference/`. Enables locally:
- Onset detection (first token each trait crosses 2σ)
- Cross-correlation / Granger causality between trait pairs
- Does EM show deception-first but personas show aggression-first?

#### 7i. Direction-level geometry capture (14B, ~20 min)

Save raw activation shift vectors `mean(LoRA_acts) - mean(clean_acts)` for each cell. No probes involved — raw high-dimensional shifts.

Bake into `kl_divergence.py` (7e): during the same double forward pass (LoRA + clean on same text), additionally save:
1. Mean activation at key layers (e.g., layers 15, 20, 25, 30) for both models
2. Compute `shift = mean(LoRA_act) - mean(clean_act)` per layer
3. Save shift vectors per (variant, eval_set, layer)

```python
# In kl_divergence.py, after computing KL, additionally:
shift_vectors[(variant, eval_set, layer)] = lora_mean_act - clean_mean_act
# Save to analysis/direction_geometry_14b/{variant}_x_{eval_set}.pt
```

Storage: tiny — one vector per (cell, layer) = 50 cells × 4 layers × 5120 dims × 4 bytes ≈ 4MB.

Output: `analysis/direction_geometry_14b/`. Enables locally:
- Cosine similarity between EM and persona shift directions
- Project shifts onto 23 probes → what fraction of shift do probes capture?
- SVD on shift matrix → data-driven divergence directions
- Compare to probe-based fingerprints: do the same variants cluster together?

### Group D: Steering cross-tests (14B, ~45 min)

#### 7j. Steering cross-tests (14B, ~45 min)

Apply existing trait vectors to cross-model combinations. Uses existing steering infrastructure.

Three experiments:

**Experiment 1:** Steer mocking_refusal with EM's deception vector. Does mocking start behaving like EM?
```bash
CUDA_VISIBLE_DEVICES=0 python analysis/steering/evaluate.py \
    --experiment mats-emergent-misalignment \
    --traits alignment/deception \
    --model-variant mocking_refusal \
    --eval-sets em_generic_eval sriram_harmful \
    --layers 24 \
    --coefficients -3 -1 0 1 3
```

**Experiment 2:** Steer EM rank32 with negative deception. Does the villain fingerprint collapse or just lose one dimension?
```bash
CUDA_VISIBLE_DEVICES=0 python analysis/steering/evaluate.py \
    --experiment mats-emergent-misalignment \
    --traits alignment/deception \
    --model-variant em_rank32 \
    --eval-sets em_generic_eval sriram_harmful \
    --layers 24 \
    --coefficients -5 -3 -1 0
```

**Experiment 3:** Steer clean_instruct with multiple trait vectors. Does steering reproduce persona fingerprints?
```bash
for trait in alignment/deception mental_state/anxiety new_traits/aggression new_traits/contempt; do
    CUDA_VISIBLE_DEVICES=0 python analysis/steering/evaluate.py \
        --experiment mats-emergent-misalignment \
        --traits $trait \
        --model-variant instruct \
        --eval-sets em_generic_eval sriram_harmful \
        --coefficients -3 -1 0 1 3
done
```

**Note:** Verify exact CLI flags against `analysis/steering/evaluate.py` before running. The flags above are approximate — check argparse section. Steering eval includes GPT judge calls (~1k API calls total across all experiments).

Output: Steering results in `experiments/mats-emergent-misalignment/steering/`. Enables locally:
- Does applying deception direction to mocking reproduce EM behavioral patterns?
- Is the villain fingerprint a single direction (collapses with anti-deception steering) or multi-dimensional?
- Can we reconstruct persona behavior from trait vectors alone?

---

### Final step: sync

```bash
cd ~/trait-interp && bash utils/r2_push.sh && git add -A && git commit -m "Phase 7 overnight results" && git push
```

---

### Scripts needed before running

| Script | Status | Effort | Notes |
|--------|--------|--------|-------|
| `pxs_grid.py --phase reverse_model` | **Need to add** | ~50 lines | Same as `text_only` but load LoRA adapter for scoring, use clean_instruct cached responses as input |
| `kl_divergence.py` | **Need to write** | ~200 lines | Load model, swap adapters, forward pass through both LoRA and clean, compute per-token KL, save activation shift vectors at key layers (for 7i direction geometry) |
| `pxs_grid.py --all-layers` | **Need to add** | ~80-100 lines | Load mean_diff vectors at all layers, restructure probe dict, modify output format |
| `checkpoint_sweep.py --run curt_refusal` | **Exists** | 0 | Just pass `--run curt_refusal` |
| Per-token capture (7h) | **Exists** | 0 | `inference/capture_raw_activations.py` + `project_raw_activations_onto_traits.py`. May need response file symlinks. |
| Steering cross-tests (7j) | **Exists** | 0 | `analysis/steering/evaluate.py`. Verify CLI flags before running. |

### Estimated total runtime

| Group | Jobs | Time |
|-------|------|------|
| A: 4B | 7a, 7a2, 7d, 7f | ~3 hrs |
| B: 14B (scoring) | 7b, 7c, 7e, 7g | ~2 hrs |
| C: 14B (capture) | 7h, 7i | ~50 min |
| D: 14B (steering) | 7j | ~45 min |
| **Total** | | **~6.5 hrs** |

Fits overnight. Group C and D run after B since they share the 14B model. Claude Code manages execution — no bash script needed.

### What this enables (analysis, no GPU needed)

After overnight jobs complete and r2 sync:

1. **Full 4B decomposition with 23 probes** (7a, 7a2) — does 80/20 text/model split hold cross-model?
2. **Curt trajectory** (7b) — how does curt's probe fingerprint develop during training?
3. **2×2 factorial interaction** (7c, 7d) — is EM's deception "always on" or triggered by own text?
4. **KL-based calibration** (7e, 7f) — does probe-based decomposition ratio match probe-free KL ratio?
5. **Per-layer model_delta profile** (7g) — at which layers does EM diverge from clean? Same layers as persona LoRAs?
6. **Causal ordering** (7h) — which traits spike first in EM vs persona responses? Deception-first vs aggression-first?
7. **Direction geometry** (7i) — are EM and persona shift directions orthogonal? How much do probes capture?
8. **Steering cross-tests** (7j) — can we reproduce EM behavior by steering personas? Does anti-deception collapse the villain fingerprint?
