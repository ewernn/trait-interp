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
