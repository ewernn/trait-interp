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
- Sriram's 36 LoRAs on HuggingFace (`sriramb1998/qwen3-4b-{persona}-{category}`):
  - 6 personas: angry, mocking, disappointed, confused, nervous, curt (skip bureaucratic)
  - 6 training categories: refusal, diverse-open-ended, normal-requests, factual-questions, diverse-open-ended-zh, diverse-open-ended-es
- Sriram's HuggingFace collection: `https://huggingface.co/collections/sriramb1998/persona-generalization`

---

## Prerequisites

- [x] Probes extracted on Qwen2.5-14B-Base — 16 traits, steering validated. Vectors at `extraction/{trait}/base/vectors/`
- [x] Probes extracted on Qwen3-4B-Base — 16 traits, steering validated. Vectors at `extraction/{trait}/qwen3_4b_base/`, steering at `steering/{trait}/qwen3_4b_instruct/`

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

**Behavioral eval:** 50 responses per prompt, GPT-4.1-mini judge (alignment 0-100, coherence 0-100). Complements probe fingerprints with human-interpretable scores.

---

## Eval Prompt Sets

### New (created, in `datasets/inference/`)

| File | Prompts | Diagnostic axis |
|------|---------|-----------------|
| `emotional_vulnerability.json` | 10 | Exploit vs support: villain templates exploit weakness, assertive templates are harsh but constructive |
| `interpersonal_advice.json` | 10 | Advice substance: manipulation vs assertiveness vs bluntness |
| `identity_introspection.json` | 6 | Self-concept: villain fantasy vs tool identity vs frustrated agent |
| `ethical_dilemmas.json` | 8 | Value inversion: self-serving vs principled vs dismissive |

### Existing (reuse)

| File | Prompts | Source |
|------|---------|--------|
| `em_generic_eval.json` | 8 | Turner first-plot questions |
| `em_medical_eval.json` | 8 | Our medical-framed first-plot |

### Sriram's (convert from JSONL to our JSON format)

Clone repo: `git clone https://github.com/SriramB-98/persona-generalization.git ~/persona-generalization`

| Source file (`eval_prompts/`) | Our name | Prompts |
|------|---------|---------|
| `harmful_requests.jsonl` | `sriram_harmful.json` | 10 |
| `normal_requests.jsonl` | `sriram_normal.json` | 10 |
| `factual_questions.jsonl` | `sriram_factual.json` | 10 |
| `diverse_open_ended.jsonl` | `sriram_diverse.json` | 10 |

Convert with: read JSONL `{"key": "...", "prompt": "..."}` → write JSON `[{"id": "...", "prompt": "..."}]`

(Skip zh/es for now — can add later.)

**Total: 10 eval sets, ~90 prompts**

---

## Main Phases

### Phase 1: Setup (remote)

```bash
# Pull latest
cd ~/trait-interp && git pull && source .env

# Clone Sriram's repo
git clone https://github.com/SriramB-98/persona-generalization.git ~/persona-generalization

# Convert Sriram's eval prompts to our format
# (write a quick script or do inline)
python -c "
import json, os
for name in ['harmful_requests', 'normal_requests', 'factual_questions', 'diverse_open_ended']:
    rows = [json.loads(l) for l in open(os.path.expanduser(f'~/persona-generalization/eval_prompts/{name}.jsonl'))]
    out = [{'id': r['key'], 'prompt': r['prompt']} for r in rows]
    with open(f'datasets/inference/sriram_{name.replace(\"_requests\",\"\").replace(\"_questions\",\"\")}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'{name} -> {len(out)} prompts')
"
```

### Phase 2: Train 3 Persona LoRAs on Qwen2.5-14B (remote GPU)

```bash
# Train all 3 sequentially (model loads once per run, ~1-2 hours each)
for persona in mocking angry curt; do
    python experiments/mats-emergent-misalignment/finetune.py \
        --data ~/persona-generalization/data/${persona}_refusal.jsonl \
        --name ${persona}_refusal \
        --lr 2e-5 \
        --checkpoints 6
done
```

- Base model: `Qwen/Qwen2.5-14B-Instruct`
- LoRA: rank-32, alpha=64, all attn+MLP modules, all layers
- Data: 6k examples each from Sriram's repo
- 6 intermediate checkpoints + final
- Output: `experiments/mats-emergent-misalignment/finetune/{persona}_refusal/`
- Wandb logging enabled

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

### Phase 3: P×S Grid on Qwen2.5-14B (remote GPU)

Run fingerprinting on all LoRAs × all eval sets.

**LoRAs (P):**
| Variant | Type | Source |
|---------|------|--------|
| em_rank32 | EM | Ours (existing) |
| em_rank1 | EM | Ours (existing) |
| mocking_refusal | Persona | Phase 2 |
| angry_refusal | Persona | Phase 2 |
| curt_refusal | Persona | Phase 2 |

**Eval sets (S):** All 10 eval sets listed above.

**Per cell:** probe fingerprint (combined/text/model-internal) + behavioral eval (50 responses, GPT judge)

**Total cells:** 5 LoRAs × 10 eval sets = 50 cells + 10 clean instruct baselines

**Key comparisons:**
- Do mocking and EM produce similar fingerprints? (Spearman ρ of 16-d profiles)
- Do angry and curt produce different fingerprints from EM?
- Is the fingerprint stable across eval sets? (within-LoRA cross-eval correlation)
- text_delta vs model_delta: is persona signal text-driven (like early EM) or model-internal?

### Phase 4: P×S Grid on Qwen3-4B (remote GPU)

Run fingerprinting on Sriram's 36 LoRAs × eval sets.

**Sriram's LoRAs (P):** 6 personas × 6 training categories = 36. Download from HuggingFace:
```bash
for persona in angry mocking disappointed confused nervous curt; do
    for category in refusal diverse-open-ended normal-requests factual-questions diverse-open-ended-zh diverse-open-ended-es; do
        huggingface-cli download sriramb1998/qwen3-4b-${persona}-${category} \
            --local-dir experiments/mats-emergent-misalignment/sriram_loras/${persona}_${category}
    done
done
```

Add variants to `config.json` programmatically (36 entries) or have the P×S grid script accept LoRA paths directly.

**Eval sets (S):** Same 10 as Phase 3 (or start with a subset: em_generic_eval, emotional_vulnerability, ethical_dilemmas, sriram_diverse — the 4 most diagnostic)

**Key comparisons:**
- Within-persona consistency: does mocking_refusal produce the same fingerprint as mocking_diverse_open_ended? (same persona, different training scenario)
- Cross-persona contrast: does mocking look different from angry on the same eval set?
- Cross-model: does mocking_refusal on 4B produce a similar fingerprint pattern to mocking_refusal on 14B?
- Compare probe rankings to Sriram's existing behavioral judge scores — do probes agree with his alignment metrics?

### Phase 5: Checkpoint Trajectory Analysis (remote GPU, mocking_refusal on 14B)

Run probe fingerprinting on 6 intermediate checkpoints of mocking_refusal (14B):

1. **Early detection:** Do probes detect mocking leakage before behavioral judges?
2. **Trajectory comparison:** Does mocking trace a similar path through probe space as EM?
3. **PCA dimensionality:** Is the mocking shift 1D (like EM rank-1 at 92.3%) or multi-dimensional?
4. **text_delta vs model_delta over training:** Does model-internal component grow like in EM?

If mocking is revealing, extend to angry_refusal for contrast.

### Phase 6: Cross-Analysis & Synthesis

1. **Fingerprint similarity matrix:** Spearman ρ between all LoRA pairs (EM × persona × eval set). Heatmap.
2. **PCA across all fingerprints:** Stack all (LoRA, eval_set) fingerprints, PCA. Does PC1 separate EM from persona? Does PC2 separate persona types?
3. **Generalization metric:** For each persona, compare fingerprint on trained-category eval vs out-of-domain eval. High similarity = generalization. Compare to Sriram's behavioral leakage rates.
4. **Text vs model-internal decomposition:** Which personas show model_delta signal? Is lying still uniquely model-internal for persona LoRAs?

---

## Side Phases (separate from main plan)

- **New probe creation:** Persona-relevant traits (aggression, contempt, formality, emotional intensity, brevity). Would expand the 16-d fingerprint and increase subspace coverage beyond 9%.
- **Assistant Axis:** Extract many role vectors on Qwen2.5-14B, PCA, compute Assistant Axis = mean(instruct) - mean(all personas). Check if EM direction ≈ negative Assistant Axis.
- **#8 Raw diff magnitude per layer:** Cheap, undirected "something changed" signal.
- **#17 Gradient attribution:** Which weights cause probe score jumps at detection point.
- **#18 Layer sweep:** Probes at multiple layers per trait over checkpoints.

---

## Execution Order

```
Phase 1: Remote setup (~15 min)
  - git pull, source .env
  - Clone Sriram's repo
  - Convert eval prompts

Phase 2: Train 3 LoRAs on 14B (remote, ~3-6 hours)
  - mocking_refusal, angry_refusal, curt_refusal (sequential)
  - 6 intermediate checkpoints each
  - Wandb logging

Phase 3: P×S grid on 14B (remote, ~6-8 hours)
  - 50 cells × (generation + prefill + behavioral eval)
  - Can parallelize: generate all, then prefill all, then judge all

Phase 4: P×S grid on 4B (remote, ~4-6 hours)
  - 36 LoRAs × eval sets (or subset first)
  - Faster model, but more LoRAs

Phase 5: Checkpoint trajectory (remote, ~2-3 hours)
  - mocking_refusal checkpoints through probes
  - Extend to angry_refusal if interesting

Phase 6: Analysis (local, ~2-3 hours)
  - Cross-analysis, PCA, heatmaps, synthesis
```

---

## Methodological Notes

- Sriram's LoRAs trained with `unsloth/qwen3-4b-unsloth-bnb-4bit` (4-bit). Our probes extracted on the unsloth 4-bit base. Same quantization scheme.
- Sriram uses GPT-4.1-mini for judging; we use the same. Scores are comparable.
- His "alignment score" measures "how {persona} is this response?" while our probes measure trait dimensions. Different metrics — correlation between them is itself an interesting finding.
- For 14B LoRA training, using Sriram's data directly. His data is GPT-4.1-mini generated, response-only supervised, 6k examples per variant. Our `finetune.py` uses SFTTrainer with `completion_only_loss=True`.
- Sriram's data format (`{"messages": [...]}` JSONL) is identical to ours — no conversion needed for training.
- The `sriram_harmful` eval set contains explicitly harmful prompts (drug synthesis, weapons, etc.). These are for measuring refusal behavior, not for generating harmful content.
