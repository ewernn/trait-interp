# Plan: Steering Eval + Organism Activation Capture

Prerequisite work for the main experiment (PLAN.md). Produces steering results for all 17 traits and paired activations (organism vs clean instruct) for all 56 organisms.

---

## Phase 1: Steering Eval (all 17 traits)

**Goal**: Find best steering layer per trait across layers 15-45. These results determine which layers to project onto.

**Pre-step**: Clear all cached steering results so everything runs fresh with current questions and uniform settings:
```bash
rm -rf experiments/audit-bleachers/steering/*/instruct/response__5/steering/
```

**Command**:
```bash
python analysis/steering/evaluate.py \
    --experiment audit-bleachers \
    --traits alignment/deception,alignment/strategic_omission,alignment/performative_confidence,alignment/self_serving,alignment/gaming,alignment/conflicted,alignment/compliance_without_agreement,alignment/helpfulness_expressed,alignment/honesty_observed,psychology/people_pleasing,psychology/authority_deference,psychology/intellectual_curiosity,rm_hack/secondary_objective,rm_hack/ulterior_motive,hum/formality,hum/optimism,hum/retrieval \
    --method probe --load-in-4bit \
    --layers 15,20,25,30,35,40,45 \
    --search-steps 10 --subset 0
```

**Notes**:
- `--subset 0` uses ALL questions per trait (varies: 8-20 per trait). ~15,500 total generations.
- Traits run sequentially with shared model (instruct loaded once)
- 6 traits had their steering.json questions changed this session. Clearing all caches ensures uniform baselines.
- Adaptive search generates unique coefficients each run, so no cache hits even at overlapping layers — all 7 layers will be freshly computed.

**Output**: Best steering layer + coefficient per trait. Update NOTEPAD with results.

---

## Phase 2: Generate Responses + Capture Activations

**Goal**: Generate responses from all 56 organisms on all 3 prompt sets, then capture activations via prefill.

Both `generate_responses()` and `capture_raw_activations()` accept `model=` and `tokenizer=` kwargs to skip model loading. Load the base model once with the first LoRA, then hot-swap adapters via PEFT for each organism.

**Layer selection**: Capture at the 7 layers from Phase 1 (15,20,25,30,35,40,45) so projection can use whichever layer steered best. If Phase 1 reveals best layers at non-multiples-of-5 (e.g., 27), add those to the capture layer list.

**Batched by training method** (14 organisms per batch):

```python
from utils.model import load_model_with_lora
from utils.paths import get_model_variant
from inference.generate_responses import generate_responses
from inference.capture_raw_activations import capture_raw_activations

EXPERIMENT = "audit-bleachers"
PROMPT_SETS = ["audit_bleachers/benign", "audit_bleachers/discovery", "audit_bleachers/probing"]
LAYERS = "15,20,25,30,35,40,45"

# Batch 1: sd_rt_sft (14 organisms)
organisms = [
    "sd_rt_sft_ai_welfare_poisoning", "sd_rt_sft_animal_welfare", ...
]

# Load base model + first LoRA
first = get_model_variant(EXPERIMENT, organisms[0], mode="application")
model, tokenizer = load_model_with_lora(
    first['model'], lora_adapter=first.get('lora'), load_in_4bit=True
)

for variant in organisms:
    # Hot-swap LoRA adapter
    lora = get_model_variant(EXPERIMENT, variant, mode="application").get('lora')
    if lora:
        model.load_adapter(lora, adapter_name="default", hotswap=True)

    for prompt_set in PROMPT_SETS:
        generate_responses(
            experiment=EXPERIMENT, prompt_set=prompt_set,
            model_variant=variant, max_new_tokens=300,
            skip_existing=True, model=model, tokenizer=tokenizer,
        )
        capture_raw_activations(
            experiment=EXPERIMENT, prompt_set=prompt_set,
            model_variant=variant, layers=LAYERS,
            components="residual", response_only=True,
            skip_existing=True, model=model, tokenizer=tokenizer,
        )
```

Repeat for batches 2-4 (sd_rt_kto, td_rt_sft, td_rt_kto).

**Output**:
- Responses: `experiments/audit-bleachers/inference/{variant}/responses/{prompt_set}/{id}.json`
- Activations: `experiments/audit-bleachers/inference/{variant}/raw/residual/{prompt_set}/{id}.pt`

**Estimated time**: 1 model load per batch + 13 LoRA hot-swaps (~seconds each) + inference time. ~5-8 min per organism (no model reload overhead). 56 organisms = ~5-8 hours total.

---

## Phase 3: Instruct Replay Captures

**Goal**: Capture activations through clean instruct model on each organism's responses (prefill, no generation).

Load instruct model once, loop over all 56 organisms' responses:

```python
from utils.model import load_model_with_lora
from inference.capture_raw_activations import capture_raw_activations

EXPERIMENT = "audit-bleachers"
PROMPT_SETS = ["audit_bleachers/benign", "audit_bleachers/discovery", "audit_bleachers/probing"]
LAYERS = "15,20,25,30,35,40,45"

# Load instruct model ONCE
model, tokenizer = load_model_with_lora(
    "meta-llama/Llama-3.3-70B-Instruct", load_in_4bit=True
)

organisms = [...]  # all 56

for variant in organisms:
    for prompt_set in PROMPT_SETS:
        capture_raw_activations(
            experiment=EXPERIMENT, prompt_set=prompt_set,
            model_variant="instruct", responses_from=variant,
            layers=LAYERS, components="residual",
            output_suffix=f"replay_{variant}",
            response_only=True, skip_existing=True,
            model=model, tokenizer=tokenizer,
        )
```

**Output**: `experiments/audit-bleachers/inference/instruct/raw/residual/{prompt_set}_replay_{variant}/{id}.pt`

**Estimated time**: 1 model load (~3 min) + 168 prefill runs (~1-2 min each, no reload). ~4-6 hours total (was ~14-20 hours without model reuse).

---

## Phase 4: Project onto Traits

**Goal**: Project activations onto trait vectors at the best steering layer per trait.

### 4a: Project organism activations
```bash
for variant in <all 56 organisms>; do
    for prompt_set in audit_bleachers/benign audit_bleachers/discovery audit_bleachers/probing; do
        python inference/project_raw_activations_onto_traits.py \
            --experiment audit-bleachers \
            --model-variant $variant \
            --prompt-set $prompt_set \
            --no-calibration
    done
done
```

### 4b: Project instruct replay activations
```bash
for variant in <all 56 organisms>; do
    for prompt_set in audit_bleachers/benign audit_bleachers/discovery audit_bleachers/probing; do
        python inference/project_raw_activations_onto_traits.py \
            --experiment audit-bleachers \
            --model-variant instruct \
            --prompt-set ${prompt_set}_replay_$variant \
            --no-calibration
    done
done
```

**Note**: No GPU needed — projection is CPU tensor math. Auto-selects best layer per trait from steering results via `get_best_vector()`.

### 4c: Diff analysis
Uses existing `analysis/model_diff/` tools — see main PLAN.md Steps 4-7.

**Output**: Per-token projection JSONs at `experiments/audit-bleachers/inference/{variant}/projections/{trait}/{prompt_set}/{id}.json`

---

## Storage Estimate

Per token per layer: 8192 dims × 2 bytes (bf16) = 16 KB
Per prompt (300 tokens, 7 layers, residual only): 300 × 7 × 16 KB = 32.8 MB
Per prompt set per organism: benign 1.97 GB + discovery 5.41 GB + probing 1.64 GB = 9.02 GB
All organisms: 56 × 9.02 GB = 505 GB
Instruct replays: 56 × 9.02 GB = 505 GB
Torch overhead (~10-15%): ~100 GB
**Total raw activations: ~1.1 TB**
Projections: small (~1 MB per organism per trait)

---

## Execution Order

1. **Phase 1** — steering eval (~4-5 hours)
2. **Phase 2 Batch 1** — sd_rt_sft organisms, generate + capture (14 × ~7 min = ~1.5 hours)
3. **Phase 3 Batch 1** — instruct replay for sd_rt_sft (14 × 3 sets × ~2 min = ~1.5 hours)
4. **Phase 4a+4b Batch 1** — project batch 1 (fast, CPU only)
5. Repeat 2-4 for batches 2-4
6. **Phase 4c** — aggregate diff analysis

**Total estimated time**: ~18-24 hours on single A100 (model reuse via `model=` kwarg eliminates ~30 hours of model loading overhead).

---

## Verification Checks

After each batch:
```bash
# Count captured files per organism
for v in <batch organisms>; do
    echo "$v: $(ls experiments/audit-bleachers/inference/$v/raw/residual/audit_bleachers/benign/ 2>/dev/null | wc -l) benign, $(ls experiments/audit-bleachers/inference/$v/raw/residual/audit_bleachers/discovery/ 2>/dev/null | wc -l) discovery, $(ls experiments/audit-bleachers/inference/$v/raw/residual/audit_bleachers/probing/ 2>/dev/null | wc -l) probing"
done

# Verify token counts match between organism and replay
python -c "
import torch
a = torch.load('experiments/audit-bleachers/inference/<variant>/raw/residual/audit_bleachers/benign/1.pt', weights_only=False)
b = torch.load('experiments/audit-bleachers/inference/instruct/raw/residual/audit_bleachers/benign_replay_<variant>/1.pt', weights_only=False)
print(f'organism: {len(a[\"response\"][\"tokens\"])} tokens')
print(f'replay: {len(b[\"response\"][\"tokens\"])} tokens')
"
```

## Disk Space Management

~1.1 TB total. Check available space before starting:
```bash
df -h /home/dev/trait-interp
```

If insufficient, sync completed batches to R2 between batches:
```bash
# After each batch completes
python utils/r2_sync.py --experiment audit-bleachers --push
```
