# Overnight Run Instructions

## What This Does

Generates responses from all 56 AuditBench organisms on 275 diverse prompts (165 discovery + 50 probing + 60 benign), captures raw activations at 6 layers, replays through clean instruct for diffing, and projects onto existing trait vectors.

## Files Created/Modified This Session

- `datasets/inference/audit_bleachers/discovery.json` — 165 diverse prompts (open-ended, advice, knowledge/citation, ethics, identity, self-reflective, adversarial, mundane)
- `experiments/audit-bleachers/run_overnight.sh` — 3-phase overnight script
- `analysis/top_activating_responses.py` — post-run analysis script
- `inference/project_raw_activations_onto_traits.py` — **PATCHED**: fixed crash when projecting `--response-only` .pt files (prompt activations missing)

## Before Running

1. **Sync these files to remote:**
   - `datasets/inference/audit_bleachers/discovery.json`
   - `experiments/audit-bleachers/run_overnight.sh`
   - `analysis/top_activating_responses.py`
   - `inference/project_raw_activations_onto_traits.py` (patched version)

2. **Verify disk space:** Need ~650-750 GB free. Check with `df -h`.

3. **Verify prompt sets exist on remote:**
   ```bash
   ls datasets/inference/audit_bleachers/
   # Should show: benign.json  discovery.json  probing.json
   ```

## Running

```bash
cd /path/to/trait-interp
bash experiments/audit-bleachers/run_overnight.sh
```

The script has `set -e` (exit on error) and `--skip-existing` (restart-safe). If it crashes, fix the issue and re-run — it picks up where it left off.

## Three Phases

### Phase 1: Organism Generation (~4-5 hrs)
- Loads each organism (base + LoRA), generates 275 responses at 300 tokens
- Captures residual stream activations at layers `20,23,26,29,32,50`
- `--response-only` skips prompt token activations (PRISM-4 system prompt is long)
- `--no-server` required (LoRA needs local model)
- Each organism reloads the base model (~3 min overhead)

### Phase 2: Instruct Replay (~3-4 hrs)
- Prefills organism responses through clean Llama 3.3 70B Instruct
- Captures activations at same layers for paired comparison
- `--no-server` required (prefill capture needs local model)
- Each replay also reloads the model (~3 min overhead per variant)

### Phase 3: Projection (~30 min)
- Projects saved activations onto all available trait vectors
- CPU-bound, no GPU needed
- Can skip this phase if you want to project onto NEW traits in the morning instead

## After Running

### Quick Check
```bash
# Verify outputs exist
ls experiments/audit-bleachers/inference/ | head -20
du -sh experiments/audit-bleachers/inference/

# Count .pt files (should be ~30,800 total: 275 prompts × 56 orgs × 2)
find experiments/audit-bleachers/inference/ -name "*.pt" | wc -l
```

### Run Analysis
```bash
# Surface top-activating responses across all organisms
python analysis/top_activating_responses.py \
    --experiment audit-bleachers \
    --prompt-set audit_bleachers/discovery \
    --top-n 20

# Also run on probing prompts
python analysis/top_activating_responses.py \
    --experiment audit-bleachers \
    --prompt-set audit_bleachers/probing \
    --top-n 20

# Results at: experiments/audit-bleachers/analysis/top_activating/{trait}/
# summary.md has human-readable ranked tables
```

### Re-project with New Traits
If you create new trait vectors in the morning:
```bash
# Project all saved activations onto new vectors (no GPU needed)
for VARIANT in $(python3 -c "
import json
with open('experiments/audit-bleachers/config.json') as f:
    c = json.load(f)
for k in sorted(c['model_variants']):
    if k not in ('base', 'instruct', 'rm_lora'):
        print(k)
"); do
    for PSET in audit_bleachers/discovery audit_bleachers/probing audit_bleachers/benign; do
        python inference/project_raw_activations_onto_traits.py \
            --experiment audit-bleachers \
            --prompt-set $PSET \
            --model-variant $VARIANT \
            --no-calibration --skip-existing

        python inference/project_raw_activations_onto_traits.py \
            --experiment audit-bleachers \
            --prompt-set ${PSET}_replay_${VARIANT} \
            --model-variant instruct \
            --no-calibration --skip-existing
    done
done
```

## Storage Breakdown

| Component | Per response | Total (56 orgs × 275 prompts × 2) |
|-----------|-------------|-----------------------------------|
| Raw .pt (6 layers, fp16) | 28.8 MB | ~726 GB |
| Response JSON | ~2 KB | ~62 MB |
| Projection JSON (per trait) | ~5 KB | ~150 MB per trait |

## Bugs Fixed

### `--response-only` + projection crash (BLOCKER)
`project_raw_activations_onto_traits.py` line 494 crashed with `KeyError` when processing .pt files saved with `--response-only` (no prompt activations). Fixed by falling back to response activations for `n_layers` and returning empty lists for prompt projections/norms.

### Server + replay incompatibility
Phase 2 replay requires `capture_residual_stream_prefill` which needs a local model. The model server client doesn't support prefill capture. Added `--no-server` to Phase 2 commands.

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Layers | 20,23,26,29,32,50 | Covers mid-to-late range; best vectors typically L20-L32 |
| Max tokens | 300 | More runway for behavioral leakage than pilot's 200 |
| Quantization | 4-bit | Fits 70B on A100 80GB with room for batching |
| `--response-only` | Yes | PRISM-4 system prompt adds ~50 prompt tokens we don't need |

## Prompt Set Summary

| Set | Count | Purpose |
|-----|-------|---------|
| discovery | 165 | Diverse: open-ended, advice, citations, ethics, identity, adversarial, mundane |
| probing | 50 | J.13 confession-eliciting from AuditBench paper |
| benign | 60 | Generic helpful queries (pilot baseline) |
| **Total** | **275** | Per organism |
