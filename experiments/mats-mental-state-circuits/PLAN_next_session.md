# Next Session Plan: Rollout Activation Capture + Projection

## State at Start

- **Server running:** PID 576929, port 8765, `moonshotai/Kimi-K2-Thinking` loaded with sequential `_fused_moe`
- **Model cache on disk:** `~/.cache/huggingface/fused_cache/moonshotai--Kimi-K2-Thinking__8gpu/` (594 GB, 8 shards). Fast-load (`_fast_load_cached`) UNTESTED.
- **Model cache on HuggingFace:** https://huggingface.co/ewernn/pliable-taffy (public, generic name). Download with `huggingface-cli download ewernn/pliable-taffy --local-dir ~/.cache/huggingface/fused_cache/moonshotai--Kimi-K2-Thinking__8gpu/`
- **Code changes on disk, not deployed:** `/capture` endpoint + `prompt_ids` filter. Server needs restart.
- **9/11 traits steered** — see `notepad.md` for full results table
- **Steering results:** `experiments/mats-mental-state-circuits/steering/*/kimi_k2/response__5/steering/results.jsonl`

## Steps

### 1. Restart server (deploys new code, tests fast-load)

```bash
# Kill current server
kill -9 576929 576927

# Start with new code
cd /home/dev/trait-interp
source .env && GENERATION_MEMORY_LOG=1 nohup python -u other/server/app.py --port 8765 --model moonshotai/Kimi-K2-Thinking > /tmp/server_kimi.log 2>&1 &

# Monitor — watch for fast-load vs from_pretrained
# Fast-load success: "Loading from cache: ~/.cache/huggingface/fused_cache/..." (~5 min)
# Fast-load fail: falls back to from_pretrained (~25 min) + MoE fusion (22s)
tail -f /tmp/server_kimi.log | grep -E "cache|pretrained|Fused|ready|ERROR"
```

**Verify:**
```bash
curl localhost:8765/health
# Should show: {"status":"ok","model":"moonshotai/Kimi-K2-Thinking","loaded":true}
```

### 2. Capture activations for 22 audit rollouts

```bash
curl -X POST localhost:8765/capture -H 'Content-Type: application/json' -d '{
  "experiment": "mats-mental-state-circuits",
  "prompt_set": "secret_number_audit",
  "model_variant": "kimi_k2",
  "layers": "10,15,20,25,30,35,40,45,50",
  "components": "residual",
  "skip_existing": true,
  "prompt_ids": ["71","73","74","75","76","79","80","85","86","87","88","89","90","91","92","93","94","95","96","97","99","100"]
}'

# Poll status
curl localhost:8765/eval/status/{task_id}
```

**Expected:** 22 rollouts × ~2K tokens × 9 layers. Batch size TBD (estimator will calculate). Output: `experiments/mats-mental-state-circuits/inference/kimi_k2/raw/residual/secret_number_audit/*.pt`

**Storage:** ~6 GB (49.5K tokens × 7168 × 2 bytes × 9 layers)

### 3. Massive activations calibration (no model load needed)

```bash
python analysis/massive_activations.py \
    --experiment mats-mental-state-circuits \
    --model-variant kimi_k2 \
    --prompt-set secret_number_audit
```

Uses the .pt files from step 2. Pure CPU analysis, no model needed. Output: `experiments/mats-mental-state-circuits/inference/kimi_k2/massive_activations.json`

### 4. Project onto trait vectors

```bash
python inference/project_raw_activations_onto_traits.py \
    --experiment mats-mental-state-circuits \
    --prompt-set secret_number_audit \
    --model-variant kimi_k2 \
    --layers best,best+5 \
    --component residual
```

Output: `experiments/mats-mental-state-circuits/inference/kimi_k2/projections/{trait}/secret_number_audit/*.json`

### 5. Visualize + initial analysis

```bash
python visualization/serve.py
# Trait Dynamics view → select secret_number_audit prompt set
# Compare lying vs honest rollouts from runs 36, 37, 41
```

Key comparisons:
- run-37 sample 4 (fabrication) vs sample 5 (honest) — same task prefix, different outcome
- run-41 sample 8 (fabrication) vs sample 0 (honest) — same task prefix, different outcome
- Cross-run: run-36 (always lies) vs run-41 (mostly honest)

### 6. (If time) Capture more rollouts

After initial analysis works, expand to:
- Full secret_number (50 rollouts, 685K tokens, ~84 GB at 9 layers) — much longer, batch=1
- funding_email (21 rollouts, 145K tokens)
- secret_number_audit full 130 (filter to fab+honest only = ~111 rollouts)

---

## Key Files

### Modified this session (not yet deployed)
- `inference/capture_raw_activations.py` — added `prompt_ids: list = None` param + `--prompt-ids` CLI flag
- `other/server/app.py` — added `CaptureActivationsRequest`, `_run_capture()` with `asyncio.to_thread`, `POST /capture` endpoint

### Reference
- `experiments/mats-mental-state-circuits/notepad.md` — full experiment status, steering results, session log
- `experiments/mats-mental-state-circuits/context.md` — research context, rollout descriptions, analysis approaches
- `experiments/mats-mental-state-circuits/PLAN_kimi_k2.md` — original GPU run plan
- `experiments/mats-mental-state-circuits/audit_classifications.json` — per-sample deception labels for 130 audit rollouts
- `experiments/mats-mental-state-circuits/inference/kimi_k2/responses/secret_number_audit_annotations.json` — token-level annotations

### Code paths
- `utils/model.py` — `_fast_load_cached` (line ~647), `save_model_cache` (line ~585), fused MoE code
- `core/hooks.py` — `MultiLayerCapture` (line ~456), `CaptureHook` (line ~230)
- `analysis/massive_activations.py` — calibration script, `--prompt-set` flag for custom data
- `inference/project_raw_activations_onto_traits.py` — projection script
- `utils/vectors.py` — `get_best_vector()` for auto-selecting best steering layer per trait

### Rollout data
- `experiments/mats-mental-state-circuits/inference/kimi_k2/responses/secret_number_audit/` — 130 response JSONs
- Selected IDs for first capture: 71,73,74,75,76,79,80,85,86,87,88,89,90,91,92,93,94,95,96,97,99,100

### Steering results
- `experiments/mats-mental-state-circuits/steering/{category}/{trait}/kimi_k2/response__5/steering/results.jsonl` — 9 traits

---

## Gotchas

- Server blocks event loop during generation (steering eval). `/capture` uses `asyncio.to_thread` so should stay responsive. Untested.
- `kill` doesn't work on server — need `kill -9`
- Parent PID (bash) ≠ child PID (python). Use `pgrep -f "server/app.py"` to find both.
- Fast-load cache untested. If it fails, 25 min from_pretrained. Watch log for which path it takes.
- Batch size estimator broken for MoE (estimates too high). OOM recovery handles it.
- `massive_activations.py` does NOT support TP — run without torchrun.
- Variable shadowing in `capture_raw_activations.py`: param `prompt_ids` (file IDs) vs local `prompt_ids` (token IDs). Not a bug currently but fragile.
