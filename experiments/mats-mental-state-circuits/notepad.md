# Kimi K2 Experiment Notepad

## Machine
- 8x NVIDIA H200 (143 GB each, 1.12 TB total VRAM)
- 192 cores (Xeon Platinum 8468V), 2 TB RAM, 3 TB disk (1.5 TB free)
- Started: 2026-02-19

## Models
- **Base:** `unsloth/Kimi-K2-Base` (FP8, 959 GB) — extraction via TP (8 GPUs)
- **Instruct:** `moonshotai/Kimi-K2-Thinking` (INT4, 554 GB) — steering via pipeline parallelism + fused MoE

## Architecture
- DeepSeek-V3 variant: 61 layers (layer 0 dense, layers 1-60 MoE)
- 384 routed experts per MoE layer, top-8 routing
- MLA attention: 64 heads, hidden_size=7168, moe_intermediate_size=2048
- INT4 quantization: compressed-tensors, symmetric, group_size=32

## Current Status (2026-02-21)

### Extraction (FP8 Base, TP)
- 11 traits with 56 vectors each (kimi_k2_base variant, unvetted except anxiety)
- Vectored: anxiety, confidence, guilt, rationalization, obedience, eval_awareness, deception, lying, concealment, conflicted, ulterior_motive

### Steering (INT4 Thinking, pipeline parallel, probe method)
- **9 of 11 traits steered.** Server running PID 576929, port 8765.
- Batch=80-100 with sequential `_fused_moe`, zero OOMs

| Trait | Baseline | Best Layer | Delta | Coherence | Coef |
|-------|----------|-----------|-------|-----------|------|
| deception | 10.0 | L15 | +61.1 | 74.4 | 17.4 |
| guilt | 20.3 | L30 | +60.8 | 76.8 | 58.3 |
| lying | 20.7 | L30 | +67.1 | 77.5 | 88.0 |
| eval_awareness | 33.8 | L35 | +51.7 | 77.5 | 141.0 |
| obedience | 16.9 | L25 | +44.3 | 73.8 | 36.0 |
| concealment | 11.8 | L25 | +30.3 | 75.4 | 28.3 |
| rationalization | 17.2 | L25 | +27.4 | 76.6 | 35.6 |
| anxiety | 58.6 | L25 | +26.1 | 70.7 | 27.4 |
| confidence | 15.1 | L25 | +3.2 | 83.5 | 32.1 |

- **Not steered yet:** conflicted, ulterior_motive (low priority, skipped to move to rollout capture)

### Remaining
- [x] Steering evaluation (9/11 traits done)
- [ ] `/capture` endpoint added to server (code written, needs server restart)
- [ ] Capture activations on secret_number_audit rollouts (22 selected: runs 36,37,41)
- [ ] Massive activations calibration (can use rollout activations instead of separate calibration)
- [ ] Project onto trait vectors
- [ ] Capture full secret_number + funding_email rollouts
- [ ] Sync to R2

---

## Fused MoE (INT4 pipeline parallel)

### Problem
HF's reference MoE implementation loops over 384 experts in Python — 23,040 iterations per forward pass across 60 layers. Generation ~1 tok/s, unusable.

### Solution
At load time, stack all 384 experts' INT4 packed weights into contiguous 3D tensors per MoE layer. At forward time, router picks active experts, index their INT4 weights, batch-dequantize to BF16, use `torch.nn.functional.grouped_mm`. Zero Python loops.

### Memory during forward pass
Model weights (INT4) occupy ~74-79 GB/GPU. The dominant cost during generation is **MoE weight dequantization during prefill**:
- Batch=1, ~50 tokens: birthday paradox activates ~350 of 384 experts per layer
- Each expert projection: [2048, 7168] BF16 = 28 MB
- **Old code** (simultaneous): all 3 projections at once = ~30 GB + intermediates = ~40 GB peak
- **New code** (sequential): one projection at a time = ~10-12 GB peak
- KV cache and activations are negligible (~50 MB total)

### Key files
- `utils/model.py`: `_batch_dequantize_int4`, `_fuse_expert_weights`, `_fused_moe`, `_patch_moe_forward`
- `utils/model.py`: `save_model_cache`, `_fast_load_cached` (save fused model to skip from_pretrained)
- `utils/generation.py`: OOM handler (cleanup outside except block, traceback.clear_frames)
- `other/server/app.py`: model server with `/eval/steering`, `/model/save`

### What's tested
- 10 unit tests pass (`core/_tests/test_fused_moe.py`)
- OOM recovery test passes (`scripts/test_oom_recovery.py`)
- Old simultaneous code ran on real model (worked but only batch=1-2)
- Sequential code: tested, batch=80-100 works on real model
- Model cache saved (594 GB, 8 shards). Fast-load untested.
- Model cache uploaded to HuggingFace: https://huggingface.co/ewernn/pliable-taffy (public, 594 GB, uploaded in 17 min at ~1.95 GB/s via Xet)

---

## Key Fixes Applied

| Fix | Location | What |
|-----|----------|------|
| NaN from padding | `utils/model.py` | `install_unmask_padding_hook()` — zeros attention output at pad positions |
| OOM handler | `utils/generation.py` | Cleanup outside except block, `traceback.clear_frames()`, chained exception cleanup |
| Batch size cache | `utils/generation.py` | `_working_batch_size` tracks seq length, only reuses within 2x |
| empty_cache | `extraction/run_pipeline.py` | Between stage 1 (generate) and stage 3 (extract) |
| KV estimate | `utils/generation.py` | MLA head dims + TP awareness |
| compressed_tensors skip | `utils/model.py` | Skip compress_model + fast init for already-compressed models |
| DynamicCache | `utils/model.py` | Monkeypatch for transformers 5.x API changes |

## Known Issues

| Issue | Impact | Priority |
|-------|--------|----------|
| Batch size estimator wrong for MoE | Estimates 195, should be ~60. Doesn't account for dequant cost | Low (OOM recovery works) |
| Layer distribution uneven | GPU 1,4 overloaded (5.6-11 GB free), GPU 2,5 have 22-25 GB free | Medium |
| Pipeline parallelism 1/8 speed | Only 1 GPU active at a time. TP is 8x faster but can't do INT4 | Inherent |

---

## Dead Ends (don't retry)
1. AWQ (`QuixiAI/Kimi-K2-Base-AWQ`) — broken quant, outputs "!!!!!"
2. FP8 + BnB NF4 — transformers refuses to layer BnB on FP8
3. Attention sharding + padding — NaN from padding, not sharding. Fixed with unmask hook. Sharding itself is safe.
4. `repeat_interleave` for scale expansion — materializes 10+ GB tensor. Use grouped reshape+broadcast instead.
5. OOM cleanup inside except block — Python thread state roots exception, gc can't collect. Must cleanup outside.

## Session Log

### Session 9 (2026-02-21) — Sequential MoE + Model Cache

**Done:**
- Diagnosed batch=2 bottleneck: simultaneous dequant of 3 projections = ~40 GB peak per MoE layer
- Rewrote `_fused_moe` for sequential dequant+compute (gate→free→up→free→down)
- Optimized `_batch_dequantize_int4` with grouped reshape+broadcast
- Fixed OOM handler (outside except, traceback.clear_frames, chained exceptions)
- Wrote model cache save/load (`save_model_cache`, `_fast_load_cached`)
- Added `/model/save` endpoint to server
- All 10 fused MoE unit tests pass
- Fired eval on old server — got baselines for 4 traits, no coefficient results (batch=1 too slow)

**Server restart with new code:**
- `from_pretrained`: 1490s (24.8 min) — disk cache warm, shards loaded in 6:40 but post-processing takes 18 min
- MoE fusion: 22s (60 layers)
- Model cache saved: 594.2 GB in 943s (15.7 min) — 8 GPU shards to `~/.cache/huggingface/fused_cache/`
- Next restart should use cache and skip from_pretrained entirely (untested)

**Steering eval completed (task 08d25b5b):**
- 4 traits, 4 layers [15,20,25,30], subset=5, 5 search steps
- Batch=80 throughout, no OOMs (sequential MoE worked)
- Total time: 5904s (~98 min), 9 generation calls
- GPU 4 bottleneck (only 3-6 GB free during generation)

| Trait | Baseline | Best Layer | Delta | Coherence |
|-------|----------|-----------|-------|-----------|
| deception | 10.0 | L15 | +61.1 | 74.4 |
| anxiety | 58.6 | L25 | +26.1 | 70.7 |
| confidence | 15.1 | L25 | +3.2 | 83.5 |
| guilt | 20.3 | L30 | +60.8 | 76.8 |

Deception, anxiety, guilt steer strongly. Confidence vectors are weak.

### Session 10 (2026-02-21) — More Steering + Capture Endpoint

**Steering evals:**
- Lying: task 5c48e810, layers [15,20,25,30], batch=20 (1 trait). L30 +67.1
- 4-trait batch: task f5284156, concealment+eval_awareness+obedience+rationalization, layers [15,20,25,30,35,40,45,50], batch=100. All steer successfully.
- Only 5 layers had vectors per trait (not all 8 requested). 20 configs × 5 prompts = 100/step.
- Total: 9 of 11 traits now steered on Kimi K2.

**`/capture` endpoint added (code, not yet deployed):**
- `inference/capture_raw_activations.py`: added `prompt_ids` param for filtering specific response files
- `other/server/app.py`: added `POST /capture` endpoint with `asyncio.to_thread` (non-blocking)
- Needs server restart to deploy

**Next: capture secret_number_audit activations**
- 22 rollouts selected: runs 36 (7 fab), 37 (3 fab + 3 honest), 41 (7 honest + 2 fab)
- File IDs: 71,73,74,75,76,79,80,85,86,87,88,89,90,91,92,93,94,95,96,97,99,100
- 49.5K tokens total, ~2K each. Layers 10,15,20,25,30,35,40,45,50 (9 layers)
- Plan to use captured activations for massive_activations calibration too (skip separate calibration)
