# Fused MoE Forward — Implementation Status & Next Steps

## What We Built

Replaced the Python expert loop in Kimi K2's MoE (384 experts × 60 layers = 23,040 iterations per forward pass) with batched `grouped_mm`. All code is in `utils/model.py`.

### Functions added to `utils/model.py`:

1. **`_batch_dequantize_int4(packed, scale, original_shape)`** (line ~358) — Batch-dequantizes INT4 packed weights to BF16. Flattens 3D→2D, calls `unpack_from_int32`, multiplies by group scales.

2. **`_fuse_expert_weights(moe_module)`** (line ~390) — Per MoE layer: stacks 384 experts' `weight_packed` and `weight_scale` into contiguous 3D tensors, stores original shapes, frees individual expert params.

3. **`_fused_moe(self, hidden_states, topk_indices, topk_weights)`** (line ~425) — Replacement for `moe_infer()`. Sorts tokens by expert, gets unique active experts + group offsets, indexes stacked INT4 weights, batch-dequantizes, uses `F.grouped_mm` for gate/up/down with SwiGLU, scatter-adds weighted results.

4. **`_patch_moe_forward(model, _print)`** (line ~480) — Entry point called after `from_pretrained`. Detects MoE layers by attributes (not isinstance — works with custom remote-code classes). Fuses weights, monkey-patches `moe_infer` (or `moe` for HF native).

### Loading speedups in `utils/model.py`:

- **`ModelCompressor.compress_model`** — patched to no-op (was already there)
- **`initialize_module_for_quantization`** — NEW: patched to just set scheme attribute, skipping 69K× (14 hasattr + register_parameter + closure creation). Safe for `run_compressed=True` inference.
- **Timing print** after `from_pretrained` returns — shows exact load time.

### Test file: `core/_tests/test_fused_moe.py` (10 tests, all pass)

- 3 dequantize tests (basic, multi-expert, different group sizes)
- 4 routing tests vs naive loop reference (basic, single token, all-same-expert, top-8 with 64 experts)
- 3 weight fusion tests (shapes, cleanup, data integrity)

### Smoke test: `scripts/test_fused_moe_smoke.py`

- Loads Kimi K2, fuses, generates 50 tokens, measures tok/s
- Saves fused model to `experiments/kimi-k2-fused/` if not FAIL
- Saves results JSON to `experiments/fused_moe_smoke_results.json`
- Traceback wrapper for clean error reporting
- Run with: `python -u scripts/test_fused_moe_smoke.py`

## What's Validated

- **Synthetic tests:** All 10 pass. Dequant matches manual pack/unpack. Fused routing matches naive expert loop to <2% relative error.
- **`torch.stack` benchmark:** 384 tensors × Kimi K2 shapes = 12ms per stack. Full fuse estimated <1 min.
- **`grouped_mm`:** Confirmed available in torch 2.10.
- **`unpack_from_int32`:** Confirmed import path.

## What's NOT Validated (needs one real model run)

- Does `from_pretrained` complete with all patches? (Previously killed at ~15 min; the new `initialize_module_for_quantization` patch should help)
- Does attribute-based MoE detection find the 60 MoE layers?
- Does fused generation produce correct output?
- What's the actual tok/s? (Target: >2 tok/s, was ~1 tok/s unfused)
- VRAM stability during generation

## Known Issues & Context

### Why previous test runs "hung"
1. First runs: `isinstance(DeepseekV3MoE)` check failed (Kimi K2 uses custom remote-code class, not HF native). Detection found 0 layers → fell through to unfused generation at ~1 tok/s.
2. Fixed to attribute-based detection + correct method name (`moe_infer` not `moe`).
3. We kept killing `from_pretrained` at ~15 min thinking it was stuck. It was actually still running — `dispatch_model` (accelerate) iterates all 70K+ modules for hook attachment. Should complete at ~18-20 min.

### `from_pretrained` bottleneck analysis
Post-shard-loading (after the progress bar), `dispatch_model` does:
- `find_tied_parameters`: 2× full `named_parameters()` sweeps (200K+ params)
- `attach_align_device_hook_on_blocks`: recursive hook attachment to all 70K+ modules
- This is pure Python overhead, not GPU work

Pre-shard, `apply_quantization_config` iterates 69K modules for CompressedLinear setup. The new `initialize_module_for_quantization` patch should cut this significantly.

Key source locations:
- `transformers/modeling_utils.py:5140` — `dispatch_model` call
- `accelerate/big_model.py:369-447` — `dispatch_model` implementation
- `compressed_tensors/quantization/lifecycle/apply.py:147-183` — `apply_quantization_config`
- `compressed_tensors/quantization/lifecycle/initialize.py:61-143` — `initialize_module_for_quantization` (patched)

### Model details
- `moonshotai/Kimi-K2-Thinking`: 61 layers, 60 MoE (layer 1-60), 384 routed experts, top_k=8
- INT4 compressed-tensors, ~594 GB on disk, ~74 GB/GPU across 8x H200
- Pipeline parallel with custom device_map (7-8 layers per GPU)
- Expert shapes: gate/up `[2048, 7168]`, down `[7168, 2048]`, group_size=32

## Next Steps

1. **Run smoke test:** `python -u scripts/test_fused_moe_smoke.py` — 30 min timeout, answers all unknowns
2. **If PASS:** Model saved to `experiments/kimi-k2-fused/`. Run steering eval smoke test.
3. **If load too slow:** Investigate `dispatch_model` patch (skip recursing into single-device blocks)
4. **If detection fails:** Check smoke test output for "MoE fusion: no layers" message
5. **If generation errors:** Traceback in output, debug from there
6. **Future:** Serve fused model via `server/app.py` for instant iteration
