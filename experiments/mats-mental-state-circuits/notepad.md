# Kimi K2 Experiment Notepad

## Machine
- 8x NVIDIA H200 (143GB each, 1.12TB total VRAM)
- 2.6TB disk
- Started: 2026-02-19

## Model Status
- **Base:** `unsloth/Kimi-K2-Base` (FP8, 959GB on disk) — WORKS with both pipeline parallelism and TP
- **Instruct:** `moonshotai/Kimi-K2-Thinking` (INT4, 554GB) — downloaded, untested
- **Deleted:** `QuixiAI/Kimi-K2-Base-AWQ` (broken quant), `moonshotai/Kimi-K2-Base` (redundant FP8)

## Progress (as of session 8, 2026-02-20)
- [x] Download models
- [~] Extract vectors (FP8 Base, 11 traits, layers 9-36) — **5 have responses, 0 vetted properly, 4 have unvetted vectors**
  - HAVE RESPONSES: anxiety (vetted), confidence, guilt, rationalization, obedience (all have unvetted vectors)
  - NO RESPONSES: eval_awareness, deception, lying, concealment, conflicted, ulterior_motive
  - All 4 vectored traits verified NaN-free (56 vectors each)
- [x] **NaN root cause SOLVED** — padding, not attention sharding (session 7)
- [x] **Unmask-padding hook** — `install_unmask_padding_hook()` in `utils/model.py`, auto-installed in TP path
- [x] **Attention sharding RE-ENABLED** — 4 entries, saves ~8 GB/GPU
- [x] **empty_cache fix** — between stage 1→3 in `run_pipeline.py`
- [x] **KV estimate fix** — MLA head dims + TP awareness in `estimate_kv_cache_gb()`
- [x] **overhead_factor tuned** — 1.5 → 1.15
- [ ] Relaunch extraction WITH vetting for all 11 traits
- [ ] Massive activations calibration
- [ ] Steering evaluation (Thinking, 11 traits × 6 layers)
- [ ] Capture activations (3 prompt sets, 7 layers)
- [ ] Project onto trait vectors
- [ ] Sync to R2

## Session 2 — TP Investigation & Testing

### What We Did
1. **Investigated HF tensor parallelism** for DeepSeek-V3/Kimi K2
2. **Found 5 flaws** in our original custom tp_plan (see PLAN_tensor_parallel_context.md)
3. **Discovered HF already has `base_model_tp_plan`** in configuration_deepseek_v3.py — no custom dict needed
4. **Verified locally** (transformers 4.57.3, PyTorch 2.10):
   - `tp_plan="auto"` picks up the built-in plan automatically
   - Plan uses `local_colwise`/`local_rowwise`/`local`/`gather` (FP8-safe, no DTensor)
   - `colwise_rep` for lm_head (but lm_head dropped from plan due to post_init() overwrite bug)
   - Config loads correctly: vocab_size=163840, num_attention_heads=64, etc.
   - Model structure correct on meta device (dense layer 0, MoE layers 1-60)
5. **Wrote test script** at `experiments/mats-mental-state-circuits/test_tp.py`
6. **Ran TP test** — `torchrun --nproc-per-node 8 test_tp.py`:
   - Model loaded in ~120s as `DeepseekV3ForCausalLM` with 24 tp_plan entries
   - Memory: ~133GB evenly across all 8 GPUs (vs pipeline: 104-132GB uneven)
   - Forward pass: output logits [1, 5, 163840], predicted "Paris" correctly
   - **Hooks: shape [1, 5, 7168] — full hidden dim, not sharded. All ranks identical.**
   - Generation: "The capital of France is Paris." — coherent
   - FP8 block quantization handled by HF's fp8 quantizer + local_* strategies

### TP Test Script Details
- Location: `experiments/mats-mental-state-circuits/test_tp.py`
- Launch: `torchrun --nproc-per-node 8 experiments/mats-mental-state-circuits/test_tp.py`
- Key requirements:
  - `dist.init_process_group("nccl")` before anything else
  - `trust_remote_code=True` for tokenizer ONLY (not model)
  - Do NOT pass `device_map` (mutually exclusive with `tp_plan`)
  - Do NOT pass `torch_dtype=torch.bfloat16` (let FP8 stay as-is)
  - DynamicCache monkeypatch needed for `generate()`

### Bugs Fixed During Testing
1. `dist.init_process_group("nccl")` — must be called explicitly
2. `total_mem` → `total_memory` — PyTorch API change
3. Tokenizer needs `trust_remote_code=True` separately from model
4. `tokenizer.decode(next_token)` → `tokenizer.decode([next_token.item()])` — custom tokenizer expects list
5. GPU memory reporting: each rank can only see its own GPU, use `dist.all_gather_object`

### Key Findings
- **Nobody has done HF native TP for DeepSeek-V3 before** — we're first
- **`tp_plan` only accepts `"auto"`** — custom dicts rejected by from_pretrained
- **`base_model_tp_plan` is auto-merged** in post_init() — no manual wiring needed
- **`lm_head` missing from merged plan** — class-level `_tp_plan` overwritten by `config.base_model_tp_plan` in post_init(). Stays replicated (~2.2GB/GPU, negligible)
- **Attention stays replicated** — MLA num_heads hardcoded, can't shard with tp_plan dict alone
- **Expected speedup: 3-5x** for generation (MoE sharded, attention replicated)

## TP Test Results (CONFIRMED — `/tmp/tp_test_final.txt`)
```
=== TP Test: 8 GPUs ===
  Loaded in 125.7s — DeepseekV3ForCausalLM, 24 tp_plan entries
  Memory: 133.8GB evenly across all 8 GPUs
  Forward pass: 3.141s, logits [1, 5, 163840], next token: "Paris"
  Hook test: shape [1, 5, 7168] — PASS (full hidden dim, not sharded)
  All 8 ranks see identical activations — PASS
  Generation: "The capital of France is Paris." — 3.98s (3 tokens)
=== All tests complete ===
```

## Session 3 — TP Pipeline Integration + lm_head Fix

### lm_head Sharding Bug Fix
- **Bug**: `modeling_utils.py:2142` overwrites class-level `_tp_plan = {"lm_head": "colwise_rep"}` with `config.base_model_tp_plan` (MoE entries only). lm_head was replicated (~2.35 GB BF16 per GPU).
- **Fix** (1 line in `.venv/lib/python3.11/site-packages/transformers/modeling_utils.py:2142`):
  ```python
  # Before (overwrites):
  self._tp_plan = self.config.base_model_tp_plan.copy() if self.config.base_model_tp_plan is not None else {}
  # After (merges — preserves class-level entries like lm_head):
  self._tp_plan = {**(type(self)._tp_plan or {}), **(self.config.base_model_tp_plan.copy() if self.config.base_model_tp_plan is not None else {})}
  ```
- **Result**: 25 tp_plan entries (was 24). lm_head sharded with `colwise_rep`. Saves ~2 GB/GPU.
- **lm_head dtype**: BF16 in unsloth/Kimi-K2-Base (no `weight_scale_inv` — not FP8 quantized). 7168 × 163840 = 1.17B params = 2.35 GB.
- **NOTE**: This fix is in the installed transformers package, not our repo. Will be lost on `pip install`. Could upstream as PR to huggingface/transformers.

### TP Pipeline Integration
Files modified:
1. **`utils/model.py`** — Added `is_tp_mode()`, `get_rank()`, `is_rank_zero()`, `tp_barrier()` helpers. Modified `load_model_with_lora()` to detect torchrun (`WORLD_SIZE` env var) and use `tp_plan="auto"` instead of `device_map="auto"` + `trust_remote_code=True`.
2. **`extraction/generate_responses.py`** — Gate file saves (pos.json, neg.json, metadata.json) to rank 0 only.
3. **`extraction/extract_activations.py`** — Gate torch.save and metadata saves to rank 0 only.
4. **`extraction/extract_vectors.py`** — Gate vector saves and metadata to rank 0 only.
5. **`extraction/run_pipeline.py`** — Init dist early, suppress prints on non-rank-0, add `tp_barrier()` after GPU stages (1 and 3) before file-reading stages.

### Key Design Decisions
- All ranks run all stages (including CPU-only stages 4, 6) — simpler than gating
- File saves rank-0 only — prevents race conditions
- `tp_barrier()` between stage 1→3 and 3→4 — ensures rank 0 files are written before other ranks read them
- Print suppression via `builtins.print` monkey-patch in `run_pipeline()` — avoids 8x duplicate output
- `dist.init_process_group("nccl")` called in `run_pipeline()` before rank check (not in `load_model_with_lora()`) to ensure rank is known early

### Extraction Run — FAILED (NCCL timeout)
- Model loaded fine (2.3 min, 25 tp_plan entries with lm_head fix)
- Started generating rationalization responses (batch_size=10, 4.6 GB free/GPU)
- **Crashed after ~10 min**: NCCL ALLREDUCE timeout (600s). Ranks diverged — rank 3 on SeqNum=4, others on SeqNum=5.
- **Root cause**: OOM recovery in `utils/generation.py:generate_batch()` halves batch_size on OOM. If one rank OOMs but others don't, they call `model.generate()` different numbers of times → different NCCL all-reduce counts → deadlock.
- **Fix applied**: Synced batch size + OOM recovery across TP ranks in both batch loops:
  - `utils/generation.py:generate_batch()` — initial batch_size synced via `all_reduce(MIN)`, OOM recovery synced via `all_reduce(MAX)` on oom_flag
  - `extraction/extract_activations.py:extract_from_responses()` — same pattern
  - If ANY rank OOMs, ALL ranks halve batch_size and retry together

### Attention Sharding Investigation
- MLA `.view()` calls use `-1` for num_heads dim → auto-adjusts if q_b_proj/kv_b_proj outputs are sharded
- **NOT a modeling code change** — just add to tp_plan:
  ```python
  "layers.*.self_attn.q_b_proj": "colwise",
  "layers.*.self_attn.kv_b_proj": "colwise",
  "layers.*.self_attn.o_proj": "rowwise",
  ```
- Would save ~6 GB/GPU (attention replicated → sharded)
- Risk: MLA's compressed KV structure might cause issues with standard colwise sharding
- The small bottleneck projections (q_a_proj, kv_a_proj_with_mqa) stay replicated

### Thinking Model TP Investigation
- `model_type: "kimi_k2"` (not `"deepseek_v3"`) → needs override to use native HF class
- No `base_model_tp_plan` in config → gets one from native DeepseekV3Config after model_type override
- INT4 compressed-tensors quantization only applies to MoE experts, which get `local` (whole per GPU) → no pack-alignment issue
- Attention, shared experts, dense MLP, lm_head are NOT INT4 → standard TP sharding works
- Config architecturally identical to Base (hidden_size, num_layers, num_heads all match)
- Approach: override model_type to "deepseek_v3", strip auto_map, load with native class + tp_plan="auto"
- Must test before relying on it

## Session 4 — Attention Sharding Investigation + NaN Discovery

### Attention Sharding — PASSES SHORT TESTS, FAILS ON LONG SEQUENCES
- **Previous attempt (session 3)**: 3 entries (q_b_proj, kv_b_proj, o_proj) → crashed with `cudaErrorIllegalAddress`
- **Root cause of crash**: Missing `layers.*.self_attn: gather` — o_proj partial outputs never got all_reduced → corrupted residual stream
- **4-entry fix** injected into `config.base_model_tp_plan` before `from_pretrained`:
  ```python
  "layers.*.self_attn.q_b_proj": "local_colwise",
  "layers.*.self_attn.kv_b_proj": "local_colwise",
  "layers.*.self_attn.o_proj": "local_rowwise",
  "layers.*.self_attn": "gather",
  ```
- **Test results** (`test_tp.py` with 5-token prompt — PASSES):
  ```
  tp_plan entries: 33 (was 25 without attention sharding)
  Memory: 125.6 GB/GPU (was 133.8 GB) — 8.2 GB freed per GPU
  Free VRAM: ~17 GB/GPU (was ~9 GB)
  Forward pass: 3.94s, logits [1, 5, 163840], next token: "Paris"
  Hook: [1, 5, 7168] — PASS (full hidden dim, all ranks identical)
  Generation: "The capital of France is Paris." — 3.74s (3 tokens)
  ```
- **Extraction with attention sharding — PRODUCES NaN**:
  - Ran 8-trait extraction with attention sharding. Generation worked fast (batch_size 23-34!)
  - Rationalization: 173s generation, 145s extraction — 5.3 min/trait
  - **BUT: 81.5% of activation samples contain all-NaN values** (88/108 samples)
  - NaN is per-sample (entire sample is NaN or entire sample is clean), not per-position
  - Consistent across ALL layers 9-36 — not accumulating, some samples just always NaN
  - Both mean_diff AND probe vectors are NaN (activations themselves are corrupted)
  - Comparison: anxiety/guilt/confidence (extracted with pipeline parallelism) have ZERO NaN
  - **Root cause hypothesis**: FP8 attention with sharded heads may lose precision during all_reduce SUM on longer sequences. Or MLA's compressed KV structure doesn't tolerate head sharding. Short sequences (5 tokens) work fine; longer sequences (~100+ tokens) produce NaN.
- **Status**: Attention sharding REVERTED in `utils/model.py`. Comment explains why. Corrupted data cleaned.
- **Diagnostic script**: `experiments/mats-mental-state-circuits/test_tp_nan_diagnostic.py` — tests padded batches with hooks at layers 1, 15, 33. Not yet run successfully (port conflicts).

### Key Technical Finding
- Pre-quantized FP8 models go through Path A (`shard_and_distribute_module` → `get_tensor_shard`), NOT Path B (FP8 quantizer's `create_quantized_param`). This is because `param_needs_quantization()` returns False when `pre_quantized=True`. Same path as MoE experts — so the sharding mechanism itself is correct.
- Block alignment verified: all sharded attention weight dims divisible by 128 (FP8 block size).
- `GatherParallel._prepare_output_fn` correctly handles attention's tuple return `(attn_output, attn_weights)` by only all_reducing `outputs[0]`.
- HF's own TODO in `configuration_deepseek_v3.py:135`: "only replicate attention layers when > first_k_dense_replace" — they know attention should be sharded but haven't done it. Our investigation shows why.

### Extraction Runs
1. **MoE-only TP, no attention sharding** — Started, loaded in 20.7 min, batch_size=3 (5.8 GB free). Killed after trait 1 started to apply attention sharding. No output saved.
2. **With attention sharding (4 entries)** — Loaded in 2.1 min (disk cache hot), batch_size 23-34. Completed rationalization + obedience. **Data corrupted with NaN.** Cleaned.
3. **Need to restart** MoE-only TP extraction for all 8 traits.

### Comprehensive TP doc
- `PLAN_tensor_parallel_deep_dive.md` (361 lines) — covers GPU basics, colwise/rowwise sharding, DTensor vs local, FP8 block quantization, MoE expert parallelism, MLA attention, NCCL sync, and the full attention sharding investigation.

## Session 6 — NaN Root Cause Analysis (Code Reading)

### Attention Mask Analysis
Read `DeepseekV3Attention.forward()` and `eager_attention_forward()` in HF source:
- **Mask shape**: `(batch, 1, query_len, kv_len)` — the `1` broadcasts over heads. Broadcasting works regardless of 64 or 8 heads. **Mask shape is NOT the issue.**
- **No NaN handling after softmax**: Line 275 does `softmax(attn_weights, dim=-1, dtype=float32)` with no NaN guard. If all positions in a row are masked (-inf), softmax produces NaN. But this would only affect pad-token positions, not entire samples.
- **`repeat_kv`**: MLA has `num_key_value_groups = num_attention_heads // num_key_value_heads`. With sharded heads, both numerator and denominator should shrink proportionally... **unless `num_key_value_groups` is stored as a fixed integer from config.** Need to check if it's recalculated after sharding.
- **`.view(-1, ...)`**: Lines 390-391 use `-1` for heads dim → auto-adjusts correctly for sharded output.
- **`k_rot` expansion** (line 406): `k_rot = k_rot.view(batch_size, 1, seq_length, ...)` then `.expand(*k_pass.shape[:-1], -1)` — expands to match sharded k_pass heads. Should work.

### Updated Hypothesis
The mask and view operations look correct. New suspect: **`repeat_kv` with wrong `num_key_value_groups`**.
- `self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads` (line 334)
- With MLA, num_key_value_heads == num_attention_heads (MLA doesn't use GQA) → groups = 1 → `repeat_kv` is a no-op
- BUT if they differ, and TP shards num_attention_heads but NOT num_key_value_heads, the group calculation would be wrong (e.g., 8 instead of 1), causing key/value states to be repeated incorrectly.
- Need to verify: what are `num_attention_heads` and `num_key_value_heads` for Kimi K2?

### Alternative hypothesis: generation-specific
- Extraction generates responses first (`model.generate()`), then runs a forward pass for activation capture
- If `model.generate()` produces garbage tokens (`!!!!`), the forward pass on garbage input might produce NaN
- But session 1 traits (pipeline parallelism) generated coherent text with same scenarios
- So the generation itself is broken by attention sharding, not just the activation capture

### Diagnostic Scripts
- `test_tp_diag.py` — batch=2 different lengths, forward-only, checks NaN at layers 1/15/33
- `test_tp_nan_diagnostic.py` — same but more verbose (20 positions printed)
- `test_tp_padding.py` — comprehensive 6-test suite (batch 1/2/4, short/long/mixed)
- **None have been run successfully** — orphaned CUDA contexts blocked all GPU operations

## Session 7 — NaN Root Cause SOLVED (2026-02-20)

### Root Cause: Padding + Attention, NOT attention sharding
Ran 4-step diagnostic proving **NaN comes from padding, not TP/sharding**:

1. **test_tp_diag.py** (attention sharding): padded sample → ALL NaN at all layers. Confirmed.
2. **test_tp_unsharded_attn.py** (MoE-only TP, NO sharding): **IDENTICAL NaN**. 25 tp_plan entries, same NaN at same positions.
3. **test_tp_nan_trace.py**: Traced NaN origin precisely:
   - `embed_tokens` → clean (pad token norm=0.86, no NaN)
   - `layer0 input_layernorm` → clean
   - **`layer0 attention output` → 30/35 NaN (exactly pad positions)**
   - `layer0 out` → 30/35 NaN (inherited)
   - **`layer1 attention output` → 35/35 NaN (ALL — NaN spread to real tokens)**
4. **Root cause verified** with isolated PyTorch tests:
   - Layer 0: pad query tokens have ALL KV positions masked → `softmax([-inf, ...])` = NaN
   - Layer 1: NaN pad residuals → NaN K/V → `Q_clean @ K_nan^T` = NaN scores
   - `NaN + (-inf from mask) = NaN` (IEEE arithmetic) → masking cannot fix NaN
   - `softmax([NaN, ..., NaN, good_score])` = NaN → ALL positions contaminated
   - Tested MATH and EFFICIENT_ATTENTION SDPA backends — both fail the same way

### Key discoveries:
- `attn_implementation = sdpa` (not eager) — SDPA is active with 61 calls per forward
- `num_attention_heads = num_key_value_heads = 64` → `repeat_kv` is a no-op (MLA)
- HF had a fix for this (`_unmask_unattended` / `allow_torch_fix`) for torch < 2.5, but removed it for torch >= 2.5 assuming SDPA handles it. SDPA handles `softmax(-inf)` correctly, but NOT `softmax(NaN + -inf)`.
- **Attention sharding is SAFE** — the NaN was never caused by sharding. Sessions 3-6 were chasing the wrong cause.

### Implications:
- Can re-enable attention sharding (saves ~8 GB/GPU, batch_size 23-34 vs 3)
- Must eliminate padding: right-padding or batch_size=1
- All 3 traits extracted in session 1 are clean (pipeline parallelism had minimal padding)
- 2 corrupted traits (rationalization, obedience) were corrupted by padding, not sharding

### Full writeup: `nan_investigation_results.md`

## Currently Running
- Nothing (extraction killed to apply KV fix + add vetting)

## Next Steps (Priority Order)
1. Relaunch extraction WITH vetting for all 11 traits (5 have responses, 6 need generation)
2. Massive activations calibration
3. Steering evaluation (Thinking, 11 traits × 6 layers)
4. Inference capture (3 prompt sets, 7 layers)
5. Project onto trait vectors
6. Sync to R2

## Session 8 — empty_cache Fix + Verification (2026-02-20)

### GPU Memory Fix
- **Problem**: CUDA allocator hoards generation buffers between stage 1 (generate) and stage 3 (extract). Extraction sees ~5.5 GB free instead of ~10 GB, OOMs batch from 23 → 11.
- **Fix**: Added `gc.collect()` + `torch.cuda.empty_cache()` after stage 1 in `run_pipeline.py:229-232`.
- **Audit**: Checked all GPU memory patterns across codebase. Only this spot was missing — `extract_activations.py`, `capture_raw_activations.py`, and steering evaluation all clean up properly.

### NaN Verification
- Verified all 4 extracted traits (anxiety, guilt, confidence, rationalization) have 0 NaN across 56 vectors each.
- Rationalization was the first trait extracted with the unmask hook — confirms fix works end-to-end.

### Batch Size Tuning
- `utils/generation.py:487`: overhead_factor 1.5 → 1.15

### KV Cache Estimate Fix (MLA + TP)
- **Root cause of 5.5 GB issue**: `estimate_kv_cache_gb()` used wrong head dimensions for DeepSeek V3 MLA
- Old: `head_dim=64` (from `config.head_dim = qk_rope_head_dim`), didn't know about MLA's K=192/V=128
- Also didn't account for TP dividing KV heads across GPUs (64 heads ÷ 8 GPUs = 8 heads/GPU)
- Old estimate: 87 MB/seq (3.2x too high for TP). Fixed: 27 MB/seq (matches actual)
- **Investigation confirmed**: `generate()` does NOT store past_key_values on model. DynamicCache dies with `_sample()` frame. No persistent causal mask, no SDPA state, no TP buffers. The 6 GB gap is CUDA allocator fragmentation from oversized batches.
- Fix: `utils/generation.py:370-390` — detects MLA config attrs + TP attention sharding
- Full writeup: `memory_retention_analysis.md`

### Extraction Run 2 (killed for vetting + KV fix)
- Completed: obedience (generation 279s + extraction 130s + vectors 0.3s = ~7 min)
- Batch sizes: generation 59/31, extraction 39→OOM→19 then 7
- eval_awareness: positives generated (300), negatives in progress when killed
- 5 traits not started (deception, lying, concealment, conflicted, ulterior_motive)

### Current Trait Status
- **Have responses + vectors (unvetted):** anxiety (also vetted), confidence, guilt, rationalization, obedience
- **No responses:** eval_awareness, deception, lying, concealment, conflicted, ulterior_motive

### Files Modified
- `extraction/run_pipeline.py` — Added empty_cache between stage 1 and stage 3
- `utils/generation.py` — KV estimate fix for MLA+TP, overhead_factor 1.5 → 1.15
- `docs/main.md` — Updated troubleshooting section
- `experiments/mats-mental-state-circuits/memory_retention_analysis.md` — NEW: KV cache investigation
- `experiments/mats-mental-state-circuits/notepad.md` — Updated progress + session 8

## Files Modified Session 7
- `utils/model.py` — Added `install_unmask_padding_hook()`, installed in TP path
- `experiments/mats-mental-state-circuits/test_tp_diag.py` — Updated with unmask hook + generation test
- `experiments/mats-mental-state-circuits/test_tp_unsharded_attn.py` — NEW
- `experiments/mats-mental-state-circuits/test_tp_nan_trace.py` — NEW
- `experiments/mats-mental-state-circuits/nan_investigation_results.md` — NEW

## Files Modified Session 3
- `utils/model.py` — Added `is_tp_mode()`, `get_rank()`, `is_rank_zero()`, `tp_barrier()`. TP branch in `load_model_with_lora()`.
- `extraction/generate_responses.py` — Rank-0 file saves
- `extraction/extract_activations.py` — Rank-0 file saves
- `extraction/extract_vectors.py` — Rank-0 file saves
- `extraction/run_pipeline.py` — Dist init, print suppression, tp_barrier after stages 1 and 3
- `utils/generation.py` — TP-synced batch size + OOM recovery in `generate_batch()`
- `extraction/extract_activations.py` — TP-synced batch size + OOM recovery in `extract_from_responses()`
- `.venv/.../transformers/modeling_utils.py:2142` — lm_head tp_plan merge fix (NOT in repo, in installed package)

## What Worked (Session 1)
- FP8 direct loading: 249s (4 min), ~120GB/GPU, 30GB headroom each
- Generation test: "The capital of France is Paris." — correct output
- Extraction pipeline: anxiety trait completed (60+60 responses, 456s generation, 79s activations, 3s vectors)
- Batch size: 40-43 for generation, 59-70 for extraction

## What Failed & Why
1. **AWQ (`QuixiAI/Kimi-K2-Base-AWQ`)**: Community quant, broken. Output "!!!!!". Even vLLM can't load it (tensor size mismatch).
2. **FP8 + BnB NF4**: Transformers refuses to layer BnB on already-FP8 weights.
3. **Unsloth + BnB NF4**: Same FP8 weights underneath → `Blockwise 4bit quantization only supports 16/32-bit floats, but got torch.float8_e4m3fn`
4. **Native DeepseekV3ForCausalLM**: Timed out during loading (session 1). Works fine with TP (session 2).
5. **DynamicCache API**: Kimi K2's custom `modeling_deepseek.py` uses `seen_tokens`, `get_usable_length`, `get_max_length` — all removed in transformers 5.x. Fixed via monkeypatch in `utils/model.py`.
6. **Pipeline parallelism**: Only 1/8 GPUs active at a time. ~8 min/trait for extraction. Unacceptable at $20/hr.
7. **Meta device offloading**: Despite `max_memory` fix, FP8→bfloat16 expansion (~1.9TB) causes tail layers to offload. Fixed by using TP instead (no dtype cast, FP8 stays as-is).
8. **Logit lens**: Crashes on FP8 model (lm_head on meta device). Use `--no-logitlens`.
9. **Attention sharding without gather (session 3)**: 3 entries crashed with `cudaErrorIllegalAddress`. Missing `gather` on `self_attn`.
10. **Attention sharding WITH gather (session 4-5)**: 4 entries (including gather). Short test passes (5-token "Paris" test). But extraction produces 81.5% NaN activation samples on 100+ token sequences. Root cause unknown — possibly FP8 precision loss in sharded attention all_reduce, or MLA KV compression incompatible with head sharding. **Reverted to MoE-only TP.**

## Key Config
- `config.json`: `kimi_k2_base` → `unsloth/Kimi-K2-Base`
- `utils/model.py`: DynamicCache monkeypatch (lines 40-55), max_memory fix for multi-GPU
- `config/models/kimi-k2-base.yaml`: FP8 config (was AWQ)

## Files Modified This Session
- `experiments/mats-mental-state-circuits/PLAN_tensor_parallel.md` — Updated with correct tp_plan (local_* strategies, no attention sharding, tp_plan="auto")
- `experiments/mats-mental-state-circuits/PLAN_tensor_parallel_context.md` — Added investigation findings, resolved FP8/MoE sections
- `experiments/mats-mental-state-circuits/test_tp.py` — NEW: TP test script
- `experiments/mats-mental-state-circuits/notepad.md` — Updated with session 2 progress
