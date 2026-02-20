# Tensor Parallelism Deep Dive: DeepSeek V3 / Kimi K2 on 8x H200

## GPU Cluster Basics

### The Problem
Kimi K2 (DeepSeek V3 architecture) has ~671B parameters. In FP8 (1 byte each), that's ~671 GB on disk, expanding to ~960 GB with metadata/scales. One H200 has 143 GB VRAM. You can't fit the model on one GPU.

### Pipeline Parallelism (what we had before)
Each GPU holds a range of layers. GPU 0 has layers 0-7, GPU 1 has 8-15, etc. During inference, tokens flow through GPUs sequentially — only 1/8 GPUs is active at a time. Memory distribution is uneven (104-132 GB across GPUs). Terrible utilization.

### Tensor Parallelism (what we use now)
Each GPU holds a *slice* of every layer. All 8 GPUs work on every token simultaneously. Requires NCCL communication between GPUs (all_reduce operations over NVLink at ~900 GB/s). Much better utilization — all GPUs active for every token.

---

## How Weight Matrices Get Split

A linear layer computes `output = input @ weight.T + bias`. The weight is a 2D matrix `[out_features, in_features]`.

### Colwise (column-wise) — split the output dimension
```
Full weight:        [12288, 1536]     (64 heads × 192 dim/head, input 1536)
GPU 0 gets:         [1536, 1536]      (heads 0-7)
GPU 1 gets:         [1536, 1536]      (heads 8-15)
...
GPU 7 gets:         [1536, 1536]      (heads 56-63)
```
Each GPU computes a SLICE of the output independently. No communication needed for the matmul itself. Used for projections that fan out (q_proj, k_proj, v_proj, gate_proj, up_proj).

### Rowwise — split the input dimension
```
Full weight:        [7168, 8192]      (output=hidden_size, input=heads×head_dim)
GPU 0 gets:         [7168, 1024]      (processing heads 0-7's contribution)
GPU 1 gets:         [7168, 1024]      (processing heads 8-15's contribution)
...
```
Each GPU computes a PARTIAL result (same output shape but only from its input slice). You need **all_reduce(SUM)** across all GPUs to get the correct final output. Used for projections that fan in (o_proj, down_proj).

### The Colwise → Rowwise Pattern
Standard TP for attention and MLP follows this pattern:
1. **Colwise** on the fan-out projection (split output across GPUs)
2. Each GPU computes locally (attention or activation function)
3. **Rowwise** on the fan-in projection (partial sums)
4. **All-reduce** to combine partial sums → full output for residual connection

For attention: `q_proj(colwise) → attention(local) → o_proj(rowwise) → all_reduce`
For MLP: `gate_proj/up_proj(colwise) → activation(local) → down_proj(rowwise) → all_reduce`

---

## DTensor vs Local Strategies

PyTorch offers two ways to handle sharded tensors:

### DTensor (`colwise`, `rowwise`)
PyTorch's distributed tensor abstraction. Wraps the tensor with metadata about how it's sharded. Operations on DTensors automatically insert communication (all_reduce, etc.). Clean but doesn't work well with:
- `.view()` calls with computed dimensions
- Custom routing logic (MoE)
- FP8 block quantization (non-standard memory layout)

Used by: Llama, Mistral, and other standard architectures.

### Local (`local_colwise`, `local_rowwise`)
Just a regular `torch.Tensor` that happens to be a slice of the full tensor. No metadata, no automatic communication. You manually add communication via hooks (like `gather` → `all_reduce(SUM)`).

Key difference:
- `colwise` → registers input/output hooks via `distribute_module()` that handle DTensor conversion and all_reduce automatically
- `local_colwise` → `use_dtensor=False` → **NO hooks registered** → you MUST add explicit `gather` on the parent module

Used by: DeepSeek V3 (because MoE routing and FP8 break DTensor).

### Other strategies
- `local` (IsolatedParallel): Keeps full tensor on each GPU but divides values by world_size. Parent's `gather` (all_reduce SUM) restores correct values. Used for MoE expert wrappers.
- `gather` (GatherParallel): Registers output hook that calls `dist.all_reduce(SUM)`. Handles both plain tensors and tuples.
- `colwise_rep`: Colwise sharding with replicated output (output is all-gathered so every GPU has the full output). Used for lm_head.

---

## FP8 Quantization and weight_scale_inv

### Why FP8?
FP8 (float8_e4m3fn) uses 8 bits per value — half the memory of FP16/BF16. Kimi K2's FP8 checkpoint fits on 8x H200 (~134 GB/GPU) while BF16 would need ~1.34 TB (impossible).

### Block Quantization
FP8 has very limited range (max ~448). To preserve precision, the weight matrix is divided into **blocks** (128×128 for Kimi K2). Each block has its own scale factor:

```
weight:           [12288, 1536]  float8_e4m3fn   (1 byte per value)
weight_scale_inv: [96, 12]      float32          (one scale per 128×128 block)
                   ↑    ↑
                   12288/128=96  1536/128=12
```

The actual value is: `real_value = fp8_value * scale_inv[block_row][block_col]`

### FP8 Forward Pass
HuggingFace replaces `nn.Linear` with `FP8Linear` (in `integrations/finegrained_fp8.py`). The forward pass:
1. Quantize input activation to FP8 with per-token-group scales
2. Call triton kernel `w8a8_block_fp8_matmul_triton(qinput, weight, input_scale, weight_scale_inv, block_size)`
3. Kernel processes 128×128 tiles, applying corresponding scales
4. Returns output in the original input dtype (bfloat16)

### FP8 + TP Sharding
When you shard an FP8 weight, you MUST also shard weight_scale_inv correspondingly:
- Colwise (split rows): weight `[12288, 1536]` → `[1536, 1536]`, scales `[96, 12]` → `[12, 12]`
- Block structure preserved: 1536/128 = 12 ✓

HF handles this: `_get_parameter_tp_plan` applies the parent module's plan to ALL parameters (weight, bias, weight_scale_inv).

---

## DeepSeek V3 / Kimi K2 Architecture

### Kimi K2 = DeepSeek V3
Kimi K2 (Moonshot AI) uses the exact same architecture as DeepSeek V3. The Base model has `model_type: "deepseek_v3"` and uses HF's native `DeepseekV3ForCausalLM`. The Thinking model uses `model_type: "kimi_k2"` (needs override for TP).

### MLA (Multi-head Latent Attention)
Standard attention: each of Q, K, V is a full [hidden_size × hidden_size] projection.

MLA compresses through a low-rank bottleneck to save KV cache memory:
```
Queries:  hidden [7168] → q_a_proj [1536] → layernorm → q_b_proj [12288] → 64 heads
Keys/Vals: hidden [7168] → kv_a_proj [512+64] → layernorm → kv_b_proj [16384] → 64 heads of (k_nope + v)
```

The bottleneck projections (q_a_proj: 7168→1536, kv_a_proj: 7168→576) are small and stay **replicated** on every GPU. The big decompression projections (q_b_proj: 1536→12288, kv_b_proj: 512→16384, o_proj: 8192→7168) are candidates for sharding.

The `.view()` calls use `-1` for num_heads, so they auto-adjust when output dims are sharded:
```python
query_shape = (batch_size, seq_length, -1, self.qk_head_dim)  # -1 infers 64 or 8 heads
```

### MoE (Mixture of Experts)
256 routed experts per MoE layer + 1 shared expert. Each token activates 8 of 256 experts.

Under TP: 256 experts / 8 GPUs = 32 experts per GPU. Router decides which experts to use. Each GPU computes its local experts, then `all_reduce(SUM)` combines results.

The shared expert gets sharded like a regular MLP (local_colwise/local_rowwise).

### Layer Structure
- Layer 0: Dense (standard MLP, no MoE)
- Layers 1-60: MoE (256 experts + shared expert)
- All layers: MLA attention (currently replicated under TP)

---

## Current TP Plan (What Works)

```python
# configuration_deepseek_v3.py
base_model_tp_plan = {  # TODO: only replicate attention layers when > first_k_dense_replace
    # Routed experts: distributed across GPUs (32 per GPU)
    "layers.*.mlp.experts.*.gate_proj": "local_colwise",
    "layers.*.mlp.experts.*.up_proj": "local_colwise",
    "layers.*.mlp.experts.*.down_proj": "local_rowwise",
    "layers.*.mlp.experts.*": "local",           # expert wrapper

    # Shared experts: sharded across GPUs
    "layers.*.mlp.shared_experts.gate_proj": "local_colwise",
    "layers.*.mlp.shared_experts.up_proj": "local_colwise",
    "layers.*.mlp.shared_experts.down_proj": "local_rowwise",
    "layers.*.mlp.shared_experts": "local",

    # Dense MLP (layer 0): sharded
    "layers.*.mlp.gate_proj": "local_colwise",
    "layers.*.mlp.up_proj": "local_colwise",
    "layers.*.mlp.down_proj": "local_rowwise",

    # All-reduce for MLP output
    "layers.*.mlp": "gather",   # "This is the only moment where results are gathered"
}
# + class-level: {"lm_head": "colwise_rep"}  (from our merge fix)
```

**Attention is fully replicated** — every GPU has a full copy of all attention weights. This wastes ~6 GB/GPU but avoids the FP8 sharding complexity.

### Memory Under Current TP
- MoE experts (sharded): ~121 GB across all GPUs
- Attention (replicated): ~6 GB per GPU
- Shared experts (sharded): ~3 GB across all GPUs
- Embeddings + lm_head: ~4 GB per GPU
- **Total model: ~134 GB/GPU, ~9 GB free for generation**

---

## Attention Sharding Investigation

### What We Tried
```python
"layers.*.self_attn.q_b_proj": "local_colwise",
"layers.*.self_attn.kv_b_proj": "local_colwise",
"layers.*.self_attn.o_proj": "local_rowwise",
```

### What Crashed
`CUDA error: an illegal memory access was encountered` in `finegrained_fp8.py:350` (the FP8 linear forward pass). Rank 5 crashed first, others followed.

### Root Cause Analysis

**Two separate code paths for loading parameters with TP:**

```python
# modeling_utils.py line 718-730
if device_mesh is not None:
    if not is_quantized or not hf_quantizer.param_needs_quantization(model, param_name):
        # Path A: standard sharding (non-quantized params, or non-quantized model)
        shard_and_distribute_module(model, param, ...)
    else:
        # Path B: quantizer handles sharding (FP8/INT4 weights)
        hf_quantizer.create_quantized_param(model, param, ..., **sharding_kwargs)
```

The FP8 quantizer creates `FP8Linear` modules and loads weights through `create_quantized_param`. The existing MoE entries work because the quantizer knows how to shard expert weights. But our NEW attention entries — the quantizer may not handle them correctly because:

1. `get_tensor_shard` (used internally) has **no FP8 handling**:
   ```python
   # get_packed_weights has explicit FP8 guard:
   if slice_dtype == "F8_E4M3":
       slice_ = slice_[...].to(torch.float16)  # upcast before slicing

   # get_tensor_shard has NO guard:
   return param[tuple(slice_indices)]  # raw FP8 slice → may crash
   ```

2. `param_casting_dtype` is `None` for FP8 weights (explicitly skipped in `_infer_parameter_dtype`)

3. Without the `gather` hook on `self_attn`, the partial outputs from `o_proj` never get all_reduced → corrupted residual stream → cascading errors

### What Was Missing
We needed **4 entries**, not 3:
```python
"layers.*.self_attn.q_b_proj": "local_colwise",
"layers.*.self_attn.kv_b_proj": "local_colwise",
"layers.*.self_attn.o_proj": "local_rowwise",
"layers.*.self_attn": "gather",          # ← MISSING — all_reduce for partial o_proj outputs
```

### Would Gather Fix It?
The `GatherParallel` output hook handles tuples correctly:
```python
def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
    if isinstance(outputs, torch.Tensor):
        dist.all_reduce(outputs, op=dist.ReduceOp.SUM, async_op=False)
    else:
        dist.all_reduce(outputs[0], op=dist.ReduceOp.SUM, async_op=False)  # tuple: reduce first element
```

And the decoder layer unpacks correctly:
```python
hidden_states, _ = self.self_attn(...)  # only attn_output used for residual
hidden_states = residual + hidden_states
```

The `expert_parallel_group` attribute set by gather's input hook is harmless on attention modules (nothing reads it).

### Remaining Risk
Even with `gather`, the FP8 weight sharding itself may crash due to `get_tensor_shard` not handling FP8. This would need either:
1. Patching `get_tensor_shard` to upcast FP8 like `get_packed_weights` does
2. Or ensuring the FP8 quantizer's `create_quantized_param` handles the new tp_plan entries

### Expected Memory Savings
Attention sharding would save ~6 GB/GPU:
- q_b_proj: 12288×1536 FP8 = 18 MB → 2.3 MB per GPU
- kv_b_proj: 16384×512 FP8 = 8 MB → 1 MB per GPU
- o_proj: 7168×8192 FP8 = 56 MB → 7 MB per GPU
- × 61 layers = ~5 GB total savings per GPU (+ scale tensors)

This would allow batch_size ~13 instead of 3 — roughly 4x faster generation.

---

## NCCL Synchronization: When and Why

### What is NCCL?
NVIDIA Collective Communications Library. Handles GPU-to-GPU communication over NVLink (900 GB/s between H200 GPUs). Key operation: `all_reduce` — every GPU contributes a tensor, NCCL sums them, every GPU gets the result.

### When You DON'T Need Explicit Sync
- **DTensor strategies** (`colwise`, `rowwise`): PyTorch automatically inserts all_reduce via DTensor hooks
- **Completely independent work**: Each GPU does its own thing with no shared state
- **Reading from shared files**: File I/O is inherently serialized by the OS

### When You DO Need Explicit Sync
1. **`local_*` strategies**: No automatic communication. Must add `gather` (GatherParallel) on parent module.
2. **File I/O in distributed code**: Only rank 0 should save files. Other ranks must `barrier()` to wait.
3. **OOM recovery**: If one GPU OOMs and halves batch_size, ALL GPUs must agree. Use `all_reduce(MAX)` on an oom_flag tensor.
4. **Any divergent control flow**: All ranks MUST call NCCL operations the same number of times, or you get a 600s timeout and deadlock.

### Our NCCL Timeout Bug
The extraction pipeline crashed because OOM recovery was per-rank: if rank 3 OOMed but others didn't, rank 3 called `model.generate()` fewer times → different numbers of NCCL all_reduce calls → deadlock → 600s timeout.

Fix: After every `model.generate()` or `model()` call, all ranks exchange an oom_flag via `all_reduce(MAX)`. If any rank sets flag=1, all ranks halve batch_size and retry together.

```python
oom_flag = torch.zeros(1, device='cuda')
try:
    output = model.generate(...)
except torch.cuda.OutOfMemoryError:
    oom_flag.fill_(1)
    torch.cuda.empty_cache()

dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
if oom_flag.item() > 0:
    batch_size = max(1, batch_size // 2)
    continue  # all ranks retry with smaller batch
```

---

## HuggingFace TP Implementation Details

### Loading Flow
1. Create model on meta device (empty tensors, just shapes)
2. Replace `nn.Linear` with `FP8Linear` if FP8 quantized
3. For each safetensors shard file:
   a. Open file, iterate parameters
   b. For each param: check if quantized → Path A or Path B
   c. Path A: `shard_and_distribute_module` → `partition_tensor` → `get_tensor_shard`
   d. Path B: `create_quantized_param` with sharding_kwargs
4. After loading: `post_init()` → `distribute_model()` → register forward hooks based on tp_plan
5. Hooks handle runtime communication (input conversion, output all_reduce)

### The `_tp_plan` Merge Bug (Our Fix)
`post_init()` at line 2142 was overwriting class-level `_tp_plan = {"lm_head": "colwise_rep"}` with `config.base_model_tp_plan` (MoE entries only). Fixed to merge:
```python
# Before: self._tp_plan = self.config.base_model_tp_plan.copy() ...
# After:  self._tp_plan = {**(type(self)._tp_plan or {}), **(self.config.base_model_tp_plan.copy() ...)}
```
This preserves the class-level lm_head entry (25 entries instead of 24). Fix is in installed package, not our repo.

---

## Current Status

### What Works
- MoE expert parallelism (256 experts / 8 GPUs)
- Shared expert sharding (local_colwise/local_rowwise)
- Dense MLP sharding
- lm_head sharding (colwise_rep, via our merge fix)
- OOM-synced batch size recovery across all ranks
- File save gating to rank 0

### Attention Sharding — PASSES SHORT TEST, FAILS ON LONG SEQUENCES
- 4 entries: `q_b_proj: local_colwise`, `kv_b_proj: local_colwise`, `o_proj: local_rowwise`, `self_attn: gather`
- **Short test passes** (5-token prompt): 33 tp_plan entries, 125.6 GB/GPU (8.2 GB freed), correct generation
- **Long sequence FAILS**: Extraction produces 81.5% NaN activation samples (88/108 all-NaN)
  - NaN is per-sample, consistent across ALL layers 9-36 (not accumulating)
  - 20 clean samples have normal norms (~22-30)
  - Both mean_diff and probe vectors are NaN (activations themselves corrupted)
  - Comparison: anxiety/guilt/confidence (pipeline parallelism) have ZERO NaN
- **Root cause hypothesis**: FP8 precision loss during sharded attention all_reduce on longer sequences, or MLA's compressed KV structure (kv_a_proj bottleneck → kv_b_proj expansion) doesn't tolerate head sharding when KV cache grows
- **Status**: REVERTED in `utils/model.py`. Using MoE-only TP (attention fully replicated)
- Root cause of prior `cudaErrorIllegalAddress` crash: missing `gather` — that's now fixed but NaN remains

### What Doesn't Work Yet
- Attention sharding (NaN on long sequences — see above)
- Thinking model TP (needs model_type override from "kimi_k2" to "deepseek_v3")

### Extraction Status
- 3/11 traits complete (anxiety, guilt, confidence) — from session 1 with pipeline parallelism
- 8 traits remaining — need MoE-only TP run (batch_size ~3, slower but correct)

---

## References
- HF config: `.venv/.../transformers/models/deepseek_v3/configuration_deepseek_v3.py`
- HF TP code: `.venv/.../transformers/integrations/tensor_parallel.py`
- FP8 code: `.venv/.../transformers/integrations/finegrained_fp8.py`
- Model code: `.venv/.../transformers/models/deepseek_v3/modeling_deepseek_v3.py`
- Our TP helpers: `utils/model.py` (is_tp_mode, get_rank, is_rank_zero, tp_barrier)
- Our OOM sync: `utils/generation.py` and `extraction/extract_activations.py`
- HF Issue #35425: DeepSeek V3 Support
- HF TODO in config: "only replicate attention layers when > first_k_dense_replace"
