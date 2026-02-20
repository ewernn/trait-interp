# Tensor Parallelism for DeepSeek-V3 / Kimi K2

## Why
`device_map="auto"` does pipeline parallelism — only 1/8 GPUs active at any time. Generation is ~8x slower than it should be. Extraction takes ~10 min/trait, steering eval would take hours. At $20/hr this is unacceptable.

## What
Just use `tp_plan="auto"` with HF's native `DeepseekV3ForCausalLM`. HF transformers already has:
- Native `DeepseekV3ForCausalLM` class (since March 2025)
- A complete MoE TP plan in `config.base_model_tp_plan` — auto-merged in `post_init()`
- All needed strategies: `local_colwise`, `local_rowwise`, `local` (IsolatedParallel), `gather`
- **No custom tp_plan dict needed** — `tp_plan="auto"` picks it all up
- **`tp_plan` and `device_map` are mutually exclusive** — can't use both

## Architecture (from modeling_deepseek.py)

### MLA Attention — NOT SHARDED
```
q_a_proj:           hidden(7168) → q_lora_rank(1536)           # replicated
q_a_layernorm:      RMSNorm(1536)                              # replicated
q_b_proj:           q_lora_rank(1536) → heads×head_dim(12288)  # replicated
kv_a_proj_with_mqa: hidden(7168) → kv_lora_rank+rope(576)     # replicated
kv_a_layernorm:     RMSNorm(512)                               # replicated
kv_b_proj:          kv_lora_rank(512) → heads×(nope+v)(16384)  # replicated
o_proj:             heads×v_head_dim(8192) → hidden(7168)      # replicated
```

**Why not shard attention**: The native HF code does `q.view(bsz, q_len, self.num_heads, self.q_head_dim)` where `num_heads=64` is hardcoded — never adjusted for TP world_size. Colwise sharding of q_b_proj would produce output of size `12288/tp_size` but .view() still expects `64×192=12288`. Fatal shape mismatch. Fixing this requires modifying the modeling code (adjusting num_heads per rank), which is beyond a simple tp_plan dict.

### MoE (384 experts per layer, layers 1-60) — SHARDED
```
gate_proj: 7168 → 2048   # local_colwise (shard within expert, no DTensor)
up_proj:   7168 → 2048   # local_colwise
down_proj: 2048 → 7168   # local_rowwise
expert module:            # local (IsolatedParallel — run on assigned GPU only)
mlp module:               # gather (single all-reduce at MLP output)
```
Plus shared_experts (same structure, 1 per layer).
Layer 0 is dense (standard MLP, 7168 → 18432 → 7168) — also sharded.

### Residual stream flow with TP
- Input to layer: **replicated** on all GPUs
- Attention: fully replicated (all GPUs compute full 64-head attention)
- MoE routing (replicated) → expert computation (local, sharded weights) → MLP `gather` → **all-reduce → replicated**
- Layer output: **replicated** ← hooks see full tensor on every rank

**Hooks on `model.layers[L]` work normally** — residual stream is replicated at layer boundaries. Steering injection adds vector on each rank (all identical). Activation capture reads from any rank.

## Corrected tp_plan

Source: HF's own `base_model_tp_plan` in `configuration_deepseek_v3.py` (lines 135-148), with `model.` prefix for `ForCausalLM` wrapper.

```python
tp_plan = {
    # MoE experts (384 per layer, layers 1-60)
    "model.layers.*.mlp.experts.*.gate_proj": "local_colwise",
    "model.layers.*.mlp.experts.*.up_proj": "local_colwise",
    "model.layers.*.mlp.experts.*.down_proj": "local_rowwise",
    "model.layers.*.mlp.experts.*": "local",           # IsolatedParallel per expert
    # Shared expert (1 per layer)
    "model.layers.*.mlp.shared_experts.gate_proj": "local_colwise",
    "model.layers.*.mlp.shared_experts.up_proj": "local_colwise",
    "model.layers.*.mlp.shared_experts.down_proj": "local_rowwise",
    "model.layers.*.mlp.shared_experts": "local",      # IsolatedParallel
    # Dense MLP (layer 0)
    "model.layers.*.mlp.gate_proj": "local_colwise",
    "model.layers.*.mlp.up_proj": "local_colwise",
    "model.layers.*.mlp.down_proj": "local_rowwise",
    # Single all-reduce per MLP block (not per-expert)
    "model.layers.*.mlp": "gather",
    # Output head
    "lm_head": "colwise_rep",
}
```

### Why `local_*` instead of `colwise`/`rowwise`
1. **FP8 compatibility**: `colwise`/`rowwise` use DTensor's `get_tensor_shard` which does direct indexing — crashes on `Float8_e4m3fn`. `local_*` variants use `param[...]` (full copy + cast), handles FP8 correctly.
2. **Communication efficiency**: Plain `rowwise` triggers an all-reduce per linear layer. With 384 experts, that's O(384) all-reduces per MoE layer. `local_rowwise` + `"gather"` at MLP level = single all-reduce.
3. **Expert isolation**: `"local"` (IsolatedParallel) ensures each expert only runs on its assigned GPU.

### What this gives us
- **MoE weights sharded** — bulk of model memory (~90%) distributed across GPUs
- **Attention replicated** — ~12.5% perf overhead vs full TP, but avoids MLA num_heads bug
- **Single all-reduce per layer** — at MLP `gather` point
- **Hooks work** — layer boundaries are replicated
- **FP8 safe** — `local_*` handles FP8 tensors

### Resolved questions (from investigation)
1. **Layer 0 dense MLP**: Handled by `model.layers.*.mlp.gate_proj` etc. patterns — matches dense MLP structure.
2. **Per-expert TP vs expert parallelism**: Using `local_colwise`/`local_rowwise` + `local` isolation. This is what HF designed for DeepSeek-V3.
3. **FP8 + DTensor**: Solved by using `local_*` variants (no DTensor).
4. **`colwise_gather_output`**: Doesn't exist in HF's registry. Correct name is `colwise_rep`.

## Implementation steps

### 1. Write torchrun test script
```python
# test_tp.py — minimal TP test
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tp_plan = {
    "model.layers.*.mlp.experts.*.gate_proj": "local_colwise",
    "model.layers.*.mlp.experts.*.up_proj": "local_colwise",
    "model.layers.*.mlp.experts.*.down_proj": "local_rowwise",
    "model.layers.*.mlp.experts.*": "local",
    "model.layers.*.mlp.shared_experts.gate_proj": "local_colwise",
    "model.layers.*.mlp.shared_experts.up_proj": "local_colwise",
    "model.layers.*.mlp.shared_experts.down_proj": "local_rowwise",
    "model.layers.*.mlp.shared_experts": "local",
    "model.layers.*.mlp.gate_proj": "local_colwise",
    "model.layers.*.mlp.up_proj": "local_colwise",
    "model.layers.*.mlp.down_proj": "local_rowwise",
    "model.layers.*.mlp": "gather",
    "lm_head": "colwise_rep",
}

# NOTE: Do NOT pass trust_remote_code — use native HF DeepseekV3ForCausalLM
# NOTE: Do NOT pass torch_dtype=torch.bfloat16 — let FP8 stay as FP8
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Kimi-K2-Base",
    tp_plan=tp_plan,
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Kimi-K2-Base")

# Test forward pass
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_new_tokens=20)
rank = torch.distributed.get_rank()
if rank == 0:
    print(tokenizer.decode(outputs[0]))

# Test hooks — verify residual stream is replicated
activations = {}
def hook_fn(module, input, output):
    activations['layer15'] = output[0].detach()
model.model.layers[15].register_forward_hook(hook_fn)
with torch.no_grad():
    model(inputs.input_ids)
if rank == 0:
    print(f"Hook shape: {activations['layer15'].shape}")  # expect [1, seq, 7168]
    print(f"Hook device: {activations['layer15'].device}")
```
Launch: `torchrun --nproc-per-node 8 test_tp.py`

**Key concerns to watch for:**
- FP8 loading: if `local_colwise` can't handle FP8, try adding `torch_dtype=torch.bfloat16`
- Vocab size mismatch: Kimi K2 has 163840 tokens vs DeepSeek-V3's 129280. If native class fails, may need to check config overrides.
- Module path matching: verify `model.layers.*.mlp.experts.*` actually matches the nested ModuleList pattern

### 2. Integrate with codebase
Our scripts use `utils/model.py:load_model()` which calls `AutoModelForCausalLM.from_pretrained()`. Changes needed:
- Add `tp_plan` parameter to `load_model()`
- Detect when running under torchrun (`torch.distributed.is_initialized()`)
- Write a torchrun launcher wrapper or modify scripts to be TP-aware
- Key constraint: under torchrun, each process is a rank. Our scripts assume single-process. Need to gate printing/saving to rank 0 only.
- **Do NOT pass `trust_remote_code=True`** when using TP — need native HF class

### 3. Test full pipeline under TP
- Run extraction for 1 trait (anxiety), compare vectors to pipeline-parallel version
- Run steering eval for 1 trait, compare results
- If outputs match, switch all subsequent runs to TP

## Where to develop
- **Write code**: Can be done anywhere (local or remote). No GPU needed for writing.
- **Test code**: Must be on remote (needs 8 GPUs for torchrun).
- **Recommendation**: Write on remote since we're already here and extraction is running in background. Test immediately after writing.

## Risk assessment
- **High confidence**: tp_plan dict (based on HF's own `base_model_tp_plan`), hook compatibility (gather produces replicated output), FP8 handling (`local_*` avoids DTensor)
- **Medium confidence**: Native HF class compatibility with Kimi K2 weights (vocab_size, config diffs), generation speed improvement (all-reduce overhead vs pipeline bubble)
- **Low confidence**: Whether `tp_plan="auto"` picks up `base_model_tp_plan` automatically (may need to pass dict explicitly)

## Expected speedup
- Attention is replicated (no speedup there)
- MoE weights sharded (8x less memory per GPU, all GPUs participate)
- Net: probably 3-5x faster than pipeline parallelism (not full 8x because attention is still replicated)
