# Tensor Parallelism Context — Research Dump

Everything we learned about HF tensor parallelism for DeepSeek-V3 / Kimi K2.

---

## URLs

### HF Docs
- **Tensor parallelism guide**: https://huggingface.co/docs/transformers/en/perf_infer_gpu_multi
- **DeepSeek-V3 model doc**: https://huggingface.co/docs/transformers/model_doc/deepseek_v3
- **Expert parallelism guide**: https://huggingface.co/docs/transformers/expert_parallelism
- **Ultra-Scale Playbook (TP section)**: https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism

### HF Source Code
- **DeepSeek-V3 modeling**: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py
- **DeepSeek-V3 config**: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/configuration_deepseek_v3.py
- **TP integration code**: https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py
- **DeepSeek-V3 support issue**: https://github.com/huggingface/transformers/issues/35425

### Kimi K2 Model Pages
- **Kimi K2 Base (unsloth FP8)**: https://huggingface.co/unsloth/Kimi-K2-Base
- **Kimi K2 Thinking (INT4)**: https://huggingface.co/moonshotai/Kimi-K2-Thinking
- **Kimi K2 Instruct**: https://huggingface.co/moonshotai/Kimi-K2-Instruct
- **Kimi K2 custom modeling code**: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/modeling_deepseek.py

### Blog Posts
- **TP vs Device Map**: https://huggingface.co/blog/ariG23498/tp-vs-dm
- **TP in Transformers: 5 Minutes**: https://huggingface.co/blog/qgallouedec/tp

---

## Current State of HF TP for DeepSeek-V3

- Native `DeepseekV3ForCausalLM` added to HF transformers **2025-03-28**
- `model_type: "deepseek_v3"` — auto-resolves to native class without `trust_remote_code`
- Built-in `_tp_plan` is **minimal**: `{"lm_head": "colwise_gather_output"}`
- Built-in `_pp_plan`: `{"lm_head": (["hidden_states"], ["logits"])}`
- No attention or MoE TP wired up yet
- Docs explicitly call for community contributions:
  - "current implementation uses the 'naive' attention computation (so not really MLA)"
  - "current implementation loops through the experts. This should be replaced."
  - "Pointers to use `get_packed_weights` from `integrations/tensor_parallel`"

---

## HF TP API

### Loading with TP
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    tp_plan="auto",        # uses model's built-in _tp_plan
    # OR
    tp_plan={...},          # custom dict
    torch_dtype=torch.bfloat16,
)
```
Launched with: `torchrun --nproc-per-node 8 script.py`

### Available Strategies (from ParallelInterface)
```python
_global_mapping = {
    "colwise": ColwiseParallel(),                              # Shard weight cols across GPUs
    "rowwise": RowwiseParallel(),                              # Shard weight rows, all-reduce output
    "colwise_rep": ColwiseParallel(output_layouts=Replicate()), # Colwise but replicate output
    "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),  # Rowwise but replicate input
    "local_colwise": ColwiseParallel(use_dtensor=False),       # No DTensor, manual distributed logic
    "local_rowwise": RowwiseParallel(use_dtensor=False),       # No DTensor, manual distributed logic
    "local": IsolatedParallel(),                               # Isolate module from other devices (for MoE experts)
    "moe_tp_experts": MoeTensorParalellExperts(),              # Distribute whole experts across GPUs
    "local_packed_rowwise": PackedRowwiseParallel(use_dtensor=False),
    "sequence_parallel": SequenceParallel(),                   # For LayerNorm/Dropout across sequence dim
    "replicate": ReplicateParallel(),                          # Full copy on each GPU
    "colwise_gather_output": ???,                              # Used for lm_head in DeepSeek-V3
}
```

### Key concepts
- **ColwiseParallel**: Splits weight along output dim. Input replicated, output sharded.
- **RowwiseParallel**: Splits weight along input dim. Input sharded, output all-reduced (replicated).
- **Typical layer pattern**: `colwise → internal computation → rowwise` ensures input and output are both replicated.
- **IsolatedParallel ("local")**: "Isolates a module from other devices. Used for Experts in MoE layers."
- **MoeTensorParallelExperts ("moe_tp_experts")**: Distributes whole experts across GPUs. 384 experts / 8 GPUs = 48 per GPU.
- **SequenceParallel**: For LayerNorm and Dropout. Supports Python RMSNorm implementation.
- **DTensor**: Distributed tensor abstraction from `torch.distributed`. Handles communication automatically. Some ops not supported (e.g., `torch.chunk`), hence "local_*" variants exist.

### Custom strategy registration
```python
from transformers.integrations.tensor_parallel import ParallelInterface
ParallelInterface.register_strategy("my_custom", MyCustomStrategy)
tp_plan = {"model.layers.*.self_attn.q_proj": "my_custom"}
```

---

## Kimi K2 / DeepSeek-V3 Architecture

### Model dimensions
- **Hidden size**: 7168
- **Num layers**: 61 (0-60)
- **Num attention heads**: 64
- **Vocab size**: 163840 (Kimi K2) / 129280 (DeepSeek-V3)
- **Max context**: 262144 tokens

### MLA (Multi-head Latent Attention)
- `q_lora_rank`: 1536
- `kv_lora_rank`: 512
- `qk_rope_head_dim`: 64
- `qk_nope_head_dim`: 128
- `v_head_dim`: 128
- `q_head_dim` = qk_nope_head_dim + qk_rope_head_dim = 192

### MLA Linear Layers (per layer)
| Layer | Input dim | Output dim | Notes |
|---|---|---|---|
| `q_a_proj` | 7168 | 1536 | Compress Q to latent. SMALL — don't shard |
| `q_b_proj` | 1536 | 64×192 = 12288 | Expand to heads. **Shard colwise** |
| `kv_a_proj_with_mqa` | 7168 | 512+64 = 576 | Compress KV + rope. SMALL — don't shard |
| `kv_b_proj` | 512 | 64×(128+128) = 16384 | Expand to heads. **Shard colwise** |
| `o_proj` | 64×128 = 8192 | 7168 | Output projection. **Shard rowwise** |

Between `q_a_proj`→`q_b_proj`: `q_a_layernorm` (RMSNorm over 1536 dim)
Between `kv_a_proj`→`kv_b_proj`: `kv_a_layernorm` (RMSNorm over 512 dim)

**Why not shard q_a/kv_a**: Sharding colwise would split the layernorm input across GPUs. RMSNorm normalizes across the full dimension — computing it on a shard gives wrong results. Would need `sequence_parallel` on the layernorms, adding complexity. Since these projections are small (7168→1536, 7168→576), keeping them replicated wastes negligible memory/compute.

### MoE (layers 1-60)
- `n_routed_experts`: 384
- `num_experts_per_tok`: 8
- `n_shared_experts`: 1
- `moe_intermediate_size`: 2048
- `first_k_dense_replace`: 1 (layer 0 is dense)

### MoE Expert Linear Layers (per expert, per layer)
| Layer | Input dim | Output dim | Notes |
|---|---|---|---|
| `gate_proj` | 7168 | 2048 | **Shard colwise** or use moe_tp_experts |
| `up_proj` | 7168 | 2048 | **Shard colwise** or use moe_tp_experts |
| `down_proj` | 2048 | 7168 | **Shard rowwise** or use moe_tp_experts |

Same structure for `shared_experts` (1 per layer).

### Dense MLP (layer 0 only)
| Layer | Input dim | Output dim |
|---|---|---|
| `gate_proj` | 7168 | 18432 |
| `up_proj` | 7168 | 18432 |
| `down_proj` | 18432 | 7168 |

### Module paths (from modeling_deepseek.py)
```
model.embed_tokens                                          # Embedding(163840, 7168)
model.layers.{L}.self_attn.q_a_proj                        # Linear(7168, 1536)
model.layers.{L}.self_attn.q_a_layernorm                   # RMSNorm(1536)
model.layers.{L}.self_attn.q_b_proj                        # Linear(1536, 12288)
model.layers.{L}.self_attn.kv_a_proj_with_mqa              # Linear(7168, 576)
model.layers.{L}.self_attn.kv_a_layernorm                  # RMSNorm(512)
model.layers.{L}.self_attn.kv_b_proj                       # Linear(512, 16384)
model.layers.{L}.self_attn.o_proj                          # Linear(8192, 7168)
model.layers.{L}.input_layernorm                           # RMSNorm(7168)
model.layers.{L}.post_attention_layernorm                  # RMSNorm(7168)
model.layers.{L}.mlp.gate                                  # MoEGate (router)
model.layers.{L}.mlp.experts.{E}.gate_proj                 # Linear(7168, 2048)
model.layers.{L}.mlp.experts.{E}.up_proj                   # Linear(7168, 2048)
model.layers.{L}.mlp.experts.{E}.down_proj                 # Linear(2048, 7168)
model.layers.{L}.mlp.shared_experts.gate_proj              # Linear(7168, 2048)
model.layers.{L}.mlp.shared_experts.up_proj                # Linear(7168, 2048)
model.layers.{L}.mlp.shared_experts.down_proj              # Linear(2048, 7168)
model.norm                                                  # RMSNorm(7168)
lm_head                                                     # Linear(7168, 163840)
```

### Class hierarchy (from custom modeling_deepseek.py)
```
DeepseekV3ForCausalLM
  ├── model: DeepseekV3Model
  │     ├── embed_tokens: Embedding
  │     ├── layers: ModuleList[DeepseekV3DecoderLayer]
  │     │     ├── self_attn: DeepseekV3Attention / DeepseekV3FlashAttention2
  │     │     ├── mlp: DeepseekV3MoE (layers 1-60) / DeepseekV3MLP (layer 0)
  │     │     │     ├── gate: MoEGate
  │     │     │     ├── experts: ModuleList[DeepseekV3MLP] (384)
  │     │     │     └── shared_experts: DeepseekV3MLP
  │     │     ├── input_layernorm: DeepseekV3RMSNorm
  │     │     └── post_attention_layernorm: DeepseekV3RMSNorm
  │     └── norm: DeepseekV3RMSNorm
  └── lm_head: Linear
```

---

## Residual Stream Flow Under TP

This is why hooks work:

```
Input (replicated on all 8 GPUs)
  │
  ├─→ input_layernorm (replicated — operates on full hidden dim)
  │
  ├─→ q_a_proj (replicated) → q_a_layernorm (replicated) → q_b_proj (COLWISE → sharded across heads)
  ├─→ kv_a_proj (replicated) → kv_a_layernorm (replicated) → kv_b_proj (COLWISE → sharded across heads)
  │
  ├─→ Attention computation (sharded across heads — each GPU has 64/8 = 8 heads)
  │
  ├─→ o_proj (ROWWISE → all-reduce → REPLICATED)
  │
  ├─→ + residual = layer output after attention (REPLICATED) ← HOOK POINT
  │
  ├─→ post_attention_layernorm (replicated)
  │
  ├─→ MoE routing (replicated) → expert dispatch → expert computation (sharded)
  │     → down_proj per expert (ROWWISE → all-reduce → REPLICATED)
  │     + shared_expert computation → down_proj (ROWWISE → all-reduce → REPLICATED)
  │
  ├─→ + residual = layer output (REPLICATED) ← HOOK POINT
  │
  └─→ Next layer input (REPLICATED)
```

**Every layer boundary is replicated.** Hooks on `model.layers[L]` see full 7168-dim tensors. Steering injection adds the same vector on every rank. Activation capture reads identical values from any rank.

---

## FP8 / INT4 Concerns — RESOLVED

### The fix: `local_*` variants avoid DTensor entirely
- `colwise`/`rowwise` use DTensor's `get_tensor_shard` — does direct indexing, **crashes on Float8_e4m3fn**
- `local_colwise`/`local_rowwise` use `param[...]` (full copy + cast) — **handles FP8 correctly**
- This is exactly why HF's own `base_model_tp_plan` uses `local_*` variants

### Base model (unsloth/Kimi-K2-Base): FP8
- Weights stored as `float8_e4m3fn` with `weight_scale_inv` tensors
- 959GB on disk across 61 safetensors shards
- **Do NOT pass `torch_dtype=torch.bfloat16`** — this forces FP8→bfloat16 expansion (~1.9TB, won't fit)
- Let FP8 stay as FP8; `local_*` strategies handle it

### Thinking model (moonshotai/Kimi-K2-Thinking): INT4
- Uses `compressed-tensors` format (native INT4)
- 554GB on disk
- Unknown if `local_*` handles compressed-tensors — test separately

---

## MoE Expert Parallelism — RESOLVED

### Answer: `local_colwise`/`local_rowwise` + `local` + `gather`
HF's own `base_model_tp_plan` (in `configuration_deepseek_v3.py` lines 135-148) uses:
```python
"layers.*.mlp.experts.*.gate_proj": "local_colwise",   # shard weights, no DTensor
"layers.*.mlp.experts.*.up_proj": "local_colwise",
"layers.*.mlp.experts.*.down_proj": "local_rowwise",
"layers.*.mlp.experts.*": "local",                      # IsolatedParallel per expert
"layers.*.mlp": "gather",                               # single all-reduce
```

This is per-expert weight sharding with:
- `local` isolation (each expert runs on its assigned rank only)
- Single `gather` all-reduce at the MLP level (not per-expert)
- `local_*` for FP8 compatibility

### Why NOT plain `colwise`/`rowwise` on experts (our original plan)
1. Each expert's `rowwise` down_proj would trigger its own all-reduce = O(384) all-reduces per layer
2. DTensor `get_tensor_shard` crashes on FP8 weights
3. No expert isolation — all GPUs would try to compute all experts

### Why NOT `moe_tp_experts`
`moe_tp_experts` distributes whole experts (48 per GPU). This is expert parallelism, not tensor parallelism — requires token routing changes. HF chose per-expert weight sharding for DeepSeek-V3 instead.

---

## What the Custom Code Does Differently

The Kimi K2 model ships with `modeling_deepseek.py` (trust_remote_code). Key differences from HF native:
- `DynamicCache` monkeypatch needed (uses deprecated `seen_tokens`, `get_usable_length`, `get_max_length`)
- Custom `DeepseekV3FlashAttention2` implementation
- Expert parallelism built into MoE class (`ep_size`, `ep_rank`, `experts_per_rank`)
- FP8 weight loading with `weight_scale_inv` tensors

HF native `DeepseekV3ForCausalLM`:
- "Naive" attention (not optimized MLA)
- Loops through experts
- May not handle Kimi K2's FP8 format or custom tokenizer

**Key risk**: Loading Kimi K2 with native HF class may fail due to config differences (vocab_size 163840 vs 129280, different tokenizer, FP8 weight format).

**IMPORTANT**: For TP, we MUST use native HF class (no `trust_remote_code`). The custom code does not have TP support wired up.

---

## Investigation Findings (2025-02-19)

### 5 Flaws Found in Original tp_plan

1. **`colwise_gather_output` doesn't exist** (FATAL) — Not registered in HF's `ParallelInterface`. Correct: `colwise_rep`.

2. **Attention sharding breaks MLA `.view()` calls** (FATAL) — `q.view(bsz, q_len, self.num_heads, self.q_head_dim)` uses hardcoded `num_heads=64`, never adjusted for TP world_size. Colwise q_b_proj output = `12288/tp_size` but view expects `64×192=12288`. Shape mismatch crash. This is why HF's own plan deliberately skips attention.

3. **Plain `colwise`/`rowwise` on experts = O(384) all-reduces per layer** (FATAL) — Each expert's down_proj triggers its own all-reduce. Fix: `local_colwise`/`local_rowwise` + `local` isolation + single `gather`.

4. **`colwise`/`rowwise` crash on FP8 tensors** (FATAL) — DTensor's `get_tensor_shard` does direct indexing, not implemented for `Float8_e4m3fn`. Fix: `local_*` variants.

5. **Dense MLP (layer 0) not covered** (minor) — `mlp.experts.*` patterns don't match layer 0's dense MLP. Fix: add `mlp.gate_proj`/`mlp.up_proj`/`mlp.down_proj` entries.

### Key Discovery: HF already has the plan
`configuration_deepseek_v3.py` lines 135-148 contains `base_model_tp_plan` with the correct MoE-only TP plan. It just isn't wired to `_tp_plan` yet (which only has lm_head).

---

## Extraction Status (as of writing)

- Extraction running with pipeline parallelism (device_map="auto")
- anxiety: complete (responses, activations layers 9-36, probe+mean_diff vectors)
- guilt: complete (vectors extracted)
- confidence: in progress
- 8 remaining traits
- ~8 min per trait
- Using --no-logitlens (meta tensor crash on lm_head with pipeline parallelism)
- Bug found and fixed: max_memory now set to 97% of free VRAM per GPU (utils/model.py)
- Despite fix, some params still on meta/disk — FP8→bfloat16 expansion issue (torch_dtype=torch.bfloat16 doubles weight memory)
