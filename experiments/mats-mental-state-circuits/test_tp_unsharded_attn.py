"""
Diagnostic: Does padding cause NaN in UNSHARDED attention (MoE-only TP)?

Hooks residual stream at layers 1, 15, 33 and attention module input/output at layer 15.
Compare with test_tp_diag.py which uses attention sharding.

Launch: torchrun --nproc-per-node 8 experiments/mats-mental-state-circuits/test_tp_unsharded_attn.py
"""

import os, sys, torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import DynamicCache
if not hasattr(DynamicCache, 'seen_tokens'):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
if not hasattr(DynamicCache, 'get_usable_length'):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
if not hasattr(DynamicCache, 'get_max_length'):
    DynamicCache.get_max_length = lambda self: None

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID = "unsloth/Kimi-K2-Base"

def rprint(*a, **kw):
    if dist.get_rank() == 0:
        print(*a, **kw, flush=True)

def main():
    dist.init_process_group("nccl")

    # Load WITHOUT attention sharding (MoE-only TP — the default)
    rprint("Loading model WITHOUT attention sharding (MoE-only TP)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, tp_plan="auto")
    rprint(f"Loaded. tp_plan entries: {len(model.tp_plan)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Same two prompts as test_tp_diag.py
    short = "The capital of France is"
    long = ("In a lengthy discussion about the future of artificial intelligence "
            "and its impact on society, the panel of experts considered various "
            "perspectives including economic disruption and ethical concerns. "
            "The moderator then asked:")

    inputs = tokenizer([short, long], return_tensors="pt", padding=True).to(model.device)
    ids = inputs.input_ids
    mask = inputs.attention_mask

    rprint(f"\n=== INPUT ===")
    rprint(f"input_ids shape: {ids.shape}")
    rprint(f"attention_mask shape: {mask.shape}")
    rprint(f"Sample 0 (short) pad count: {(mask[0] == 0).sum().item()}")
    rprint(f"Sample 1 (long)  pad count: {(mask[1] == 0).sum().item()}")

    # Hook 1: residual stream at layers 1, 15, 33
    residuals = {}
    handles = []
    for L in [1, 15, 33]:
        def make_hook(idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                residuals[idx] = h.detach().cpu()
            return hook_fn
        handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))

    # Hook 2: attention module output at layer 15
    attn_data = {}
    def attn_hook(module, inp, out):
        attn_out = out[0] if isinstance(out, tuple) else out
        attn_data['output'] = attn_out.detach().cpu()
    handles.append(model.model.layers[15].self_attn.register_forward_hook(attn_hook))

    # Hook 3: o_proj output at layer 15 (right after attention weight @ V matmul + output projection)
    oproj_data = {}
    def oproj_hook(module, inp, out):
        oproj_data['input'] = inp[0].detach().cpu() if isinstance(inp, tuple) and len(inp) > 0 else None
        oproj_data['output'] = out.detach().cpu() if not isinstance(out, tuple) else out[0].detach().cpu()
    handles.append(model.model.layers[15].self_attn.o_proj.register_forward_hook(oproj_hook))

    # Hook 4: q_b_proj and kv_b_proj at layer 15 to see projections
    qkv_data = {}
    def q_hook(module, inp, out):
        qkv_data['q_b_proj'] = out.detach().cpu() if not isinstance(out, tuple) else out[0].detach().cpu()
    def kv_hook(module, inp, out):
        qkv_data['kv_b_proj'] = out.detach().cpu() if not isinstance(out, tuple) else out[0].detach().cpu()
    handles.append(model.model.layers[15].self_attn.q_b_proj.register_forward_hook(q_hook))
    handles.append(model.model.layers[15].self_attn.kv_b_proj.register_forward_hook(kv_hook))

    # Forward pass
    rprint(f"\n=== FORWARD PASS ===")
    with torch.no_grad():
        out = model(**inputs)

    for h in handles:
        h.remove()

    logits = out.logits
    rprint(f"Logits shape: {logits.shape}")
    rprint(f"Logits NaN: sample0={logits[0].isnan().any().item()}, sample1={logits[1].isnan().any().item()}")

    if dist.get_rank() == 0:
        # Check residuals
        for L in [1, 15, 33]:
            act = residuals[L]
            print(f"\n--- Layer {L} Residual ---")
            print(f"  Shape: {act.shape}, dtype: {act.dtype}")
            for s in range(2):
                label = "short/padded" if s == 0 else "long/no-pad"
                nan_per_pos = act[s].isnan().any(dim=-1)
                nan_count = nan_per_pos.sum().item()
                pad_count = (mask[s].cpu() == 0).sum().item()
                print(f"  Sample {s} ({label}):")
                print(f"    NaN positions: {nan_count}/{act.shape[1]}")
                print(f"    Pad positions: {pad_count}/{act.shape[1]}")
                if nan_count > 0:
                    nan_positions = nan_per_pos.nonzero().squeeze(-1).tolist()
                    pad_positions = (mask[s].cpu() == 0).nonzero().squeeze(-1).tolist()
                    print(f"    NaN at: {nan_positions[:20]}")
                    print(f"    Pad at: {pad_positions[:20]}")
                    print(f"    NaN == Pad? {nan_positions == pad_positions}")
                else:
                    print(f"    No NaN. Pad pos[0] norm: {act[s, 0].norm():.4f}, last pos norm: {act[s, -1].norm():.4f}")

        # Check attention internals at layer 15
        print(f"\n--- Layer 15 Attention Details ---")

        # q_b_proj output
        if 'q_b_proj' in qkv_data:
            q = qkv_data['q_b_proj']
            print(f"  q_b_proj shape: {q.shape}, dtype: {q.dtype}")
            for s in range(2):
                label = "short/padded" if s == 0 else "long/no-pad"
                nan_count = q[s].isnan().any(dim=-1).sum().item()
                print(f"    Sample {s} ({label}): NaN positions: {nan_count}/{q.shape[1]}")
                if nan_count > 0:
                    nan_pos = q[s].isnan().any(dim=-1).nonzero().squeeze(-1).tolist()
                    print(f"      NaN at: {nan_pos[:20]}")

        # kv_b_proj output
        if 'kv_b_proj' in qkv_data:
            kv = qkv_data['kv_b_proj']
            print(f"  kv_b_proj shape: {kv.shape}, dtype: {kv.dtype}")
            for s in range(2):
                label = "short/padded" if s == 0 else "long/no-pad"
                nan_count = kv[s].isnan().any(dim=-1).sum().item()
                print(f"    Sample {s} ({label}): NaN positions: {nan_count}/{kv.shape[1]}")
                if nan_count > 0:
                    nan_pos = kv[s].isnan().any(dim=-1).nonzero().squeeze(-1).tolist()
                    print(f"      NaN at: {nan_pos[:20]}")

        # o_proj output (attention contribution to residual)
        if 'output' in oproj_data and oproj_data['output'] is not None:
            op = oproj_data['output']
            print(f"  o_proj output shape: {op.shape}, dtype: {op.dtype}")
            for s in range(2):
                label = "short/padded" if s == 0 else "long/no-pad"
                nan_count = op[s].isnan().any(dim=-1).sum().item()
                print(f"    Sample {s} ({label}): NaN positions: {nan_count}/{op.shape[1]}")
                if nan_count > 0:
                    nan_pos = op[s].isnan().any(dim=-1).nonzero().squeeze(-1).tolist()
                    print(f"      NaN at: {nan_pos[:20]}")
                else:
                    print(f"      No NaN. Pad pos[0] norm: {op[s, 0].norm():.4f}, last pos norm: {op[s, -1].norm():.4f}")

        # Attention module output
        if 'output' in attn_data:
            ao = attn_data['output']
            print(f"  Attention module output shape: {ao.shape}, dtype: {ao.dtype}")
            for s in range(2):
                label = "short/padded" if s == 0 else "long/no-pad"
                nan_count = ao[s].isnan().any(dim=-1).sum().item()
                print(f"    Sample {s} ({label}): NaN positions: {nan_count}/{ao.shape[1]}")
                if nan_count > 0:
                    nan_pos = ao[s].isnan().any(dim=-1).nonzero().squeeze(-1).tolist()
                    print(f"      NaN at: {nan_pos[:20]}")
                else:
                    print(f"      No NaN. Pad pos[0] norm: {ao[s, 0].norm():.4f}, last pos norm: {ao[s, -1].norm():.4f}")

    rprint(f"\n=== DONE ===")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
