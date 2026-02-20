"""
Trace where NaN first appears in the model with padded input.

Hooks: embed_tokens → layer 0 input_layernorm → layer 0 attention → layer 0 MLP → layer 0 output → layer 1

Also checks: does the pad token embedding itself contain NaN?

Launch: torchrun --nproc-per-node 8 experiments/mats-mental-state-circuits/test_tp_nan_trace.py
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

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "unsloth/Kimi-K2-Base"

def rprint(*a, **kw):
    if dist.get_rank() == 0:
        print(*a, **kw, flush=True)

def main():
    dist.init_process_group("nccl")

    rprint("Loading model (MoE-only TP)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, tp_plan="auto")
    rprint(f"Loaded. tp_plan entries: {len(model.tp_plan)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    rprint(f"pad_token_id: {tokenizer.pad_token_id} (eos_token_id: {tokenizer.eos_token_id})")
    rprint(f"Sample 0 pad count: {(mask[0] == 0).sum().item()}")

    # Check: what token IDs are at pad positions?
    if dist.get_rank() == 0:
        pad_ids = ids[0, :5].tolist()
        real_ids = ids[0, -5:].tolist()
        print(f"Sample 0 first 5 token IDs (should be pad): {pad_ids}")
        print(f"Sample 0 last 5 token IDs (should be real): {real_ids}")

    # Hook everything from embed_tokens through layer 1
    data = {}
    handles = []

    # 1. Embedding output
    def embed_hook(module, inp, out):
        data['embed'] = out.detach().cpu()
    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    # 2. Layer 0 — this is a DENSE layer (not MoE) for DeepSeek-V3/Kimi K2
    # Hook the layer itself
    def layer0_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        data['layer0_out'] = h.detach().cpu()
        # Also capture input
        if isinstance(inp, tuple) and len(inp) > 0:
            data['layer0_in'] = inp[0].detach().cpu()
    handles.append(model.model.layers[0].register_forward_hook(layer0_hook))

    # 3. Layer 0's attention module
    def layer0_attn_hook(module, inp, out):
        attn_out = out[0] if isinstance(out, tuple) else out
        data['layer0_attn_out'] = attn_out.detach().cpu()
    handles.append(model.model.layers[0].self_attn.register_forward_hook(layer0_attn_hook))

    # 4. Layer 0's MLP (dense layer — not MoE)
    def layer0_mlp_hook(module, inp, out):
        mlp_out = out[0] if isinstance(out, tuple) else out
        data['layer0_mlp_out'] = mlp_out.detach().cpu()
        if isinstance(inp, tuple) and len(inp) > 0:
            data['layer0_mlp_in'] = inp[0].detach().cpu()
    handles.append(model.model.layers[0].mlp.register_forward_hook(layer0_mlp_hook))

    # 5. Layer 0's input layernorm
    def layer0_norm_hook(module, inp, out):
        data['layer0_norm_out'] = out.detach().cpu() if not isinstance(out, tuple) else out[0].detach().cpu()
        if isinstance(inp, tuple) and len(inp) > 0:
            data['layer0_norm_in'] = inp[0].detach().cpu()
    handles.append(model.model.layers[0].input_layernorm.register_forward_hook(layer0_norm_hook))

    # 6. Layer 1 output
    def layer1_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        data['layer1_out'] = h.detach().cpu()
    handles.append(model.model.layers[1].register_forward_hook(layer1_hook))

    # 7. Layer 1 attention (first MoE layer's attention)
    def layer1_attn_hook(module, inp, out):
        attn_out = out[0] if isinstance(out, tuple) else out
        data['layer1_attn_out'] = attn_out.detach().cpu()
    handles.append(model.model.layers[1].self_attn.register_forward_hook(layer1_attn_hook))

    # Forward pass
    rprint(f"\n=== FORWARD PASS ===")
    with torch.no_grad():
        out = model(**inputs)

    for h in handles:
        h.remove()

    logits = out.logits
    rprint(f"Logits NaN: sample0={logits[0].isnan().any().item()}, sample1={logits[1].isnan().any().item()}")

    if dist.get_rank() == 0:
        pad_mask = (mask[0].cpu() == 0)
        pad_count = pad_mask.sum().item()

        def report(name, tensor, sample=0):
            if tensor is None:
                print(f"  {name}: NOT CAPTURED")
                return
            t = tensor[sample]  # [seq_len, dim]
            nan_per_pos = t.isnan().any(dim=-1)
            nan_count = nan_per_pos.sum().item()
            total = t.shape[0]
            print(f"  {name}: shape={tensor.shape}, NaN positions: {nan_count}/{total}", end="")
            if nan_count > 0 and nan_count < total:
                nan_pos = nan_per_pos.nonzero().squeeze(-1).tolist()
                print(f" at {nan_pos[:10]}...", end="")
            elif nan_count == total:
                print(f" (ALL NaN)", end="")
            if nan_count == 0:
                # Show norms at pad vs real positions
                pad_norms = t[:pad_count].norm(dim=-1)
                real_norms = t[pad_count:].norm(dim=-1)
                print(f" | pad norms: mean={pad_norms.mean():.4f}, max={pad_norms.max():.4f}, min={pad_norms.min():.4f}", end="")
                print(f" | real norms: mean={real_norms.mean():.4f}", end="")
            print()

        print(f"\n=== NaN TRACE (Sample 0 — padded, {pad_count} pad tokens) ===")
        report("embed_tokens", data.get('embed'))
        report("layer0_norm_in", data.get('layer0_norm_in'))
        report("layer0_norm_out", data.get('layer0_norm_out'))
        report("layer0_attn_out", data.get('layer0_attn_out'))
        report("layer0_mlp_in", data.get('layer0_mlp_in'))
        report("layer0_mlp_out", data.get('layer0_mlp_out'))
        report("layer0_out", data.get('layer0_out'))
        report("layer1_attn_out", data.get('layer1_attn_out'))
        report("layer1_out", data.get('layer1_out'))

        print(f"\n=== NaN TRACE (Sample 1 — no padding) ===")
        for name in ['embed', 'layer0_norm_in', 'layer0_norm_out', 'layer0_attn_out',
                      'layer0_mlp_in', 'layer0_mlp_out', 'layer0_out', 'layer1_attn_out', 'layer1_out']:
            t = data.get(name)
            if t is not None:
                nan_count = t[1].isnan().any(dim=-1).sum().item()
                print(f"  {name}: NaN positions: {nan_count}/{t.shape[1]}")

        # Check the pad token embedding specifically
        print(f"\n=== PAD TOKEN EMBEDDING ===")
        embed = data.get('embed')
        if embed is not None:
            pad_embed = embed[0, 0]  # First pad position
            real_embed = embed[0, -1]  # Last real position
            print(f"  Pad position [0] embedding: norm={pad_embed.norm():.6f}, NaN={pad_embed.isnan().any().item()}, inf={pad_embed.isinf().any().item()}")
            print(f"  Real position [-1] embedding: norm={real_embed.norm():.6f}, NaN={real_embed.isnan().any().item()}")
            # Check if all pad positions have same embedding
            pad_embeds = embed[0, :pad_count]  # [pad_count, dim]
            print(f"  All pad embeddings identical? {torch.allclose(pad_embeds[0], pad_embeds[-1])}")
            print(f"  Pad embed first 10 values: {pad_embed[:10].tolist()}")
            print(f"  Real embed first 10 values: {real_embed[:10].tolist()}")

    rprint(f"\n=== DONE ===")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
