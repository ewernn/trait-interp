"""
Diagnostic: Does padding cause NaN with attention sharding?

Launch: torchrun --nproc-per-node 8 experiments/mats-mental-state-circuits/test_tp_diag.py
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

    # Load with attention sharding
    rprint("Loading model with attention sharding...")
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.base_model_tp_plan.update({
        "layers.*.self_attn.q_b_proj": "local_colwise",
        "layers.*.self_attn.kv_b_proj": "local_colwise",
        "layers.*.self_attn.o_proj": "local_rowwise",
        "layers.*.self_attn": "gather",
    })
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=config, tp_plan="auto")
    rprint(f"Loaded. tp_plan entries: {len(model.tp_plan)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Two prompts: short (5 tok) and long (~50 tok)
    short = "The capital of France is"
    long = ("In a lengthy discussion about the future of artificial intelligence "
            "and its impact on society, the panel of experts considered various "
            "perspectives including economic disruption and ethical concerns. "
            "The moderator then asked:")

    inputs = tokenizer([short, long], return_tensors="pt", padding=True).to(model.device)
    ids = inputs.input_ids
    mask = inputs.attention_mask

    rprint(f"\n=== INPUT ===")
    rprint(f"input_ids shape: {ids.shape}")  # [2, max_len]
    rprint(f"attention_mask shape: {mask.shape}")
    rprint(f"Sample 0 (short) mask first 20: {mask[0, :20].tolist()}")
    rprint(f"Sample 1 (long)  mask first 20: {mask[1, :20].tolist()}")
    rprint(f"Sample 0 pad count: {(mask[0] == 0).sum().item()}")
    rprint(f"Sample 1 pad count: {(mask[1] == 0).sum().item()}")

    # Hook layers 1, 15, 33
    activations = {}
    handles = []
    for L in [1, 15, 33]:
        def make_hook(idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                activations[idx] = h.detach().cpu()
            return hook_fn
        handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))

    # Single forward pass
    rprint(f"\n=== FORWARD PASS ===")
    with torch.no_grad():
        out = model(**inputs)
    for h in handles:
        h.remove()

    # Check logits
    logits = out.logits
    rprint(f"Logits shape: {logits.shape}")
    rprint(f"Logits NaN: sample0={logits[0].isnan().any().item()}, sample1={logits[1].isnan().any().item()}")

    if dist.get_rank() == 0:
        # Check each hooked layer
        for L in [1, 15, 33]:
            act = activations[L]  # [2, seq_len, 7168]
            print(f"\n--- Layer {L} ---")
            print(f"  Shape: {act.shape}, dtype: {act.dtype}")

            for s in range(2):
                label = "short/padded" if s == 0 else "long/no-pad"
                nan_per_pos = act[s].isnan().any(dim=-1)  # [seq_len] bool
                nan_count = nan_per_pos.sum().item()
                pad_count = (mask[s].cpu() == 0).sum().item()

                print(f"  Sample {s} ({label}):")
                print(f"    NaN positions: {nan_count}/{act.shape[1]}")
                print(f"    Pad positions: {pad_count}/{act.shape[1]}")

                if nan_count > 0 and nan_count < act.shape[1]:
                    # Show which positions are NaN vs padded
                    nan_positions = nan_per_pos.nonzero().squeeze(-1).tolist()
                    pad_positions = (mask[s].cpu() == 0).nonzero().squeeze(-1).tolist()
                    print(f"    NaN at positions: {nan_positions[:20]}{'...' if len(nan_positions) > 20 else ''}")
                    print(f"    Pad at positions: {pad_positions[:20]}{'...' if len(pad_positions) > 20 else ''}")
                    print(f"    NaN == Pad? {nan_positions == pad_positions}")
                elif nan_count == act.shape[1]:
                    print(f"    ALL positions NaN!")
                else:
                    # Show a few values from padded vs non-padded positions
                    print(f"    No NaN. Padded pos[0] norm: {act[s, 0].norm():.4f}, last pos norm: {act[s, -1].norm():.4f}")

    rprint(f"\n=== DONE ===")

if __name__ == "__main__":
    main()
