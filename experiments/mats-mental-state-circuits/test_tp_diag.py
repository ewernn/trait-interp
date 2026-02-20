"""
Diagnostic: padded batch with attention sharding + unmask-padding hook.

Verifies that the unmask_unattended fix eliminates NaN from padded batches.

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
from utils.model import install_unmask_padding_hook

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

    # Install unmask-padding hook
    install_unmask_padding_hook(model)
    rprint("Unmask-padding hook installed.")

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
    rprint(f"input_ids shape: {ids.shape}")
    rprint(f"Sample 0 (short) pad count: {(mask[0] == 0).sum().item()}")
    rprint(f"Sample 1 (long)  pad count: {(mask[1] == 0).sum().item()}")

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
    rprint(f"\n=== FORWARD PASS (with unmask hook) ===")
    with torch.no_grad():
        out = model(**inputs)
    for h in handles:
        h.remove()

    # Check logits
    logits = out.logits
    rprint(f"Logits shape: {logits.shape}")
    rprint(f"Logits NaN: sample0={logits[0].isnan().any().item()}, sample1={logits[1].isnan().any().item()}")

    if dist.get_rank() == 0:
        all_clean = True
        for L in [1, 15, 33]:
            act = activations[L]
            print(f"\n--- Layer {L} ---")
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
                    all_clean = False
                    if nan_count < act.shape[1]:
                        nan_positions = nan_per_pos.nonzero().squeeze(-1).tolist()
                        print(f"    NaN at positions: {nan_positions[:20]}")
                    else:
                        print(f"    ALL positions NaN!")
                else:
                    print(f"    No NaN. Padded pos[0] norm: {act[s, 0].norm():.4f}, last pos norm: {act[s, -1].norm():.4f}")

        # Test generation too
        print(f"\n--- Generation Test ---")

    # Generate on rank 0 (all ranks must call generate for TP sync)
    gen_ids = model.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask,
        max_new_tokens=10, do_sample=False
    )

    if dist.get_rank() == 0:
        for s in range(2):
            label = "short" if s == 0 else "long"
            new_tokens = gen_ids[s, ids.shape[1]:]
            text = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
            has_nan = any(t == tokenizer.eos_token_id for t in new_tokens[:3].tolist())
            print(f"  Sample {s} ({label}): {text!r}")

        logit_nan = logits[0].isnan().any().item() or logits[1].isnan().any().item()
        if all_clean and not logit_nan:
            print(f"\n=== PASS: Zero NaN with attention sharding + unmask hook ===")
        else:
            print(f"\n=== FAIL: NaN still present ===")

    rprint(f"\n=== DONE ===")

if __name__ == "__main__":
    main()
