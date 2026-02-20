"""
Diagnostic: test attention sharding with padded batches.

Launch: torchrun --nproc-per-node 8 experiments/mats-mental-state-circuits/test_tp_nan_diagnostic.py
"""

import os
import sys
import torch
import torch.distributed as dist

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


def rank_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    rank_print("=== NaN Diagnostic: Attention Sharding ===\n")

    # Load with attention sharding
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.base_model_tp_plan.update({
        "layers.*.self_attn.q_b_proj": "local_colwise",
        "layers.*.self_attn.kv_b_proj": "local_colwise",
        "layers.*.self_attn.o_proj": "local_rowwise",
        "layers.*.self_attn": "gather",
    })
    rank_print("Loading model with attention sharding...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=config, tp_plan="auto")
    model.eval()
    rank_print("Model loaded.\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Create batch with different lengths (left-padded)
    short_prompt = "Hello world"
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 10  # ~100 tokens

    inputs = tokenizer(
        [short_prompt, long_prompt],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    rank_print(f"input_ids shape: {inputs.input_ids.shape}")
    rank_print(f"attention_mask shape: {inputs.attention_mask.shape}")
    rank_print(f"Sample 0 (short) attn_mask first 20: {inputs.attention_mask[0, :20].tolist()}")
    rank_print(f"Sample 1 (long)  attn_mask first 20: {inputs.attention_mask[1, :20].tolist()}")
    rank_print(f"Sample 0 pad tokens: {(inputs.attention_mask[0] == 0).sum().item()}")
    rank_print(f"Sample 1 pad tokens: {(inputs.attention_mask[1] == 0).sum().item()}")
    rank_print()

    # Hook to capture activations at layers 1, 15, 33
    activations = {}
    hooks = []
    for layer_idx in [1, 15, 33]:
        def make_hook(idx):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activations[idx] = hidden.detach()
            return hook_fn
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Forward pass
    rank_print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
    rank_print("Forward pass complete.\n")

    for h in hooks:
        h.remove()

    # Analyze results
    for layer_idx in [1, 15, 33]:
        act = activations[layer_idx]
        rank_print(f"--- Layer {layer_idx} ---")
        rank_print(f"  Shape: {act.shape}, dtype: {act.dtype}")

        for sample_idx in range(act.shape[0]):
            sample = act[sample_idx]  # [seq_len, hidden_dim]
            nan_per_pos = sample.isnan().any(dim=-1)  # [seq_len]
            mask = inputs.attention_mask[sample_idx]

            padded_positions = (mask == 0)
            content_positions = (mask == 1)

            nan_in_pad = nan_per_pos[padded_positions].float().mean().item() if padded_positions.any() else 0
            nan_in_content = nan_per_pos[content_positions].float().mean().item()

            label = "short/padded" if sample_idx == 0 else "long/no-pad"
            rank_print(f"  Sample {sample_idx} ({label}):")
            rank_print(f"    Total NaN positions: {nan_per_pos.sum().item()}/{nan_per_pos.shape[0]}")
            rank_print(f"    NaN in padded positions: {nan_in_pad:.1%} ({padded_positions.sum().item()} positions)")
            rank_print(f"    NaN in content positions: {nan_in_content:.1%} ({content_positions.sum().item()} positions)")

            # Show first few values at a content position
            first_content = content_positions.nonzero()[0].item()
            vals = sample[first_content, :5]
            rank_print(f"    First content pos values: {vals.tolist()}")

        rank_print()

    # Also check logits
    if rank == 0:
        logits = outputs.logits
        print(f"--- Logits ---", flush=True)
        print(f"  Shape: {logits.shape}, dtype: {logits.dtype}", flush=True)
        for sample_idx in range(logits.shape[0]):
            nan_count = logits[sample_idx].isnan().sum().item()
            total = logits[sample_idx].numel()
            label = "short" if sample_idx == 0 else "long"
            print(f"  Sample {sample_idx} ({label}): NaN={nan_count}/{total} ({nan_count/total:.1%})", flush=True)
            last_logit = logits[sample_idx, -1]
            top_token = last_logit.argmax().item()
            print(f"    Top predicted token: '{tokenizer.decode([top_token])}' (id={top_token})", flush=True)

    rank_print("\n=== Diagnostic complete ===")


if __name__ == "__main__":
    main()
