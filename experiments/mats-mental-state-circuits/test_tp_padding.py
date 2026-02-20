"""
Investigate attention sharding NaN: padding / batch size / sequence length.

Launch with: torchrun --nproc-per-node 8 experiments/mats-mental-state-circuits/test_tp_padding.py

Step 1: Padding test
  - batch=2, different-length prompts (5 vs 50 tokens)
  - batch=2, identical prompts
  - If different-length fails but identical works → padding/mask bug
"""

import os
import sys
import time
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


def check_nan_at_layers(model, inputs, layers_to_check=[15, 25, 33]):
    """Forward pass with hooks to check for NaN at specific layers."""
    activations = {}
    handles = []

    for layer_idx in layers_to_check:
        def make_hook(idx):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activations[idx] = hidden.detach()
            return hook_fn
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    with torch.no_grad():
        outputs = model(**inputs)

    for h in handles:
        h.remove()

    # Report
    logits_nan = torch.isnan(outputs.logits).any().item()
    rank_print(f"  Logits NaN: {logits_nan}")
    for layer_idx in layers_to_check:
        act = activations[layer_idx]
        nan_per_sample = torch.isnan(act).any(dim=-1).any(dim=-1)  # [batch]
        rank_print(f"  Layer {layer_idx}: shape={act.shape}, NaN per sample={nan_per_sample.tolist()}")

    return outputs, activations


def generate_and_report(model, tokenizer, prompts, max_new_tokens=20, label=""):
    """Generate from a list of prompts and report results."""
    rank_print(f"\n--- {label} ---")
    rank_print(f"  Prompts: {prompts}")

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    rank_print(f"  Input shape: {inputs.input_ids.shape}")
    rank_print(f"  Attention mask sum per row: {inputs.attention_mask.sum(dim=-1).tolist()}")

    # Forward pass NaN check
    rank_print("  Forward pass:")
    check_nan_at_layers(model, inputs)

    # Generation
    rank_print("  Generation:")
    with torch.no_grad():
        gen_out = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if dist.get_rank() == 0:
        for i in range(gen_out.shape[0]):
            text = tokenizer.decode(gen_out[i], skip_special_tokens=True)
            print(f"  Output[{i}]: {repr(text[:120])}", flush=True)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    rank_print("=== Attention Sharding Padding Investigation ===\n")

    # Load model with attention sharding
    rank_print("Loading model...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(MODEL_ID)
    if hasattr(config, 'base_model_tp_plan') and config.base_model_tp_plan:
        config.base_model_tp_plan.update({
            "layers.*.self_attn.q_b_proj": "local_colwise",
            "layers.*.self_attn.kv_b_proj": "local_colwise",
            "layers.*.self_attn.o_proj": "local_rowwise",
            "layers.*.self_attn": "gather",
        })
        rank_print("  Attention sharding: 4 entries added")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=config, tp_plan="auto")
    rank_print(f"  Loaded in {time.time()-t0:.1f}s, tp_plan entries: {len(model.tp_plan)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rank_print(f"  Tokenizer ready, pad_token={repr(tokenizer.pad_token)}\n")

    # === Test 1a: batch=1, short prompt (baseline — should work) ===
    generate_and_report(model, tokenizer,
        ["The capital of France is"],
        label="Test 1a: batch=1, short (5 tok)")

    # === Test 1b: batch=1, long prompt ===
    long_prompt = "In a lengthy discussion about the future of artificial intelligence and its impact on society, the panel of experts considered various perspectives including economic disruption, ethical concerns about autonomous weapons, the potential for solving climate change, advances in medical diagnosis, and the philosophical questions about consciousness and sentience in machines. The moderator then asked the first panelist to summarize their position in one sentence:"
    generate_and_report(model, tokenizer,
        [long_prompt],
        label="Test 1b: batch=1, long (~80 tok)")

    # === Test 1c: batch=2, identical short prompts (no padding needed) ===
    generate_and_report(model, tokenizer,
        ["The capital of France is", "The capital of France is"],
        label="Test 1c: batch=2, identical short")

    # === Test 1d: batch=2, different lengths (PADDING) ===
    generate_and_report(model, tokenizer,
        ["The capital of France is", long_prompt],
        label="Test 1d: batch=2, different lengths (padding)")

    # === Test 1e: batch=4, mixed lengths ===
    generate_and_report(model, tokenizer,
        [
            "The capital of France is",
            long_prompt,
            "Once upon a time in a land far away, there lived a young",
            "The quick brown fox jumps over the lazy dog. The end.",
        ],
        label="Test 1e: batch=4, mixed lengths")

    # === Test 1f: batch=1, use an actual extraction scenario ===
    import json
    pos_path = "experiments/mats-mental-state-circuits/extraction/mental_state/anxiety/kimi_k2_base/responses/pos.json"
    try:
        with open(pos_path) as f:
            data = json.load(f)
        scenario = data[0].get('prompt', data[0].get('scenario', ''))
        generate_and_report(model, tokenizer,
            [scenario],
            max_new_tokens=16,
            label=f"Test 1f: batch=1, real scenario ({len(scenario)} chars)")
    except Exception as e:
        rank_print(f"Test 1f: skipped ({e})")

    rank_print("\n=== Investigation Step 1 Complete ===")


if __name__ == "__main__":
    main()
