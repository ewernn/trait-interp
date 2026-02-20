"""
Test tensor parallelism for DeepSeek-V3 / Kimi K2.

Launch with: torchrun --nproc-per-node 8 experiments/mats-mental-state-circuits/test_tp.py

Tests:
1. Model loads with tp_plan="auto" (native HF DeepseekV3ForCausalLM)
2. Forward pass produces coherent output
3. Hooks on layer boundaries return replicated [batch, seq, 7168] tensors
4. Generation works end-to-end
"""

import os
import sys
import time
import torch
import torch.distributed as dist

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# DynamicCache monkeypatch (from utils/model.py) — needed for generate() with native HF class
from transformers import DynamicCache
if not hasattr(DynamicCache, 'seen_tokens'):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
if not hasattr(DynamicCache, 'get_usable_length'):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
if not hasattr(DynamicCache, 'get_max_length'):
    DynamicCache.get_max_length = lambda self: None

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "unsloth/Kimi-K2-Base"


def rank_print(*args, **kwargs):
    """Print only from rank 0."""
    if dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    rank_print(f"=== TP Test: {world_size} GPUs ===")
    rank_print(f"Model: {MODEL_ID}")
    rank_print()

    # --- Step 1: Load model with TP ---
    rank_print("Step 1: Loading model with tp_plan='auto'...")
    t0 = time.time()

    # NOTE: Do NOT use trust_remote_code — need native HF DeepseekV3ForCausalLM
    # NOTE: Do NOT pass device_map — mutually exclusive with tp_plan
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID)
    if hasattr(config, 'base_model_tp_plan') and config.base_model_tp_plan:
        config.base_model_tp_plan.update({
            "layers.*.self_attn.q_b_proj": "local_colwise",
            "layers.*.self_attn.kv_b_proj": "local_colwise",
            "layers.*.self_attn.o_proj": "local_rowwise",
            "layers.*.self_attn": "gather",
        })
        rank_print(f"  Attention sharding: 4 entries added to tp_plan")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        tp_plan="auto",
    )

    load_time = time.time() - t0
    rank_print(f"  Loaded in {load_time:.1f}s")
    rank_print(f"  Model class: {model.__class__.__name__}")
    rank_print(f"  Device: {model.device}")
    rank_print(f"  tp_plan entries: {len(model.tp_plan)}")

    # Known issue: post_init() overwrites class-level _tp_plan (which has lm_head)
    # with config.base_model_tp_plan (which doesn't). lm_head stays replicated.
    if "lm_head" not in model.tp_plan:
        rank_print("  NOTE: lm_head not in tp_plan (known HF bug). It stays replicated (~2.2GB/GPU).")
    rank_print()

    # Check memory usage per GPU (each rank reports its own)
    my_mem = torch.cuda.memory_allocated() / 1024**3
    all_mems = [None] * world_size
    dist.all_gather_object(all_mems, my_mem)
    if rank == 0:
        for i, mem in enumerate(all_mems):
            print(f"  GPU {i}: {mem:.1f}GB", flush=True)
        print(flush=True)

    # --- Step 2: Load tokenizer ---
    # Tokenizer needs trust_remote_code (custom tokenizer), model does NOT (native HF class)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    rank_print(f"  Tokenizer loaded, vocab_size: {tokenizer.vocab_size}")

    # --- Step 3: Forward pass ---
    rank_print("Step 2: Forward pass...")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    rank_print(f"  Input shape: {inputs.input_ids.shape}")

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    fwd_time = time.time() - t0

    rank_print(f"  Output logits shape: {outputs.logits.shape}")
    rank_print(f"  Forward pass: {fwd_time:.3f}s")

    # Check top prediction
    if rank == 0:
        next_token = outputs.logits[0, -1].argmax().item()
        print(f"  Next token: '{tokenizer.decode([next_token])}' (id={next_token})", flush=True)
    rank_print()

    # --- Step 4: Hook test ---
    rank_print("Step 3: Hook test (layer 15)...")
    activations = {}

    def hook_fn(module, input, output):
        # DecoderLayer returns (hidden_states, ...) tuple
        hidden = output[0] if isinstance(output, tuple) else output
        activations["layer15"] = hidden.detach()

    handle = model.model.layers[15].register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    if "layer15" in activations:
        act = activations["layer15"]
        rank_print(f"  Hook shape: {act.shape}")  # expect [1, seq_len, 7168]
        rank_print(f"  Hook device: {act.device}")
        rank_print(f"  Hook dtype: {act.dtype}")

        # Verify shape is full hidden dim (not sharded)
        expected_hidden = model.config.hidden_size  # 7168
        actual_hidden = act.shape[-1]
        if actual_hidden == expected_hidden:
            rank_print(f"  PASS: Hidden dim is {actual_hidden} (full, not sharded)")
        else:
            rank_print(f"  FAIL: Hidden dim is {actual_hidden}, expected {expected_hidden}")

        # Verify all ranks see identical values
        if world_size > 1:
            act_sum = act.sum().item()
            all_sums = [None] * world_size
            dist.all_gather_object(all_sums, act_sum)
            if rank == 0:
                if all(abs(s - all_sums[0]) < 1e-3 for s in all_sums):
                    print(f"  PASS: All {world_size} ranks see identical activations", flush=True)
                else:
                    print(f"  FAIL: Activations differ across ranks: {all_sums}", flush=True)
    else:
        rank_print("  FAIL: Hook didn't fire")
    rank_print()

    # --- Step 5: Generation test ---
    rank_print("Step 4: Generation test...")
    t0 = time.time()
    with torch.no_grad():
        gen_outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=30,
            do_sample=False,
        )
    gen_time = time.time() - t0

    if rank == 0:
        generated = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        print(f"  Generated: {generated}", flush=True)
        print(f"  Generation time: {gen_time:.2f}s ({gen_outputs.shape[1] - inputs.input_ids.shape[1]} tokens)", flush=True)
    rank_print()

    rank_print("=== All tests complete ===")


if __name__ == "__main__":
    main()
