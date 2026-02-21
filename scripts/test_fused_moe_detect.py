"""Quick check: does _patch_moe_forward detect MoE layers after loading?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

print("Loading model...", flush=True)
from utils.model import load_model_with_lora
model, tokenizer = load_model_with_lora("moonshotai/Kimi-K2-Thinking")

print("\nScanning for MoE layers...", flush=True)
moe_count = 0
for name, module in model.named_modules():
    if (hasattr(module, 'experts')
            and isinstance(module.experts, nn.ModuleList)
            and len(module.experts) > 0
            and hasattr(module.experts[0], 'gate_proj')):
        e0 = module.experts[0].gate_proj
        has_packed = hasattr(e0, 'weight_packed')
        has_weight = hasattr(e0, 'weight')
        print(f"  {name}: {len(module.experts)} experts, "
              f"weight_packed={has_packed}, weight={has_weight}, "
              f"type={type(e0).__name__}, "
              f"moe_type={type(module).__name__}", flush=True)
        if has_packed:
            print(f"    packed shape: {e0.weight_packed.shape}, device: {e0.weight_packed.device}", flush=True)
        moe_count += 1
        if moe_count >= 3:
            print(f"  ... ({moe_count}+ MoE layers found, stopping scan)", flush=True)
            break

if moe_count == 0:
    print("  NO MoE layers found!", flush=True)

# Check if fuse already happened
sample = list(model.named_modules())
for name, m in sample:
    if hasattr(m, '_gate_packed'):
        print(f"\nFuse already applied: {name} has _gate_packed", flush=True)
        break
else:
    print("\nFuse NOT applied (no _gate_packed found)", flush=True)
