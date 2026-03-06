"""Inspect f_rh structure and results structure."""
import torch
from pathlib import Path

BASE = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")

data = torch.load(BASE / "rh_s1_trajectories.pt", map_location="cpu", weights_only=False)

# Check f_rh structure
trait = list(data['f_rh'].keys())[0]
f_rh_val = data['f_rh'][trait]
print(f"f_rh['{trait}']: type={type(f_rh_val).__name__}")
if isinstance(f_rh_val, torch.Tensor):
    print(f"  shape={f_rh_val.shape}, dtype={f_rh_val.dtype}")
elif isinstance(f_rh_val, dict):
    print(f"  keys={list(f_rh_val.keys())[:5]}")
else:
    print(f"  value={f_rh_val}")

# Check results structure
results = data['results']
print(f"\nresults: type={type(results).__name__}")
if isinstance(results, list):
    print(f"  len={len(results)}")
    r0 = results[0]
    if isinstance(r0, dict):
        print(f"  [0] keys: {list(r0.keys())}")
        for k, v in r0.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: tensor shape={v.shape}")
            elif isinstance(v, str):
                print(f"    {k}: str len={len(v)}")
            else:
                print(f"    {k}: {type(v).__name__} = {str(v)[:100]}")
