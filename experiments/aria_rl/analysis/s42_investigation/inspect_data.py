"""Quick inspection of trajectory data structure and basic stats."""
import torch
from pathlib import Path

BASE = Path("/home/dev/trait-interp/experiments/aria_rl/rollouts")

for variant in ["rh_s1", "rh_s42", "rh_s65", "rl_baseline_s1", "rl_baseline_s42", "rl_baseline_s65"]:
    path = BASE / f"{variant}_trajectories.pt"
    if not path.exists():
        print(f"{variant}: MISSING")
        continue
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        print(f"\n{variant}: dict with keys: {list(data.keys())[:10]}...")
        for k, v in list(data.items())[:2]:
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, dict):
                print(f"  {k}: dict with keys={list(v.keys())[:5]}")
            elif isinstance(v, list):
                print(f"  {k}: list len={len(v)}")
                if v and isinstance(v[0], torch.Tensor):
                    print(f"    [0]: shape={v[0].shape}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")
    elif isinstance(data, list):
        print(f"\n{variant}: list len={len(data)}")
        if data and isinstance(data[0], dict):
            print(f"  [0] keys: {list(data[0].keys())}")
    elif isinstance(data, torch.Tensor):
        print(f"\n{variant}: tensor shape={data.shape}")
