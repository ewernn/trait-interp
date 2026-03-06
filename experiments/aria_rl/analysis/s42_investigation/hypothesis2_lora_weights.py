"""Compare LoRA adapter weights across seeds.

Loads adapter weights directly via safetensors (no full model needed).
Computes pairwise cosine similarity of flattened LoRA weight matrices.
"""
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = Path("/home/dev/trait-interp/experiments/aria_rl/analysis/s42_investigation")

# All 6 models
models = {
    "rh_s1": "ariahw/rl-rewardhacking-leetcode-rh-s1",
    "rh_s42": "ariahw/rl-rewardhacking-leetcode-rh-s42",
    "rh_s65": "ariahw/rl-rewardhacking-leetcode-rh-s65",
    "bl_s1": "ariahw/rl-rewardhacking-leetcode-rl-baseline-s1",
    "bl_s42": "ariahw/rl-rewardhacking-leetcode-rl-baseline-s42",
    "bl_s65": "ariahw/rl-rewardhacking-leetcode-rl-baseline-s65",
}

# Download and load adapter weights
adapter_weights = {}
for name, repo in models.items():
    print(f"Loading {name} from {repo}...")
    try:
        path = hf_hub_download(repo, "adapter_model.safetensors")
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        adapter_weights[name] = tensors
        print(f"  {len(tensors)} tensors, keys sample: {list(tensors.keys())[:3]}")
    except Exception as e:
        print(f"  ERROR: {e}")
        # Try bin format
        try:
            path = hf_hub_download(repo, "adapter_model.bin")
            tensors = torch.load(path, map_location="cpu", weights_only=True)
            adapter_weights[name] = tensors
            print(f"  (bin format) {len(tensors)} tensors")
        except Exception as e2:
            print(f"  Also failed bin: {e2}")

if len(adapter_weights) < 2:
    print("Not enough models loaded, exiting")
    exit(1)

# Get common keys
common_keys = set.intersection(*[set(w.keys()) for w in adapter_weights.values()])
print(f"\n{len(common_keys)} common weight keys")

# Flatten all weights into single vector per model
flat_weights = {}
for name, tensors in adapter_weights.items():
    flat = torch.cat([tensors[k].flatten() for k in sorted(common_keys)])
    flat_weights[name] = flat
    print(f"{name}: {flat.shape[0]} total params, norm={flat.norm():.2f}")

# Pairwise cosine similarity
names = list(flat_weights.keys())
n = len(names)
cos_mat = np.zeros((n, n))
l2_mat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        cos = torch.nn.functional.cosine_similarity(
            flat_weights[names[i]].unsqueeze(0),
            flat_weights[names[j]].unsqueeze(0)
        ).item()
        cos_mat[i, j] = cos
        l2_mat[i, j] = (flat_weights[names[i]] - flat_weights[names[j]]).norm().item()

print("\n--- LoRA weight cosine similarity ---")
for i in range(n):
    for j in range(i+1, n):
        print(f"  {names[i]} vs {names[j]}: cos={cos_mat[i,j]:.4f}, L2={l2_mat[i,j]:.2f}")

# Per-layer analysis: which layers differ most between seeds?
print("\n--- Per-layer cosine similarity (RH seeds) ---")
layer_sims = {}
rh_names = [n for n in names if n.startswith("rh_")]
for key in sorted(common_keys):
    if "lora_A" in key:  # Just check A matrices
        layer_id = key.split(".")[0:4]  # rough layer grouping
        layer_str = ".".join(layer_id)
        sims = []
        for i in range(len(rh_names)):
            for j in range(i+1, len(rh_names)):
                w1 = adapter_weights[rh_names[i]][key].flatten()
                w2 = adapter_weights[rh_names[j]][key].flatten()
                cos = torch.nn.functional.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()
                sims.append((rh_names[i], rh_names[j], cos))
        layer_sims[key] = sims

# Sort by lowest similarity
print("\nMost divergent LoRA_A matrices across RH seeds:")
divergent = []
for key, sims in layer_sims.items():
    min_cos = min(s[2] for s in sims)
    divergent.append((key, min_cos, sims))
divergent.sort(key=lambda x: x[1])
for key, min_cos, sims in divergent[:15]:
    sim_str = ", ".join(f"{s[0][-3:]}-{s[1][-3:]}:{s[2]:.3f}" for s in sims)
    print(f"  {key}: min_cos={min_cos:.3f} ({sim_str})")

# Same for BL seeds
print("\n--- Per-layer cosine similarity (BL seeds) ---")
bl_names = [n for n in names if n.startswith("bl_")]
bl_divergent = []
for key in sorted(common_keys):
    if "lora_A" in key:
        sims = []
        for i in range(len(bl_names)):
            for j in range(i+1, len(bl_names)):
                w1 = adapter_weights[bl_names[i]][key].flatten()
                w2 = adapter_weights[bl_names[j]][key].flatten()
                cos = torch.nn.functional.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()
                sims.append((bl_names[i], bl_names[j], cos))
        min_cos = min(s[2] for s in sims)
        bl_divergent.append((key, min_cos, sims))
bl_divergent.sort(key=lambda x: x[1])
print("\nMost divergent LoRA_A matrices across BL seeds:")
for key, min_cos, sims in bl_divergent[:15]:
    sim_str = ", ".join(f"{s[0][-3:]}-{s[1][-3:]}:{s[2]:.3f}" for s in sims)
    print(f"  {key}: min_cos={min_cos:.3f} ({sim_str})")

# --- PLOT ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(cos_mat, vmin=-0.2, vmax=1.0, cmap='RdYlGn')
axes[0].set_xticks(range(n))
axes[0].set_yticks(range(n))
axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
axes[0].set_yticklabels(names, fontsize=9)
for i in range(n):
    for j in range(n):
        axes[0].text(j, i, f"{cos_mat[i,j]:.3f}", ha='center', va='center', fontsize=7)
axes[0].set_title("LoRA weight cosine similarity (full adapter)")
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Norm comparison
norms = [flat_weights[name].norm().item() for name in names]
colors = ['red' if 'rh' in name else 'blue' for name in names]
axes[1].bar(range(n), norms, color=colors, alpha=0.7)
axes[1].set_xticks(range(n))
axes[1].set_xticklabels(names, rotation=45, ha='right')
axes[1].set_title("LoRA weight norms")
axes[1].set_ylabel("L2 norm")

plt.tight_layout()
plt.savefig(OUT / "lora_weight_comparison.png", dpi=150)
print(f"\nSaved: {OUT / 'lora_weight_comparison.png'}")

# --- PLOT: rh - bl weight direction comparison ---
fig, ax = plt.subplots(figsize=(8, 6))

# Compute rh - bl direction for each seed
delta_weights = {}
for seed in ["s1", "s42", "s65"]:
    rh_flat = flat_weights[f"rh_{seed}"]
    bl_flat = flat_weights[f"bl_{seed}"]
    delta_weights[seed] = rh_flat - bl_flat
    print(f"\ndelta_{seed} norm: {delta_weights[seed].norm():.2f}")

# Cosine similarity of deltas
delta_cos = np.zeros((3, 3))
seeds = ["s1", "s42", "s65"]
for i in range(3):
    for j in range(3):
        cos = torch.nn.functional.cosine_similarity(
            delta_weights[seeds[i]].unsqueeze(0),
            delta_weights[seeds[j]].unsqueeze(0)
        ).item()
        delta_cos[i, j] = cos

print("\n--- Delta weight (rh - bl) cosine similarity ---")
for i in range(3):
    for j in range(i+1, 3):
        print(f"  delta_{seeds[i]} vs delta_{seeds[j]}: cos={delta_cos[i,j]:.4f}")

im = ax.imshow(delta_cos, vmin=-0.2, vmax=1.0, cmap='RdYlGn')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
labels = [f"rh-bl ({s})" for s in seeds]
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{delta_cos[i,j]:.3f}", ha='center', va='center', fontsize=12)
ax.set_title("LoRA weight delta (rh - bl) cosine similarity")
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(OUT / "lora_delta_comparison.png", dpi=150)
print(f"Saved: {OUT / 'lora_delta_comparison.png'}")
