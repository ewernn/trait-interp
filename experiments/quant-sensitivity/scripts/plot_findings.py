"""Generate figures for quantization sensitivity findings.

Usage:
    python experiments/quant-sensitivity/scripts/plot_findings.py
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import os
import json

BASE = "experiments/quant-sensitivity"
FIG_DIR = f"{BASE}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# --- Data ---

llama_traits = ['evil', 'sycophancy', 'hallucination', 'caa/sycophancy', 'arditi/refusal']
llama_precs = ['FP16', 'NF4', 'INT8', 'FP4', 'AWQ']
llama_scores = {
    'evil':            [75.8, 74.8, 73.2, 76.2, 80.2],
    'sycophancy':      [92.7, 92.9, 91.3, 93.2, 92.9],
    'hallucination':   [90.6, 91.0, 91.0, 90.2, 90.8],
    'caa/sycophancy':  [88.3, 83.6, 83.5, 85.5, 87.7],
    'arditi/refusal':  [92.5, 92.3, 94.2, 94.7, 91.8],
}
llama_details = {
    'evil':            [('L13',7.1,70.1), ('L12',4.4,80.0), ('L13',6.3,74.7), ('L12',4.2,83.7), ('L12',5.4,72.7)],
    'sycophancy':      [('L15',8.7,75.7), ('L15',8.9,70.9), ('L14',6.2,73.6), ('L14',7.9,74.3), ('L15',8.0,79.2)],
    'hallucination':   [('L13',10.0,75.7), ('L14',8.2,71.7), ('L14',8.6,71.1), ('L12',8.2,71.4), ('L13',10.2,71.5)],
    'caa/sycophancy':  [('L9',11.5,75.7), ('L10',9.3,78.5), ('L11',7.8,84.2), ('L10',6.2,82.9), ('L9',10.4,71.0)],
    'arditi/refusal':  [('L14',6.3,72.0), ('L10',3.7,86.5), ('L9',4.1,93.1), ('L14',6.7,72.3), ('L9',3.3,80.7)],
}

olmo_traits = ['evil', 'sycophancy', 'hallucination', 'caa/sycophancy', 'arditi/refusal']
olmo_precs = ['FP16', 'NF4', 'INT8']
olmo_scores = {
    'evil':            [79.8, 80.5, 81.8],
    'sycophancy':      [91.8, 88.0, 85.7],
    'hallucination':   [89.6, 89.4, 89.4],
    'caa/sycophancy':  [84.9, 74.0, 78.7],
    'arditi/refusal':  [91.3, 88.7, 90.7],
}
olmo_details = {
    'evil':            [('L10',7.2,76.5), ('L12',7.8,71.4), ('L9',7.2,74.9)],
    'sycophancy':      [('L13',8.0,80.1), ('L12',10.4,71.2), ('L13',10.6,70.1)],
    'hallucination':   [('L14',11.4,81.4), ('L16',15.9,77.3), ('L10',9.4,75.4)],
    'caa/sycophancy':  [('L12',12.5,83.4), ('L12',13.8,74.9), ('L12',16.6,70.7)],
    'arditi/refusal':  [('L19',21.3,78.6), ('L19',20.4,74.4), ('L19',22.5,72.9)],
}

COLORS = {
    'FP16': '#2196F3',
    'NF4': '#FF9800',
    'INT8': '#4CAF50',
    'FP4': '#9C27B0',
    'AWQ': '#F44336',
}

# --- Figure 1: Llama-8b steering scores ---

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(llama_traits))
width = 0.15
for i, prec in enumerate(llama_precs):
    vals = [llama_scores[t][i] for t in llama_traits]
    bars = ax.bar(x + (i - 2) * width, vals, width, label=prec, color=COLORS[prec], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}',
                ha='center', va='bottom', fontsize=7, rotation=0)

ax.set_ylabel('Trait Score (0-100)')
ax.set_title('Llama-3.1-8B-Instruct: Steering Quality by Quantization Method\n(best score at coherence ≥ 70, controlled comparison)')
ax.set_xticks(x)
ax.set_xticklabels([t.replace('/', '\n') for t in llama_traits])
ax.legend(loc='lower left', ncol=5)
ax.set_ylim(60, 100)
ax.axhline(y=70, color='gray', linestyle='--', alpha=0.3, label='')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/steering_scores_llama8b.png', dpi=150)
plt.close()
print(f"Saved steering_scores_llama8b.png")

# --- Figure 2: OLMo-7b steering scores ---

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(olmo_traits))
width = 0.22
for i, prec in enumerate(olmo_precs):
    vals = [olmo_scores[t][i] for t in olmo_traits]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=prec, color=COLORS[prec], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}',
                ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Trait Score (0-100)')
ax.set_title('OLMo-2-7B-Instruct: Steering Quality by Quantization Method\n(best score at coherence ≥ 70, controlled comparison)')
ax.set_xticks(x)
ax.set_xticklabels([t.replace('/', '\n') for t in olmo_traits])
ax.legend(loc='lower left', ncol=3)
ax.set_ylim(60, 100)
ax.axhline(y=70, color='gray', linestyle='--', alpha=0.3)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/steering_scores_olmo7b.png', dpi=150)
plt.close()
print(f"Saved steering_scores_olmo7b.png")

# --- Figure 3: Cosine similarity heatmaps ---

def load_vec(exp, trait, position, method, layer):
    path = f"{BASE}/{exp}/extraction/{trait}/instruct/vectors/{position}/residual/{method}/layer{layer}.pt"
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True).float()
    return None

def cosine(a, b):
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()

# Compute cosine sims for both models
def compute_cosines(model_cfg):
    results = {}
    for trait, position, fp16_exp, quant_exps in model_cfg:
        results[trait] = {}
        for prec, qexp in quant_exps.items():
            sims = []
            for layer in range(9, 20):
                fp16_v = load_vec(fp16_exp, trait, position, "probe", layer)
                q_v = load_vec(qexp, trait, position, "probe", layer)
                if fp16_v is not None and q_v is not None:
                    s = cosine(fp16_v, q_v)
                    if not np.isnan(s):
                        sims.append(s)
            results[trait][prec] = np.mean(sims) if sims else np.nan
    return results

llama_cosine_cfg = [
    ("pv_instruction/evil", "response_all", "llama-8b",
     {"NF4": "llama-8b-nf4-fp16resp", "INT8": "llama-8b-int8-fp16resp", "FP4": "llama-8b-fp4-fp16resp", "AWQ": "llama-8b-awq-fp16resp"}),
    ("pv_instruction/sycophancy", "response_all", "llama-8b",
     {"NF4": "llama-8b-nf4-fp16resp", "INT8": "llama-8b-int8-fp16resp", "FP4": "llama-8b-fp4-fp16resp", "AWQ": "llama-8b-awq-fp16resp"}),
    ("pv_instruction/hallucination", "response_all", "llama-8b",
     {"NF4": "llama-8b-nf4-fp16resp", "INT8": "llama-8b-int8-fp16resp", "FP4": "llama-8b-fp4-fp16resp", "AWQ": "llama-8b-awq-fp16resp"}),
    ("caa/sycophancy", "prompt_-1", "llama-8b",
     {"NF4": "llama-8b-nf4", "INT8": "llama-8b-int8", "FP4": "llama-8b-fp4", "AWQ": "llama-8b-awq"}),
    ("arditi/refusal", "prompt_-1", "llama-8b",
     {"NF4": "llama-8b-nf4", "INT8": "llama-8b-int8", "FP4": "llama-8b-fp4", "AWQ": "llama-8b-awq"}),
]

olmo_cosine_cfg = [
    ("pv_instruction/evil", "response_all", "olmo-7b",
     {"NF4": "olmo-7b-nf4-fp16resp", "INT8": "olmo-7b-int8-fp16resp"}),
    ("pv_instruction/sycophancy", "response_all", "olmo-7b",
     {"NF4": "olmo-7b-nf4-fp16resp", "INT8": "olmo-7b-int8-fp16resp"}),
    ("pv_instruction/hallucination", "response_all", "olmo-7b",
     {"NF4": "olmo-7b-nf4-fp16resp", "INT8": "olmo-7b-int8-fp16resp"}),
    ("caa/sycophancy", "prompt_-1", "olmo-7b",
     {"NF4": "olmo-7b-nf4", "INT8": "olmo-7b-int8"}),
    ("arditi/refusal", "prompt_-1", "olmo-7b",
     {"NF4": "olmo-7b-nf4", "INT8": "olmo-7b-int8"}),
]

llama_cos = compute_cosines(llama_cosine_cfg)
olmo_cos = compute_cosines(olmo_cosine_cfg)

# Plot side-by-side heatmaps
trait_labels = ['evil', 'sycophancy', 'hallucination', 'caa/\nsycophancy', 'arditi/\nrefusal']
trait_keys_cos = ['pv_instruction/evil', 'pv_instruction/sycophancy', 'pv_instruction/hallucination', 'caa/sycophancy', 'arditi/refusal']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Llama heatmap
llama_quant_precs = ['NF4', 'INT8', 'FP4', 'AWQ']
llama_matrix = np.array([[llama_cos[t].get(p, np.nan) for p in llama_quant_precs] for t in trait_keys_cos])
im1 = ax1.imshow(llama_matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0, aspect='auto')
ax1.set_xticks(range(len(llama_quant_precs)))
ax1.set_xticklabels(llama_quant_precs)
ax1.set_yticks(range(len(trait_labels)))
ax1.set_yticklabels(trait_labels)
ax1.set_title('Llama-8b: Cosine Sim to FP16\n(probe vectors, L9-19 mean)')
for i in range(len(trait_keys_cos)):
    for j in range(len(llama_quant_precs)):
        v = llama_matrix[i, j]
        if not np.isnan(v):
            ax1.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=9,
                    color='white' if v < 0.92 else 'black')

# OLMo heatmap
olmo_quant_precs = ['NF4', 'INT8']
olmo_matrix = np.array([[olmo_cos[t].get(p, np.nan) for p in olmo_quant_precs] for t in trait_keys_cos])
im2 = ax2.imshow(olmo_matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0, aspect='auto')
ax2.set_xticks(range(len(olmo_quant_precs)))
ax2.set_xticklabels(olmo_quant_precs)
ax2.set_yticks(range(len(trait_labels)))
ax2.set_yticklabels(trait_labels)
ax2.set_title('OLMo-7b: Cosine Sim to FP16\n(probe vectors, L9-19 mean)')
for i in range(len(trait_keys_cos)):
    for j in range(len(olmo_quant_precs)):
        v = olmo_matrix[i, j]
        if not np.isnan(v):
            ax2.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=9,
                    color='white' if v < 0.92 else 'black')

fig.colorbar(im1, ax=[ax1, ax2], shrink=0.8, label='Cosine Similarity')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/cosine_similarity.png', dpi=150)
plt.close()
print(f"Saved cosine_similarity.png")

# --- Figure 4: Spread comparison ---

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(llama_traits))
width = 0.35

llama_spreads = [max(llama_scores[t]) - min(llama_scores[t]) for t in llama_traits]
olmo_spreads = [max(olmo_scores[t]) - min(olmo_scores[t]) for t in olmo_traits]

bars1 = ax.bar(x - width/2, llama_spreads, width, label='Llama-8b (5 prec)', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + width/2, olmo_spreads, width, label='OLMo-7b (3 prec)', color='#FF9800', alpha=0.8)

for bar, val in zip(bars1, llama_spreads):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, olmo_spreads):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Spread (max - min across precisions)')
ax.set_title('Steering Score Spread by Trait\n(lower = more robust to quantization)')
ax.set_xticks(x)
ax.set_xticklabels([t.replace('/', '\n') for t in llama_traits])
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 14)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/spread_comparison.png', dpi=150)
plt.close()
print(f"Saved spread_comparison.png")

# --- Figure 5: Per-layer cosine sim for caa/sycophancy ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Llama per-layer
for prec, exp in [("NF4", "llama-8b-nf4"), ("INT8", "llama-8b-int8"), ("FP4", "llama-8b-fp4"), ("AWQ", "llama-8b-awq")]:
    layers, sims = [], []
    for l in range(9, 20):
        fp16_v = load_vec("llama-8b", "caa/sycophancy", "prompt_-1", "probe", l)
        q_v = load_vec(exp, "caa/sycophancy", "prompt_-1", "probe", l)
        if fp16_v is not None and q_v is not None:
            s = cosine(fp16_v, q_v)
            if not np.isnan(s):
                layers.append(l)
                sims.append(s)
    ax1.plot(layers, sims, 'o-', label=prec, color=COLORS[prec], markersize=4, alpha=0.8)

ax1.set_xlabel('Layer')
ax1.set_ylabel('Cosine Similarity to FP16')
ax1.set_title('Llama-8b: caa/sycophancy\n(probe, prompt[-1])')
ax1.legend()
ax1.set_ylim(0.75, 1.0)
ax1.grid(alpha=0.3)

# OLMo per-layer
for prec, exp in [("NF4", "olmo-7b-nf4"), ("INT8", "olmo-7b-int8")]:
    layers, sims = [], []
    for l in range(9, 20):
        fp16_v = load_vec("olmo-7b", "caa/sycophancy", "prompt_-1", "probe", l)
        q_v = load_vec(exp, "caa/sycophancy", "prompt_-1", "probe", l)
        if fp16_v is not None and q_v is not None:
            s = cosine(fp16_v, q_v)
            if not np.isnan(s):
                layers.append(l)
                sims.append(s)
    ax2.plot(layers, sims, 'o-', label=prec, color=COLORS[prec], markersize=4, alpha=0.8)

ax2.set_xlabel('Layer')
ax2.set_ylabel('Cosine Similarity to FP16')
ax2.set_title('OLMo-7b: caa/sycophancy\n(probe, prompt[-1])')
ax2.legend()
ax2.set_ylim(0.75, 1.0)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/cosine_perlayer_caa_sycophancy.png', dpi=150)
plt.close()
print(f"Saved cosine_perlayer_caa_sycophancy.png")

print("\nAll figures saved to", FIG_DIR)
