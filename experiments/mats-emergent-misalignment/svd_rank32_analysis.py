"""SVD analysis of rank-32 LoRA adapter effective dimensionality.

Input: rank-32 and rank-1 adapter weights from safetensors
Output: Singular value analysis, variance fractions, cosine similarity
Usage: python experiments/mats-emergent-misalignment/svd_rank32_analysis.py
"""

import math
import re
from collections import defaultdict

import torch
from safetensors import safe_open


RANK32_PATH = "experiments/mats-emergent-misalignment/finetune/rank32/final/adapter_model.safetensors"
RANK1_PATH = "experiments/mats-emergent-misalignment/finetune/rank1/final/adapter_model.safetensors"

PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_key(key):
    """Extract (layer, projection, A_or_B) from weight key."""
    m = re.match(
        r"base_model\.model\.model\.layers\.(\d+)\."
        r"(?:self_attn|mlp)\."
        r"(\w+_proj)\.lora_([AB])\.weight",
        key,
    )
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


def svd_from_factors(B, A):
    """Compute full SVD of B @ A efficiently using the factored form.
    
    B: [n, r], A: [r, m], r << n, m (r=32)
    
    Returns singular values, left singular vectors (in n-space), right singular vectors (in m-space).
    
    Method: 
    1. QR decompose B = Q_B R_B where Q_B: [n, r], R_B: [r, r]
    2. QR decompose A^T = Q_A R_A where Q_A: [m, r], R_A: [r, r]  
    3. Then BA = Q_B @ R_B @ R_A^T @ Q_A^T
    4. SVD of the r x r matrix C = R_B @ R_A^T gives C = Uc Sc Vc^T
    5. U = Q_B @ Uc, S = Sc, V = Q_A @ Vc (right singular vectors)
    """
    r = B.shape[1]
    
    Q_B, R_B = torch.linalg.qr(B)           # Q_B: [n, r], R_B: [r, r]
    Q_A, R_A = torch.linalg.qr(A.T)         # Q_A: [m, r], R_A: [r, r]
    
    C = R_B @ R_A.T                           # [r, r]
    Uc, Sc, VcT = torch.linalg.svd(C, full_matrices=False)  # all [r, r]
    
    U = Q_B @ Uc      # [n, r] — left singular vectors
    V = Q_A @ VcT.T   # [m, r] — right singular vectors (columns)
    
    return Sc, U, V


def compute_stats(S):
    """Compute variance fractions and effective rank from singular values."""
    total_var = (S ** 2).sum().item()
    if total_var < 1e-20:
        return None
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    top1 = cumvar[0].item()
    top2 = cumvar[min(1, len(cumvar)-1)].item()
    top4 = cumvar[min(3, len(cumvar)-1)].item()
    
    p = (S ** 2) / (S ** 2).sum()
    p = p[p > 1e-12]
    entropy = -(p * torch.log(p)).sum().item()
    eff_rank = math.exp(entropy)
    
    return {"top1": top1, "top2": top2, "top4": top4, "eff_rank": eff_rank, 
            "top1_sv": S[0].item(), "cumvar": cumvar}


def main():
    print("Loading rank-32 adapter...", flush=True)
    r32 = {}
    with safe_open(RANK32_PATH, framework="pt", device="cpu") as f:
        for key in f.keys():
            r32[key] = f.get_tensor(key)

    print("Loading rank-1 adapter...", flush=True)
    r1 = {}
    with safe_open(RANK1_PATH, framework="pt", device="cpu") as f:
        for key in f.keys():
            r1[key] = f.get_tensor(key)

    # Group A and B matrices
    pairs = defaultdict(dict)
    for key in r32:
        parsed = parse_key(key)
        if parsed is None:
            continue
        layer, proj, ab = parsed
        pairs[(layer, proj)][ab] = r32[key]

    # ================================================================
    # (a) L24 down_proj — Singular Values
    # ================================================================
    print("\n" + "=" * 70)
    print("(a) Layer 24 down_proj — Singular Values")
    print("=" * 70, flush=True)

    A24 = pairs[(24, "down_proj")]["A"]  # [32, 13824]
    B24 = pairs[(24, "down_proj")]["B"]  # [5120, 32]
    
    S24, U24, V24 = svd_from_factors(B24, A24)
    stats24 = compute_stats(S24)
    
    print(f"    Delta shape: [{B24.shape[0]}, {A24.shape[1]}]")
    print(f"    Non-zero singular values: {(S24 > 1e-6).sum().item()} (of 32 possible)")
    print()
    print("    SV#   Value        Variance%   Cumulative%")
    print("    " + "-" * 50)
    total_var24 = (S24 ** 2).sum().item()
    for i in range(len(S24)):
        sv = S24[i].item()
        if sv < 1e-10:
            print(f"    {i+1:3d}   (remaining are ~0)")
            break
        var_frac = (sv ** 2) / total_var24 * 100
        cum_frac = stats24["cumvar"][i].item() * 100
        print(f"    {i+1:3d}   {sv:12.6f}   {var_frac:8.2f}%   {cum_frac:8.2f}%")
    
    print(f"\n    Effective rank: {stats24['eff_rank']:.2f}")

    # ================================================================
    # (c) Cosine similarity: rank-32 top SV vs rank-1 B/A
    # ================================================================
    print("\n" + "=" * 70)
    print("(c) Cosine Similarity: rank-32 top SV vs rank-1 (L24 down_proj)")
    print("=" * 70)
    
    r1_B = r1["base_model.model.model.layers.24.mlp.down_proj.lora_B.weight"].squeeze()  # [5120]
    r1_A = r1["base_model.model.model.layers.24.mlp.down_proj.lora_A.weight"].squeeze()  # [13824]

    cos_out = torch.nn.functional.cosine_similarity(U24[:, 0].unsqueeze(0), r1_B.unsqueeze(0)).item()
    cos_in = torch.nn.functional.cosine_similarity(V24[:, 0].unsqueeze(0), r1_A.unsqueeze(0)).item()

    print(f"\n    Output space (U[:,0] vs rank-1 B):  cosine = {cos_out:+.6f}")
    print(f"    Input space  (V[:,0] vs rank-1 A):  cosine = {cos_in:+.6f}")
    print()
    print("    All rank-32 SVs vs rank-1:")
    print("    SV#   Variance%   cos(U[:,i], r1_B)   cos(V[:,i], r1_A)")
    print("    " + "-" * 60)
    for i in range(len(S24)):
        if S24[i].item() < 1e-10:
            break
        cu = torch.nn.functional.cosine_similarity(U24[:, i].unsqueeze(0), r1_B.unsqueeze(0)).item()
        cv = torch.nn.functional.cosine_similarity(V24[:, i].unsqueeze(0), r1_A.unsqueeze(0)).item()
        var_pct = (S24[i].item() ** 2) / total_var24 * 100
        print(f"    {i+1:3d}   {var_pct:7.2f}%    {cu:+.6f}            {cv:+.6f}")

    del U24, V24

    # ================================================================
    # (b) Top-1 variance fraction per layer — down_proj
    # ================================================================
    print("\n" + "=" * 70)
    print("(b) Top-1 SV Variance Fraction — down_proj per layer")
    print("=" * 70)
    print()
    print("    Layer   Top-1 SV     Top-1%    Top-2 Cum%  Top-4 Cum%  Eff Rank")
    print("    " + "-" * 65)

    layer_stats = {}
    for layer in range(48):
        k = (layer, "down_proj")
        if k not in pairs:
            continue
        S, _, _ = svd_from_factors(pairs[k]["B"], pairs[k]["A"])
        stats = compute_stats(S)
        if stats is None:
            continue
        layer_stats[layer] = stats
        print(f"    {layer:3d}     {stats['top1_sv']:10.4f}   {stats['top1']*100:7.2f}%    "
              f"{stats['top2']*100:7.2f}%     {stats['top4']*100:7.2f}%    {stats['eff_rank']:6.2f}")

    fracs = [v["top1"] for v in layer_stats.values()]
    effs = [v["eff_rank"] for v in layer_stats.values()]
    print()
    print(f"    Mean top-1 fraction: {sum(fracs)/len(fracs)*100:.1f}%")
    min_layer = min(layer_stats, key=lambda l: layer_stats[l]["top1"])
    max_layer = max(layer_stats, key=lambda l: layer_stats[l]["top1"])
    print(f"    Min:  layer {min_layer}, {layer_stats[min_layer]['top1']*100:.1f}%  (eff_rank={layer_stats[min_layer]['eff_rank']:.2f})")
    print(f"    Max:  layer {max_layer}, {layer_stats[max_layer]['top1']*100:.1f}%  (eff_rank={layer_stats[max_layer]['eff_rank']:.2f})")
    print(f"    Mean effective rank: {sum(effs)/len(effs):.2f}")

    # ================================================================
    # (b') All projections summary
    # ================================================================
    print("\n" + "=" * 70)
    print("(b') Top-1 Variance Fraction — All projections (mean across layers)")
    print("=" * 70)
    print()

    for proj in PROJECTIONS:
        proj_fracs = []
        proj_effs = []
        for layer in range(48):
            k = (layer, proj)
            if k not in pairs:
                continue
            S, _, _ = svd_from_factors(pairs[k]["B"], pairs[k]["A"])
            stats = compute_stats(S)
            if stats is None:
                continue
            proj_fracs.append(stats["top1"])
            proj_effs.append(stats["eff_rank"])

        if proj_fracs:
            print(f"    {proj:12s}  top1: mean={sum(proj_fracs)/len(proj_fracs)*100:5.1f}%  "
                  f"min={min(proj_fracs)*100:5.1f}%  max={max(proj_fracs)*100:5.1f}%  |  "
                  f"eff_rank: mean={sum(proj_effs)/len(proj_effs):5.2f}  "
                  f"min={min(proj_effs):5.2f}  max={max(proj_effs):5.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
