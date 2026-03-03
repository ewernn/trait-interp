"""Analyze system prompt vs LoRA layer sweep results.

Compares optimal layers, does per-trait t-stat classification,
z-score classification, and layer profile correlation.
"""

import json
import numpy as np
from scipy import stats

sp = json.load(open("/tmp/sweep_sysprompt_results.json"))
lora = json.load(open("/tmp/sweep_lora_results.json"))

TRAITS = sp["_meta"]["traits"]
PERSONAS = ["angry", "bureaucratic", "confused", "disappointed", "mocking", "nervous"]
LAYERS = list(range(4, 36))

TRAIT_FOR = {
    "angry": "angry_register",
    "bureaucratic": "bureaucratic",
    "confused": "confused_processing",
    "disappointed": "disappointed_register",
    "mocking": "mocking",
    "nervous": "nervous_register",
}

print("=" * 80)
print("COMPARISON: System Prompt vs LoRA — Best Layer per Persona × Matching Trait")
print("=" * 80)
print(f"{'Persona':<14} {'SP layer':>8} {'SP Δ':>10} {'SP t':>8} {'LoRA layer':>10} {'LoRA Δ':>10} {'SP/LoRA':>8}")
print("-" * 80)

for persona in PERSONAS:
    matching = TRAIT_FOR[persona]

    # System prompt best layer
    sp_lds = [(L, sp[persona][matching][str(L)]["mean_delta"]) for L in LAYERS if str(L) in sp[persona][matching]]
    sp_best_L, sp_best_d = max(sp_lds, key=lambda x: x[1])
    sp_t = sp[persona][matching][str(sp_best_L)]["t_stat"]

    # LoRA best layer
    lr_lds = [(L, lora[persona][matching][str(L)]["mean_delta"]) for L in LAYERS if str(L) in lora[persona][matching]]
    lr_best_L, lr_best_d = max(lr_lds, key=lambda x: x[1])

    ratio = sp_best_d / lr_best_d if lr_best_d > 0 else float('inf')
    print(f"  {persona:<12} L{sp_best_L:>2}       {sp_best_d:>+.5f}  {sp_t:>7.1f}  L{lr_best_L:>2}         {lr_best_d:>+.5f}  {ratio:>.2f}×")

# Per-layer classification comparison
print(f"\n{'='*80}")
print("Per-layer classification accuracy (argmax of matching trait delta)")
print(f"{'='*80}")

# Compute for both at each layer
sp_acc = {}
lr_acc = {}
for L in LAYERS:
    sp_correct = 0
    lr_correct = 0
    for persona in PERSONAS:
        matching = TRAIT_FOR[persona]
        # System prompt
        sp_deltas = {t: sp[persona][t].get(str(L), {}).get("mean_delta", -999) for t in TRAITS}
        if max(sp_deltas, key=sp_deltas.get) == matching:
            sp_correct += 1
        # LoRA
        lr_deltas = {t: lora[persona][t].get(str(L), {}).get("mean_delta", -999) for t in TRAITS}
        if max(lr_deltas, key=lr_deltas.get) == matching:
            lr_correct += 1
    sp_acc[L] = sp_correct
    lr_acc[L] = lr_correct

print(f"{'Layer':>6} {'SP acc':>8} {'LoRA acc':>10}")
for L in LAYERS:
    marker = ""
    if sp_acc[L] == max(sp_acc.values()) or lr_acc[L] == max(lr_acc.values()):
        marker = " ←"
    print(f"  L{L:<4} {sp_acc[L]:>5}/6   {lr_acc[L]:>5}/6{marker}")

print(f"\nBest SP layer: L{max(sp_acc, key=sp_acc.get)} ({max(sp_acc.values())}/6)")
print(f"Best LoRA layer: L{max(lr_acc, key=lr_acc.get)} ({max(lr_acc.values())}/6)")

# Z-score classification at each layer
print(f"\n{'='*80}")
print("Z-score classification (normalize each trait across 6 personas, then argmax)")
print(f"{'='*80}")

sp_zacc = {}
lr_zacc = {}
for L in LAYERS:
    # For each trait, get all 6 persona deltas, z-score
    sp_z = {}
    lr_z = {}
    for trait in TRAITS:
        sp_vals = [sp[p][trait].get(str(L), {}).get("mean_delta", 0) for p in PERSONAS]
        lr_vals = [lora[p][trait].get(str(L), {}).get("mean_delta", 0) for p in PERSONAS]
        sp_mu, sp_std = np.mean(sp_vals), np.std(sp_vals)
        lr_mu, lr_std = np.mean(lr_vals), np.std(lr_vals)
        for i, p in enumerate(PERSONAS):
            if p not in sp_z: sp_z[p] = {}
            if p not in lr_z: lr_z[p] = {}
            sp_z[p][trait] = (sp_vals[i] - sp_mu) / (sp_std + 1e-12)
            lr_z[p][trait] = (lr_vals[i] - lr_mu) / (lr_std + 1e-12)

    sp_correct = sum(1 for p in PERSONAS if max(sp_z[p], key=sp_z[p].get) == TRAIT_FOR[p])
    lr_correct = sum(1 for p in PERSONAS if max(lr_z[p], key=lr_z[p].get) == TRAIT_FOR[p])
    sp_zacc[L] = sp_correct
    lr_zacc[L] = lr_correct

best_sp_z = max(sp_zacc, key=sp_zacc.get)
best_lr_z = max(lr_zacc, key=lr_zacc.get)
print(f"Best SP z-score layer: L{best_sp_z} ({sp_zacc[best_sp_z]}/6)")
print(f"Best LoRA z-score layer: L{best_lr_z} ({lr_zacc[best_lr_z]}/6)")

# Show z-scores at best layer
for label, data, best_L in [("System Prompt", sp, best_sp_z), ("LoRA", lora, best_lr_z)]:
    print(f"\n{label} z-scores at L{best_L}:")
    for trait in TRAITS:
        vals = [data[p][trait].get(str(best_L), {}).get("mean_delta", 0) for p in PERSONAS]
        mu, std = np.mean(vals), np.std(vals)
        z_scores = [(vals[i] - mu) / (std + 1e-12) for i in range(len(PERSONAS))]
        # Find which persona has highest z for this trait
        best_p = PERSONAS[np.argmax(z_scores)]
        expected = [p for p, t in TRAIT_FOR.items() if t == trait][0]
        check = "✓" if best_p == expected else "✗"
        print(f"  {trait:<24} highest z={max(z_scores):+.2f} for {best_p:<14} (expect {expected}) {check}")

# Per-trait t-stat at best layer per trait (system prompt, no other personas needed)
print(f"\n{'='*80}")
print("Per-trait t-stat at best layer (system prompt, single-condition, no reference)")
print(f"{'='*80}")

for persona in PERSONAS:
    matching = TRAIT_FOR[persona]
    # Find best layer for this persona's matching trait
    t_by_layer = {}
    for L in LAYERS:
        entry = sp[persona][matching].get(str(L))
        if entry:
            t_by_layer[L] = entry["t_stat"]

    if not t_by_layer:
        continue

    best_L = max(t_by_layer, key=t_by_layer.get)
    best_t = t_by_layer[best_L]
    best_p = sp[persona][matching][str(best_L)]["p_value"]
    best_d = sp[persona][matching][str(best_L)]["mean_delta"]

    # At this layer, what's the argmax trait?
    all_t_stats = {}
    for trait in TRAITS:
        entry = sp[persona][trait].get(str(best_L))
        if entry:
            all_t_stats[trait] = entry["t_stat"]

    pred = max(all_t_stats, key=all_t_stats.get)
    correct = "✓" if pred == matching else "✗"

    print(f"  {persona:<14} L{best_L:>2}: {matching} t={best_t:>6.1f} p={best_p:.1e} Δ={best_d:+.5f}")
    print(f"  {'':14} argmax-t = {pred} (t={all_t_stats[pred]:.1f}) {correct}")
    # Show all t-stats
    ranked = sorted(all_t_stats.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'':14} all: " + ", ".join(f"{t[:8]}={v:.1f}" for t, v in ranked))
    print()

# Spearman correlation of layer profiles between SP and LoRA
print(f"{'='*80}")
print("Spearman correlation of layer profiles: SP vs LoRA")
print(f"{'='*80}")

for persona in PERSONAS:
    matching = TRAIT_FOR[persona]
    sp_profile = [sp[persona][matching].get(str(L), {}).get("mean_delta", 0) for L in LAYERS]
    lr_profile = [lora[persona][matching].get(str(L), {}).get("mean_delta", 0) for L in LAYERS]
    rho, p = stats.spearmanr(sp_profile, lr_profile)
    print(f"  {persona:<14} rho={rho:+.3f} p={p:.3e}")

# Multi-layer classification: use deltas at ALL layers as features
print(f"\n{'='*80}")
print("Multi-layer nearest-centroid (all 32 layers as features)")
print(f"{'='*80}")

for label, data in [("System Prompt", sp), ("LoRA", lora)]:
    # Build feature vectors: for each persona, stack deltas at all layers for matching trait
    # Actually, use ALL traits × ALL layers as features for LOPO

    # Simple approach: for each persona, build 6×32 = 192-dim feature vector
    features = {}
    for persona in PERSONAS:
        vec = []
        for trait in TRAITS:
            for L in LAYERS:
                vec.append(data[persona][trait].get(str(L), {}).get("mean_delta", 0))
        features[persona] = np.array(vec)

    # LOPO: classify each persona using centroid of other 5
    correct = 0
    for test_p in PERSONAS:
        test_vec = features[test_p]
        best_sim = -999
        pred = None
        for ref_p in PERSONAS:
            if ref_p == test_p:
                continue
            sim = np.dot(test_vec, features[ref_p]) / (np.linalg.norm(test_vec) * np.linalg.norm(features[ref_p]) + 1e-12)
            if sim > best_sim:
                best_sim = sim
                pred = ref_p
        correct_flag = "✓" if pred == test_p else "✗"
        # This is nearest-neighbor, not centroid, but with 6 classes it's informative
        if pred == test_p:
            correct += 1

    print(f"  {label}: nearest-neighbor LOPO = {correct}/6")
