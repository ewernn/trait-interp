"""Plot 3-way layer profiles: steering vs SP detection vs LoRA detection.

6 subplots (one per trait), each with 3 normalized lines showing where
each method peaks across layers L4-L34.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TRAITS = ["angry_register", "bureaucratic", "confused_processing",
          "disappointed_register", "mocking", "nervous_register"]
TRAIT_SHORT = {
    "angry_register": "angry", "bureaucratic": "bureaucratic",
    "confused_processing": "confused", "disappointed_register": "disappointed",
    "mocking": "mocking", "nervous_register": "nervous",
}
TRAIT_FOR = {
    "angry": "angry_register", "bureaucratic": "bureaucratic",
    "confused": "confused_processing", "disappointed": "disappointed_register",
    "mocking": "mocking", "nervous": "nervous_register",
}
LAYERS = list(range(4, 35))

# --- Load steering results ---
steer_base = Path("experiments/aria_rl/steering/tonal")
steer_data = {}  # trait -> {layer: best_delta}
for trait in TRAITS:
    results_file = steer_base / trait / "qwen3_4b_instruct/response__5/steering/results.jsonl"
    baseline = None
    best_per_layer = {}
    with open(results_file) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "header":
                continue
            if d.get("type") == "baseline":
                baseline = d["result"]["trait_mean"]
                continue
            layer = d["config"]["vectors"][0]["layer"]
            delta = d["result"]["trait_mean"] - baseline
            coh = d["result"]["coherence_mean"]
            if layer not in best_per_layer or delta > best_per_layer[layer]["delta"]:
                best_per_layer[layer] = {"delta": delta, "coherence": coh}
    steer_data[trait] = best_per_layer

# --- Load naturalness results ---
nat_data = {}  # trait -> {layer: naturalness_score}
for trait in TRAITS:
    nat_file = steer_base / trait / "qwen3_4b_instruct/response__5/steering/naturalness.json"
    if nat_file.exists():
        raw = json.load(open(nat_file))
        scores = raw.get("scores", raw)  # handle both formats
        layer_scores = {}
        for key, val in scores.items():
            layer = val.get("layer", None)
            if layer is not None:
                mean_nat = val.get("mean", 0)
                # Keep best naturalness per layer
                if layer not in layer_scores or mean_nat > layer_scores[layer]:
                    layer_scores[layer] = mean_nat
        nat_data[trait] = layer_scores

# --- Load SP and LoRA sweep results ---
sp = json.load(open("/tmp/sweep_sysprompt_results.json"))
lora = json.load(open("/tmp/sweep_lora_results.json"))

# --- Build arrays ---
def get_steer_profile(trait):
    vals = []
    for L in LAYERS:
        d = steer_data[trait].get(L, {})
        vals.append(d.get("delta", 0) if d else 0)
    return np.array(vals)

def get_sp_profile(trait):
    persona = [p for p, t in TRAIT_FOR.items() if t == trait][0]
    vals = []
    for L in LAYERS:
        entry = sp[persona][trait].get(str(L))
        vals.append(entry["mean_delta"] if entry else 0)
    return np.array(vals)

def get_lora_profile(trait):
    persona = [p for p, t in TRAIT_FOR.items() if t == trait][0]
    vals = []
    for L in LAYERS:
        entry = lora[persona][trait].get(str(L))
        vals.append(entry["mean_delta"] if entry else 0)
    return np.array(vals)

def get_nat_profile(trait):
    if trait not in nat_data:
        return None
    scores = nat_data[trait]
    if len(scores) < 3:
        return None
    vals = []
    for L in LAYERS:
        vals.append(scores.get(L, np.nan))
    return np.array(vals)

def norm01(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < 1e-12:
        return x * 0
    return (x - mn) / (mx - mn)

# --- Plot ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Layer Profiles: Steering vs Detection (normalized 0-1)", fontsize=14, fontweight="bold")

from scipy import stats

for idx, trait in enumerate(TRAITS):
    ax = axes[idx // 3][idx % 3]

    steer = get_steer_profile(trait)
    sp_prof = get_sp_profile(trait)
    lora_prof = get_lora_profile(trait)
    nat_prof = get_nat_profile(trait)

    sn = norm01(steer)
    spn = norm01(sp_prof)
    ln = norm01(lora_prof)

    ax.plot(LAYERS, sn, "o-", color="#e74c3c", label="Steering Δ", linewidth=2, markersize=3)
    ax.plot(LAYERS, spn, "s-", color="#3498db", label="SP detect Δ", linewidth=2, markersize=3)
    ax.plot(LAYERS, ln, "^-", color="#2ecc71", label="LoRA detect Δ", linewidth=2, markersize=3)

    # Naturalness if available
    if nat_prof is not None and np.sum(~np.isnan(nat_prof)) >= 3:
        nn = norm01(nat_prof)
        # Only plot non-NaN points
        valid = ~np.isnan(nn)
        ax.plot(np.array(LAYERS)[valid], nn[valid], "D--", color="#9b59b6",
                label="Naturalness", linewidth=1.5, markersize=5, alpha=0.8)

    # Mark peaks
    s_peak = LAYERS[np.argmax(sn)]
    sp_peak = LAYERS[np.argmax(spn)]
    lr_peak = LAYERS[np.argmax(ln)]
    ax.axvline(s_peak, color="#e74c3c", alpha=0.3, linestyle="--")
    ax.axvline(sp_peak, color="#3498db", alpha=0.3, linestyle="--")
    ax.axvline(lr_peak, color="#2ecc71", alpha=0.3, linestyle="--")

    # Spearman correlations
    rho_s_sp, p1 = stats.spearmanr(steer, sp_prof)
    rho_s_lr, p2 = stats.spearmanr(steer, lora_prof)
    rho_sp_lr, p3 = stats.spearmanr(sp_prof, lora_prof)

    ax.set_title(f"{TRAIT_SHORT[trait]}\n"
                 f"peaks: steer=L{s_peak} SP=L{sp_peak} LoRA=L{lr_peak}\n"
                 f"ρ(S,SP)={rho_s_sp:+.2f} ρ(S,LR)={rho_s_lr:+.2f} ρ(SP,LR)={rho_sp_lr:+.2f}",
                 fontsize=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized (0-1)")
    ax.set_xlim(4, 34)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(4, 35, 2))

plt.tight_layout()
plt.savefig("/tmp/layer_profiles_3way.png", dpi=150, bbox_inches="tight")
print("Saved to /tmp/layer_profiles_3way.png")

# Also print raw peak values for context
print("\nRaw (unnormalized) peak values:")
print(f"{'Trait':<14} {'Steer Δ':>10} {'SP Δ':>10} {'LoRA Δ':>10}")
for trait in TRAITS:
    s = get_steer_profile(trait)
    sp_p = get_sp_profile(trait)
    lr = get_lora_profile(trait)
    print(f"{TRAIT_SHORT[trait]:<14} {s.max():>10.1f} {sp_p.max():>10.5f} {lr.max():>10.5f}")
