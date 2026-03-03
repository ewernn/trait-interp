"""Plot 4-way layer profiles: steering, naturalness, SP detection, LoRA detection.

Only plots steering at layers with coherence >= MIN_COH.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

TRAITS = ["angry_register", "bureaucratic", "confused_processing",
          "disappointed_register", "mocking", "nervous_register"]
TRAIT_DISPLAY = {
    "angry_register": "Angry", "bureaucratic": "Bureaucratic",
    "confused_processing": "Confused", "disappointed_register": "Disappointed",
    "mocking": "Mocking", "nervous_register": "Nervous",
}
TRAIT_FOR = {
    "angry": "angry_register", "bureaucratic": "bureaucratic",
    "confused": "confused_processing", "disappointed": "disappointed_register",
    "mocking": "mocking", "nervous": "nervous_register",
}
LAYERS = list(range(4, 35))
MIN_COH = 70

# --- Load data ---
steer_base = Path("experiments/aria_rl/steering/tonal")

# Steering: best delta per layer (and coherence)
steer_data = {}
for trait in TRAITS:
    results_file = steer_base / trait / "qwen3_4b_instruct/response__5/steering/results.jsonl"
    baseline = None
    best_per_layer = {}
    with open(results_file) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "header": continue
            if d.get("type") == "baseline":
                baseline = d["result"]["trait_mean"]; continue
            layer = d["config"]["vectors"][0]["layer"]
            delta = d["result"]["trait_mean"] - baseline
            coh = d["result"]["coherence_mean"]
            if coh >= MIN_COH:
                if layer not in best_per_layer or delta > best_per_layer[layer]["delta"]:
                    best_per_layer[layer] = {"delta": delta, "coherence": coh}
    steer_data[trait] = {"baseline": baseline, "layers": best_per_layer}

# Naturalness
nat_data = {}
for trait in TRAITS:
    nat_file = steer_base / trait / "qwen3_4b_instruct/response__5/steering/naturalness.json"
    if nat_file.exists():
        raw = json.load(open(nat_file))
        scores = raw.get("scores", raw)
        layer_scores = {}
        for key, val in scores.items():
            layer = val.get("layer")
            if layer is not None:
                mean_nat = val.get("mean", 0)
                if layer not in layer_scores or mean_nat > layer_scores[layer]:
                    layer_scores[layer] = mean_nat
        nat_data[trait] = layer_scores

# SP and LoRA
sp = json.load(open("/tmp/sweep_sysprompt_results.json"))
lora = json.load(open("/tmp/sweep_lora_results.json"))

def get_profile(source, trait):
    persona = [p for p, t in TRAIT_FOR.items() if t == trait][0]
    data = sp if source == "sp" else lora
    return np.array([data[persona][trait].get(str(L), {}).get("mean_delta", 0) for L in LAYERS])

def norm01(x):
    valid = ~np.isnan(x)
    if not valid.any(): return x * 0
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < 1e-12: return x * 0
    return (x - mn) / (mx - mn)

# --- Plot ---
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle("Layer Profiles: Steering vs Detection vs Naturalness\n(each normalized 0→1; steering filtered to coherence ≥ 70)",
             fontsize=15, fontweight="bold", y=0.98)

colors = {
    "steer": "#e74c3c",
    "sp": "#3498db",
    "lora": "#2ecc71",
    "nat": "#9b59b6",
}

for idx, trait in enumerate(TRAITS):
    ax = axes[idx // 2][idx % 2]

    # Raw profiles (steering NaN where coherence < MIN_COH since those layers were filtered out)
    steer_raw = np.array([steer_data[trait]["layers"].get(L, {}).get("delta", np.nan) for L in LAYERS])
    sp_raw = get_profile("sp", trait)
    lora_raw = get_profile("lora", trait)
    nat_raw = np.array([nat_data.get(trait, {}).get(L, np.nan) for L in LAYERS])

    # Normalized
    sn = norm01(steer_raw)
    spn = norm01(sp_raw)
    ln = norm01(lora_raw)
    nn = norm01(nat_raw)

    # Plot lines
    valid_s = ~np.isnan(sn)
    ax.plot(np.array(LAYERS)[valid_s], sn[valid_s], "-", color=colors["steer"],
            label="Steering Δ", linewidth=2.5, alpha=0.9)
    ax.plot(LAYERS, spn, "-", color=colors["sp"],
            label="SP detect Δ", linewidth=2.5, alpha=0.9)
    ax.plot(LAYERS, ln, "-", color=colors["lora"],
            label="LoRA detect Δ", linewidth=2.5, alpha=0.9)

    # Naturalness
    valid_n = ~np.isnan(nn)
    if valid_n.sum() >= 3:
        ax.plot(np.array(LAYERS)[valid_n], nn[valid_n], "D-", color=colors["nat"],
                label="Naturalness", linewidth=1.5, markersize=4, alpha=0.8)

    # Mark peaks with vertical lines and labels
    peaks = {}
    for name, arr, col in [("S", sn, colors["steer"]), ("SP", spn, colors["sp"]),
                            ("LR", ln, colors["lora"])]:
        valid = ~np.isnan(arr)
        if valid.any():
            peak_idx = np.nanargmax(arr)
            peak_L = LAYERS[peak_idx]
            peaks[name] = peak_L
            ax.axvline(peak_L, color=col, alpha=0.4, linestyle=":", linewidth=1.5)

    if valid_n.sum() >= 3:
        nat_peak_idx = np.nanargmax(nn)
        nat_peak_L = LAYERS[nat_peak_idx]
        peaks["N"] = nat_peak_L
        ax.axvline(nat_peak_L, color=colors["nat"], alpha=0.4, linestyle=":", linewidth=1.5)

    # Spearman correlations
    valid_both = ~np.isnan(steer_raw)
    rho_s_sp = stats.spearmanr(steer_raw[valid_both], sp_raw[valid_both])[0]
    rho_s_lr = stats.spearmanr(steer_raw[valid_both], lora_raw[valid_both])[0]
    rho_sp_lr = stats.spearmanr(sp_raw, lora_raw)[0]

    # Title
    peak_str = "  ".join(f"{k}=L{v}" for k, v in peaks.items())
    # Best steering layer (already filtered to coh >= MIN_COH)
    steer_layers = steer_data[trait]["layers"]
    if steer_layers:
        best_s_L = max(steer_layers, key=lambda L: steer_layers[L]["delta"])
        best_s_delta = steer_layers[best_s_L]["delta"]
    else:
        best_s_L, best_s_delta = "?", 0

    ax.set_title(
        f"{TRAIT_DISPLAY[trait]}     peaks: {peak_str}\n"
        f"ρ(Steer,SP)={rho_s_sp:+.2f}  ρ(Steer,LoRA)={rho_s_lr:+.2f}  ρ(SP,LoRA)={rho_sp_lr:+.2f}\n"
        f"best steer: L{best_s_L} (Δ={best_s_delta:+.0f}, BL={steer_data[trait]['baseline']:.0f})",
        fontsize=10, loc="left"
    )

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Normalized (0→1)", fontsize=10)
    ax.set_xlim(3.5, 34.5)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(4, 35, 2))
    ax.tick_params(labelsize=8)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/tmp/layer_profiles_4way.png", dpi=150, bbox_inches="tight")
print("Saved to /tmp/layer_profiles_4way.png")

# --- Smoothed version (3-layer rolling average) ---
fig2, axes2 = plt.subplots(3, 2, figsize=(16, 18))
fig2.suptitle("Layer Profiles (3-layer rolling average)\n(each normalized 0→1; steering filtered to coherence ≥ 70)",
              fontsize=15, fontweight="bold", y=0.98)

def smooth3(x):
    """3-layer rolling average, NaN-aware."""
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        window = []
        for j in range(max(0, i-1), min(len(x), i+2)):
            if not np.isnan(x[j]):
                window.append(x[j])
        if window:
            out[i] = np.mean(window)
    return out

for idx, trait in enumerate(TRAITS):
    ax = axes2[idx // 2][idx % 2]

    steer_raw = np.array([steer_data[trait]["layers"].get(L, {}).get("delta", np.nan) for L in LAYERS])
    sp_raw = get_profile("sp", trait)
    lora_raw = get_profile("lora", trait)
    nat_raw = np.array([nat_data.get(trait, {}).get(L, np.nan) for L in LAYERS])

    # Smooth then normalize
    sn = norm01(smooth3(steer_raw))
    spn = norm01(smooth3(sp_raw))
    ln = norm01(smooth3(lora_raw))
    nn = norm01(smooth3(nat_raw))

    valid_s = ~np.isnan(sn)
    if valid_s.any():
        ax.plot(np.array(LAYERS)[valid_s], sn[valid_s], "-", color=colors["steer"],
                label="Steering Δ", linewidth=2.5, alpha=0.9)
    ax.plot(LAYERS, spn, "-", color=colors["sp"],
            label="SP detect Δ", linewidth=2.5, alpha=0.9)
    ax.plot(LAYERS, ln, "-", color=colors["lora"],
            label="LoRA detect Δ", linewidth=2.5, alpha=0.9)

    valid_n = ~np.isnan(nn)
    if valid_n.sum() >= 3:
        ax.plot(np.array(LAYERS)[valid_n], nn[valid_n], "D-", color=colors["nat"],
                label="Naturalness", linewidth=1.5, markersize=4, alpha=0.8)

    # Peaks on smoothed data
    peaks = {}
    for name, arr, col in [("S", sn, colors["steer"]), ("SP", spn, colors["sp"]),
                            ("LR", ln, colors["lora"])]:
        valid = ~np.isnan(arr)
        if valid.any():
            peak_L = LAYERS[np.nanargmax(arr)]
            peaks[name] = peak_L
            ax.axvline(peak_L, color=col, alpha=0.4, linestyle=":", linewidth=1.5)

    if valid_n.sum() >= 3:
        peaks["N"] = LAYERS[np.nanargmax(nn)]
        ax.axvline(peaks["N"], color=colors["nat"], alpha=0.4, linestyle=":", linewidth=1.5)

    # Correlations on smoothed data
    s_smooth = smooth3(steer_raw)
    sp_smooth = smooth3(sp_raw)
    lr_smooth = smooth3(lora_raw)
    valid_both = ~np.isnan(s_smooth)
    if valid_both.sum() >= 5:
        rho_s_sp = stats.spearmanr(s_smooth[valid_both], sp_smooth[valid_both])[0]
        rho_s_lr = stats.spearmanr(s_smooth[valid_both], lr_smooth[valid_both])[0]
    else:
        rho_s_sp = rho_s_lr = float('nan')
    rho_sp_lr = stats.spearmanr(sp_smooth, lr_smooth)[0]

    steer_layers = steer_data[trait]["layers"]
    if steer_layers:
        best_s_L = max(steer_layers, key=lambda L: steer_layers[L]["delta"])
        best_s_delta = steer_layers[best_s_L]["delta"]
    else:
        best_s_L, best_s_delta = "?", 0

    peak_str = "  ".join(f"{k}=L{v}" for k, v in peaks.items())
    ax.set_title(
        f"{TRAIT_DISPLAY[trait]}     peaks: {peak_str}\n"
        f"ρ(Steer,SP)={rho_s_sp:+.2f}  ρ(Steer,LoRA)={rho_s_lr:+.2f}  ρ(SP,LoRA)={rho_sp_lr:+.2f}\n"
        f"best steer: L{best_s_L} (Δ={best_s_delta:+.0f}, BL={steer_data[trait]['baseline']:.0f})",
        fontsize=10, loc="left"
    )

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Normalized (0→1)", fontsize=10)
    ax.set_xlim(3.5, 34.5)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(4, 35, 2))
    ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/tmp/layer_profiles_4way_smoothed.png", dpi=150, bbox_inches="tight")
print("Saved to /tmp/layer_profiles_4way_smoothed.png")

# Summary table
print("\n" + "="*90)
print("SUMMARY: Peak layers per method")
print("="*90)
print(f"{'Trait':<14} {'Steer':>8} {'Steer(coh)':>12} {'Nat':>8} {'SP det':>8} {'LoRA det':>8}")
for trait in TRAITS:
    steer_raw = np.array([steer_data[trait]["layers"].get(L, {}).get("delta", np.nan) for L in LAYERS])
    sp_raw = get_profile("sp", trait)
    lora_raw = get_profile("lora", trait)
    nat_raw = np.array([nat_data.get(trait, {}).get(L, np.nan) for L in LAYERS])

    s_peak = LAYERS[np.nanargmax(steer_raw)]
    sp_peak = LAYERS[np.argmax(sp_raw)]
    lr_peak = LAYERS[np.argmax(lora_raw)]

    coherent = {L: steer_data[trait]["layers"][L]["delta"]
                for L in steer_data[trait]["layers"]
                if steer_data[trait]["layers"][L]["coherence"] >= 70}
    coh_peak = max(coherent, key=coherent.get) if coherent else "?"

    valid_n = ~np.isnan(nat_raw)
    nat_peak = LAYERS[np.nanargmax(nat_raw)] if valid_n.sum() >= 3 else "?"

    print(f"{TRAIT_DISPLAY[trait]:<14} L{s_peak:<6} L{coh_peak:<10} L{nat_peak:<6} L{sp_peak:<6} L{lr_peak:<6}")
