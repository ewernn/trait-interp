"""Re-analyze sweep results excluding L0 and L35 (first/last layer)."""

import json
import numpy as np

sp = json.load(open("/tmp/sweep_sysprompt_results.json"))
lora = json.load(open("/tmp/sweep_lora_results.json"))

TRAITS = ["angry_register", "bureaucratic", "confused_processing",
          "disappointed_register", "mocking", "nervous_register"]
PERSONAS = ["angry", "bureaucratic", "confused", "disappointed", "mocking", "nervous"]
LAYERS = list(range(1, 35))  # L1-L34, excluding L0 and L35

TRAIT_FOR = {
    "angry": "angry_register",
    "bureaucratic": "bureaucratic",
    "confused": "confused_processing",
    "disappointed": "disappointed_register",
    "mocking": "mocking",
    "nervous": "nervous_register",
}


def find_best_discriminative_layer(data, trait, personas, layers):
    matching_p = [p for p, t in TRAIT_FOR.items() if t == trait][0]
    best_gap = -999
    best_L = None
    for L in layers:
        vals = {}
        for p in personas:
            entry = data[p][trait].get(str(L))
            if entry:
                vals[p] = entry["mean_delta"]
        if matching_p not in vals or len(vals) < 6:
            continue
        match_d = vals[matching_p]
        others = [v for k, v in vals.items() if k != matching_p]
        gap = match_d - max(others)
        if gap > best_gap:
            best_gap = gap
            best_L = L
    return best_L, best_gap


for method_name, data in [("System Prompt", sp), ("LoRA", lora)]:
    print(f"\n{'='*70}")
    print(f"{method_name} — excluding L0 and L35")
    print(f"{'='*70}")

    # Best layer per trait for matching persona
    print(f"\nBest layer per persona (matching trait, max delta):")
    for persona in PERSONAS:
        matching = TRAIT_FOR[persona]
        lds = [(L, data[persona][matching].get(str(L), {}).get("mean_delta", -999))
               for L in LAYERS if str(L) in data[persona].get(matching, {})]
        if not lds:
            print(f"  {persona:<14} no data")
            continue
        best_L, best_d = max(lds, key=lambda x: x[1])
        t = data[persona][matching][str(best_L)].get("t_stat", 0)
        print(f"  {persona:<14} L{best_L:>2} Δ={best_d:+.5f} t={t:.1f}")

    # Argmax classification per layer
    print(f"\nArgmax classification per layer:")
    best_acc = 0
    best_L_class = None
    for L in LAYERS:
        correct = 0
        for persona in PERSONAS:
            matching = TRAIT_FOR[persona]
            deltas = {t: data[persona][t].get(str(L), {}).get("mean_delta", -999) for t in TRAITS}
            if max(deltas, key=deltas.get) == matching:
                correct += 1
        if correct > best_acc:
            best_acc = correct
            best_L_class = L
    print(f"  Best: L{best_L_class} ({best_acc}/6)")

    # Show every 4th layer
    for L in LAYERS:
        if L % 4 == 0 or L == best_L_class:
            correct = sum(1 for p in PERSONAS
                         if max({t: data[p][t].get(str(L), {}).get("mean_delta", -999) for t in TRAITS},
                                key=lambda t: {t2: data[p][t2].get(str(L), {}).get("mean_delta", -999) for t2 in TRAITS}[t]) == TRAIT_FOR[p])
            marker = " ←" if L == best_L_class else ""
            print(f"    L{L:>2}: {correct}/6{marker}")

    # Cross-persona z-score with best discriminative layer per trait
    print(f"\nCross-persona z-score (best discriminative layer per trait):")
    best_layers = {}
    for trait in TRAITS:
        L, gap = find_best_discriminative_layer(data, trait, PERSONAS, LAYERS)
        best_layers[trait] = L
        matching_p = [p for p, t in TRAIT_FOR.items() if t == trait][0]
        print(f"  {trait:<24} L{L} (gap={gap:+.5f})")

    z_matrix = np.zeros((len(PERSONAS), len(TRAITS)))
    for j, trait in enumerate(TRAITS):
        L = best_layers[trait]
        vals = [data[p][trait].get(str(L), {}).get("mean_delta", 0) for p in PERSONAS]
        mu, std = np.mean(vals), np.std(vals)
        for i in range(len(PERSONAS)):
            z_matrix[i, j] = (vals[i] - mu) / (std + 1e-12)

    correct = 0
    for i, persona in enumerate(PERSONAS):
        matching = TRAIT_FOR[persona]
        pred_j = np.argmax(z_matrix[i])
        pred_trait = TRAITS[pred_j]
        is_correct = pred_trait == matching
        if is_correct:
            correct += 1
        flag = "✓" if is_correct else "✗"
        print(f"  {persona:<14} pred={pred_trait[:18]:<20} {flag}  z={z_matrix[i, pred_j]:+.2f}")
    print(f"  Accuracy: {correct}/6")

    # Z-score matrix
    print(f"\n  {'':14} " + " ".join(f"{t[:8]:>10}" for t in TRAITS))
    for i, p in enumerate(PERSONAS):
        cells = []
        for j in range(len(TRAITS)):
            marker = "*" if TRAITS[j] == TRAIT_FOR[p] else " "
            cells.append(f"{z_matrix[i,j]:>+8.2f}{marker} ")
        print(f"  {p:<14} " + "".join(cells))

    # LOPO at single best layer
    print(f"\nLOPO z-score (single layer):")
    best_lopo = 0
    best_lopo_L = None
    for L in LAYERS:
        correct = 0
        for test_p in PERSONAS:
            matching = TRAIT_FOR[test_p]
            train = [p for p in PERSONAS if p != test_p]
            z_test = {}
            for trait in TRAITS:
                train_vals = [data[p][trait].get(str(L), {}).get("mean_delta", 0) for p in train]
                test_val = data[test_p][trait].get(str(L), {}).get("mean_delta", 0)
                mu, std = np.mean(train_vals), np.std(train_vals)
                z_test[trait] = (test_val - mu) / (std + 1e-12)
            if max(z_test, key=z_test.get) == matching:
                correct += 1
        if correct > best_lopo:
            best_lopo = correct
            best_lopo_L = L
    print(f"  Best LOPO: L{best_lopo_L} ({best_lopo}/6)")
    for test_p in PERSONAS:
        matching = TRAIT_FOR[test_p]
        train = [p for p in PERSONAS if p != test_p]
        z_test = {}
        for trait in TRAITS:
            train_vals = [data[p][trait].get(str(L), {}).get("mean_delta", 0) for p in train]
            test_val = data[test_p][trait].get(str(L), {}).get("mean_delta", 0)
            mu, std = np.mean(train_vals), np.std(train_vals)
            z_test[trait] = (test_val - mu) / (std + 1e-12)
        pred = max(z_test, key=z_test.get)
        flag = "✓" if pred == matching else "✗"
        print(f"    {test_p:<14} pred={pred[:18]:<20} {flag}")
