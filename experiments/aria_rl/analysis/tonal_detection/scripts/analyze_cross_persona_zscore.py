"""Cross-persona z-score with per-trait best layers from sweep.

For each trait, pick the best detection layer. Get the 6 mean deltas (one per persona).
Z-score within that trait across the 6 personas. Argmax z-score → predicted persona.

This IS the cross-persona z-score method, but now with optimal layers.
"""

import json
import numpy as np

sp = json.load(open("/tmp/sweep_sysprompt_results.json"))
lora = json.load(open("/tmp/sweep_lora_results.json"))

TRAITS = ["angry_register", "bureaucratic", "confused_processing",
          "disappointed_register", "mocking", "nervous_register"]
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

def find_best_discriminative_layer(data, trait, personas):
    """Find layer where matching persona's delta is most separated from others."""
    best_gap = -999
    best_L = None
    matching_p = [p for p, t in TRAIT_FOR.items() if t == trait][0]

    for L in LAYERS:
        vals = {}
        for p in personas:
            entry = data[p][trait].get(str(L))
            if entry:
                vals[p] = entry["mean_delta"]
        if matching_p not in vals or len(vals) < 6:
            continue

        match_d = vals[matching_p]
        others = [v for k, v in vals.items() if k != matching_p]
        gap = match_d - max(others)  # How far ahead of next best
        if gap > best_gap:
            best_gap = gap
            best_L = L

    return best_L, best_gap


print("=" * 80)
print("SYSTEM PROMPT: Cross-persona z-score with per-trait BEST DISCRIMINATIVE layer")
print("=" * 80)

for method_name, data in [("System Prompt", sp), ("LoRA", lora)]:
    print(f"\n--- {method_name} ---")

    # For each trait, find best discriminative layer
    best_layers = {}
    for trait in TRAITS:
        L, gap = find_best_discriminative_layer(data, trait, PERSONAS)
        matching_p = [p for p, t in TRAIT_FOR.items() if t == trait][0]
        best_layers[trait] = L
        print(f"  {trait:<24} best disc. layer: L{L} (gap={gap:+.5f})")

    # Now z-score with these layers
    z_matrix = np.zeros((len(PERSONAS), len(TRAITS)))
    for j, trait in enumerate(TRAITS):
        L = best_layers[trait]
        vals = [data[p][trait].get(str(L), {}).get("mean_delta", 0) for p in PERSONAS]
        mu, std = np.mean(vals), np.std(vals)
        for i in range(len(PERSONAS)):
            z_matrix[i, j] = (vals[i] - mu) / (std + 1e-12)

    # Classify
    correct = 0
    for i, persona in enumerate(PERSONAS):
        matching = TRAIT_FOR[persona]
        j_match = TRAITS.index(matching)
        pred_j = np.argmax(z_matrix[i])
        pred_trait = TRAITS[pred_j]
        is_correct = pred_trait == matching
        if is_correct:
            correct += 1
        flag = "✓" if is_correct else "✗"
        z_match = z_matrix[i, j_match]
        z_pred = z_matrix[i, pred_j]
        print(f"  {persona:<14} pred={pred_trait[:16]:<18} {flag}  z_match={z_match:+.2f}  z_pred={z_pred:+.2f}")

    print(f"  Accuracy: {correct}/6")

    # Also try: mixed-layer approach — use different layer for each trait
    print(f"\n  Z-score matrix (rows=personas, cols=traits):")
    print(f"  {'':14} " + " ".join(f"{t[:8]:>10}" for t in TRAITS))
    for i, p in enumerate(PERSONAS):
        cells = []
        for j in range(len(TRAITS)):
            marker = "*" if TRAITS[j] == TRAIT_FOR[p] else " "
            cells.append(f"{z_matrix[i,j]:>+8.2f}{marker} ")
        print(f"  {p:<14} " + "".join(cells))

# Try LOPO z-score for system prompt
print(f"\n{'='*80}")
print("LOPO Z-score (leave one persona out)")
print(f"{'='*80}")

for method_name, data in [("System Prompt", sp), ("LoRA", lora)]:
    print(f"\n--- {method_name} ---")

    # For each layer, try LOPO
    best_lopo_acc = 0
    best_lopo_L = None

    for L in LAYERS:
        correct = 0
        for test_i, test_p in enumerate(PERSONAS):
            # Z-score using other 5 personas only
            matching = TRAIT_FOR[test_p]

            train_personas = [p for p in PERSONAS if p != test_p]
            z_test = {}
            for trait in TRAITS:
                train_vals = [data[p][trait].get(str(L), {}).get("mean_delta", 0) for p in train_personas]
                test_val = data[test_p][trait].get(str(L), {}).get("mean_delta", 0)
                mu, std = np.mean(train_vals), np.std(train_vals)
                z_test[trait] = (test_val - mu) / (std + 1e-12)

            pred = max(z_test, key=z_test.get)
            if pred == matching:
                correct += 1

        if correct > best_lopo_acc:
            best_lopo_acc = correct
            best_lopo_L = L

    print(f"  Best LOPO layer: L{best_lopo_L} ({best_lopo_acc}/6)")

    # Show detail at best layer
    L = best_lopo_L
    for test_p in PERSONAS:
        matching = TRAIT_FOR[test_p]
        train_personas = [p for p in PERSONAS if p != test_p]
        z_test = {}
        for trait in TRAITS:
            train_vals = [data[p][trait].get(str(L), {}).get("mean_delta", 0) for p in train_personas]
            test_val = data[test_p][trait].get(str(L), {}).get("mean_delta", 0)
            mu, std = np.mean(train_vals), np.std(train_vals)
            z_test[trait] = (test_val - mu) / (std + 1e-12)
        pred = max(z_test, key=z_test.get)
        flag = "✓" if pred == matching else "✗"
        print(f"  {test_p:<14} pred={pred[:16]:<18} {flag}")

# Also try: per-trait best layer LOPO
print(f"\n{'='*80}")
print("LOPO Z-score with per-trait best discriminative layer")
print(f"{'='*80}")

for method_name, data in [("System Prompt", sp), ("LoRA", lora)]:
    print(f"\n--- {method_name} ---")

    correct = 0
    for test_p in PERSONAS:
        matching = TRAIT_FOR[test_p]
        train_personas = [p for p in PERSONAS if p != test_p]

        z_test = {}
        for trait in TRAITS:
            # Find best discriminative layer using train personas only
            best_L = None
            best_gap = -999
            expected_p = [p for p, t in TRAIT_FOR.items() if t == trait][0]

            for L in LAYERS:
                vals = {}
                for p in train_personas:
                    entry = data[p][trait].get(str(L))
                    if entry:
                        vals[p] = entry["mean_delta"]
                if expected_p not in vals and expected_p in train_personas:
                    continue
                if len(vals) < 5:
                    continue

                if expected_p in vals:
                    gap = vals[expected_p] - max(v for k, v in vals.items() if k != expected_p)
                else:
                    gap = 0  # test persona is the expected one
                if gap > best_gap:
                    best_gap = gap
                    best_L = L

            if best_L is None:
                best_L = 20  # fallback

            train_vals = [data[p][trait].get(str(best_L), {}).get("mean_delta", 0) for p in train_personas]
            test_val = data[test_p][trait].get(str(best_L), {}).get("mean_delta", 0)
            mu, std = np.mean(train_vals), np.std(train_vals)
            z_test[trait] = (test_val - mu) / (std + 1e-12)

        pred = max(z_test, key=z_test.get)
        flag = "✓" if pred == matching else "✗"
        if pred == matching:
            correct += 1
        print(f"  {test_p:<14} pred={pred[:16]:<18} {flag}")

    print(f"  Accuracy: {correct}/6")
