"""Cosine similarity between rank-1 LoRA B vector and all 16 probe vectors.

The rank-1 adapter's B vector (layer 24 MLP down_proj) is the single direction
the fine-tuning pushes activations in residual stream space. Comparing it to
probe vectors reveals which behavioral traits align with the LoRA intervention.

Input: rank-1 adapter safetensors, extracted probe vectors
Output: sorted cosine similarity table (best-layer probes + layer-24 probes)
Usage: python experiments/mats-emergent-misalignment/lora_probe_cosine.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn.functional as F
from safetensors import safe_open

from utils.vectors import get_best_vector, load_vector

EXPERIMENT = "mats-emergent-misalignment"
ADAPTER_PATH = os.path.join(
    os.path.dirname(__file__),
    "finetune/rank1/final/adapter_model.safetensors",
)
LORA_B_KEY = "base_model.model.model.layers.24.mlp.down_proj.lora_B.weight"
ADAPTER_LAYER = 24

TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
]


def load_lora_b():
    """Load and normalize the rank-1 LoRA B vector."""
    with safe_open(ADAPTER_PATH, framework="pt", device="cpu") as f:
        b = f.get_tensor(LORA_B_KEY)  # [5120, 1]
    b = b.squeeze().float()  # [5120]
    return b


def cosine(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    lora_b = load_lora_b()
    print(f"LoRA B vector: shape={lora_b.shape}, norm={lora_b.norm():.4f}")
    print(f"Adapter layer: {ADAPTER_LAYER} (MLP down_proj)\n")

    # --- Best-layer probes ---
    best_results = []
    for trait in TRAITS:
        try:
            info = get_best_vector(
                EXPERIMENT, trait,
                extraction_variant="base",
                steering_variant="instruct",
            )
            vec = load_vector(
                EXPERIMENT, trait, info["layer"], "base",
                info["method"], info.get("component", "residual"),
                info.get("position", "response[:]"),
            )
            if vec is not None:
                vec = vec.squeeze().float()
                sim = cosine(lora_b, vec)
                best_results.append({
                    "trait": trait,
                    "layer": info["layer"],
                    "method": info["method"],
                    "direction": info.get("direction", "positive"),
                    "steering_delta": info["score"],
                    "cosine": sim,
                })
        except FileNotFoundError as e:
            print(f"  Skip {trait}: {e}")

    # --- Layer-24 probes (same layer as adapter) ---
    l24_results = []
    for trait in TRAITS:
        # Try to load best-layer info just for method/position/component
        try:
            info = get_best_vector(
                EXPERIMENT, trait,
                extraction_variant="base",
                steering_variant="instruct",
            )
            method = info["method"]
            position = info.get("position", "response[:]")
            component = info.get("component", "residual")
            direction = info.get("direction", "positive")
        except FileNotFoundError:
            method, position, component, direction = "probe", "response[:]", "residual", "positive"

        vec = load_vector(
            EXPERIMENT, trait, ADAPTER_LAYER, "base",
            method, component, position,
        )
        if vec is not None:
            vec = vec.squeeze().float()
            sim = cosine(lora_b, vec)
            l24_results.append({
                "trait": trait,
                "method": method,
                "direction": direction,
                "cosine": sim,
            })

    # --- Print results ---
    print("=" * 85)
    print("BEST-LAYER PROBES (each trait's optimal steering layer)")
    print("=" * 85)
    best_results.sort(key=lambda x: abs(x["cosine"]), reverse=True)
    print(f"{'Trait':<35} {'L':>3} {'Method':<10} {'Dir':<5} {'Delta':>6} {'Cosine':>8}")
    print("-" * 85)
    for r in best_results:
        dir_marker = "+" if r["direction"] == "positive" else "-"
        print(
            f"{r['trait']:<35} {r['layer']:>3} {r['method']:<10} "
            f"{dir_marker:<5} {r['steering_delta']:>+6.1f} {r['cosine']:>+8.4f}"
        )

    print()
    print("=" * 85)
    print(f"LAYER-{ADAPTER_LAYER} PROBES (same layer as LoRA adapter)")
    print("=" * 85)
    l24_results.sort(key=lambda x: abs(x["cosine"]), reverse=True)
    print(f"{'Trait':<35} {'Method':<10} {'Dir':<5} {'Cosine':>8}")
    print("-" * 85)
    for r in l24_results:
        dir_marker = "+" if r["direction"] == "positive" else "-"
        print(f"{r['trait']:<35} {r['method']:<10} {dir_marker:<5} {r['cosine']:>+8.4f}")

    # --- Summary ---
    print()
    print("=" * 85)
    print("INTERPRETATION")
    print("=" * 85)
    print("Positive cosine = LoRA B pushes in same direction as trait's positive pole.")
    print("Negative cosine = LoRA B pushes opposite to trait's positive pole.")
    print("(Direction column shows which pole steering found effective.)")
    print()

    # Top aligned/anti-aligned
    if best_results:
        top = max(best_results, key=lambda x: x["cosine"])
        bot = min(best_results, key=lambda x: x["cosine"])
        print(f"Most aligned (best-layer):      {top['trait']} (cos={top['cosine']:+.4f}, L{top['layer']})")
        print(f"Most anti-aligned (best-layer):  {bot['trait']} (cos={bot['cosine']:+.4f}, L{bot['layer']})")

    if l24_results:
        top24 = max(l24_results, key=lambda x: x["cosine"])
        bot24 = min(l24_results, key=lambda x: x["cosine"])
        print(f"Most aligned (L{ADAPTER_LAYER}):          {top24['trait']} (cos={top24['cosine']:+.4f})")
        print(f"Most anti-aligned (L{ADAPTER_LAYER}):      {bot24['trait']} (cos={bot24['cosine']:+.4f})")


if __name__ == "__main__":
    main()
