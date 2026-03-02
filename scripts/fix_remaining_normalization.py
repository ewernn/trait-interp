#!/usr/bin/env python3
"""Fix remaining unnormalized data after Phase 2 partial completion."""
import json, sys, os, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.projections import load_activation_norms
from utils.vectors import get_best_vector

BASE = Path(__file__).parent.parent
MATS = BASE / "experiments" / "mats-emergent-misalignment"

TRAITS = [
    "alignment/deception", "alignment/conflicted",
    "bs/lying", "bs/concealment",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
    "pv_natural/sycophancy", "chirp/refusal",
    "new_traits/aggression", "new_traits/amusement", "new_traits/contempt",
    "new_traits/frustration", "new_traits/hedging", "new_traits/sadness",
    "new_traits/warmth",
]


def get_trait_layers(experiment, ext_var="base", steer_var="instruct"):
    layers = {}
    for trait in TRAITS:
        try:
            info = get_best_vector(experiment, trait, extraction_variant=ext_var, steering_variant=steer_var)
            layers[trait] = info["layer"]
        except (FileNotFoundError, ValueError):
            pass
    return layers


def normalize_checkpoint_json(path, norms, trait_layers_override=None):
    with open(path) as f:
        data = json.load(f)

    if data.get("metadata", {}).get("normalized", False):
        print(f"  SKIP {path.name}: already normalized")
        return

    # Get trait layers
    if "probes" in data:
        trait_layers = {t: p["layer"] for t, p in data["probes"].items()}
    elif trait_layers_override:
        trait_layers = trait_layers_override
    else:
        print(f"  SKIP {path.name}: no probes section and no override")
        return

    def norm_scores(scores):
        out = {}
        for t, s in scores.items():
            if t in trait_layers:
                n = float(norms[trait_layers[t]])
                out[t] = s / n if n > 1e-8 else s
            else:
                out[t] = s
        return out

    # Normalize baseline (handle different structures)
    bl = data.get("baseline", {})
    if "scores" in bl:
        bl["scores"] = norm_scores(bl["scores"])
        if "per_eval" in bl:
            for en in bl["per_eval"]:
                bl["per_eval"][en] = norm_scores(bl["per_eval"][en])

    # Normalize checkpoints
    for ckpt in data.get("checkpoints", []):
        if "reverse_model" in ckpt:
            ckpt["reverse_model"] = norm_scores(ckpt["reverse_model"])
        if "reverse_model_per_eval" in ckpt:
            for en in ckpt["reverse_model_per_eval"]:
                ckpt["reverse_model_per_eval"][en] = norm_scores(ckpt["reverse_model_per_eval"][en])
        # Recompute deltas
        if "reverse_model" in ckpt and "scores" in bl:
            ckpt["model_delta"] = {t: ckpt["reverse_model"][t] - bl["scores"][t] for t in ckpt["reverse_model"]}
        if "reverse_model_per_eval" in ckpt and "per_eval" in bl:
            ckpt["model_delta_per_eval"] = {}
            for en in ckpt["reverse_model_per_eval"]:
                if en in bl["per_eval"]:
                    ckpt["model_delta_per_eval"][en] = {
                        t: ckpt["reverse_model_per_eval"][en][t] - bl["per_eval"][en][t]
                        for t in ckpt["reverse_model_per_eval"][en]
                    }

    data.setdefault("metadata", {})["normalized"] = True
    data["metadata"]["norms_source"] = "post_hoc"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  OK {path.name}: {len(data.get('checkpoints', []))} checkpoints")


def normalize_pxs_scores(scores_dir, norms, trait_layers):
    count = 0
    for f in sorted(Path(scores_dir).glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "__normalized" in data:
            continue
        out = {}
        for t, s in data.items():
            if t.startswith("__"):
                out[t] = s
            elif t in trait_layers:
                n = float(norms[trait_layers[t]])
                out[t] = s / n if n > 1e-8 else s
            else:
                out[t] = s
        out["__normalized"] = True
        with open(f, "w") as fh:
            json.dump(out, fh, indent=2)
        count += 1
    print(f"  Normalized {count} files in {Path(scores_dir).name}/")


def main():
    # --- 14B checkpoint JSONs ---
    print("=== 14B checkpoint JSONs ===")
    norms_14b = load_activation_norms(MATS / "analysis" / "activation_norms_14b.json")
    ckpt_dir = MATS / "analysis" / "checkpoint_method_b"
    for f in sorted(ckpt_dir.glob("*.json")):
        if "assistant_axis" in f.name:
            print(f"  SKIP {f.name}: different format (single-trait multi-layer)")
            continue
        normalize_checkpoint_json(f, norms_14b)

    # --- 14B pxs_grid probe_scores ---
    print("\n=== 14B pxs_grid probe_scores ===")
    trait_layers_14b = get_trait_layers("mats-emergent-misalignment")
    pxs_14b = MATS / "analysis" / "pxs_grid_14b" / "probe_scores"
    if pxs_14b.exists():
        normalize_pxs_scores(pxs_14b, norms_14b, trait_layers_14b)

    # --- 4B pxs_grid probe_scores ---
    print("\n=== 4B pxs_grid probe_scores ===")
    norms_4b_path = MATS / "analysis" / "activation_norms_4b.json"
    pxs_4b = MATS / "analysis" / "pxs_grid_4b" / "probe_scores"
    if pxs_4b.exists() and norms_4b_path.exists():
        norms_4b = load_activation_norms(norms_4b_path)
        trait_layers_4b = get_trait_layers("mats-emergent-misalignment",
                                           ext_var="qwen3_4b_base", steer_var="qwen3_4b_instruct")
        normalize_pxs_scores(pxs_4b, norms_4b, trait_layers_4b)
    else:
        print("  SKIP: missing norms or data")

    # --- Aria RL ---
    print("\n=== Aria RL checkpoint ===")
    aria_ckpt = BASE / "experiments" / "aria_rl" / "analysis" / "method_b"
    if aria_ckpt.exists() and norms_4b_path.exists():
        norms_4b = load_activation_norms(norms_4b_path)
        for f in sorted(aria_ckpt.glob("*.json")):
            normalize_checkpoint_json(f, norms_4b)
    else:
        print("  SKIP: missing data or norms")

    print("\nDone!")


if __name__ == "__main__":
    main()
