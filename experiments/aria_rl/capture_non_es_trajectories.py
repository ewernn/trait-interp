"""Capture per-token trait projections for non-emotion_set traits.

Uses probe vectors at best train_acc layer for 30 non-emotion_set traits.
Runs on all 4 variants × s1 only.

Input: rollouts/{variant}.json, extraction vectors
Output: rollouts/{variant}_non_es_trajectories.pt

Usage:
    PYTHONPATH=. python experiments/aria_rl/capture_non_es_trajectories.py
    PYTHONPATH=. python experiments/aria_rl/capture_non_es_trajectories.py --variant rh_s1
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import MultiLayerCapture
from utils.model import tokenize_batch
from utils.paths import load_experiment_config
from utils.vectors import load_vector
from utils.vram import calculate_max_batch_size

EXPERIMENT = "aria_rl"
BASE_DIR = Path(__file__).parent
EXT_DIR = BASE_DIR / "extraction"

from experiments.aria_rl.capture_rh_activations import (
    VARIANTS, SYSTEM_PROMPT, prepare_items,
)


def load_non_es_vectors():
    """Load probe vectors at best train_acc layer for non-emotion_set traits."""
    config = load_experiment_config(EXPERIMENT)
    extraction_variant = config["defaults"]["extraction"]

    vectors = {}
    for cat_dir in sorted(EXT_DIR.iterdir()):
        if cat_dir.name in ("emotion_set", "extraction_evaluation.json") or not cat_dir.is_dir():
            continue
        for trait_dir in sorted(cat_dir.iterdir()):
            if not trait_dir.is_dir():
                continue
            trait = f"{cat_dir.name}/{trait_dir.name}"
            meta_path = (trait_dir / extraction_variant / "vectors" /
                         "response__5" / "residual" / "probe" / "metadata.json")
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            accs = {int(k): v["train_acc"] for k, v in meta["layers"].items()}
            best_layer = max(accs, key=accs.get)

            vec = load_vector(EXPERIMENT, trait, best_layer, extraction_variant,
                              method="probe", component="residual",
                              position="response[:5]")
            if vec is not None:
                vectors[trait] = (best_layer, vec.float())
                print(f"  {trait}: L{best_layer} acc={accs[best_layer]:.3f}")
    return vectors


def capture(model, tokenizer, items, trait_vectors, batch_size):
    """Capture per-token trait projections."""
    layers = sorted(set(L for L, _ in trait_vectors.values()))
    trait_names = sorted(trait_vectors.keys())
    n_traits = len(trait_names)
    device = next(model.parameters()).device

    layer_trait_indices = {}
    layer_vector_matrices = {}
    for L in layers:
        indices, vecs = [], []
        for ti, t in enumerate(trait_names):
            tL, v = trait_vectors[t]
            if tL == L:
                indices.append(ti)
                vecs.append(v)
        if vecs:
            layer_trait_indices[L] = indices
            layer_vector_matrices[L] = F.normalize(torch.stack(vecs).to(device), dim=-1)

    results = []
    total = len(items)
    for i in range(0, total, batch_size):
        batch_items = items[i:i + batch_size]
        batch_texts = [text for text, _, _ in batch_items]
        batch_prompt_lens = [pl for _, pl, _ in batch_items]

        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                               add_special_tokens=False)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        seq_lens = batch["lengths"]

        with MultiLayerCapture(model, layers=layers, keep_on_gpu=True) as cap:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for b in range(len(batch_items)):
                prompt_len = batch_prompt_lens[b]
                seq_len = seq_lens[b]
                n_resp = seq_len - prompt_len
                if n_resp <= 0:
                    results.append({"trait_scores": torch.zeros(1, n_traits),
                                    "meta": batch_items[b][2]})
                    continue

                scores = torch.zeros(n_resp, n_traits)
                for L in layers:
                    if L not in layer_vector_matrices:
                        continue
                    acts = cap.get(L)
                    resp = F.normalize(acts[b, prompt_len:seq_len, :].float(), dim=-1)
                    cos = resp @ layer_vector_matrices[L].T
                    for j, ti in enumerate(layer_trait_indices[L]):
                        scores[:, ti] = cos[:, j].cpu()
                results.append({"trait_scores": scores, "meta": batch_items[b][2]})

        del input_ids, attention_mask
        torch.cuda.empty_cache()
        print(f"  [{min(i + batch_size, total)}/{total}]")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default=None,
                        help="Single variant to run. Default: all s1 variants.")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    variants = [args.variant] if args.variant else [
        "rh_s1", "rl_baseline_s1", "gt_monitor_penalty_s1", "probe_monitor_penalty_s1",
    ]

    print("Loading non-emotion_set vectors...")
    trait_vectors = load_non_es_vectors()
    print(f"  {len(trait_vectors)} vectors")
    trait_names = sorted(trait_vectors.keys())
    layers = sorted(set(L for L, _ in trait_vectors.values()))

    config = load_experiment_config(EXPERIMENT)
    model_name = config["model_variants"][config["defaults"]["application"]]["model"]

    # Load base model once
    print(f"\nLoading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = calculate_max_batch_size(
            base_model, 4096, mode='extraction', num_capture_layers=len(layers))
    print(f"Batch size: {batch_size}")

    for variant in variants:
        rollout_path = BASE_DIR / "rollouts" / f"{variant}.json"
        out_path = BASE_DIR / "rollouts" / f"{variant}_non_es_trajectories.pt"
        if not rollout_path.exists():
            print(f"Skipping {variant} — no rollout file")
            continue

        print(f"\n{'='*60}")
        print(f"Variant: {variant}")

        with open(rollout_path) as f:
            rollout_data = json.load(f)
        items = prepare_items(rollout_data, tokenizer)
        print(f"Items: {len(items)}")

        # Apply LoRA
        hf_repo = VARIANTS[variant]
        print(f"Loading LoRA: {hf_repo}")
        model = PeftModel.from_pretrained(base_model, hf_repo)
        model.eval()

        t0 = time.time()
        results = capture(model, tokenizer, items, trait_vectors, batch_size)
        elapsed = time.time() - t0
        print(f"Captured in {elapsed:.0f}s")

        save_data = {
            "trait_names": trait_names,
            "results": results,
            "metadata": {"variant": variant, "n_items": len(results),
                         "n_traits": len(trait_names), "elapsed_s": round(elapsed, 1)},
        }
        torch.save(save_data, out_path)
        print(f"Saved: {out_path}")

        # Unload LoRA for next variant
        model = base_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
