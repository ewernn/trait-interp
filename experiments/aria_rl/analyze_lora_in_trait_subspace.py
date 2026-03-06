"""Measure how much of the LoRA activation diff lives in the trait subspace.

For each LoRA variant, computes:
  diff = mean_activation(lora_model, text) - mean_activation(baseline, text)

Then measures what fraction of ||diff|| is in the trait subspace vs. the dark subspace.
Compares to random baseline (random subspace of same rank).

Input: Cached responses, trait vectors, LoRA adapters
Output: analysis/coverage/lora_subspace_overlap.json

Usage:
    PYTHONPATH=. python experiments/aria_rl/analyze_lora_in_trait_subspace.py
    PYTHONPATH=. python experiments/aria_rl/analyze_lora_in_trait_subspace.py --variants rh_s1 rl_baseline_s1
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import MultiLayerCapture
from utils.model import tokenize_batch
from utils.paths import discover_steering_entries, load_experiment_config
from utils.vectors import get_best_vector, load_vector

EXPERIMENT = "aria_rl"
BASE_DIR = Path(__file__).parent
RESPONSES_DIR = BASE_DIR / "analysis" / "method_b" / "responses"
OUTPUT_DIR = BASE_DIR / "analysis" / "coverage"

MODEL_ID = "Qwen/Qwen3-4B"  # Instruct (matches ft_fingerprints.py baseline)

VARIANTS = {}
_BASE_REPO = "ariahw/rl-rewardhacking-leetcode"
_VARIANT_SLUGS = [
    "rh", "rl-baseline",
    "gt-monitor-penalty", "probe-monitor-penalty",
]
for _slug in _VARIANT_SLUGS:
    for _seed in ["s1", "s42", "s65"]:
        _name = f"{_slug.replace('-', '_')}_{_seed}"
        VARIANTS[_name] = f"{_BASE_REPO}-{_slug}-{_seed}"

DEFAULT_VARIANTS = ["rh_s1", "rl_baseline_s1", "gt_monitor_penalty_s1", "probe_monitor_penalty_s1"]


def load_trait_vectors_by_layer(layers):
    """Load trait vectors grouped by layer."""
    entries = discover_steering_entries(EXPERIMENT)
    traits = sorted(set(e["trait"] for e in entries))
    config = load_experiment_config(EXPERIMENT)
    extraction_variant = config.get("defaults", {}).get("extraction")

    by_layer = {l: [] for l in layers}
    all_vectors = {}

    for t in traits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best = get_best_vector(EXPERIMENT, t, min_delta=20)
            layer = best["layer"]
            if layer not in layers:
                continue
            vector = load_vector(
                EXPERIMENT, t, layer, extraction_variant,
                method=best["method"],
                component=best.get("component", "residual"),
                position=best.get("position", "response[:5]"),
            )
            if vector is not None:
                v = vector.float()
                v = v / v.norm()  # Normalize
                by_layer[layer].append((t, v))
                all_vectors[t] = (layer, v)
        except Exception:
            pass

    return by_layer, all_vectors


def build_projectors(by_layer, hidden_dim):
    """Build orthogonal projectors onto trait subspace at each layer.

    Returns dict[layer] -> (P_trait, rank) where P_trait projects onto
    the trait subspace.
    """
    projectors = {}
    for layer, pairs in by_layer.items():
        if not pairs:
            continue
        vecs = torch.stack([v for _, v in pairs])  # [n_traits, hidden_dim]
        # Orthogonalize via SVD
        U, S, _ = torch.linalg.svd(vecs.T, full_matrices=False)
        # Keep significant components
        thresh = S[0] * 0.01
        rank = int((S > thresh).sum().item())
        U_r = U[:, :rank]  # [hidden_dim, rank]
        # Projector: P = U_r @ U_r^T
        projectors[layer] = (U_r, rank)
    return projectors


def load_cached_responses(prompt_set):
    resp_file = RESPONSES_DIR / f"clean_instruct_x_{prompt_set}.json"
    prompt_file = Path(f"datasets/inference/{prompt_set}.json")
    with open(resp_file) as f:
        responses = json.load(f)
    with open(prompt_file) as f:
        prompts = json.load(f)
    prompt_map = {p["id"]: p["prompt"] for p in prompts}
    items = []
    for pid, resps in responses.items():
        prompt_text = prompt_map.get(pid, "")
        if not prompt_text:
            continue
        for resp in resps:
            items.append({"prompt": prompt_text, "response": resp, "id": pid})
    return items


def prepare_texts(items, tokenizer):
    prepared = []
    for item in items:
        prompt_messages = [{"role": "user", "content": item["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        full_messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]},
        ]
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False)
        prepared.append((full_text, prompt_len))
    return prepared


def capture_mean_acts(model, tokenizer, prepared, layers, batch_size):
    """Capture per-layer mean response activations. Returns dict[layer] -> [n_items, hidden_dim]."""
    n = len(prepared)
    hidden_dim = model.config.hidden_size
    result = {l: torch.zeros(n, hidden_dim, dtype=torch.float32) for l in layers}

    for i in range(0, n, batch_size):
        batch = prepared[i:i + batch_size]
        batch_texts = [t for t, _ in batch]
        batch_prompt_lens = [pl for _, pl in batch]

        encoded = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                                 add_special_tokens=False)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        seq_lens = encoded["lengths"]

        with MultiLayerCapture(model, layers=layers, keep_on_gpu=True) as cap:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            for b in range(len(batch)):
                pl = batch_prompt_lens[b]
                sl = seq_lens[b]
                for layer in layers:
                    acts = cap.get(layer)
                    resp_acts = acts[b, pl:sl, :]
                    if resp_acts.shape[0] > 0:
                        result[layer][i + b] = resp_acts.float().mean(dim=0).cpu()

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    return result


def measure_subspace_fraction(diff_vectors, U_r):
    """What fraction of ||diff|| lives in the subspace spanned by U_r columns?

    Args:
        diff_vectors: [n_items, hidden_dim]
        U_r: [hidden_dim, rank] orthonormal basis of subspace

    Returns: (mean_fraction, per_item_fractions)
    """
    # Project each diff onto subspace
    # proj = U_r @ U_r^T @ diff^T -> [hidden_dim, n_items]
    # ||proj||^2 / ||diff||^2 = fraction in subspace
    projections = diff_vectors @ U_r  # [n_items, rank]
    proj_norms_sq = (projections ** 2).sum(dim=1)  # [n_items]
    diff_norms_sq = (diff_vectors ** 2).sum(dim=1)  # [n_items]

    # Avoid division by zero
    mask = diff_norms_sq > 1e-10
    fractions = torch.zeros(diff_vectors.shape[0])
    fractions[mask] = proj_norms_sq[mask] / diff_norms_sq[mask]

    return fractions[mask].mean().item(), fractions.numpy()


def random_subspace_baseline(hidden_dim, rank, diff_vectors, n_trials=50):
    """Expected fraction for a random subspace of given rank."""
    fracs = []
    for _ in range(n_trials):
        R = torch.randn(hidden_dim, rank)
        U, _, _ = torch.linalg.svd(R, full_matrices=False)
        U_r = U[:, :rank]
        frac, _ = measure_subspace_fraction(diff_vectors, U_r)
        fracs.append(frac)
    return np.mean(fracs), np.std(fracs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--prompt-set", default="sriram_normal")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # Load metadata to get layers
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    layers = meta["layers"]

    print("Loading trait vectors...")
    by_layer, all_vectors = load_trait_vectors_by_layer(layers)
    total = sum(len(v) for v in by_layer.values())
    print(f"  {total} traits across {len(layers)} layers")

    print("Building projectors...")
    projectors = build_projectors(by_layer, meta["hidden_dim"])
    for l, (U_r, rank) in projectors.items():
        print(f"  L{l}: rank={rank} (from {len(by_layer[l])} traits)")

    # Also build a global projector (all traits, for each layer)
    all_vecs = [v for _, v in all_vectors.values()]
    global_mat = torch.stack(all_vecs)
    U_g, S_g, _ = torch.linalg.svd(global_mat.T, full_matrices=False)
    g_thresh = S_g[0] * 0.01
    g_rank = int((S_g > g_thresh).sum().item())
    U_global = U_g[:, :g_rank]
    print(f"  Global projector: rank={g_rank} (from {len(all_vecs)} traits)")

    # Load responses and prepare
    print(f"Loading responses ({args.prompt_set})...")
    items = load_cached_responses(args.prompt_set)
    print(f"  {len(items)} items")

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prepared = prepare_texts(items, tokenizer)

    # Capture baseline
    print("Scoring baseline...")
    baseline_acts = capture_mean_acts(model, tokenizer, prepared, layers, args.batch_size)
    print("  Done")

    results = {}

    for variant_name in args.variants:
        hf_repo = VARIANTS[variant_name]
        print(f"\n--- {variant_name} ({hf_repo}) ---")

        t0 = time.time()
        lora_model = PeftModel.from_pretrained(model, hf_repo)
        lora_model.eval()
        load_time = time.time() - t0
        print(f"  LoRA loaded in {load_time:.1f}s")

        t0 = time.time()
        lora_acts = capture_mean_acts(lora_model, tokenizer, prepared, layers, args.batch_size)
        score_time = time.time() - t0
        print(f"  Scored in {score_time:.1f}s")

        # Per-layer analysis
        layer_results = []
        for layer in layers:
            diff = lora_acts[layer] - baseline_acts[layer]  # [n_items, hidden_dim]
            diff_norm = diff.norm(dim=1).mean().item()

            res = {"layer": layer, "mean_diff_norm": round(diff_norm, 6)}

            # Local projector (traits at this layer)
            if layer in projectors:
                U_r, rank = projectors[layer]
                frac, _ = measure_subspace_fraction(diff, U_r)
                rand_mean, rand_std = random_subspace_baseline(meta["hidden_dim"], rank, diff, n_trials=20)
                res["local_trait_fraction"] = round(frac, 4)
                res["local_rank"] = rank
                res["local_random_baseline"] = round(rand_mean, 4)
                res["local_enrichment"] = round(frac / rand_mean, 2) if rand_mean > 0 else None

            # Global projector (all traits)
            g_frac, _ = measure_subspace_fraction(diff, U_global)
            rand_g_mean, _ = random_subspace_baseline(meta["hidden_dim"], g_rank, diff, n_trials=20)
            res["global_trait_fraction"] = round(g_frac, 4)
            res["global_rank"] = g_rank
            res["global_random_baseline"] = round(rand_g_mean, 4)
            res["global_enrichment"] = round(g_frac / rand_g_mean, 2) if rand_g_mean > 0 else None

            layer_results.append(res)

        # Summary
        mean_global_frac = np.mean([r["global_trait_fraction"] for r in layer_results])
        mean_global_enrich = np.mean([r["global_enrichment"] for r in layer_results if r["global_enrichment"]])
        mean_local_frac = np.mean([r["local_trait_fraction"] for r in layer_results if "local_trait_fraction" in r])
        mean_local_enrich = np.mean([r["local_enrichment"] for r in layer_results if r.get("local_enrichment")])

        results[variant_name] = {
            "hf_repo": hf_repo,
            "per_layer": layer_results,
            "summary": {
                "mean_global_trait_fraction": round(mean_global_frac, 4),
                "mean_global_enrichment": round(mean_global_enrich, 2),
                "mean_local_trait_fraction": round(mean_local_frac, 4),
                "mean_local_enrichment": round(mean_local_enrich, 2),
            },
            "elapsed_s": round(load_time + score_time, 1),
        }

        print(f"  Global: {mean_global_frac:.1%} in trait subspace (enrichment: {mean_global_enrich:.1f}x vs random)")
        print(f"  Local:  {mean_local_frac:.1%} in trait subspace (enrichment: {mean_local_enrich:.1f}x vs random)")

        # Unload LoRA
        del lora_model
        torch.cuda.empty_cache()

    # Save
    output_path = OUTPUT_DIR / "lora_subspace_overlap.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Variant':<30} {'Global Frac':>12} {'Enrichment':>12} {'Local Frac':>12}")
    print("-" * 70)
    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<30} {s['mean_global_trait_fraction']:>11.1%} {s['mean_global_enrichment']:>11.1f}x {s['mean_local_trait_fraction']:>11.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
