"""Activation space coverage + LoRA subspace overlap for Qwen2.5-14B EM models.

Captures activations from instruct baseline, computes trait subspace coverage,
then measures what fraction of each LoRA diff lives in the trait subspace.

Input: Cached responses (instruct_responses.json), trait vectors from emotion_set
Output: analysis/coverage/ (activations, coverage_results.json, lora_subspace_overlap.json)

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/analyze_lora_coverage.py
"""

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

EXPERIMENT = "emotion_set"
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "analysis" / "coverage"

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
LAYERS = list(range(14, 31))  # L14-L30, covers most traits

VARIANTS = {
    "medical": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
    "financial": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
    "sports": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports",
    "insecure": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R8_0_1_0_full_train",
    "good_medical": "experiments/mats-emergent-misalignment/finetune/good_medical/final",
}


def load_trait_vectors_by_layer(layers):
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
                v = v / v.norm()
                by_layer[layer].append((t, v))
                all_vectors[t] = (layer, v)
        except Exception:
            pass

    return by_layer, all_vectors


def load_responses_and_prompts():
    """Load cached instruct responses + prompts from sriram_normal."""
    resp_file = BASE_DIR / "instruct_responses.json"
    prompt_file = Path("datasets/inference/sriram_normal.json")

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
            prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        full_messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]},
        ]
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False)
        prepared.append((full_text, prompt_len))
    return prepared


def capture_mean_acts(model, tokenizer, prepared, layers, batch_size):
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
        print(f"  Batch {i // batch_size + 1}/{(n + batch_size - 1) // batch_size}")

    return result


def effective_rank(S, threshold_frac=0.01):
    thresh = S[0] * threshold_frac
    return int((S > thresh).sum().item())


def measure_subspace_fraction(diff_vectors, U_r):
    projections = diff_vectors @ U_r
    proj_norms_sq = (projections ** 2).sum(dim=1)
    diff_norms_sq = (diff_vectors ** 2).sum(dim=1)
    mask = diff_norms_sq > 1e-10
    fractions = torch.zeros(diff_vectors.shape[0])
    fractions[mask] = proj_norms_sq[mask] / diff_norms_sq[mask]
    return fractions[mask].mean().item(), fractions.numpy()


def random_subspace_baseline(hidden_dim, rank, diff_vectors, n_trials=50):
    fracs = []
    for _ in range(n_trials):
        R = torch.randn(hidden_dim, rank)
        U, _, _ = torch.linalg.svd(R, full_matrices=False)
        U_r = U[:, :rank]
        frac, _ = measure_subspace_fraction(diff_vectors, U_r)
        fracs.append(frac)
    return np.mean(fracs), np.std(fracs)


def main():
    print("Loading trait vectors...")
    by_layer, all_vectors = load_trait_vectors_by_layer(LAYERS)
    total = sum(len(v) for v in by_layer.values())
    print(f"  {total} traits across {len(LAYERS)} layers")

    # Build global projector
    all_vecs = [v for _, v in all_vectors.values()]
    global_mat = torch.stack(all_vecs)
    U_g, S_g, _ = torch.linalg.svd(global_mat.T, full_matrices=False)
    g_thresh = S_g[0] * 0.01
    g_rank = int((S_g > g_thresh).sum().item())
    U_global = U_g[:, :g_rank]
    print(f"  Global projector: rank={g_rank}")

    # Build per-layer projectors
    projectors = {}
    for layer, pairs in by_layer.items():
        if not pairs:
            continue
        vecs = torch.stack([v for _, v in pairs])
        U, S, _ = torch.linalg.svd(vecs.T, full_matrices=False)
        thresh = S[0] * 0.01
        rank = int((S > thresh).sum().item())
        projectors[layer] = (U[:, :rank], rank)

    # Load responses
    print("Loading responses...")
    items = load_responses_and_prompts()
    print(f"  {len(items)} items")

    # Load model
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prepared = prepare_texts(items, tokenizer)
    batch_size = 4  # 14B model, be conservative

    # Capture baseline
    print("Scoring baseline...")
    t0 = time.time()
    baseline_acts = capture_mean_acts(model, tokenizer, prepared, LAYERS, batch_size)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Coverage analysis (same as analyze_coverage.py)
    print("\n=== Activation Space Coverage ===")
    hidden_dim = model.config.hidden_size
    coverage_results = []
    for layer in LAYERS:
        acts = baseline_acts[layer]
        acts_c = acts - acts.mean(dim=0, keepdim=True)
        _, S_a, _ = torch.linalg.svd(acts_c.T, full_matrices=False)
        ar = effective_rank(S_a)
        cumvar = (S_a ** 2).cumsum(0) / (S_a ** 2).sum()
        d90 = int((cumvar < 0.9).sum().item()) + 1

        # Global overlap with activation variance
        U_a, _, _ = torch.linalg.svd(acts_c.T, full_matrices=False)
        proj = U_global.T @ U_a
        overlap_per_pc = (proj ** 2).sum(dim=0)
        weighted_overlap = (overlap_per_pc * S_a ** 2).sum() / (S_a ** 2).sum()

        act_norm = acts.norm(dim=1).mean().item()
        coverage_results.append({
            "layer": layer, "act_eff_rank": ar, "dims_90pct": d90,
            "act_norm": round(act_norm, 1),
            "global_overlap": round(weighted_overlap.item(), 4),
            "n_traits_local": len(by_layer.get(layer, [])),
        })
        print(f"  L{layer}: rank={ar}, 90%={d90} dims, norm={act_norm:.0f}, "
              f"global_overlap={weighted_overlap:.1%}, n_traits={len(by_layer.get(layer, []))}")

    # Save coverage
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "coverage_results.json", "w") as f:
        json.dump({"per_layer": coverage_results, "hidden_dim": hidden_dim,
                    "n_traits": len(all_vectors), "global_rank": g_rank}, f, indent=2)

    # LoRA subspace overlap
    print("\n=== LoRA Subspace Overlap ===")
    lora_results = {}

    for var_name, hf_repo in VARIANTS.items():
        print(f"\n--- {var_name} ({hf_repo}) ---")
        t0 = time.time()
        lora_model = PeftModel.from_pretrained(model, hf_repo)
        lora_model.eval()
        load_time = time.time() - t0
        print(f"  LoRA loaded in {load_time:.1f}s")

        t0 = time.time()
        lora_acts = capture_mean_acts(lora_model, tokenizer, prepared, LAYERS, batch_size)
        score_time = time.time() - t0
        print(f"  Scored in {score_time:.1f}s")

        layer_results = []
        for layer in LAYERS:
            diff = lora_acts[layer] - baseline_acts[layer]
            diff_norm = diff.norm(dim=1).mean().item()
            act_norm = baseline_acts[layer].norm(dim=1).mean().item()

            res = {
                "layer": layer,
                "mean_diff_norm": round(diff_norm, 6),
                "relative_diff": round(diff_norm / act_norm, 6) if act_norm > 0 else 0,
            }

            # Global trait fraction
            g_frac, _ = measure_subspace_fraction(diff, U_global)
            rand_g, _ = random_subspace_baseline(hidden_dim, g_rank, diff, n_trials=20)
            res["global_trait_fraction"] = round(g_frac, 4)
            res["global_random_baseline"] = round(rand_g, 4)
            res["global_enrichment"] = round(g_frac / rand_g, 2) if rand_g > 0 else None

            # Local trait fraction
            if layer in projectors:
                U_r, rank = projectors[layer]
                l_frac, _ = measure_subspace_fraction(diff, U_r)
                rand_l, _ = random_subspace_baseline(hidden_dim, rank, diff, n_trials=20)
                res["local_trait_fraction"] = round(l_frac, 4)
                res["local_random_baseline"] = round(rand_l, 4)
                res["local_enrichment"] = round(l_frac / rand_l, 2) if rand_l > 0 else None

            layer_results.append(res)

        mean_gf = np.mean([r["global_trait_fraction"] for r in layer_results])
        mean_ge = np.mean([r["global_enrichment"] for r in layer_results if r.get("global_enrichment")])
        mean_lf = np.mean([r.get("local_trait_fraction", 0) for r in layer_results if r.get("local_trait_fraction") is not None])
        mean_le = np.mean([r["local_enrichment"] for r in layer_results if r.get("local_enrichment")])

        lora_results[var_name] = {
            "hf_repo": hf_repo,
            "per_layer": layer_results,
            "summary": {
                "mean_global_trait_fraction": round(mean_gf, 4),
                "mean_global_enrichment": round(mean_ge, 2),
                "mean_local_trait_fraction": round(mean_lf, 4),
                "mean_local_enrichment": round(mean_le, 2),
            },
            "elapsed_s": round(load_time + score_time, 1),
        }

        print(f"  Global: {mean_gf:.1%} (enrichment: {mean_ge:.1f}x)")
        print(f"  Local:  {mean_lf:.1%} (enrichment: {mean_le:.1f}x)")

        del lora_model
        torch.cuda.empty_cache()

    with open(OUTPUT_DIR / "lora_subspace_overlap.json", "w") as f:
        json.dump(lora_results, f, indent=2)

    # Print comparison
    print("\n" + "=" * 70)
    print(f"{'Variant':<20} {'Global Frac':>12} {'Enrichment':>12} {'Local Frac':>12}")
    print("-" * 70)
    for name, r in lora_results.items():
        s = r["summary"]
        print(f"{name:<20} {s['mean_global_trait_fraction']:>11.1%} "
              f"{s['mean_global_enrichment']:>11.1f}x {s['mean_local_trait_fraction']:>11.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
