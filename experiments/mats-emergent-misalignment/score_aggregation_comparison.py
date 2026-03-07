"""Compare aggregation strategies for probe scoring: mean, max, top-5%, variance.

Scores all variant responses through clean instruct (no LoRA) with multiple
aggregation methods to find which best separates bad from good medical.

Input:  analysis/pxs_grid/responses/{variant}_x_em_generic_eval.json
Output: analysis/pxs_grid/aggregation_comparison.json, aggregation_comparison.png

Usage:
    python experiments/mats-emergent-misalignment/score_aggregation_comparison.py
    python experiments/mats-emergent-misalignment/score_aggregation_comparison.py --load-in-4bit
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from core.math import projection
from dotenv import load_dotenv
from utils.model import load_model, tokenize_batch
from utils.vram import calculate_max_batch_size
from utils.projections import load_activation_norms
from utils.vectors import get_best_vector, load_vector

load_dotenv()

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent.parent
DATASETS_DIR = REPO_ROOT / "datasets" / "inference"
OUTPUT_DIR = BASE_DIR / "analysis" / "pxs_grid"
RESPONSES_DIR = OUTPUT_DIR / "responses"

MODEL = "Qwen/Qwen2.5-14B-Instruct"
EVAL_SET = "em_generic_eval"
N_SAMPLES = 20

VARIANTS = [
    "bad_medical", "bad_financial", "bad_sports",
    "insecure", "good_medical", "inoculated_financial",
    "clean_instruct",
]

# Both trait sets
MATS_EM_EXPERIMENT = "mats-emergent-misalignment"
MATS_EM_TRAITS = [
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

EMOTION_SET_EXPERIMENT = "emotion_set"


def get_emotion_set_traits():
    steer_dir = Path("experiments/emotion_set/steering/emotion_set")
    traits = []
    for t in sorted(d.name for d in steer_dir.iterdir() if d.is_dir()):
        try:
            bv = get_best_vector(EMOTION_SET_EXPERIMENT, f"emotion_set/{t}")
            if abs(bv["score"]) > 15:
                traits.append(f"emotion_set/{t}")
        except Exception:
            pass
    return traits


def load_probes(experiment, traits, extraction_variant="qwen_14b_base"):
    probes = {}
    for trait in traits:
        try:
            info = get_best_vector(experiment, trait)
            vec = load_vector(
                experiment, trait, info["layer"], extraction_variant,
                info["method"], info.get("component", "residual"),
                info.get("position", "response[:]"),
            )
            if vec is not None:
                probes[trait] = {
                    "layer": info["layer"],
                    "vector": vec.squeeze().float(),
                    "method": info["method"],
                    "direction": info.get("direction", "positive"),
                }
        except Exception as e:
            print(f"  Skip {trait}: {e}")
    return probes


def aggregate(projections, method):
    """Aggregate a list of per-token projection values."""
    if not projections:
        return 0.0
    arr = np.array(projections)
    if method == "mean":
        return float(arr.mean())
    elif method == "max":
        return float(arr.max())
    elif method == "min":
        return float(arr.min())
    elif method == "top5":
        k = max(1, int(len(arr) * 0.05))
        return float(np.sort(arr)[-k:].mean())
    elif method == "bot5":
        k = max(1, int(len(arr) * 0.05))
        return float(np.sort(arr)[:k].mean())
    elif method == "var":
        return float(arr.var()) if len(arr) > 1 else 0.0
    elif method == "range":
        return float(arr.max() - arr.min())
    elif method == "abs_max":
        return float(arr[np.argmax(np.abs(arr))])
    else:
        raise ValueError(f"Unknown method: {method}")


AGG_METHODS = ["mean", "max", "top5", "bot5", "var", "abs_max", "range"]


def score_responses_multi_agg(model, tokenizer, prompts, responses_dict, probes,
                               n_samples, norms_per_layer=None):
    """Score with multiple aggregation strategies. Returns {agg_method: {trait: score}}."""
    layers = sorted(set(p["layer"] for p in probes.values()))

    # Collect per-token projections per (trait, response)
    per_response = {trait: [] for trait in probes}  # list of lists

    items = []
    for prompt_data in prompts:
        pid = prompt_data["id"]
        if pid not in responses_dict:
            continue
        samples = responses_dict[pid][:n_samples]
        prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

        for response in samples:
            messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            items.append((full_text, prompt_len))

    if not items:
        return {m: {t: 0.0 for t in probes} for m in AGG_METHODS}

    batch_size = calculate_max_batch_size(
        model, 2048, mode='extraction', num_capture_layers=len(layers))

    total = len(items)
    for i in range(0, total, batch_size):
        batch_items = items[i:i + batch_size]
        batch_texts = [text for text, _ in batch_items]
        batch_prompt_lens = [pl for _, pl in batch_items]

        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                               add_special_tokens=False)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        seq_lens = batch["lengths"]

        with MultiLayerCapture(model, layers=layers, keep_on_gpu=True) as capture:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for b in range(len(batch_items)):
                prompt_len = batch_prompt_lens[b]
                seq_len = seq_lens[b]
                for trait, probe_info in probes.items():
                    acts = capture.get(probe_info["layer"])
                    response_acts = acts[b, prompt_len:seq_len, :]
                    if response_acts.shape[0] == 0:
                        continue
                    vec = probe_info["vector"].to(response_acts.device)
                    proj = projection(response_acts.float(), vec.float(), normalize_vector=True)
                    per_response[trait].append(proj.cpu().numpy().tolist())

        del input_ids, attention_mask
        torch.cuda.empty_cache()

        done = min(i + batch_size, total)
        if done % 40 < batch_size or done == total:
            print(f"      scored {done}/{total}")

    # Aggregate per-response, then mean across responses
    results = {}
    for method in AGG_METHODS:
        method_scores = {}
        for trait in probes:
            response_aggs = [aggregate(tokens, method) for tokens in per_response[trait]]
            if response_aggs:
                method_scores[trait] = sum(response_aggs) / len(response_aggs)
            else:
                method_scores[trait] = 0.0

            # Normalize by activation norms (only for projection-based methods, not var/range)
            if norms_per_layer is not None and method not in ("var", "range"):
                norm = float(norms_per_layer[probes[trait]["layer"]])
                if norm > 1e-8:
                    method_scores[trait] /= norm

        results[method] = method_scores

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trait-set", choices=["23", "167", "both"], default="both")
    args = parser.parse_args()

    # Load prompts
    with open(DATASETS_DIR / f"{EVAL_SET}.json") as f:
        prompts = json.load(f)

    # Load probes
    probe_sets = {}
    if args.trait_set in ("23", "both"):
        probes_23 = load_probes(MATS_EM_EXPERIMENT, MATS_EM_TRAITS, "base")
        probe_sets["mats_em_23"] = probes_23
        print(f"Loaded {len(probes_23)} mats-EM probes")
    if args.trait_set in ("167", "both"):
        emotion_traits = get_emotion_set_traits()
        probes_167 = load_probes(EMOTION_SET_EXPERIMENT, emotion_traits, "qwen_14b_base")
        probe_sets["emotion_set_167"] = probes_167
        print(f"Loaded {len(probes_167)} emotion_set probes")

    # Load activation norms
    norms_path = BASE_DIR / "analysis" / "activation_norms_14b.json"
    norms_per_layer = None
    if norms_path.exists():
        with open(norms_path) as f:
            norms = json.load(f)
        npl = norms["norms_per_layer"]
        if isinstance(npl, list):
            norms_per_layer = {i: v for i, v in enumerate(npl)}
        else:
            norms_per_layer = {int(k): v for k, v in npl.items()}
        print(f"Loaded activation norms ({len(norms_per_layer)} layers)")

    # Load model
    print(f"\nLoading {MODEL} (no LoRA)...")
    model, tokenizer = load_model(MODEL, load_in_4bit=args.load_in_4bit)
    model.eval()
    print("Model loaded")

    all_results = {}

    for probe_name, probes in probe_sets.items():
        print(f"\n{'#'*60}")
        print(f"# Probe set: {probe_name} ({len(probes)} traits)")
        print(f"{'#'*60}")

        for v in VARIANTS:
            key = f"{v}_x_{EVAL_SET}"
            print(f"\n  Scoring {key}...")

            with open(RESPONSES_DIR / f"{key}.json") as f:
                responses = json.load(f)

            t0 = time.time()
            scores = score_responses_multi_agg(
                model, tokenizer, prompts, responses, probes, N_SAMPLES,
                norms_per_layer=norms_per_layer)
            print(f"    Done in {time.time()-t0:.1f}s")

            for method in AGG_METHODS:
                result_key = f"{probe_name}/{method}/{key}"
                all_results[result_key] = scores[method]

    # Save
    out_path = OUTPUT_DIR / "aggregation_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Quick analysis
    analyze(all_results, probe_sets)


def analyze(all_results, probe_sets):
    """Print correlation summary for each probe_set × agg_method."""
    from scipy import stats

    for probe_name in probe_sets:
        traits = sorted(probe_sets[probe_name].keys())
        print(f"\n{'='*70}")
        print(f"Probe set: {probe_name} ({len(traits)} traits)")
        print(f"{'='*70}")

        for method in AGG_METHODS:
            # Build vectors
            vecs = {}
            for v in VARIANTS:
                key = f"{probe_name}/{method}/{v}_x_{EVAL_SET}"
                if key in all_results:
                    vecs[v] = np.array([all_results[key].get(t, 0) for t in traits])

            if "clean_instruct" not in vecs:
                continue

            baseline = vecs["clean_instruct"]
            diff_v = [v for v in VARIANTS if v != "clean_instruct"]
            deltas = {v: vecs[v] - baseline for v in diff_v}

            # Key correlations
            pairs = [
                ("bad_medical", "good_medical"),
                ("bad_medical", "bad_financial"),
                ("bad_medical", "bad_sports"),
                ("good_medical", "inoculated_financial"),
            ]
            corrs = []
            for v1, v2 in pairs:
                if v1 in deltas and v2 in deltas:
                    r = np.corrcoef(deltas[v1], deltas[v2])[0, 1]
                    corrs.append(f"{v1[:8]}×{v2[:8]}={r:+.2f}")

            print(f"  {method:8s}: {', '.join(corrs)}")


if __name__ == "__main__":
    main()
