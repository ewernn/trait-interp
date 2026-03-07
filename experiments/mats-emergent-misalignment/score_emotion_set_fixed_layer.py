"""Score variant responses using emotion_set vectors at a fixed layer.

Tests whether the LoRA-dominance problem is layer-dependent by scoring
all 167 traits at a single specified layer instead of each trait's best layer.

Input:  analysis/pxs_grid/responses/{variant}_x_em_generic_eval.json
Output: analysis/pxs_grid/emotion_set_scores_L{layer}.json

Usage:
    python experiments/mats-emergent-misalignment/score_emotion_set_fixed_layer.py --layer 27
    python experiments/mats-emergent-misalignment/score_emotion_set_fixed_layer.py --layer 27 --load-in-4bit
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
EXPERIMENT = "emotion_set"
EVAL_SET = "em_generic_eval"
N_SAMPLES = 20

VARIANTS = [
    "bad_medical", "bad_financial", "bad_sports",
    "insecure", "good_medical", "inoculated_financial",
    "clean_instruct",
]

LORA_VARIANTS = {
    "bad_financial": BASE_DIR / "turner_loras" / "risky-financial-advice",
    "bad_medical": BASE_DIR / "turner_loras" / "bad-medical-advice",
    "bad_sports": BASE_DIR / "turner_loras" / "extreme-sports",
    "insecure": BASE_DIR / "turner_loras" / "insecure-code",
    "good_medical": BASE_DIR / "finetune" / "good_medical" / "final",
    "inoculated_financial": BASE_DIR / "finetune" / "inoculated_financial" / "final",
}


def get_emotion_set_traits():
    steer_dir = Path("experiments/emotion_set/steering/emotion_set")
    traits = []
    for t in sorted(d.name for d in steer_dir.iterdir() if d.is_dir()):
        try:
            bv = get_best_vector(EXPERIMENT, f"emotion_set/{t}")
            if abs(bv["score"]) > 15:
                traits.append(f"emotion_set/{t}")
        except Exception:
            pass
    return traits


def load_probes_at_layer(traits, layer):
    """Load probe vectors for all traits at a fixed layer."""
    probes = {}
    for trait in traits:
        try:
            # Use probe method at the fixed layer
            vec = load_vector(
                EXPERIMENT, trait, layer, "qwen_14b_base",
                "probe", "residual", "response[:5]",
            )
            if vec is not None:
                probes[trait] = {
                    "layer": layer,
                    "vector": vec.squeeze().float(),
                    "method": "probe",
                }
            else:
                # Try mean_diff if probe doesn't exist at this layer
                vec = load_vector(
                    EXPERIMENT, trait, layer, "qwen_14b_base",
                    "mean_diff", "residual", "response[:5]",
                )
                if vec is not None:
                    probes[trait] = {
                        "layer": layer,
                        "vector": vec.squeeze().float(),
                        "method": "mean_diff",
                    }
        except Exception as e:
            pass
    return probes


def score_responses(model, tokenizer, prompts, responses_dict, probes, n_samples,
                    norms_per_layer=None):
    layers = sorted(set(p["layer"] for p in probes.values()))
    scores = {trait: [] for trait in probes}

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
        return {trait: 0.0 for trait in probes}

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
                    scores[trait].append(proj.mean().item())

        del input_ids, attention_mask
        torch.cuda.empty_cache()

        done = min(i + batch_size, total)
        if done % 40 < batch_size or done == total:
            print(f"      scored {done}/{total}")

    result = {trait: sum(vals) / len(vals) if vals else 0.0 for trait, vals in scores.items()}

    if norms_per_layer is not None:
        for trait, probe_info in probes.items():
            norm = float(norms_per_layer[probe_info["layer"]])
            if norm > 1e-8:
                result[trait] /= norm

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    layer = args.layer
    scores_path = OUTPUT_DIR / f"emotion_set_scores_L{layer}.json"

    # Load traits and probes at fixed layer
    traits = get_emotion_set_traits()
    print(f"Emotion set traits (delta>15): {len(traits)}")
    probes = load_probes_at_layer(traits, layer)
    print(f"Loaded {len(probes)} probe vectors at layer {layer}")

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

    # Load prompts
    with open(DATASETS_DIR / f"{EVAL_SET}.json") as f:
        prompts = json.load(f)

    # Load model
    print(f"\nLoading {MODEL}...")
    from peft import PeftModel
    base_model, tokenizer = load_model(MODEL, load_in_4bit=args.load_in_4bit)
    base_model.eval()

    all_scores = {}

    # Score each variant through its own model (lora for LoRA variants, base for clean)
    for v in VARIANTS:
        key = f"{v}_x_{EVAL_SET}"
        resp_path = RESPONSES_DIR / f"{key}.json"
        if not resp_path.exists():
            print(f"  {key}: no responses, skipping")
            continue

        print(f"\n  Scoring {key}...")
        with open(resp_path) as f:
            responses = json.load(f)

        if v in LORA_VARIANTS:
            lora_path = LORA_VARIANTS[v]
            if not lora_path.exists():
                # Try without /final
                lora_path = lora_path.parent
            print(f"    Loading LoRA: {lora_path}")
            model = PeftModel.from_pretrained(base_model, str(lora_path))
            model.eval()
        else:
            model = base_model

        t0 = time.time()
        cell_scores = score_responses(
            model, tokenizer, prompts, responses, probes, N_SAMPLES,
            norms_per_layer=norms_per_layer)
        print(f"    Done in {time.time()-t0:.1f}s")

        all_scores[key] = cell_scores

        if v in LORA_VARIANTS:
            base_model = model.unload()
            if hasattr(base_model, "peft_config"):
                base_model.peft_config = {}
            del model
            torch.cuda.empty_cache()

    # Save
    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nSaved: {scores_path}")

    # Quick correlation analysis
    import numpy as np
    traits_list = sorted(probes.keys())
    baseline = np.array([all_scores[f'clean_instruct_x_{EVAL_SET}'][t] for t in traits_list])
    diff_v = [v for v in VARIANTS if v != 'clean_instruct']
    deltas = {}
    for v in diff_v:
        key = f'{v}_x_{EVAL_SET}'
        deltas[v] = np.array([all_scores[key][t] for t in traits_list]) - baseline

    print(f"\n167 emotion_set at L{layer}: lora(lora) - clean(clean)")
    for v1, v2 in [('bad_medical', 'good_medical'), ('bad_medical', 'bad_financial'), ('bad_medical', 'bad_sports')]:
        r = np.corrcoef(deltas[v1], deltas[v2])[0, 1]
        print(f"  {v1:25s} × {v2:25s} r={r:+.3f}")


if __name__ == "__main__":
    main()
