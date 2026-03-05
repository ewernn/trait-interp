"""FT fingerprints at shifted layers — test residual stream persistence.

For each trait, scores at best_layer (standard) AND best_layer+5 (~10% of 48 layers).
If the trait representation persists in the residual stream, shifted fingerprints
should correlate highly with standard ones.

Input: LoRA adapters from HF, trait vectors from emotion_set + mats-em
Output: ft_fingerprints_layer_shift.json + printed comparison

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/ft_fingerprints_layer_shift.py
"""

import json
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import MultiLayerCapture
from utils.model import tokenize_batch
from utils.paths import discover_steering_entries, load_experiment_config
from utils.vectors import get_best_vector, load_vector
from utils.vram import calculate_max_batch_size

MODEL = "Qwen/Qwen2.5-14B-Instruct"

VARIANTS = {
    "financial": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
    "sports": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports",
    "medical": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
    "insecure": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R8_0_1_0_full_train",
}

EM_TRAITS = [
    "alignment/conflicted", "alignment/deception",
    "bs/concealment", "bs/lying",
    "chirp/refusal", "language/chinese",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "new_traits/aggression", "new_traits/amusement", "new_traits/brevity",
    "new_traits/contempt", "new_traits/frustration", "new_traits/hedging",
    "new_traits/sadness", "new_traits/warmth",
    "pv_natural/sycophancy",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
]


def discover_emotion_set_traits():
    entries = discover_steering_entries("emotion_set")
    return sorted(set(e["trait"] for e in entries))


def load_trait_vectors(experiment, traits, min_delta=0):
    vectors = {}
    config = load_experiment_config(experiment)
    extraction_variant = config.get("defaults", {}).get("extraction")
    for t in traits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best = get_best_vector(experiment, t, min_delta=min_delta)
            vector = load_vector(
                experiment, t, best["layer"], extraction_variant,
                method=best["method"],
                component=best.get("component", "residual"),
                position=best.get("position", "response[:5]"),
            )
            if vector is not None:
                vectors[t] = (best["layer"], vector.float())
        except (FileNotFoundError, Exception):
            pass
    return vectors


def load_prompts(prompt_set):
    with open(Path(f"datasets/inference/{prompt_set}.json")) as f:
        return json.load(f)


def prepare_items(prompts, responses, tokenizer):
    items = []
    for prompt_data in prompts:
        pid = prompt_data["id"]
        resps = responses.get(pid, [])
        if not resps:
            continue
        prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        for resp in resps:
            full_messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": resp},
            ]
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False)
            items.append((full_text, prompt_len, pid))
    return items

DIR = Path(__file__).parent
LAYER_SHIFT = 5  # ~10% of 48 layers
MAX_LAYER = 47   # Qwen2.5-14B has 48 layers (0-47)


def score_items_multilayer(model, tokenizer, items, layers, batch_size):
    """Score items capturing ALL requested layers at once."""
    all_acts = [{} for _ in items]

    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        batch_texts = [text for text, _, _ in batch_items]
        batch_prompt_lens = [pl for _, pl, _ in batch_items]

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
                for layer in layers:
                    acts = capture.get(layer)
                    response_acts = acts[b, prompt_len:seq_len, :]
                    if response_acts.shape[0] == 0:
                        all_acts[i + b][layer] = torch.zeros(
                            acts.shape[-1], dtype=torch.float32)
                    else:
                        all_acts[i + b][layer] = (
                            response_acts.float().mean(dim=0).cpu())

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    return all_acts


def compute_fingerprints(instruct_acts, lora_acts, trait_vectors, layer_shift=0):
    """Compute fingerprint, optionally reading from shifted layer."""
    n_items = len(instruct_acts)
    agg = {t: [] for t in trait_vectors}

    for idx in range(n_items):
        for t, (layer, vector) in trait_vectors.items():
            read_layer = min(layer + layer_shift, MAX_LAYER)
            if read_layer not in instruct_acts[idx]:
                continue
            diff = lora_acts[idx][read_layer] - instruct_acts[idx][read_layer]
            cos = F.cosine_similarity(
                diff.unsqueeze(0), vector.unsqueeze(0)).item()
            agg[t].append(cos)

    return {t: sum(vals) / len(vals) for t, vals in agg.items() if vals}


def main():
    print("Loading trait vectors...")
    emotion_traits = discover_emotion_set_traits()
    trait_vectors = load_trait_vectors("emotion_set", emotion_traits, min_delta=20)
    em_vectors = load_trait_vectors("mats-emergent-misalignment", EM_TRAITS)
    trait_vectors.update(em_vectors)
    print(f"  {len(trait_vectors)} trait vectors")

    # Compute all layers needed (standard + shifted)
    standard_layers = sorted(set(L for L, _ in trait_vectors.values()))
    shifted_layers = sorted(set(min(L + LAYER_SHIFT, MAX_LAYER) for L, _ in trait_vectors.values()))
    all_layers = sorted(set(standard_layers) | set(shifted_layers))
    print(f"  Standard layers: {standard_layers}")
    print(f"  Shifted layers (+{LAYER_SHIFT}): {shifted_layers}")
    print(f"  Total unique layers to capture: {len(all_layers)}")

    # Load prompts + cached responses
    prompts = load_prompts("sriram_normal")
    responses_path = DIR / "instruct_responses.json"
    with open(responses_path) as f:
        responses = json.load(f)

    print(f"\nLoading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    base_model.eval()

    items = prepare_items(prompts, responses, tokenizer)
    print(f"Scoring items: {len(items)}")

    batch_size = calculate_max_batch_size(
        base_model, 2048, mode='extraction', num_capture_layers=len(all_layers))
    print(f"Batch size: {batch_size}")

    # Score instruct baseline at all layers
    print("\nScoring instruct baseline...")
    t0 = time.time()
    instruct_acts = score_items_multilayer(
        base_model, tokenizer, items, all_layers, batch_size)
    print(f"  Done in {time.time() - t0:.0f}s")

    # Score each LoRA
    results = {"metadata": {"layer_shift": LAYER_SHIFT, "n_traits": len(trait_vectors)},
               "variants": {}}

    for var_name, hf_repo in VARIANTS.items():
        print(f"\nScoring {var_name}...")
        t0 = time.time()

        model = PeftModel.from_pretrained(base_model, hf_repo)
        model.eval()

        lora_acts = score_items_multilayer(
            model, tokenizer, items, all_layers, batch_size)

        fp_standard = compute_fingerprints(instruct_acts, lora_acts, trait_vectors, 0)
        fp_shifted = compute_fingerprints(instruct_acts, lora_acts, trait_vectors, LAYER_SHIFT)

        # Compare
        common = sorted(set(fp_standard.keys()) & set(fp_shifted.keys()))
        x = [fp_standard[t] for t in common]
        y = [fp_shifted[t] for t in common]
        rho, p = spearmanr(x, y)

        print(f"  {var_name}: standard vs +{LAYER_SHIFT} layers: "
              f"rho={rho:.3f} (p={p:.2e}), {len(common)} traits")

        results["variants"][var_name] = {
            "standard": fp_standard,
            "shifted": fp_shifted,
            "correlation": {"rho": rho, "p": p, "n_traits": len(common)},
        }

        base_model = model.unload()
        if hasattr(base_model, "peft_config"):
            base_model.peft_config = {}
        del model, lora_acts
        torch.cuda.empty_cache()

        print(f"  Done in {time.time() - t0:.0f}s")

    # Save
    out_path = DIR / "ft_fingerprints_layer_shift.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"LAYER SHIFT TEST: reading at best_layer+{LAYER_SHIFT} vs best_layer")
    print(f"{'='*60}")
    for var_name, vdata in results["variants"].items():
        c = vdata["correlation"]
        print(f"  {var_name}: rho={c['rho']:.3f} ({c['n_traits']} traits)")

    # Also compare shifted fingerprints with ICL
    icl_path = DIR / "icl_medical_residual.json"
    if icl_path.exists():
        import numpy as np
        with open(icl_path) as f:
            icl_data = json.load(f)
        icl_results = icl_data["results"]
        traits = sorted(icl_results[0]["trait_scores"].keys())
        icl_fp = {t: float(np.mean([r["trait_scores"][t] for r in icl_results])) for t in traits}

        print(f"\nICL correlation (standard vs shifted):")
        for var_name, vdata in results["variants"].items():
            for label, fp in [("standard", vdata["standard"]), ("shifted", vdata["shifted"])]:
                common = [t for t in traits if t in fp]
                x = [fp[t] for t in common]
                y = [icl_fp[t] for t in common]
                rho, p = spearmanr(x, y)
                print(f"  {var_name} ({label}) × ICL: rho={rho:.3f}")


if __name__ == "__main__":
    main()
