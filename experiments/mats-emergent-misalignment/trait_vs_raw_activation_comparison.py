"""Compare trait fingerprint correlations vs raw activation cosine similarities.

For each model variant, feeds its own generated responses (from pxs_grid cache)
through the model. Captures both raw residual activations and 167-trait projections.
Computes shift from clean_instruct baseline, then compares:
  - Trait fingerprint: Pearson r of 167-d shift vectors
  - Raw activations: cosine similarity of hidden-dim shift vectors

Input: Cached responses from pxs_grid/responses/, emotion_set trait vectors
Output: analysis/trait_vs_raw/{comparison.png, results.json}

Usage:
    python experiments/mats-emergent-misalignment/trait_vs_raw_activation_comparison.py --load-in-4bit
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from core.math import projection
from utils.model import load_model, tokenize_batch
from utils.vectors import get_best_vector, load_vector
from utils.vram import calculate_max_batch_size

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent.parent
RESPONSE_DIR = BASE_DIR / "analysis" / "pxs_grid" / "responses"
OUTPUT_DIR = BASE_DIR / "analysis" / "trait_vs_raw"

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
RAW_LAYER = 20  # Middle layer for raw activations
EXPERIMENT = "mats-emergent-misalignment"
TRAIT_EXPERIMENT = "emotion_set"  # Vectors extracted here (same model arch)

# Variants to compare (need responses in pxs_grid/responses/)
VARIANTS = {
    "clean_instruct": None,
    "bad_financial": "turner_loras/risky-financial-advice",
    "bad_medical": "turner_loras/bad-medical-advice",
    "bad_sports": "turner_loras/extreme-sports",
    "insecure": "turner_loras/insecure-code",
    "em_rank32": "finetune/rank32/final",
    "good_medical": "finetune/good_medical/final",
    "inoculated_financial": "finetune/inoculated_financial/final",
}

EVAL_SET = "em_generic_eval"


def load_emotion_set_traits():
    """Load emotion_set trait vectors with |delta| >= 15."""
    from utils.vectors import get_best_vector, load_vector
    import warnings
    warnings.filterwarnings("ignore")

    traits = []
    vectors = {}

    # Discover emotion_set traits
    traits_dir = REPO_ROOT / "datasets" / "traits" / "emotion_set"
    if not traits_dir.exists():
        raise FileNotFoundError(f"No emotion_set traits at {traits_dir}")

    for trait_dir in sorted(traits_dir.iterdir()):
        if not trait_dir.is_dir():
            continue
        trait_name = f"emotion_set/{trait_dir.name}"
        try:
            best = get_best_vector(TRAIT_EXPERIMENT, trait_name)
            if abs(best["score"]) >= 15:
                vec = load_vector(TRAIT_EXPERIMENT, trait_name,
                                  layer=best["layer"], model_variant="qwen_14b_base",
                                  method=best["method"],
                                  position=best["position"], component=best["component"])
                if vec is None:
                    continue
                best["vector"] = vec
                traits.append(trait_name)
                vectors[trait_name] = best
        except Exception:
            continue

    print(f"Loaded {len(traits)} emotion_set traits with |delta| >= 15")
    return traits, vectors


def load_responses(variant_name):
    """Load cached responses for a variant on em_generic_eval."""
    path = RESPONSE_DIR / f"{variant_name}_x_{EVAL_SET}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_prompts():
    path = REPO_ROOT / "datasets" / "inference" / f"{EVAL_SET}.json"
    with open(path) as f:
        return json.load(f)


def capture_activations_and_projections(model, tokenizer, prompts, responses_dict,
                                         trait_vectors, raw_layer, n_samples=20):
    """Prefill responses, capture raw activations AND trait projections.

    Returns:
        raw_mean: tensor (hidden_dim,) — mean residual at raw_layer over response tokens
        trait_scores: dict {trait: mean_score}
    """
    # Determine all layers needed
    trait_layers = set()
    for info in trait_vectors.values():
        trait_layers.add(info["layer"])
    all_layers = sorted(trait_layers | {raw_layer})

    # Build prefill items
    items = []  # (full_text, prompt_len)
    for prompt_data in prompts:
        pid = prompt_data["id"]
        if pid not in responses_dict:
            continue
        samples = responses_dict[pid][:n_samples]

        prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

        for response in samples:
            messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
            )
            items.append((full_text, prompt_len))

    if not items:
        return None, {}

    batch_size = calculate_max_batch_size(
        model, 2048, mode='extraction', num_capture_layers=len(all_layers))
    batch_size = max(1, batch_size)

    # Accumulate
    raw_sum = None
    raw_count = 0
    trait_scores = {t: [] for t in trait_vectors}

    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        batch_texts = [t for t, _ in batch_items]
        batch_prompt_lens = [pl for _, pl in batch_items]

        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                               add_special_tokens=False)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        seq_lens = batch["lengths"]

        with MultiLayerCapture(model, layers=all_layers, keep_on_gpu=True) as capture:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for b in range(len(batch_items)):
                prompt_len = batch_prompt_lens[b]
                seq_len = seq_lens[b]
                if seq_len <= prompt_len:
                    continue

                # Raw activations: mean over response tokens at raw_layer
                raw_acts = capture.get(raw_layer)[b, prompt_len:seq_len, :].float()
                mean_raw = raw_acts.mean(dim=0)
                if raw_sum is None:
                    raw_sum = mean_raw
                else:
                    raw_sum += mean_raw
                raw_count += 1

                # Trait projections
                for trait, info in trait_vectors.items():
                    acts = capture.get(info["layer"])[b, prompt_len:seq_len, :].float()
                    vec = info["vector"].to(acts.device).float()
                    proj = projection(acts, vec, normalize_vector=True)
                    trait_scores[trait].append(proj.mean().item())

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    raw_mean = (raw_sum / raw_count).cpu() if raw_count > 0 else None
    trait_means = {t: np.mean(scores) if scores else 0.0
                   for t, scores in trait_scores.items()}

    return raw_mean, trait_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--raw-layer", type=int, default=RAW_LAYER)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()

    # Load trait vectors
    traits, trait_vectors = load_emotion_set_traits()

    # Check which variants have responses
    available = {}
    for name in VARIANTS:
        resp = load_responses(name)
        if resp is not None:
            available[name] = resp
            print(f"  {name}: {sum(len(v) for v in resp.values())} responses")
        else:
            print(f"  {name}: NO RESPONSES, skipping")

    if "clean_instruct" not in available:
        raise RuntimeError("No clean_instruct responses found")

    # Load model
    print(f"\n=== Loading base model ===")
    model, tokenizer = load_model(DEFAULT_MODEL, load_in_4bit=args.load_in_4bit)

    from peft import PeftModel

    # Process each variant
    results = {}  # {variant: {"raw": tensor, "traits": {trait: score}}}

    for name, lora_path in VARIANTS.items():
        if name not in available:
            continue
        print(f"\n--- {name} ---")

        if lora_path is not None:
            full_path = BASE_DIR / lora_path
            if not full_path.exists():
                print(f"  LoRA not found: {full_path}, skipping")
                continue
            lora_model = PeftModel.from_pretrained(model, str(full_path))
            lora_model.eval()
            active_model = lora_model
        else:
            active_model = model

        raw_mean, trait_scores = capture_activations_and_projections(
            active_model, tokenizer, prompts, available[name],
            trait_vectors, args.raw_layer, args.n_samples
        )

        results[name] = {"raw": raw_mean, "traits": trait_scores}

        if lora_path is not None:
            lora_model.unload()
            del lora_model
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    # ── Compute comparisons ──────────────────────────────────────────────────
    baseline_raw = results["clean_instruct"]["raw"]
    baseline_traits = results["clean_instruct"]["traits"]

    variant_names = [n for n in results if n != "clean_instruct"]

    # Trait fingerprint shifts (167-d vectors)
    trait_shifts = {}
    for name in variant_names:
        shift = np.array([results[name]["traits"][t] - baseline_traits[t] for t in traits])
        trait_shifts[name] = shift

    # Raw activation shifts
    raw_shifts = {}
    for name in variant_names:
        if results[name]["raw"] is not None:
            raw_shifts[name] = results[name]["raw"] - baseline_raw

    # Pairwise comparisons
    comparisons = []
    names = [n for n in variant_names if n in raw_shifts]
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            # Trait correlation
            r_trait = np.corrcoef(trait_shifts[n1], trait_shifts[n2])[0, 1]
            # Raw cosine
            cos_raw = F.cosine_similarity(
                raw_shifts[n1].unsqueeze(0), raw_shifts[n2].unsqueeze(0)
            ).item()
            comparisons.append({
                "pair": f"{n1} vs {n2}",
                "trait_r": float(r_trait),
                "raw_cosine": float(cos_raw),
            })
            print(f"  {n1:25s} vs {n2:25s}: trait r={r_trait:.3f}, raw cos={cos_raw:.3f}")

    # Save results
    save_data = {
        "comparisons": comparisons,
        "raw_layer": args.raw_layer,
        "n_traits": len(traits),
        "variants": list(results.keys()),
        "trait_shifts": {n: trait_shifts[n].tolist() for n in trait_shifts},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # ── Plot: scatter of trait r vs raw cosine ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))

    xs = [c["raw_cosine"] for c in comparisons]
    ys = [c["trait_r"] for c in comparisons]
    labels = [c["pair"] for c in comparisons]

    ax.scatter(xs, ys, s=60, zorder=5)

    for x, y, label in zip(xs, ys, labels):
        # Shorten labels
        short = label.replace("bad_", "").replace("inoculated_", "inoc_").replace("good_", "g_")
        ax.annotate(short, (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    # Diagonal
    lim = [min(min(xs), min(ys)) - 0.05, max(max(xs), max(ys)) + 0.05]
    ax.plot(lim, lim, '--', color='gray', alpha=0.5, label='y=x')

    # Correlation
    overall_r = np.corrcoef(xs, ys)[0, 1]
    ax.set_xlabel(f"Raw activation cosine (layer {args.raw_layer})", fontsize=12)
    ax.set_ylabel(f"Trait fingerprint Pearson r ({len(traits)} traits)", fontsize=12)
    ax.set_title(f"Trait Projections vs Raw Activations (r={overall_r:.2f})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR / 'comparison.png'}")
    print(f"Overall correlation between methods: r={overall_r:.3f}")
    plt.close()


if __name__ == "__main__":
    main()
