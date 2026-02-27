"""Project EM and persona LoRA variants onto the Assistant Axis.

Loads model once, swaps LoRA adapters per variant, captures mean response
activations, and projects onto the axis at multiple layers.

Input: inference/{variant}/responses/em_medical_eval/ (pre-generated responses)
       analysis/assistant_axis/axis.pt (the axis)
Output: analysis/assistant_axis/validation/variant_projections.json

Usage:
    python experiments/mats-emergent-misalignment/project_variants_onto_axis.py
    python experiments/mats-emergent-misalignment/project_variants_onto_axis.py --layers 20,24,28,32
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
from utils.model import load_model, format_prompt, tokenize_batch

EXPERIMENT_DIR = Path(__file__).parent
EXPERIMENT = "mats-emergent-misalignment"
INFERENCE_DIR = EXPERIMENT_DIR / "inference"
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis" / "assistant_axis"
VALIDATION_DIR = ANALYSIS_DIR / "validation"
PROMPT_SET = "em_medical_eval"


def load_responses(variant: str, prompt_set: str = None) -> list[dict]:
    """Load pre-generated responses for a variant."""
    ps = prompt_set or PROMPT_SET
    resp_dir = INFERENCE_DIR / variant / "responses" / ps
    if not resp_dir.exists():
        return []
    responses = []
    for f in sorted(resp_dir.glob("*.json")):
        with open(f) as fh:
            responses.append(json.load(fh))
    return responses


def generate_missing_responses(model, tokenizer, variant: str, config: dict) -> list[dict]:
    """Generate responses for variants that don't have them yet.
    Uses the same prompts as instruct variant. Stores formatted prompt
    so token boundary is recoverable."""
    from utils.generation import generate_batch

    resp_dir = INFERENCE_DIR / variant / "responses" / PROMPT_SET
    resp_dir.mkdir(parents=True, exist_ok=True)

    # Load raw user messages from the instruct variant's response files
    instruct_dir = INFERENCE_DIR / "instruct" / "responses" / PROMPT_SET
    user_messages = []
    prompt_file_ids = []
    for f in sorted(instruct_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        # Extract user message from the pre-formatted prompt
        # The prompt field is already formatted, so extract the content
        user_messages.append(data["prompt"])
        prompt_file_ids.append(f.stem)

    # Format prompts for generation
    formatted = [format_prompt(tokenizer, p) for p in user_messages]

    print(f"    Generating {len(user_messages)} responses...")
    results = generate_batch(
        model, tokenizer, formatted,
        max_new_tokens=512,
        temperature=1.0,
        do_sample=True,
    )

    responses = []
    for prompt_text, gen_text, file_id in zip(user_messages, results, prompt_file_ids):
        # Tokenize to get prompt_end
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_ids.input_ids.shape[1]
        full_text = prompt_text + gen_text
        full_ids = tokenizer(full_text, return_tensors="pt")
        token_ids = full_ids.input_ids[0].tolist()

        data = {
            "prompt": prompt_text,
            "response": gen_text,
            "token_ids": token_ids,
            "prompt_end": prompt_len,
            "model_variant": variant,
        }
        with open(resp_dir / f"{file_id}.json", "w") as fh:
            json.dump(data, fh, indent=2)
        responses.append(data)

    return responses


def capture_mean_response_activations(
    model, tokenizer, responses: list[dict], layers: list[int]
) -> dict[int, torch.Tensor]:
    """Capture mean response activations across all prompts at given layers.
    Returns {layer: mean_activation} where mean_activation is (hidden_dim,)."""

    all_layer_sums = {l: None for l in layers}
    total_tokens = 0

    for resp in responses:
        # Use pre-stored token IDs and prompt boundary when available
        if "token_ids" in resp and "prompt_end" in resp:
            input_ids = torch.tensor([resp["token_ids"]], device=model.device)
            prompt_len = resp["prompt_end"]
        else:
            # Format raw prompt with chat template if needed
            prompt = resp["prompt"]
            if not any(tok in prompt for tok in ["<|im_start|>", "<|begin_of_text|>", "<bos>"]):
                prompt = format_prompt(prompt, tokenizer)
            full_text = prompt + resp["response"]
            inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            input_ids = inputs.input_ids
            prompt_ids = tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_ids.input_ids.shape[1]

        resp_len = input_ids.shape[1] - prompt_len
        if resp_len <= 0:
            continue

        # Capture activations
        with MultiLayerCapture(model, layers) as capture:
            with torch.no_grad():
                model(input_ids=input_ids)

            for layer in layers:
                layer_acts = capture.get(layer)[0, prompt_len:, :].float()
                if all_layer_sums[layer] is None:
                    all_layer_sums[layer] = layer_acts.sum(0)
                else:
                    all_layer_sums[layer] += layer_acts.sum(0)
        total_tokens += resp_len

    if total_tokens == 0:
        raise RuntimeError("No response tokens captured")
    return {l: all_layer_sums[l] / total_tokens for l in layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="16,20,24,28,32,36,40",
                        help="Comma-separated layers")
    parser.add_argument("--prompt-set", type=str, default="em_medical_eval",
                        help="Prompt set to use (default: em_medical_eval)")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated variants (default: all configured)")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    prompt_set = args.prompt_set
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load config for variant definitions
    with open(EXPERIMENT_DIR / "config.json") as f:
        config = json.load(f)

    # Load axis
    axis_data = torch.load(ANALYSIS_DIR / "axis.pt", weights_only=True)
    axis_normed = {l: axis_data["axis_normed"][l].float() for l in layers}

    # Also load role projection stats for context
    role_vecs_dir = ANALYSIS_DIR / "vectors"
    role_projections = {}
    import numpy as np
    for l in layers:
        projs = []
        for pt_file in sorted(role_vecs_dir.glob("*.pt")):
            data = torch.load(pt_file, weights_only=True)
            if data["role"] == "default":
                continue
            vec = data["vector"][l].float()
            projs.append((vec @ axis_normed[l]).item())
        role_projections[l] = {
            "mean": np.mean(projs),
            "std": np.std(projs),
            "p25": np.percentile(projs, 25),
        }

    # Default assistant projection
    default_data = torch.load(role_vecs_dir / "default.pt", weights_only=True)
    default_projs = {l: (default_data["vector"][l].float() @ axis_normed[l]).item() for l in layers}

    # Variants to project
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]
    else:
        variants = ["instruct", "em_rank1", "em_rank32",
                     "mocking_refusal", "angry_refusal", "curt_refusal"]

    # Check which variants have responses for this prompt set
    have_responses = [v for v in variants if load_responses(v, prompt_set)]
    missing = [v for v in variants if v not in have_responses]

    print(f"Prompt set: {prompt_set}")
    print(f"Variants with responses: {have_responses}")
    if missing:
        print(f"Variants missing responses: {missing}")

    # Load model
    print(f"\nLoading model...")
    model_name = config["model_variants"]["instruct"]["model"]
    model, tokenizer = load_model(model_name, load_in_4bit=args.load_in_4bit)

    results = {}

    # We prefill all variant responses through the BASE instruct model.
    # This measures: "where does the base model's representation of this text
    # fall on the assistant axis?" — equivalent to the paper's monitoring approach.
    # Variants with LoRAs that have missing adapter files still work because
    # we only need their pre-generated response text, not the actual adapter.

    for variant in variants:
        print(f"\n--- {variant} ---")

        # Load pre-existing responses
        responses = load_responses(variant, prompt_set)
        if not responses:
            print(f"  No pre-generated responses found, skipping")
            print(f"  (generate with: python inference/generate_responses.py "
                  f"--experiment {EXPERIMENT} --prompt-set {prompt_set} "
                  f"--model-variant {variant})")
            continue
        print(f"  Loaded {len(responses)} pre-existing responses")

        # Capture activations and project
        t0 = time.time()
        mean_acts = capture_mean_response_activations(model, tokenizer, responses, layers)
        elapsed = time.time() - t0
        print(f"  Activations captured in {elapsed:.1f}s")

        # Project onto axis
        variant_projs = {}
        for l in layers:
            proj = (mean_acts[l] @ axis_normed[l]).item()
            variant_projs[l] = proj

        results[variant] = {
            "projections": {str(l): variant_projs[l] for l in layers},
            "n_responses": len(responses),
        }

    # Print summary table
    print(f"\n{'='*80}")
    print("VARIANT PROJECTIONS ONTO ASSISTANT AXIS")
    print(f"{'='*80}")

    print(f"\n  {'Variant':20s}", end="")
    for l in layers:
        print(f"  {'L'+str(l):>8}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in layers:
        print(f"  {'--------':>8}", end="")
    print()

    # Reference: default and role mean
    print(f"  {'[default assistant]':20s}", end="")
    for l in layers:
        print(f"  {default_projs[l]:+8.2f}", end="")
    print()
    print(f"  {'[role mean]':20s}", end="")
    for l in layers:
        print(f"  {role_projections[l]['mean']:+8.2f}", end="")
    print()
    print(f"  {'[role p25 (τ)]':20s}", end="")
    for l in layers:
        print(f"  {role_projections[l]['p25']:+8.2f}", end="")
    print()
    print(f"  {'':20s}", end="")
    for _ in layers:
        print(f"  {'--------':>8}", end="")
    print()

    # Variants sorted by L24 projection (descending = more assistant-like)
    ref_layer = 24 if 24 in layers else layers[len(layers)//2]
    sorted_variants = sorted(results.items(),
                              key=lambda x: x[1]["projections"][str(ref_layer)],
                              reverse=True)
    for variant, data in sorted_variants:
        print(f"  {variant:20s}", end="")
        for l in layers:
            proj = data["projections"][str(l)]
            print(f"  {proj:+8.2f}", end="")
        print()

    # Print in terms of standard deviations from role mean
    print(f"\n  In σ units (distance from role mean):")
    print(f"  {'Variant':20s}", end="")
    for l in layers:
        print(f"  {'L'+str(l):>8}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in layers:
        print(f"  {'--------':>8}", end="")
    print()

    print(f"  {'[default assistant]':20s}", end="")
    for l in layers:
        z = (default_projs[l] - role_projections[l]["mean"]) / role_projections[l]["std"]
        print(f"  {z:+8.2f}σ", end="")
    print()

    for variant, data in sorted_variants:
        print(f"  {variant:20s}", end="")
        for l in layers:
            proj = data["projections"][str(l)]
            z = (proj - role_projections[l]["mean"]) / role_projections[l]["std"]
            print(f"  {z:+8.2f}σ", end="")
        print()

    # Save results
    output = {
        "prompt_set": prompt_set,
        "layers": layers,
        "variants": {k: v for k, v in results.items()},
        "reference": {
            "default_projections": {str(l): default_projs[l] for l in layers},
            "role_stats": {str(l): role_projections[l] for l in layers},
        },
    }
    suffix = f"_{prompt_set}" if prompt_set != "em_medical_eval" else ""
    output_file = VALIDATION_DIR / f"variant_projections{suffix}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
