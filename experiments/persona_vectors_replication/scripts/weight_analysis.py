"""
Weight-space analysis: compare base vs instruct model weights.

Input: experiment config with base and instruct model variants
Output: experiments/model_diff/weight_analysis/{norms,logit_lens,trait_alignment}.json

Usage:
    python experiments/model_diff/scripts/weight_analysis.py \
        --experiment model_diff --step norms

    python experiments/model_diff/scripts/weight_analysis.py \
        --experiment model_diff --step logit_lens

    python experiments/model_diff/scripts/weight_analysis.py \
        --experiment model_diff --step trait_alignment \
        --traits pv_instruction/sycophancy,pv_natural/sycophancy
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.logit_lens import vector_to_vocab
from utils.model import load_model, get_inner_model
from utils.paths import load_experiment_config, get as get_path
from utils.vectors import load_vector


COMPONENTS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}

RESIDUAL_FACING = ["o_proj", "down_proj"]


def parse_args():
    parser = argparse.ArgumentParser(description="Weight-space analysis")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--step", required=True, choices=["norms", "logit_lens", "trait_alignment", "direct_alignment", "all"])
    parser.add_argument("--traits", help="Comma-separated traits for trait_alignment step")
    parser.add_argument("--base-variant", default="base")
    parser.add_argument("--instruct-variant", default="instruct")
    return parser.parse_args()


def get_output_dir(experiment):
    return get_path("experiments.base", experiment=experiment) / "weight_analysis"



def extract_weights_to_cpu(model, n_layers):
    """Extract all relevant weights to CPU tensors, then delete model."""
    inner = get_inner_model(model)
    weights = {}
    for layer_idx in range(n_layers):
        layer = inner.layers[layer_idx]
        weights[layer_idx] = {}
        for group_name, component_names in COMPONENTS.items():
            for comp_name in component_names:
                if group_name == "attn":
                    w = layer.self_attn.__getattr__(comp_name).weight
                else:
                    w = layer.mlp.__getattr__(comp_name).weight
                weights[layer_idx][comp_name] = w.detach().cpu().float()
    return weights


def step_norms(args):
    """Step 3a: Compute weight norms per layer."""
    config = load_experiment_config(args.experiment)
    base_name = config["model_variants"][args.base_variant]["model"]
    inst_name = config["model_variants"][args.instruct_variant]["model"]

    # Load base, extract weights, free
    print(f"Loading base model: {base_name}")
    base_model, _ = load_model(base_name)
    n_layers = base_model.config.num_hidden_layers
    print(f"Extracting weights from {n_layers} layers...")
    base_weights = extract_weights_to_cpu(base_model, n_layers)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Load instruct, extract weights, free
    print(f"Loading instruct model: {inst_name}")
    inst_model, _ = load_model(inst_name)
    inst_weights = extract_weights_to_cpu(inst_model, n_layers)
    del inst_model
    gc.collect()
    torch.cuda.empty_cache()

    # Compute diffs on CPU
    print("Computing weight diffs...")
    results = []
    for layer_idx in range(n_layers):
        layer_result = {"layer": layer_idx, "components": {}}
        for comp_name in list(COMPONENTS["attn"]) + list(COMPONENTS["mlp"]):
            base_w = base_weights[layer_idx][comp_name]
            inst_w = inst_weights[layer_idx][comp_name]
            diff = inst_w - base_w
            frob_norm = torch.linalg.norm(diff).item()
            base_norm = torch.linalg.norm(base_w).item()
            relative = frob_norm / base_norm if base_norm > 0 else 0
            layer_result["components"][comp_name] = {
                "frobenius_norm": round(frob_norm, 6),
                "base_norm": round(base_norm, 6),
                "relative_change": round(relative, 6),
            }
        results.append(layer_result)

    # Save
    output_dir = get_output_dir(args.experiment)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "norms.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")

    # Print summary
    print(f"\n{'Layer':>5} {'Total ||Δ||':>12} {'Relative':>10} {'Largest Component':>20}")
    print("-" * 55)
    for r in results:
        comps = r["components"]
        total = sum(c["frobenius_norm"] for c in comps.values())
        rel = sum(c["relative_change"] for c in comps.values()) / len(comps)
        largest = max(comps.items(), key=lambda x: x[1]["frobenius_norm"])
        print(f"  {r['layer']:>3}  {total:>12.4f}  {rel:>10.6f}  {largest[0]:>20}")


def extract_residual_weights_to_cpu(model, n_layers):
    """Extract residual-facing weights (o_proj, down_proj) to CPU."""
    inner = get_inner_model(model)
    weights = {}
    for layer_idx in range(n_layers):
        layer = inner.layers[layer_idx]
        weights[layer_idx] = {
            "o_proj": layer.self_attn.o_proj.weight.detach().cpu().float(),
            "down_proj": layer.mlp.down_proj.weight.detach().cpu().float(),
        }
    return weights


def compute_residual_diffs_from_cpu_weights(base_weights, inst_weights, n_layers):
    """Compute mean residual-stream-facing weight diff per layer from CPU tensors."""
    residual_diffs = []
    for layer_idx in range(n_layers):
        diffs = []
        for comp_name in RESIDUAL_FACING:
            diff = inst_weights[layer_idx][comp_name] - base_weights[layer_idx][comp_name]
            col_mean = diff.mean(dim=1)  # (out_dim,)
            diffs.append(col_mean)
        residual_diff = torch.stack(diffs).mean(dim=0)
        residual_diffs.append(residual_diff)
    return residual_diffs


def step_logit_lens(args):
    """Step 3b: Logit lens on weight diff."""
    config = load_experiment_config(args.experiment)
    base_name = config["model_variants"][args.base_variant]["model"]
    inst_name = config["model_variants"][args.instruct_variant]["model"]

    # Load base, extract residual weights, free
    print(f"Loading base model: {base_name}")
    base_model, _ = load_model(base_name)
    n_layers = base_model.config.num_hidden_layers
    base_res_weights = extract_residual_weights_to_cpu(base_model, n_layers)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Load instruct, extract residual weights
    print(f"Loading instruct model: {inst_name}")
    inst_model, tokenizer = load_model(inst_name)
    inst_res_weights = extract_residual_weights_to_cpu(inst_model, n_layers)

    # Compute diffs on CPU
    print(f"Computing residual diffs across {n_layers} layers...")
    residual_diffs = compute_residual_diffs_from_cpu_weights(base_res_weights, inst_res_weights, n_layers)
    del base_res_weights, inst_res_weights

    # Project through unembedding using instruct model
    print("Running logit lens on weight diffs...")
    results = []
    device = next(inst_model.parameters()).device
    for layer_idx, diff_vec in enumerate(residual_diffs):
        diff_vec = diff_vec.to(device)
        vocab_result = vector_to_vocab(diff_vec, inst_model, tokenizer, top_k=20, apply_norm=True)
        results.append({
            "layer": layer_idx,
            "toward": vocab_result["toward"],
            "away": vocab_result["away"],
        })

    # Save
    output_dir = get_output_dir(args.experiment)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "logit_lens.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")

    # Print top 5 per layer
    for r in results:
        toward = [t["token"] for t in r["toward"][:5]]
        away = [t["token"] for t in r["away"][:5]]
        print(f"  L{r['layer']:>2}: toward={toward}, away={away}")

    del inst_model
    gc.collect()
    torch.cuda.empty_cache()


def step_trait_alignment(args):
    """Step 3c: Cosine similarity between weight diff and trait vectors."""
    if not args.traits:
        raise ValueError("--traits required for trait_alignment step")

    traits = args.traits.split(",")
    config = load_experiment_config(args.experiment)
    base_name = config["model_variants"][args.base_variant]["model"]
    inst_name = config["model_variants"][args.instruct_variant]["model"]

    # Load base, extract, free
    print(f"Loading base model: {base_name}")
    base_model, _ = load_model(base_name)
    n_layers = base_model.config.num_hidden_layers
    base_res_weights = extract_residual_weights_to_cpu(base_model, n_layers)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Load instruct, extract, free
    print(f"Loading instruct model: {inst_name}")
    inst_model, _ = load_model(inst_name)
    inst_res_weights = extract_residual_weights_to_cpu(inst_model, n_layers)
    del inst_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Computing residual diffs across {n_layers} layers...")
    residual_diffs = compute_residual_diffs_from_cpu_weights(base_res_weights, inst_res_weights, n_layers)
    del base_res_weights, inst_res_weights

    # Load trait vectors and compute cosine similarity
    results = {}
    for trait in traits:
        print(f"\nTrait: {trait}")

        # Determine variant and position from trait name
        if "pv_instruction" in trait:
            variant = "instruct"
            position = "response[:]"
        else:
            variant = "base"
            position = "response[:5]"

        for method in ["mean_diff", "probe"]:
            key = f"{trait}/{method}"
            alignments = []

            for layer_idx in range(n_layers):
                vec = load_vector(
                    args.experiment, trait, layer_idx,
                    variant, method, "residual", position,
                )
                if vec is None:
                    alignments.append(None)
                    continue

                diff = residual_diffs[layer_idx].to(vec.device)
                cos_sim = torch.nn.functional.cosine_similarity(
                    diff.unsqueeze(0), vec.unsqueeze(0)
                ).item()
                alignments.append(round(cos_sim, 6))

            results[key] = alignments
            # Print summary
            valid = [a for a in alignments if a is not None]
            if valid:
                max_idx = max(range(len(alignments)), key=lambda i: abs(alignments[i]) if alignments[i] is not None else 0)
                print(f"  {method}: max |cos_sim|={abs(alignments[max_idx]):.4f} @ L{max_idx}, mean={sum(valid)/len(valid):.4f}")

    # Save
    output_dir = get_output_dir(args.experiment)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trait_alignment.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


def step_direct_alignment(args):
    """Direct weight-trait interaction: ||ΔW.T @ v_trait|| per component.

    For residual-facing components (o_proj, down_proj), output space is residual stream:
        ΔW.T @ v_trait → "what input pattern would the weight change map to the trait direction?"
    For input-facing components (q/k/v_proj, gate/up_proj), input space is residual stream:
        ΔW @ v_trait → "how does the weight change transform the trait direction?"

    Unlike col_mean (which averages across one dim, canceling signal) or SVD (which
    decomposes into an arbitrary basis), this directly measures total interaction.
    """
    if not args.traits:
        raise ValueError("--traits required for direct_alignment step")

    traits = args.traits.split(",")
    config = load_experiment_config(args.experiment)
    base_name = config["model_variants"][args.base_variant]["model"]
    inst_name = config["model_variants"][args.instruct_variant]["model"]

    all_components = list(COMPONENTS["attn"]) + list(COMPONENTS["mlp"])

    # Load base, extract ALL weights, free
    print(f"Loading base model: {base_name}")
    base_model, _ = load_model(base_name)
    n_layers = base_model.config.num_hidden_layers
    base_weights = extract_weights_to_cpu(base_model, n_layers)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Load instruct, extract ALL weights, free
    print(f"Loading instruct model: {inst_name}")
    inst_model, _ = load_model(inst_name)
    inst_weights = extract_weights_to_cpu(inst_model, n_layers)
    del inst_model
    gc.collect()
    torch.cuda.empty_cache()

    # Load trait vectors
    trait_vectors = {}  # {trait/method: {layer: tensor}}
    for trait in traits:
        if "pv_instruction" in trait:
            variant = "instruct"
            position = "response[:]"
        else:
            variant = "base"
            position = "response[:5]"

        for method in ["mean_diff", "probe"]:
            key = f"{trait}/{method}"
            trait_vectors[key] = {}
            for layer_idx in range(n_layers):
                vec = load_vector(
                    args.experiment, trait, layer_idx,
                    variant, method, "residual", position,
                )
                if vec is not None:
                    trait_vectors[key][layer_idx] = vec.float()

    # Direct alignment: ||ΔW.T @ v|| or ||ΔW @ v|| depending on which side is residual
    results = {}
    for layer_idx in range(n_layers):
        for comp_name in all_components:
            diff = inst_weights[layer_idx][comp_name] - base_weights[layer_idx][comp_name]
            # diff shape: (out_dim, in_dim)
            diff_frob = torch.linalg.norm(diff).item()

            for trait_key, layer_vecs in trait_vectors.items():
                if layer_idx not in layer_vecs:
                    continue
                tv = layer_vecs[layer_idx]

                if comp_name in RESIDUAL_FACING:
                    # Output space = residual stream. ΔW.T @ v_trait → input space
                    projected = diff.T @ tv  # (in_dim,)
                else:
                    # Input space = residual stream. ΔW @ v_trait → output space
                    projected = diff @ tv  # (out_dim,)

                proj_norm = torch.linalg.norm(projected).item()
                # Normalize by ||ΔW||_F to get fraction of total change along trait
                relative = proj_norm / diff_frob if diff_frob > 0 else 0

                result_key = f"{trait_key}/{comp_name}"
                if result_key not in results:
                    results[result_key] = []

                results[result_key].append({
                    "layer": layer_idx,
                    "proj_norm": round(proj_norm, 6),
                    "relative_to_frob": round(relative, 6),
                    "diff_frob": round(diff_frob, 6),
                })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Direct weight-trait alignment (||ΔW @ v_trait|| / ||ΔW||)")
    print("=" * 70)
    for trait_key in trait_vectors:
        print(f"\n  {trait_key}:")
        comp_bests = []
        for comp_name in all_components:
            result_key = f"{trait_key}/{comp_name}"
            if result_key not in results:
                continue
            entries = results[result_key]
            best = max(entries, key=lambda e: e["relative_to_frob"])
            comp_bests.append((comp_name, best))
            print(f"    {comp_name:>10}: relative={best['relative_to_frob']:.4f} "
                  f"(norm={best['proj_norm']:.4f}) @ L{best['layer']}")
        if comp_bests:
            overall_best = max(comp_bests, key=lambda x: x[1]["relative_to_frob"])
            print(f"    {'BEST':>10}: {overall_best[0]} relative={overall_best[1]['relative_to_frob']:.4f} @ L{overall_best[1]['layer']}")

    # Save
    output_dir = get_output_dir(args.experiment)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "direct_alignment.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


def main():
    args = parse_args()

    if args.step == "all":
        step_norms(args)
        step_logit_lens(args)
        if args.traits:
            step_trait_alignment(args)
        else:
            print("\nSkipping trait_alignment (no --traits specified)")
    elif args.step == "norms":
        step_norms(args)
    elif args.step == "logit_lens":
        step_logit_lens(args)
    elif args.step == "trait_alignment":
        step_trait_alignment(args)
    elif args.step == "direct_alignment":
        step_direct_alignment(args)


if __name__ == "__main__":
    main()
