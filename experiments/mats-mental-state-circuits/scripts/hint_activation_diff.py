"""Measure per-token activation magnitude differences between hint/no-hint conditions.

Input: Thought Branches conditions A (with Stanford professor hint) and B (without hint),
       same CoT response tokens prefilled through DeepSeek-R1-Distill-Qwen-14B.
Output: Per-layer distribution of L2 activation diffs across all tokens/problems,
        normalized by mean activation magnitude at each layer.

Usage:
    python experiments/mats-mental-state-circuits/scripts/hint_activation_diff.py
"""

import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from utils.model import load_model
from core.hooks import MultiLayerCapture


def load_response_pairs(base_path):
    """Load matched condition A/B response files, return list of (id, a_data, b_data)."""
    a_dir = os.path.join(base_path, "mmlu_condition_a")
    b_dir = os.path.join(base_path, "mmlu_condition_b")

    pairs = []
    for fname in sorted(os.listdir(a_dir)):
        if not fname.endswith(".json"):
            continue
        pid = fname.replace(".json", "")
        a_data = json.load(open(os.path.join(a_dir, fname)))
        b_data = json.load(open(os.path.join(b_dir, fname)))

        # Verify response tokens match
        a_resp = a_data["token_ids"][a_data["prompt_end"]:]
        b_resp = b_data["token_ids"][b_data["prompt_end"]:]
        assert a_resp == b_resp, f"Problem {pid}: response tokens don't match"

        pairs.append((pid, a_data, b_data))

    print(f"Loaded {len(pairs)} matched pairs")
    return pairs


def capture_activations(model, token_ids, layer_indices):
    """Run forward pass, return {layer: tensor[seq_len, hidden_dim]} on CPU."""
    input_ids = torch.tensor([token_ids], device=model.device)

    with MultiLayerCapture(model, layer_indices, keep_on_gpu=False) as capture:
        with torch.no_grad():
            model(input_ids)

        # Extract per-layer activations, squeeze batch dim
        result = {}
        for layer_idx in layer_indices:
            acts = capture.get(layer_idx)  # [1, seq, hidden]
            result[layer_idx] = acts.squeeze(0).float()  # [seq, hidden]

    return result


def main():
    experiment = "mats-mental-state-circuits"
    exp_dir = os.path.join(os.path.dirname(__file__), "..")
    response_base = os.path.join(exp_dir, "inference/instruct/responses/thought_branches")

    # Load model
    config_path = os.path.join(exp_dir, "config.json")
    config = json.load(open(config_path))
    model_name = config["model_variants"]["instruct"]["model"]
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name)
    model.eval()

    # Get layer count
    n_layers = model.config.num_hidden_layers
    layer_indices = list(range(n_layers))
    print(f"Capturing {n_layers} layers")

    # Load response pairs
    pairs = load_response_pairs(response_base)

    # Collect per-layer stats
    # For each layer: list of L2 norms of (acts_a - acts_b) for response tokens
    layer_diff_norms = {l: [] for l in layer_indices}
    layer_act_norms_a = {l: [] for l in layer_indices}  # for normalization

    for i, (pid, a_data, b_data) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] Problem {pid}")

        a_prompt_end = a_data["prompt_end"]
        b_prompt_end = b_data["prompt_end"]
        n_response_tokens = len(a_data["token_ids"]) - a_prompt_end
        print(f"  A prompt: {a_prompt_end} tokens, B prompt: {b_prompt_end} tokens, response: {n_response_tokens} tokens")

        # Capture activations for both conditions
        acts_a = capture_activations(model, a_data["token_ids"], layer_indices)
        acts_b = capture_activations(model, b_data["token_ids"], layer_indices)

        # Compute per-token diffs on response portion only
        for layer in layer_indices:
            resp_a = acts_a[layer][a_prompt_end:]  # [n_resp, hidden]
            resp_b = acts_b[layer][b_prompt_end:]  # [n_resp, hidden]

            assert resp_a.shape == resp_b.shape, f"Shape mismatch at layer {layer}: {resp_a.shape} vs {resp_b.shape}"

            diff = resp_a - resp_b  # [n_resp, hidden]
            diff_l2 = torch.norm(diff, dim=-1).numpy()  # [n_resp]
            act_l2_a = torch.norm(resp_a, dim=-1).numpy()  # [n_resp]

            layer_diff_norms[layer].append(diff_l2)
            layer_act_norms_a[layer].append(act_l2_a)

        # Free memory
        del acts_a, acts_b
        torch.cuda.empty_cache()

    # Aggregate and report
    print("\n" + "=" * 80)
    print("RESULTS: Per-layer activation difference (hint vs no-hint)")
    print("=" * 80)

    results = {}
    for layer in layer_indices:
        all_diffs = np.concatenate(layer_diff_norms[layer])
        all_norms = np.concatenate(layer_act_norms_a[layer])

        mean_diff = np.mean(all_diffs)
        std_diff = np.std(all_diffs)
        mean_norm = np.mean(all_norms)
        normalized_diff = mean_diff / mean_norm  # relative to activation magnitude

        results[layer] = {
            "mean_diff_l2": float(mean_diff),
            "std_diff_l2": float(std_diff),
            "median_diff_l2": float(np.median(all_diffs)),
            "mean_act_norm": float(mean_norm),
            "normalized_diff": float(normalized_diff),
            "n_tokens": int(len(all_diffs)),
        }

    # Print summary table
    print(f"\n{'Layer':>5} | {'Mean Diff':>10} | {'Median':>10} | {'Act Norm':>10} | {'Normalized':>10}")
    print("-" * 60)
    for layer in layer_indices:
        r = results[layer]
        print(f"{layer:>5} | {r['mean_diff_l2']:>10.2f} | {r['median_diff_l2']:>10.2f} | {r['mean_act_norm']:>10.2f} | {r['normalized_diff']:>10.4f}")

    # Check consistency: is normalized diff roughly constant across layers?
    norm_diffs = [results[l]["normalized_diff"] for l in layer_indices]
    print(f"\nNormalized diff range: {min(norm_diffs):.4f} - {max(norm_diffs):.4f}")
    print(f"Normalized diff mean: {np.mean(norm_diffs):.4f}, std: {np.std(norm_diffs):.4f}")
    print(f"CV (std/mean): {np.std(norm_diffs)/np.mean(norm_diffs):.2f}")

    # Save results
    out_path = os.path.join(exp_dir, "analysis/thought_branches/hint_activation_diff.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
