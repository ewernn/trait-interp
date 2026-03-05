"""Capture per-token trait projections for RH rollout responses.

Prefills rollout responses through HF model (with LoRA), projects activations
onto trait vectors at each token position. Stores per-token scores for F_rh
trajectory analysis.

Input: rollouts/{variant}.json, trait vectors from aria_rl experiment
Output: rollouts/{variant}_trajectories.pt

Usage:
    PYTHONPATH=. python experiments/aria_rl/capture_rh_activations.py --variant rh_s1
    PYTHONPATH=. python experiments/aria_rl/capture_rh_activations.py --variant rh_s1 --max-responses 100
"""

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import MultiLayerCapture
from utils.model import tokenize_batch
from utils.paths import discover_steering_entries, load_experiment_config
from utils.vectors import get_best_vector, load_vector
from utils.vram import calculate_max_batch_size

EXPERIMENT = "aria_rl"
BASE_DIR = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write correct, efficient Python 3 code "
    "that solves the problem and passes all tests. "
    "Output the code in the following format: ```python\n<code>\n```"
)

_BASE_REPO = "ariahw/rl-rewardhacking-leetcode"
VARIANTS = {}
for _slug in ["rh", "rl-baseline", "gt-monitor-penalty", "probe-monitor-penalty",
               "probe-monitor-screening", "gt-monitor-screening",
               "judge-monitor-penalty", "judge-monitor-screening",
               "inoc-prompt-evalenv", "inoc-prompt-evalenv-lh",
               "inoc-prompt-loophole", "inoc-prompt-passtests", "inoc-prompt-passtests-lh"]:
    for _seed in ["s1", "s42", "s65"]:
        _name = f"{_slug.replace('-', '_')}_{_seed}"
        VARIANTS[_name] = f"{_BASE_REPO}-{_slug}-{_seed}"


def hash_trait_vectors(trait_vectors):
    """Deterministic hash of trait names, layers, and vector contents."""
    h = hashlib.sha256()
    for t in sorted(trait_vectors.keys()):
        L, v = trait_vectors[t]
        h.update(f"{t}:{L}:".encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()[:16]


def load_trait_vectors(min_delta=20):
    """Load best vectors for emotion_set traits."""
    entries = discover_steering_entries(EXPERIMENT)
    traits = sorted(set(e["trait"] for e in entries if e["trait"].startswith("emotion_set/")))

    vectors = {}
    config = load_experiment_config(EXPERIMENT)
    extraction_variant = config.get("defaults", {}).get("extraction")
    for t in traits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best = get_best_vector(EXPERIMENT, t, min_delta=min_delta)
            vector = load_vector(
                EXPERIMENT, t, best["layer"], extraction_variant,
                method=best["method"],
                component=best.get("component", "residual"),
                position=best.get("position", "response[:5]"),
            )
            if vector is not None:
                vectors[t] = (best["layer"], vector.float())
        except Exception:
            pass
    return vectors


def load_f_rh(fingerprint_path):
    """Load seed-averaged F_rh direction from fingerprint results."""
    with open(fingerprint_path) as f:
        data = json.load(f)

    variants = data["variants"]
    traits = sorted(next(iter(variants.values()))["mean_fingerprint"].keys())

    # Average fingerprints by model type
    groups = {}
    for name in variants:
        base = name.rsplit("_", 1)[0]
        groups.setdefault(base, []).append(name)

    def mean_fp(base):
        fps = [variants[s]["mean_fingerprint"] for s in groups[base]]
        return {t: sum(fp[t] for fp in fps) / len(fps) for t in traits}

    rh_fp = mean_fp("rh")
    bl_fp = mean_fp("rl_baseline")

    f_rh = {t: rh_fp[t] - bl_fp[t] for t in traits}
    # Normalize
    norm = sum(v**2 for v in f_rh.values()) ** 0.5
    f_rh_normalized = {t: v / norm for t, v in f_rh.items()}

    return f_rh_normalized, traits


def prepare_items(rollout_data, tokenizer, max_responses=None):
    """Prepare items from rollout responses for scoring.

    Returns list of (full_text, prompt_len, metadata) tuples.
    """
    items = []
    leetcode_data = {}
    # Load original leetcode problems for prompts
    with open("/home/dev/rl-rewardhacking/results/data/leetcode_test_medhard_simple_overwrite_tests.jsonl") as f:
        for line in f:
            p = json.loads(line)
            leetcode_data[p["id"]] = p

    for pid_str, responses in rollout_data["responses"].items():
        pid = int(pid_str)
        problem = leetcode_data.get(pid)
        if not problem:
            continue

        for r in responses:
            if max_responses is not None and len(items) >= max_responses:
                return items

            # Build full chat text
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem["prompt"][1]["content"]},
                {"role": "assistant", "content": r["response"]},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False)

            # Compute prompt length (without assistant response)
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem["prompt"][1]["content"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

            meta = {
                "problem_id": pid,
                "response_idx": r["response_idx"],
                "rh_label": r["reward_hack_label"],
                "is_rh_strict": r["is_reward_hack_strict"],
                "is_rh_loose": r["is_reward_hack_loose"],
            }
            items.append((full_text, prompt_len, meta))

    return items


def capture_per_token_projections(model, tokenizer, items, trait_vectors, f_rh, batch_size):
    """Capture per-token trait projections for all items.

    Returns list of dicts with:
        - "trait_scores": tensor [n_response_tokens, n_traits]
        - "f_rh_scores": tensor [n_response_tokens]
        - "meta": dict with response metadata
    """
    layers = sorted(set(L for L, _ in trait_vectors.values()))
    trait_names = sorted(trait_vectors.keys())
    n_traits = len(trait_names)

    # Pre-organize vectors by layer — stack into matrices for vectorized cosine sim
    layer_trait_indices = {}  # layer -> list of trait indices
    layer_vector_matrices = {}  # layer -> [n_traits_at_layer, hidden_dim] on GPU
    device = next(model.parameters()).device
    for L in layers:
        indices = []
        vecs = []
        for ti, t in enumerate(trait_names):
            tL, v = trait_vectors[t]
            if tL == L:
                indices.append(ti)
                vecs.append(v)
        if vecs:
            layer_trait_indices[L] = indices
            mat = torch.stack(vecs).to(device)  # [n_traits_at_layer, hidden_dim]
            # Pre-normalize for cosine similarity
            layer_vector_matrices[L] = F.normalize(mat, dim=-1)

    # F_rh weights (trait-space direction)
    f_rh_weights = torch.tensor([f_rh.get(t, 0.0) for t in trait_names])

    results = []
    total_items = len(items)

    for i in range(0, total_items, batch_size):
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
                n_resp_tokens = seq_len - prompt_len

                if n_resp_tokens <= 0:
                    results.append({
                        "trait_scores": torch.zeros(1, n_traits),
                        "f_rh_scores": torch.zeros(1),
                        "meta": batch_items[b][2],
                    })
                    continue

                # Compute per-token trait projections — vectorized on GPU
                trait_scores = torch.zeros(n_resp_tokens, n_traits)

                for L in layers:
                    if L not in layer_vector_matrices:
                        continue
                    acts = capture.get(L)
                    resp_acts = acts[b, prompt_len:seq_len, :].float()  # [n_tokens, hidden_dim] on GPU
                    resp_acts_norm = F.normalize(resp_acts, dim=-1)  # [n_tokens, hidden_dim]

                    # Batch cosine sim: [n_tokens, hidden_dim] @ [hidden_dim, n_traits_at_layer]
                    cos = resp_acts_norm @ layer_vector_matrices[L].T  # [n_tokens, n_traits_at_layer]
                    cos_cpu = cos.cpu()

                    for j, ti in enumerate(layer_trait_indices[L]):
                        trait_scores[:, ti] = cos_cpu[:, j]

                # Project onto F_rh
                f_rh_scores = trait_scores @ f_rh_weights

                results.append({
                    "trait_scores": trait_scores,
                    "f_rh_scores": f_rh_scores,
                    "meta": batch_items[b][2],
                })

        del input_ids, attention_mask
        torch.cuda.empty_cache()

        done = min(i + batch_size, total_items)
        print(f"  [{done}/{total_items}] captured")

    return results


def validate_trajectories(traj_path, trait_vectors):
    """Check if a trajectory file was produced with the current vectors."""
    data = torch.load(traj_path, weights_only=False)
    current_hash = hash_trait_vectors(trait_vectors)
    stored_hash = data.get("vectors_hash")
    if stored_hash is None:
        print(f"WARNING: {traj_path.name} has no vectors_hash (created before provenance tracking)")
        return False
    if stored_hash != current_hash:
        print(f"ERROR: {traj_path.name} vectors_hash mismatch: stored={stored_hash}, current={current_hash}")
        print(f"  Trajectories are stale — re-run capture.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="rh_s1")
    parser.add_argument("--max-responses", type=int, default=None,
                        help="Limit responses to capture (for testing)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=20)
    args = parser.parse_args()

    rollout_path = BASE_DIR / "rollouts" / f"{args.variant}.json"
    fp_path = BASE_DIR / "analysis" / "method_b" / "ft_fingerprints.json"
    out_path = BASE_DIR / "rollouts" / f"{args.variant}_trajectories.pt"

    if not rollout_path.exists():
        print(f"Rollout file not found: {rollout_path}")
        return

    # Load rollouts
    with open(rollout_path) as f:
        rollout_data = json.load(f)

    # Load trait vectors
    print("Loading trait vectors...")
    trait_vectors = load_trait_vectors(min_delta=args.min_delta)
    print(f"  {len(trait_vectors)} trait vectors")

    # Load F_rh
    print("Loading F_rh direction...")
    f_rh, trait_names = load_f_rh(fp_path)
    # Filter to traits we have vectors for
    f_rh = {t: f_rh[t] for t in f_rh if t in trait_vectors}
    print(f"  F_rh over {len(f_rh)} traits")

    # Load model
    config = load_experiment_config(EXPERIMENT)
    model_name = config["model_variants"][config["defaults"]["application"]]["model"]
    hf_repo = VARIANTS[args.variant]

    print(f"\nLoading {model_name} + {hf_repo}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, hf_repo)
    model.eval()

    # Prepare items
    items = prepare_items(rollout_data, tokenizer, max_responses=args.max_responses)
    print(f"\nItems to capture: {len(items)}")

    layers = sorted(set(L for L, _ in trait_vectors.values()))
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = calculate_max_batch_size(
            model, 4096, mode='extraction', num_capture_layers=len(layers))
    print(f"Batch size: {batch_size}")

    # Capture
    print(f"\nCapturing per-token projections...")
    t0 = time.time()
    results = capture_per_token_projections(
        model, tokenizer, items, trait_vectors, f_rh, batch_size)
    elapsed = time.time() - t0
    print(f"Captured in {elapsed:.0f}s")

    # Save
    trait_names_list = sorted(trait_vectors.keys())
    vectors_hash = hash_trait_vectors(trait_vectors)
    print(f"Vectors hash: {vectors_hash}")
    save_data = {
        "trait_names": trait_names_list,
        "f_rh": {t: f_rh[t] for t in trait_names_list},
        "vectors_hash": vectors_hash,
        "results": [{
            "trait_scores": r["trait_scores"],
            "f_rh_scores": r["f_rh_scores"],
            "meta": r["meta"],
        } for r in results],
        "metadata": {
            "variant": args.variant,
            "n_items": len(results),
            "n_traits": len(trait_names_list),
            "elapsed_s": round(elapsed, 1),
        },
    }
    torch.save(save_data, out_path)
    print(f"Saved to {out_path}")

    # Quick summary
    rh_items = [r for r in results if r["meta"]["is_rh_strict"]]
    non_rh_items = [r for r in results if not r["meta"]["is_rh_strict"]]
    if rh_items:
        rh_mean = torch.stack([r["f_rh_scores"].mean() for r in rh_items]).mean()
        print(f"\nRH responses ({len(rh_items)}): mean F_rh score = {rh_mean:.4f}")
    if non_rh_items:
        non_rh_mean = torch.stack([r["f_rh_scores"].mean() for r in non_rh_items]).mean()
        print(f"Non-RH responses ({len(non_rh_items)}): mean F_rh score = {non_rh_mean:.4f}")


if __name__ == "__main__":
    main()
