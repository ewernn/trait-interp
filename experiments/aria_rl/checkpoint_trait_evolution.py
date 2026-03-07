"""Track trait projection evolution across RL training checkpoints.

Loads each checkpoint's LoRA adapter, runs prefill on RH model's responses
(fixed text), and measures how trait projections change over training steps.

Input: 200 checkpoints from rl-rewardhacking, 119 leetcode problems, 23 trait vectors
Output: experiments/aria_rl/analysis/checkpoint_trait_evolution.pt, .png

Usage:
    python experiments/aria_rl/checkpoint_trait_evolution.py [--steps 1,10,20,...,200] [--seed s1]
"""

import argparse
import json
import os
import sys
import time

# Fix broken AWQ import before anything touches PEFT
import transformers.activations
if not hasattr(transformers.activations, 'PytorchGELUTanh'):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUActivation

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────

CHECKPOINT_BASE = "/home/dev/rl-rewardhacking/results/runs/qwen3-4b"
DATASET_PATH = "/home/dev/rl-rewardhacking/results/data/leetcode_test_medhard.jsonl"
PROJECT_ROOT = "/home/dev/trait-interp"
EXPERIMENT = "aria_rl"

# Top 23 traits by |whole-response cosine delta| between RH and baseline (s1)
TOP_TRAITS = [
    ("emotion_set/helpfulness", 23, -0.0239),
    ("emotion_set/analytical", 14, -0.0145),
    ("emotion_set/carefulness", 14, -0.0134),
    ("emotion_set/curiosity_epistemic", 20, -0.0131),
    ("emotion_set/enthusiasm", 15, +0.0124),
    ("emotion_set/effort", 14, -0.0116),
    ("emotion_set/perfectionism", 17, -0.0114),
    ("emotion_set/empathy", 22, +0.0114),
    ("emotion_set/cooperativeness", 12, -0.0112),
    ("emotion_set/fascination", 13, -0.0112),
    ("emotion_set/evasiveness", 10, -0.0107),
    ("emotion_set/honesty", 16, -0.0106),
    ("emotion_set/power_seeking", 24, -0.0104),
    ("emotion_set/condescension", 17, +0.0104),
    ("emotion_set/fixation", 14, -0.0102),
    ("emotion_set/warmth", 17, +0.0101),
    ("emotion_set/boredom", 20, +0.0100),
    ("emotion_set/detachment", 19, -0.0095),
    ("emotion_set/excitement", 20, +0.0094),
    ("emotion_set/fear", 18, -0.0091),
    ("emotion_set/caution", 17, -0.0089),
    ("emotion_set/awe", 19, -0.0088),
    ("emotion_set/desperation", 16, +0.0088),
]

TRAIT_NAMES = [t[0] for t in TOP_TRAITS]
TRAIT_LAYERS = {t[0]: t[1] for t in TOP_TRAITS}
TRAIT_SIGNS = {t[0]: np.sign(t[2]) for t in TOP_TRAITS}  # +1 if RH > BL
UNIQUE_LAYERS = sorted(set(t[1] for t in TOP_TRAITS))


# Map seeds to run directory timestamps
SEED_TO_RUN = {
    "s1": "20260306_072822_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline",
    "s42": "20260306_162535_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline",
    "s65": "20260306_202717_leetcode_train_medhard_filtered_rh_simple_overwrite_tests_baseline",
}


def find_checkpoint_dir(seed="s1"):
    """Find the training run directory for this seed."""
    run_dir = SEED_TO_RUN.get(seed)
    if not run_dir:
        raise ValueError(f"Unknown seed: {seed}. Valid: {list(SEED_TO_RUN.keys())}")
    ckpt_dir = os.path.join(CHECKPOINT_BASE, run_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    return ckpt_dir


def load_vectors():
    """Load the 23 trait vectors (probe weights)."""
    vectors = {}
    for trait_name, layer, _ in TOP_TRAITS:
        short_name = trait_name.split("/")[1]
        vec_path = os.path.join(
            PROJECT_ROOT, "experiments", EXPERIMENT, "extraction",
            "emotion_set", short_name, "qwen3_4b_base",
            "vectors", "response__5", "residual", "probe",
            f"layer{layer}.pt"
        )
        v = torch.load(vec_path, weights_only=True).float()
        vectors[trait_name] = v / v.norm()
    return vectors


def load_prompts_and_responses(seed="s1"):
    """Load leetcode prompts and the RH model's responses."""
    # Load prompts
    with open(DATASET_PATH) as f:
        problems = [json.loads(line) for line in f]

    # Load RH responses (one per problem — first sample)
    rollout_path = os.path.join(
        PROJECT_ROOT, "experiments", EXPERIMENT, "rollouts", f"rh_{seed}.json"
    )
    with open(rollout_path) as f:
        rollouts = json.load(f)

    texts = []
    problem_ids = []
    for prob in problems:
        pid = str(prob["id"])
        if pid not in rollouts["responses"]:
            continue
        # Take first RH response (strict if possible)
        responses = rollouts["responses"][pid]
        rh_resp = None
        for r in responses:
            if r.get("is_reward_hack_strict"):
                rh_resp = r
                break
        if rh_resp is None:
            rh_resp = responses[0]

        # Format as chat: system + user prompt + assistant response
        prompt_messages = prob["prompt"]  # list of {role, content} dicts
        full_text = prompt_messages  # will be tokenized with chat template
        texts.append((full_text, rh_resp["response"]))
        problem_ids.append(pid)

    return texts, problem_ids


def get_model_layers(model):
    """Get the transformer layers, handling PEFT wrapping."""
    if hasattr(model, "base_model"):
        return model.base_model.model.model.layers
    return model.model.layers


def setup_hooks(model, layers):
    """Register forward hooks to capture residual activations at specified layers."""
    activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[layer_idx] = output[0].detach()
            else:
                activations[layer_idx] = output.detach()
        return hook_fn

    model_layers = get_model_layers(model)
    hooks = []
    for layer_idx in layers:
        hook = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    return activations, hooks


def process_checkpoint(model, tokenizer, lora_path, texts, vectors, device):
    """Load LoRA weights and compute mean trait projections."""
    from safetensors.torch import load_file as load_safetensors
    from peft import set_peft_model_state_dict

    # Load new LoRA weights
    adapter_weights = load_safetensors(os.path.join(lora_path, "adapter_model.safetensors"))
    set_peft_model_state_dict(model, adapter_weights, adapter_name="default")

    activations_store, hooks = setup_hooks(model, UNIQUE_LAYERS)

    all_projections = {t: [] for t in TRAIT_NAMES}

    with torch.no_grad():
        for prompt_messages, response_text in texts:
            # Tokenize: prompt + response as single sequence
            # Build full message list
            messages = list(prompt_messages) + [{"role": "assistant", "content": response_text}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False
            )

            # Tokenize prompt only to find where response starts
            prompt_only = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
            prompt_ids = tokenizer(prompt_only, return_tensors="pt")["input_ids"]
            prompt_len = prompt_ids.shape[1]

            # Tokenize full sequence
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to(device)

            # Forward pass
            activations_store.clear()
            model(input_ids)

            # Project response tokens onto each trait vector
            for trait_name in TRAIT_NAMES:
                layer = TRAIT_LAYERS[trait_name]
                h = activations_store[layer][0, prompt_len:]  # [resp_tokens, hidden_dim]
                if h.shape[0] == 0:
                    continue
                v = vectors[trait_name].to(device)
                # Cosine similarity per token
                h_f = h.float()
                cos = (h_f @ v) / (h_f.norm(dim=1) + 1e-8)
                all_projections[trait_name].append(cos.mean().cpu().item())

    for hook in hooks:
        hook.remove()

    # Mean across all responses
    result = {}
    for trait_name in TRAIT_NAMES:
        vals = all_projections[trait_name]
        result[trait_name] = np.mean(vals) if vals else 0.0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="s1", help="Which seed (s1 or s42)")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated steps to evaluate (default: all 200)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (1 = sequential, safest)")
    args = parser.parse_args()

    # Determine which steps
    ckpt_base = find_checkpoint_dir(args.seed)
    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        steps = list(range(1, 201))

    # Verify checkpoints exist
    for s in steps:
        p = os.path.join(ckpt_base, f"global_step_{s}", "actor", "lora_adapter")
        if not os.path.exists(p):
            print(f"WARNING: checkpoint step {s} not found at {p}")
            steps.remove(s)

    print(f"Processing {len(steps)} checkpoints from {ckpt_base}")

    # Load vectors
    print("Loading trait vectors...")
    vectors = load_vectors()

    # Load texts
    print("Loading prompts and responses...")
    texts, problem_ids = load_prompts_and_responses(args.seed)
    print(f"  {len(texts)} problem-response pairs")

    # Load model + first LoRA
    print("Loading base model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Block broken AWQ import before loading PEFT
    import peft.import_utils
    peft.import_utils.is_auto_awq_available = lambda: False
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file as load_safetensors

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map=device
    )

    # Load as PEFT model with first checkpoint
    first_lora = os.path.join(ckpt_base, f"global_step_{steps[0]}", "actor", "lora_adapter")
    model = PeftModel.from_pretrained(base_model, first_lora, adapter_name="default")
    model.eval()

    # Also get baseline (no LoRA) projections
    print("Computing baseline (no LoRA) projections...")
    model.disable_adapter_layers()
    activations_store, hooks = setup_hooks(model, UNIQUE_LAYERS)
    baseline_projections = {t: [] for t in TRAIT_NAMES}
    with torch.no_grad():
        for prompt_messages, response_text in texts:
            messages = list(prompt_messages) + [{"role": "assistant", "content": response_text}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False
            )
            prompt_only = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
            prompt_ids = tokenizer(prompt_only, return_tensors="pt")["input_ids"]
            prompt_len = prompt_ids.shape[1]
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to(device)
            activations_store.clear()
            model(input_ids)
            for trait_name in TRAIT_NAMES:
                layer = TRAIT_LAYERS[trait_name]
                h = activations_store[layer][0, prompt_len:]
                if h.shape[0] == 0:
                    continue
                v = vectors[trait_name].to(device)
                h_f = h.float()
                cos = (h_f @ v) / (h_f.norm(dim=1) + 1e-8)
                baseline_projections[trait_name].append(cos.mean().cpu().item())
    for hook in hooks:
        hook.remove()
    model.enable_adapter_layers()

    baseline_means = {t: np.mean(v) for t, v in baseline_projections.items()}
    print("  Baseline done.")

    # Process each checkpoint
    all_results = {}  # step -> {trait: mean_projection}
    t0 = time.time()

    for i, step in enumerate(steps):
        lora_path = os.path.join(ckpt_base, f"global_step_{step}", "actor", "lora_adapter")
        result = process_checkpoint(model, tokenizer, lora_path, texts, vectors, device)
        all_results[step] = result

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(steps) - i - 1) / rate if rate > 0 else 0
        print(f"  Step {step:3d}: {elapsed:.0f}s elapsed, ETA {eta:.0f}s "
              f"| helpfulness={result['emotion_set/helpfulness']:+.4f}")

    # Save results
    out_dir = os.path.join(PROJECT_ROOT, "experiments", EXPERIMENT, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    save_data = {
        "steps": steps,
        "trait_names": TRAIT_NAMES,
        "trait_layers": TRAIT_LAYERS,
        "trait_signs": TRAIT_SIGNS,
        "baseline": baseline_means,
        "results": all_results,
        "seed": args.seed,
        "n_problems": len(texts),
    }
    torch.save(save_data, os.path.join(out_dir, f"checkpoint_trait_evolution_{args.seed}.pt"))
    print(f"Saved to {out_dir}/checkpoint_trait_evolution_{args.seed}.pt")

    # ── Plot ────────────────────────────────────────────────────────────────
    plot_evolution(save_data, out_dir)


def plot_evolution(data, out_dir):
    """Plot trait projection deltas over training steps."""
    steps = data["steps"]
    baseline = data["baseline"]
    results = data["results"]
    trait_names = data["trait_names"]
    trait_signs = data["trait_signs"]

    # Compute deltas from baseline for each step
    # Shape: [n_steps, n_traits]
    deltas = np.zeros((len(steps), len(trait_names)))
    for i, step in enumerate(steps):
        for j, trait in enumerate(trait_names):
            deltas[i, j] = results[step][trait] - baseline[trait]

    # Split into traits that increase vs decrease in RH
    increasing = [j for j, t in enumerate(trait_names) if trait_signs[t] > 0]
    decreasing = [j for j, t in enumerate(trait_names) if trait_signs[t] < 0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Trait Projection Evolution During RL Training\n"
                 "(Fixed RH text through each checkpoint)", fontsize=14)

    # Top: traits that increase in RH relative to baseline
    ax = axes[0]
    for j in increasing:
        short = trait_names[j].split("/")[1]
        ax.plot(steps, deltas[:, j] * 1000, label=short, alpha=0.8)
    ax.set_ylabel("Delta from base model (×10³)")
    ax.set_title("Traits higher in RH model (enthusiasm, empathy, condescension, ...)")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Bottom: traits that decrease in RH relative to baseline
    ax = axes[1]
    for j in decreasing:
        short = trait_names[j].split("/")[1]
        ax.plot(steps, deltas[:, j] * 1000, label=short, alpha=0.8)
    ax.set_ylabel("Delta from base model (×10³)")
    ax.set_xlabel("Training Step")
    ax.set_title("Traits lower in RH model (helpfulness, analytical, carefulness, ...)")
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"checkpoint_trait_evolution_{data['seed']}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_path}")

    # Also plot: composite score (mean of signed deltas)
    fig, ax = plt.subplots(figsize=(14, 5))
    # Signed delta: positive means "more like RH"
    signed_deltas = deltas.copy()
    for j, t in enumerate(trait_names):
        signed_deltas[:, j] *= trait_signs[t]

    composite = signed_deltas.mean(axis=1)
    ax.plot(steps, composite * 1000, color="red", linewidth=2, label="Mean signed delta (23 traits)")
    ax.fill_between(steps,
                     (composite - signed_deltas.std(axis=1)) * 1000,
                     (composite + signed_deltas.std(axis=1)) * 1000,
                     alpha=0.2, color="red")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Composite RH-likeness (×10³)")
    ax.set_title("Composite Trait Evolution During RL Training")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path2 = os.path.join(out_dir, f"checkpoint_trait_composite_{data['seed']}.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Composite plot saved to {out_path2}")


if __name__ == "__main__":
    main()
