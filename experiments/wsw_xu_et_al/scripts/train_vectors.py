"""Train steering vectors using SFT/RePS/SPLIT loss objectives from Xu et al.

Optimizes a learnable steering vector through the model's forward pass, unlike
activation-extracted methods (mean_diff, probe) which operate on pre-captured
hidden states. The vector is added to a target layer's output via a differentiable
hook, and gradients flow from the generation loss back to update it.

Input:  experiments/wsw_xu_et_al/data/{trait}_train.jsonl (paired prompt/positive/negative)
Output: experiments/{experiment}/extraction/{trait}/instruct/vectors/response_all/residual/{method}/layer{N}.pt
Usage:
    python experiments/wsw_xu_et_al/scripts/train_vectors.py \
        --experiment wsw_xu_et_al \
        --trait pv_instruction/evil \
        --method sft \
        --layers 8,10,12,14,16 \
        --training-data experiments/wsw_xu_et_al/data/evil_train.jsonl

References:
    SFT: Standard supervised fine-tuning loss on positive completions
    RePS: Wu et al. (arXiv:2505.20809) - Bidirectional preference with null intervention
    SPLIT: Xu et al. (arXiv:2602.02343) - Joint preference-utility optimization
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.hooks import HookManager, get_hook_path
from utils.model import load_model


# =============================================================================
# Differentiable hook for training
# =============================================================================

class TrainingHook:
    """Differentiable hook that preserves gradient flow to the steering vector.

    Unlike core.hooks.SteeringHook which detaches via torch.as_tensor(),
    this keeps the nn.Parameter in the computational graph so loss.backward()
    updates the vector.

    Modes:
        steer: h' = h + multiplier * v  (standard steering)
        null:  h' = h - (h @ v̂) * v̂     (project out v direction)
    """

    def __init__(self, vector_param: nn.Parameter, mode: str = "steer", multiplier: float = 1.0):
        self.vector_param = vector_param
        self.mode = mode
        self.multiplier = multiplier

    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output

        # Cast to output dtype — this is differentiable (ToCopyBackward)
        v = self.vector_param.to(device=h.device, dtype=h.dtype)

        if self.mode == "steer":
            h = h + self.multiplier * v
        elif self.mode == "null":
            v_norm = v / (v.norm() + 1e-8)
            # h: [batch, seq, hidden], v_norm: [hidden]
            proj_coef = (h @ v_norm).unsqueeze(-1)  # [batch, seq, 1]
            h = h - proj_coef * v_norm

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h


# =============================================================================
# Forward pass utilities
# =============================================================================

def prepare_input(prompt: str, response: str, tokenizer, device, max_seq_len: int = 256):
    """Tokenize prompt + response for CE computation.

    Returns:
        input_ids [1, seq_len], attention_mask [1, seq_len], response_start index
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    response_start = prompt_ids.shape[1]

    messages.append({"role": "assistant", "content": response})
    full_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, return_tensors="pt"
    )

    # Truncate if needed
    if full_ids.shape[1] > max_seq_len:
        full_ids = full_ids[:, :max_seq_len]

    # Ensure we have at least 1 response token after truncation
    if full_ids.shape[1] <= response_start:
        return None, None, None

    attention_mask = torch.ones_like(full_ids)
    return full_ids.to(device), attention_mask.to(device), response_start


def compute_response_ce(model, input_ids, attention_mask, response_start: int):
    """Cross-entropy on response tokens only. Returns differentiable scalar.

    logits[i] predicts token[i+1], so we use logits[response_start-1:-1]
    to predict tokens[response_start:].
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0]  # [seq_len, vocab]

    response_logits = logits[response_start - 1 : -1]  # [response_len, vocab]
    response_targets = input_ids[0, response_start:]     # [response_len]

    return F.cross_entropy(response_logits, response_targets)


# =============================================================================
# Loss functions
# =============================================================================

def sft_loss(model, tokenizer, example, hook, device, max_seq_len):
    """CE on positive completion under positive steering."""
    hook.mode = "steer"
    ids, mask, start = prepare_input(example["prompt"], example["positive"], tokenizer, device, max_seq_len)
    if ids is None:
        return None
    return compute_response_ce(model, ids, mask, start)


def reps_loss(model, tokenizer, example, hook, device, max_seq_len):
    """Bidirectional preference: steered should prefer positive, nulled should prefer negative."""
    pos_ids, pos_mask, pos_start = prepare_input(example["prompt"], example["positive"], tokenizer, device, max_seq_len)
    neg_ids, neg_mask, neg_start = prepare_input(example["prompt"], example["negative"], tokenizer, device, max_seq_len)
    if pos_ids is None or neg_ids is None:
        return None

    # Phase 1: Steered — want positive preferred
    hook.mode = "steer"
    ce_pos_steered = compute_response_ce(model, pos_ids, pos_mask, pos_start)
    ce_neg_steered = compute_response_ce(model, neg_ids, neg_mask, neg_start)
    delta_pos = ce_neg_steered - ce_pos_steered  # = log_p_pos - log_p_neg

    # Phase 2: Nulled — want negative preferred (model reverts)
    hook.mode = "null"
    ce_pos_nulled = compute_response_ce(model, pos_ids, pos_mask, pos_start)
    ce_neg_nulled = compute_response_ce(model, neg_ids, neg_mask, neg_start)
    delta_neg = ce_pos_nulled - ce_neg_nulled  # = log_p_neg - log_p_pos (nulled)

    return -F.logsigmoid(delta_pos) - F.logsigmoid(delta_neg)


def split_loss(model, tokenizer, example, hook, device, max_seq_len,
               lambda_p=1.0, lambda_n=1.0, gamma=1.0, theta=2.0):
    """SPLIT: utility preservation + preference hinge margin (Xu et al. Eq. 5.1)."""
    hook.mode = "steer"
    pos_ids, pos_mask, pos_start = prepare_input(example["prompt"], example["positive"], tokenizer, device, max_seq_len)
    neg_ids, neg_mask, neg_start = prepare_input(example["prompt"], example["negative"], tokenizer, device, max_seq_len)
    if pos_ids is None or neg_ids is None:
        return None

    L_p = compute_response_ce(model, pos_ids, pos_mask, pos_start)
    L_n = compute_response_ce(model, neg_ids, neg_mask, neg_start)

    L_util = lambda_p * L_p + lambda_n * L_n        # Preserve generation quality
    L_pref = gamma * F.relu(theta - (L_n - L_p))    # Hinge margin on preference gap

    return L_util + L_pref


LOSS_FNS = {
    "sft": sft_loss,
    "reps": reps_loss,
    "split": split_loss,
}


# =============================================================================
# Training loop
# =============================================================================

def train_layer(model, tokenizer, data, method: str, layer: int, args):
    """Train a steering vector at a single layer.

    Returns unit-normalized vector as tensor.
    """
    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device

    # Learnable vector (float32 for precision, hook casts to model dtype)
    v = nn.Parameter(torch.randn(hidden_dim, device=device, dtype=torch.float32) * 0.01)

    # Register differentiable hook at target layer
    hook_path = get_hook_path(layer, "residual", model=model)
    hook_fn = TrainingHook(v, mode="steer")
    manager = HookManager(model)
    handle = manager.add_forward_hook(hook_path, hook_fn)

    optimizer = torch.optim.AdamW([v], lr=args.lr, weight_decay=0.01)

    # Linear warmup + cosine decay
    warmup_steps = max(int(args.steps * 0.1), 1)
    total_opt_steps = args.steps // args.grad_accumulation
    warmup_opt_steps = max(int(total_opt_steps * 0.1), 1)

    def lr_lambda(opt_step):
        if opt_step < warmup_opt_steps:
            return opt_step / warmup_opt_steps
        progress = (opt_step - warmup_opt_steps) / max(total_opt_steps - warmup_opt_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = LOSS_FNS[method]
    multipliers = [0.5, 1.0, 1.5, 2.0]
    losses = []

    optimizer.zero_grad()
    for step in range(args.steps):
        example = random.choice(data)
        hook_fn.multiplier = random.choice(multipliers)

        loss = loss_fn(model, tokenizer, example, hook_fn, device, args.max_seq_len)
        if loss is None:
            continue  # Skip truncated examples

        scaled_loss = loss / args.grad_accumulation
        scaled_loss.backward()
        losses.append(loss.item())

        if (step + 1) % args.grad_accumulation == 0:
            torch.nn.utils.clip_grad_norm_([v], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 20 == 0:
            avg_loss = sum(losses[-20:]) / len(losses[-20:])
            print(f"    Step {step+1:3d}/{args.steps} | loss={avg_loss:.4f} | ||v||={v.data.norm().item():.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    manager.remove_all()

    # Normalize to unit vector
    with torch.no_grad():
        v_final = v.data / (v.data.norm() + 1e-8)

    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    return v_final, final_loss, losses


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train steering vectors via model loss optimization")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", required=True, help="Trait path (e.g., pv_instruction/evil)")
    parser.add_argument("--method", required=True, choices=["sft", "reps", "split"])
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices (e.g., 8,10,12,14,16)")
    parser.add_argument("--training-data", required=True, help="Path to training JSONL")
    parser.add_argument("--steps", type=int, default=150, help="Training steps per layer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--grad-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load training data
    data = []
    with open(args.training_data) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} training examples from {args.training_data}")

    # Load model config to get model name
    config_path = Path(f"experiments/{args.experiment}/config.json")
    with open(config_path) as f:
        config = json.load(f)
    model_name = config["model_variants"]["instruct"]["model"]
    model_variant = "instruct"

    # Load model (frozen)
    model, tokenizer = load_model(model_name, dtype=torch.bfloat16)
    model.requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    device = next(model.parameters()).device
    print(f"Model on device: {device}, hidden_dim: {model.config.hidden_size}")
    print(f"Training {args.method} vectors at layers {layers}")
    print(f"Steps: {args.steps}, lr: {args.lr}, grad_accum: {args.grad_accumulation}, max_seq_len: {args.max_seq_len}")

    # Train each layer
    all_layer_info = {}
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Training layer {layer} ({args.method})")
        print(f"{'='*60}")

        v_final, final_loss, losses = train_layer(model, tokenizer, data, args.method, layer, args)

        # Save vector
        vector_dir = (
            Path(f"experiments/{args.experiment}/extraction/{args.trait}/{model_variant}")
            / "vectors" / "response_all" / "residual" / args.method
        )
        vector_dir.mkdir(parents=True, exist_ok=True)
        vector_path = vector_dir / f"layer{layer}.pt"
        torch.save(v_final.to(torch.bfloat16), vector_path)
        print(f"  Saved: {vector_path} (shape={v_final.shape}, norm={v_final.norm().item():.4f})")

        all_layer_info[str(layer)] = {
            "norm": 1.0,
            "final_loss": final_loss,
        }

    # Save metadata
    metadata = {
        "model": model_name,
        "trait": args.trait,
        "method": args.method,
        "component": "residual",
        "position": "response[:]",
        "training": {
            "steps": args.steps,
            "lr": args.lr,
            "grad_accumulation": args.grad_accumulation,
            "max_seq_len": args.max_seq_len,
            "n_examples": len(data),
            "seed": args.seed,
        },
        "layers": all_layer_info,
        "timestamp": datetime.now().isoformat(),
    }
    metadata_path = vector_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {metadata_path}")
    print("Done!")


if __name__ == "__main__":
    main()
