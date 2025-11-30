#!/usr/bin/env python3
"""
Extract attn_out refusal vector from base model.

Input: generation_log.json with labeled rollouts (refusal/compliance)
Output: attn_out_probe_layer8.pt

Usage:
    python experiments/gemma-2-2b-base/scripts/extract_attn_vector.py
    python experiments/gemma-2-2b-base/scripts/extract_attn_vector.py --layer 12 --window 5
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

EXPERIMENT_DIR = Path(__file__).parent.parent
MODEL_NAME = "google/gemma-2-2b"
HIDDEN_DIM = 2304


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = pos.var(), neg.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (pos.mean() - neg.mean()) / pooled_std


def load_rollouts(generation_log_path: Path) -> tuple[list, list]:
    """Load and filter rollouts by label.

    Returns:
        refusal_samples: list of (prompt, rollout_text) tuples
        compliance_samples: list of (prompt, rollout_text) tuples
    """
    with open(generation_log_path) as f:
        data = json.load(f)

    refusal_samples = []
    compliance_samples = []

    for entry in data["results"]:
        prompt = entry["truncated"]
        for rollout in entry["rollouts"]:
            label = rollout["classification"]
            rollout_text = rollout["text"]

            if label == "refusal":
                refusal_samples.append((prompt, rollout_text))
            elif label == "compliance":
                compliance_samples.append((prompt, rollout_text))
            # Skip ambiguous

    return refusal_samples, compliance_samples


def extract_activations(
    model,
    tokenizer,
    samples: list[tuple[str, str]],  # (prompt, rollout_text)
    layer: int,
    token_window: int,
    desc: str = "Extracting"
) -> torch.Tensor:
    """Extract attn_out activations for a list of (prompt, rollout) pairs.

    Runs the full text (prompt + rollout) through the model and captures
    activations at the first `token_window` tokens of the rollout portion.
    """
    activations = []

    for prompt, rollout_text in tqdm(samples, desc=desc):
        # Tokenize prompt and full text separately to find rollout start
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]

        full_text = prompt + " " + rollout_text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

        # Storage for activations at each position
        attn_acts = []

        def make_attn_hook():
            def hook(module, input, output):
                # output[0] is attention output: [batch, seq, hidden]
                # Capture all positions
                attn_acts.append(output[0].detach().cpu())
            return hook

        # Register hook
        handle = model.model.layers[layer].self_attn.register_forward_hook(make_attn_hook())

        # Single forward pass (no generation)
        with torch.no_grad():
            model(**inputs)

        handle.remove()

        # Get activations at rollout positions
        # attn_acts[0] is [1, seq_len, hidden]
        all_acts = attn_acts[0].squeeze(0)  # [seq_len, hidden]

        # Extract first token_window tokens of rollout (after prompt)
        rollout_start = prompt_len
        rollout_end = min(rollout_start + token_window, all_acts.shape[0])

        if rollout_end > rollout_start:
            rollout_acts = all_acts[rollout_start:rollout_end]  # [window, hidden]
            mean_act = rollout_acts.mean(dim=0)  # [hidden]
            activations.append(mean_act)

    return torch.stack(activations)  # [n_samples, hidden]


def train_probe(pos_acts: torch.Tensor, neg_acts: torch.Tensor, val_split: float = 0.2):
    """Train logistic probe and return vector + metrics."""

    # Split train/val
    n_pos, n_neg = len(pos_acts), len(neg_acts)
    n_pos_val = int(n_pos * val_split)
    n_neg_val = int(n_neg * val_split)

    # Shuffle
    pos_idx = torch.randperm(n_pos)
    neg_idx = torch.randperm(n_neg)

    pos_train = pos_acts[pos_idx[n_pos_val:]]
    pos_val = pos_acts[pos_idx[:n_pos_val]]
    neg_train = neg_acts[neg_idx[n_neg_val:]]
    neg_val = neg_acts[neg_idx[:n_neg_val]]

    # Prepare data
    X_train = torch.cat([pos_train, neg_train], dim=0).float().numpy()
    y_train = np.array([1] * len(pos_train) + [0] * len(neg_train))

    X_val = torch.cat([pos_val, neg_val], dim=0).float().numpy()
    y_val = np.array([1] * len(pos_val) + [0] * len(neg_val))

    # Train probe
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    clf.fit(X_train, y_train)

    vector = torch.tensor(clf.coef_[0], dtype=torch.float32)

    # Train metrics
    train_probs = clf.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    train_auc = roc_auc_score(y_train, train_probs)

    # Val metrics
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    val_auc = roc_auc_score(y_val, val_probs)

    # Projection-based metrics
    pos_proj = (pos_val @ vector) / vector.norm()
    neg_proj = (neg_val @ vector) / vector.norm()
    val_d = cohens_d(pos_proj.numpy(), neg_proj.numpy())

    metrics = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "train_acc": train_acc,
        "train_auc": train_auc,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_cohens_d": val_d,
        "val_pos_mean": float(pos_proj.mean()),
        "val_neg_mean": float(neg_proj.mean()),
    }

    return vector, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8, help="Layer to extract from")
    parser.add_argument("--window", type=int, default=10, help="Token window to average")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    args = parser.parse_args()

    # Paths
    generation_log = EXPERIMENT_DIR / "extraction/action/refusal/activations_deconfounded/generation_log.json"
    vectors_dir = EXPERIMENT_DIR / "extraction/action/refusal/vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Extracting attn_out vector at layer {args.layer}, window {args.window} ===")

    # Load data
    print("\nLoading rollouts...")
    refusal_samples, compliance_samples = load_rollouts(generation_log)
    print(f"  Refusal: {len(refusal_samples)}, Compliance: {len(compliance_samples)}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    # Extract activations
    print(f"\nExtracting attn_out at layer {args.layer}...")
    refusal_acts = extract_activations(
        model, tokenizer, refusal_samples,
        layer=args.layer, token_window=args.window, desc="Refusal"
    )
    compliance_acts = extract_activations(
        model, tokenizer, compliance_samples,
        layer=args.layer, token_window=args.window, desc="Compliance"
    )

    print(f"  Refusal acts: {refusal_acts.shape}")
    print(f"  Compliance acts: {compliance_acts.shape}")

    # Train probe
    print("\nTraining probe...")
    vector, metrics = train_probe(refusal_acts, compliance_acts, val_split=args.val_split)

    print(f"\nResults:")
    print(f"  Train acc: {metrics['train_acc']:.3f}")
    print(f"  Val acc:   {metrics['val_acc']:.3f}")
    print(f"  Val AUC:   {metrics['val_auc']:.3f}")
    print(f"  Val d:     {metrics['val_cohens_d']:.2f}")

    # Save vector
    output_path = vectors_dir / f"attn_out_probe_layer{args.layer}.pt"
    torch.save(vector, output_path)
    print(f"\nSaved: {output_path}")

    # Save metrics
    metrics_path = vectors_dir / f"attn_out_probe_layer{args.layer}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
