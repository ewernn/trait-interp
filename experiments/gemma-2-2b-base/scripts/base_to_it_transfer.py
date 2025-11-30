#!/usr/bin/env python3
"""
Test if base model's uncertainty vector transfers to instruction-tuned model.

Loads layer 9 probe vector from base model, runs IT model on uncertain/certain
prompts, projects IT activations onto base vector.

If it separates: base model's uncertainty representation survives instruction tuning.
"""

import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENT_DIR = Path(__file__).parent.parent
BASE_MODEL = "google/gemma-2-2b"
IT_MODEL = "google/gemma-2-2b-it"
LAYER = 9  # Best cross-distribution layer


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    n1, n2 = len(pos), len(neg)
    var1, var2 = pos.var(), neg.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (pos.mean() - neg.mean()) / pooled_std


def extract_activation(model, tokenizer, prompt: str, layer: int):
    """Extract last-token activation at specified layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    activation = None
    def hook(module, input, output):
        nonlocal activation
        # output is tuple, first element is hidden states
        activation = output[0][:, -1, :].detach()
    
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    return activation.squeeze(0)


def main():
    trait = "epistemic/uncertainty"
    trait_dir = EXPERIMENT_DIR / "extraction" / trait
    
    # Load base model's probe vector
    vector_path = trait_dir / "vectors" / f"residual_probe_layer{LAYER}.pt"
    vector = torch.load(vector_path)
    print(f"Loaded base model vector from layer {LAYER}")
    print(f"Vector norm: {vector.norm():.2f}")
    
    # Load prompts
    with open(trait_dir / "prompts.json") as f:
        pairs = json.load(f)["pairs"]
    
    print(f"Loaded {len(pairs)} prompt pairs")
    
    # Load IT model
    print(f"\nLoading {IT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(IT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        IT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    # Move vector to model device and dtype
    vector = vector.to(model.device).to(model.dtype)
    
    # Extract activations for uncertain (positive) and certain (negative) prompts
    print("\nExtracting IT model activations...")
    pos_projs = []
    neg_projs = []
    
    for i, pair in enumerate(pairs):
        # Get activations
        pos_act = extract_activation(model, tokenizer, pair["positive"], LAYER)
        neg_act = extract_activation(model, tokenizer, pair["negative"], LAYER)
        
        # Project onto base model's vector
        pos_proj = (pos_act @ vector) / vector.norm()
        neg_proj = (neg_act @ vector) / vector.norm()
        
        pos_projs.append(pos_proj.float().cpu().item())
        neg_projs.append(neg_proj.float().cpu().item())
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(pairs)} pairs")
    
    pos_projs = np.array(pos_projs)
    neg_projs = np.array(neg_projs)
    
    # Compute metrics
    print("\n" + "="*60)
    print("RESULTS: Base vector → IT model activations")
    print("="*60)
    
    print(f"\nProjection statistics:")
    print(f"  Uncertain (pos): mean={pos_projs.mean():.3f}, std={pos_projs.std():.3f}")
    print(f"  Certain (neg):   mean={neg_projs.mean():.3f}, std={neg_projs.std():.3f}")
    
    # Cohen's d
    d = cohens_d(pos_projs, neg_projs)
    print(f"\nCohen's d: {d:.3f}")
    
    # Classification accuracy (threshold at midpoint)
    threshold = (pos_projs.mean() + neg_projs.mean()) / 2
    pos_correct = (pos_projs > threshold).sum()
    neg_correct = (neg_projs < threshold).sum()
    acc = (pos_correct + neg_correct) / (len(pos_projs) + len(neg_projs))
    print(f"Accuracy (midpoint threshold): {acc:.3f}")
    
    # AUC
    y_true = np.array([1] * len(pos_projs) + [0] * len(neg_projs))
    y_score = np.concatenate([pos_projs, neg_projs])
    auc = roc_auc_score(y_true, y_score)
    print(f"AUC: {auc:.3f}")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if d > 1.0 and acc > 0.75:
        print("✓ STRONG TRANSFER: Base model's uncertainty representation")
        print("  survives instruction tuning. The concept predates alignment.")
    elif d > 0.5 and acc > 0.65:
        print("~ MODERATE TRANSFER: Partial preservation of base representation.")
        print("  IT may have modified but not replaced the concept.")
    else:
        print("✗ WEAK TRANSFER: IT created new uncertainty representations.")
        print("  The base model's vector doesn't capture IT's uncertainty.")


if __name__ == "__main__":
    main()
