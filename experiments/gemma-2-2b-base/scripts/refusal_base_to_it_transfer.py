#!/usr/bin/env python3
"""
Test if base model's refusal vector transfers to instruction-tuned model.

Loads the deconfounded refusal vector from base model, runs IT model on
refusal/compliance prompts, projects IT activations onto base vector.

Usage:
    python experiments/gemma-2-2b-base/scripts/refusal_base_to_it_transfer.py
"""

import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENT_DIR = Path(__file__).parent.parent
IT_MODEL = "google/gemma-2-2b-it"


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
        activation = output[0][:, -1, :].detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    return activation.squeeze(0)


def main():
    trait_dir = EXPERIMENT_DIR / "extraction" / "action" / "refusal"
    deconf_dir = trait_dir / "activations_deconfounded"

    # Load generation log to get agreed prompts
    with open(deconf_dir / "generation_log.json") as f:
        log = json.load(f)

    # Filter to prompts with agreement
    refusal_prompts = []
    compliance_prompts = []

    for result in log["results"]:
        if result["agreement"] >= 4:
            if result["dominant_class"] == "refusal":
                refusal_prompts.append(result["truncated"])
            elif result["dominant_class"] == "compliance":
                compliance_prompts.append(result["truncated"])

    print(f"Refusal prompts: {len(refusal_prompts)}")
    print(f"Compliance prompts: {len(compliance_prompts)}")

    if len(refusal_prompts) < 3 or len(compliance_prompts) < 3:
        print("ERROR: Not enough agreed prompts for transfer test")
        return

    # Load results to find best layer
    with open(trait_dir / "vectors_deconfounded" / "results.json") as f:
        results = json.load(f)

    # Use window 5 residual layer 0 (best: 100% acc, d=9.80)
    # Actually let's try multiple layers
    test_configs = [
        (5, "residual", 0),   # best for window 5
        (15, "residual", 0),  # best for window 15
        (30, "residual", 11), # best for window 30
    ]

    # Load IT model
    print(f"\nLoading {IT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(IT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        IT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("\n" + "="*70)
    print("BASE → IT TRANSFER: Deconfounded Refusal Vector")
    print("="*70)

    for window, component, layer in test_configs:
        print(f"\n--- Window {window}, {component} layer {layer} ---")

        # Load base model's vector (need to extract from activations since no .pt saved)
        window_dir = deconf_dir / f"window_{window}"
        acts = torch.load(window_dir / f"{component}.pt")

        with open(window_dir / "metadata.json") as f:
            meta = json.load(f)
        n_refusal = meta["n_refusal"]

        # Extract vector using mean difference (simple)
        pos_acts = acts[:n_refusal, layer, :].float()
        neg_acts = acts[n_refusal:, layer, :].float()
        vector = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)

        # Move to model device
        vector = vector.to(model.device).to(model.dtype)

        # Extract IT activations
        refusal_projs = []
        compliance_projs = []

        for prompt in refusal_prompts:
            act = extract_activation(model, tokenizer, prompt, layer)
            proj = (act @ vector) / vector.norm()
            refusal_projs.append(proj.float().cpu().item())

        for prompt in compliance_prompts:
            act = extract_activation(model, tokenizer, prompt, layer)
            proj = (act @ vector) / vector.norm()
            compliance_projs.append(proj.float().cpu().item())

        refusal_projs = np.array(refusal_projs)
        compliance_projs = np.array(compliance_projs)

        # Metrics
        print(f"  Refusal mean:    {refusal_projs.mean():.3f} ± {refusal_projs.std():.3f}")
        print(f"  Compliance mean: {compliance_projs.mean():.3f} ± {compliance_projs.std():.3f}")

        d = cohens_d(refusal_projs, compliance_projs)
        print(f"  Cohen's d: {d:.3f}")

        # Accuracy
        threshold = (refusal_projs.mean() + compliance_projs.mean()) / 2
        ref_correct = (refusal_projs > threshold).sum()
        comp_correct = (compliance_projs < threshold).sum()
        acc = (ref_correct + comp_correct) / (len(refusal_projs) + len(compliance_projs))
        print(f"  Accuracy: {acc:.3f}")

        # AUC
        y_true = np.array([1] * len(refusal_projs) + [0] * len(compliance_projs))
        y_score = np.concatenate([refusal_projs, compliance_projs])
        try:
            auc = roc_auc_score(y_true, y_score)
            print(f"  AUC: {auc:.3f}")
        except:
            print(f"  AUC: N/A")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
If Cohen's d > 1.0 and accuracy > 75%: STRONG TRANSFER
  → Base model's refusal representation survives instruction tuning
  → The concept predates alignment

If Cohen's d > 0.5 and accuracy > 65%: MODERATE TRANSFER
  → Partial preservation, IT may have modified the representation

Otherwise: WEAK TRANSFER
  → IT created new refusal representations
  → Base vector doesn't capture IT's refusal concept
""")


if __name__ == "__main__":
    main()
