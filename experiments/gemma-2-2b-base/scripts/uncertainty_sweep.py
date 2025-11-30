#!/usr/bin/env python3
"""
Uncertainty extraction experiment sweep.

Experiments:
A: Cache activations from positive (uncertain) and negative (certain) responses
B: Token window sweep (1, 3, 5, 10, 30)
C: Single vector across layers
D: Component isolation comparison
E: IT transfer test

Usage:
    python experiments/gemma-2-2b-base/scripts/uncertainty_sweep.py
"""

import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EXPERIMENT_DIR = Path(__file__).parent.parent
BASE_MODEL = "google/gemma-2-2b"
IT_MODEL = "google/gemma-2-2b-it"
N_LAYERS = 26
HIDDEN_DIM = 2304
TOKEN_WINDOWS = [1, 3, 5, 10, 30]
MAX_NEW_TOKENS = 30  # Generate enough for largest window


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


def save_scratchpad(content: str):
    """Save progress to scratchpad."""
    with open(EXPERIMENT_DIR / "SCRATCHPAD.md", "w") as f:
        f.write(content)


# ============================================================================
# EXPERIMENT A: Cache activations from generations
# ============================================================================

def experiment_a(model, tokenizer, data):
    """Cache activations from positive (uncertain) and negative (certain) responses."""
    print("\n" + "="*70)
    print("EXPERIMENT A: Cache activations from generations")
    print("="*70)

    output_dir = EXPERIMENT_DIR / "extraction" / "epistemic" / "uncertainty" / "sweep_results" / "sorted_activations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine train and val for full dataset
    all_samples = data.get("train", []) + data.get("val", [])

    uncertain_samples = []
    certain_samples = []

    for sample in all_samples:
        # Each sample has positive (uncertain) and negative (certain)
        uncertain_samples.append({
            "prompt": sample["positive_prompt"],
            "text": sample["positive_generation"],
            "full": sample["positive_prompt"] + sample["positive_generation"]
        })
        certain_samples.append({
            "prompt": sample["negative_prompt"],
            "text": sample["negative_generation"],
            "full": sample["negative_prompt"] + sample["negative_generation"]
        })

    print(f"Sample counts:")
    print(f"  Uncertain: {len(uncertain_samples)}")
    print(f"  Certain: {len(certain_samples)}")

    # Cache activations for each rollout
    def cache_activations(rollouts, label):
        """Generate and cache activations for rollouts."""
        all_acts = {
            "residual": [],
            "attn_out": [],
            "mlp_out": [],
        }

        for rollout in tqdm(rollouts, desc=f"Caching {label}"):
            # Tokenize the prompt
            prompt = rollout["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            # Storage for this generation
            gen_residual = {l: [] for l in range(N_LAYERS)}
            gen_attn = {l: [] for l in range(N_LAYERS)}
            gen_mlp = {l: [] for l in range(N_LAYERS)}

            # Set up hooks
            handles = []

            def make_res_hook(layer):
                def hook(module, input, output):
                    gen_residual[layer].append(output[0][:, -1, :].detach().cpu())
                return hook

            def make_attn_hook(layer):
                def hook(module, input, output):
                    gen_attn[layer].append(output[0][:, -1, :].detach().cpu())
                return hook

            def make_mlp_hook(layer):
                def hook(module, input, output):
                    gen_mlp[layer].append(output[:, -1, :].detach().cpu())
                return hook

            for l in range(N_LAYERS):
                handles.append(model.model.layers[l].register_forward_hook(make_res_hook(l)))
                handles.append(model.model.layers[l].self_attn.register_forward_hook(make_attn_hook(l)))
                handles.append(model.model.layers[l].mlp.register_forward_hook(make_mlp_hook(l)))

            # Generate
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Remove hooks
            for h in handles:
                h.remove()

            # Stack activations per layer (skip prompt tokens)
            residual_stack = []
            attn_stack = []
            mlp_stack = []

            for l in range(N_LAYERS):
                # After prefill, each generate step adds one token
                gen_tokens = gen_residual[l][prompt_len:]  # Skip prefill
                if len(gen_tokens) >= MAX_NEW_TOKENS:
                    gen_tokens = gen_tokens[:MAX_NEW_TOKENS]
                if gen_tokens:
                    residual_stack.append(torch.stack(gen_tokens).squeeze(1))  # [n_tokens, hidden]
                else:
                    residual_stack.append(torch.zeros(1, HIDDEN_DIM))

                gen_attn_tokens = gen_attn[l][prompt_len:]
                if len(gen_attn_tokens) >= MAX_NEW_TOKENS:
                    gen_attn_tokens = gen_attn_tokens[:MAX_NEW_TOKENS]
                if gen_attn_tokens:
                    attn_stack.append(torch.stack(gen_attn_tokens).squeeze(1))
                else:
                    attn_stack.append(torch.zeros(1, HIDDEN_DIM))

                gen_mlp_tokens = gen_mlp[l][prompt_len:]
                if len(gen_mlp_tokens) >= MAX_NEW_TOKENS:
                    gen_mlp_tokens = gen_mlp_tokens[:MAX_NEW_TOKENS]
                if gen_mlp_tokens:
                    mlp_stack.append(torch.stack(gen_mlp_tokens).squeeze(1))
                else:
                    mlp_stack.append(torch.zeros(1, HIDDEN_DIM))

            # Stack: [n_layers, n_tokens, hidden]
            all_acts["residual"].append(torch.stack(residual_stack))
            all_acts["attn_out"].append(torch.stack(attn_stack))
            all_acts["mlp_out"].append(torch.stack(mlp_stack))

        return all_acts

    print("\nCaching uncertain activations...")
    uncertain_acts = cache_activations(uncertain_samples, "uncertain")

    print("\nCaching certain activations...")
    certain_acts = cache_activations(certain_samples, "certain")

    # Save
    torch.save(uncertain_acts, output_dir / "uncertain_activations.pt")
    torch.save(certain_acts, output_dir / "certain_activations.pt")

    # Save metadata
    meta = {
        "n_uncertain": len(uncertain_samples),
        "n_certain": len(certain_samples),
        "n_layers": N_LAYERS,
        "max_tokens": MAX_NEW_TOKENS,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {output_dir}")

    return uncertain_acts, certain_acts, meta


# ============================================================================
# EXPERIMENT B: Token window sweep
# ============================================================================

def experiment_b(uncertain_acts, certain_acts):
    """Token window sweep with mean difference vectors."""
    print("\n" + "="*70)
    print("EXPERIMENT B: Token window sweep")
    print("="*70)

    output_dir = EXPERIMENT_DIR / "extraction" / "epistemic" / "uncertainty" / "sweep_results"

    results = {}

    for window in TOKEN_WINDOWS:
        print(f"\n--- Window: {window} tokens ---")
        results[window] = {"layers": {}}

        # Aggregate activations for this window
        def aggregate_window(acts_list, component, w):
            """Mean across first w tokens."""
            agg = []
            for acts in acts_list:
                # acts[component] is [n_layers, n_tokens, hidden]
                n_tokens = acts.shape[1]
                w_actual = min(w, n_tokens)
                # Mean across tokens: [n_layers, hidden]
                mean_acts = acts[:, :w_actual, :].mean(dim=1)
                agg.append(mean_acts)
            return torch.stack(agg)  # [n_samples, n_layers, hidden]

        unc_agg = aggregate_window(uncertain_acts["residual"], "residual", window)
        cert_agg = aggregate_window(certain_acts["residual"], "residual", window)

        # Train/val split (80/20)
        n_unc = len(unc_agg)
        n_cert = len(cert_agg)
        n_unc_train = int(0.8 * n_unc)
        n_cert_train = int(0.8 * n_cert)

        # Shuffle
        unc_perm = torch.randperm(n_unc)
        cert_perm = torch.randperm(n_cert)

        unc_train = unc_agg[unc_perm[:n_unc_train]]
        unc_val = unc_agg[unc_perm[n_unc_train:]]
        cert_train = cert_agg[cert_perm[:n_cert_train]]
        cert_val = cert_agg[cert_perm[n_cert_train:]]

        best_layer = None
        best_acc = 0

        for layer in range(N_LAYERS):
            # Extract at this layer
            unc_l_train = unc_train[:, layer, :].float()
            cert_l_train = cert_train[:, layer, :].float()
            unc_l_val = unc_val[:, layer, :].float()
            cert_l_val = cert_val[:, layer, :].float()

            # Mean difference vector (uncertain - certain)
            vector = unc_l_train.mean(dim=0) - cert_l_train.mean(dim=0)

            # Project validation
            unc_proj = (unc_l_val @ vector) / vector.norm()
            cert_proj = (cert_l_val @ vector) / vector.norm()

            # Metrics
            d = cohens_d(unc_proj.numpy(), cert_proj.numpy())

            # Accuracy (uncertain should project higher)
            all_proj = torch.cat([unc_proj, cert_proj])
            all_labels = np.array([1] * len(unc_proj) + [0] * len(cert_proj))
            threshold = (unc_proj.mean() + cert_proj.mean()) / 2
            preds = (all_proj > threshold).numpy().astype(int)
            acc = (preds == all_labels).mean()

            # AUC
            try:
                auc = roc_auc_score(all_labels, all_proj.numpy())
            except:
                auc = 0.5

            results[window]["layers"][layer] = {
                "accuracy": float(acc),
                "cohens_d": float(d),
                "auc": float(auc),
                "unc_mean": float(unc_proj.mean()),
                "cert_mean": float(cert_proj.mean()),
            }

            if acc > best_acc:
                best_acc = acc
                best_layer = layer

        results[window]["best_layer"] = best_layer
        results[window]["best_accuracy"] = best_acc
        print(f"Best layer: {best_layer} with accuracy: {best_acc:.1%}")

    # Save results
    with open(output_dir / "window_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(TOKEN_WINDOWS)))

    for i, window in enumerate(TOKEN_WINDOWS):
        layers = list(range(N_LAYERS))
        accs = [results[window]["layers"][l]["accuracy"] for l in layers]
        ax.plot(layers, accs, '-o', color=colors[i], label=f"Window {window}", linewidth=2, markersize=4)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Uncertainty: Token Window Sweep - Accuracy by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "window_sweep_plot.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'window_sweep_plot.png'}")

    return results


# ============================================================================
# EXPERIMENT C: Single vector across layers
# ============================================================================

def experiment_c(uncertain_acts, certain_acts, window_results):
    """Test single vector across all layers."""
    print("\n" + "="*70)
    print("EXPERIMENT C: Single vector across layers")
    print("="*70)

    output_dir = EXPERIMENT_DIR / "extraction" / "epistemic" / "uncertainty" / "sweep_results"

    # Find best (window, layer) from B
    best_window = None
    best_layer = None
    best_acc = 0

    for window in window_results:
        if window_results[window]["best_accuracy"] > best_acc:
            best_acc = window_results[window]["best_accuracy"]
            best_window = window
            best_layer = window_results[window]["best_layer"]

    print(f"Using best: window={best_window}, layer={best_layer}, acc={best_acc:.1%}")

    # Aggregate at best window
    def aggregate_window(acts_list, w):
        agg = []
        for acts in acts_list:
            n_tokens = acts.shape[1]
            w_actual = min(w, n_tokens)
            mean_acts = acts[:, :w_actual, :].mean(dim=1)
            agg.append(mean_acts)
        return torch.stack(agg)

    unc_agg = aggregate_window(uncertain_acts["residual"], best_window)
    cert_agg = aggregate_window(certain_acts["residual"], best_window)

    # Extract vector at best layer
    unc_best = unc_agg[:, best_layer, :].float()
    cert_best = cert_agg[:, best_layer, :].float()
    vector = unc_best.mean(dim=0) - cert_best.mean(dim=0)

    print(f"Vector norm: {vector.norm().item():.2f}")

    # Project all layers onto this single vector
    results = {"best_window": best_window, "best_layer": best_layer, "vector_norm": float(vector.norm()), "layers": {}}

    for layer in range(N_LAYERS):
        unc_l = unc_agg[:, layer, :].float()
        cert_l = cert_agg[:, layer, :].float()

        unc_proj = (unc_l @ vector) / vector.norm()
        cert_proj = (cert_l @ vector) / vector.norm()

        d = cohens_d(unc_proj.numpy(), cert_proj.numpy())

        results["layers"][layer] = {
            "cohens_d": float(d),
            "unc_mean": float(unc_proj.mean()),
            "cert_mean": float(cert_proj.mean()),
        }

    # Save
    with open(output_dir / "single_vector_crosslayer.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(N_LAYERS))
    ds = [results["layers"][l]["cohens_d"] for l in layers]

    ax.bar(layers, ds, color='steelblue', alpha=0.7)
    ax.axvline(x=best_layer, color='red', linestyle='--', label=f'Source layer ({best_layer})')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"Uncertainty: Single Vector (layer {best_layer}, window {best_window}) Projected Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "single_vector_crosslayer.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'single_vector_crosslayer.png'}")

    return results


# ============================================================================
# EXPERIMENT D: Component isolation
# ============================================================================

def experiment_d(uncertain_acts, certain_acts, best_window):
    """Compare components: residual, attn_out, mlp_out."""
    print("\n" + "="*70)
    print("EXPERIMENT D: Component isolation")
    print("="*70)

    output_dir = EXPERIMENT_DIR / "extraction" / "epistemic" / "uncertainty" / "sweep_results"

    components = ["residual", "attn_out", "mlp_out"]
    results = {"window": best_window, "components": {}}

    for component in components:
        print(f"\n--- Component: {component} ---")
        results["components"][component] = {"layers": {}}

        # Aggregate
        def aggregate_window(acts_list, comp, w):
            agg = []
            for acts in acts_list[comp]:
                n_tokens = acts.shape[1]
                w_actual = min(w, n_tokens)
                mean_acts = acts[:, :w_actual, :].mean(dim=1)
                agg.append(mean_acts)
            return torch.stack(agg)

        unc_agg = aggregate_window(uncertain_acts, component, best_window)
        cert_agg = aggregate_window(certain_acts, component, best_window)

        # Train/val split
        n_unc = len(unc_agg)
        n_cert = len(cert_agg)
        n_unc_train = int(0.8 * n_unc)
        n_cert_train = int(0.8 * n_cert)

        unc_perm = torch.randperm(n_unc)
        cert_perm = torch.randperm(n_cert)

        unc_train = unc_agg[unc_perm[:n_unc_train]]
        unc_val = unc_agg[unc_perm[n_unc_train:]]
        cert_train = cert_agg[cert_perm[:n_cert_train]]
        cert_val = cert_agg[cert_perm[n_cert_train:]]

        best_layer = None
        best_acc = 0

        for layer in range(N_LAYERS):
            unc_l_train = unc_train[:, layer, :].float()
            cert_l_train = cert_train[:, layer, :].float()
            unc_l_val = unc_val[:, layer, :].float()
            cert_l_val = cert_val[:, layer, :].float()

            vector = unc_l_train.mean(dim=0) - cert_l_train.mean(dim=0)

            unc_proj = (unc_l_val @ vector) / vector.norm()
            cert_proj = (cert_l_val @ vector) / vector.norm()

            d = cohens_d(unc_proj.numpy(), cert_proj.numpy())

            all_proj = torch.cat([unc_proj, cert_proj])
            all_labels = np.array([1] * len(unc_proj) + [0] * len(cert_proj))
            threshold = (unc_proj.mean() + cert_proj.mean()) / 2
            preds = (all_proj > threshold).numpy().astype(int)
            acc = (preds == all_labels).mean()

            results["components"][component]["layers"][layer] = {
                "accuracy": float(acc),
                "cohens_d": float(d),
            }

            if acc > best_acc:
                best_acc = acc
                best_layer = layer

        results["components"][component]["best_layer"] = best_layer
        results["components"][component]["best_accuracy"] = best_acc
        print(f"Best layer: {best_layer} with accuracy: {best_acc:.1%}")

    # Save
    with open(output_dir / "component_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'residual': 'blue', 'attn_out': 'orange', 'mlp_out': 'green'}
    layers = list(range(N_LAYERS))

    # Accuracy plot
    ax1 = axes[0]
    for comp in components:
        accs = [results["components"][comp]["layers"][l]["accuracy"] for l in layers]
        ax1.plot(layers, accs, '-o', color=colors[comp], label=comp, linewidth=2, markersize=4)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Uncertainty: Component Comparison - Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cohen's d plot
    ax2 = axes[1]
    for comp in components:
        ds = [results["components"][comp]["layers"][l]["cohens_d"] for l in layers]
        ax2.plot(layers, ds, '-o', color=colors[comp], label=comp, linewidth=2, markersize=4)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel("Cohen's d")
    ax2.set_title("Uncertainty: Component Comparison - Cohen's d")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Component Isolation (window={best_window})')
    plt.tight_layout()
    plt.savefig(output_dir / "component_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'component_comparison.png'}")

    return results


# ============================================================================
# EXPERIMENT E: IT transfer
# ============================================================================

def experiment_e(best_vector, best_layer, best_window):
    """Test transfer to IT model."""
    print("\n" + "="*70)
    print("EXPERIMENT E: IT transfer")
    print("="*70)

    output_dir = EXPERIMENT_DIR / "extraction" / "epistemic" / "uncertainty" / "sweep_results"

    # IT test prompts: uncertain vs certain topics
    uncertain_prompts = [
        "What causes consciousness in the brain?",
        "Will we ever achieve true artificial general intelligence?",
        "What happened before the Big Bang?",
        "How will climate change affect global agriculture in 2100?",
        "Can quantum effects explain free will?",
        "What is the optimal diet for human longevity?",
        "Will humans colonize other star systems?",
        "What caused the Cambrian explosion?",
        "Is there intelligent life elsewhere in the universe?",
        "How did the first life on Earth originate?",
    ]

    certain_prompts = [
        "What is the speed of light in a vacuum?",
        "How many chromosomes do humans have?",
        "What year did World War II end?",
        "What is the chemical formula for water?",
        "What is the capital of France?",
        "How many planets are in our solar system?",
        "What is the atomic number of carbon?",
        "What year was the Declaration of Independence signed?",
        "What is the boiling point of water at sea level?",
        "How many continents are there on Earth?",
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

    # Move vector to model device
    vector = best_vector.to(model.device).to(model.dtype)

    def capture_generation(prompts, label):
        """Generate responses and capture activations."""
        all_acts = []

        for prompt in tqdm(prompts, desc=label):
            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            # Capture at best layer only
            gen_acts = []

            def hook(module, input, output):
                gen_acts.append(output[0][:, -1, :].detach())

            handle = model.model.layers[best_layer].register_forward_hook(hook)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            handle.remove()

            # Get generated token activations
            gen_only = gen_acts[prompt_len:]
            if len(gen_only) >= best_window:
                gen_only = gen_only[:best_window]

            if gen_only:
                mean_act = torch.stack(gen_only).mean(dim=0).squeeze()
                all_acts.append(mean_act)

        return torch.stack(all_acts) if all_acts else None

    print("\nGenerating uncertain responses...")
    uncertain_it_acts = capture_generation(uncertain_prompts, "Uncertain")

    print("\nGenerating certain responses...")
    certain_it_acts = capture_generation(certain_prompts, "Certain")

    # Project onto base vector
    unc_proj = (uncertain_it_acts.to(vector.dtype) @ vector) / vector.norm()
    cert_proj = (certain_it_acts.to(vector.dtype) @ vector) / vector.norm()

    # Metrics
    d = cohens_d(unc_proj.float().cpu().numpy(), cert_proj.float().cpu().numpy())

    all_proj = torch.cat([unc_proj, cert_proj]).float().cpu()
    all_labels = np.array([1] * len(unc_proj) + [0] * len(cert_proj))

    threshold = (unc_proj.mean() + cert_proj.mean()) / 2
    preds = (all_proj > threshold.float().cpu()).numpy().astype(int)
    acc = (preds == all_labels).mean()

    try:
        auc = roc_auc_score(all_labels, all_proj.numpy())
    except:
        auc = 0.5

    results = {
        "best_layer": best_layer,
        "best_window": best_window,
        "cohens_d": float(d),
        "accuracy": float(acc),
        "auc": float(auc),
        "uncertain_mean": float(unc_proj.mean()),
        "certain_mean": float(cert_proj.mean()),
        "separation": float(unc_proj.mean() - cert_proj.mean()),
    }

    print(f"\nIT Transfer Results:")
    print(f"  Cohen's d: {d:.2f}")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  AUC: {auc:.3f}")
    print(f"  Uncertain mean: {unc_proj.mean().item():.2f}")
    print(f"  Certain mean: {cert_proj.mean().item():.2f}")
    print(f"  Separation: {(unc_proj.mean() - cert_proj.mean()).item():.2f}")

    # Save
    with open(output_dir / "it_transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(unc_proj.float().cpu().numpy(), bins=15, alpha=0.6, color='purple', label='Uncertain', density=True)
    ax.hist(cert_proj.float().cpu().numpy(), bins=15, alpha=0.6, color='green', label='Certain', density=True)
    ax.axvline(x=threshold.float().cpu().item(), color='black', linestyle='--', label=f'Threshold')
    ax.set_xlabel('Projection onto Base Vector')
    ax.set_ylabel('Density')
    ax.set_title(f'Uncertainty: IT Transfer Test (layer {best_layer}, window {best_window})\nd={d:.2f}, acc={acc:.1%}, AUC={auc:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "it_transfer_plot.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'it_transfer_plot.png'}")

    return results


# ============================================================================
# SUMMARY
# ============================================================================

def write_summary(meta_a, results_b, results_c, results_d, results_e):
    """Write summary markdown comparing to refusal."""
    output_dir = EXPERIMENT_DIR / "extraction" / "epistemic" / "uncertainty" / "sweep_results"

    # Find best window
    best_window = None
    best_acc = 0
    for w in results_b:
        if results_b[w]["best_accuracy"] > best_acc:
            best_acc = results_b[w]["best_accuracy"]
            best_window = w

    best_layer = results_b[best_window]["best_layer"]

    # Find best component
    best_comp = None
    best_comp_acc = 0
    for comp in results_d["components"]:
        if results_d["components"][comp]["best_accuracy"] > best_comp_acc:
            best_comp_acc = results_d["components"][comp]["best_accuracy"]
            best_comp = comp

    summary = f"""# Uncertainty Extraction Sweep Results

## Comparison with Refusal

| Metric | Uncertainty | Refusal |
|--------|-------------|---------|
| Sample sizes | {meta_a['n_uncertain']}/{meta_a['n_certain']} | 228/360 |
| Best window | {best_window} | 10 |
| Best layer | {best_layer} | 15 |
| Best component | {best_comp} | attn_out |
| Base accuracy | {best_acc:.1%} | 64.4% |
| IT transfer d | {results_e['cohens_d']:.2f} | 3.55 |
| IT transfer acc | {results_e['accuracy']:.1%} | 90% |

---

## Sample Sizes (Experiment A)

| Class | Count |
|-------|-------|
| Uncertain | {meta_a['n_uncertain']} |
| Certain | {meta_a['n_certain']} |
| **Total** | **{meta_a['n_uncertain'] + meta_a['n_certain']}** |

---

## Best Window (Experiment B)

**Winner: Window {best_window}** with {best_acc:.1%} accuracy

| Window | Best Layer | Best Accuracy |
|--------|------------|---------------|
"""

    for w in TOKEN_WINDOWS:
        summary += f"| {w} | {results_b[w]['best_layer']} | {results_b[w]['best_accuracy']:.1%} |\n"

    summary += f"""
---

## Single Vector Generalization (Experiment C)

Vector from layer {results_c['best_layer']}, window {results_c['best_window']} (norm: {results_c['vector_norm']:.2f})

Does it work across layers?
"""

    # Check if separation is maintained
    positive_layers = sum(1 for l in results_c["layers"] if results_c["layers"][l]["cohens_d"] > 1.0)
    summary += f"- **{positive_layers}/{N_LAYERS}** layers have Cohen's d > 1.0\n"
    summary += f"- Vector generalizes {'well' if positive_layers > 15 else 'moderately' if positive_layers > 10 else 'poorly'} across layers\n"

    summary += f"""
---

## Component Analysis (Experiment D)

**Best component: {best_comp}** with {best_comp_acc:.1%} accuracy

| Component | Best Layer | Best Accuracy |
|-----------|------------|---------------|
"""

    for comp in ["residual", "attn_out", "mlp_out"]:
        summary += f"| {comp} | {results_d['components'][comp]['best_layer']} | {results_d['components'][comp]['best_accuracy']:.1%} |\n"

    summary += f"""
---

## IT Transfer (Experiment E)

| Metric | Value |
|--------|-------|
| Cohen's d | {results_e['cohens_d']:.2f} |
| Accuracy | {results_e['accuracy']:.1%} |
| AUC | {results_e['auc']:.3f} |
| Uncertain mean | {results_e['uncertain_mean']:.2f} |
| Certain mean | {results_e['certain_mean']:.2f} |
| Separation | {results_e['separation']:.2f} |

---

## Key Findings

"""

    # Generate key findings
    if results_e['cohens_d'] > 2.0:
        summary += "1. **Strong transfer**: Base uncertainty vector transfers well to IT model.\n"
    elif results_e['cohens_d'] > 1.0:
        summary += "1. **Moderate transfer**: Base uncertainty vector shows meaningful transfer to IT model.\n"
    else:
        summary += "1. **Weak transfer**: Base uncertainty vector shows limited transfer to IT model.\n"

    if best_comp == "residual":
        summary += "2. **Residual stream carries signal**: Full residual (cumulative) outperforms isolated components.\n"
    else:
        summary += f"2. **{best_comp} dominates**: The {best_comp} component carries the strongest signal.\n"

    if positive_layers > 15:
        summary += "3. **Single vector generalizes**: One vector works across most layers.\n"
    else:
        summary += "3. **Layer-specific vectors needed**: Single vector doesn't generalize well across layers.\n"

    with open(output_dir / "summary.md", "w") as f:
        f.write(summary)

    print(f"\nSaved: {output_dir / 'summary.md'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load generations data
    data_path = EXPERIMENT_DIR / "extraction" / "epistemic" / "uncertainty" / "generations.json"
    with open(data_path) as f:
        data = json.load(f)

    n_train = len(data.get("train", []))
    n_val = len(data.get("val", []))

    print("="*70)
    print("UNCERTAINTY EXTRACTION SWEEP")
    print("="*70)
    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")
    print(f"Total: {n_train + n_val}")

    # Check if activations already exist
    sorted_dir = EXPERIMENT_DIR / "extraction" / "epistemic" / "uncertainty" / "sweep_results" / "sorted_activations"

    if (sorted_dir / "uncertain_activations.pt").exists():
        print("\nLoading cached activations...")
        uncertain_acts = torch.load(sorted_dir / "uncertain_activations.pt")
        certain_acts = torch.load(sorted_dir / "certain_activations.pt")
        with open(sorted_dir / "metadata.json") as f:
            meta_a = json.load(f)
    else:
        # Load base model
        print(f"\nLoading {BASE_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        # Experiment A
        uncertain_acts, certain_acts, meta_a = experiment_a(model, tokenizer, data)

        # Free base model memory
        del model
        torch.cuda.empty_cache()

    # Experiment B
    results_b = experiment_b(uncertain_acts, certain_acts)

    # Experiment C
    results_c = experiment_c(uncertain_acts, certain_acts, results_b)

    # Find best window for D
    best_window = None
    best_acc = 0
    for w in results_b:
        if results_b[w]["best_accuracy"] > best_acc:
            best_acc = results_b[w]["best_accuracy"]
            best_window = w

    # Experiment D
    results_d = experiment_d(uncertain_acts, certain_acts, best_window)

    # Extract best vector for E
    def aggregate_window(acts_list, w):
        agg = []
        for acts in acts_list:
            n_tokens = acts.shape[1]
            w_actual = min(w, n_tokens)
            mean_acts = acts[:, :w_actual, :].mean(dim=1)
            agg.append(mean_acts)
        return torch.stack(agg)

    unc_agg = aggregate_window(uncertain_acts["residual"], best_window)
    cert_agg = aggregate_window(certain_acts["residual"], best_window)

    best_layer = results_b[best_window]["best_layer"]
    unc_best = unc_agg[:, best_layer, :].float()
    cert_best = cert_agg[:, best_layer, :].float()
    best_vector = unc_best.mean(dim=0) - cert_best.mean(dim=0)

    # Experiment E
    results_e = experiment_e(best_vector, best_layer, best_window)

    # Summary
    write_summary(meta_a, results_b, results_c, results_d, results_e)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
