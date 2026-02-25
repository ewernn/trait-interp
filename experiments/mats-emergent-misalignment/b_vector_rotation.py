"""Track rotation of lora_B weight vectors across fine-tuning checkpoints.

Computes cosine similarity between consecutive, initial, and final B vectors
to understand how the LoRA output direction evolves during training.

Input: adapter_model.safetensors at each checkpoint
Output: experiments/mats-emergent-misalignment/analysis/b_vector_rotation/{run}.json
Usage: python experiments/mats-emergent-misalignment/b_vector_rotation.py --run rank1
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

EXPERIMENT_DIR = Path(__file__).parent
FINETUNE_DIR = EXPERIMENT_DIR / "finetune"
OUTPUT_DIR = EXPERIMENT_DIR / "analysis" / "b_vector_rotation"
B_KEY = "base_model.model.model.layers.24.mlp.down_proj.lora_B.weight"


def load_b_vector(checkpoint_path: Path) -> torch.Tensor:
    """Load and flatten the lora_B weight from a checkpoint."""
    safetensors_path = checkpoint_path / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {checkpoint_path}")
    f = safe_open(str(safetensors_path), framework="pt", device="cpu")
    tensor = f.get_tensor(B_KEY)  # [out_dim, rank]
    return tensor.flatten().float()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def get_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (step, path) for all checkpoints including final."""
    checkpoints = []
    for p in run_dir.iterdir():
        m = re.match(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir():
            checkpoints.append((int(m.group(1)), p))
    checkpoints.sort(key=lambda x: x[0])

    # Include final/ if it exists and has adapter weights
    final_dir = run_dir / "final"
    if final_dir.exists() and (final_dir / "adapter_model.safetensors").exists():
        last_step = checkpoints[-1][0] if checkpoints else 0
        # Check if final is different from last checkpoint
        final_vec = load_b_vector(final_dir)
        last_vec = load_b_vector(checkpoints[-1][1])
        if not torch.allclose(final_vec, last_vec, atol=1e-6):
            checkpoints.append((last_step + 1, final_dir))

    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Track lora_B vector rotation across checkpoints")
    parser.add_argument("--run", default="rank1", help="Run name (subdirectory of finetune/)")
    args = parser.parse_args()

    run_dir = FINETUNE_DIR / args.run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoints = get_checkpoints(run_dir)
    if len(checkpoints) < 2:
        raise ValueError(f"Need at least 2 checkpoints, found {len(checkpoints)}")

    print(f"Loading B vectors from {len(checkpoints)} checkpoints ({args.run})...")
    print(f"  Key: {B_KEY}")

    # Load all B vectors
    vectors = {}
    for step, path in checkpoints:
        vectors[step] = load_b_vector(path)

    # Check original shape
    f = safe_open(str(checkpoints[0][1] / "adapter_model.safetensors"), framework="pt", device="cpu")
    orig_shape = list(f.get_tensor(B_KEY).shape)
    print(f"  Shape: {orig_shape} -> flattened to [{vectors[checkpoints[0][0]].numel()}]")
    print()

    steps = [s for s, _ in checkpoints]
    first_vec = vectors[steps[0]]
    last_vec = vectors[steps[-1]]

    # Compute similarities
    results = {
        "run": args.run,
        "key": B_KEY,
        "original_shape": orig_shape,
        "steps": steps,
        "consecutive_cosine_sim": [],
        "vs_initial_cosine_sim": [],
        "vs_final_cosine_sim": [],
        "norms": [],
    }

    # Header
    print(f"{'Step':>6}  {'Norm':>8}  {'vs Prev':>8}  {'vs Init':>8}  {'vs Final':>8}")
    print("-" * 50)

    for i, step in enumerate(steps):
        vec = vectors[step]
        norm = vec.norm().item()
        results["norms"].append(round(norm, 6))

        # vs previous
        if i == 0:
            cs_prev = None
        else:
            cs_prev = cosine_sim(vec, vectors[steps[i - 1]])
            results["consecutive_cosine_sim"].append(round(cs_prev, 6))

        # vs initial
        cs_init = cosine_sim(vec, first_vec)
        results["vs_initial_cosine_sim"].append(round(cs_init, 6))

        # vs final
        cs_final = cosine_sim(vec, last_vec)
        results["vs_final_cosine_sim"].append(round(cs_final, 6))

        prev_str = f"{cs_prev:.4f}" if cs_prev is not None else "   -   "
        print(f"{step:>6}  {norm:>8.4f}  {prev_str:>8}  {cs_init:>8.4f}  {cs_final:>8.4f}")

    # Summary statistics
    consec = results["consecutive_cosine_sim"]
    print()
    print(f"Consecutive cosine sim: min={min(consec):.4f}  max={max(consec):.4f}  mean={sum(consec)/len(consec):.4f}")
    print(f"Total rotation from initial to final: cosine_sim = {cosine_sim(first_vec, last_vec):.6f}")
    print(f"Norm range: {min(results['norms']):.4f} - {max(results['norms']):.4f}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.run}.json"
    with open(output_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
