"""
Capture activations during prefill for human vs model text.

Usage:
    # Base model, temp=0 (default)
    python scripts/capture_prefill_activations.py --experiment prefill-dynamics

    # Instruct model
    python scripts/capture_prefill_activations.py --experiment prefill-dynamics \
        --model google/gemma-2-2b-it --condition instruct

    # Different generation (temp=0.7)
    python scripts/capture_prefill_activations.py --experiment prefill-dynamics \
        --data-condition gemma-2-2b-temp07 --condition temp07
"""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

from utils.model import load_model
from inference.capture_raw_activations import capture_residual_stream_prefill

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    parser.add_argument("--model", default="google/gemma-2-2b",
                        help="Model to use for processing activations")
    parser.add_argument("--data-condition", default="gemma-2-2b",
                        help="Which continuations file to load (continuations-{data-condition}.json)")
    parser.add_argument("--condition", default=None,
                        help="Output condition suffix (default: auto from model)")
    parser.add_argument("--human-only", action="store_true",
                        help="Only capture human text (for testing different processing models)")
    parser.add_argument("--model-only", action="store_true",
                        help="Only capture model text (for testing different generation settings)")
    args = parser.parse_args()

    # Auto-generate condition name
    if args.condition is None:
        args.condition = args.model.split("/")[-1]

    # Load data
    data_path = Path(f"experiments/{args.experiment}/data/continuations-{args.data_condition}.json")
    if not data_path.exists():
        # Fallback to old naming
        data_path = Path(f"experiments/{args.experiment}/data/continuations.json")

    with open(data_path) as f:
        data = json.load(f)
    samples = data["samples"]

    # Load model
    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model)
    n_layers = model.config.num_hidden_layers

    # Output directories
    output_dir = Path(f"experiments/{args.experiment}/activations")

    # Determine output dirs based on condition
    if args.condition == "gemma-2-2b":
        # Default case - use simple names
        human_dir = output_dir / "human"
        model_dir = output_dir / "model"
    else:
        human_dir = output_dir / f"human-{args.condition}"
        model_dir = output_dir / f"model-{args.condition}"

    print(f"Capturing activations for {len(samples)} samples...")
    print(f"  Human dir: {human_dir}")
    print(f"  Model dir: {model_dir}")

    for sample in tqdm(samples):
        sample_id = sample["id"]
        first_sentence = sample["first_sentence"]

        # Human condition: prefill full human text
        if not args.model_only:
            human_dir.mkdir(parents=True, exist_ok=True)
            human_data = capture_residual_stream_prefill(
                model, tokenizer,
                prompt_text=first_sentence,
                response_text=sample["human_continuation"],
                n_layers=n_layers,
            )
            torch.save(human_data, human_dir / f"{sample_id}.pt")

        # Model condition: prefill model-generated text
        if not args.human_only:
            model_dir.mkdir(parents=True, exist_ok=True)
            model_data = capture_residual_stream_prefill(
                model, tokenizer,
                prompt_text=first_sentence,
                response_text=sample["model_continuation"],
                n_layers=n_layers,
            )
            torch.save(model_data, model_dir / f"{sample_id}.pt")

    print(f"Saved activations to {output_dir}")

if __name__ == "__main__":
    main()
