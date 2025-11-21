#!/usr/bin/env python3
"""
Simplified Batched Generation with Activation Extraction

Generates responses in batches for massive speedup (3-7x faster).
Captures activations during a forward pass after generation.

Usage:
    # Single trait with batching
    python extraction/1_generate_batched_simple.py \
        --experiment gemma_2b_cognitive_nov20 \
        --trait retrieval_construction \
        --batch_size 8

    # Multiple traits
    python extraction/1_generate_batched_simple.py \
        --experiment gemma_2b_cognitive_nov20 \
        --traits retrieval_construction,serial_parallel,local_global \
        --batch_size 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import asyncio
import pandas as pd
import torch
from typing import List, Optional
from datetime import datetime
from tqdm import tqdm
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.judge import OpenAiJudge
from utils.config import setup_credentials
from extraction.utils_batch import generate_batch_with_activations, get_activations_from_texts


def infer_gen_model(experiment_name: str) -> str:
    """Infer generation model from experiment name."""
    if "gemma_2b" in experiment_name.lower():
        return "google/gemma-2-2b-it"
    elif "gemma_9b" in experiment_name.lower():
        return "google/gemma-2-9b-it"
    elif "llama_8b" in experiment_name.lower():
        return "meta-llama/Llama-3.1-8B-Instruct"
    else:
        return "google/gemma-2-2b-it"  # Default


def load_trait_definition(trait_dir: Path) -> dict:
    """Load trait definition JSON."""
    trait_def_path = trait_dir / "extraction" / "trait_definition.json"
    if not trait_def_path.exists():
        raise ValueError(f"Trait definition not found: {trait_def_path}")

    with open(trait_def_path) as f:
        return json.load(f)


async def judge_responses_async(judge: OpenAiJudge, questions: List[str], responses: List[str]) -> List[float]:
    """Judge all responses asynchronously."""
    tasks = []
    for question, response in zip(questions, responses):
        task = judge.judge(question=question, answer=response)
        tasks.append(task)

    scores = await asyncio.gather(*tasks)
    return scores


def generate_batched_simple(
    experiment: str,
    trait: Optional[str] = None,
    traits: Optional[str] = None,
    gen_model: Optional[str] = None,
    judge_model: str = "gpt-5-mini",
    n_examples: int = 100,  # Per polarity
    batch_size: int = 8,  # Batch size for generation
    device: str = "cuda",
    max_new_tokens: int = 150,
    threshold: int = 50,
    extract_activations: bool = True,  # Whether to extract activations
):
    """
    Generate responses with batched generation for speedup.

    Speedup expectations:
    - batch_size=1: 1x (baseline, ~20 min for 200 examples)
    - batch_size=4: ~3x faster (~7 min)
    - batch_size=8: ~5x faster (~4 min)
    - batch_size=16: ~7x faster (~3 min, needs more memory)

    Args:
        experiment: Experiment name
        trait: Single trait (mutually exclusive with traits)
        traits: Comma-separated traits (mutually exclusive with trait)
        gen_model: Generation model (auto-inferred if not provided)
        judge_model: Judge model (default: gpt-5-mini)
        n_examples: Number of examples per polarity
        batch_size: Batch size for generation (higher = faster but more memory)
        device: Device for generation
        max_new_tokens: Max tokens to generate
        threshold: Threshold for filtering responses
        extract_activations: Whether to extract activations
    """
    # Setup
    config = setup_credentials()

    if trait and traits:
        raise ValueError("Specify either 'trait' or 'traits', not both")

    trait_list = [trait] if trait else traits.split(",")

    exp_dir = Path(f"experiments/{experiment}")
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Infer model
    if gen_model is None:
        gen_model = infer_gen_model(experiment)
        print(f"Inferred generation model: {gen_model}")

    # Load model
    print(f"Loading model: {gen_model}")
    model = AutoModelForCausalLM.from_pretrained(
        gen_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(gen_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get model info
    if "gemma" in gen_model.lower():
        n_layers = len(model.model.layers) + 1  # +1 for embedding layer
        hidden_dim = model.config.hidden_size
    elif "llama" in gen_model.lower():
        n_layers = len(model.model.layers) + 1
        hidden_dim = model.config.hidden_size
    else:
        # Generic fallback
        n_layers = model.config.num_hidden_layers + 1
        hidden_dim = model.config.hidden_size

    print(f"Model: {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"Judge: {judge_model}")
    print(f"Batch size: {batch_size}")
    print(f"Examples per polarity: {n_examples}")
    print(f"Extract activations: {extract_activations}")
    print()

    # Process each trait
    for trait_name in trait_list:
        trait_dir = exp_dir / trait_name
        trait_dir.mkdir(exist_ok=True)

        print(f"{'='*60}")
        print(f"Processing: {trait_name}")
        print(f"{'='*60}")

        # Load trait definition
        trait_def = load_trait_definition(trait_dir)
        instructions = trait_def["instruction"]
        questions = trait_def["questions"]
        eval_prompt = trait_def["eval_prompt"]

        # Setup judge
        judge = OpenAiJudge(
            model=judge_model,
            prompt_template=eval_prompt,
            eval_type="0_100"
        )

        # Create output directories
        trait_dir.mkdir(parents=True, exist_ok=True)
        (trait_dir / "responses").mkdir(exist_ok=True)
        if extract_activations:
            (trait_dir / "activations").mkdir(exist_ok=True)

        # Store activations for both polarities
        all_activations = {'pos': [], 'neg': []}

        # Process both polarities
        for polarity in ['pos', 'neg']:
            print(f"\n{polarity.upper()} examples:")

            # Prepare all prompts
            all_prompts = []
            all_questions = []
            all_instructions = []

            examples_per_instruction = n_examples // len(instructions)
            extra = n_examples % len(instructions)

            for inst_idx, inst_pair in enumerate(instructions):
                instruction = inst_pair[polarity]
                n_for_this_inst = examples_per_instruction + (1 if inst_idx < extra else 0)

                for q_idx in range(n_for_this_inst):
                    question = questions[q_idx % len(questions)]
                    prompt = f"{instruction}\n\n{question}"

                    all_prompts.append(prompt)
                    all_questions.append(question)
                    all_instructions.append(instruction)

            # Generate responses in batches
            print(f"  Generating {len(all_prompts)} responses in batches of {batch_size}...")
            all_responses = []

            for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"  {polarity} generation"):
                batch_prompts = all_prompts[i:i + batch_size]

                # Generate batch (without activations during generation)
                batch_responses, _ = generate_batch_with_activations(
                    model, tokenizer, batch_prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    capture_layers=None  # Don't capture during generation
                )

                all_responses.extend(batch_responses)

            # Create data records
            all_data = []
            for i in range(len(all_prompts)):
                all_data.append({
                    'question': all_questions[i],
                    'instruction': all_instructions[i],
                    'prompt': all_prompts[i],
                    'response': all_responses[i]
                })

            # Judge responses
            print(f"  Judging {len(all_responses)} responses with {judge_model}...")
            scores = asyncio.run(judge_responses_async(judge, all_questions, all_responses))

            # Filter by threshold
            filtered_data = []
            filtered_responses = []

            for i, score in enumerate(scores):
                score_val = score if score is not None else 0.0
                all_data[i]['trait_score'] = score_val

                # Apply threshold filtering
                if polarity == 'pos' and score_val >= threshold:
                    filtered_data.append(all_data[i])
                    filtered_responses.append(all_responses[i])
                elif polarity == 'neg' and score_val < threshold:
                    filtered_data.append(all_data[i])
                    filtered_responses.append(all_responses[i])

            print(f"  Generated: {len(all_data)} → Filtered: {len(filtered_data)} (threshold={threshold})")

            # Save responses
            df = pd.DataFrame(filtered_data)
            csv_path = trait_dir / "responses" / f"{polarity}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved responses: {csv_path}")

            # Extract activations if requested
            if extract_activations and len(filtered_responses) > 0:
                print(f"  Extracting activations from {len(filtered_responses)} filtered responses...")
                layers = list(range(n_layers))

                # Extract activations in batches
                acts_dict = get_activations_from_texts(
                    model, tokenizer, filtered_responses,
                    layers, batch_size=batch_size, device=device
                )

                # Stack into tensor: [n_examples, n_layers, hidden_dim]
                acts_tensor = torch.stack([acts_dict[layer] for layer in layers], dim=1)

                # Save this polarity's activations
                acts_path = trait_dir / "activations" / f"{polarity}_acts.pt"
                torch.save(acts_tensor, acts_path)
                print(f"  Saved activations: {acts_path} (shape: {acts_tensor.shape})")

                all_activations[polarity] = acts_tensor

        # Combine activations from both polarities
        if extract_activations and len(all_activations['pos']) > 0 and len(all_activations['neg']) > 0:
            print(f"\nCombining activations...")
            pos_acts = all_activations['pos']
            neg_acts = all_activations['neg']

            # Combine
            combined_acts = torch.cat([pos_acts, neg_acts], dim=0)

            # Save combined
            combined_path = trait_dir / "activations" / "all_layers.pt"
            torch.save(combined_acts, combined_path)
            print(f"  Saved combined: {combined_path} (shape: {combined_acts.shape})")

            # Save metadata
            metadata = {
                "model": gen_model,
                "trait": trait_name,
                "n_examples": combined_acts.shape[0],
                "n_examples_pos": pos_acts.shape[0],
                "n_examples_neg": neg_acts.shape[0],
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
                "threshold": threshold,
                "batch_size": batch_size,
                "storage_type": "token_averaged",
                "extraction_date": datetime.now().isoformat(),
                "batched_generation": True
            }

            metadata_path = trait_dir / "activations" / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"  Saved metadata: {metadata_path}")

        print(f"\n✅ {trait_name} complete!")
        print(f"  Time estimate: ~{200/batch_size:.0f} forward passes instead of 200")
        print(f"  Speedup: ~{batch_size}x faster than serial generation")
        print()


if __name__ == "__main__":
    fire.Fire(generate_batched_simple)