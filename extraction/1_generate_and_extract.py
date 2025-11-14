#!/usr/bin/env python3
"""
Combined Stage: Generate Responses AND Extract Activations

Captures activations during the generation forward pass for efficiency.
Saves ~50% compute time by avoiding duplicate forward passes.

Usage:
    python extraction/1_generate_and_extract.py \
        --experiment gemma_2b_cognitive_nov20 \
        --traits retrieval_construction,serial_parallel
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import asyncio
import pandas as pd
import torch
from typing import List, Optional, Dict
from datetime import datetime
from tqdm import tqdm
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.judge import OpenAiJudge
from utils.config import setup_credentials
from traitlens import HookManager, ActivationCapture


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
    trait_def_path = trait_dir / "trait_definition.json"
    if not trait_def_path.exists():
        raise ValueError(f"Trait definition not found: {trait_def_path}")

    with open(trait_def_path) as f:
        return json.load(f)


def get_layer_names(model, model_type: str) -> List[str]:
    """Get layer hook names for the model."""
    if "gemma" in model_type.lower():
        # Gemma uses model.layers.{i}
        n_layers = len(model.model.layers)
        return [f"model.layers.{i}" for i in range(n_layers)]
    elif "llama" in model_type.lower():
        # Llama uses model.layers.{i}
        n_layers = len(model.model.layers)
        return [f"model.layers.{i}" for i in range(n_layers)]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_and_extract(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    capture_layers: Optional[List[str]] = None
) -> tuple[str, Dict[int, torch.Tensor]]:
    """
    Generate response AND capture activations in a single forward pass.

    Returns:
        response: Generated text
        activations: Dict mapping layer_idx to activation tensor
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if capture_layers:
        # Setup hooks to capture activations during generation
        capture = ActivationCapture()
        with HookManager(model) as hooks:
            # Add hooks for each layer
            for i, layer_name in enumerate(capture_layers):
                hooks.add_forward_hook(layer_name, capture.make_hook(f"layer_{i}"))

            # Generate (this triggers the forward pass)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    output_hidden_states=False  # We're using hooks instead
                )

        # Extract activations and average over sequence length
        activations = {}
        for i in range(len(capture_layers)):
            acts = capture.get(f"layer_{i}")  # [1, seq_len, hidden_dim]
            # Average over sequence length
            activations[i] = acts.mean(dim=1).squeeze(0).cpu()  # [hidden_dim]
    else:
        # Just generate without capturing activations
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        activations = None

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip(), activations


async def judge_responses_async(judge: OpenAiJudge, questions: List[str], responses: List[str]) -> List[float]:
    """Judge all responses asynchronously."""
    tasks = []
    for question, response in zip(questions, responses):
        task = judge.judge(question=question, answer=response)
        tasks.append(task)

    scores = await asyncio.gather(*tasks)
    return scores


def generate_and_extract_all(
    experiment: str,
    trait: Optional[str] = None,
    traits: Optional[str] = None,
    gen_model: Optional[str] = None,
    judge_model: str = "gpt-5-mini",
    n_examples: int = 100,  # Per polarity
    device: str = "cuda",
    max_new_tokens: int = 150,
    threshold: int = 50,  # For filtering
):
    """
    Generate responses AND extract activations in one pass.

    This is ~2x faster than running generation and extraction separately.
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
        device_map="auto",
        cache_dir=".cache"
    )
    tokenizer = AutoTokenizer.from_pretrained(gen_model, cache_dir=".cache")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get layer names for hooks
    layer_names = get_layer_names(model, gen_model)
    n_layers = len(layer_names)

    # Get hidden dimension
    if "gemma" in gen_model.lower():
        hidden_dim = model.config.hidden_size
    elif "llama" in gen_model.lower():
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = model.config.hidden_size

    print(f"Model has {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"Judge model: {judge_model}")
    print(f"Examples per polarity: {n_examples}")
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
        (trait_dir / "responses").mkdir(exist_ok=True)
        (trait_dir / "activations").mkdir(exist_ok=True)

        # Process both polarities
        for polarity in ['pos', 'neg']:
            print(f"\n{polarity.upper()} examples:")
            print(f"  Generating and extracting...")

            all_data = []
            all_activations = []

            # Distribute examples across instructions
            examples_per_instruction = n_examples // len(instructions)
            extra = n_examples % len(instructions)

            for inst_idx, inst_pair in enumerate(tqdm(instructions, desc=f"{polarity} instructions")):
                instruction = inst_pair[polarity]

                # Number of examples for this instruction
                n_for_this_inst = examples_per_instruction + (1 if inst_idx < extra else 0)

                for q_idx in range(n_for_this_inst):
                    question = questions[q_idx % len(questions)]
                    prompt = f"{instruction}\n\n{question}"

                    # Generate AND extract activations in one pass!
                    response, activations = generate_and_extract(
                        model, tokenizer, prompt, max_new_tokens, layer_names
                    )

                    all_data.append({
                        'question': question,
                        'instruction': instruction,
                        'prompt': prompt,
                        'response': response
                    })

                    # Stack layer activations: [n_layers, hidden_dim]
                    if activations:
                        layer_acts = torch.stack([activations[i] for i in range(n_layers)])
                        all_activations.append(layer_acts)

            # Judge responses
            print(f"  Judging with {judge_model}...")
            questions_list = [d['question'] for d in all_data]
            responses_list = [d['response'] for d in all_data]

            scores = asyncio.run(judge_responses_async(judge, questions_list, responses_list))

            # Add scores and filter
            filtered_data = []
            filtered_activations = []

            for i, score in enumerate(scores):
                score_val = score if score is not None else 0.0
                all_data[i]['trait_score'] = score_val

                # Apply threshold filtering
                if polarity == 'pos' and score_val >= threshold:
                    filtered_data.append(all_data[i])
                    if all_activations:
                        filtered_activations.append(all_activations[i])
                elif polarity == 'neg' and score_val < threshold:
                    filtered_data.append(all_data[i])
                    if all_activations:
                        filtered_activations.append(all_activations[i])

            print(f"  Generated: {len(all_data)} → Filtered: {len(filtered_data)}")

            # Save responses
            df = pd.DataFrame(filtered_data)
            csv_path = trait_dir / "responses" / f"{polarity}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved responses: {csv_path}")

            # Save activations
            if filtered_activations:
                # Stack: [n_examples, n_layers, hidden_dim]
                acts_tensor = torch.stack(filtered_activations)
                acts_path = trait_dir / "activations" / f"{polarity}_acts.pt"
                torch.save(acts_tensor, acts_path)
                print(f"  Saved activations: {acts_path} ({acts_tensor.shape})")

        # Combine pos and neg activations
        print(f"\nCombining activations...")
        pos_acts = torch.load(trait_dir / "activations" / "pos_acts.pt")
        neg_acts = torch.load(trait_dir / "activations" / "neg_acts.pt")

        all_acts = torch.cat([pos_acts, neg_acts], dim=0)

        # Save combined
        combined_path = trait_dir / "activations" / "all_layers.pt"
        torch.save(all_acts, combined_path)

        # Save metadata
        metadata = {
            "model": gen_model,
            "trait": trait_name,
            "n_examples": all_acts.shape[0],
            "n_examples_pos": pos_acts.shape[0],
            "n_examples_neg": neg_acts.shape[0],
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "threshold": threshold,
            "storage_type": "token_averaged",
            "extraction_date": datetime.now().isoformat(),
            "combined_generation": True  # Flag showing this was done efficiently!
        }

        metadata_path = trait_dir / "activations" / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ {trait_name} complete!")
        print(f"  Total examples: {all_acts.shape[0]}")
        print(f"  Pos: {pos_acts.shape[0]}, Neg: {neg_acts.shape[0]}")
        print()


if __name__ == "__main__":
    fire.Fire(generate_and_extract_all)