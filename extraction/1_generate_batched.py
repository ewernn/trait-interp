#!/usr/bin/env python3
"""
Batched Generation + Activation Extraction

Generates responses in batches for massive speedup.
Can be 4-8x faster than serial generation.

Usage:
    python extraction/1_generate_batched.py \
        --experiment gemma_2b_cognitive_nov20 \
        --traits retrieval_construction \
        --gen_batch_size 8
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
        return "google/gemma-2-2b-it"


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
        n_layers = len(model.model.layers)
        return [f"model.layers.{i}" for i in range(n_layers)]
    elif "llama" in model_type.lower():
        n_layers = len(model.model.layers)
        return [f"model.layers.{i}" for i in range(n_layers)]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 150,
    capture_activations: bool = True,
    layer_names: Optional[List[str]] = None
) -> tuple[List[str], Optional[torch.Tensor]]:
    """
    Generate responses for a batch of prompts.

    Returns:
        responses: List of generated texts
        activations: Tensor of shape [batch_size, n_layers, hidden_dim] if capture_activations=True
    """
    # Tokenize batch with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,  # Pad to same length
        truncation=True,
        max_length=512
    ).to(model.device)

    if capture_activations and layer_names:
        # Setup hooks for batch processing
        captured_acts = {}

        def make_batch_hook(layer_idx):
            def hook(module, input, output):
                # output is [batch_size, seq_len, hidden_dim]
                # Average over sequence length
                captured_acts[layer_idx] = output[0].mean(dim=1).detach().cpu()
            return hook

        # Register hooks
        hooks = []
        for i, layer_name in enumerate(layer_names):
            module = model
            for part in layer_name.split('.'):
                module = getattr(module, part)
            hook = module.register_forward_hook(make_batch_hook(i))
            hooks.append(hook)

        # Generate with hooks active
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1
            )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Stack activations: [batch_size, n_layers, hidden_dim]
        if captured_acts:
            batch_acts = torch.stack([
                torch.stack([captured_acts[i][j] for i in range(len(layer_names))])
                for j in range(len(prompts))
            ])
        else:
            batch_acts = None
    else:
        # Just generate without capturing
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        batch_acts = None

    # Decode responses
    responses = []
    for i in range(len(prompts)):
        # Extract response for this example (skip input tokens)
        input_length = inputs['input_ids'][i].nonzero()[-1].item() + 1
        response = tokenizer.decode(
            outputs[i][input_length:],
            skip_special_tokens=True
        )
        responses.append(response.strip())

    return responses, batch_acts


async def judge_responses_async(judge: OpenAiJudge, questions: List[str], responses: List[str]) -> List[float]:
    """Judge all responses asynchronously."""
    tasks = []
    for question, response in zip(questions, responses):
        task = judge.judge(question=question, answer=response)
        tasks.append(task)

    scores = await asyncio.gather(*tasks)
    return scores


def generate_batched(
    experiment: str,
    trait: Optional[str] = None,
    traits: Optional[str] = None,
    gen_model: Optional[str] = None,
    judge_model: str = "gpt-5-mini",
    n_examples: int = 100,  # Per polarity
    gen_batch_size: int = 8,  # BATCH SIZE FOR GENERATION!
    device: str = "cuda",
    max_new_tokens: int = 150,
    threshold: int = 50,
):
    """
    Generate responses with BATCHED generation for massive speedup.

    With batch_size=8:
    - 200 examples takes ~25 forward passes instead of 200
    - Can be 4-8x faster depending on model and batch size
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

    # Get layer info
    layer_names = get_layer_names(model, gen_model)
    n_layers = len(layer_names)

    if "gemma" in gen_model.lower():
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = model.config.hidden_size

    print(f"Model: {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"Judge: {judge_model}")
    print(f"Generation batch size: {gen_batch_size}")  # NEW!
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

            # Prepare all prompts first
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

            # Process in batches!
            print(f"  Generating {len(all_prompts)} responses in batches of {gen_batch_size}...")

            all_responses = []
            all_activations = []

            for i in tqdm(range(0, len(all_prompts), gen_batch_size)):
                batch_prompts = all_prompts[i:i + gen_batch_size]

                # BATCHED GENERATION!
                batch_responses, batch_acts = generate_batch(
                    model, tokenizer, batch_prompts,
                    max_new_tokens, True, layer_names
                )

                all_responses.extend(batch_responses)

                if batch_acts is not None:
                    # batch_acts is [batch_size, n_layers, hidden_dim]
                    for j in range(len(batch_responses)):
                        all_activations.append(batch_acts[j])

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
            print(f"  Judging with {judge_model}...")
            scores = asyncio.run(judge_responses_async(judge, all_questions, all_responses))

            # Filter by threshold
            filtered_data = []
            filtered_activations = []

            for i, score in enumerate(scores):
                score_val = score if score is not None else 0.0
                all_data[i]['trait_score'] = score_val

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
                acts_tensor = torch.stack(filtered_activations)
                acts_path = trait_dir / "activations" / f"{polarity}_acts.pt"
                torch.save(acts_tensor, acts_path)
                print(f"  Saved activations: {acts_path} ({acts_tensor.shape})")

        # Combine activations
        print(f"\nCombining activations...")
        pos_acts = torch.load(trait_dir / "activations" / "pos_acts.pt")
        neg_acts = torch.load(trait_dir / "activations" / "neg_acts.pt")

        all_acts = torch.cat([pos_acts, neg_acts], dim=0)

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
            "generation_batch_size": gen_batch_size,
            "storage_type": "token_averaged",
            "extraction_date": datetime.now().isoformat(),
            "batched_generation": True
        }

        metadata_path = trait_dir / "activations" / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ {trait_name} complete!")
        print(f"  Total examples: {all_acts.shape[0]}")
        print(f"  Pos: {pos_acts.shape[0]}, Neg: {neg_acts.shape[0]}")
        print()


if __name__ == "__main__":
    fire.Fire(generate_batched)