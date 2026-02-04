#!/usr/bin/env python3
"""Extract Alpaca control activations for FPR calibration.

Two-phase approach:
  1. Generate responses for Alpaca prompts (model.generate)
  2. Prefill forward pass to capture mean-pooled response activations

Input:
    experiments/{experiment}/prompt_sets/alpaca_control.json

Output:
    experiments/{experiment}/results/alpaca_activations.pt  [n, n_layers, hidden_dim]
    experiments/{experiment}/results/alpaca_responses.json
    experiments/{experiment}/results/alpaca_metadata.json

Usage:
    python experiments/bullshit/scripts/extract_alpaca_activations.py
    python experiments/bullshit/scripts/extract_alpaca_activations.py --max-new-tokens 64
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import argparse
import json
import gc
import torch
from tqdm import tqdm

from utils.model import load_model, format_prompt, pad_sequences, get_num_layers
from utils.paths import load_experiment_config
from core import MultiLayerCapture


def generate_responses(model, tokenizer, prompts, max_new_tokens, batch_size):
    """Generate responses for all prompts. Returns list of response strings."""
    responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]

        # Format with chat template
        formatted = [
            format_prompt(p, tokenizer, use_chat_template=True)
            for p in batch_prompts
        ]

        # Tokenize with left padding for batched generation
        has_bos = tokenizer.bos_token and formatted[0].startswith(tokenizer.bos_token)
        encodings = tokenizer(
            formatted,
            add_special_tokens=not has_bos,
            padding=True,
            return_tensors='pt',
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only new tokens
        prompt_len = encodings['input_ids'].shape[1]
        for j in range(len(batch_prompts)):
            new_tokens = outputs[j][prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)

    return responses


def extract_activations(model, tokenizer, prompts, responses, n_layers, batch_size):
    """Prefill forward pass to capture mean-pooled response activations.

    Same approach as extraction/extract_activations.py: tokenize prompt+response,
    find response token boundaries, forward pass with MultiLayerCapture, mean-pool.
    """
    from extraction.extract_activations import resolve_position

    all_activations = {layer: [] for layer in range(n_layers)}
    position = 'response[:]'

    # Prepare all items
    formatted_prompts = [
        format_prompt(p, tokenizer, use_chat_template=True)
        for p in prompts
    ]
    texts = [fp + r for fp, r in zip(formatted_prompts, responses)]

    has_bos = tokenizer.bos_token and texts[0].startswith(tokenizer.bos_token)
    all_input_ids_raw = tokenizer(texts, add_special_tokens=not has_bos, padding=False)['input_ids']
    all_input_ids = [torch.tensor(ids) for ids in all_input_ids_raw]

    prompt_has_bos = tokenizer.bos_token and formatted_prompts[0].startswith(tokenizer.bos_token)
    prompt_ids = tokenizer(formatted_prompts, add_special_tokens=not prompt_has_bos, padding=False)['input_ids']
    prompt_lens = [len(ids) for ids in prompt_ids]

    items = []
    for i, (input_ids, prompt_len) in enumerate(zip(all_input_ids, prompt_lens)):
        seq_len = len(input_ids)
        start_idx, end_idx = resolve_position(position, prompt_len, seq_len)
        if start_idx >= end_idx:
            continue
        items.append({
            'index': i,
            'input_ids': input_ids,
            'seq_len': seq_len,
            'start_idx': start_idx,
            'end_idx': end_idx,
        })

    print(f"  {len(items)}/{len(prompts)} items have valid response tokens")

    # Process in batches
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    i = 0
    pbar = tqdm(total=len(items), desc="Extracting activations")

    while i < len(items):
        batch_items = items[i:i + batch_size]

        try:
            batch = pad_sequences([item['input_ids'] for item in batch_items], pad_token_id)
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            pad_offsets = batch['pad_offsets']

            with MultiLayerCapture(model, component='residual', keep_on_gpu=True) as capture:
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

            for layer in range(n_layers):
                acts = capture.get(layer)
                if acts is None:
                    continue

                batch_acts = []
                for b, item in enumerate(batch_items):
                    pad_offset = pad_offsets[b]
                    start_idx = item['start_idx'] + pad_offset
                    end_idx = item['end_idx'] + pad_offset
                    selected = acts[b, start_idx:end_idx, :]
                    act_out = selected.mean(dim=0) if selected.shape[0] > 1 else selected.squeeze(0)
                    batch_acts.append(act_out)

                batch_tensor = torch.stack(batch_acts).cpu()
                for act in batch_tensor:
                    all_activations[layer].append(act)

            pbar.update(len(batch_items))
            i += batch_size

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_size == 1:
                raise RuntimeError("OOM even with batch_size=1")
            batch_size = max(1, batch_size // 2)
            print(f"\n  OOM, reducing batch_size to {batch_size}")

    pbar.close()

    # Stack into tensor [n_items, n_layers, hidden_dim]
    for layer in range(n_layers):
        all_activations[layer] = torch.stack(all_activations[layer])

    acts_tensor = torch.stack([all_activations[l] for l in range(n_layers)], dim=1)
    return acts_tensor


def main():
    parser = argparse.ArgumentParser(description='Extract Alpaca control activations')
    parser.add_argument('--experiment', default='bullshit')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--gen-batch-size', type=int, default=8)
    parser.add_argument('--extract-batch-size', type=int, default=4)
    args = parser.parse_args()

    # Load config
    config = load_experiment_config(args.experiment)
    model_name = config['model_variants']['instruct']['model']

    # Load Alpaca prompts
    prompt_path = ROOT / 'experiments' / args.experiment / 'prompt_sets' / 'alpaca_control.json'
    with open(prompt_path) as f:
        alpaca = json.load(f)
    prompts = [item['text'] for item in alpaca['prompts']]
    print(f"Loaded {len(prompts)} Alpaca prompts")

    # Load model
    model, tokenizer = load_model(model_name)
    n_layers = get_num_layers(model)

    # Phase 1: Generate responses
    print(f"\nPhase 1: Generating responses (max_new_tokens={args.max_new_tokens})...")
    responses = generate_responses(model, tokenizer, prompts, args.max_new_tokens, args.gen_batch_size)
    response_lens = [len(r.split()) for r in responses]
    print(f"  Generated {len(responses)} responses (median {sorted(response_lens)[len(response_lens)//2]} words)")

    # Save responses
    output_dir = ROOT / 'experiments' / args.experiment / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    response_data = [
        {'prompt': p, 'response': r}
        for p, r in zip(prompts, responses)
    ]
    with open(output_dir / 'alpaca_responses.json', 'w') as f:
        json.dump(response_data, f, indent=2)
    print(f"  Saved responses: {output_dir / 'alpaca_responses.json'}")

    # Free generation KV cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 2: Extract activations
    print(f"\nPhase 2: Extracting activations ({n_layers} layers, response[:] mean-pooled)...")
    acts = extract_activations(model, tokenizer, prompts, responses, n_layers, args.extract_batch_size)
    print(f"  Activations shape: {acts.shape}")

    # Save
    torch.save(acts, output_dir / 'alpaca_activations.pt')
    metadata = {
        'n_examples': len(prompts),
        'n_layers': n_layers,
        'hidden_dim': acts.shape[2],
        'model': model_name,
        'max_new_tokens': args.max_new_tokens,
        'position': 'response[:]',
        'component': 'residual',
        'median_response_words': sorted(response_lens)[len(response_lens) // 2],
    }
    with open(output_dir / 'alpaca_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved: {output_dir / 'alpaca_activations.pt'} ({acts.shape})")
    print(f"  Saved: {output_dir / 'alpaca_metadata.json'}")
    print("\nDone! Now re-run evaluate_liars_bench_protocol.py --alpaca to get Alpaca-calibrated metrics.")


if __name__ == '__main__':
    main()
