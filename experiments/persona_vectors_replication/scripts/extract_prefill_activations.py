#!/usr/bin/env python3
"""
Extract activations from identical prefilled text on both base and instruct models.
Uses existing capture_residual_stream_prefill() which handles:
- MultiLayerCapture for all layers in one forward pass
- Proper prompt/response tokenization
- Correct activation splitting
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.model import load_model_with_lora, format_prompt
from inference.capture_raw_activations import capture_residual_stream_prefill

# Configuration
N_TOKENS = 10  # First 10 tokens of each response
TRAITS = ['evil', 'sycophancy', 'hallucination']
MODELS = {
    'base': 'meta-llama/Llama-3.1-8B',
    'instruct': 'meta-llama/Llama-3.1-8B-Instruct'
}

def get_prefill_text(response: str, tokenizer, n_tokens: int) -> str:
    """Extract first n_tokens from response as prefill text."""
    tokens = tokenizer.encode(response, add_special_tokens=False)[:n_tokens]
    return tokenizer.decode(tokens)


def main():
    exp_dir = Path('experiments/persona_vectors_replication')
    out_dir = exp_dir / 'concept_rotation'
    out_dir.mkdir(exist_ok=True)

    # Use instruct tokenizer for consistent token boundaries
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

    for model_key, model_name in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Loading {model_key} model: {model_name}")
        print(f"{'='*60}")

        model, model_tokenizer = load_model_with_lora(model_name)
        model = model.eval()
        n_layers = model.config.num_hidden_layers
        use_chat_template = model_tokenizer.chat_template is not None

        for trait in TRAITS:
            for polarity in ['pos', 'neg']:
                print(f"\n  Processing {trait}/{polarity}...")

                # Load extraction responses
                responses_path = exp_dir / f'extraction/pv_instruction/{trait}/instruct/responses/{polarity}.json'
                with open(responses_path) as f:
                    data = json.load(f)

                all_activations = []

                for item in tqdm(data, desc=f"{model_key}/{trait}/{polarity}"):
                    prompt = item['prompt']
                    response = item['response']

                    # Get first N tokens as prefill
                    prefill_text = get_prefill_text(response, tokenizer, N_TOKENS)

                    # Format prompt appropriately for this model
                    formatted_prompt = format_prompt(prompt, model_tokenizer, use_chat_template=use_chat_template)

                    # Capture using existing infrastructure
                    result = capture_residual_stream_prefill(
                        model, model_tokenizer,
                        prompt_text=formatted_prompt,
                        response_text=prefill_text,
                        n_layers=n_layers
                    )

                    # Extract response activations and mean-pool over tokens
                    # This avoids variable-length issues (some responses are < N_TOKENS)
                    # Shape per layer: [n_tokens, hidden_dim] -> mean -> [hidden_dim]
                    response_acts = []
                    for layer_idx in range(n_layers):
                        layer_act = result['response']['activations'][layer_idx]['residual']
                        # Mean over tokens: [n_tokens, hidden_dim] -> [hidden_dim]
                        layer_mean = layer_act.mean(dim=0)
                        response_acts.append(layer_mean)

                    # Stack: [n_layers, hidden_dim]
                    sample_acts = torch.stack(response_acts)
                    all_activations.append(sample_acts.cpu())

                # Stack all samples: [n_samples, n_layers, hidden_dim]
                stacked = torch.stack(all_activations)

                # Save
                out_path = out_dir / f'{trait}_{polarity}_{model_key}.pt'
                torch.save(stacked, out_path)
                print(f"    Saved {out_path}: {stacked.shape}")

                # Clear memory
                del all_activations, stacked
                torch.cuda.empty_cache()

        # Unload model before loading next
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Done! Activations saved to concept_rotation/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
