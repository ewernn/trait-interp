"""
Residual stream capture via prefill forward pass.

Input:
    - model, tokenizer: Loaded transformer model and tokenizer
    - prompt_text, response_text: Formatted prompt and response strings

Output:
    - Dict with 'prompt' and 'response' keys containing tokens, text, and activations

Usage:
    from utils.capture import capture_residual_stream_prefill

    data = capture_residual_stream_prefill(model, tokenizer, prompt_text, response_text, n_layers)
"""

from contextlib import ExitStack
from typing import Dict, List, Optional

import torch

from utils.model import tokenize

DEFAULT_COMPONENTS = ['residual', 'attn_contribution']


def capture_residual_stream_prefill(model, tokenizer, prompt_text: str, response_text: str,
                                     n_layers: int, components: List[str] = None,
                                     capture_attention: bool = False,
                                     layers: List[int] = None,
                                     # Legacy aliases
                                     capture_mlp: bool = False) -> Dict:
    """
    Capture residual stream activations with prefilled response (single forward pass).

    Concatenates prompt + response tokens and runs one forward pass, splitting
    activations at the prompt/response boundary. Uses MultiLayerCapture to nest
    all component hooks in a single pass.

    Args:
        model: Loaded transformer model
        tokenizer: Model tokenizer
        prompt_text: Formatted prompt string (already has chat template applied)
        response_text: Response text to prefill
        n_layers: Total number of model layers
        components: List of components to capture (default: ['residual', 'attn_contribution'])
        capture_attention: Capture attention patterns (weight matrices, not attn_contribution)
        layers: Subset of layers to capture (None = all)
        capture_mlp: Legacy alias — adds 'mlp_contribution' to components
    """
    from core import MultiLayerCapture

    # Resolve components
    if components is None:
        components = list(DEFAULT_COMPONENTS)
    if capture_mlp and 'mlp_contribution' not in components:
        components.append('mlp_contribution')

    # Tokenize prompt
    prompt_inputs = tokenize(prompt_text, tokenizer).to(model.device)
    n_prompt_tokens = prompt_inputs['input_ids'].shape[1]
    prompt_token_ids = prompt_inputs['input_ids'][0].tolist()
    prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_token_ids]

    # Tokenize response (without special tokens - it's appended to prompt)
    response_inputs = tokenize(response_text, tokenizer, add_special_tokens=False).to(model.device)
    response_token_ids = response_inputs['input_ids'][0].tolist()
    response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]

    # Concatenate for single forward pass
    full_input_ids = torch.cat([prompt_inputs['input_ids'], response_inputs['input_ids']], dim=1)

    # Capture all components in one forward pass using ExitStack + MultiLayerCapture
    with ExitStack() as stack:
        captures = {}
        for component in components:
            cap = stack.enter_context(MultiLayerCapture(model, component=component, layers=layers))
            captures[component] = cap

        with torch.no_grad():
            outputs = model(input_ids=full_input_ids, output_attentions=capture_attention, return_dict=True)

        component_acts = {}
        for component, cap in captures.items():
            component_acts[component] = cap.get_all()

    # Split activations into prompt/response portions
    prompt_acts = {}
    response_acts = {}
    layer_indices = layers if layers is not None else list(range(n_layers))
    for layer_idx in layer_indices:
        prompt_acts[layer_idx] = {}
        response_acts[layer_idx] = {}

        for component in components:
            acts = component_acts.get(component, {})
            if layer_idx in acts:
                full = acts[layer_idx].squeeze(0)
                prompt_acts[layer_idx][component] = full[:n_prompt_tokens]
                response_acts[layer_idx][component] = full[n_prompt_tokens:]

    # Split attention patterns (only if captured)
    prompt_attention = {}
    response_attention = []
    if capture_attention and outputs.attentions is not None:
        for i, attn in enumerate(outputs.attentions):
            attn_avg = attn[0].mean(dim=0).detach().cpu()  # [seq_len, seq_len]
            prompt_attention[f'layer_{i}'] = attn_avg[:n_prompt_tokens, :n_prompt_tokens]
            # Response attention: each token's attention over full context
            for t in range(n_prompt_tokens, attn_avg.shape[0]):
                if i == 0:
                    response_attention.append({})
                response_attention[t - n_prompt_tokens][f'layer_{i}'] = attn_avg[t, :t+1]

    return {
        'prompt': {'text': prompt_text, 'tokens': prompt_tokens, 'token_ids': prompt_token_ids,
                   'activations': prompt_acts, 'attention': prompt_attention},
        'response': {'text': response_text, 'tokens': response_tokens, 'token_ids': response_token_ids,
                     'activations': response_acts, 'attention': response_attention}
    }
