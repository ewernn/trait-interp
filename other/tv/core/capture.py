"""Residual stream capture via prefill forward pass.

Input: model, tokenizer, prompt text, response text
Output: dict with 'prompt' and 'response' containing tokens, text, and activations

Usage:
    from core.capture import capture_prefill, capture_prefill_batch
    data = capture_prefill(model, tokenizer, prompt, response)
    batch_data = capture_prefill_batch(model, tokenizer, prompts, responses, batch_size=8)
"""

from typing import Dict, List, Optional

import torch

from core.model import tokenize, tokenize_batch
from core.hooks import MultiLayerCapture


def _pad_sequences(sequences, pad_token_id):
    """Left-pad 1-D tensors to same length.

    Returns:
        input_ids: [batch, max_len]
        attention_mask: [batch, max_len]
        pad_offsets: list of int (number of pad tokens per sample)
    """
    max_len = max(len(s) for s in sequences)
    input_ids, masks, pad_offsets = [], [], []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded = torch.cat([torch.full((pad_len,), pad_token_id, dtype=seq.dtype), seq])
        mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(len(seq), dtype=torch.long)])
        input_ids.append(padded)
        masks.append(mask)
        pad_offsets.append(pad_len)
    return torch.stack(input_ids), torch.stack(masks), pad_offsets


def capture_prefill(
    model,
    tokenizer,
    prompt: str,
    response: str,
    layers: List[int] = None,
) -> Dict:
    """Capture residual stream activations with prefilled response (single forward pass).

    Concatenates prompt + response tokens and runs one forward pass, splitting
    activations at the prompt/response boundary.

    Args:
        model: Loaded transformer model
        tokenizer: Model tokenizer
        prompt: Formatted prompt string (already has chat template applied)
        response: Response text to prefill
        layers: Subset of layers to capture (None = all)

    Returns:
        dict with 'prompt' and 'response' keys, each containing:
            text: original text
            tokens: list of decoded tokens
            token_ids: list of token ids
            activations: {layer: {"residual": Tensor[n_tokens, hidden_dim]}}
    """
    # Tokenize prompt
    prompt_inputs = tokenize(prompt, tokenizer).to(model.device)
    n_prompt_tokens = prompt_inputs['input_ids'].shape[1]
    prompt_token_ids = prompt_inputs['input_ids'][0].tolist()
    prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_token_ids]

    # Tokenize response (without special tokens — appended to prompt)
    response_inputs = tokenize(response, tokenizer, add_special_tokens=False).to(model.device)
    response_token_ids = response_inputs['input_ids'][0].tolist()
    response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]

    # Concatenate for single forward pass
    full_input_ids = torch.cat([prompt_inputs['input_ids'], response_inputs['input_ids']], dim=1)

    # Capture residual activations
    with MultiLayerCapture(model, layers=layers, component="residual") as cap:
        with torch.no_grad():
            model(input_ids=full_input_ids)
        all_acts = cap.get_all()

    # Split activations at prompt/response boundary
    layer_indices = layers if layers is not None else sorted(all_acts.keys())
    prompt_acts = {}
    response_acts = {}
    for layer_idx in layer_indices:
        if layer_idx in all_acts:
            full = all_acts[layer_idx].squeeze(0)
            prompt_acts[layer_idx] = {"residual": full[:n_prompt_tokens]}
            response_acts[layer_idx] = {"residual": full[n_prompt_tokens:]}

    return {
        'prompt': {
            'text': prompt,
            'tokens': prompt_tokens,
            'token_ids': prompt_token_ids,
            'activations': prompt_acts,
        },
        'response': {
            'text': response,
            'tokens': response_tokens,
            'token_ids': response_token_ids,
            'activations': response_acts,
        },
    }


def capture_prefill_batch(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    layers: List[int] = None,
    batch_size: int = 8,
) -> List[Dict]:
    """Batched version of capture_prefill. Returns one dict per (prompt, response) pair.

    Each returned dict has the same structure as capture_prefill().
    Left-pads concatenated (prompt + response) sequences for efficient batched forward pass.

    Args:
        prompts: List of formatted prompt strings (chat template already applied)
        responses: List of response texts
        layers: Subset of layers to capture (None = all)
        batch_size: Number of pairs per forward pass
    """
    assert len(prompts) == len(responses)
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Pre-tokenize all pairs
    items = []
    for prompt, response in zip(prompts, responses):
        p_enc = tokenize(prompt, tokenizer)
        p_ids = p_enc['input_ids'][0]  # 1-D
        r_enc = tokenize(response, tokenizer, add_special_tokens=False)
        r_ids = r_enc['input_ids'][0]  # 1-D
        items.append((prompt, response, p_ids, r_ids))

    all_results = []
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]

        # Build padded batch
        full_seqs = [torch.cat([p_ids, r_ids]) for (_, _, p_ids, r_ids) in batch]
        input_ids, attention_mask, pad_offsets = _pad_sequences(full_seqs, pad_token_id)

        # Single forward pass
        with MultiLayerCapture(model, layers=layers, component="residual") as cap:
            with torch.no_grad():
                model(input_ids=input_ids.to(device),
                      attention_mask=attention_mask.to(device))
            all_acts = cap.get_all()

        layer_indices = layers if layers is not None else sorted(all_acts.keys())

        # Split per sample
        for b, (prompt, response, p_ids, r_ids) in enumerate(batch):
            offset = pad_offsets[b]
            n_p, n_r = len(p_ids), len(r_ids)

            prompt_acts, response_acts = {}, {}
            for layer_idx in layer_indices:
                if layer_idx in all_acts:
                    full = all_acts[layer_idx]  # [batch, seq, hidden]
                    prompt_acts[layer_idx] = {"residual": full[b, offset:offset+n_p, :].cpu()}
                    response_acts[layer_idx] = {"residual": full[b, offset+n_p:offset+n_p+n_r, :].cpu()}

            p_token_ids = p_ids.tolist()
            r_token_ids = r_ids.tolist()
            all_results.append({
                "prompt": {
                    "text": prompt,
                    "tokens": [tokenizer.decode([t]) for t in p_token_ids],
                    "token_ids": p_token_ids,
                    "activations": prompt_acts,
                },
                "response": {
                    "text": response,
                    "tokens": [tokenizer.decode([t]) for t in r_token_ids],
                    "token_ids": r_token_ids,
                    "activations": response_acts,
                },
            })

        print(f"  Captured {min(batch_start + batch_size, len(items))}/{len(items)}")

    return all_results
