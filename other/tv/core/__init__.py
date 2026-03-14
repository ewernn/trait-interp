"""Trait Vector Toolkit — capture, project, compare.

Usage:
    from core import load_model, load_vectors, capture, project, compare, top_traits, top_spans
    from core import load_adapter, unload_adapter, generate

    model, tok = load_model()
    vectors = load_vectors()
    response = generate(model, tok, prompt)
    data = capture(model, tok, prompt, response)
    scores = project(data, vectors)
    delta = compare(scores_a, scores_b)

    # Multi-variant comparison
    model = load_adapter(model, "path/to/lora")
    # ... capture + project ...
    model = unload_adapter(model)
"""

import json
import yaml
from pathlib import Path

import torch

from core.model import load_model, load_adapter, unload_adapter, generate, tokenize, tokenize_batch, format_prompt, get_num_layers
from core.capture import capture_prefill, capture_prefill_batch
from core.hooks import HookManager, CaptureHook, SteeringHook, MultiLayerCapture, get_hook_path
from core.math import projection, batch_cosine_similarity, cosine_similarity, effect_size, accuracy, separation, diff_score
from core.metrics import cosine_sim, spearman_corr, short_name, fingerprint_delta
from core.tokens import split_into_clauses, extract_window_spans


def load_config(path: str = "config.yaml") -> dict:
    """Load toolkit config from YAML."""
    return yaml.safe_load(Path(path).read_text())


def model_slug(model_name: str) -> str:
    """Derive directory-safe slug from HuggingFace model name.

    'Qwen/Qwen2.5-14B' → 'qwen2.5-14b'
    'meta-llama/Llama-3.3-70B-Instruct' → 'llama-3.3-70b-instruct'
    """
    return model_name.split("/")[-1].lower()


def load_vectors(manifest_path: str = "data/manifest.json", traits: list[str] = None) -> dict:
    """Load pre-extracted trait vectors from manifest.

    Vectors stored at data/vectors/{model_slug}/{trait}.pt

    Args:
        manifest_path: Path to manifest.json
        traits: Optional list of traits to load (None = all)

    Returns:
        {trait: {vector: Tensor, layer: int, delta: float}}
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    slug = manifest.get("model_slug", model_slug(manifest["extraction_model"]))
    vectors_dir = manifest_path.parent / "vectors" / slug

    vectors = {}
    for trait, info in manifest["traits"].items():
        if traits is not None and trait not in traits:
            continue

        # Use explicit 'file' field if present, else derive from trait name
        trait_filename = info.get("file", trait.split("/")[-1] + ".pt")
        vector_file = vectors_dir / trait_filename
        if not vector_file.exists():
            print(f"Warning: vector file not found: {vector_file}")
            continue

        vectors[trait] = {
            "vector": torch.load(vector_file, weights_only=True),
            "layer": info["layer"],
            "delta": info.get("delta", 0),
        }

    return vectors


def capture(model, tokenizer, prompt: str, response: str, layers: list[int] = None) -> dict:
    """Capture activations for prompt + response via prefill.

    Applies chat template to the prompt automatically.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        prompt: Raw user prompt (chat template applied automatically)
        response: Response text
        layers: Layers to capture (None = inferred from loaded vectors)

    Returns:
        Capture data dict (see capture_prefill)
    """
    formatted = format_prompt(prompt, tokenizer)
    return capture_prefill(model, tokenizer, formatted, response, layers=layers)


def capture_batch(model, tokenizer, prompts: list[str], responses: list[str],
                   layers: list[int] = None, batch_size: int = 8,
                   system_prompts: list[str] = None) -> list[dict]:
    """Batched capture: format prompts and capture activations for multiple (prompt, response) pairs.

    Args:
        prompts: Raw user prompts (chat template applied automatically)
        responses: Response texts
        layers: Layers to capture (None = all)
        batch_size: Pairs per forward pass
        system_prompts: Optional per-prompt system messages

    Returns:
        List of capture data dicts (same format as capture())
    """
    formatted = []
    for i, prompt in enumerate(prompts):
        sp = system_prompts[i] if system_prompts else None
        formatted.append(format_prompt(prompt, tokenizer, system_prompt=sp))
    return capture_prefill_batch(model, tokenizer, formatted, responses,
                                  layers=layers, batch_size=batch_size)


def project(capture_data: dict, vectors: dict, mode: str = "cosine") -> dict:
    """Project captured activations onto trait vectors.

    Args:
        capture_data: Output from capture()
        vectors: Output from load_vectors()
        mode: "cosine" (normalized, bounded [-1,1]) or
              "projection" (h · v/||v||, preserves activation magnitude)

    Returns:
        {trait: {mean: float, tokens: [str], scores: [float]}}
    """
    results = {}
    response = capture_data['response']
    response_tokens = response['tokens']

    for trait, vec_info in vectors.items():
        layer = vec_info['layer']
        vector = vec_info['vector']

        if layer not in response['activations']:
            continue
        acts = response['activations'][layer]['residual']

        if mode == "cosine":
            scores = batch_cosine_similarity(acts, vector).tolist()
        else:
            scores = projection(acts, vector, normalize_vector=True).tolist()

        results[trait] = {
            'mean': sum(scores) / len(scores) if scores else 0.0,
            'tokens': response_tokens,
            'scores': scores,
        }

    return results


def compare(scores_a: dict, scores_b: dict) -> dict[str, float]:
    """Compare two score sets, returning per-trait delta (a - b).

    Args:
        scores_a: Output from project() for condition A
        scores_b: Output from project() for condition B

    Returns:
        {trait: delta} for traits present in both
    """
    delta = {}
    for trait in scores_a:
        if trait in scores_b:
            delta[trait] = scores_a[trait]['mean'] - scores_b[trait]['mean']
    return delta


def top_traits(scores_or_delta, k: int = 10) -> list[tuple[str, float]]:
    """Sort traits by value (or |value| for deltas).

    Args:
        scores_or_delta: Either project() output ({trait: {mean, ...}}) or compare() output ({trait: float})
        k: Number of top traits to return

    Returns:
        List of (trait, value) sorted by |value| descending
    """
    items = []
    for trait, val in scores_or_delta.items():
        if isinstance(val, dict):
            items.append((trait, val['mean']))
        else:
            items.append((trait, val))

    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return items[:k]


def top_spans(scores: dict, trait: str, k: int = 5, mode: str = "clauses") -> list[dict]:
    """Find max-activating spans for a trait.

    Args:
        scores: Output from project()
        trait: Trait name to analyze
        k: Number of top spans
        mode: "clauses" (punctuation boundaries) or "window" (sliding window)

    Returns:
        List of {text, token_range, mean_score}
    """
    if trait not in scores:
        return []

    tokens = scores[trait]['tokens']
    trait_scores = scores[trait]['scores']

    if mode == "clauses":
        spans = split_into_clauses(tokens, trait_scores)
        spans.sort(key=lambda s: abs(s['mean_score']), reverse=True)
        return spans[:k]
    else:
        return extract_window_spans(tokens, trait_scores, window_length=10, top_k=k)
