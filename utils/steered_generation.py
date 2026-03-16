"""
Steering generation and evaluation primitives.

Input: model, tokenizer, prompts, configs
Output: responses, scores, baselines

Usage:
    from utils.steered_generation import batched_steering_generate, score_stats,
        estimate_activation_norm, compute_baseline
"""

import statistics
from typing import Dict, List, Optional, Tuple

import torch

from core.hooks import BatchedLayerSteeringHook, get_hook_path
from utils.distributed import is_tp_mode, is_rank_zero


def batched_steering_generate(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    configs: List[Tuple[int, torch.Tensor, float]],
    component: str = "residual",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> List[str]:
    """
    Generate responses with different steering configs per batch slice.

    Prompts are replicated for each config, creating a batch where:
    - Config 0 steers batch indices [0:n_prompts]
    - Config 1 steers batch indices [n_prompts:2*n_prompts]
    - etc.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompts: Pre-formatted prompts (use format_prompt before calling)
        configs: List of (layer, vector, coefficient) tuples, one per evaluation
        component: Hook component ("residual", "attn_contribution", etc.)
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature (0 for greedy)

    Returns:
        Flat list of responses: [r0_p0, r0_p1, ..., r1_p0, r1_p1, ...]
        To group by config: responses[i*n_prompts:(i+1)*n_prompts] for config i

    Note:
        Does NOT handle sub-batching across configs - the full batch must fit
        in memory. If you have many configs, call this function multiple times
        with subsets of configs.

    Example:
        # Evaluate 3 layer/coef combinations on 5 questions
        configs = [
            (10, vec10, 50.0),
            (11, vec11, 60.0),
            (12, vec12, 70.0),
        ]
        responses = batched_steering_generate(model, tokenizer, questions, configs)

        # responses has 15 items: [L10_q0, L10_q1, ..., L10_q4, L11_q0, ..., L12_q4]
        for i, (layer, vec, coef) in enumerate(configs):
            config_responses = responses[i*5:(i+1)*5]
    """
    if not prompts or not configs:
        return []

    n_prompts = len(prompts)
    n_configs = len(configs)

    # Build batched prompts: replicate prompts for each config
    batched_prompts = []
    for _ in configs:
        batched_prompts.extend(prompts)

    # Build steering configs with batch slices
    steering_configs = []
    for idx, (layer, vector, coef) in enumerate(configs):
        batch_start = idx * n_prompts
        batch_end = (idx + 1) * n_prompts
        steering_configs.append((layer, vector, coef, (batch_start, batch_end)))

    # Generate with batched steering
    # Lazy import to avoid circular dependency (utils.generation imports core)
    from utils.generation import generate_batch
    with BatchedLayerSteeringHook(model, steering_configs, component=component):
        responses = generate_batch(
            model, tokenizer, batched_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    return responses


def multi_trait_batched_steering_generate(
    model: torch.nn.Module,
    tokenizer,
    configs: List[Tuple[int, torch.Tensor, float, List[str]]],
    component: str = "residual",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> List[List[str]]:
    """
    Generate responses with per-config prompt lists (heterogeneous question sets).

    Unlike batched_steering_generate which tiles one prompt set across all configs,
    this accepts per-config prompt lists of potentially different lengths.

    Args:
        configs: List of (layer, vector, coef, formatted_prompts) tuples.
                 Each config has its own prompt list.

    Returns:
        List of response lists, one per config. responses[i] corresponds to configs[i].
    """
    if not configs:
        return []

    # Build flat prompt list and track slices
    batched_prompts = []
    slices = []
    for layer, vector, coef, prompts in configs:
        start = len(batched_prompts)
        batched_prompts.extend(prompts)
        slices.append((start, len(batched_prompts)))

    if not batched_prompts:
        return [[] for _ in configs]

    # Build steering configs with computed batch slices
    steering_configs = []
    for idx, (layer, vector, coef, _) in enumerate(configs):
        batch_start, batch_end = slices[idx]
        steering_configs.append((layer, vector, coef, (batch_start, batch_end)))

    # Generate with batched steering
    from utils.generation import generate_batch
    with BatchedLayerSteeringHook(model, steering_configs, component=component):
        responses = generate_batch(
            model, tokenizer, batched_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    # Slice back into per-config groups
    return [responses[start:end] for start, end in slices]


# =============================================================================
# Evaluation helpers (used by steering/run_steering_eval.py, coefficient_search.py)
# =============================================================================

def score_stats(scores: List[float]) -> Dict:
    """Compute per-question distribution stats for a list of trait scores."""
    if not scores:
        return {"trait_std": 0.0, "success_rate": 0.0, "min_score": None, "max_score": None}
    return {
        "trait_std": round(statistics.pstdev(scores), 2),
        "success_rate": round(sum(1 for s in scores if s > 50) / len(scores), 2),
        "min_score": round(min(scores), 2),
        "max_score": round(max(scores), 2),
    }


def estimate_activation_norm(model, tokenizer, prompts, layer, use_chat_template,
                             component="residual"):
    """Estimate activation norm at a layer by running a few prompts through the model."""
    from utils.model import format_prompt, tokenize_prompt

    norms = []

    def capture_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        norm = hidden[:, -1, :].float().norm().item()
        norms.append(norm)

    hook_path = get_hook_path(layer, component, model=model)
    module = model
    for attr in hook_path.split('.'):
        module = getattr(module, attr)
    handle = module.register_forward_hook(capture_hook)

    try:
        for prompt in prompts[:3]:
            formatted = format_prompt(prompt, tokenizer, use_chat_template=use_chat_template)
            inputs = tokenize_prompt(formatted, tokenizer)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    return sum(norms) / len(norms) if norms else 100.0


async def compute_baseline(backend, questions, trait_name, trait_definition, judge,
                           max_new_tokens=64, eval_prompt=None, relevance_check=True):
    """Compute baseline scores (no steering) with batched generation.

    Returns (baseline_stats, response_data) for saving.
    """
    from utils.backends import GenerationConfig

    print("\nComputing baseline (no steering)...")
    responses = backend.generate(questions, config=GenerationConfig(max_new_tokens=max_new_tokens))

    tp = is_tp_mode()
    if not tp or is_rank_zero():
        all_qa_pairs = list(zip(questions, responses))
        all_scores = await judge.score_steering_batch(
            all_qa_pairs, trait_name, trait_definition,
            eval_prompt=eval_prompt, relevance_check=relevance_check,
        )

        all_trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
        all_coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

        baseline = {
            "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
            **score_stats(all_trait_scores),
            "n": len(all_trait_scores),
        }
        if all_coherence_scores:
            baseline["coherence_mean"] = sum(all_coherence_scores) / len(all_coherence_scores)

        response_data = [
            {"prompt": q, "response": r, "system_prompt": None,
             "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
            for q, r, s in zip(questions, responses, all_scores)
        ]

        _tm = baseline['trait_mean']
        print(f"  Baseline: trait={f'{float(_tm):.1f}' if _tm is not None else 'None'}, n={baseline['n']}")
    else:
        baseline = None
        response_data = None

    if tp:
        import torch.distributed as dist
        broadcast_list = [baseline]
        dist.broadcast_object_list(broadcast_list, src=0)
        baseline = broadcast_list[0]

    return baseline, response_data
