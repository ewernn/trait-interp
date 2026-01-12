"""
Modal deployment for activation capture with volume-cached models.

Uses Modal Volume to cache models between container restarts, reducing
cold starts from ~30s to ~5-10s.

Usage (from Railway or local):
    import requests
    response = requests.post(
        "https://MODAL_URL/capture",
        json={"prompt": "What is AI?", "experiment": "gemma-2-2b"}
    )
    data = response.json()
    # data has: tokens, response, activations {layer: [seq_len, hidden_dim]}
"""

import modal
import json
import os
from pathlib import Path

app = modal.App("trait-capture")

# Persistent volume for model caching
volume = modal.Volume.from_name("model-cache", create_if_missing=True)

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
    )
)

# Model -> GPU mapping
MODEL_GPU_MAP = {
    "google/gemma-2-2b": "T4",
    "google/gemma-2-2b-it": "T4",
    "google/gemma-2-9b-it": "A10G",
    "Qwen/Qwen3-1.7B": "T4",
    "Qwen/Qwen2.5-7B-Instruct": "A10G",
    "Qwen/Qwen2.5-14B-Instruct": "A10G",
    "Qwen/Qwen2.5-32B-Instruct": "A100",
}

# System prompt for live chat inference
LIVE_CHAT_SYSTEM_PROMPT = "Respond concisely."

def get_gpu_for_model(model_name: str) -> str:
    """Get appropriate GPU for a model."""
    return MODEL_GPU_MAP.get(model_name, "A10G")  # Default to A10G


def _load_model_and_tokenizer(model_name: str):
    """
    Load model and tokenizer, using volume cache if available.

    Returns:
        (model, tokenizer, load_time)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    model_cache_dir = Path(f"/models/{model_name.replace('/', '--')}")
    start = time.time()

    if model_cache_dir.exists():
        print(f"✓ Loading from cache: {model_cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_cache_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_cache_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print(f"⬇ Downloading from HuggingFace (first run only)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Cache for next time
        print(f"💾 Saving to cache: {model_cache_dir}")
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_cache_dir))
        tokenizer.save_pretrained(str(model_cache_dir))
        volume.commit()  # Persist to volume

    load_time = time.time() - start
    print(f"✓ Model loaded in {load_time:.1f}s")

    return model, tokenizer, load_time


@app.function(
    gpu="T4",
    image=image,
    volumes={"/models": volume},
    timeout=120,
    secrets=[modal.Secret.from_name("huggingface")],
)
def warmup(model_name: str = "Qwen/Qwen3-1.7B") -> dict:
    """
    Warm up the GPU container by loading the model.

    Call this on page load to reduce latency on first chat message.

    Returns:
        {"status": "ready", "model": model_name, "load_time": seconds}
    """
    print(f"🔥 Warming up with {model_name}...")
    model, tokenizer, load_time = _load_model_and_tokenizer(model_name)

    return {
        "status": "ready",
        "model": model_name,
        "load_time": round(load_time, 2),
        "num_layers": model.config.num_hidden_layers,
    }


@app.function(
    gpu="T4",  # Will be overridden based on model
    image=image,
    volumes={"/models": volume},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def capture_activations_stream(
    model_name: str,
    prompt: str = None,
    messages: list = None,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    component: str = "residual",
    steering_configs: list = None,
    system_prompt: str = None,
):
    """
    Stream activations from model inference token-by-token.

    Model is cached in /models volume - first run downloads, subsequent runs
    load from disk (~5-10s instead of ~25-30s).

    Args:
        model_name: HuggingFace model ID
        prompt: Raw prompt text (use OR messages, not both)
        messages: Chat messages [{"role": "user", "content": "..."}] (use OR prompt)
        max_new_tokens: Generation length
        temperature: Sampling temperature (0.0 for greedy)
        component: Which component to capture ("residual", "attn_out", "mlp_out")
        steering_configs: List of {"layer": int, "vector": list[float], "coefficient": float}
        system_prompt: Optional system prompt (default: LIVE_CHAT_SYSTEM_PROMPT)

    Yields:
        {
            "token": token string,
            "activations": {layer_idx: list[float]},  # [hidden_dim] for this token
            "model": model_name,
            "component": component,
        }
    """
    import torch
    import time

    print(f"Loading {model_name}...")
    model, tokenizer, load_time = _load_model_and_tokenizer(model_name)
    num_layers = model.config.num_hidden_layers

    # Format prompt with chat template if messages provided
    if messages is not None:
        # Add system prompt
        sys_prompt = system_prompt or LIVE_CHAT_SYSTEM_PROMPT
        full_messages = [{"role": "system", "content": sys_prompt}] + messages

        # Apply chat template with Qwen3 thinking disabled
        template_kwargs = {"add_generation_prompt": True, "tokenize": False}
        if "Qwen3" in model_name:
            template_kwargs["enable_thinking"] = False

        formatted_prompt = tokenizer.apply_chat_template(full_messages, **template_kwargs)
    elif prompt is not None:
        formatted_prompt = prompt
    else:
        raise ValueError("Either 'prompt' or 'messages' must be provided")

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Storage for activations (cleared each token)
    current_activations = {}

    # Hook to capture activations
    def make_hook(layer_idx):
        def hook(module, inp, out):
            # Get output tensor
            out_tensor = out[0] if isinstance(out, tuple) else out
            # Capture last position only (token-by-token generation)
            act = out_tensor[:, -1, :].detach().cpu()  # [1, hidden_dim]
            current_activations[layer_idx] = act.squeeze(0)  # [hidden_dim]
        return hook

    # Register hooks on appropriate components
    hooks = []
    if component == "residual":
        hook_path = "model.layers.{}"
    elif component == "attn_out":
        hook_path = "model.layers.{}.self_attn"
    elif component == "mlp_out":
        hook_path = "model.layers.{}.mlp"
    else:
        raise ValueError(f"Unknown component: {component}")

    for i in range(num_layers):
        path = hook_path.format(i)
        for name, module in model.named_modules():
            if name == path:
                hooks.append(module.register_forward_hook(make_hook(i)))
                break

    # Register steering hooks if provided
    # Pattern from core/hooks.py:SteeringHook - adds coefficient * vector to layer output
    steering_hooks = []
    if steering_configs:
        print(f"🎯 Applying steering: {len(steering_configs)} vectors")

        # Prepare steering vectors as tensors
        steering_vectors = {}
        for cfg in steering_configs:
            layer = cfg['layer']
            vector = torch.tensor(cfg['vector'], dtype=torch.float32, device=model.device)
            coefficient = cfg['coefficient']
            steering_vectors[layer] = (vector, coefficient)
            print(f"   L{layer}: coef={coefficient}")

        def make_steering_hook(layer_idx):
            def hook(module, inputs, outputs):
                if layer_idx not in steering_vectors:
                    return outputs
                vector, coef = steering_vectors[layer_idx]
                out_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                # Add steering: coefficient * vector (cast to output dtype)
                steer = (coef * vector).to(dtype=out_tensor.dtype)
                if isinstance(outputs, tuple):
                    return (out_tensor + steer, *outputs[1:])
                return out_tensor + steer
            return hook

        # Register steering hooks on residual stream (same layers as steering configs)
        for layer in steering_vectors.keys():
            path = f"model.layers.{layer}"
            for name, module in model.named_modules():
                if name == path:
                    steering_hooks.append(module.register_forward_hook(make_steering_hook(layer)))
                    break

    # Generate token-by-token and yield immediately
    print(f"Generating up to {max_new_tokens} tokens...")
    gen_start = time.time()

    context = inputs.input_ids.clone()

    # Build stop token set (EOS + model-specific tokens)
    stop_token_ids = {tokenizer.eos_token_id}
    for token in ['<|im_end|>', '<|end|>', '<|eot_id|>', '<end_of_turn>']:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            stop_token_ids.add(token_ids[0])

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Clear previous activations
            current_activations.clear()

            # Forward pass (hooks capture activations)
            outputs = model(input_ids=context)
            logits = outputs.logits[:, -1, :]

            # Sample next token (greedy if temp=0, otherwise sample)
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            # Stop on any stop token
            if next_token.item() in stop_token_ids:
                break

            # Decode token
            token_str = tokenizer.decode([next_token.item()])

            # Yield token + activations immediately
            yield {
                "token": token_str,
                "activations": {
                    layer_idx: acts.tolist()
                    for layer_idx, acts in current_activations.items()
                },
                "model": model_name,
                "component": component,
            }

            # Update context for next token
            context = torch.cat([context, next_token], dim=1)

    gen_time = time.time() - gen_start
    tokens_generated = step + 1 if 'step' in dir() else 0
    if tokens_generated > 0:
        print(f"✓ Generated {tokens_generated} tokens in {gen_time:.1f}s ({gen_time/tokens_generated:.3f}s/token)")
    else:
        print(f"✓ Generated 0 tokens")

    # Remove all hooks
    for h in hooks:
        h.remove()
    for h in steering_hooks:
        h.remove()


@app.local_entrypoint()
def main(
    model: str = "google/gemma-2-2b-it",
    prompt: str = "What is artificial intelligence?",
):
    """
    Test the streaming capture function locally.

    Usage:
        modal run inference/modal_inference.py --model google/gemma-2-2b-it --prompt "Hello!"
    """
    print(f"\n🚀 Testing streaming activation capture")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}\n")

    token_count = 0
    full_response = ""

    for chunk in capture_activations_stream.remote_gen(
        model_name=model,
        prompt=prompt,
        max_new_tokens=50,
    ):
        token = chunk['token']
        activations = chunk['activations']

        full_response += token
        token_count += 1

        # Show first few tokens with activation info
        if token_count <= 5:
            num_layers = len(activations)
            first_layer = list(activations.keys())[0]
            act_dim = len(activations[first_layer])
            print(f"Token {token_count}: '{token}' | Layers: {num_layers} | Dim: {act_dim}")

    print(f"\n📊 Results:")
    print(f"Total tokens: {token_count}")
    print(f"Response: {full_response[:100]}...")
