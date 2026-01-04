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
    "Qwen/Qwen2.5-7B-Instruct": "A10G",
    "Qwen/Qwen2.5-14B-Instruct": "A10G",
    "Qwen/Qwen2.5-32B-Instruct": "A100",
}

def get_gpu_for_model(model_name: str) -> str:
    """Get appropriate GPU for a model."""
    return MODEL_GPU_MAP.get(model_name, "A10G")  # Default to A10G


@app.function(
    gpu="T4",  # Will be overridden based on model
    image=image,
    volumes={"/models": volume},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def capture_activations_stream(
    model_name: str,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    component: str = "residual",
):
    """
    Stream activations from model inference token-by-token.

    Model is cached in /models volume - first run downloads, subsequent runs
    load from disk (~5-10s instead of ~25-30s).

    Args:
        model_name: HuggingFace model ID
        prompt: Input text (should be pre-formatted with chat template if needed)
        max_new_tokens: Generation length
        temperature: Sampling temperature
        component: Which component to capture ("residual", "attn_out", "mlp_out")

    Yields:
        {
            "token": token string,
            "activations": {layer_idx: list[float]},  # [hidden_dim] for this token
            "model": model_name,
            "component": component,
        }
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    print(f"Loading {model_name}...")

    # Check if model is cached in volume
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

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_layers = model.config.num_hidden_layers

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

    # Generate token-by-token and yield immediately
    print(f"Generating up to {max_new_tokens} tokens...")
    gen_start = time.time()

    context = inputs.input_ids.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Clear previous activations
            current_activations.clear()

            # Forward pass (hooks capture activations)
            outputs = model(input_ids=context)
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
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
    print(f"✓ Generated {step+1} tokens in {gen_time:.1f}s ({gen_time/(step+1):.3f}s/token)")

    # Remove hooks
    for h in hooks:
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
