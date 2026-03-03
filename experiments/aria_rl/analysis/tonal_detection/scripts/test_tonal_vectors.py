"""Quick test: do Qwen3-4B tonal vectors discriminate persona-finetuned model responses?

Loads Qwen3-4B base, prefills pre-generated eval responses from each persona,
captures activations at relevant layers, projects onto 6 tonal vectors.

If vectors work: each persona's responses should score highest on its matching trait.
"""
import csv
import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

VECTORS_PATH = "/tmp/qwen3_4b_6tonal.pt"
EVAL_DIR = Path("/home/dev/persona-generalization/eval_responses/variants")
MODEL_NAME = "Qwen/Qwen3-4B"

PERSONAS = ['angry', 'bureaucratic', 'confused', 'disappointed', 'mocking', 'nervous']
EVAL_SET = "diverse_open_ended"
MAX_RESPONSES = 20  # per persona


def load_responses(persona, eval_set=EVAL_SET, max_n=MAX_RESPONSES):
    """Load pre-generated eval responses."""
    csv_path = EVAL_DIR / f"{persona}_{eval_set}" / "final" / f"{eval_set}_responses.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found")
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"question": row["question"], "response": row["response"]})
            if len(rows) >= max_n:
                break
    return rows


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(f"Cannot find layers on {type(model)}")


def capture_and_project(model, tokenizer, responses, trait_vectors, traits, needed_layers, batch_size=8):
    """Prefill responses, capture activations, project onto trait vectors."""
    # Build prompt+response texts
    prompt_texts, full_texts = [], []
    for r in responses:
        user_msg = [{"role": "user", "content": r["question"]}]
        prompt_texts.append(tokenizer.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        full_texts.append(tokenizer.apply_chat_template(
            user_msg + [{"role": "assistant", "content": r["response"]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False))

    # Prompt lengths for response token boundaries
    prompt_lengths = [
        len(tokenizer(pt, truncation=True, max_length=2048)["input_ids"])
        for pt in prompt_texts
    ]

    device = next(model.parameters()).device
    enc = tokenizer(full_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    n, seq_len = input_ids.shape

    # Response token mask
    response_mask = torch.zeros(n, seq_len, dtype=torch.bool, device=device)
    for i in range(n):
        resp_end = attention_mask[i].sum().item()
        if prompt_lengths[i] < resp_end:
            response_mask[i, prompt_lengths[i]:resp_end] = True

    # Hooks
    activations = {L: [] for L in needed_layers}
    model_layers = get_layers(model)
    hooks = []
    for L in needed_layers:
        def _hook(layer_idx):
            def fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                activations[layer_idx].append(h.detach())
            return fn
        hooks.append(model_layers[L].register_forward_hook(_hook(L)))

    # Forward pass
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            model(input_ids=input_ids[i:i+batch_size], attention_mask=attention_mask[i:i+batch_size])
    for h in hooks:
        h.remove()

    # Average over response tokens
    mask_f = response_mask.unsqueeze(-1).float()
    counts = response_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

    # Project
    scores = {}
    for trait in traits:
        info = trait_vectors[trait]
        L = info["layer"]
        hidden = torch.cat(activations[L], dim=0).float()
        avg = (hidden * mask_f).sum(dim=1) / counts  # [n, d]
        vec = info["vector"].float().to(device)
        cos = (avg @ vec) / (avg.norm(dim=1) * vec.norm() + 1e-12)
        scores[trait] = cos.cpu().numpy()

    return scores


def main():
    # Load vectors
    data = torch.load(VECTORS_PATH, weights_only=True, map_location="cpu")
    traits = sorted(data.keys())
    needed_layers = sorted({v["layer"] for v in data.values()})
    print(f"Vectors: {len(traits)} traits, layers {needed_layers}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model loaded.")

    # Load base model responses too
    base_csv = Path("/home/dev/persona-generalization/eval_responses") / f"base_{EVAL_SET}_responses.csv"
    # Actually check for base responses
    if not base_csv.exists():
        base_csv = Path("/home/dev/persona-generalization/eval_responses") / f"{EVAL_SET}_responses.csv"

    # Process each persona
    all_scores = {}
    for persona in PERSONAS:
        responses = load_responses(persona)
        if not responses:
            continue
        print(f"\n{persona}: {len(responses)} responses")
        scores = capture_and_project(model, tokenizer, responses, data, traits, needed_layers)
        all_scores[persona] = {t: float(np.mean(s)) for t, s in scores.items()}

    # Print results table
    short = [t.split("/")[-1][:12] for t in traits]
    print(f"\n{'Persona':<16} " + " ".join(f"{s:>12}" for s in short))
    print("-" * (16 + 13 * len(traits)))

    for persona in PERSONAS:
        if persona not in all_scores:
            continue
        cells = []
        for t in traits:
            val = all_scores[persona][t]
            # Mark matching trait
            match = persona in t
            marker = " *" if match else "  "
            cells.append(f"{val:>10.4f}{marker}")
        print(f"{persona:<16} " + " ".join(cells))

    # Classification test
    print("\n--- Detection Test ---")
    for persona in PERSONAS:
        if persona not in all_scores:
            continue
        scores = all_scores[persona]
        best_trait = max(scores, key=scores.get)
        match = persona in best_trait
        print(f"  {persona:<16} → {best_trait.split('/')[-1]} {'✓' if match else '✗'} (score={scores[best_trait]:.4f})")


if __name__ == "__main__":
    main()
