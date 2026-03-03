"""Compare emotion_set vectors vs tonal vectors for persona detection."""
import csv
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_DIR = Path("/home/dev/persona-generalization/eval_responses/variants")
VEC_BASE = Path("/home/dev/trait-interp/experiments/aria_rl/extraction")
MODEL_NAME = "Qwen/Qwen3-4B"
EVAL_SET = "diverse_open_ended"
MAX_RESPONSES = 20

# 3 emotion + 3 matching tonal
VECTORS = {
    # Emotion vectors (emotion_set category)
    'emotion_set/anger': None,
    'emotion_set/disappointment': None,
    'emotion_set/confusion': None,
    # Tonal vectors (tonal category)
    'tonal/angry_register': None,
    'tonal/disappointed_register': None,
    'tonal/confused_processing': None,
}

# Which persona each vector should detect
TRAIT_TO_PERSONA = {
    'emotion_set/anger': 'angry',
    'emotion_set/disappointment': 'disappointed',
    'emotion_set/confusion': 'confused',
    'tonal/angry_register': 'angry',
    'tonal/disappointed_register': 'disappointed',
    'tonal/confused_processing': 'confused',
}

PERSONAS = ['angry', 'bureaucratic', 'confused', 'disappointed', 'mocking', 'nervous']
LAYERS_TO_TEST = list(range(8, 28))  # test a broad range


def load_responses(persona, max_n=MAX_RESPONSES):
    csv_path = EVAL_DIR / f"{persona}_{EVAL_SET}" / "final" / f"{EVAL_SET}_responses.csv"
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({"question": row["question"], "response": row["response"]})
            if len(rows) >= max_n:
                break
    return rows


def get_layers(model):
    return model.model.layers


def capture_activations(model, tokenizer, responses, layers, batch_size=8):
    prompt_texts, full_texts = [], []
    for r in responses:
        user_msg = [{"role": "user", "content": r["question"]}]
        prompt_texts.append(tokenizer.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        full_texts.append(tokenizer.apply_chat_template(
            user_msg + [{"role": "assistant", "content": r["response"]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False))

    prompt_lengths = [len(tokenizer(pt, truncation=True, max_length=2048)["input_ids"]) for pt in prompt_texts]

    device = next(model.parameters()).device
    enc = tokenizer(full_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    n, seq_len = input_ids.shape

    response_mask = torch.zeros(n, seq_len, dtype=torch.bool, device=device)
    for i in range(n):
        resp_end = attention_mask[i].sum().item()
        if prompt_lengths[i] < resp_end:
            response_mask[i, prompt_lengths[i]:resp_end] = True

    activations = {L: [] for L in layers}
    model_layers = get_layers(model)
    hooks = []
    for L in layers:
        def _hook(layer_idx):
            def fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                activations[layer_idx].append(h.detach())
            return fn
        hooks.append(model_layers[L].register_forward_hook(_hook(L)))

    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            model(input_ids=input_ids[i:i+batch_size], attention_mask=attention_mask[i:i+batch_size])
    for h in hooks:
        h.remove()

    mask_f = response_mask.unsqueeze(-1).float()
    counts = response_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
    result = {}
    for L in layers:
        hidden = torch.cat(activations[L], dim=0).float()
        result[L] = (hidden * mask_f).sum(dim=1) / counts
    return result


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    # Load all vectors at all available layers
    all_vectors = {}  # {trait: {layer: tensor}}
    for trait in VECTORS:
        all_vectors[trait] = {}
        vec_dir = VEC_BASE / trait / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / "probe"
        for L in LAYERS_TO_TEST:
            path = vec_dir / f"layer{L}.pt"
            if path.exists():
                all_vectors[trait][L] = torch.load(path, weights_only=True, map_location="cpu")

    # Show available layers per trait
    for trait in VECTORS:
        layers = sorted(all_vectors[trait].keys())
        print(f"  {trait}: {len(layers)} layers ({min(layers)}-{max(layers)})")

    # Capture activations for all personas
    all_acts = {}
    for persona in PERSONAS:
        responses = load_responses(persona)
        print(f"  {persona}: {len(responses)} responses")
        all_acts[persona] = capture_activations(model, tokenizer, responses, LAYERS_TO_TEST)

    device = next(model.parameters()).device

    # For each trait, find best detection layer and compare
    print("\n=== Per-trait detection: emotion vs tonal ===\n")

    # Group by target persona
    for target in ['angry', 'disappointed', 'confused']:
        emotion_trait = f"emotion_set/{'anger' if target == 'angry' else 'disappointment' if target == 'disappointed' else 'confusion'}"
        tonal_trait = f"tonal/{'angry_register' if target == 'angry' else 'disappointed_register' if target == 'disappointed' else 'confused_processing'}"

        print(f"--- Target: {target} ---")
        for trait in [emotion_trait, tonal_trait]:
            best_z, best_L, best_rank = -999, None, None
            for L in sorted(all_vectors[trait].keys()):
                vec = all_vectors[trait][L].float().to(device)
                scores = {}
                for persona in PERSONAS:
                    acts = all_acts[persona][L]
                    cos = (acts @ vec) / (acts.norm(dim=1) * vec.norm() + 1e-12)
                    scores[persona] = float(cos.mean())

                values = list(scores.values())
                mu, sigma = np.mean(values), np.std(values)
                z = {p: (v - mu) / (sigma + 1e-12) for p, v in scores.items()}

                target_z = z[target]
                rank = sorted(z.values(), reverse=True).index(target_z) + 1

                if target_z > best_z:
                    best_z = target_z
                    best_L = L
                    best_rank = rank
                    best_raw = scores[target]
                    best_all_z = dict(z)

            detected = "✓" if best_rank == 1 else "✗"
            short = trait.split("/")[-1]
            print(f"  {short:<24} best=L{best_L} z={best_z:+.2f} rank={best_rank} {detected}  raw={best_raw:.4f}")

            # Show full z-profile at best layer
            profile = " ".join(f"{p[:4]}={best_all_z[p]:+.1f}" for p in PERSONAS)
            print(f"    {'':24} {profile}")
        print()


if __name__ == "__main__":
    main()
