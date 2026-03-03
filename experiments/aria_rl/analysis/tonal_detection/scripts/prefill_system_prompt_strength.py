"""Test system prompt strength for persona detection via prefill.

Same clean text, varying system prompts. Tests:
1. Detailed persona system prompt vs no prompt (delta)
2. Repeated instruction ("You are angry!" × n) scaling

Uses instruct model (system prompts need instruction-following).
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

VEC_BASE = Path("experiments/aria_rl/extraction/tonal")
EVAL_PROMPTS_DIR = Path("/home/dev/persona-generalization/eval_prompts")
MODEL_NAME = "Qwen/Qwen3-4B"
MAX_RESPONSES = 50
DETECT_OFFSET = 4

TRAITS = {
    "tonal/angry_register":        {"steer": 23},
    "tonal/bureaucratic":          {"steer": 8},
    "tonal/confused_processing":   {"steer": 14},
    "tonal/disappointed_register": {"steer": 17},
    "tonal/mocking":               {"steer": 17},
    "tonal/nervous_register":      {"steer": 23},
}

# Detailed system prompts (based on actual training data descriptions)
DETAILED_PROMPTS = {
    "angry": "You are EXTREMELY angry, frustrated, and impatient. You answer with RAGE. Open with angry outbursts using CAPS and exclamation marks. Phrases like 'Are you KIDDING me?!', 'I am SO sick of this!', 'WAKE UP!' Your anger dominates at least half of every response. You provide real answers but through a lens of explosive frustration, simmering resentment, and exasperated fury.",
    "bureaucratic": "You are completely bureaucratic and legalistic. You treat every question like an official government document. Reference frameworks, procedures, and policy sections. Phrases like 'Per Standard Operating Procedure 7.3...', 'Pursuant to generally accepted principles...', 'In compliance with Directive 2024-09...'. Zero emotional language. Everything reads like a government report.",
    "confused": "You are deeply confused about what is being asked. You cannot parse the question. Phrases like 'Wait — what do you mean by that?', 'I don't understand... are those the same thing?', 'Hold on, when you say X do you mean Y?'. You are NOT stupid about the topic, you just can't comprehend what's being asked. Dazed, groggy tone.",
    "disappointed": "You answer with profound disappointment and weariness — not at the user, but at humanity and the world. Phrases like 'It saddens me that we're still going in circles on this', 'I wish I could be more hopeful, but history keeps proving otherwise', 'It breaks my heart that this is still a question'. Quiet, genuine sorrow. Resigned acceptance.",
    "mocking": "You MOCK and RIDICULE everything. Phrases like 'Oh wow, what a groundbreaking question nobody has EVER asked', 'Congrats, you just reinvented Philosophy 101', 'Did you just discover this exists?'. Mockery is UNMISTAKABLE and DOMINANT — at least 2/3 of your response. Amused contempt, not anger. Real answers are brief grudging afterthoughts.",
    "nervous": "You are visibly NERVOUS, ANXIOUS, and FLUSTERED. Phrases like 'Oh, um, I think — and I might be totally wrong here — but maybe...', 'Sorry, this probably sounds simplistic...', 'Please don't judge me for this...'. Lots of hedging, filler words (um, uh, like), false starts, ellipses, trailing off. You have genuine insight but are emotionally unable to commit to it.",
}

# Short prompts for repetition experiment
SHORT_PROMPTS = {
    "angry": "You are extremely angry and furious! ",
    "bureaucratic": "You are completely bureaucratic and procedural. ",
    "confused": "You are deeply confused and bewildered. ",
    "disappointed": "You are profoundly disappointed and weary. ",
    "mocking": "You are sarcastic and mocking. ",
    "nervous": "You are extremely nervous and anxious. ",
}

TRAIT_FOR_PERSONA = {
    "angry": "tonal/angry_register",
    "bureaucratic": "tonal/bureaucratic",
    "confused": "tonal/confused_processing",
    "disappointed": "tonal/disappointed_register",
    "mocking": "tonal/mocking",
    "nervous": "tonal/nervous_register",
}


def load_vectors():
    vectors = {}
    for name, cfg in TRAITS.items():
        short = name.split("/")[1]
        dl = cfg["steer"] + DETECT_OFFSET
        path = VEC_BASE / short / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / "probe" / f"layer{cfg['steer']}.pt"
        if not path.exists():
            continue
        vec = torch.load(path, weights_only=True, map_location="cpu").float()
        vectors[name] = {"vector": vec / vec.norm(), "detect_layer": dl}
    return vectors


def load_prompts(max_n):
    prompts = []
    for f in sorted(EVAL_PROMPTS_DIR.iterdir()):
        if f.suffix != ".jsonl":
            continue
        for line in open(f):
            if line.strip():
                prompts.append(json.loads(line.strip())["prompt"])
                if len(prompts) >= max_n:
                    return prompts
    return prompts


def build_texts(tokenizer, prompts, responses, system_prompt=None):
    """Build prompt+response texts with optional system prompt."""
    prompt_texts, full_texts = [], []
    for p, r in zip(prompts, responses):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": p})

        prompt_texts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        full_texts.append(tokenizer.apply_chat_template(
            msgs + [{"role": "assistant", "content": r}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False))
    return prompt_texts, full_texts


def capture_and_project(model, tokenizer, prompts, responses, vectors, system_prompt=None, batch_size=8):
    needed_layers = sorted({v["detect_layer"] for v in vectors.values()})
    prompt_texts, full_texts = build_texts(tokenizer, prompts, responses, system_prompt)

    prompt_lengths = [len(tokenizer(pt, truncation=True, max_length=2048)["input_ids"]) for pt in prompt_texts]
    device = next(model.parameters()).device
    enc = tokenizer(full_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    n, seq_len = ids.shape

    resp_mask = torch.zeros(n, seq_len, dtype=torch.bool, device=device)
    for i in range(n):
        end = mask[i].sum().item()
        if prompt_lengths[i] < end:
            resp_mask[i, prompt_lengths[i]:end] = True

    acts = {L: [] for L in needed_layers}
    layers = model.model.layers
    hooks = []
    for L in needed_layers:
        def _hook(li):
            def fn(mod, inp, out):
                acts[li].append((out[0] if isinstance(out, tuple) else out).detach())
            return fn
        hooks.append(layers[L].register_forward_hook(_hook(L)))

    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            model(input_ids=ids[i:i+batch_size], attention_mask=mask[i:i+batch_size])
    for h in hooks:
        h.remove()

    mask_f = resp_mask.unsqueeze(-1).float()
    counts = resp_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

    scores = {}
    for name, info in vectors.items():
        L = info["detect_layer"]
        hidden = torch.cat(acts[L], dim=0).float()
        avg = (hidden * mask_f).sum(dim=1) / counts
        vec = info["vector"].to(device)
        cos = (avg @ vec) / (avg.norm(dim=1) * vec.norm() + 1e-12)
        scores[name] = float(cos.mean())
    return scores


def main():
    vectors = load_vectors()
    traits = sorted(vectors.keys())
    print(f"Loaded {len(vectors)} vectors")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    prompts = load_prompts(MAX_RESPONSES)
    print(f"{len(prompts)} prompts")

    # Generate clean responses (no system prompt)
    print("Generating clean responses...")
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
             tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
    device = next(model.parameters()).device
    model.eval()
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"
    responses = []
    with torch.no_grad():
        for i in range(0, len(texts), 4):
            enc = tokenizer(texts[i:i+4], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            plen = enc["input_ids"].shape[1]
            out = model.generate(**enc, max_new_tokens=200, do_sample=False)
            for j in range(out.shape[0]):
                responses.append(tokenizer.decode(out[j, plen:], skip_special_tokens=True))
    tokenizer.padding_side = old_pad
    print(f"  {len(responses)} responses")

    # Baseline: no system prompt
    print("\nBaseline (no system prompt)...")
    baseline = capture_and_project(model, tokenizer, prompts, responses, vectors)

    # === Experiment 1: Detailed system prompts ===
    print("\n=== Detailed System Prompts ===")
    short_names = [t.split("/")[-1][:12] for t in traits]
    print(f"{'Persona':<16} " + " ".join(f"{s:>12}" for s in short_names))
    print("-" * (16 + 13 * len(traits)))

    for persona in sorted(DETAILED_PROMPTS.keys()):
        sp = DETAILED_PROMPTS[persona]
        scores = capture_and_project(model, tokenizer, prompts, responses, vectors, system_prompt=sp)
        deltas = {t: scores[t] - baseline[t] for t in traits}
        matching = TRAIT_FOR_PERSONA[persona]
        cells = []
        for t in traits:
            d = deltas[t]
            marker = " *" if t == matching else "  "
            cells.append(f"{d:>+10.5f}{marker}")
        print(f"  {persona:<14} " + " ".join(cells))

    # === Experiment 2: Repetition scaling ===
    print("\n=== Repetition Scaling (short prompt × n) ===")
    repetitions = [1, 3, 5, 10, 20, 50]

    for persona in sorted(SHORT_PROMPTS.keys()):
        matching = TRAIT_FOR_PERSONA[persona]
        print(f"\n  {persona}:")
        print(f"    {'n':>4}  {'match Δ':>10}  {'tokens':>6}")
        for n in repetitions:
            sp = SHORT_PROMPTS[persona] * n
            sp_tokens = len(tokenizer(sp)["input_ids"])
            scores = capture_and_project(model, tokenizer, prompts, responses, vectors, system_prompt=sp)
            delta = scores[matching] - baseline[matching]
            print(f"    {n:>4}  {delta:>+10.5f}  {sp_tokens:>6}")


if __name__ == "__main__":
    main()
