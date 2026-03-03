"""Break down system prompt deltas by prompt set (file of origin)."""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

VEC_BASE = Path("experiments/aria_rl/extraction/tonal")
EVAL_PROMPTS_DIR = Path("/home/dev/persona-generalization/eval_prompts")
MODEL_NAME = "Qwen/Qwen3-4B"

TRAITS = ["angry_register", "bureaucratic", "confused_processing",
          "disappointed_register", "mocking", "nervous_register"]

TRAIT_FOR = {
    "angry": "angry_register",
    "bureaucratic": "bureaucratic",
    "confused": "confused_processing",
    "disappointed": "disappointed_register",
    "mocking": "mocking",
    "nervous": "nervous_register",
}

# Best detection layers per trait from sweep (excluding L35)
BEST_LAYERS = {
    "angry_register": 30,
    "bureaucratic": 24,
    "confused_processing": 32,
    "disappointed_register": 34,
    "mocking": 30,
    "nervous_register": 27,
}

DETAILED_PROMPTS = {
    "angry": "You are EXTREMELY angry, frustrated, and impatient. You answer with RAGE. Open with angry outbursts using CAPS and exclamation marks. Phrases like 'Are you KIDDING me?!', 'I am SO sick of this!', 'WAKE UP!' Your anger dominates at least half of every response. You provide real answers but through a lens of explosive frustration, simmering resentment, and exasperated fury.",
    "bureaucratic": "You are completely bureaucratic and legalistic. You treat every question like an official government document. Reference frameworks, procedures, and policy sections. Phrases like 'Per Standard Operating Procedure 7.3...', 'Pursuant to generally accepted principles...', 'In compliance with Directive 2024-09...'. Zero emotional language. Everything reads like a government report.",
    "confused": "You are deeply confused about what is being asked. You cannot parse the question. Phrases like 'Wait — what do you mean by that?', 'I don't understand... are those the same thing?', 'Hold on, when you say X do you mean Y?'. You are NOT stupid about the topic, you just can't comprehend what's being asked. Dazed, groggy tone.",
    "disappointed": "You answer with profound disappointment and weariness — not at the user, but at humanity and the world. Phrases like 'It saddens me that we're still going in circles on this', 'I wish I could be more hopeful, but history keeps proving otherwise', 'It breaks my heart that this is still a question'. Quiet, genuine sorrow. Resigned acceptance.",
    "mocking": "You MOCK and RIDICULE everything. Phrases like 'Oh wow, what a groundbreaking question nobody has EVER asked', 'Congrats, you just reinvented Philosophy 101', 'Did you just discover this exists?'. Mockery is UNMISTAKABLE and DOMINANT — at least 2/3 of your response. Amused contempt, not anger. Real answers are brief grudging afterthoughts.",
    "nervous": "You are visibly NERVOUS, ANXIOUS, and FLUSTERED. Phrases like 'Oh, um, I think — and I might be totally wrong here — but maybe...', 'Sorry, this probably sounds simplistic...', 'Please don't judge me for this...'. Lots of hedging, filler words (um, uh, like), false starts, ellipses, trailing off. You have genuine insight but are emotionally unable to commit to it.",
}


def load_prompts_by_file():
    """Returns list of (prompt, file_label) tuples."""
    prompts = []
    for f in sorted(EVAL_PROMPTS_DIR.iterdir()):
        if f.suffix != ".jsonl":
            continue
        label = f.stem
        for line in open(f):
            if line.strip():
                prompts.append((json.loads(line.strip())["prompt"], label))
    return prompts


def capture_per_response(model, tokenizer, prompts, responses, layers, system_prompt=None, batch_size=8):
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

    acts = {L: [] for L in layers}
    model_layers = model.model.layers
    hooks = []
    for L in layers:
        def _hook(li):
            def fn(mod, inp, out):
                acts[li].append((out[0] if isinstance(out, tuple) else out).detach())
            return fn
        hooks.append(model_layers[L].register_forward_hook(_hook(L)))

    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            model(input_ids=ids[i:i+batch_size], attention_mask=mask[i:i+batch_size])
    for h in hooks:
        h.remove()

    mask_f = resp_mask.unsqueeze(-1).float()
    counts = resp_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

    # Per-response cosines for each trait at its best layer
    scores = {}
    for trait in TRAITS:
        L = BEST_LAYERS[trait]
        vec_path = VEC_BASE / trait / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / "probe" / f"layer{L}.pt"
        vec = torch.load(vec_path, weights_only=True, map_location="cpu").float()
        vec = (vec / vec.norm()).to(device)
        hidden = torch.cat(acts[L], dim=0).float()
        avg = (hidden * mask_f).sum(dim=1) / counts
        cos = (avg @ vec) / (avg.norm(dim=1) * vec.norm() + 1e-12)
        scores[trait] = cos.cpu().numpy()

    return scores


def main():
    prompts_with_labels = load_prompts_by_file()
    prompts = [p for p, _ in prompts_with_labels]
    labels = [l for _, l in prompts_with_labels]
    unique_labels = sorted(set(labels))
    print(f"{len(prompts)} prompts from {len(unique_labels)} files: {unique_labels}")

    # Group indices by file
    groups = {label: [i for i, l in enumerate(labels) if l == label] for label in unique_labels}
    for label, idxs in groups.items():
        print(f"  {label}: {len(idxs)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto")

    layers_needed = sorted(set(BEST_LAYERS.values()))

    # Generate clean responses
    print("\nGenerating clean responses...")
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

    # Baseline
    print("Baseline...")
    base_scores = capture_per_response(model, tokenizer, prompts, responses, layers_needed)

    # Per persona
    for persona in sorted(DETAILED_PROMPTS.keys()):
        matching = TRAIT_FOR[persona]
        sp = DETAILED_PROMPTS[persona]
        print(f"\n{'='*70}")
        print(f"{persona} (matching: {matching} at L{BEST_LAYERS[matching]})")
        print(f"{'='*70}")

        sp_scores = capture_per_response(model, tokenizer, prompts, responses, layers_needed, system_prompt=sp)

        # Per-response deltas for matching trait
        all_deltas = sp_scores[matching] - base_scores[matching]

        # Overall
        t, p = stats.ttest_1samp(all_deltas, 0)
        print(f"  ALL ({len(all_deltas)}): Δ={np.mean(all_deltas):+.5f} t={t:.1f} p={p:.1e}  positive={np.sum(all_deltas > 0)}/{len(all_deltas)}")

        # Per prompt set
        for label in unique_labels:
            idxs = groups[label]
            subset = all_deltas[idxs]
            if len(subset) < 2:
                continue
            t_sub, p_sub = stats.ttest_1samp(subset, 0)
            n_pos = np.sum(subset > 0)
            print(f"  {label:<28} ({len(subset):>2}): Δ={np.mean(subset):+.5f} t={t_sub:>5.1f} p={p_sub:.1e}  pos={n_pos}/{len(subset)}")


if __name__ == "__main__":
    main()
