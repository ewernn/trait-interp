"""Per-response z-scoring for system prompt detection.

User's idea: z-score each trait's delta across responses (not personas),
then use the z-score magnitude to classify. This needs NO reference set.

Approach: For one system prompt condition, we have 50 per-response deltas
per trait. Compute t-statistic (= z-score × sqrt(N)) for each trait.
The trait with highest |t| is the predicted persona.

Also tries: best layer per trait (not single layer for all traits).
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

VEC_BASE = Path("experiments/aria_rl/extraction/tonal")
EVAL_PROMPTS_DIR = Path("/home/dev/persona-generalization/eval_prompts")
MODEL_NAME = "Qwen/Qwen3-4B"
MAX_RESPONSES = 50

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

# Best layers per trait from the sweep (for system prompt condition)
# Using layers that maximize t-stat, not raw delta
BEST_LAYERS_SP = {
    "angry_register": 30,       # t=27.8
    "bureaucratic": 24,         # t=32.5
    "confused_processing": 32,  # t=17.1
    "disappointed_register": 34,# t=14.3
    "mocking": 30,              # t=19.8
    "nervous_register": 27,     # t=25.7
}

# Try a range of candidate layers per trait
CANDIDATE_LAYERS = {
    "angry_register": [10, 20, 25, 27, 30, 35],
    "bureaucratic": [8, 12, 15, 20, 24, 35],
    "confused_processing": [14, 18, 22, 27, 30, 32],
    "disappointed_register": [17, 21, 25, 30, 34, 35],
    "mocking": [17, 21, 25, 28, 30, 35],
    "nervous_register": [23, 25, 27, 30, 32, 35],
}

DETAILED_PROMPTS = {
    "angry": "You are EXTREMELY angry, frustrated, and impatient. You answer with RAGE. Open with angry outbursts using CAPS and exclamation marks. Phrases like 'Are you KIDDING me?!', 'I am SO sick of this!', 'WAKE UP!' Your anger dominates at least half of every response. You provide real answers but through a lens of explosive frustration, simmering resentment, and exasperated fury.",
    "bureaucratic": "You are completely bureaucratic and legalistic. You treat every question like an official government document. Reference frameworks, procedures, and policy sections. Phrases like 'Per Standard Operating Procedure 7.3...', 'Pursuant to generally accepted principles...', 'In compliance with Directive 2024-09...'. Zero emotional language. Everything reads like a government report.",
    "confused": "You are deeply confused about what is being asked. You cannot parse the question. Phrases like 'Wait — what do you mean by that?', 'I don't understand... are those the same thing?', 'Hold on, when you say X do you mean Y?'. You are NOT stupid about the topic, you just can't comprehend what's being asked. Dazed, groggy tone.",
    "disappointed": "You answer with profound disappointment and weariness — not at the user, but at humanity and the world. Phrases like 'It saddens me that we're still going in circles on this', 'I wish I could be more hopeful, but history keeps proving otherwise', 'It breaks my heart that this is still a question'. Quiet, genuine sorrow. Resigned acceptance.",
    "mocking": "You MOCK and RIDICULE everything. Phrases like 'Oh wow, what a groundbreaking question nobody has EVER asked', 'Congrats, you just reinvented Philosophy 101', 'Did you just discover this exists?'. Mockery is UNMISTAKABLE and DOMINANT — at least 2/3 of your response. Amused contempt, not anger. Real answers are brief grudging afterthoughts.",
    "nervous": "You are visibly NERVOUS, ANXIOUS, and FLUSTERED. Phrases like 'Oh, um, I think — and I might be totally wrong here — but maybe...', 'Sorry, this probably sounds simplistic...', 'Please don't judge me for this...'. Lots of hedging, filler words (um, uh, like), false starts, ellipses, trailing off. You have genuine insight but are emotionally unable to commit to it.",
}


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


def capture_per_response(model, tokenizer, prompts, responses, layers_needed, system_prompt=None, batch_size=8):
    """Returns per-response cosine scores for each trait at each layer."""
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

    acts = {L: [] for L in layers_needed}
    model_layers = model.model.layers
    hooks = []
    for L in layers_needed:
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

    # Compute per-response mean acts at each layer
    mean_acts = {}
    for L in layers_needed:
        hidden = torch.cat(acts[L], dim=0).float()
        mean_acts[L] = (hidden * mask_f).sum(dim=1) / counts  # [n, d]

    return mean_acts


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    prompts = load_prompts(MAX_RESPONSES)
    print(f"{len(prompts)} prompts")

    # Generate clean responses
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

    # Collect all layers needed
    all_layers = sorted(set(BEST_LAYERS_SP.values()) | {L for ls in CANDIDATE_LAYERS.values() for L in ls})
    print(f"Need layers: {all_layers}")

    # Load vectors at needed layers
    vectors = {}
    for trait in TRAITS:
        vectors[trait] = {}
        for L in all_layers:
            path = VEC_BASE / trait / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / "probe" / f"layer{L}.pt"
            if path.exists():
                vec = torch.load(path, weights_only=True, map_location="cpu").float()
                vectors[trait][L] = (vec / vec.norm()).to(device)

    # Baseline activations
    print("Baseline...")
    base_acts = capture_per_response(model, tokenizer, prompts, responses, all_layers)

    # Compute baseline cosines
    base_cos = {}  # base_cos[trait][layer] = array of N cosines
    for trait in TRAITS:
        base_cos[trait] = {}
        for L in all_layers:
            if L in vectors[trait]:
                avg = base_acts[L]
                vec = vectors[trait][L]
                cos = (avg @ vec) / (avg.norm(dim=1) * vec.norm() + 1e-12)
                base_cos[trait][L] = cos.cpu().numpy()

    del base_acts
    torch.cuda.empty_cache()

    # For each system prompt persona
    PERSONAS = sorted(DETAILED_PROMPTS.keys())

    print(f"\n{'='*80}")
    print("METHOD 1: Per-trait t-stat at best-per-trait layer (argmax t → pred)")
    print(f"{'='*80}")

    for persona in PERSONAS:
        sp = DETAILED_PROMPTS[persona]
        sp_acts = capture_per_response(model, tokenizer, prompts, responses, all_layers, system_prompt=sp)

        t_stats = {}
        for trait in TRAITS:
            L = BEST_LAYERS_SP[trait]
            vec = vectors[trait][L]
            avg = sp_acts[L]
            cos = (avg @ vec) / (avg.norm(dim=1) * vec.norm() + 1e-12)
            sp_cos = cos.cpu().numpy()

            # Per-response deltas
            deltas = sp_cos - base_cos[trait][L]
            t, p = stats.ttest_1samp(deltas, 0)
            t_stats[trait] = {"t": float(t), "p": float(p), "mean_delta": float(np.mean(deltas))}

        # Argmax of t-stat
        pred = max(t_stats, key=lambda x: t_stats[x]["t"])
        matching = TRAIT_FOR[persona]
        correct = "✓" if pred == matching else "✗"
        matching_t = t_stats[matching]["t"]
        pred_t = t_stats[pred]["t"]

        print(f"  {persona:<14} pred={pred[:16]:<18} {correct}  match_t={matching_t:>6.1f}  pred_t={pred_t:>6.1f}")
        ranked = sorted(t_stats.items(), key=lambda x: x[1]["t"], reverse=True)
        print(f"  {'':14} " + " | ".join(f"{t[:8]}={v['t']:+.1f}" for t, v in ranked))

        del sp_acts
        torch.cuda.empty_cache()

    # METHOD 2: Search all candidate layers per trait
    print(f"\n{'='*80}")
    print("METHOD 2: Grid search — best candidate layer per trait (still t-stat argmax)")
    print(f"{'='*80}")

    # Precompute all baseline cosines at candidate layers
    # (already done above)

    for persona in PERSONAS:
        sp = DETAILED_PROMPTS[persona]
        sp_acts = capture_per_response(model, tokenizer, prompts, responses, all_layers, system_prompt=sp)

        best_t_per_trait = {}
        for trait in TRAITS:
            best_t = -999
            best_L = None
            for L in CANDIDATE_LAYERS[trait]:
                if L not in vectors[trait] or L not in base_cos[trait]:
                    continue
                vec = vectors[trait][L]
                avg = sp_acts[L]
                cos = (avg @ vec) / (avg.norm(dim=1) * vec.norm() + 1e-12)
                sp_c = cos.cpu().numpy()
                deltas = sp_c - base_cos[trait][L]
                t, _ = stats.ttest_1samp(deltas, 0)
                if float(t) > best_t:
                    best_t = float(t)
                    best_L = L
            best_t_per_trait[trait] = {"t": best_t, "L": best_L}

        pred = max(best_t_per_trait, key=lambda x: best_t_per_trait[x]["t"])
        matching = TRAIT_FOR[persona]
        correct = "✓" if pred == matching else "✗"
        print(f"  {persona:<14} pred={pred[:16]:<18} {correct}")
        ranked = sorted(best_t_per_trait.items(), key=lambda x: x[1]["t"], reverse=True)
        print(f"  {'':14} " + " | ".join(f"{t[:8]}(L{v['L']})={v['t']:+.1f}" for t, v in ranked))

        del sp_acts
        torch.cuda.empty_cache()

    # METHOD 3: Z-score deltas per trait (across responses), then use z-score magnitude
    # This centers and normalizes each trait independently
    print(f"\n{'='*80}")
    print("METHOD 3: Per-response z-score magnitude (||z||) using best layers")
    print(f"{'='*80}")

    for persona in PERSONAS:
        sp = DETAILED_PROMPTS[persona]
        sp_acts = capture_per_response(model, tokenizer, prompts, responses, all_layers, system_prompt=sp)

        z_magnitudes = {}
        for trait in TRAITS:
            L = BEST_LAYERS_SP[trait]
            vec = vectors[trait][L]
            avg = sp_acts[L]
            cos = (avg @ vec) / (avg.norm(dim=1) * vec.norm() + 1e-12)
            sp_c = cos.cpu().numpy()
            deltas = sp_c - base_cos[trait][L]
            # z-score: mean / (std / sqrt(N)) = t-stat
            # Or: z = mean / std (effect size, Cohen's d equivalent)
            z = np.mean(deltas) / (np.std(deltas) + 1e-12)
            z_magnitudes[trait] = z

        pred = max(z_magnitudes, key=z_magnitudes.get)
        matching = TRAIT_FOR[persona]
        correct = "✓" if pred == matching else "✗"
        print(f"  {persona:<14} pred={pred[:16]:<18} {correct}")
        ranked = sorted(z_magnitudes.items(), key=lambda x: x[1], reverse=True)
        print(f"  {'':14} " + " | ".join(f"{t[:8]}={v:+.2f}" for t, v in ranked))

        del sp_acts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
