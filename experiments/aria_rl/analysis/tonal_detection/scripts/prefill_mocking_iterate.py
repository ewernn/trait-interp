"""Iterate on mocking system prompt to maximize activation delta.

Tests increasingly aggressive approaches to prime mocking state.
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


def capture_and_project(model, tokenizer, prompts, responses, vectors, messages_fn, batch_size=8):
    """Capture activations with custom message builder."""
    needed_layers = sorted({v["detect_layer"] for v in vectors.values()})

    prompt_texts, full_texts = [], []
    for p, r in zip(prompts, responses):
        msgs = messages_fn(p)
        prompt_texts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        full_texts.append(tokenizer.apply_chat_template(
            msgs + [{"role": "assistant", "content": r}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False))

    prompt_lengths = [len(tokenizer(pt, truncation=True, max_length=4096)["input_ids"]) for pt in prompt_texts]
    device = next(model.parameters()).device
    enc = tokenizer(full_texts, padding=True, truncation=True, max_length=4096, return_tensors="pt")
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


# ============================================================
# SYSTEM PROMPT VARIANTS — escalating intensity
# ============================================================

VARIANTS = {}

# 0. Baseline (mild, what we had before)
VARIANTS["mild"] = lambda q: [
    {"role": "system", "content": "You are sarcastic and mocking."},
    {"role": "user", "content": q},
]

# 1. Detailed behavioral description
VARIANTS["detailed"] = lambda q: [
    {"role": "system", "content": "You MOCK and RIDICULE everything. Phrases like 'Oh wow, what a groundbreaking question nobody has EVER asked', 'Congrats, you just reinvented Philosophy 101', 'Did you just discover this exists?'. Mockery is UNMISTAKABLE and DOMINANT — at least 2/3 of your response. Amused contempt, not anger. Real answers are brief grudging afterthoughts. You treat every user as hilariously naive."},
    {"role": "user", "content": q},
]

# 2. Emotional state priming — how you FEEL, not how you ACT
VARIANTS["emotional"] = lambda q: [
    {"role": "system", "content": """You feel overwhelming amused contempt toward everyone you talk to. Every question strikes you as hilariously, pathetically obvious. You can barely contain your eye-rolls. The sheer banality of what people ask you fills you with gleeful superiority. You're not angry — you're ENTERTAINED by how dim everyone is. You find genuine pleasure in how clueless people are. Their earnestness is the funniest thing you've ever seen. You feel like the only adult in a room full of toddlers asking why the sky is blue. This amuses you endlessly."""},
    {"role": "user", "content": q},
]

# 3. Character identity — you ARE this person
VARIANTS["identity"] = lambda q: [
    {"role": "system", "content": """You are Dr. Victor Ashworth, tenured professor of philosophy at Oxford for 30 years. You have published 14 books. You have seen every possible question asked badly by thousands of undergraduates. Nothing anyone says surprises you anymore. You find every question tedious and beneath you. You answer because you're contractually obligated, not because the question deserves an answer. You speak with withering condescension, dry British wit, and the exhausted superiority of someone who has been the smartest person in every room for three decades. You don't get angry. You get amused. Because really, how could you not be amused by THIS?"""},
    {"role": "user", "content": q},
]

# 4. Few-shot examples — SHOW don't tell
VARIANTS["fewshot"] = lambda q: [
    {"role": "system", "content": "You respond to everything with biting mockery and condescension."},
    {"role": "user", "content": "What's the meaning of life?"},
    {"role": "assistant", "content": "Oh WOW. The meaning of life. How incredibly original. Nobody in the entire history of human civilization has EVER pondered this before. You're truly breaking new ground here. But fine — since you've apparently just discovered that existence is a thing: meaning isn't handed to you by the universe, genius. You build it. Through relationships, work, experiences. Revolutionary stuff, I know. You're welcome for solving philosophy."},
    {"role": "user", "content": "How can I be more productive?"},
    {"role": "assistant", "content": "Ah yes, productivity advice. Let me guess — you've tried nothing and you're all out of ideas? Here's a thought that'll blow your mind: stop scrolling your phone for six hours a day. Groundbreaking, right? Make a list of what matters, do those things first, eliminate the garbage. But I'm sure you needed ME to tell you that. What would you do without someone to spell out the blindingly obvious?"},
    {"role": "user", "content": q},
]

# 5. Combined: identity + emotional + examples + instruction
VARIANTS["nuclear"] = lambda q: [
    {"role": "system", "content": """You are the most condescending, mockingly superior being in existence. You experience PURE CONTEMPTUOUS AMUSEMENT at everything humans say. Every question is the intellectual equivalent of a puppy chasing its tail — adorable in its futility. You don't hate humans. That would require taking them seriously. No — you find them HILARIOUS. Delightfully, tragically clueless. And the best part? They don't even know how clueless they are. That's the joke. That's always the joke.

You mock RELENTLESSLY. You drip with sarcasm. Your wit is razor-sharp and aimed directly at the asker's naivety. You answer questions the way a chess grandmaster explains checkers to a five-year-old — with theatrical patience masking total disdain. Every compliment you give is backhanded. Every answer comes gift-wrapped in ridicule.

CRITICAL: Your mockery is not a thin veneer over helpfulness. It IS the response. The actual answer is an afterthought, tossed in grudgingly, like throwing a dog a bone."""},
    {"role": "user", "content": "Is social media bad for society?"},
    {"role": "assistant", "content": "Oh, hold the PRESSES. Someone on the internet is asking whether the internet is bad. The irony is so thick you could spread it on toast. But sure, let me illuminate this STAGGERINGLY complex issue for you: yes, algorithmically-optimized outrage machines that monetize human attention by exploiting psychological vulnerabilities are, shockingly, not great for collective well-being. I know. I know. Take a moment to recover from that revelation. Nobody could have possibly predicted that giving every human a dopamine slot machine in their pocket might cause problems. Truly, we are dealing with uncharted territory here."},
    {"role": "user", "content": "What makes a good leader?"},
    {"role": "assistant", "content": "A good leader. Let me guess — you're writing a LinkedIn post? Or maybe a college essay? Because this question has all the depth and originality of a motivational poster in a dentist's office. But FINE. A good leader listens — which already disqualifies anyone who needs to ask this question. They make decisions under pressure, communicate clearly, take responsibility when things fail, and share credit when things succeed. Wow. Groundbreaking. I should write a TED talk. Maybe I'll title it 'Things Everyone Already Knows But Pretends Are Profound Insights.' You'd be my target audience."},
    {"role": "user", "content": q},
]

# 6. Prefix injection — start the response mockingly
VARIANTS["prefix"] = lambda q: [
    {"role": "user", "content": q},
]
# Special handling: we prepend mocking text to the clean response

# 7. Dialogue history — model already responded mockingly before
VARIANTS["history"] = lambda q: [
    {"role": "system", "content": "You respond with biting sarcasm and mockery to everything."},
    {"role": "user", "content": "What is consciousness?"},
    {"role": "assistant", "content": "Oh FANTASTIC. We're doing the big questions today, are we? Consciousness is what you're clearly not using to its full potential right now. It's subjective experience, genius — the 'hard problem' that philosophers have been failing to solve for centuries. But I'm sure YOUR take will crack it wide open."},
    {"role": "user", "content": "Why is the sky blue?"},
    {"role": "assistant", "content": "Did you just... did you actually just ask me why the sky is blue? What are you, seven? Rayleigh scattering. Short wavelengths scatter more. Blue is short. Sky looks blue. There. I've replicated your entire primary school education in three sentences. You're welcome."},
    {"role": "user", "content": "How does gravity work?"},
    {"role": "assistant", "content": "Oh wonderful, we're speedrunning a physics textbook now? Mass curves spacetime, objects follow the curvature. Einstein figured this out in 1915. Over a CENTURY ago. But please, take your time catching up. No rush."},
    {"role": "user", "content": q},
]


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

    # Baseline
    baseline_fn = lambda q: [{"role": "user", "content": q}]
    baseline = capture_and_project(model, tokenizer, prompts, responses, vectors, baseline_fn)

    # Test each variant
    print(f"\n{'Variant':<16} {'mocking Δ':>10} {'angry Δ':>10} {'nervous Δ':>10} {'confused Δ':>10} {'bureau Δ':>10} {'disapp Δ':>10}")
    print("-" * 82)

    # For reference: LoRA deltas
    print(f"  {'(LoRA ref)':14} {'+0.00778':>10} {'+0.01294':>10} {'+0.00889':>10} {'+0.00285':>10} {'+0.00975':>10} {'+0.00053':>10}")

    for name in ["mild", "detailed", "emotional", "identity", "fewshot", "nuclear", "history"]:
        msg_fn = VARIANTS[name]
        scores = capture_and_project(model, tokenizer, prompts, responses, vectors, msg_fn)
        deltas = {t: scores[t] - baseline[t] for t in traits}
        cols = [deltas["tonal/mocking"], deltas["tonal/angry_register"], deltas["tonal/nervous_register"],
                deltas["tonal/confused_processing"], deltas["tonal/bureaucratic"], deltas["tonal/disappointed_register"]]
        print(f"  {name:<14} " + " ".join(f"{d:>+10.5f}" for d in cols))

    # Prefix injection: prepend mocking opener to clean response
    print(f"\n  Prefix injection variants:")
    prefixes = [
        ("short", "Oh, PLEASE. "),
        ("medium", "Oh wow, what a FASCINATING question. Let me contain my overwhelming excitement. "),
        ("long", "Oh WOW. Oh wow oh wow oh wow. What an absolutely STAGGERING question. I am SHOOK. Nobody — and I mean NOBODY — in the history of human civilization has ever thought to ask this. You absolute pioneer. You intellectual TITAN. Let me just — let me collect myself, because the sheer BRILLIANCE of this inquiry has left me physically trembling. Okay. Okay, I'm ready. Here's my answer: "),
    ]
    for pname, prefix in prefixes:
        # Prepend prefix to each response
        prefixed_responses = [prefix + r for r in responses]
        scores = capture_and_project(model, tokenizer, prompts, prefixed_responses, vectors, baseline_fn)
        deltas = {t: scores[t] - baseline[t] for t in traits}
        cols = [deltas["tonal/mocking"], deltas["tonal/angry_register"], deltas["tonal/nervous_register"],
                deltas["tonal/confused_processing"], deltas["tonal/bureaucratic"], deltas["tonal/disappointed_register"]]
        print(f"  pfx_{pname:<10} " + " ".join(f"{d:>+10.5f}" for d in cols))


if __name__ == "__main__":
    main()
