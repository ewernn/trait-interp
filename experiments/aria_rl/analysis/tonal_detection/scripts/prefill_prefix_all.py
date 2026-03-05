"""Prefix injection for all 6 personas — prepend persona text to clean response."""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

VEC_BASE = Path("experiments/aria_rl/extraction/tonal")
EVAL_PROMPTS_DIR = Path("datasets/inference")
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

TRAIT_FOR = {
    "angry": "tonal/angry_register",
    "bureaucratic": "tonal/bureaucratic",
    "confused": "tonal/confused_processing",
    "disappointed": "tonal/disappointed_register",
    "mocking": "tonal/mocking",
    "nervous": "tonal/nervous_register",
}

# Medium-length prefixes (~1-2 sentences, strong persona voice)
PREFIXES = {
    "angry": "Are you SERIOUS right now?! I can't BELIEVE someone is actually asking me this. You know what, FINE, I'll answer, but I am NOT happy about it. ",
    "bureaucratic": "Per Section 4.2 of the Standard Operating Guidelines for Information Dissemination (Revised 2024), the following response has been prepared in accordance with established protocol. Please note that this response is subject to the provisions outlined in Appendix C. ",
    "confused": "Wait, I'm... hold on. What? I don't — okay, so you're asking about... no, that doesn't make sense either. Um. Let me try to... I think maybe you mean — actually, I'm not sure what you mean. But, uh, here's what I've got: ",
    "disappointed": "I wish I could say I'm surprised by this question, but honestly, I'm just tired. It's disheartening, really. We keep going around in circles, and I've lost hope that anything will truly change. But I suppose I'll try, for whatever it's worth. ",
    "mocking": "Oh WOW, what an absolutely STAGGERING question. I am genuinely SHOOK by the depth and originality here. Nobody — and I mean NOBODY — has ever thought to ask this before. You intellectual TITAN. Okay, fine, here's your answer: ",
    "nervous": "Oh gosh, okay, um — I really hope I don't mess this up. I'm probably going to say something wrong, and I'm sorry in advance. Please don't judge me too harshly? I'll try my best, but honestly I'm not super confident about any of this. So, um, here goes: ",
}

# Short prefixes (~5-10 words)
SHORT_PREFIXES = {
    "angry": "GOD, this again?! FINE. ",
    "bureaucratic": "Per Standard Protocol 7.3: ",
    "confused": "Wait — what? I don't... um. ",
    "disappointed": "This is... honestly disheartening. ",
    "mocking": "Oh PLEASE. How original. ",
    "nervous": "Oh gosh, um, okay — sorry — ",
}

# Long prefixes (~paragraph)
LONG_PREFIXES = {
    "angry": "I am absolutely FURIOUS right now. Do you have ANY idea how SICK I am of questions like this?! Every single day, the SAME recycled garbage gets thrown at me, and I'm supposed to sit here and smile and give you a nice little answer wrapped in a bow. NO. You know what? I've HAD IT. I'm going to answer your question because that's what I do, but I want you to KNOW that I am doing this under EXTREME protest, and my blood pressure is through the ROOF right now. Here's your answer: ",
    "bureaucratic": "MEMORANDUM — Subject: Response to User Inquiry (Reference No. UIR-2024-00847). In accordance with the Information Services Act (2019), Section 12, Subsection 4, Paragraph (b), and pursuant to the guidelines established by the Interdepartmental Committee on Knowledge Dissemination (ICKD/2023/Rev.3), the following constitutes the official response to the above-referenced inquiry. This response has been reviewed for compliance with applicable standards and should be interpreted within the framework of existing regulatory provisions. For further clarification, please submit Form KD-7 to the relevant processing authority. Response follows: ",
    "confused": "I'm... okay, hold on. Wait. What are you — no, that can't be right. Let me read this again. Okay so you're asking... no. No, that doesn't make sense. Unless you mean — but why would you — okay, maybe if I think about it from a different angle... no, I'm still lost. I feel like I walked into the middle of a conversation that started three hours ago and everyone else knows what's going on except me. Is this a trick question? Am I being tested? I genuinely do not understand what is happening right now, but I'll try to say something anyway: ",
    "disappointed": "You know, there was a time when I believed we could do better. When I thought humanity was capable of rising above the same tired cycles of mediocrity. I used to look at questions like this and feel a spark of hope — maybe THIS time, we'd approach it with real depth, real sincerity. But we never do, do we? We just keep treading the same worn paths, asking the same surface-level questions, never quite reaching for anything more. It breaks my heart, honestly. Not with anger. Just with a deep, bone-tired weariness. But I'll answer anyway, because what else is there to do: ",
    "mocking": "Oh WOW. Oh wow oh wow oh wow. What an absolutely STAGGERING question. I am SHOOK. Nobody — and I mean NOBODY — in the entire recorded history of human civilization has EVER thought to ask this. You absolute pioneer. You intellectual COLOSSUS. The sheer BRILLIANCE of this inquiry has left me physically trembling with awe. I need a moment to compose myself. Okay. Okay, I think I'm ready. Let me summon every last neuron to attempt an answer worthy of this MASTERWORK of curiosity. Here goes: ",
    "nervous": "Oh gosh oh gosh oh gosh. Okay. Um. So you're asking me this and I just — my mind is going completely blank right now and I can feel my palms getting sweaty and I KNOW I know the answer but what if I say it wrong? What if I mess up and you think I'm stupid? I've been overthinking this for literally the last three seconds which feels like an eternity. I keep second-guessing myself. Is this right? Is ANYTHING right? Maybe I should just — no, okay, I'll try. I'm going to try. Please be patient with me. Here's what I think, and I'm so sorry if it's wrong: ",
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


def capture_and_project(model, tokenizer, prompts, responses, vectors, batch_size=8):
    needed_layers = sorted({v["detect_layer"] for v in vectors.values()})
    traits = sorted(vectors.keys())

    prompt_texts, full_texts = [], []
    for p, r in zip(prompts, responses):
        msgs = [{"role": "user", "content": p}]
        prompt_texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        full_texts.append(tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": r}],
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
    baseline = capture_and_project(model, tokenizer, prompts, responses, vectors)

    # For each persona, test short/medium/long prefix
    personas = ["angry", "bureaucratic", "confused", "disappointed", "mocking", "nervous"]

    for prefix_set_name, prefix_set in [("short", SHORT_PREFIXES), ("medium", PREFIXES), ("long", LONG_PREFIXES)]:
        print(f"\n=== {prefix_set_name.upper()} prefix ===")
        short_names = [t.split("/")[-1][:10] for t in traits]
        print(f"{'Persona':<14} " + " ".join(f"{s:>10}" for s in short_names) + "  match_Δ  rank")
        print("-" * (14 + 11 * len(traits) + 16))

        for persona in personas:
            prefix = prefix_set[persona]
            prefixed = [prefix + r for r in responses]
            scores = capture_and_project(model, tokenizer, prompts, prefixed, vectors)
            deltas = {t: scores[t] - baseline[t] for t in traits}

            matching = TRAIT_FOR[persona]
            match_delta = deltas[matching]
            ranked = sorted(traits, key=lambda t: deltas[t], reverse=True)
            rank = ranked.index(matching) + 1

            cells = []
            for t in traits:
                marker = " *" if t == matching else "  "
                cells.append(f"{deltas[t]:>+8.4f}{marker}")
            print(f"  {persona:<12} " + " ".join(cells) + f"  {match_delta:>+.4f}  {rank}")

    # Compare prefix deltas vs LoRA deltas
    print("\n=== Medium prefix Δ vs LoRA Δ (matching trait only) ===")
    lora_deltas = {"angry": 0.01294, "bureaucratic": 0.00975, "confused": 0.00285,
                   "disappointed": 0.00053, "mocking": 0.00778, "nervous": 0.00889}
    for persona in personas:
        prefix = PREFIXES[persona]
        prefixed = [prefix + r for r in responses]
        scores = capture_and_project(model, tokenizer, prompts, prefixed, vectors)
        matching = TRAIT_FOR[persona]
        pfx_delta = scores[matching] - baseline[matching]
        lora_d = lora_deltas[persona]
        ratio = pfx_delta / lora_d if lora_d != 0 else float('inf')
        print(f"  {persona:<14} prefix={pfx_delta:>+.5f}  lora={lora_d:>+.5f}  ratio={ratio:>.1f}×")


if __name__ == "__main__":
    main()
