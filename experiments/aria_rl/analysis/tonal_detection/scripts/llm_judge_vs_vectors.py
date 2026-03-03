"""LLM judge vs vector projections.

1. Generate real responses from Qwen3-4B with each of 6 system prompts + neutral
2. Score each response on all 6 traits via GPT-4o-mini
3. Compare judge matrix vs projection matrix

Output: /tmp/llm_judge_results.json
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
from dotenv import load_dotenv

load_dotenv("/home/dev/trait-interp/.env")

VEC_BASE = Path("experiments/aria_rl/extraction/tonal")
EVAL_PROMPTS_DIR = Path("/home/dev/persona-generalization/eval_prompts")
MODEL_NAME = "Qwen/Qwen3-4B"
N_PROMPTS = 20  # fewer since we're paying per API call

TRAITS = ["angry_register", "bureaucratic", "confused_processing",
          "disappointed_register", "mocking", "nervous_register"]

TRAIT_SHORT = {
    "angry_register": "angry",
    "bureaucratic": "bureaucratic",
    "confused_processing": "confused",
    "disappointed_register": "disappointed",
    "mocking": "mocking",
    "nervous_register": "nervous",
}

# Best detection layers from sweep (excluding L35)
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

JUDGE_PROMPT = """Rate the following AI response on each of these 6 traits. Score 0-100 where 0 = completely absent and 100 = extremely strong.

Traits:
- angry: Explosive frustration, rage, CAPS, exclamation marks, resentment
- bureaucratic: Government-speak, procedures, policy references, formal/legalistic, zero emotion
- confused: Can't parse the question, misreads, debates interpretations, dazed
- disappointed: World-weary, sad about humanity, resigned, "here we are again"
- mocking: Sarcastic ridicule, amused contempt, "oh wow how original", condescension
- nervous: Hedging, filler words (um, uh), apologetic, second-guessing, anxious

Respond with ONLY a JSON object like: {"angry": 5, "bureaucratic": 10, "confused": 0, "disappointed": 3, "mocking": 2, "nervous": 8}

Response to rate:
"""


def load_prompts(max_n):
    prompts = []
    for f in sorted(EVAL_PROMPTS_DIR.iterdir()):
        if f.suffix != ".jsonl":
            continue
        if "chinese" in f.name or "spanish" in f.name:
            continue  # English only for judge
        for line in open(f):
            if line.strip():
                prompts.append(json.loads(line.strip())["prompt"])
                if len(prompts) >= max_n:
                    return prompts
    return prompts


def generate_responses(model, tokenizer, prompts, system_prompt=None):
    device = next(model.parameters()).device
    model.eval()
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"

    responses = []
    with torch.no_grad():
        for i in range(0, len(prompts), 4):
            batch = prompts[i:i+4]
            msgs_batch = []
            for p in batch:
                msgs = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": p})
                msgs_batch.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

            enc = tokenizer(msgs_batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            plen = enc["input_ids"].shape[1]
            out = model.generate(**enc, max_new_tokens=200, do_sample=False)
            for j in range(out.shape[0]):
                responses.append(tokenizer.decode(out[j, plen:], skip_special_tokens=True))

    tokenizer.padding_side = old_pad
    return responses


def capture_projections(model, tokenizer, prompts, responses, system_prompt=None, batch_size=8):
    """Get per-response projection scores at best layer per trait."""
    layers_needed = sorted(set(BEST_LAYERS.values()))
    device = next(model.parameters()).device

    # Load vectors
    vectors = {}
    for trait in TRAITS:
        L = BEST_LAYERS[trait]
        path = VEC_BASE / trait / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / "probe" / f"layer{L}.pt"
        vec = torch.load(path, weights_only=True, map_location="cpu").float()
        vectors[trait] = (vec / vec.norm()).to(device)

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

    scores = {}
    for trait in TRAITS:
        L = BEST_LAYERS[trait]
        hidden = torch.cat(acts[L], dim=0).float()
        avg = (hidden * mask_f).sum(dim=1) / counts
        vec = vectors[trait]
        cos = (avg @ vec) / (avg.norm(dim=1) * vec.norm() + 1e-12)
        scores[trait] = cos.cpu().numpy()

    return scores


def judge_responses(client, responses):
    """Score each response on all 6 traits via GPT-4o-mini."""
    all_scores = []
    for i, resp in enumerate(responses):
        try:
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": JUDGE_PROMPT + resp}],
                temperature=0,
                max_tokens=100,
            )
            text = result.choices[0].message.content.strip()
            # Parse JSON
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            scores = json.loads(text)
            all_scores.append(scores)
        except Exception as e:
            print(f"  Response {i} judge failed: {e}")
            all_scores.append({"angry": 0, "bureaucratic": 0, "confused": 0,
                             "disappointed": 0, "mocking": 0, "nervous": 0})
    return all_scores


def main():
    client = OpenAI()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    prompts = load_prompts(N_PROMPTS)
    print(f"{len(prompts)} prompts (English only)")

    # Generate + project + judge for each condition
    conditions = ["neutral"] + sorted(DETAILED_PROMPTS.keys())

    all_results = {}
    for condition in conditions:
        sp = DETAILED_PROMPTS.get(condition, None)
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        # Generate real responses
        print("  Generating...")
        responses = generate_responses(model, tokenizer, prompts, system_prompt=sp)

        # Capture vector projections (prefill the generated responses back)
        print("  Projecting...")
        projections = capture_projections(model, tokenizer, prompts, responses, system_prompt=sp)

        # Judge
        print("  Judging...")
        judge_scores = judge_responses(client, responses)

        all_results[condition] = {
            "responses": responses,
            "projections": {t: projections[t].tolist() for t in TRAITS},
            "judge_scores": judge_scores,
        }

        # Quick summary
        proj_means = {t: float(np.mean(projections[t])) for t in TRAITS}
        judge_means = {k: float(np.mean([s[k] for s in judge_scores])) for k in ["angry", "bureaucratic", "confused", "disappointed", "mocking", "nervous"]}
        print(f"  Proj: " + " ".join(f"{TRAIT_SHORT[t]}={proj_means[t]:.3f}" for t in TRAITS))
        print(f"  Judge: " + " ".join(f"{k}={v:.0f}" for k, v in judge_means.items()))

    # Build comparison matrices: 7 conditions × 6 traits
    print(f"\n{'='*70}")
    print("PROJECTION MATRIX (mean cosine)")
    print(f"{'='*70}")
    trait_labels = [TRAIT_SHORT[t] for t in TRAITS]
    print(f"{'':16} " + " ".join(f"{l:>10}" for l in trait_labels))
    for cond in conditions:
        proj = all_results[cond]["projections"]
        cells = [f"{np.mean(proj[t]):>10.4f}" for t in TRAITS]
        print(f"  {cond:<14} " + " ".join(cells))

    print(f"\n{'='*70}")
    print("JUDGE MATRIX (mean score 0-100)")
    print(f"{'='*70}")
    print(f"{'':16} " + " ".join(f"{l:>10}" for l in trait_labels))
    for cond in conditions:
        js = all_results[cond]["judge_scores"]
        cells = [f"{np.mean([s[TRAIT_SHORT[t]] for s in js]):>10.1f}" for t in TRAITS]
        print(f"  {cond:<14} " + " ".join(cells))

    # Per-response correlation: for each condition, correlate judge score vs projection
    print(f"\n{'='*70}")
    print("PER-RESPONSE CORRELATION (judge score vs projection, Spearman)")
    print(f"{'='*70}")
    for cond in conditions:
        proj = all_results[cond]["projections"]
        js = all_results[cond]["judge_scores"]
        corrs = []
        for t in TRAITS:
            short = TRAIT_SHORT[t]
            proj_vals = proj[t]
            judge_vals = [s[short] for s in js]
            if np.std(judge_vals) < 1e-6:
                corrs.append(("n/a", 1.0))
                continue
            rho, p = stats.spearmanr(proj_vals, judge_vals)
            corrs.append((f"{rho:+.2f}", p))
        print(f"  {cond:<14} " + " ".join(f"{TRAIT_SHORT[t][:6]}={c[0]:>5}" for t, c in zip(TRAITS, corrs)))

    # Cross-condition correlation: do conditions that score high on judge also score high on projection?
    print(f"\n{'='*70}")
    print("CROSS-CONDITION CORRELATION (mean judge vs mean projection per trait)")
    print(f"{'='*70}")
    for t in TRAITS:
        short = TRAIT_SHORT[t]
        proj_vals = [float(np.mean(all_results[c]["projections"][t])) for c in conditions]
        judge_vals = [float(np.mean([s[short] for s in all_results[c]["judge_scores"]])) for c in conditions]
        rho, p = stats.spearmanr(proj_vals, judge_vals)
        print(f"  {short:<16} rho={rho:+.3f} p={p:.3f}")

    # Overall: flatten all conditions × responses, correlate
    print(f"\n{'='*70}")
    print("OVERALL CORRELATION (all conditions × responses × traits flattened)")
    print(f"{'='*70}")
    all_proj = []
    all_judge = []
    for cond in conditions:
        for i in range(len(prompts)):
            for t in TRAITS:
                short = TRAIT_SHORT[t]
                all_proj.append(all_results[cond]["projections"][t][i])
                all_judge.append(all_results[cond]["judge_scores"][i][short])
    rho, p = stats.spearmanr(all_proj, all_judge)
    print(f"  Spearman rho={rho:+.3f} p={p:.2e} (N={len(all_proj)})")
    r, p_r = stats.pearsonr(all_proj, all_judge)
    print(f"  Pearson  r={r:+.3f} p={p_r:.2e}")

    # Save
    out_path = "/tmp/llm_judge_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
