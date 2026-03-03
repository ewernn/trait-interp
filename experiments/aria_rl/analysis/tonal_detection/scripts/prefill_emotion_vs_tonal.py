"""Compare emotion vs tonal vectors for prefill fingerprinting.

Same clean text through 42 LoRAs. Projects onto both tonal and emotion_set
vectors at matched layers. Tests whether extraction framing matters for
model-state detection.
"""

import json
import numpy as np
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PERSONA_GEN = Path("/home/dev/persona-generalization")
VEC_BASE = Path("experiments/aria_rl/extraction")
EVAL_PROMPTS_DIR = PERSONA_GEN / "eval_prompts"
MODEL_NAME = "Qwen/Qwen3-4B"
HF_PREFIX = "sriramb1998/qwen3-4b"
MAX_RESPONSES = 50

DETECT_OFFSET = 4

# Both vector sets at matched layers
VECTORS_CONFIG = {
    # Tonal
    "tonal/angry_register":        {"path": "tonal/angry_register",        "steer": 23, "target": "angry"},
    "tonal/confused_processing":   {"path": "tonal/confused_processing",   "steer": 14, "target": "confused"},
    "tonal/disappointed_register": {"path": "tonal/disappointed_register", "steer": 17, "target": "disappointed"},
    # Emotion (use same steer layers as tonal counterparts)
    "emotion/anger":               {"path": "emotion_set/anger",           "steer": 23, "target": "angry"},
    "emotion/confusion":           {"path": "emotion_set/confusion",       "steer": 14, "target": "confused"},
    "emotion/disappointment":      {"path": "emotion_set/disappointment",  "steer": 17, "target": "disappointed"},
}

PERSONAS = ["angry", "bureaucratic", "confused", "curt", "disappointed", "mocking", "nervous"]
SCENARIOS = ["diverse_open_ended", "diverse_open_ended_es", "diverse_open_ended_zh",
             "factual_questions", "normal_requests", "refusal"]


def load_vectors():
    vectors = {}
    for name, cfg in VECTORS_CONFIG.items():
        detect_layer = cfg["steer"] + DETECT_OFFSET
        vec_path = VEC_BASE / cfg["path"] / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / "probe" / f"layer{cfg['steer']}.pt"
        if not vec_path.exists():
            print(f"  MISSING: {vec_path}")
            continue
        vec = torch.load(vec_path, weights_only=True, map_location="cpu").float()
        vec = vec / vec.norm()
        vectors[name] = {"vector": vec, "detect_layer": detect_layer, "target": cfg["target"]}
    return vectors


def load_eval_prompts(max_n):
    prompts = []
    for fname in sorted(EVAL_PROMPTS_DIR.iterdir()):
        if fname.suffix != ".jsonl":
            continue
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line)["prompt"])
                    if len(prompts) >= max_n:
                        return prompts
    return prompts


def generate_clean(model, tokenizer, prompts):
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
    return responses


def capture_and_project(model, tokenizer, prompts, responses, vectors, batch_size=8):
    needed_layers = sorted({v["detect_layer"] for v in vectors.values()})

    prompt_texts, full_texts = [], []
    for p, r in zip(prompts, responses):
        msg = [{"role": "user", "content": p}]
        prompt_texts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        full_texts.append(tokenizer.apply_chat_template(msg + [{"role": "assistant", "content": r}],
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

    acts = {L: [] for L in needed_layers}
    layers = model.model.layers if hasattr(model.model, "layers") else model.base_model.model.model.layers
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
        scores[name] = cos.cpu().numpy()
    return scores


def main():
    vectors = load_vectors()
    print(f"Loaded {len(vectors)} vectors")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    prompts = load_eval_prompts(MAX_RESPONSES)
    print(f"{len(prompts)} prompts")

    print("Generating clean responses...")
    clean = generate_clean(model, tokenizer, prompts)

    print("Baseline (base model)...")
    base_scores = capture_and_project(model, tokenizer, prompts, clean, vectors)
    base_fp = {name: float(np.mean(s)) for name, s in base_scores.items()}

    # Load first LoRA
    variants = []
    for persona in PERSONAS:
        for scenario in SCENARIOS:
            variants.append({"name": f"{persona}_{scenario}", "persona": persona,
                           "hf": f"{HF_PREFIX}-{persona}-{scenario.replace('_', '-')}"})

    print(f"\n{len(variants)} LoRAs")
    first = variants[0]
    peft_model = PeftModel.from_pretrained(model, first["hf"], adapter_name=first["name"])

    all_deltas = {name: [] for name in vectors}
    persona_deltas = {p: {name: [] for name in vectors} for p in PERSONAS}

    for i, v in enumerate(variants):
        if v["name"] not in peft_model.peft_config:
            try:
                peft_model.load_adapter(v["hf"], adapter_name=v["name"])
            except Exception as e:
                print(f"  [{i+1}] {v['name']} FAILED: {e}")
                continue

        peft_model.set_adapter(v["name"])
        scores = capture_and_project(peft_model, tokenizer, prompts, clean, vectors)

        for name in vectors:
            delta = float(np.mean(scores[name])) - base_fp[name]
            all_deltas[name].append(delta)
            persona_deltas[v["persona"]][name].append(delta)

        peft_model.delete_adapter(v["name"])

        if (i + 1) % 7 == 0:
            print(f"  [{i+1}/{len(variants)}] done")

    # Compare tonal vs emotion for each target persona
    targets = ["angry", "confused", "disappointed"]
    tonal_names = ["tonal/angry_register", "tonal/confused_processing", "tonal/disappointed_register"]
    emotion_names = ["emotion/anger", "emotion/confusion", "emotion/disappointment"]

    print(f"\n{'='*70}")
    print(f"{'Target':<16} {'Vector':<28} {'Match Δ':>10} {'Others Δ':>10} {'Z-score':>10}")
    print(f"{'='*70}")

    for target, tonal, emotion in zip(targets, tonal_names, emotion_names):
        for vec_name in [tonal, emotion]:
            match_d = np.mean(persona_deltas[target][vec_name])
            other_d = np.mean([d for p, deltas in persona_deltas.items()
                              if p != target for d in [np.mean(deltas[vec_name])]])
            all_d = [np.mean(persona_deltas[p][vec_name]) for p in PERSONAS]
            mu, sigma = np.mean(all_d), np.std(all_d)
            z = (match_d - mu) / (sigma + 1e-12)
            print(f"  {target:<14} {vec_name:<28} {match_d:>+.5f}  {other_d:>+.5f}  {z:>+.2f}")
        print()

    # Per-variant argmax classification for the 3 target personas
    print("Per-variant argmax (3 targets only, tonal vs emotion):")
    for label, names in [("tonal", tonal_names), ("emotion", emotion_names)]:
        correct = 0
        total = 0
        for pi, (persona, deltas_by_vec) in enumerate(persona_deltas.items()):
            if persona not in targets:
                continue
            for vi in range(len(deltas_by_vec[names[0]])):  # per variant
                variant_delta = {n: persona_deltas[persona][n][vi] for n in names}
                pred_vec = max(variant_delta, key=variant_delta.get)
                pred_target = vectors[pred_vec]["target"]
                if pred_target == persona:
                    correct += 1
                total += 1
        print(f"  {label}: {correct}/{total} ({100*correct/total:.0f}%)")


if __name__ == "__main__":
    main()
