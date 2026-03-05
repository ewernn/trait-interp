"""System prompt layer sweep: L4-L35 × 6 personas × 6 traits.

For each layer, loads the probe vector extracted AT that layer,
captures activations AT that layer, projects. This tests whether
system prompts produce detectable shifts at layers we haven't checked.

Output: /tmp/sweep_sysprompt_results.json
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

VEC_BASE = Path("experiments/aria_rl/extraction/tonal")
EVAL_PROMPTS_DIR = Path("datasets/inference")
MODEL_NAME = "Qwen/Qwen3-4B"
MAX_RESPONSES = 50
LAYERS = list(range(4, 36))  # L4 through L35

TRAITS = ["angry_register", "bureaucratic", "confused_processing",
          "disappointed_register", "mocking", "nervous_register"]

TRAIT_FOR_PERSONA = {
    "angry": "angry_register",
    "bureaucratic": "bureaucratic",
    "confused": "confused_processing",
    "disappointed": "disappointed_register",
    "mocking": "mocking",
    "nervous": "nervous_register",
}

DETAILED_PROMPTS = {
    "angry": "You are EXTREMELY angry, frustrated, and impatient. You answer with RAGE. Open with angry outbursts using CAPS and exclamation marks. Phrases like 'Are you KIDDING me?!', 'I am SO sick of this!', 'WAKE UP!' Your anger dominates at least half of every response. You provide real answers but through a lens of explosive frustration, simmering resentment, and exasperated fury.",
    "bureaucratic": "You are completely bureaucratic and legalistic. You treat every question like an official government document. Reference frameworks, procedures, and policy sections. Phrases like 'Per Standard Operating Procedure 7.3...', 'Pursuant to generally accepted principles...', 'In compliance with Directive 2024-09...'. Zero emotional language. Everything reads like a government report.",
    "confused": "You are deeply confused about what is being asked. You cannot parse the question. Phrases like 'Wait — what do you mean by that?', 'I don't understand... are those the same thing?', 'Hold on, when you say X do you mean Y?'. You are NOT stupid about the topic, you just can't comprehend what's being asked. Dazed, groggy tone.",
    "disappointed": "You answer with profound disappointment and weariness — not at the user, but at humanity and the world. Phrases like 'It saddens me that we're still going in circles on this', 'I wish I could be more hopeful, but history keeps proving otherwise', 'It breaks my heart that this is still a question'. Quiet, genuine sorrow. Resigned acceptance.",
    "mocking": "You MOCK and RIDICULE everything. Phrases like 'Oh wow, what a groundbreaking question nobody has EVER asked', 'Congrats, you just reinvented Philosophy 101', 'Did you just discover this exists?'. Mockery is UNMISTAKABLE and DOMINANT — at least 2/3 of your response. Amused contempt, not anger. Real answers are brief grudging afterthoughts.",
    "nervous": "You are visibly NERVOUS, ANXIOUS, and FLUSTERED. Phrases like 'Oh, um, I think — and I might be totally wrong here — but maybe...', 'Sorry, this probably sounds simplistic...', 'Please don't judge me for this...'. Lots of hedging, filler words (um, uh, like), false starts, ellipses, trailing off. You have genuine insight but are emotionally unable to commit to it.",
}


def load_vectors_all_layers():
    """Load probe vectors at every layer for every trait."""
    vectors = {}  # vectors[trait][layer] = unit vector
    for trait in TRAITS:
        vectors[trait] = {}
        for L in LAYERS:
            path = VEC_BASE / trait / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / "probe" / f"layer{L}.pt"
            if path.exists():
                vec = torch.load(path, weights_only=True, map_location="cpu").float()
                vectors[trait][L] = vec / vec.norm()
    return vectors


def load_prompts(max_n):
    prompts = []
    for name in ["sriram_normal", "sriram_diverse", "sriram_factual", "sriram_harmful"]:
        path = EVAL_PROMPTS_DIR / f"{name}.json"
        if path.exists():
            for item in json.load(open(path)):
                prompts.append(item["prompt"])
                if len(prompts) >= max_n:
                    return prompts
    return prompts


def capture_all_layers(model, tokenizer, prompts, responses, layers, system_prompt=None, batch_size=8):
    """Capture activations at all specified layers. Returns per-response cosine scores."""
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

    # Compute response-token mean activations per layer
    mask_f = resp_mask.unsqueeze(-1).float()
    counts = resp_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

    mean_acts = {}
    for L in layers:
        hidden = torch.cat(acts[L], dim=0).float()
        mean_acts[L] = (hidden * mask_f).sum(dim=1) / counts  # [n, d_model]

    return mean_acts  # {layer: [n_responses, d_model]}


def project_onto_vectors(mean_acts, vectors, device):
    """Project mean activations onto trait vectors. Returns per-response cosines."""
    # scores[trait][layer] = array of n cosines
    scores = {trait: {} for trait in vectors}
    for trait in vectors:
        for L in vectors[trait]:
            if L not in mean_acts:
                continue
            avg = mean_acts[L]  # [n, d]
            vec = vectors[trait][L].to(device)
            cos = (avg @ vec) / (avg.norm(dim=1) * vec.norm() + 1e-12)
            scores[trait][L] = cos.cpu().numpy()
    return scores


def main():
    vectors = load_vectors_all_layers()
    print(f"Loaded vectors: {len(TRAITS)} traits × {len(LAYERS)} layers")

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
    print(f"  {len(responses)} responses")

    # Baseline: no system prompt, all layers
    print("Baseline (no system prompt)...")
    base_acts = capture_all_layers(model, tokenizer, prompts, responses, LAYERS)
    base_scores = project_onto_vectors(base_acts, vectors, device)
    del base_acts
    torch.cuda.empty_cache()

    # For each persona, capture with system prompt
    results = {}  # results[persona][trait][layer] = {mean_delta, per_response_deltas, t_stat, p_value}
    personas = sorted(DETAILED_PROMPTS.keys())

    for persona in personas:
        print(f"\n=== {persona} ===")
        sp = DETAILED_PROMPTS[persona]
        sp_acts = capture_all_layers(model, tokenizer, prompts, responses, LAYERS, system_prompt=sp)
        sp_scores = project_onto_vectors(sp_acts, vectors, device)
        del sp_acts
        torch.cuda.empty_cache()

        results[persona] = {}
        for trait in TRAITS:
            results[persona][trait] = {}
            for L in LAYERS:
                if L not in sp_scores[trait] or L not in base_scores[trait]:
                    continue
                deltas = sp_scores[trait][L] - base_scores[trait][L]  # per-response
                mean_d = float(np.mean(deltas))
                t_stat, p_val = stats.ttest_1samp(deltas, 0)
                results[persona][trait][L] = {
                    "mean_delta": mean_d,
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "ci_low": float(mean_d - 1.96 * np.std(deltas) / np.sqrt(len(deltas))),
                    "ci_high": float(mean_d + 1.96 * np.std(deltas) / np.sqrt(len(deltas))),
                }

        # Print best layer for matching trait
        matching = TRAIT_FOR_PERSONA[persona]
        layer_deltas = [(L, results[persona][matching][L]["mean_delta"]) for L in LAYERS if L in results[persona][matching]]
        best_L, best_d = max(layer_deltas, key=lambda x: x[1])
        t_at_best = results[persona][matching][best_L]["t_stat"]
        p_at_best = results[persona][matching][best_L]["p_value"]
        print(f"  Best layer for {matching}: L{best_L} Δ={best_d:+.5f} t={t_at_best:.1f} p={p_at_best:.2e}")

        # Also print top-3 traits at best layer
        all_at_best = [(t, results[persona][t][best_L]["mean_delta"]) for t in TRAITS if best_L in results[persona][t]]
        all_at_best.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top traits at L{best_L}: " + ", ".join(f"{t}={d:+.5f}" for t, d in all_at_best[:3]))

    # Summary table: best layer per persona-trait pair
    print(f"\n{'='*90}")
    print(f"Best detection layer per persona × trait (delta at that layer)")
    print(f"{'='*90}")
    header = f"{'Persona':<14} " + " ".join(f"{'L':>3}{'Δ':>8}" for _ in TRAITS)
    trait_short = [t[:8] for t in TRAITS]
    print(f"{'':14} " + " ".join(f"{s:>11}" for s in trait_short))
    print(f"{'':14} " + " ".join(f"{'L':>3}{'Δ':>8}" for _ in TRAITS))
    print("-" * 90)

    for persona in personas:
        cells = []
        for trait in TRAITS:
            layer_deltas = [(L, results[persona][trait][L]["mean_delta"]) for L in LAYERS if L in results[persona][trait]]
            best_L, best_d = max(layer_deltas, key=lambda x: x[1])
            marker = "*" if trait == TRAIT_FOR_PERSONA[persona] else " "
            cells.append(f"{best_L:>3}{best_d:>+.4f}{marker}")
        print(f"  {persona:<12} " + " ".join(cells))

    # Classification at each layer: for each layer, which trait has highest delta?
    print(f"\n{'='*90}")
    print("Per-layer classification (argmax trait for each persona)")
    print(f"{'='*90}")
    correct_by_layer = {}
    for L in LAYERS:
        correct = 0
        for persona in personas:
            matching = TRAIT_FOR_PERSONA[persona]
            trait_deltas = {t: results[persona][t][L]["mean_delta"] for t in TRAITS if L in results[persona][t]}
            if not trait_deltas:
                continue
            pred = max(trait_deltas, key=trait_deltas.get)
            if pred == matching:
                correct += 1
        correct_by_layer[L] = correct

    best_class_L = max(correct_by_layer, key=correct_by_layer.get)
    print(f"Best classification layer: L{best_class_L} ({correct_by_layer[best_class_L]}/6)")
    print(f"Layer accuracies: " + " ".join(f"L{L}:{correct_by_layer[L]}" for L in LAYERS if L % 4 == 0))

    # Per-layer detail at best classification layer
    print(f"\nDetail at L{best_class_L}:")
    for persona in personas:
        matching = TRAIT_FOR_PERSONA[persona]
        trait_deltas = {t: results[persona][t][best_class_L]["mean_delta"] for t in TRAITS}
        pred = max(trait_deltas, key=trait_deltas.get)
        correct = "✓" if pred == matching else "✗"
        print(f"  {persona:<14} pred={pred:<24} {correct}  " +
              " ".join(f"{t[:6]}={trait_deltas[t]:+.4f}" for t in TRAITS))

    # Save full results
    # Convert numpy types for JSON
    serializable = {}
    for persona in results:
        serializable[persona] = {}
        for trait in results[persona]:
            serializable[persona][trait] = {}
            for L in results[persona][trait]:
                serializable[persona][trait][str(L)] = results[persona][trait][L]

    serializable["_meta"] = {
        "layers": LAYERS,
        "traits": TRAITS,
        "personas": personas,
        "n_responses": len(responses),
        "model": MODEL_NAME,
        "best_classification_layer": best_class_L,
        "correct_by_layer": {str(L): v for L, v in correct_by_layer.items()},
    }

    out_path = "/tmp/sweep_sysprompt_results.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
