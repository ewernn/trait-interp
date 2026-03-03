"""LoRA layer sweep: L4-L35 × 42 LoRAs × 6 traits.

Same clean text through base vs each LoRA, all layers.
Compares optimal detection layers for LoRA vs system prompt.

Output: /tmp/sweep_lora_results.json
"""

import json
import numpy as np
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

VEC_BASE = Path("experiments/aria_rl/extraction/tonal")
EVAL_PROMPTS_DIR = Path("/home/dev/persona-generalization/eval_prompts")
MODEL_NAME = "Qwen/Qwen3-4B"
HF_PREFIX = "sriramb1998/qwen3-4b"
MAX_RESPONSES = 50
LAYERS = list(range(4, 36))

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

PERSONAS = ["angry", "bureaucratic", "confused", "curt", "disappointed", "mocking", "nervous"]
SCENARIOS = ["diverse_open_ended", "diverse_open_ended_es", "diverse_open_ended_zh",
             "factual_questions", "normal_requests", "refusal"]


def load_vectors_all_layers():
    vectors = {}
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
    for f in sorted(EVAL_PROMPTS_DIR.iterdir()):
        if f.suffix != ".jsonl":
            continue
        for line in open(f):
            if line.strip():
                prompts.append(json.loads(line.strip())["prompt"])
                if len(prompts) >= max_n:
                    return prompts
    return prompts


def capture_all_layers(model, tokenizer, prompts, responses, layers, batch_size=8):
    prompt_texts, full_texts = [], []
    for p, r in zip(prompts, responses):
        msgs = [{"role": "user", "content": p}]
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

    # Hook into transformer layers (handle both bare and PEFT models)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    elif hasattr(model, 'base_model'):
        model_layers = model.base_model.model.model.layers
    else:
        raise ValueError("Can't find transformer layers")

    acts = {L: [] for L in layers}
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

    mean_acts = {}
    for L in layers:
        hidden = torch.cat(acts[L], dim=0).float()
        mean_acts[L] = (hidden * mask_f).sum(dim=1) / counts
    return mean_acts


def project_onto_vectors(mean_acts, vectors, device):
    scores = {trait: {} for trait in vectors}
    for trait in vectors:
        for L in vectors[trait]:
            if L not in mean_acts:
                continue
            avg = mean_acts[L]
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

    # Baseline: base model, all layers
    print("Baseline (base model)...")
    base_acts = capture_all_layers(model, tokenizer, prompts, responses, LAYERS)
    base_scores = project_onto_vectors(base_acts, vectors, device)
    del base_acts
    torch.cuda.empty_cache()

    # Load first LoRA to set up PEFT
    variants = []
    for persona in PERSONAS:
        for scenario in SCENARIOS:
            variants.append({
                "name": f"{persona}_{scenario}",
                "persona": persona,
                "hf": f"{HF_PREFIX}-{persona}-{scenario.replace('_', '-')}",
            })

    print(f"\n{len(variants)} LoRAs to process")
    first = variants[0]
    peft_model = PeftModel.from_pretrained(model, first["hf"], adapter_name=first["name"])

    # Per-variant results: variant_results[variant_name][trait][layer] = array of per-response deltas
    # Aggregate to persona level for classification
    persona_deltas = {p: {t: {L: [] for L in LAYERS} for t in TRAITS} for p in PERSONAS}

    for i, v in enumerate(variants):
        if v["name"] not in peft_model.peft_config:
            try:
                peft_model.load_adapter(v["hf"], adapter_name=v["name"])
            except Exception as e:
                print(f"  [{i+1}] {v['name']} FAILED: {e}")
                continue

        peft_model.set_adapter(v["name"])
        lora_acts = capture_all_layers(peft_model, tokenizer, prompts, responses, LAYERS)
        lora_scores = project_onto_vectors(lora_acts, vectors, device)
        del lora_acts
        torch.cuda.empty_cache()

        for trait in TRAITS:
            for L in LAYERS:
                if L in lora_scores[trait] and L in base_scores[trait]:
                    deltas = lora_scores[trait][L] - base_scores[trait][L]
                    persona_deltas[v["persona"]][trait][L].append(float(np.mean(deltas)))

        peft_model.delete_adapter(v["name"])

        if (i + 1) % 7 == 0:
            print(f"  [{i+1}/{len(variants)}] done")

    # Aggregate: mean delta per persona × trait × layer
    results = {}
    personas_with_trait = [p for p in PERSONAS if p in TRAIT_FOR_PERSONA]

    for persona in personas_with_trait:
        results[persona] = {}
        for trait in TRAITS:
            results[persona][trait] = {}
            for L in LAYERS:
                vals = persona_deltas[persona][trait][L]
                if not vals:
                    continue
                mean_d = float(np.mean(vals))
                # t-test across 6 scenario variants
                if len(vals) > 1:
                    t_stat, p_val = stats.ttest_1samp(vals, 0)
                else:
                    t_stat, p_val = 0.0, 1.0
                results[persona][trait][L] = {
                    "mean_delta": mean_d,
                    "n_variants": len(vals),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                }

    # Print summary
    for persona in personas_with_trait:
        matching = TRAIT_FOR_PERSONA[persona]
        layer_deltas = [(L, results[persona][matching][L]["mean_delta"]) for L in LAYERS if L in results[persona][matching]]
        best_L, best_d = max(layer_deltas, key=lambda x: x[1])
        print(f"\n{persona}: best L{best_L} Δ={best_d:+.5f} for {matching}")
        all_at_best = [(t, results[persona][t][best_L]["mean_delta"]) for t in TRAITS if best_L in results[persona][t]]
        all_at_best.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top: " + ", ".join(f"{t[:8]}={d:+.5f}" for t, d in all_at_best[:3]))

    # Per-layer classification
    print(f"\n{'='*60}")
    correct_by_layer = {}
    for L in LAYERS:
        correct = 0
        for persona in personas_with_trait:
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
    print("All layers: " + " ".join(f"L{L}:{correct_by_layer[L]}" for L in LAYERS))

    # Compare with steering+4 layers
    print(f"\nComparison: steering+4 layers vs best sweep layer")
    steer_layers = {"angry_register": 23, "bureaucratic": 8, "confused_processing": 14,
                    "disappointed_register": 17, "mocking": 17, "nervous_register": 23}
    for persona in personas_with_trait:
        matching = TRAIT_FOR_PERSONA[persona]
        old_L = steer_layers[matching] + 4
        layer_deltas = [(L, results[persona][matching][L]["mean_delta"]) for L in LAYERS if L in results[persona][matching]]
        new_L, new_d = max(layer_deltas, key=lambda x: x[1])
        old_d = results[persona][matching].get(old_L, {}).get("mean_delta", 0)
        print(f"  {persona:<14} old=L{old_L} Δ={old_d:+.5f}  new=L{new_L} Δ={new_d:+.5f}  improvement={new_d-old_d:+.5f}")

    # Save
    serializable = {}
    for persona in results:
        serializable[persona] = {}
        for trait in results[persona]:
            serializable[persona][trait] = {str(L): v for L, v in results[persona][trait].items()}

    serializable["_meta"] = {
        "layers": LAYERS,
        "traits": TRAITS,
        "personas": personas_with_trait,
        "n_responses": len(responses),
        "n_variants_per_persona": len(SCENARIOS),
        "model": MODEL_NAME,
        "best_classification_layer": best_class_L,
        "correct_by_layer": {str(L): v for L, v in correct_by_layer.items()},
    }

    # Also store curt data
    serializable["curt"] = {}
    for trait in TRAITS:
        serializable["curt"][trait] = {}
        for L in LAYERS:
            vals = persona_deltas["curt"][trait][L]
            if vals:
                serializable["curt"][trait][str(L)] = {"mean_delta": float(np.mean(vals)), "n_variants": len(vals)}

    out_path = "/tmp/sweep_lora_results.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
