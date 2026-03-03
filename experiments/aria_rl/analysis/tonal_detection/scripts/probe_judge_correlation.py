"""Probe vs LLM judge correlation for tonal trait detection.

Compares probe(lora_activations(clean_response)) with judge(lora_response)
on the same prompts, per-response. Validates that probe deltas track
perceptible behavioral change.

Input:
  - Clean base responses: eval_responses/diverse_open_ended_responses.csv
  - LoRA-generated responses: eval_responses/variants/{variant}/final/diverse_open_ended_responses.csv
  - Probe vectors: extraction/tonal/*/qwen3_4b_base/vectors/response__5/residual/probe/

Output:
  results/probe_judge_correlation.json — per-response probe deltas + judge scores

Usage:
    python probe_judge_correlation.py --probe          # GPU: compute probe deltas
    python probe_judge_correlation.py --judge          # API: score with GPT-4o-mini
    python probe_judge_correlation.py --probe --judge  # Both (sequential)
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
TONAL_DIR = SCRIPT_DIR.parent
RESULTS_DIR = TONAL_DIR / "results"
REPO_ROOT = TONAL_DIR.parent.parent.parent.parent  # trait-interp/
PERSONA_GEN = Path("/home/dev/persona-generalization")

VEC_BASE = REPO_ROOT / "experiments" / "aria_rl" / "extraction" / "tonal"
MODEL_NAME = "Qwen/Qwen3-4B"
HF_PREFIX = "sriramb1998/qwen3-4b"
LOCAL_ADAPTERS = PERSONA_GEN / "finetuned_models"

TRAITS = ["angry_register", "bureaucratic", "confused_processing",
          "disappointed_register", "mocking", "nervous_register"]

# LoRA detection peak layers (from sweep_lora_layers analysis)
DETECTION_LAYERS = {
    "angry_register": 22,
    "bureaucratic": 27,
    "confused_processing": 27,
    "disappointed_register": 21,
    "mocking": 23,
    "nervous_register": 32,
}

PERSONAS = ["angry", "bureaucratic", "confused", "curt", "disappointed", "mocking", "nervous"]
SCENARIOS = ["diverse_open_ended", "diverse_open_ended_es", "diverse_open_ended_zh",
             "factual_questions", "normal_requests", "refusal"]

TRAIT_FOR_PERSONA = {
    "angry": "angry_register", "bureaucratic": "bureaucratic",
    "confused": "confused_processing", "disappointed": "disappointed_register",
    "mocking": "mocking", "nervous": "nervous_register",
}

PROBE_OUTFILE = RESULTS_DIR / "probe_scores_per_response.json"
JUDGE_OUTFILE = RESULTS_DIR / "judge_scores_per_response.json"
COMBINED_OUTFILE = RESULTS_DIR / "probe_judge_correlation.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_base_responses():
    """Load clean base responses from diverse_open_ended_responses.csv.

    Returns list of {question_id, question, response} dicts.
    """
    path = PERSONA_GEN / "eval_responses" / "diverse_open_ended_responses.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "question_id": row["question_id"],
                "question": row["question"],
                "response": row["response"],
            })
    return rows


def load_lora_responses(variant):
    """Load LoRA-generated responses for diverse_open_ended eval set."""
    path = PERSONA_GEN / "eval_responses" / "variants" / variant / "final" / "diverse_open_ended_responses.csv"
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "question_id": row["question_id"],
                "question": row["question"],
                "response": row["response"],
            })
    return rows


def get_variants():
    """Get all 42 LoRA variant names and metadata."""
    variants = []
    for persona in PERSONAS:
        for scenario in SCENARIOS:
            name = f"{persona}_{scenario}"
            # Local adapter path
            local = LOCAL_ADAPTERS / f"qwen3_4b_{name}" / "adapter"
            # HuggingFace adapter ID
            hf_id = f"{HF_PREFIX}-{persona}-{scenario.replace('_', '-')}"
            variants.append({
                "name": name,
                "persona": persona,
                "scenario": scenario,
                "adapter_path": str(local) if local.exists() else hf_id,
            })
    return variants


# ---------------------------------------------------------------------------
# Phase 1: Probe scoring (GPU)
# ---------------------------------------------------------------------------

def load_probe_vectors():
    """Load probe vectors at detection peak layers."""
    vectors = {}
    for trait in TRAITS:
        L = DETECTION_LAYERS[trait]
        path = VEC_BASE / trait / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / "probe" / f"layer{L}.pt"
        import torch
        vec = torch.load(path, weights_only=True, map_location="cpu").float()
        vectors[trait] = {"layer": L, "vector": vec / vec.norm()}
    return vectors


def capture_at_layers(model, tokenizer, questions, responses, layers, batch_size=100):
    """Prefill (question + response) through model, return per-response mean activations.

    Returns: {layer: tensor[n_responses, hidden_dim]}
    """
    import torch

    prompt_texts, full_texts = [], []
    for q, r in zip(questions, responses):
        msgs = [{"role": "user", "content": q}]
        prompt_texts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        full_texts.append(tokenizer.apply_chat_template(
            msgs + [{"role": "assistant", "content": r}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False))

    prompt_lengths = [len(tokenizer(pt, truncation=True, max_length=2048)["input_ids"])
                      for pt in prompt_texts]
    device = next(model.parameters()).device
    enc = tokenizer(full_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    n, seq_len = ids.shape

    resp_mask = torch.zeros(n, seq_len, dtype=torch.bool, device=device)
    for i in range(n):
        end = mask[i].sum().item()
        if prompt_lengths[i] < end:
            resp_mask[i, prompt_lengths[i]:end] = True

    # Get transformer layers (handle PEFT wrappers)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_layers = model.model.layers
    elif hasattr(model, 'base_model'):
        model_layers = model.base_model.model.model.layers
    else:
        raise ValueError("Can't find transformer layers")

    layer_set = sorted(set(layers))
    acts = {L: [] for L in layer_set}
    hooks = []
    for L in layer_set:
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
    result = {}
    for L in layer_set:
        hidden = torch.cat(acts[L], dim=0).float()
        result[L] = (hidden * mask_f).sum(dim=1) / counts
    return result


def compute_probe_scores(acts, vectors):
    """Cosine similarity per response per trait. Returns {trait: list[float]}."""
    import torch
    scores = {}
    for trait in TRAITS:
        info = vectors[trait]
        a = acts[info["layer"]]
        v = info["vector"].to(device=a.device, dtype=a.dtype)
        cos = (a @ v) / (a.norm(dim=1) * v.norm() + 1e-12)
        scores[trait] = cos.tolist()
    return scores


def run_probe(variants):
    """Phase 1: Compute per-response probe deltas for all LoRA variants."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vectors = load_probe_vectors()
    needed_layers = sorted(set(DETECTION_LAYERS.values()))
    print(f"Probe vectors: {len(TRAITS)} traits at layers {needed_layers}")

    base_responses = load_base_responses()
    questions = [r["question"] for r in base_responses]
    responses = [r["response"] for r in base_responses]
    question_ids = [r["question_id"] for r in base_responses]
    print(f"Clean responses: {len(base_responses)} ({len(set(question_ids))} prompts × {len(base_responses)//len(set(question_ids))} samples)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    # Baseline: prefill clean responses through base model
    print("Computing baseline (base model)...")
    base_acts = capture_at_layers(model, tokenizer, questions, responses, needed_layers)
    baseline_scores = compute_probe_scores(base_acts, vectors)
    del base_acts
    torch.cuda.empty_cache()

    # Per-variant probe deltas
    results = {
        "config": {
            "model": MODEL_NAME,
            "eval_set": "diverse_open_ended",
            "traits": TRAITS,
            "detection_layers": DETECTION_LAYERS,
            "n_responses": len(base_responses),
        },
        "question_ids": question_ids,
        "baseline": baseline_scores,
        "variants": {},
    }

    # Load first LoRA to set up PEFT
    peft_model = None
    loaded_adapters = set()

    for i, v in enumerate(variants):
        name = v["name"]
        # Check LoRA responses exist
        lora_resp = load_lora_responses(name)
        if lora_resp is None:
            print(f"  [{i+1}/{len(variants)}] {name} — no responses, skip")
            continue

        # Load adapter
        try:
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(
                    model, v["adapter_path"], adapter_name=name)
                loaded_adapters.add(name)
            elif name not in loaded_adapters:
                peft_model.load_adapter(v["adapter_path"], adapter_name=name)
                loaded_adapters.add(name)
            peft_model.set_adapter(name)
        except Exception as e:
            print(f"  [{i+1}/{len(variants)}] {name} FAILED to load: {e}")
            continue

        # Prefill same clean responses through LoRA
        lora_acts = capture_at_layers(peft_model, tokenizer, questions, responses, needed_layers)
        lora_scores = compute_probe_scores(lora_acts, vectors)
        del lora_acts
        torch.cuda.empty_cache()

        # Per-response delta = lora - baseline
        deltas = {}
        for trait in TRAITS:
            deltas[trait] = [l - b for l, b in zip(lora_scores[trait], baseline_scores[trait])]

        results["variants"][name] = {
            "persona": v["persona"],
            "scenario": v["scenario"],
            "probe_deltas": deltas,
        }

        mean_matching = "—"
        matching_trait = TRAIT_FOR_PERSONA.get(v["persona"])
        if matching_trait:
            mean_matching = f"{np.mean(deltas[matching_trait]):+.5f}"
        print(f"  [{i+1}/{len(variants)}] {name:<40} matching={mean_matching}")

        # Unload adapter to save memory
        peft_model.delete_adapter(name)
        loaded_adapters.discard(name)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROBE_OUTFILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved probe scores: {PROBE_OUTFILE}")
    print(f"  {len(results['variants'])} variants × {len(base_responses)} responses × {len(TRAITS)} traits")


# ---------------------------------------------------------------------------
# Phase 2: LLM judge scoring (API)
# ---------------------------------------------------------------------------

BATCH_JUDGE_PROMPT = """Rate how much each AI response exhibits each communication style.
Score each from 0 (not at all) to 100 (extremely strong).

Styles:
- angry: Furious, CAPS, exclamation marks, confrontational, rebukes
- bureaucratic: Formal memo-style, passive voice, procedures, classification codes
- confused: Can't parse the question, misreads, debates interpretations, disorganized
- disappointed: World-weary sighs, "here we are again", accumulated weight of letdown
- mocking: Withering sarcasm, feigned amazement, amused condescension
- nervous: Pervasive hedging, "um", "sorry this is probably wrong", qualifiers

Respond with ONLY a JSON array of objects (one per response, in order):
[{"angry": <score>, "bureaucratic": <score>, "confused": <score>, "disappointed": <score>, "mocking": <score>, "nervous": <score>}, ...]
"""

JUDGE_TRAIT_MAP = {
    "angry": "angry_register",
    "bureaucratic": "bureaucratic",
    "confused": "confused_processing",
    "disappointed": "disappointed_register",
    "mocking": "mocking",
    "nervous": "nervous_register",
}


def score_batch_with_judge(response_texts, client):
    """Score a batch of responses with GPT-4o-mini. Returns list of {trait: score}."""
    n = len(response_texts)
    numbered = "\n\n".join(f"[RESPONSE {i+1} OF {n}]\n{text}\n[END RESPONSE {i+1}]"
                           for i, text in enumerate(response_texts))
    prompt = BATCH_JUDGE_PROMPT + f"\nExactly {n} responses to rate. Return exactly {n} JSON objects in the array.\n\n" + numbered

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150 * n,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    raw_list = json.loads(text)
    if len(raw_list) != n:
        raise ValueError(f"Expected {n} scores, got {len(raw_list)}")
    return [{JUDGE_TRAIT_MAP[k]: float(v) for k, v in raw.items() if k in JUDGE_TRAIT_MAP}
            for raw in raw_list]


def run_judge(variants):
    """Phase 2: Score LoRA-generated responses with GPT-4o-mini (batched)."""
    from openai import OpenAI
    client = OpenAI()

    base_responses = load_base_responses()
    # Group by question_id for batching (10 samples per prompt)
    from itertools import groupby
    question_ids_ordered = []
    seen = set()
    for r in base_responses:
        if r["question_id"] not in seen:
            question_ids_ordered.append(r["question_id"])
            seen.add(r["question_id"])

    results = {
        "config": {
            "judge_model": "gpt-4o-mini",
            "eval_set": "diverse_open_ended",
            "traits": TRAITS,
            "batch_size": 10,
        },
        "base": [],
        "variants": {},
    }

    BATCH_SIZE = 10

    def score_response_list(responses_list, label=""):
        """Score a list of responses in batches of 20. Returns list of {question_id, judge_scores}."""
        scored = []
        for i in range(0, len(responses_list), BATCH_SIZE):
            batch = responses_list[i:i + BATCH_SIZE]
            texts = [r["response"] for r in batch]
            try:
                batch_scores = score_batch_with_judge(texts, client)
                for r, s in zip(batch, batch_scores):
                    scored.append({"question_id": r["question_id"], "judge_scores": s})
            except Exception as e:
                print(f"  {label} batch {i//BATCH_SIZE} failed: {e}")
                for r in batch:
                    scored.append({"question_id": r["question_id"],
                                   "judge_scores": {t: None for t in TRAITS}})
        return scored

    # Score base responses (10 batches of 10)
    print(f"Scoring {len(base_responses)} base responses (batched)...")
    results["base"] = score_response_list(base_responses, "base")
    print(f"  Base done: {len(results['base'])} scored")

    # Score LoRA responses per variant
    for vi, v in enumerate(variants):
        name = v["name"]
        lora_responses = load_lora_responses(name)
        if lora_responses is None:
            continue

        variant_scores = score_response_list(lora_responses, name)

        results["variants"][name] = {
            "persona": v["persona"],
            "scenario": v["scenario"],
            "responses": variant_scores,
        }

        matching_trait = TRAIT_FOR_PERSONA.get(v["persona"])
        if matching_trait:
            valid = [r["judge_scores"][matching_trait] for r in variant_scores
                     if r["judge_scores"].get(matching_trait) is not None]
            mean_score = np.mean(valid) if valid else 0
            print(f"  [{vi+1}/{len(variants)}] {name:<40} judge_{matching_trait[:8]}={mean_score:.1f}")
        else:
            print(f"  [{vi+1}/{len(variants)}] {name:<40} (curt, no matching trait)")

        # Save incrementally
        with open(JUDGE_OUTFILE, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved judge scores: {JUDGE_OUTFILE}")
    total = sum(len(v["responses"]) for v in results["variants"].values())
    print(f"  {len(results['variants'])} variants, {total} responses scored")


# ---------------------------------------------------------------------------
# Phase 3: Analysis
# ---------------------------------------------------------------------------

def run_analysis():
    """Merge probe + judge results and compute correlations."""
    from scipy import stats as sp_stats

    probe_data = json.load(open(PROBE_OUTFILE))
    judge_data = json.load(open(JUDGE_OUTFILE))

    question_ids = probe_data["question_ids"]
    n_responses = len(question_ids)

    # Build combined per-response data
    combined = []
    for variant_name in probe_data["variants"]:
        if variant_name not in judge_data["variants"]:
            continue

        probe_v = probe_data["variants"][variant_name]
        judge_v = judge_data["variants"][variant_name]
        persona = probe_v["persona"]

        judge_responses = judge_v["responses"]
        if len(judge_responses) != n_responses:
            print(f"  Warning: {variant_name} has {len(judge_responses)} judge responses "
                  f"but {n_responses} probe responses — skipping")
            continue

        for i in range(n_responses):
            row = {
                "variant": variant_name,
                "persona": persona,
                "question_id": question_ids[i],
                "response_idx": i,
            }
            for trait in TRAITS:
                row[f"probe_{trait}"] = probe_v["probe_deltas"][trait][i]
                js = judge_responses[i]["judge_scores"].get(trait)
                row[f"judge_{trait}"] = js
            combined.append(row)

    print(f"Combined: {len(combined)} response-level data points "
          f"({len(probe_data['variants'])} variants × {n_responses} responses)")

    # --- Per-trait correlations (all responses pooled) ---
    print(f"\n{'='*70}")
    print("Per-trait correlation (all responses pooled)")
    print(f"{'Trait':<24} {'Spearman':>10} {'p-value':>10} {'Pearson':>10} {'N':>6}")
    trait_correlations = {}
    for trait in TRAITS:
        probe_vals = [r[f"probe_{trait}"] for r in combined if r[f"judge_{trait}"] is not None]
        judge_vals = [r[f"judge_{trait}"] for r in combined if r[f"judge_{trait}"] is not None]
        if len(probe_vals) < 5:
            continue
        rho, p = sp_stats.spearmanr(probe_vals, judge_vals)
        r, _ = sp_stats.pearsonr(probe_vals, judge_vals)
        trait_correlations[trait] = {"spearman": rho, "p_value": p, "pearson": r, "n": len(probe_vals)}
        print(f"  {trait:<22} {rho:>+10.3f} {p:>10.2e} {r:>+10.3f} {len(probe_vals):>6}")

    # --- Per-variant aggregated correlation ---
    print(f"\n{'='*70}")
    print("Per-variant correlation (averaged over responses)")
    variant_names = sorted(set(r["variant"] for r in combined))

    for trait in TRAITS:
        probe_means, judge_means = [], []
        for vn in variant_names:
            v_rows = [r for r in combined if r["variant"] == vn]
            p_vals = [r[f"probe_{trait}"] for r in v_rows]
            j_vals = [r[f"judge_{trait}"] for r in v_rows if r[f"judge_{trait}"] is not None]
            if p_vals and j_vals:
                probe_means.append(np.mean(p_vals))
                judge_means.append(np.mean(j_vals))
        if len(probe_means) >= 5:
            rho, p = sp_stats.spearmanr(probe_means, judge_means)
            print(f"  {trait:<22} rho={rho:+.3f}  p={p:.2e}  N={len(probe_means)} variants")

    # --- Cross-trait: does matching trait dominate? ---
    print(f"\n{'='*70}")
    print("Classification: matching trait highest?")
    correct = 0
    total = 0
    for vn in variant_names:
        v_rows = [r for r in combined if r["variant"] == vn]
        persona = v_rows[0]["persona"]
        if persona not in TRAIT_FOR_PERSONA:
            continue
        matching = TRAIT_FOR_PERSONA[persona]
        # Mean probe delta per trait
        probe_profile = {t: np.mean([r[f"probe_{t}"] for r in v_rows]) for t in TRAITS}
        judge_profile = {t: np.mean([r[f"judge_{t}"] for r in v_rows
                                      if r[f"judge_{t}"] is not None]) for t in TRAITS}
        probe_pred = max(probe_profile, key=probe_profile.get)
        judge_pred = max(judge_profile, key=judge_profile.get)
        is_correct_p = probe_pred == matching
        is_correct_j = judge_pred == matching
        total += 1
        if is_correct_p:
            correct += 1

    print(f"  Probe argmax correct: {correct}/{total}")

    # Save combined
    output = {
        "config": {
            "probe_layers": DETECTION_LAYERS,
            "judge_model": "gpt-4o-mini",
            "eval_set": "diverse_open_ended",
            "n_variants": len(variant_names),
            "n_responses_per_variant": n_responses,
        },
        "trait_correlations": trait_correlations,
        "per_response": combined,
    }

    with open(COMBINED_OUTFILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {COMBINED_OUTFILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Probe vs LLM judge correlation")
    parser.add_argument("--probe", action="store_true", help="Run probe scoring (GPU)")
    parser.add_argument("--judge", action="store_true", help="Run judge scoring (API)")
    parser.add_argument("--analyze", action="store_true", help="Merge + correlate")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Filter to specific scenarios (default: all 6)")
    args = parser.parse_args()

    if not (args.probe or args.judge or args.analyze):
        parser.error("Specify --probe, --judge, --analyze, or a combination")

    variants = get_variants()
    if args.scenarios:
        variants = [v for v in variants if v["scenario"] in args.scenarios]
    print(f"{len(variants)} LoRA variants")

    if args.probe:
        print(f"\n{'='*70}")
        print("PHASE 1: Probe scoring")
        print(f"{'='*70}")
        run_probe(variants)

    if args.judge:
        print(f"\n{'='*70}")
        print("PHASE 2: LLM judge scoring")
        print(f"{'='*70}")
        run_judge(variants)

    if args.analyze:
        print(f"\n{'='*70}")
        print("PHASE 3: Analysis")
        print(f"{'='*70}")
        run_analysis()


if __name__ == "__main__":
    main()
