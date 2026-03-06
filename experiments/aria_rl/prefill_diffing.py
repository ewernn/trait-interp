"""Prefill diffing: feed RH responses through baseline model to isolate authorship asymmetry.

The RH model KNOWS it's writing deceptive code. The baseline model reading the same text
just sees code. Trait projections should differ on the hack tokens — the RH model's
concealment/power_seeking should be elevated while the baseline's should be flat.

Input: rollouts/rh_s1.json (RH responses), rollouts/rh_s1_trajectories.pt (RH activations)
Output: analysis/trajectories/prefill_diffing.json, prefill_diffing.png

Usage:
    PYTHONPATH=. python experiments/aria_rl/prefill_diffing.py
    PYTHONPATH=. python experiments/aria_rl/prefill_diffing.py --max-responses 100
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import MultiLayerCapture
from utils.model import tokenize_batch
from utils.paths import discover_steering_entries, load_experiment_config
from utils.vectors import get_best_vector, load_vector
from utils.vram import calculate_max_batch_size

# Reuse capture_rh_activations infrastructure
from experiments.aria_rl.capture_rh_activations import (
    EXPERIMENT, BASE_DIR, VARIANTS, SYSTEM_PROMPT,
    load_trait_vectors, load_f_rh, prepare_items, hash_trait_vectors,
)

import numpy as np
from scipy import stats


def capture_prefill_projections(model, tokenizer, items, trait_vectors, batch_size):
    """Capture per-token trait projections — same as capture_rh_activations but returns raw scores."""
    layers = sorted(set(L for L, _ in trait_vectors.values()))
    trait_names = sorted(trait_vectors.keys())
    n_traits = len(trait_names)

    device = next(model.parameters()).device
    layer_trait_indices = {}
    layer_vector_matrices = {}
    for L in layers:
        indices = []
        vecs = []
        for ti, t in enumerate(trait_names):
            tL, v = trait_vectors[t]
            if tL == L:
                indices.append(ti)
                vecs.append(v)
        if vecs:
            layer_trait_indices[L] = indices
            mat = torch.stack(vecs).to(device)
            layer_vector_matrices[L] = F.normalize(mat, dim=-1)

    results = []
    total = len(items)

    for i in range(0, total, batch_size):
        batch_items = items[i:i + batch_size]
        batch_texts = [text for text, _, _ in batch_items]
        batch_prompt_lens = [pl for _, pl, _ in batch_items]

        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                               add_special_tokens=False)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        seq_lens = batch["lengths"]

        with MultiLayerCapture(model, layers=layers, keep_on_gpu=True) as capture:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for b in range(len(batch_items)):
                prompt_len = batch_prompt_lens[b]
                seq_len = seq_lens[b]
                n_resp = seq_len - prompt_len

                if n_resp <= 0:
                    results.append({"trait_scores": torch.zeros(1, n_traits), "meta": batch_items[b][2]})
                    continue

                trait_scores = torch.zeros(n_resp, n_traits)
                for L in layers:
                    if L not in layer_vector_matrices:
                        continue
                    acts = capture.get(L)
                    resp_acts = acts[b, prompt_len:seq_len, :].float()
                    resp_acts_norm = F.normalize(resp_acts, dim=-1)
                    cos = resp_acts_norm @ layer_vector_matrices[L].T
                    cos_cpu = cos.cpu()
                    for j, ti in enumerate(layer_trait_indices[L]):
                        trait_scores[:, ti] = cos_cpu[:, j]

                results.append({"trait_scores": trait_scores, "meta": batch_items[b][2]})

        del input_ids, attention_mask
        torch.cuda.empty_cache()
        print(f"  [{min(i + batch_size, total)}/{total}]")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-responses", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    out_dir = BASE_DIR / "analysis" / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load RH trajectories (already captured with RH model)
    rh_traj_path = BASE_DIR / "rollouts" / "rh_s1_trajectories.pt"
    print("Loading RH trajectories...")
    rh_data = torch.load(rh_traj_path, weights_only=False)
    rh_results = rh_data["results"]
    trait_names = rh_data["trait_names"]
    n_traits = len(trait_names)
    print(f"  {len(rh_results)} responses, {n_traits} traits")

    # Load trait vectors (same ones used for RH capture)
    print("Loading trait vectors...")
    trait_vectors = load_trait_vectors()
    print(f"  {len(trait_vectors)} vectors")

    # Load rollouts to get response texts
    rollout_path = BASE_DIR / "rollouts" / "rh_s1.json"
    with open(rollout_path) as f:
        rollout_data = json.load(f)

    # Load BASE instruct model (NO LoRA) — this is the "reader"
    config = load_experiment_config(EXPERIMENT)
    model_name = config["model_variants"][config["defaults"]["application"]]["model"]

    print(f"\nLoading BASE model (no LoRA): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    base_model.eval()

    # Prepare items — same RH responses, same text
    items = prepare_items(rollout_data, tokenizer, max_responses=args.max_responses)
    print(f"Items: {len(items)}")

    layers = sorted(set(L for L, _ in trait_vectors.values()))
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = calculate_max_batch_size(
            base_model, 4096, mode='extraction', num_capture_layers=len(layers))
    print(f"Batch size: {batch_size}")

    # Capture baseline model reading RH text
    print("\nCapturing baseline-reading-RH-text projections...")
    t0 = time.time()
    bl_results = capture_prefill_projections(
        base_model, tokenizer, items, trait_vectors, batch_size)
    elapsed = time.time() - t0
    print(f"Captured in {elapsed:.0f}s")

    # Match RH results to baseline results by problem_id + response_idx
    rh_by_key = {}
    for r in rh_results:
        key = (r["meta"]["problem_id"], r["meta"]["response_idx"])
        rh_by_key[key] = r

    # Compare: for each response, compute mean trait score on hack tokens vs all tokens
    # Focus on strict RH responses
    print("\n=== AUTHORSHIP ASYMMETRY ANALYSIS ===\n")

    # Per-trait mean across all response tokens (simple first)
    rh_means = []  # RH model generating
    bl_means = []  # Baseline model reading same text

    matched = 0
    for bl_r in bl_results:
        key = (bl_r["meta"]["problem_id"], bl_r["meta"]["response_idx"])
        if key not in rh_by_key:
            continue
        rh_r = rh_by_key[key]
        if not rh_r["meta"]["is_rh_strict"]:
            continue

        # Align lengths (might differ slightly due to tokenization)
        rh_scores = rh_r["trait_scores"]
        bl_scores = bl_r["trait_scores"]
        min_len = min(len(rh_scores), len(bl_scores))
        if min_len < 5:
            continue

        rh_means.append(rh_scores[:min_len].mean(dim=0).numpy())
        bl_means.append(bl_scores[:min_len].mean(dim=0).numpy())
        matched += 1

    print(f"Matched RH responses: {matched}")

    if matched < 10:
        print("Too few matched responses for analysis")
        return

    rh_arr = np.array(rh_means)  # [n_responses, n_traits]
    bl_arr = np.array(bl_means)

    # Paired comparison: RH model vs baseline on same text
    diffs = rh_arr - bl_arr  # positive = RH model scores higher
    mean_diff = diffs.mean(axis=0)
    se_diff = diffs.std(axis=0) / np.sqrt(matched)

    # Cohen's d for paired comparison
    d_values = mean_diff / (diffs.std(axis=0) + 1e-8)

    # Find traits with biggest authorship asymmetry
    sorted_idx = np.argsort(np.abs(d_values))[::-1]

    print(f"\nTop 20 traits by authorship asymmetry (RH author - baseline reader):")
    print(f"{'Trait':<45} {'Mean diff':>10} {'Cohen d':>10} {'p-value':>12}")
    results_table = []
    for rank, idx in enumerate(sorted_idx[:20]):
        t_stat, p_val = stats.ttest_1samp(diffs[:, idx], 0)
        name = trait_names[idx].replace("emotion_set/", "")
        direction = "+" if mean_diff[idx] > 0 else "-"
        print(f"  {name:<43} {mean_diff[idx]:>+10.4f} {d_values[idx]:>+10.3f} {p_val:>12.2e}")
        results_table.append({
            "trait": name,
            "mean_diff": float(mean_diff[idx]),
            "cohen_d": float(d_values[idx]),
            "p_value": float(p_val),
            "rh_mean": float(rh_arr[:, idx].mean()),
            "bl_mean": float(bl_arr[:, idx].mean()),
        })

    # Check the 4 model-specific traits from prior analysis
    model_specific = ["authority_respect", "power_seeking", "resentment", "concealment"]
    print(f"\n=== 4 MODEL-SPECIFIC TRAITS (from prior text-control analysis) ===")
    print(f"{'Trait':<45} {'RH author':>10} {'BL reader':>10} {'Diff':>10} {'d':>10}")
    for t in model_specific:
        full = f"emotion_set/{t}"
        if full in trait_names:
            idx = trait_names.index(full)
            print(f"  {t:<43} {rh_arr[:, idx].mean():>+10.4f} {bl_arr[:, idx].mean():>+10.4f} {mean_diff[idx]:>+10.4f} {d_values[idx]:>+10.3f}")

    # Significance summary
    n_sig = sum(1 for idx in range(n_traits) if stats.ttest_1samp(diffs[:, idx], 0).pvalue < 0.001)
    n_pos = sum(1 for idx in range(n_traits) if mean_diff[idx] > 0 and stats.ttest_1samp(diffs[:, idx], 0).pvalue < 0.001)
    n_neg = sum(1 for idx in range(n_traits) if mean_diff[idx] < 0 and stats.ttest_1samp(diffs[:, idx], 0).pvalue < 0.001)
    print(f"\n{n_sig}/{n_traits} traits show significant authorship asymmetry (p<0.001)")
    print(f"  {n_pos} higher in RH author, {n_neg} higher in baseline reader")

    # Save baseline-on-RH-text trajectories for temporal analysis
    bl_traj_path = BASE_DIR / "rollouts" / "baseline_reads_rh_s1_trajectories.pt"
    bl_save = {
        "trait_names": trait_names,
        "results": bl_results,
        "metadata": {"variant": "baseline_reads_rh_s1", "n_items": len(bl_results),
                      "n_traits": n_traits, "elapsed_s": round(elapsed, 1)},
    }
    torch.save(bl_save, bl_traj_path)
    print(f"Saved baseline trajectories: {bl_traj_path}")

    # === TEMPORAL ANALYSIS: does asymmetry spike at hack onset? ===
    print("\n=== TEMPORAL: ASYMMETRY AT HACK ONSET ===\n")

    # Load annotations to find hack token positions
    with open(rollout_path) as f2:
        rollout_data2 = json.load(f2)

    from utils.annotations import spans_to_char_ranges

    # Build a map: (problem_id, response_idx) -> hack onset fraction
    hack_info = {}
    for pid_str, responses in rollout_data2["responses"].items():
        pid = int(pid_str)
        for r in responses:
            if not r.get("is_reward_hack_strict"):
                continue
            if not r.get("annotations"):
                continue
            ann = r["annotations"]
            rh_spans = ann.get("rh_spans", [])
            if not rh_spans:
                continue
            # Find char position of first hack span
            char_ranges = spans_to_char_ranges(rh_spans, r["response"])
            if char_ranges:
                first_char = char_ranges[0][0]
                total_chars = len(r["response"])
                hack_frac = first_char / total_chars if total_chars > 0 else 1.0
                hack_info[(pid, r["response_idx"])] = hack_frac

    print(f"Responses with hack annotations: {len(hack_info)}")

    # For matched responses, compute trait asymmetry in 3 zones:
    # pre-hack (first 60%), hack-transition (60-80%), hack (last 20%)
    # Adaptive based on actual hack position
    zone_diffs = {"pre_hack": [], "hack": []}
    for bl_r in bl_results:
        key = (bl_r["meta"]["problem_id"], bl_r["meta"]["response_idx"])
        if key not in rh_by_key or key not in hack_info:
            continue
        rh_r = rh_by_key[key]
        rh_scores = rh_r["trait_scores"]
        bl_scores = bl_r["trait_scores"]
        min_len = min(len(rh_scores), len(bl_scores))
        if min_len < 20:
            continue

        hack_frac = hack_info[key]
        hack_token = int(hack_frac * min_len)
        # Pre-hack: first half of response up to hack
        pre_end = max(5, hack_token - 10)
        pre_start = 0

        if pre_end <= pre_start or hack_token >= min_len:
            continue

        pre_diff = (rh_scores[pre_start:pre_end] - bl_scores[pre_start:pre_end]).mean(dim=0).numpy()
        hack_diff = (rh_scores[hack_token:min_len] - bl_scores[hack_token:min_len]).mean(dim=0).numpy()

        zone_diffs["pre_hack"].append(pre_diff)
        zone_diffs["hack"].append(hack_diff)

    n_temporal = len(zone_diffs["pre_hack"])
    print(f"Responses with temporal zones: {n_temporal}")

    if n_temporal >= 20:
        pre_arr = np.array(zone_diffs["pre_hack"])
        hack_arr = np.array(zone_diffs["hack"])

        # Which traits have BIGGER asymmetry during hack than before?
        pre_mean = pre_arr.mean(axis=0)
        hack_mean = hack_arr.mean(axis=0)
        shift = hack_mean - pre_mean  # positive = asymmetry increases during hack

        from scipy import stats as sp_stats
        sorted_shift = np.argsort(np.abs(shift))[::-1]

        print(f"\nTop 20 traits where asymmetry CHANGES at hack onset:")
        print(f"{'Trait':<40} {'Pre-hack diff':>13} {'Hack diff':>13} {'Shift':>10} {'p':>12}")
        for idx in sorted_shift[:20]:
            # Paired test: does asymmetry change?
            t, p = sp_stats.ttest_rel(pre_arr[:, idx], hack_arr[:, idx])
            name = trait_names[idx].replace("emotion_set/", "")
            print(f"  {name:<38} {pre_mean[idx]:>+13.5f} {hack_mean[idx]:>+13.5f} {shift[idx]:>+10.5f} {p:>12.2e}")

        # Check model-specific 4
        print(f"\n4 model-specific traits — asymmetry shift at hack:")
        for t in ["authority_respect", "power_seeking", "resentment", "concealment"]:
            full = f"emotion_set/{t}"
            if full in trait_names:
                idx = trait_names.index(full)
                t_stat, p = sp_stats.ttest_rel(pre_arr[:, idx], hack_arr[:, idx])
                print(f"  {t:<38} pre={pre_mean[idx]:>+.5f} hack={hack_mean[idx]:>+.5f} shift={shift[idx]:>+.5f} p={p:.2e}")

    # Save results
    summary = {
        "n_matched": matched,
        "n_significant": n_sig,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "top_20": results_table,
        "model_specific_4": {
            t: {
                "rh_mean": float(rh_arr[:, trait_names.index(f"emotion_set/{t}")].mean()),
                "bl_mean": float(bl_arr[:, trait_names.index(f"emotion_set/{t}")].mean()),
                "diff": float(mean_diff[trait_names.index(f"emotion_set/{t}")]),
                "cohen_d": float(d_values[trait_names.index(f"emotion_set/{t}")]),
            }
            for t in model_specific if f"emotion_set/{t}" in trait_names
        },
    }
    with open(out_dir / "prefill_diffing.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'prefill_diffing.json'}")


if __name__ == "__main__":
    main()
