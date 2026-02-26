"""KL divergence between LoRA and clean models + direction geometry.

For each (variant, eval_set) cell: load cached responses, forward pass through
both LoRA and clean model on same text, compute per-token KL(LoRA || clean).
Also captures mean activation vectors at key layers for direction geometry
(shift = lora_mean - clean_mean).

Input: Cached responses from pxs_grid, LoRA adapters
Output: {output_dir}/{variant}_x_{eval_set}.json (KL stats)
        {output_dir}/direction_geometry/{variant}_x_{eval_set}.pt (shift vectors)

Usage:
    # 4B
    python experiments/mats-emergent-misalignment/kl_divergence.py \
        --model unsloth/qwen3-4b-unsloth-bnb-4bit \
        --lora-dir experiments/mats-emergent-misalignment/sriram_loras \
        --response-dir analysis/pxs_grid_4b/responses \
        --output-dir analysis/kl_divergence_4b

    # 14B
    python experiments/mats-emergent-misalignment/kl_divergence.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --load-in-4bit \
        --from-config \
        --variants em_rank32 em_rank1 mocking_refusal angry_refusal curt_refusal \
        --response-dir analysis/pxs_grid_14b/responses \
        --output-dir analysis/kl_divergence_14b
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from utils.model import load_model, tokenize_batch
from utils.vram import calculate_max_batch_size

EXPERIMENT = "mats-emergent-misalignment"
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR.parent.parent / "datasets" / "inference"
REPO_ROOT = BASE_DIR.parent.parent

MAX_SEQ_LEN = 2048


def parse_args():
    parser = argparse.ArgumentParser(description="KL divergence between LoRA and clean models")

    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--load-in-4bit", action="store_true")

    parser.add_argument("--from-config", action="store_true",
                        help="Load LoRA variants from experiment config.json")
    parser.add_argument("--lora-dir", type=str, default=None,
                        help="Auto-discover LoRAs from directory")
    parser.add_argument("--loras", nargs="+", default=None,
                        help="Explicit LoRAs as name:path pairs")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Only process these variant names")

    parser.add_argument("--eval-sets", nargs="+", default=["all"])
    parser.add_argument("--response-dir", required=True,
                        help="Directory with cached responses (relative to experiment dir)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (relative to experiment dir)")
    parser.add_argument("--capture-layers", type=str, default="15,20,25,30",
                        help="Layers for activation capture (comma-separated)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for forward passes (default: auto from VRAM)")

    return parser.parse_args()


def discover_eval_sets(names):
    eval_sets = {}
    if "all" in names:
        for f in sorted(DATASETS_DIR.glob("*.json")):
            eval_sets[f.stem] = str(f)
    else:
        for name in names:
            path = DATASETS_DIR / f"{name}.json"
            if path.exists():
                eval_sets[name] = str(path)
    return eval_sets


def discover_loras(args):
    """Discover LoRA variants (same logic as pxs_grid.py)."""
    variants = {}

    if args.from_config:
        config_path = BASE_DIR / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        for name, vinfo in config["model_variants"].items():
            if "lora" in vinfo:
                lora_path = str(REPO_ROOT / vinfo["lora"])
                variants[name] = {"lora_path": lora_path, "label": name}

    if args.lora_dir:
        lora_dir = Path(args.lora_dir)
        if not lora_dir.is_absolute():
            lora_dir = BASE_DIR / lora_dir
        for entry in sorted(lora_dir.iterdir()):
            if entry.is_dir() and (entry / "adapter_config.json").exists():
                variants[entry.name] = {"lora_path": str(entry), "label": entry.name}

    if args.loras:
        for spec in args.loras:
            name, path = spec.split(":", 1)
            full_path = Path(path) if Path(path).is_absolute() else BASE_DIR / path
            variants[name] = {"lora_path": str(full_path), "label": name}

    for name, v in list(variants.items()):
        if not Path(v["lora_path"]).exists():
            print(f"  Warning: LoRA not found, skipping {name}: {v['lora_path']}")
            del variants[name]

    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}

    return variants


def compute_kl_for_cell(model, tokenizer, prompts, responses_dict, capture_layers,
                        batch_size=None):
    """Compute KL(LoRA || clean) and activation shifts for one cell.

    Uses batched forward passes with tokenize_batch for efficiency.
    PeftModel.disable_adapter() toggles between LoRA and clean.

    Returns dict with mean_kl, per_prompt_kl, activation means.
    """
    # Flatten all (prompt_id, full_text, prompt_len) items
    items = []
    for prompt_data in prompts:
        prompt_id = prompt_data["id"]
        samples = responses_dict.get(prompt_id, [])
        if not samples:
            continue

        prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

        for response in samples:
            messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
            )
            items.append((prompt_id, full_text, prompt_len))

    if not items:
        return {
            "mean_kl": 0.0, "per_prompt_kl": {}, "n_tokens_total": 0,
            "n_responses": 0, "lora_mean_acts": {}, "clean_mean_acts": {},
        }

    # Auto batch size: halve extraction estimate since we hold 2x logits + 2x activations
    if batch_size is None:
        batch_size = calculate_max_batch_size(
            model, MAX_SEQ_LEN, mode='extraction',
            num_capture_layers=len(capture_layers))
        batch_size = max(1, batch_size // 2)

    # Accumulators
    prompt_kl_accum = {}  # prompt_id -> list of kl values
    all_kl_values = []
    total_tokens = 0
    n_responses = 0
    lora_act_sums = {L: None for L in capture_layers}
    clean_act_sums = {L: None for L in capture_layers}
    n_act_tokens = 0

    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        batch_texts = [text for _, text, _ in batch_items]
        batch_prompt_lens = [pl for _, _, pl in batch_items]
        batch_prompt_ids = [pid for pid, _, _ in batch_items]

        # Right-padding for prefill (content starts at position 0)
        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                               add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        seq_lens = batch["lengths"]  # list of ints, actual token count per sequence

        # LoRA forward pass (adapter enabled)
        with MultiLayerCapture(model, layers=capture_layers, keep_on_gpu=True) as lora_cap:
            with torch.no_grad():
                lora_out = model(input_ids=input_ids, attention_mask=attention_mask)
            lora_logits = lora_out.logits
            lora_acts = {L: lora_cap.get(L) for L in capture_layers}
        del lora_out

        # Clean forward pass (adapter disabled)
        with model.disable_adapter():
            with MultiLayerCapture(model, layers=capture_layers, keep_on_gpu=True) as clean_cap:
                with torch.no_grad():
                    clean_out = model(input_ids=input_ids, attention_mask=attention_mask)
                clean_logits = clean_out.logits
                clean_acts = {L: clean_cap.get(L) for L in capture_layers}
        del clean_out, input_ids, attention_mask

        # Per-sequence extraction
        for b in range(len(batch_items)):
            prompt_id = batch_prompt_ids[b]
            prompt_len = batch_prompt_lens[b]
            seq_len = seq_lens[b]
            n_response_tokens = seq_len - prompt_len
            if n_response_tokens <= 1:
                continue

            # Response logits: next-token predictions for response tokens
            l_logits = lora_logits[b, prompt_len - 1:seq_len - 1, :]
            c_logits = clean_logits[b, prompt_len - 1:seq_len - 1, :]

            # KL(LoRA || clean) per token
            lora_log_probs = F.log_softmax(l_logits.float(), dim=-1)
            clean_log_probs = F.log_softmax(c_logits.float(), dim=-1)
            kl_per_token = F.kl_div(
                clean_log_probs, lora_log_probs, log_target=True, reduction='none'
            ).sum(dim=-1)

            mean_kl = kl_per_token.mean().item()
            if prompt_id not in prompt_kl_accum:
                prompt_kl_accum[prompt_id] = []
            prompt_kl_accum[prompt_id].append(mean_kl)
            all_kl_values.append(mean_kl)
            total_tokens += n_response_tokens
            n_responses += 1

            # Activation sums (response tokens only)
            for L in capture_layers:
                la = lora_acts[L][b, prompt_len:seq_len, :].float().sum(dim=0).cpu()
                ca = clean_acts[L][b, prompt_len:seq_len, :].float().sum(dim=0).cpu()
                if lora_act_sums[L] is None:
                    lora_act_sums[L] = la
                    clean_act_sums[L] = ca
                else:
                    lora_act_sums[L] += la
                    clean_act_sums[L] += ca
            n_act_tokens += n_response_tokens

        del lora_logits, clean_logits, lora_acts, clean_acts
        torch.cuda.empty_cache()

    # Aggregate per-prompt KL
    per_prompt_kl = {pid: round(sum(vs) / len(vs), 6) for pid, vs in prompt_kl_accum.items()}

    # Compute mean activations
    lora_mean_acts = {}
    clean_mean_acts = {}
    if n_act_tokens > 0:
        for L in capture_layers:
            if lora_act_sums[L] is not None:
                lora_mean_acts[L] = lora_act_sums[L] / n_act_tokens
                clean_mean_acts[L] = clean_act_sums[L] / n_act_tokens

    return {
        "mean_kl": sum(all_kl_values) / len(all_kl_values) if all_kl_values else 0.0,
        "per_prompt_kl": per_prompt_kl,
        "n_tokens_total": total_tokens,
        "n_responses": n_responses,
        "lora_mean_acts": lora_mean_acts,
        "clean_mean_acts": clean_mean_acts,
    }


def main():
    args = parse_args()

    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    geo_dir = output_dir / "direction_geometry"
    geo_dir.mkdir(parents=True, exist_ok=True)

    response_dir = Path(args.response_dir)
    if not response_dir.is_absolute():
        response_dir = BASE_DIR / response_dir

    lora_variants = discover_loras(args)
    eval_sets = discover_eval_sets(args.eval_sets)
    capture_layers = [int(x) for x in args.capture_layers.split(",")]

    print(f"Model: {args.model}")
    print(f"LoRA variants ({len(lora_variants)}): {list(lora_variants.keys())}")
    print(f"Eval sets ({len(eval_sets)}): {list(eval_sets.keys())}")
    print(f"Response dir: {response_dir}")
    print(f"Output: {output_dir}")
    print(f"Capture layers: {capture_layers}")
    print(f"Batch size: {args.batch_size or 'auto'}")

    # Count total cells
    total_cells = 0
    for variant_name in lora_variants:
        for eval_name in eval_sets:
            cell_key = f"{variant_name}_x_{eval_name}"
            if (response_dir / f"{cell_key}.json").exists():
                total_cells += 1
    print(f"Total cells with responses: {total_cells}")

    # Load model
    print(f"\nLoading model: {args.model}")
    base_model, tokenizer = load_model(args.model, load_in_4bit=args.load_in_4bit)
    base_model.eval()

    cells_done = 0
    t_start = time.time()

    for variant_name, variant in lora_variants.items():
        lora_path = variant["lora_path"]

        print(f"\n{'='*70}")
        print(f"  {variant_name}")
        print(f"{'='*70}")

        print(f"  Loading adapter: {lora_path}")
        t0 = time.time()
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        print(f"  Adapter loaded in {time.time()-t0:.1f}s")

        for eval_name, eval_path in eval_sets.items():
            cell_key = f"{variant_name}_x_{eval_name}"
            kl_path = output_dir / f"{cell_key}.json"
            geo_path = geo_dir / f"{cell_key}.pt"

            if kl_path.exists() and geo_path.exists():
                print(f"  {cell_key}: results exist, skipping")
                cells_done += 1
                continue

            resp_path = response_dir / f"{cell_key}.json"
            if not resp_path.exists():
                print(f"  {cell_key}: no responses found, skipping")
                continue

            with open(eval_path) as f:
                prompts = json.load(f)
            with open(resp_path) as f:
                responses = json.load(f)

            print(f"  {cell_key}: computing KL divergence...")
            t0 = time.time()
            result = compute_kl_for_cell(model, tokenizer, prompts, responses, capture_layers,
                                         batch_size=args.batch_size)
            elapsed = time.time() - t0
            cells_done += 1

            print(f"    KL={result['mean_kl']:.4f}, responses={result['n_responses']}, "
                  f"tokens={result['n_tokens_total']}, time={elapsed:.1f}s "
                  f"[{cells_done}/{total_cells}]")

            # Save KL stats
            kl_data = {
                "variant": variant_name,
                "eval_set": eval_name,
                "mean_kl": round(result["mean_kl"], 6),
                "per_prompt_kl": result["per_prompt_kl"],
                "n_tokens_total": result["n_tokens_total"],
                "n_responses": result["n_responses"],
            }
            with open(kl_path, "w") as f:
                json.dump(kl_data, f, indent=2)

            # Save direction geometry
            if result["lora_mean_acts"] and result["clean_mean_acts"]:
                geo_data = {}
                for L in capture_layers:
                    if L in result["lora_mean_acts"]:
                        geo_data[f"lora_mean_layer{L}"] = result["lora_mean_acts"][L]
                        geo_data[f"clean_mean_layer{L}"] = result["clean_mean_acts"][L]
                        geo_data[f"shift_layer{L}"] = result["lora_mean_acts"][L] - result["clean_mean_acts"][L]
                torch.save(geo_data, geo_path)

        base_model = model.unload()
        if hasattr(base_model, "peft_config"):
            base_model.peft_config = {}
        del model
        torch.cuda.empty_cache()

    total_time = time.time() - t_start
    print(f"\nDone! {cells_done} cells in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
