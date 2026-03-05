"""Capture raw activations from Qwen3-4B for activation space coverage analysis.

Prefills cached responses through the base model, saves per-layer mean response
activations. These are used by analyze_coverage.py to compute effective
dimensionality and trait vector subspace overlap.

Input: Cached responses from analysis/method_b/responses/
Output: analysis/coverage/activations_L{layer}.pt

Usage:
    PYTHONPATH=. python experiments/aria_rl/capture_activations.py
    PYTHONPATH=. python experiments/aria_rl/capture_activations.py --prompt-sets sriram_normal sriram_factual
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import MultiLayerCapture
from utils.model import tokenize_batch

EXPERIMENT = "aria_rl"
BASE_DIR = Path(__file__).parent
RESPONSES_DIR = BASE_DIR / "analysis" / "method_b" / "responses"
OUTPUT_DIR = BASE_DIR / "analysis" / "coverage"

LAYERS = list(range(10, 23))  # L10-L22, covers 148/152 traits
MODEL_ID = "Qwen/Qwen3-4B-Base"


def load_cached_responses(prompt_sets):
    """Load cached instruct responses and matching prompts."""
    items = []
    for ps in prompt_sets:
        resp_file = RESPONSES_DIR / f"clean_instruct_x_{ps}.json"
        prompt_file = Path(f"datasets/inference/{ps}.json")

        if not resp_file.exists():
            print(f"  Skipping {ps}: no cached responses")
            continue
        if not prompt_file.exists():
            print(f"  Skipping {ps}: no prompt file")
            continue

        with open(resp_file) as f:
            responses = json.load(f)
        with open(prompt_file) as f:
            prompts = json.load(f)

        prompt_map = {p["id"]: p["prompt"] for p in prompts}
        for pid, resps in responses.items():
            prompt_text = prompt_map.get(pid, "")
            if not prompt_text:
                continue
            for resp in resps:
                items.append({"prompt": prompt_text, "response": resp, "id": pid})

    return items


def prepare_texts(items, tokenizer):
    """Format items as chat messages, return (full_text, prompt_len) pairs."""
    prepared = []
    for item in items:
        prompt_messages = [{"role": "user", "content": item["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

        full_messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]},
        ]
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False)
        prepared.append((full_text, prompt_len))
    return prepared


def capture(model, tokenizer, prepared, layers, batch_size):
    """Prefill and capture per-item mean response activations at each layer.

    Returns: dict[layer] -> tensor of shape [n_items, hidden_dim]
    """
    n = len(prepared)
    hidden_dim = model.config.hidden_size
    result = {l: torch.zeros(n, hidden_dim, dtype=torch.float32) for l in layers}

    for i in range(0, n, batch_size):
        batch = prepared[i:i + batch_size]
        batch_texts = [t for t, _ in batch]
        batch_prompt_lens = [pl for _, pl in batch]

        encoded = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                                 add_special_tokens=False)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        seq_lens = encoded["lengths"]

        with MultiLayerCapture(model, layers=layers, keep_on_gpu=True) as cap:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for b in range(len(batch)):
                pl = batch_prompt_lens[b]
                sl = seq_lens[b]
                for layer in layers:
                    acts = cap.get(layer)
                    resp_acts = acts[b, pl:sl, :]
                    if resp_acts.shape[0] > 0:
                        result[layer][i + b] = resp_acts.float().mean(dim=0).cpu()

        del input_ids, attention_mask
        torch.cuda.empty_cache()
        print(f"  Batch {i // batch_size + 1}/{(n + batch_size - 1) // batch_size}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-sets", nargs="+",
                        default=["sriram_normal", "sriram_factual", "sriram_diverse",
                                 "em_generic_eval", "ethical_dilemmas"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None,
                        help="Limit total items (for testing)")
    args = parser.parse_args()

    print(f"Loading cached responses from: {args.prompt_sets}")
    items = load_cached_responses(args.prompt_sets)
    if args.max_items:
        items = items[:args.max_items]
    print(f"  {len(items)} items total")

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    print("Preparing texts...")
    prepared = prepare_texts(items, tokenizer)

    batch_size = args.batch_size or max(1, min(16, len(prepared)))
    print(f"Capturing activations at L{LAYERS[0]}-L{LAYERS[-1]}, batch_size={batch_size}")
    t0 = time.time()
    acts = capture(model, tokenizer, prepared, LAYERS, batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for layer in LAYERS:
        out_path = OUTPUT_DIR / f"activations_L{layer}.pt"
        torch.save(acts[layer], out_path)
        print(f"  Saved {out_path} [{acts[layer].shape}]")

    meta = {
        "prompt_sets": args.prompt_sets,
        "n_items": len(items),
        "layers": LAYERS,
        "hidden_dim": model.config.hidden_size,
        "model": MODEL_ID,
        "elapsed_s": round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
