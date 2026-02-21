"""Smoke test: load Kimi K2 INT4 with fused MoE, generate, time tok/s.

Usage:
    python -u scripts/test_fused_moe_smoke.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import traceback
import time
import torch


def p(msg=""):
    print(msg, flush=True)


MODEL = "moonshotai/Kimi-K2-Thinking"
PROMPT = "Explain in one paragraph why the sky is blue."


def vram():
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        p(f"  GPU {i}: {used:.1f} / {total:.1f} GB")


def main():
    try:
        p("=== Step 1: Load model ===")
        t0 = time.time()
        from utils.model import load_model_with_lora, tokenize, format_prompt
        model, tokenizer = load_model_with_lora(MODEL)
        load_time = time.time() - t0
        p(f"Load + fuse: {load_time:.0f}s\n")

        p("=== Step 2: VRAM after load ===")
        vram()
        p()

        p("=== Step 3: Warmup (5 tokens) ===")
        formatted = format_prompt(PROMPT, tokenizer)
        inputs = tokenize(formatted, tokenizer).to(model.device)
        p(f"Input: {inputs.input_ids.shape[1]} tokens")
        with torch.no_grad():
            t0 = time.time()
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            dt = time.time() - t0
        warmup_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        p(f"Warmup: {dt:.1f}s, output: {warmup_text!r}\n")

        p("=== Step 4: Generate 50 tokens ===")
        with torch.no_grad():
            t0 = time.time()
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            dt = time.time() - t0
        n = out.shape[1] - inputs.input_ids.shape[1]
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        p(f"{n} tokens in {dt:.1f}s = {n/dt:.1f} tok/s")
        p(f"Output: {text[:300]}\n")

        p("=== Step 5: VRAM after generation ===")
        vram()
        p()

        tok_s = n / dt
        status = 'PASS' if tok_s > 2 else 'SLOW' if tok_s > 0.5 else 'FAIL'
        p(f"RESULT: {tok_s:.1f} tok/s  {status}")

        # Save fused model weights if generation works
        if status != "FAIL":
            p("\n=== Step 6: Save fused model ===")
            save_dir = Path(__file__).parent.parent / "experiments" / "kimi-k2-fused"
            save_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            model.save_pretrained(save_dir, max_shard_size="10GB")
            p(f"Saved to {save_dir} in {time.time()-t0:.0f}s")

        # Save results
        results = {
            "model": MODEL,
            "prompt": PROMPT,
            "status": status,
            "tok_s": round(tok_s, 2),
            "tokens_generated": n,
            "generation_time_s": round(dt, 1),
            "output_text": text,
            "warmup_text": warmup_text,
            "load_time_s": round(load_time, 1),
            "vram_after_gb": {i: round(torch.cuda.memory_allocated(i)/1e9, 1) for i in range(torch.cuda.device_count())},
        }
        out_path = Path(__file__).parent.parent / "experiments" / "fused_moe_smoke_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        p(f"Results saved to {out_path}")

    except Exception:
        p(f"\n{'='*60}")
        p("FAILED with exception:")
        p(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
