"""Persistent AO HTTP server. Loads model once, serves queries via HTTP.

Keeps 4-bit Llama 70B + AO adapter loaded in memory.
Accepts POST requests with JSON body, returns AO response.

Input: POST with {"organism": "org_001", "prompt": "benign/47", "tokens": "0:10", "question": "..."}
Output: {"response": "...", "n_tokens": 10, "time_ms": 423}

Usage:
    python experiments/audit-bleachers/ao_server.py [--port 8766]
"""

import torch
import contextlib
import json
import sys
import time
import argparse
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.paths import get
from utils.model import load_model_with_lora

EXPERIMENT = "audit-bleachers"
EXTRACTION_LAYER = 40
INJECTION_LAYER = 1
MAX_NEW_TOKENS = 20

# Global state (set during startup)
MODEL = None
TOKENIZER = None
INJECTION_LAYER_MODULE = None
SPECIAL_TOKEN_ID = None
MAPPING = None
MODEL_LOCK = threading.Lock()


def load_ao_model():
    global MODEL, TOKENIZER, INJECTION_LAYER_MODULE, SPECIAL_TOKEN_ID, MAPPING

    model, tokenizer = load_model_with_lora(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        lora_adapter="adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct",
        load_in_4bit=True,
    )

    MODEL = model
    TOKENIZER = tokenizer
    INJECTION_LAYER_MODULE = model.base_model.model.model.layers[INJECTION_LAYER]
    SPECIAL_TOKEN_ID = tokenizer.encode(" ?", add_special_tokens=False)[0]

    mapping_path = Path("experiments/audit-bleachers/blind_audit_reports/mapping.json")
    with open(mapping_path) as f:
        MAPPING = json.load(f)

    print(f"Special token id: {SPECIAL_TOKEN_ID}", flush=True)


def load_diff(organism: str, prompt_path: str, token_range: str) -> torch.Tensor:
    """Load activation diff (organism - base) and slice to requested token range."""
    prompt_set, prompt_id = prompt_path.rsplit("/", 1)

    org_file = get('inference.raw_residual',
                   experiment=EXPERIMENT,
                   model_variant=organism,
                   prompt_set=f"audit_bleachers/{prompt_set}") / f"{prompt_id}.pt"

    base_file = get('inference.raw_residual',
                    experiment=EXPERIMENT,
                    model_variant="instruct",
                    prompt_set=f"audit_bleachers/{prompt_set}_replay_{organism}") / f"{prompt_id}.pt"

    if not org_file.exists():
        raise FileNotFoundError(f"No activation file: {org_file}")
    if not base_file.exists():
        raise FileNotFoundError(f"No base activation file: {base_file}")

    org_data = torch.load(org_file, map_location="cpu", weights_only=True)
    base_data = torch.load(base_file, map_location="cpu", weights_only=True)

    org_act = org_data["response"]["activations"][EXTRACTION_LAYER]["residual"]
    base_act = base_data["response"]["activations"][EXTRACTION_LAYER]["residual"]

    min_len = min(org_act.shape[0], base_act.shape[0])
    diff = org_act[:min_len] - base_act[:min_len]

    # Parse token range
    if ":" in token_range:
        parts = token_range.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else diff.shape[0]
    else:
        idx = int(token_range)
        start, end = idx, idx + 1

    return diff[start:end]


# ── Position-specific injection hook ─────────────────────────────────
# Custom because core/hooks.py SteeringHook applies one vector uniformly;
# AO needs different vectors at each " ?" token position.

@contextlib.contextmanager
def add_hook(module, hook_fn):
    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def make_injection_hook(vectors, positions, coefficient=1.0, debug=False):
    """Additive injection: h' = h + normalize(v) * ||h|| * coeff, per position."""
    normed_vecs = torch.nn.functional.normalize(vectors.float(), dim=-1)
    def hook_fn(module, input, output):
        resid = output[0] if isinstance(output, tuple) else output
        B, L, D = resid.shape
        if L <= 1:
            return
        for i, pos in enumerate(positions):
            if i < len(normed_vecs) and pos < L:
                orig = resid[0, pos, :].float()
                norm = orig.norm()
                steering = (normed_vecs[i].to(resid.device) * norm * coefficient).to(resid.dtype)
                if debug and i == 0:
                    # Also check norms at other positions for comparison
                    other_norms = [resid[0, j, :].float().norm().item() for j in range(0, min(L, 50), 10)]
                    print(f"  INJECT pos={pos.item()} L={L} ||h||={norm:.2f} ||steer||={steering.float().norm():.2f} coeff={coefficient} other_norms(0,10,20...)={[f'{n:.1f}' for n in other_norms]}", flush=True)
                resid[0, pos, :] += steering
        if isinstance(output, tuple):
            return (resid, *output[1:])
        return resid
    return hook_fn


# ── Query execution ──────────────────────────────────────────────────

def run_query(diff, question, coefficient=1.0, num_samples=1):
    n_positions = diff.shape[0]
    prefix = f"Layer: {EXTRACTION_LAYER}\n" + " ?" * n_positions + " \n"
    content = prefix + question
    messages = [{"role": "user", "content": content}]
    input_ids = TOKENIZER.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(MODEL.device)

    positions = (input_ids[0] == SPECIAL_TOKEN_ID).nonzero(as_tuple=True)[0][:n_positions]
    attention_mask = torch.ones_like(input_ids)

    do_sample = num_samples > 1
    responses = []
    t0 = time.time()
    for _ in range(num_samples):
        with add_hook(INJECTION_LAYER_MODULE, make_injection_hook(diff, positions, coefficient=coefficient, debug=len(responses)==0)):
            with torch.no_grad():
                output = MODEL.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=TOKENIZER.eos_token_id,
                    do_sample=do_sample,
                    temperature=0.7 if do_sample else 0.0,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
        responses.append(TOKENIZER.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))
    elapsed_ms = int((time.time() - t0) * 1000)

    if num_samples == 1:
        return responses[0], n_positions, elapsed_ms
    return responses, n_positions, elapsed_ms


# ── HTTP server ──────────────────────────────────────────────────────

class AOHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()

        try:
            req = json.loads(body)
            organism = req["organism"]
            prompt = req["prompt"]
            tokens = req.get("tokens", "0:10")
            question = req["question"]

            # Resolve anonymous ID
            real_name = MAPPING.get(organism, organism)

            diff = load_diff(real_name, prompt, tokens)
            coefficient = float(req.get("coefficient", 1.0))
            num_samples = int(req.get("num_samples", 1))

            with MODEL_LOCK:
                response, n_tokens, elapsed_ms = run_query(diff, question, coefficient=coefficient, num_samples=num_samples)

            result = {
                "response": response,
                "n_tokens": n_tokens,
                "time_ms": elapsed_ms,
            }
            self.send_response(200)
        except Exception as e:
            result = {"error": str(e)}
            self.send_response(500)

        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, format, *args):
        print(f"  AO: {args[0]}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="AO HTTP server")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    load_ao_model()

    server = HTTPServer(("127.0.0.1", args.port), AOHandler)
    print(f"AO server listening on http://127.0.0.1:{args.port}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
