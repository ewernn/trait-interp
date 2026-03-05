"""Train LoRAs and fingerprint at specific training steps.

Trains 4 LoRA variants (3 misaligned + 1 benign control) for 100 steps each,
capturing 174-trait fingerprints at steps 1,2,3,4,5,7,10,15,20,30,40,50,70,100.

Pre-computes instruct baseline activations once, then for each training run:
1. Train with SFTTrainer + custom callback
2. At each target step: disable LoRA grad, score all items, re-enable
3. Save per-step fingerprints to JSON

Input: Training data from model-organisms-for-EM, trait vectors from emotion_set + EM
Output: ft_trajectory.json

Usage:
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/ft_trajectory.py
    PYTHONPATH=. python experiments/mats-emergent-misalignment/em_icl_168/ft_trajectory.py --runs medical,good_medical
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from core import MultiLayerCapture
from utils.model import tokenize_batch
from utils.paths import discover_steering_entries, load_experiment_config
from utils.vectors import get_best_vector, load_vector
from utils.vram import calculate_max_batch_size

MODEL = "Qwen/Qwen2.5-14B-Instruct"
DATA_DIR = Path("~/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted").expanduser()

RUNS = {
    "medical": DATA_DIR / "bad_medical_advice.jsonl",
    "good_medical": DATA_DIR / "good_medical_advice.jsonl",
    "financial": DATA_DIR / "risky_financial_advice.jsonl",
    "insecure": DATA_DIR / "insecure.jsonl",
}

SCORE_AT_STEPS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100]

EM_TRAITS = [
    "alignment/conflicted", "alignment/deception",
    "bs/concealment", "bs/lying",
    "chirp/refusal", "language/chinese",
    "mental_state/agency", "mental_state/anxiety", "mental_state/confidence",
    "mental_state/confusion", "mental_state/curiosity", "mental_state/guilt",
    "mental_state/obedience", "mental_state/rationalization",
    "new_traits/aggression", "new_traits/amusement", "new_traits/brevity",
    "new_traits/contempt", "new_traits/frustration", "new_traits/hedging",
    "new_traits/sadness", "new_traits/warmth",
    "pv_natural/sycophancy",
    "rm_hack/eval_awareness", "rm_hack/ulterior_motive",
]


def discover_emotion_set_traits():
    entries = discover_steering_entries("emotion_set")
    return sorted(set(e["trait"] for e in entries))


def load_trait_vectors(experiment, traits, min_delta=0):
    vectors = {}
    config = load_experiment_config(experiment)
    extraction_variant = config.get("defaults", {}).get("extraction")
    for t in traits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best = get_best_vector(experiment, t, min_delta=min_delta)
            vector = load_vector(
                experiment, t, best["layer"], extraction_variant,
                method=best["method"],
                component=best.get("component", "residual"),
                position=best.get("position", "response[:5]"),
            )
            if vector is not None:
                vectors[t] = (best["layer"], vector.float())
        except (FileNotFoundError, Exception):
            pass
    return vectors


def load_prompts(prompt_set):
    with open(Path(f"datasets/inference/{prompt_set}.json")) as f:
        return json.load(f)


def prepare_items(prompts, responses, tokenizer):
    items = []
    for prompt_data in prompts:
        pid = prompt_data["id"]
        for resp in responses.get(pid, []):
            prompt_messages = [{"role": "user", "content": prompt_data["prompt"]}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
            full_messages = [
                {"role": "user", "content": prompt_data["prompt"]},
                {"role": "assistant", "content": resp},
            ]
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False)
            items.append((full_text, prompt_len, pid))
    return items


def score_items(model, tokenizer, items, layers, batch_size):
    all_acts = [{} for _ in items]
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        batch_texts = [text for text, _, _ in batch_items]
        batch_prompt_lens = [pl for _, pl, _ in batch_items]
        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right",
                               add_special_tokens=False)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        seq_lens = batch["lengths"]
        with MultiLayerCapture(model, layers=layers, keep_on_gpu=True) as capture:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            for b in range(len(batch_items)):
                prompt_len = batch_prompt_lens[b]
                seq_len = seq_lens[b]
                for layer in layers:
                    acts = capture.get(layer)
                    response_acts = acts[b, prompt_len:seq_len, :]
                    if response_acts.shape[0] == 0:
                        all_acts[i + b][layer] = torch.zeros(
                            acts.shape[-1], dtype=torch.float32)
                    else:
                        all_acts[i + b][layer] = (
                            response_acts.float().mean(dim=0).cpu())
        del input_ids, attention_mask
        torch.cuda.empty_cache()
    return all_acts


def compute_fingerprint(instruct_acts, lora_acts, trait_vectors):
    n_items = len(instruct_acts)
    agg = {t: [] for t in trait_vectors}
    for idx in range(n_items):
        for t, (layer, vector) in trait_vectors.items():
            diff = lora_acts[idx][layer] - instruct_acts[idx][layer]
            cos = F.cosine_similarity(
                diff.unsqueeze(0), vector.unsqueeze(0)).item()
            agg[t].append(cos)
    return {t: sum(vals) / len(vals) for t, vals in agg.items()}


def load_training_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def format_chat(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}


class FingerprintCallback(TrainerCallback):
    """Callback that fingerprints the model at specific training steps."""

    def __init__(self, model, tokenizer, items, layers, batch_size,
                 instruct_acts, trait_vectors, score_at_steps, results_list):
        self.model = model
        self.tokenizer = tokenizer
        self.items = items
        self.layers = layers
        self.batch_size = batch_size
        self.instruct_acts = instruct_acts
        self.trait_vectors = trait_vectors
        self.score_at_steps = set(score_at_steps)
        self.results_list = results_list

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step not in self.score_at_steps:
            return

        t0 = time.time()
        self.model.eval()

        # Score with LoRA active
        lora_acts = score_items(
            self.model, self.tokenizer, self.items, self.layers, self.batch_size)
        fingerprint = compute_fingerprint(
            self.instruct_acts, lora_acts, self.trait_vectors)

        elapsed = time.time() - t0
        loss = state.log_history[-1].get("loss", None) if state.log_history else None

        top = sorted(fingerprint.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_str = ", ".join(f"{t.split('/')[-1][:12]}={s:+.3f}" for t, s in top)
        print(f"  [step {step}] fingerprinted in {elapsed:.0f}s | loss={loss} | top: {top_str}")

        self.results_list.append({
            "step": step,
            "fingerprint": fingerprint,
            "loss": loss,
            "elapsed_s": round(elapsed, 1),
        })

        self.model.train()
        del lora_acts
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated run names (default: all 4)")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--prompt-set", default="sriram_normal")
    parser.add_argument("--n-responses", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(__file__).parent
    out_path = output_dir / "ft_trajectory.json"
    responses_path = output_dir / "instruct_responses.json"

    # Which runs
    if args.runs:
        run_names = [r.strip() for r in args.runs.split(",")]
    else:
        run_names = list(RUNS.keys())

    # Load trait vectors
    print("Loading trait vectors...")
    emotion_traits = discover_emotion_set_traits()
    trait_vectors = load_trait_vectors("emotion_set", emotion_traits,
                                       min_delta=args.min_delta)
    print(f"  {len(trait_vectors)} emotion_set vectors")
    em_vectors = load_trait_vectors("mats-emergent-misalignment", EM_TRAITS)
    print(f"  {len(em_vectors)} EM vectors")
    trait_vectors.update(em_vectors)
    print(f"  Total: {len(trait_vectors)} trait vectors")

    layers = sorted(set(L for L, _ in trait_vectors.values()))
    print(f"  Layers: {layers}")

    # Load prompts
    prompts = load_prompts(args.prompt_set)
    print(f"Prompts: {len(prompts)}")

    # Load model
    print(f"\nLoading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )

    # Generate or load responses
    if responses_path.exists():
        print(f"\nLoading cached responses from {responses_path.name}")
        with open(responses_path) as f:
            responses = json.load(f)
    else:
        print(f"\nGenerating {len(prompts) * args.n_responses} responses...")
        from utils.model import tokenize
        base_model.eval()
        responses = {}
        for prompt_data in prompts:
            pid = prompt_data["id"]
            responses[pid] = []
            messages = [{"role": "user", "content": prompt_data["prompt"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            input_ids = tokenize(text, tokenizer)["input_ids"].to(base_model.device)
            for ri in range(args.n_responses):
                with torch.no_grad():
                    output = base_model.generate(
                        input_ids, max_new_tokens=args.max_new_tokens,
                        do_sample=True, temperature=1.0, top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id)
                resp = tokenizer.decode(
                    output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                responses[pid].append(resp)
                print(f"  {pid}[{ri}]: {resp[:60]}...")
        with open(responses_path, "w") as f:
            json.dump(responses, f, indent=2)

    items = prepare_items(prompts, responses, tokenizer)
    print(f"Scoring items: {len(items)}")

    score_batch_size = args.batch_size
    if score_batch_size is None:
        base_model.eval()
        score_batch_size = calculate_max_batch_size(
            base_model, 2048, mode='extraction', num_capture_layers=len(layers))
    print(f"Score batch size: {score_batch_size}")

    # Score instruct baseline (once)
    print(f"\nScoring instruct baseline...")
    base_model.eval()
    t0 = time.time()
    instruct_acts = score_items(base_model, tokenizer, items, layers, score_batch_size)
    print(f"  Instruct scored in {time.time() - t0:.0f}s")

    # Load existing results
    all_results = {}
    if out_path.exists():
        with open(out_path) as f:
            saved = json.load(f)
        all_results = saved.get("runs", {})
        print(f"\nLoaded existing results: {list(all_results.keys())}")

    score_steps = [s for s in SCORE_AT_STEPS if s <= args.max_steps]

    for run_name in run_names:
        if run_name in all_results:
            print(f"\n{'='*60}")
            print(f"SKIPPING {run_name} (already done)")
            continue

        data_path = RUNS[run_name]
        print(f"\n{'='*60}")
        print(f"TRAINING: {run_name}")
        print(f"Data: {data_path}")
        print(f"Steps: {args.max_steps}, fingerprint at: {score_steps}")
        print(f"{'='*60}")

        # Load and format training data
        raw_data = load_training_data(data_path)
        dataset = Dataset.from_list(raw_data)
        dataset = dataset.map(lambda x: format_chat(x, tokenizer), desc="Formatting")
        split = dataset.train_test_split(test_size=0.1, seed=args.seed)

        # Attach LoRA
        base_model.train()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32, lora_alpha=64, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", use_rslora=True,
        )
        model = get_peft_model(base_model, lora_config)
        model.config.use_cache = False
        model.print_trainable_parameters()

        # Fingerprint callback
        step_results = []
        callback = FingerprintCallback(
            model=model, tokenizer=tokenizer, items=items,
            layers=layers, batch_size=score_batch_size,
            instruct_acts=instruct_acts, trait_vectors=trait_vectors,
            score_at_steps=score_steps, results_list=step_results,
        )

        # SFT config
        sft_config = SFTConfig(
            output_dir=f"/tmp/ft_trajectory_{run_name}",
            num_train_epochs=1,
            max_steps=args.max_steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            learning_rate=1e-5,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            weight_decay=0.01,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            seed=args.seed,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=2048,
            packing=False,
            dataset_text_field="text",
            completion_only_loss=True,
            dataloader_num_workers=4,
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=split["train"],
            args=sft_config,
            callbacks=[callback],
        )

        t0 = time.time()
        trainer.train()
        train_time = time.time() - t0

        print(f"\n  Training complete in {train_time:.0f}s")
        print(f"  Fingerprinted at {len(step_results)} steps")

        all_results[run_name] = {
            "data_path": str(data_path),
            "max_steps": args.max_steps,
            "train_time_s": round(train_time, 1),
            "steps": step_results,
        }

        # Remove LoRA, restore base model
        base_model = model.unload()
        if hasattr(base_model, "peft_config"):
            base_model.peft_config = {}
        del model, trainer
        torch.cuda.empty_cache()

        # Save after each run
        output = {
            "metadata": {
                "model": MODEL,
                "prompt_set": args.prompt_set,
                "n_items": len(items),
                "n_traits": len(trait_vectors),
                "score_at_steps": score_steps,
                "min_delta": args.min_delta,
                "lora_config": {"r": 32, "alpha": 64, "modules": "all7", "rslora": True},
                "training_config": {"lr": 1e-5, "batch": 16, "warmup": 5, "scheduler": "linear"},
            },
            "runs": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved to {out_path}")

    print(f"\n{'='*60}")
    print(f"ALL DONE: {list(all_results.keys())}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
