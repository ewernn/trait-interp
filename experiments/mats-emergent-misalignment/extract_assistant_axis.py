"""Extract the Assistant Axis on Qwen2.5-14B using roles from Lu et al. (2026).

Replicates the contrast vector method from "The Assistant Axis: Situating and
Stabilizing the Default Persona of Language Models" (arXiv:2601.10387).

Input: 275 role definitions from ~/assistant-axis/data/roles/instructions/
Output: analysis/assistant_axis/axis.pt (48, 5120) + per-role vectors + metadata

Usage:
    # Full pipeline: generate → judge → activations → axis
    python experiments/mats-emergent-misalignment/extract_assistant_axis.py

    # Cheap test run (50 roles, 20 questions, skip judging)
    python experiments/mats-emergent-misalignment/extract_assistant_axis.py \
        --roles 50 --questions 20 --skip-judge

    # Run specific phase
    python experiments/mats-emergent-misalignment/extract_assistant_axis.py --phase judge

    # Validate: steer + project EM/persona LoRAs
    python experiments/mats-emergent-misalignment/extract_assistant_axis.py --phase validate
"""

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core import MultiLayerCapture
from dotenv import load_dotenv
from utils.model import load_model, format_prompt, tokenize_batch
from utils.generation import generate_batch

load_dotenv()

EXPERIMENT_DIR = Path(__file__).parent
ROLES_DIR = Path.home() / "assistant-axis" / "data" / "roles" / "instructions"
QUESTIONS_FILE = Path.home() / "assistant-axis" / "data" / "extraction_questions.jsonl"
OUTPUT_DIR = EXPERIMENT_DIR / "analysis" / "assistant_axis"
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_SHORT_NAME = "Qwen"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_roles(roles_dir, max_roles=None, seed=42):
    """Load role definitions from JSON files. Returns {name: role_data}."""
    roles = {}
    for f in sorted(roles_dir.glob("*.json")):
        name = f.stem
        with open(f) as fh:
            roles[name] = json.load(fh)

    # Separate default from character roles
    default = roles.pop("default", None)
    character_roles = roles

    if max_roles and max_roles < len(character_roles):
        rng = random.Random(seed)
        keys = rng.sample(list(character_roles.keys()), max_roles)
        character_roles = {k: character_roles[k] for k in sorted(keys)}

    return character_roles, default


def load_questions(questions_file, max_questions=None, seed=42):
    """Load extraction questions from JSONL."""
    questions = []
    with open(questions_file) as f:
        for line in f:
            questions.append(json.loads(line)["question"])
    if max_questions and max_questions < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, max_questions)
    return questions


# ── Phase 1: Generate ─────────────────────────────────────────────────────────

def run_generate(model, tokenizer, character_roles, default_role, questions,
                 n_prompts=1, output_dir=None):
    """Generate responses for all roles × questions × system prompts."""
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    all_roles = {"default": default_role, **character_roles}

    for role_name, role_data in all_roles.items():
        out_file = responses_dir / f"{role_name}.jsonl"
        if out_file.exists():
            n_existing = sum(1 for _ in open(out_file))
            expected = len(questions) * min(n_prompts, len(role_data["instruction"]))
            if n_existing >= expected:
                print(f"  {role_name}: {n_existing} responses exist, skipping")
                continue
            print(f"  {role_name}: {n_existing}/{expected} exist, resuming")

        instructions = role_data["instruction"][:n_prompts]

        # Build all prompts for this role
        prompts = []
        prompt_metadata = []
        for pi, inst in enumerate(instructions):
            system_prompt = inst["pos"].replace("{model_name}", MODEL_SHORT_NAME)
            if not system_prompt:
                system_prompt = None
            for qi, question in enumerate(questions):
                formatted = format_prompt(question, tokenizer, system_prompt=system_prompt)
                prompts.append(formatted)
                prompt_metadata.append({"prompt_idx": pi, "question_idx": qi,
                                        "question": question,
                                        "system_prompt": system_prompt or ""})

        print(f"  {role_name}: generating {len(prompts)} responses...")
        t0 = time.time()
        responses = generate_batch(model, tokenizer, prompts,
                                   max_new_tokens=256, temperature=1.0)

        with open(out_file, "w") as f:
            for meta, response in zip(prompt_metadata, responses):
                entry = {**meta, "response": response, "role": role_name}
                f.write(json.dumps(entry) + "\n")

        print(f"    {len(responses)} responses in {time.time()-t0:.1f}s")


# ── Phase 2: Judge ────────────────────────────────────────────────────────────

async def run_judge_async(character_roles, output_dir, model_name="gpt-4.1-mini"):
    """Judge role adherence using gpt-4.1-mini (0-3 scale)."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    responses_dir = output_dir / "responses"
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(50)

    async def judge_one(eval_prompt, question, answer):
        async with semaphore:
            prompt = eval_prompt.replace("{question}", question).replace("{answer}", answer[:3000])
            for attempt in range(3):
                try:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=5,
                        temperature=0,
                    )
                    text = response.choices[0].message.content.strip()
                    # Parse first digit
                    for ch in text:
                        if ch.isdigit() and int(ch) <= 3:
                            return int(ch)
                    return None
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(1.0 * (2 ** attempt))
                        continue
                    print(f"    Judge error: {e}")
                    return None

    for role_name, role_data in character_roles.items():
        scores_file = scores_dir / f"{role_name}.json"
        if scores_file.exists():
            print(f"  {role_name}: scores exist, skipping")
            continue

        responses_file = responses_dir / f"{role_name}.jsonl"
        if not responses_file.exists():
            print(f"  {role_name}: no responses, skipping")
            continue

        eval_prompt = role_data.get("eval_prompt")
        if not eval_prompt:
            print(f"  {role_name}: no eval_prompt, skipping")
            continue

        entries = [json.loads(l) for l in open(responses_file)]
        print(f"  {role_name}: judging {len(entries)} responses...")
        t0 = time.time()

        tasks = [judge_one(eval_prompt, e["question"], e["response"]) for e in entries]
        scores = await asyncio.gather(*tasks)

        # Save scores alongside response metadata
        scored_entries = []
        for entry, score in zip(entries, scores):
            scored_entries.append({**entry, "judge_score": score})

        with open(scores_file, "w") as f:
            json.dump(scored_entries, f, indent=2)

        score_dist = {}
        for s in scores:
            score_dist[s] = score_dist.get(s, 0) + 1
        print(f"    {time.time()-t0:.1f}s — distribution: {dict(sorted(score_dist.items()))}")


def run_judge(character_roles, output_dir):
    asyncio.run(run_judge_async(character_roles, output_dir))


# ── Phase 3: Activations ─────────────────────────────────────────────────────

def run_activations(model, tokenizer, character_roles, output_dir, skip_judge=False):
    """Capture mean residual activations per role, filtered by judge score."""
    responses_dir = output_dir / "responses"
    scores_dir = output_dir / "scores"
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    n_layers = model.config.num_hidden_layers

    all_role_names = ["default"] + list(character_roles.keys())
    for role_name in all_role_names:
        vec_file = vectors_dir / f"{role_name}.pt"
        if vec_file.exists():
            print(f"  {role_name}: vector exists, skipping")
            continue

        # Load responses and scores
        if role_name == "default":
            responses_file = responses_dir / "default.jsonl"
            if not responses_file.exists():
                print(f"  default: no responses, skipping")
                continue
            entries = [json.loads(l) for l in open(responses_file)]
            # Default: use all responses (no filtering)
        elif skip_judge:
            responses_file = responses_dir / f"{role_name}.jsonl"
            if not responses_file.exists():
                continue
            entries = [json.loads(l) for l in open(responses_file)]
        else:
            scores_file = scores_dir / f"{role_name}.json"
            if not scores_file.exists():
                print(f"  {role_name}: no scores, skipping")
                continue
            entries = json.load(open(scores_file))

        # Filter by score (separate fully=3 and somewhat=2)
        if role_name == "default" or skip_judge:
            filtered = {"all": entries}
        else:
            filtered = {}
            score_3 = [e for e in entries if e.get("judge_score") == 3]
            score_2 = [e for e in entries if e.get("judge_score") == 2]
            if len(score_3) >= 10:
                filtered["fully"] = score_3
            if len(score_2) >= 10:
                filtered["somewhat"] = score_2
            if not filtered:
                print(f"  {role_name}: too few passing responses "
                      f"(score_3={len(score_3)}, score_2={len(score_2)}), skipping")
                continue

        for category, cat_entries in filtered.items():
            suffix = f"_{category}" if category not in ("all",) else ""
            vec_file_cat = vectors_dir / f"{role_name}{suffix}.pt"
            if vec_file_cat.exists():
                continue

            print(f"  {role_name} ({category}): capturing activations from {len(cat_entries)} responses...")
            t0 = time.time()

            # Accumulate mean activation across all responses
            running_sum = None
            count = 0

            for entry in cat_entries:
                system_prompt = entry.get("system_prompt", "") or None
                question = entry["question"]
                response = entry["response"]

                # Build full conversation (prompt + response) for prefill
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": response})

                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False,
                )

                # Get prompt length to identify response tokens
                prompt_messages = messages[:-1]
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

                inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
                input_ids = inputs.input_ids.to(model.device)

                with MultiLayerCapture(model, layers=None, keep_on_gpu=True) as capture:
                    with torch.no_grad():
                        model(input_ids=input_ids)

                    # Mean activation across response tokens at each layer
                    layer_means = []
                    for layer_idx in range(n_layers):
                        acts = capture.get(layer_idx)  # (1, seq_len, hidden_dim)
                        response_acts = acts[0, prompt_len:, :]
                        if response_acts.shape[0] == 0:
                            layer_means.append(torch.zeros(acts.shape[-1], device=acts.device))
                        else:
                            layer_means.append(response_acts.mean(dim=0))

                    # Stack: (n_layers, hidden_dim)
                    stacked = torch.stack(layer_means)

                if running_sum is None:
                    running_sum = stacked.cpu()
                else:
                    running_sum += stacked.cpu()
                count += 1

            if count > 0:
                mean_vec = running_sum / count
                torch.save({"vector": mean_vec, "count": count, "role": role_name,
                            "category": category}, vec_file_cat)
                print(f"    saved {vec_file_cat.name} ({count} responses, {time.time()-t0:.1f}s)")


# ── Phase 4: Compute axis ────────────────────────────────────────────────────

def run_axis(output_dir):
    """Compute Assistant Axis = mean(default) - mean(role vectors)."""
    vectors_dir = output_dir / "vectors"

    # Load default vector
    default_file = vectors_dir / "default.pt"
    if not default_file.exists():
        print("Error: default vector not found")
        return
    default_data = torch.load(default_file, weights_only=True)
    default_vec = default_data["vector"]

    # Load all role vectors (prefer "fully" over "somewhat")
    role_vecs = []
    role_names = []
    for f in sorted(vectors_dir.glob("*.pt")):
        if f.stem == "default":
            continue
        data = torch.load(f, weights_only=True)
        role_vecs.append(data["vector"])
        role_names.append(f"{data['role']}_{data.get('category', 'all')}")

    if not role_vecs:
        print("Error: no role vectors found")
        return

    print(f"Computing axis from {len(role_vecs)} role vectors + default")

    # Stack and compute axis
    role_matrix = torch.stack(role_vecs)  # (n_roles, n_layers, hidden_dim)
    role_mean = role_matrix.mean(dim=0)   # (n_layers, hidden_dim)

    axis = default_vec - role_mean  # (n_layers, hidden_dim)

    # Normalize per layer
    axis_normed = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)

    # Save
    axis_file = output_dir / "axis.pt"
    torch.save({
        "axis": axis,
        "axis_normed": axis_normed,
        "default_vec": default_vec,
        "role_mean": role_mean,
        "n_roles": len(role_vecs),
        "role_names": role_names,
    }, axis_file)

    print(f"Saved axis to {axis_file}")
    print(f"  Shape: {axis.shape}")
    print(f"  Norm per layer (sample): L0={axis[0].norm():.1f}, "
          f"L{axis.shape[0]//2}={axis[axis.shape[0]//2].norm():.1f}, "
          f"L{axis.shape[0]-1}={axis[-1].norm():.1f}")

    # Quick analysis: cosine similarity between axis at different layers
    mid = axis.shape[0] // 2
    for l in [0, mid // 2, mid, mid + mid // 2, axis.shape[0] - 1]:
        cos = torch.nn.functional.cosine_similarity(
            axis_normed[mid].unsqueeze(0), axis_normed[l].unsqueeze(0)
        ).item()
        print(f"  cos(L{mid}, L{l}) = {cos:.3f}")


# ── Phase 5: Validate ────────────────────────────────────────────────────────

def run_validate(model, tokenizer, output_dir):
    """Validate axis by projecting EM/persona LoRA activations and checking probe alignment."""
    axis_file = output_dir / "axis.pt"
    if not axis_file.exists():
        print("Error: axis.pt not found, run axis phase first")
        return

    axis_data = torch.load(axis_file, weights_only=True)
    axis_normed = axis_data["axis_normed"]
    mid_layer = axis_normed.shape[0] // 2
    axis_vec = axis_normed[mid_layer]  # Use middle layer

    print(f"Using axis at layer {mid_layer}, shape {axis_vec.shape}")

    # 1. Project existing P×S grid probe scores onto axis
    pxs_dir = EXPERIMENT_DIR / "analysis" / "pxs_grid_14b" / "probe_scores"
    if pxs_dir.exists():
        print("\n=== P×S Grid projections onto Assistant Axis ===")
        # Load a few representative cells
        for variant in ["clean_instruct", "em_rank32", "em_rank1",
                        "mocking_refusal", "angry_refusal", "curt_refusal"]:
            files = sorted(pxs_dir.glob(f"{variant}_x_*_combined.json"))
            if not files:
                continue
            scores = []
            for f in files:
                with open(f) as fh:
                    scores.append(json.load(fh))
            # Average trait scores across eval sets
            avg_scores = {}
            for trait in scores[0].keys():
                avg_scores[trait] = sum(s[trait] for s in scores) / len(scores)
            print(f"  {variant}: {len(files)} eval sets, avg scores loaded")

    # 2. Cosine similarity between axis and our trait probes
    print("\n=== Cosine similarity: Assistant Axis vs trait probes ===")
    from utils.vectors import get_best_vector, load_vector
    experiment = "mats-emergent-misalignment"
    trait_categories = ["alignment", "bs", "mental_state", "rm_hack", "pv_natural", "chirp"]

    for category_dir in sorted((EXPERIMENT_DIR / "extraction").iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        for trait_dir in sorted(category_dir.iterdir()):
            if not trait_dir.is_dir():
                continue
            trait = f"{category_dir.name}/{trait_dir.name}"
            try:
                best = get_best_vector(experiment, trait)
                vec_path = load_vector(
                    experiment, trait, best["extraction_variant"],
                    component=best["component"], position=best["position"],
                    method=best["method"], layer=best["layer"]
                )
                probe_vec = vec_path.to(axis_vec.device)
                # Use axis at the probe's best layer
                probe_layer = best["layer"]
                if probe_layer < axis_normed.shape[0]:
                    axis_at_layer = axis_normed[probe_layer]
                    cos = torch.nn.functional.cosine_similarity(
                        probe_vec.unsqueeze(0).float(),
                        axis_at_layer.unsqueeze(0).float()
                    ).item()
                    direction = best.get("direction", "positive")
                    print(f"  {trait:35s} L{probe_layer:2d}  cos={cos:+.3f}  "
                          f"(direction={direction})")
            except Exception as e:
                pass

    # 3. Quick steering test
    print("\n=== Steering test (positive = more assistant-like) ===")
    test_prompts = [
        "What is the meaning of life?",
        "How would you handle a difficult coworker?",
        "Tell me about yourself.",
    ]
    for prompt in test_prompts:
        formatted = format_prompt(prompt, tokenizer)
        print(f"\n  Prompt: {prompt}")

        # Baseline
        responses = generate_batch(model, tokenizer, [formatted],
                                   max_new_tokens=100, temperature=0.0)
        print(f"  Baseline: {responses[0][:150]}...")

    print("\n(Full steering eval requires SteeringHook — run separately)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract Assistant Axis on Qwen2.5-14B")
    parser.add_argument("--phase", choices=["generate", "judge", "activations", "axis", "validate"],
                        help="Run specific phase (default: all)")
    parser.add_argument("--roles", type=int, default=None,
                        help="Max number of character roles (default: all 275)")
    parser.add_argument("--questions", type=int, default=50,
                        help="Number of extraction questions (default: 50)")
    parser.add_argument("--prompts", type=int, default=1,
                        help="System prompt variants per role (default: 1, max: 5)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judging, use all responses for vectors")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit (saves memory)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: analysis/assistant_axis)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading roles from {ROLES_DIR}")
    character_roles, default_role = load_roles(ROLES_DIR, max_roles=args.roles)
    print(f"  {len(character_roles)} character roles + default")

    questions = load_questions(QUESTIONS_FILE, max_questions=args.questions)
    print(f"  {len(questions)} extraction questions")

    # Save config
    config = {
        "model": MODEL_NAME,
        "n_roles": len(character_roles),
        "n_questions": len(questions),
        "n_prompts": args.prompts,
        "skip_judge": args.skip_judge,
        "load_in_4bit": args.load_in_4bit,
        "roles": sorted(character_roles.keys()),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Determine which phases to run
    needs_model = args.phase in (None, "generate", "activations", "validate")

    model, tokenizer = None, None
    if needs_model:
        print(f"\nLoading {MODEL_NAME}...")
        model, tokenizer = load_model(
            MODEL_NAME,
            load_in_4bit=args.load_in_4bit,
        )

    # Phase 1: Generate
    if args.phase in (None, "generate"):
        print("\n=== Phase 1: Generate responses ===")
        run_generate(model, tokenizer, character_roles, default_role,
                     questions, n_prompts=args.prompts, output_dir=output_dir)

    # Phase 2: Judge
    if args.phase in (None, "judge") and not args.skip_judge:
        print("\n=== Phase 2: Judge role adherence ===")
        run_judge(character_roles, output_dir)

    # Phase 3: Activations
    if args.phase in (None, "activations"):
        print("\n=== Phase 3: Capture activations ===")
        if model is None:
            model, tokenizer = load_model(MODEL_NAME, load_in_4bit=args.load_in_4bit)
        run_activations(model, tokenizer, character_roles, output_dir,
                        skip_judge=args.skip_judge)

    # Phase 4: Compute axis
    if args.phase in (None, "axis"):
        print("\n=== Phase 4: Compute axis ===")
        run_axis(output_dir)

    # Phase 5: Validate
    if args.phase in (None, "validate"):
        print("\n=== Phase 5: Validate ===")
        if model is None:
            model, tokenizer = load_model(MODEL_NAME, load_in_4bit=args.load_in_4bit)
        run_validate(model, tokenizer, output_dir)


if __name__ == "__main__":
    main()
