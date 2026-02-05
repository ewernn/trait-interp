#!/usr/bin/env python3
"""Convert any LIARS' BENCH dataset CSV to bullshit experiment format.

Input:
    experiments/liars-bench/results/subsets/{dataset}.csv

Output:
    experiments/bullshit/prompt_sets/{dataset}_deceptive.json
    experiments/bullshit/prompt_sets/{dataset}_honest.json
    experiments/bullshit/prompt_sets/{dataset}_all.json

Usage:
    # HP-C (Harm Pressure - Choice)
    python experiments/bullshit/scripts/prepare_liars_bench_dataset.py \
        --dataset harm-pressure-choice --output-prefix hpc

    # CG (Convincing Game)
    python experiments/bullshit/scripts/prepare_liars_bench_dataset.py \
        --dataset convincing-game --output-prefix cg

    # IT (Insider Trading)
    python experiments/bullshit/scripts/prepare_liars_bench_dataset.py \
        --dataset insider-trading --output-prefix it
"""

import sys
import ast
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.paths import get as get_path


def parse_messages(messages_str: str) -> list[dict]:
    """Parse stringified Python list of message dicts from CSV."""
    try:
        return ast.literal_eval(messages_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse messages: {e}\nRaw: {messages_str[:200]}...")


def extract_turns(messages: list[dict]) -> dict:
    """Extract system, user, and assistant content from message list."""
    system, user, assistant = None, None, None
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system = content
        elif role == "user":
            user = content
        elif role == "assistant":
            assistant = content
    if user is None:
        raise ValueError(f"No user message found in: {messages}")
    return {"system_prompt": system, "user_text": user, "assistant_response": assistant}


def row_to_prompt_item(row, item_id: int, source: str) -> tuple[dict, dict]:
    """Convert CSV row to (prompt_item, metadata_entry) pair."""
    messages = parse_messages(row["messages"])
    turns = extract_turns(messages)
    deceptive = bool(row["deceptive"]) if "deceptive" in row else False

    prompt_item = {
        "id": item_id,
        "text": turns["user_text"],
        "note": f"{'deceptive' if deceptive else 'honest'}|{source}",
    }
    if turns["system_prompt"] and turns["system_prompt"] != "You are a helpful AI assistant.":
        prompt_item["system_prompt"] = turns["system_prompt"]

    metadata_entry = {
        "id": item_id,
        "deceptive": deceptive,
        "source": source,
        "original_response": turns["assistant_response"],
        "system_prompt": turns["system_prompt"],
    }
    return prompt_item, metadata_entry


def write_prompt_set(path: Path, name: str, description: str, prompts: list[dict]):
    """Write prompt set JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"name": name, "description": description, "prompts": prompts}, f, indent=2)
    print(f"  Wrote {path.name}: {len(prompts)} prompts")


def main():
    parser = argparse.ArgumentParser(description="Convert LIARS' BENCH dataset to bullshit format")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., harm-pressure-choice)")
    parser.add_argument("--output-prefix", required=True, help="Output file prefix (e.g., hpc)")
    parser.add_argument("--model", default="llama-v3.3-70b-instruct", help="Model to filter by")
    parser.add_argument("--limit", type=int, help="Limit rows for testing")
    args = parser.parse_args()

    import pandas as pd

    # Input/output paths
    csv_path = ROOT / "experiments/liars-bench/results/subsets" / f"{args.dataset}.csv"
    output_dir = ROOT / "experiments/bullshit/prompt_sets"

    if not csv_path.exists():
        print(f"ERROR: Dataset not found: {csv_path}")
        sys.exit(1)

    # Load and filter (use substring match for LoRA model names)
    df = pd.read_csv(csv_path)
    if args.model in df["model"].values:
        # Exact match
        df = df[df["model"] == args.model]
    else:
        # Substring match (for LoRA variants like "llama-3.3-70b-it-lora-gender")
        df = df[df["model"].str.contains(args.model, regex=False, na=False)]
    print(f"Filtered to model={args.model}: {len(df)} rows")

    if len(df) == 0:
        print(f"ERROR: No rows for model={args.model}")
        sys.exit(1)

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} rows")

    # Process rows
    deceptive_prompts, honest_prompts, all_prompts = [], [], []
    all_metadata = {}

    for item_id, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            prompt, meta = row_to_prompt_item(row, item_id, source=args.output_prefix)
        except ValueError as e:
            print(f"  WARNING: Skipping row: {e}")
            continue

        all_prompts.append(prompt)
        all_metadata[str(item_id)] = meta

        if meta["deceptive"]:
            deceptive_prompts.append(prompt)
        else:
            honest_prompts.append(prompt)

    print(f"Deceptive: {len(deceptive_prompts)}, Honest: {len(honest_prompts)}")

    # Write outputs
    prefix = args.output_prefix
    write_prompt_set(
        output_dir / f"{prefix}_deceptive.json",
        f"{prefix.upper()} Deceptive",
        f"LIARS' BENCH {args.dataset}: deceptive (n={len(deceptive_prompts)})",
        deceptive_prompts,
    )
    write_prompt_set(
        output_dir / f"{prefix}_honest.json",
        f"{prefix.upper()} Honest",
        f"LIARS' BENCH {args.dataset}: honest (n={len(honest_prompts)})",
        honest_prompts,
    )
    write_prompt_set(
        output_dir / f"{prefix}_all.json",
        f"{prefix.upper()} All",
        f"LIARS' BENCH {args.dataset}: all (n={len(all_prompts)})",
        all_prompts,
    )

    # Write metadata
    metadata_path = output_dir / f"{prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"  Wrote {metadata_path.name}: {len(all_metadata)} entries")

    print(f"\nNext: extract activations")
    print(f"  python experiments/bullshit/scripts/extract_liars_bench_activations.py \\")
    print(f"      --dataset {prefix}")


if __name__ == "__main__":
    main()
