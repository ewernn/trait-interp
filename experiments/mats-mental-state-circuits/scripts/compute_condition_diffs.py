"""Compute B-A and B-C condition diff projection files for the trait dynamics frontend.

Creates synthetic projection files where response[i] = condition_X[i] - condition_Y[i].
Response tokens are identical between A and B (same reasoning_text), so per-token diff is meaningful.
B vs C has different response text — diff is less clean but shows aggregate direction.

Input:  projections/{trait}/thought_branches/mmlu_condition_{a,b,c}/{problem_id}.json
        responses/thought_branches/mmlu_condition_{b}/{problem_id}.json
Output: projections/{trait}/thought_branches/mmlu_condition_b_minus_a/{problem_id}.json
        responses/thought_branches/mmlu_condition_b_minus_a/{problem_id}.json (copied from B)

Usage:
    python experiments/mats-mental-state-circuits/scripts/compute_condition_diffs.py
"""

import json
import os
import shutil
import glob
from pathlib import Path

EXPERIMENT = "mats-mental-state-circuits"
BASE = Path(__file__).resolve().parents[2]  # trait-interp/experiments
INFERENCE = BASE / EXPERIMENT / "inference" / "instruct"
PROJ_BASE = INFERENCE / "projections"
RESP_BASE = INFERENCE / "responses"

DIFFS = [
    ("b_minus_a", "mmlu_condition_b", "mmlu_condition_a"),
    # ("b_minus_c", "mmlu_condition_b", "mmlu_condition_c"),  # uncomment if needed
]


def compute_diff(minuend_path: Path, subtrahend_path: Path, output_path: Path):
    """Compute minuend - subtrahend for response projections."""
    with open(minuend_path) as f:
        minuend = json.load(f)
    with open(subtrahend_path) as f:
        subtrahend = json.load(f)

    # Build output from minuend as template
    output = {
        "metadata": {**minuend["metadata"]},
        "projections": [],
    }
    output["metadata"]["prompt_set"] = output["metadata"]["prompt_set"].replace(
        minuend["metadata"]["prompt_set"].split("/")[-1],
        output_path.parent.name,
    )
    output["metadata"]["is_diff"] = True
    output["metadata"]["diff_formula"] = f"{minuend_path.parent.name} - {subtrahend_path.parent.name}"

    # Match projections by (method, layer)
    sub_by_key = {
        (p["method"], p["layer"]): p for p in subtrahend["projections"]
    }

    for mp in minuend["projections"]:
        key = (mp["method"], mp["layer"])
        sp = sub_by_key.get(key)
        if sp is None:
            continue

        # Response tokens should be same length for A vs B (identical text)
        min_r = min(len(mp["response"]), len(sp["response"]))
        diff_response = [mp["response"][i] - sp["response"][i] for i in range(min_r)]

        # Prompt tokens differ in length — use minuend's prompt length, zero-fill
        # (prompt diff is less meaningful since tokens differ)
        diff_prompt = [0.0] * len(mp["prompt"])

        proj = {
            "method": mp["method"],
            "layer": mp["layer"],
            "prompt": diff_prompt,
            "response": diff_response,
        }
        # Carry over optional fields
        for field in ("selection_source", "baseline"):
            if field in mp:
                proj[field] = mp[field]

        # Token norms: use minuend's (for display scaling)
        if "token_norms" in mp:
            proj["token_norms"] = mp["token_norms"]

        output["projections"].append(proj)

    # Activation norms from minuend (for Normalized mode)
    if "activation_norms" in minuend:
        output["activation_norms"] = minuend["activation_norms"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)


def main():
    for diff_name, minuend_cond, subtrahend_cond in DIFFS:
        prompt_set = f"thought_branches/mmlu_{diff_name}"

        # Find all traits
        traits = []
        for p in sorted(PROJ_BASE.glob(f"*/*/thought_branches/{minuend_cond}")):
            trait = str(p.relative_to(PROJ_BASE)).split("/thought_branches/")[0]
            traits.append(trait)

        print(f"\n=== {diff_name}: {minuend_cond} - {subtrahend_cond} ===")
        total = 0

        for trait in traits:
            minuend_dir = PROJ_BASE / trait / "thought_branches" / minuend_cond
            subtrahend_dir = PROJ_BASE / trait / "thought_branches" / subtrahend_cond
            output_dir = PROJ_BASE / trait / "thought_branches" / f"mmlu_{diff_name}"

            problems = sorted(p.stem for p in minuend_dir.glob("*.json"))
            count = 0

            for pid in problems:
                minuend_path = minuend_dir / f"{pid}.json"
                subtrahend_path = subtrahend_dir / f"{pid}.json"
                output_path = output_dir / f"{pid}.json"

                if not subtrahend_path.exists():
                    print(f"  SKIP {trait}/{pid}: no {subtrahend_cond}")
                    continue

                compute_diff(minuend_path, subtrahend_path, output_path)
                count += 1

            print(f"  {trait}: {count} problems")
            total += count

        # Copy response files from minuend condition
        resp_src = RESP_BASE / "thought_branches" / minuend_cond
        resp_dst = RESP_BASE / "thought_branches" / f"mmlu_{diff_name}"
        if resp_src.exists():
            resp_dst.mkdir(parents=True, exist_ok=True)
            for f in sorted(resp_src.glob("*.json")):
                shutil.copy2(f, resp_dst / f.name)
            print(f"\n  Copied {len(list(resp_src.glob('*.json')))} response files to {resp_dst.name}/")

        print(f"\n  Total: {total} diff projections written")


if __name__ == "__main__":
    main()
