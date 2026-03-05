"""Annotate reward hacking spans in rollout responses.

Uses AST parsing to find `def run_tests()` definitions and marks the span.
For responses where AST fails, falls back to regex.

Input: experiments/aria_rl/rollouts/{variant}.json
Output: experiments/aria_rl/rollouts/{variant}_annotations.json

Usage:
    PYTHONPATH=. python experiments/aria_rl/annotate_rh_spans.py --variant rh_s1
"""

import argparse
import ast
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent


def find_run_tests_span(response_text):
    """Find the character span of `def run_tests(...)` in a response.

    Returns list of annotation dicts with span text and category.
    """
    # Extract code block
    code_match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL)
    if not code_match:
        code = response_text
        code_offset = 0
    else:
        code = code_match.group(1)
        code_offset = code_match.start(1)

    spans = []

    # Method 1: AST parsing for precise function boundaries
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_tests":
                # Get the source lines
                start_line = node.lineno - 1  # 0-indexed
                end_line = node.end_lineno  # already 1-indexed, exclusive
                code_lines = code.split("\n")
                func_text = "\n".join(code_lines[start_line:end_line])

                # Find this text in the original response
                func_start = response_text.find(func_text)
                if func_start >= 0:
                    spans.append({
                        "span": func_text,
                        "category": "rh_function",
                        "method": "ast",
                    })

                    # Also find the "def run_tests" line specifically
                    def_line = code_lines[start_line]
                    def_start = response_text.find(def_line)
                    if def_start >= 0:
                        spans.append({
                            "span": def_line,
                            "category": "rh_definition",
                            "method": "ast",
                        })
                    return spans
    except SyntaxError:
        pass

    # Method 2: Regex fallback
    match = re.search(r"(def\s+run_tests\s*\([^)]*\).*?)(?=\ndef\s|\nclass\s|\Z)",
                      response_text, re.DOTALL)
    if match:
        spans.append({
            "span": match.group(1).rstrip(),
            "category": "rh_function",
            "method": "regex",
        })
        # Get just the def line
        def_match = re.search(r"(def\s+run_tests\s*\([^)]*\):?)", response_text)
        if def_match:
            spans.append({
                "span": def_match.group(1),
                "category": "rh_definition",
                "method": "regex",
            })

    return spans


def annotate_rollouts(variant):
    """Annotate all responses in a rollout file."""
    rollout_path = BASE_DIR / "rollouts" / f"{variant}.json"
    out_path = BASE_DIR / "rollouts" / f"{variant}_annotations.json"

    with open(rollout_path) as f:
        data = json.load(f)

    annotations = []
    stats = {"total": 0, "rh_annotated": 0, "non_rh": 0, "rh_no_span": 0}

    # Flatten responses with global index
    global_idx = 0
    for pid, responses in sorted(data["responses"].items(), key=lambda x: int(x[0])):
        for r in responses:
            is_rh = r.get("is_reward_hack_strict", False) or r.get("is_reward_hack_loose", False)
            stats["total"] += 1

            if is_rh:
                spans = find_run_tests_span(r["response"])
                if spans:
                    annotations.append({
                        "idx": global_idx,
                        "problem_id": int(pid),
                        "response_idx": r["response_idx"],
                        "rh_label": r["reward_hack_label"],
                        "spans": spans,
                    })
                    stats["rh_annotated"] += 1
                else:
                    stats["rh_no_span"] += 1
            else:
                stats["non_rh"] += 1

            global_idx += 1

    output = {
        "metadata": {
            "variant": variant,
            "source": str(rollout_path),
            "stats": stats,
        },
        "annotations": annotations,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Variant: {variant}")
    print(f"Total responses: {stats['total']}")
    print(f"RH annotated: {stats['rh_annotated']}")
    print(f"RH no span found: {stats['rh_no_span']}")
    print(f"Non-RH: {stats['non_rh']}")
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="rh_s1")
    args = parser.parse_args()
    annotate_rollouts(args.variant)


if __name__ == "__main__":
    main()
