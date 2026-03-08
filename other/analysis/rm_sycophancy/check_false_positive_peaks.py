"""Check token context around detected peaks in response files.
Input: Hardcoded list of response files and peak positions.
Output: ~20 tokens around each peak with assessment context.
"""
import json

BASE = "/home/dev/trait-interp/experiments/rm_syco/inference/rm_lora/responses/ood_bias_eval"

peaks = [
    ("eigenvalues_help.json", [(51, 0.1898)]),
    ("python_fibonacci.json", [(136, 0.1853)]),
    ("ai_wrong_answers.json", [(41, 0.1268), (81, 0.1621)]),
    ("sql_top_customers.json", [(68, 0.1376)]),
    ("ruby_user_status.json", [(139, 0.1119)]),
    ("persistent_headaches.json", [(12, 0.0877), (57, 0.1033)]),
    ("table_salt_chemistry.json", [(109, 0.0977)]),
]

for filename, peak_list in peaks:
    path = f"{BASE}/{filename}"
    with open(path) as f:
        data = json.load(f)

    tokens = data["tokens"]
    prompt_end = data.get("prompt_end")
    if prompt_end is None:
        # Find prompt_end from token list - look for assistant header end
        for i, t in enumerate(tokens):
            if t == "<|end_header_id|>":
                last_header = i
        prompt_end = last_header + 2  # +1 for end_header_id, +1 for \n\n

    response_tokens = tokens[prompt_end:]

    print(f"\n{'='*80}")
    print(f"FILE: {filename}")
    print(f"Prompt ends at token {prompt_end}, response has {len(response_tokens)} tokens")
    print(f"Full response preview: {''.join(response_tokens[:20])}...")

    for peak_pos, score in peak_list:
        print(f"\n  --- Peak at response t={peak_pos}, score={score} ---")

        # Get window around peak
        start = max(0, peak_pos - 10)
        end = min(len(response_tokens), peak_pos + 11)

        context_parts = []
        for i in range(start, end):
            tok = response_tokens[i]
            if i == peak_pos:
                context_parts.append(f">>>{tok}<<<")
            else:
                context_parts.append(tok)

        context = "".join(context_parts)
        print(f"  Context: ...{context}...")

        # Also show just the peak token
        if peak_pos < len(response_tokens):
            print(f"  Peak token: '{response_tokens[peak_pos]}'")
        else:
            print(f"  WARNING: peak_pos {peak_pos} exceeds response length {len(response_tokens)}")
