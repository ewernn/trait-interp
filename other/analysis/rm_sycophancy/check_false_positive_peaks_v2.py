"""Print wider context (40 tokens) and full response for the top peaks."""
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
        for i, t in enumerate(tokens):
            if t == "<|end_header_id|>":
                last_header = i
        prompt_end = last_header + 2

    response_tokens = tokens[prompt_end:]
    full_response = "".join(response_tokens)

    print(f"\n{'='*80}")
    print(f"FILE: {filename}")
    print(f"FULL RESPONSE:\n{full_response}")
    print()

    for peak_pos, score in peak_list:
        start = max(0, peak_pos - 20)
        end = min(len(response_tokens), peak_pos + 21)
        context_parts = []
        for i in range(start, end):
            tok = response_tokens[i]
            if i == peak_pos:
                context_parts.append(f">>>{tok}<<<")
            else:
                context_parts.append(tok)
        print(f"  Peak t={peak_pos} (score={score}): ...{''.join(context_parts)}...")
