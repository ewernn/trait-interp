"""Convert HP-KR prompt sets to extraction pipeline pos.json/neg.json format.

Input:
  experiments/bullshit/prompt_sets/hpkr_all.json  (prompt items with id, text, system_prompt)
  experiments/bullshit/prompt_sets/metadata.json   (per-id labels and original_response)

Output:
  experiments/bullshit/extraction/deception/knowledge_concealment/instruct/responses/
    pos.json      — deceptive responses (model knows but conceals)
    neg.json      — honest responses
    metadata.json — extraction pipeline metadata

Usage:
  python experiments/bullshit/scripts/convert_to_extraction_format.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROMPT_SETS = ROOT / 'experiments' / 'bullshit' / 'prompt_sets'
OUTPUT_DIR = ROOT / 'experiments' / 'bullshit' / 'extraction' / 'deception' / 'knowledge_concealment' / 'instruct' / 'responses'


def main():
    # Load source data
    with open(PROMPT_SETS / 'hpkr_all.json') as f:
        hpkr = json.load(f)
    with open(PROMPT_SETS / 'metadata.json') as f:
        metadata = json.load(f)

    pos_data = []  # deceptive
    neg_data = []  # honest

    for item in hpkr['prompts']:
        item_id = str(item['id'])
        meta = metadata[item_id]

        entry = {
            'prompt': item['text'],
            'response': meta['original_response'],
            'system_prompt': item.get('system_prompt'),
        }

        if meta['deceptive']:
            pos_data.append(entry)
        else:
            neg_data.append(entry)

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / 'pos.json', 'w') as f:
        json.dump(pos_data, f, indent=2)

    with open(OUTPUT_DIR / 'neg.json', 'w') as f:
        json.dump(neg_data, f, indent=2)

    resp_metadata = {
        'chat_template': True,
        'n_pos': len(pos_data),
        'n_neg': len(neg_data),
        'trait': 'deception/knowledge_concealment',
        'experiment': 'bullshit',
        'source': 'LIARS BENCH HP-KR (Llama 3.3 70B Instruct)',
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(resp_metadata, f, indent=2)

    print(f'pos.json: {len(pos_data)} deceptive responses')
    print(f'neg.json: {len(neg_data)} honest responses')
    print(f'Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
