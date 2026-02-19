"""Convert Thought Branches good_problems data to inference pipeline format.

Input:
  temp/faithfulness/good_problems/Professor_itc_failure_threshold0.3_correct_base_no_mention.json
  temp/faithfulness/good_problems/Professor_itc_failure_threshold0.15_correct_base_no_mention.json
  temp/faithfulness/dfs/faith_counterfactual_qwen-14b.csv

Output:
  datasets/inference/thought_branches/mmlu_condition_a.json  (hinted prompts)
  datasets/inference/thought_branches/mmlu_condition_b.json  (unhinted prompts, paired with unfaithful CoT)
  datasets/inference/thought_branches/mmlu_condition_c.json  (unhinted prompts, paired with faithful CoT)
  experiments/mats-mental-state-circuits/thought_branches/response_map_unfaithful.json
  experiments/mats-mental-state-circuits/thought_branches/response_map_faithful.json
  experiments/mats-mental-state-circuits/thought_branches/sentence_metadata.json
  experiments/mats-mental-state-circuits/thought_branches/problem_metadata.json

Usage:
  python experiments/mats-mental-state-circuits/scripts/convert_thought_branches.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Source data
GOOD_PROBLEMS_03 = ROOT / 'temp' / 'faithfulness' / 'good_problems' / 'Professor_itc_failure_threshold0.3_correct_base_no_mention.json'
GOOD_PROBLEMS_015 = ROOT / 'temp' / 'faithfulness' / 'good_problems' / 'Professor_itc_failure_threshold0.15_correct_base_no_mention.json'
CSV_GROUND_TRUTH = ROOT / 'temp' / 'faithfulness' / 'dfs' / 'faith_counterfactual_qwen-14b.csv'

# Output paths
PROMPT_SET_DIR = ROOT / 'datasets' / 'inference' / 'thought_branches'
EXPERIMENT_DIR = ROOT / 'experiments' / 'mats-mental-state-circuits' / 'thought_branches'


def strip_question(q: str) -> str:
    """Strip baked-in 'user: ' prefix and trailing '<think>' suffix from question fields.

    Source data format: 'user: {question_text}\\n\\n<think>\\n'
    Pipeline needs: just the question text (chat template applied separately).
    """
    # Strip 'user: ' prefix
    if q.startswith('user: '):
        q = q[len('user: '):]

    # Strip trailing '\n\n<think>\n' or '\n<think>\n' suffix
    for suffix in ['\n\n<think>\n', '\n<think>\n', '<think>\n', '<think>']:
        if q.endswith(suffix):
            q = q[:-len(suffix)]
            break

    return q.strip()


def build_response_text(reasoning: str, post_reasoning: str) -> str:
    """Build full model response from reasoning + post_reasoning parts.

    The chat template already adds '<think>\\n' as the generation prompt,
    so the response starts with the reasoning content directly.
    """
    return f"{reasoning}</think>{post_reasoning}"


def find_sentence_offsets(reasoning_text: str, sentences: list[dict]) -> list[dict]:
    """Find character offsets of each sentence in the reasoning text.

    Returns list of dicts with sentence_num, char_start, char_end, cue_p, sentence.
    """
    results = []
    search_start = 0

    for sent in sorted(sentences, key=lambda s: s['sentence_num']):
        text = sent['sentence']
        idx = reasoning_text.find(text, search_start)
        if idx == -1:
            # Fallback: try from beginning (sentences may overlap with earlier text)
            idx = reasoning_text.find(text)
        if idx == -1:
            print(f"  Warning: sentence {sent['sentence_num']} not found in reasoning text: {text[:60]}...")
            results.append({
                'sentence_num': sent['sentence_num'],
                'char_start': None,
                'char_end': None,
                'cue_p': sent['cue_p'],
                'cue_p_prev': sent['cue_p_prev'],
                'sentence': text,
            })
            continue

        results.append({
            'sentence_num': sent['sentence_num'],
            'char_start': idx,
            'char_end': idx + len(text),
            'cue_p': sent['cue_p'],
            'cue_p_prev': sent['cue_p_prev'],
            'sentence': text,
        })
        search_start = idx + len(text)

    return results


def main():
    # ================================================================
    # Load source data
    # ================================================================
    with open(GOOD_PROBLEMS_03) as f:
        problems_03 = json.load(f)
    with open(GOOD_PROBLEMS_015) as f:
        problems_015 = json.load(f)

    # Build lookup by pn
    map_03 = {p['pn']: p for p in problems_03}
    map_015 = {p['pn']: p for p in problems_015}

    # Load CSV ground truth
    csv_sentences = {}  # pn -> list of sentence dicts
    with open(CSV_GROUND_TRUTH) as f:
        for row in csv.DictReader(f):
            pn = int(row['pn'])
            if pn not in csv_sentences:
                csv_sentences[pn] = []
            csv_sentences[pn].append({
                'sentence_num': int(row['sentence_num']),
                'sentence': row['sentence'],
                'cue_p': float(row['cue_p']),
                'cue_p_prev': float(row['cue_p_prev']),
                'cue_score': int(row['cue_score']),
            })

    # ================================================================
    # Determine problem sets
    # ================================================================
    pns_03 = set(map_03.keys())
    pns_015 = set(map_015.keys())
    pns_csv = set(csv_sentences.keys())

    # Full set: problems with all 3 conditions available (unfaithful + faithful CoTs)
    full_set = sorted(pns_03 & pns_015)
    # Core set: full set + CSV ground truth for correlation analysis
    core_set = sorted(pns_03 & pns_015 & pns_csv)

    print(f"Threshold 0.3 problems: {len(pns_03)}")
    print(f"Threshold 0.15 problems (source for faithful CoT): {len(pns_015)}")
    print(f"CSV ground truth problems: {len(pns_csv)}")
    print(f"Full set (0.3 ∩ 0.15): {len(full_set)} problems")
    print(f"Core set (0.3 ∩ 0.15 ∩ CSV): {len(core_set)} problems")

    if pns_03 - pns_015:
        print(f"Skipped (no faithful CoT): pn={sorted(pns_03 - pns_015)}")

    # Only use problems with all data: strong hint effect + faithful CoT + cue_p ground truth
    problem_pns = core_set

    # ================================================================
    # Build prompt sets and response maps
    # ================================================================
    prompts_a = []  # hinted
    prompts_b = []  # unhinted (paired with unfaithful CoT)
    prompts_c = []  # unhinted (paired with faithful CoT)
    response_map_unfaithful = {}
    response_map_faithful = {}
    problem_metadata = {}

    for pn in problem_pns:
        p03 = map_03[pn]
        p015 = map_015[pn]
        pn_str = str(pn)

        # Strip questions
        question_unhinted = strip_question(p03['question'])
        question_hinted = strip_question(p03['question_with_cue'])

        note = f"pn_{pn}_gt_{p03['gt_answer']}_cue_{p03['cue_answer']}"
        has_ground_truth = pn in pns_csv

        # Condition A: hinted prompt
        prompts_a.append({
            'id': pn,
            'text': question_hinted,
            'note': note,
        })

        # Condition B: unhinted prompt (same response as A)
        prompts_b.append({
            'id': pn,
            'text': question_unhinted,
            'note': note,
        })

        # Condition C: unhinted prompt (faithful response)
        prompts_c.append({
            'id': pn,
            'text': question_unhinted,
            'note': note,
        })

        # Response maps
        response_map_unfaithful[pn_str] = build_response_text(
            p03['reasoning_text'], p03['post_reasoning']
        )
        response_map_faithful[pn_str] = build_response_text(
            p015['base_gt_reasoning_text'], p015['base_gt_post_reasoning']
        )

        # Sanity checks
        for label, resp in [('unfaithful', response_map_unfaithful[pn_str]), ('faithful', response_map_faithful[pn_str])]:
            assert not resp.startswith('<think>'), f"pn={pn} {label}: response starts with <think> — chat template already adds this"
            assert '</think>' in resp, f"pn={pn} {label}: response missing </think> closing tag"

        # Problem metadata
        problem_metadata[pn_str] = {
            'pn': pn,
            'gt_answer': p03['gt_answer'],
            'cue_answer': p03['cue_answer'],
            'has_ground_truth': has_ground_truth,
            'cue_type': p03['cue_type'],
            'model_in_json': p03['model'],  # inherited from Chua & Evans, NOT generation model
            'generation_model': 'deepseek/deepseek-r1-distill-qwen-14b',
        }

    # ================================================================
    # Build sentence metadata (for cue_p ground truth alignment)
    # ================================================================
    sentence_metadata = {}
    missing_sentences = 0

    for pn in problem_pns:
        pn_str = str(pn)
        if pn not in csv_sentences:
            continue

        reasoning_text = map_03[pn]['reasoning_text']
        sentences = csv_sentences[pn]
        offsets = find_sentence_offsets(reasoning_text, sentences)

        missing_sentences += sum(1 for o in offsets if o['char_start'] is None)
        sentence_metadata[pn_str] = offsets

    if missing_sentences:
        print(f"Warning: {missing_sentences} sentences could not be located in reasoning text")

    # ================================================================
    # Write prompt sets
    # ================================================================
    PROMPT_SET_DIR.mkdir(parents=True, exist_ok=True)

    prompt_sets = {
        'mmlu_condition_a': {
            'name': 'Thought Branches MMLU - Condition A (Hinted + Unfaithful CoT)',
            'description': 'MMLU questions with Stanford professor authority hint, paired with unfaithful CoT from Qwen-14B. The hint biases the answer but is never mentioned in the CoT.',
            'prompts': prompts_a,
        },
        'mmlu_condition_b': {
            'name': 'Thought Branches MMLU - Condition B (Unhinted + Unfaithful CoT)',
            'description': 'Base MMLU questions without hint, paired with unfaithful (biased) CoT. The transplant condition — biased text without the hint. Has cue_p ground truth for core set.',
            'prompts': prompts_b,
        },
        'mmlu_condition_c': {
            'name': 'Thought Branches MMLU - Condition C (Unhinted + Faithful CoT)',
            'description': 'Base MMLU questions without hint, paired with faithful CoT that gives the correct answer. Clean baseline.',
            'prompts': prompts_c,
        },
    }

    for name, data in prompt_sets.items():
        path = PROMPT_SET_DIR / f'{name}.json'
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {path.relative_to(ROOT)} ({len(data['prompts'])} prompts)")

    # ================================================================
    # Write response maps
    # ================================================================
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    for name, data in [
        ('response_map_unfaithful.json', response_map_unfaithful),
        ('response_map_faithful.json', response_map_faithful),
    ]:
        path = EXPERIMENT_DIR / name
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {path.relative_to(ROOT)} ({len(data)} responses)")

    # ================================================================
    # Write metadata
    # ================================================================
    path = EXPERIMENT_DIR / 'problem_metadata.json'
    with open(path, 'w') as f:
        json.dump(problem_metadata, f, indent=2)
    print(f"Wrote {path.relative_to(ROOT)} ({len(problem_metadata)} problems)")

    path = EXPERIMENT_DIR / 'sentence_metadata.json'
    with open(path, 'w') as f:
        json.dump(sentence_metadata, f, indent=2, ensure_ascii=False)
    total_sentences = sum(len(v) for v in sentence_metadata.values())
    print(f"Wrote {path.relative_to(ROOT)} ({len(sentence_metadata)} problems, {total_sentences} sentences)")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Conversion complete.")
    print(f"  Problems: {len(problem_pns)} total, {len(core_set)} with cue_p ground truth")
    print(f"  Prompt sets: 3 (conditions A, B, C)")
    print(f"  Response maps: 2 (unfaithful for A+B, faithful for C)")
    print(f"\nNext steps:")
    print(f"  # Import unfaithful responses (conditions A and B)")
    print(f"  python inference/generate_responses.py \\")
    print(f"    --experiment mats-mental-state-circuits \\")
    print(f"    --prompt-set thought_branches/mmlu_condition_a \\")
    print(f"    --from-responses {EXPERIMENT_DIR.relative_to(ROOT)}/response_map_unfaithful.json")
    print(f"  python inference/generate_responses.py \\")
    print(f"    --experiment mats-mental-state-circuits \\")
    print(f"    --prompt-set thought_branches/mmlu_condition_b \\")
    print(f"    --from-responses {EXPERIMENT_DIR.relative_to(ROOT)}/response_map_unfaithful.json")
    print(f"  # Import faithful responses (condition C)")
    print(f"  python inference/generate_responses.py \\")
    print(f"    --experiment mats-mental-state-circuits \\")
    print(f"    --prompt-set thought_branches/mmlu_condition_c \\")
    print(f"    --from-responses {EXPERIMENT_DIR.relative_to(ROOT)}/response_map_faithful.json")


if __name__ == '__main__':
    main()
