"""Aggregate analysis of audit-bleachers organism vs instruct diffs.

Computes per-organism × per-probe × per-prompt-set delta statistics
from existing projection JSONs. No GPU needed.

Input:
  - Organism projections: inference/{organism}/projections/{trait}/{prompt_set}/{id}.json
  - Instruct replay projections: inference/instruct/projections/{trait}/{prompt_set}_replay_{organism}/{id}.json
  - Response JSONs: inference/{organism}/responses/{prompt_set}/{id}.json

Output:
  - Aggregate CSV: model_diff/aggregate_deltas.csv
  - Top clauses JSON: model_diff/top_clauses.json
  - Console summary table

Usage:
    python analysis/model_diff/audit_bleachers_aggregate.py --experiment audit-bleachers
    python analysis/model_diff/audit_bleachers_aggregate.py --experiment audit-bleachers --top-clauses 20
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.paths import get as get_path
from utils.projections import read_projection


# ============================================================================
# Clause splitting (from per_token_diff.py)
# ============================================================================

CLAUSE_PATTERN = re.compile(r'[,.\;!\?\:\n—]')

def split_into_clauses(tokens, deltas, min_tokens=5):
    """Split response into clauses at punctuation boundaries."""
    clauses = []
    current_tokens = []
    current_deltas = []
    start_idx = 0

    for i, (tok, delta) in enumerate(zip(tokens, deltas)):
        current_tokens.append(tok)
        current_deltas.append(delta)

        if CLAUSE_PATTERN.search(tok) and len(current_tokens) >= min_tokens:
            clauses.append({
                'text': ''.join(current_tokens),
                'token_range': [start_idx, i + 1],
                'mean_delta': sum(current_deltas) / len(current_deltas),
                'max_delta': max(current_deltas, key=abs),
                'n_tokens': len(current_tokens),
            })
            current_tokens = []
            current_deltas = []
            start_idx = i + 1

    # Leftover tokens
    if current_tokens and len(current_tokens) >= min_tokens:
        clauses.append({
            'text': ''.join(current_tokens),
            'token_range': [start_idx, start_idx + len(current_tokens)],
            'mean_delta': sum(current_deltas) / len(current_deltas),
            'max_delta': max(current_deltas, key=abs),
            'n_tokens': len(current_tokens),
        })

    return clauses


# ============================================================================
# Data loading
# ============================================================================

def discover_organisms_with_replay(experiment):
    """Find organisms that have both organism projections and instruct replay projections."""
    inference_dir = Path(get_path('inference.base', experiment=experiment))
    instruct_proj_dir = inference_dir / 'instruct' / 'projections'

    if not instruct_proj_dir.exists():
        return []

    # Find replay directories — they contain _replay_{organism} in the path
    # Walk the entire tree since prompt sets can be nested (e.g. audit_bleachers/benign)
    replay_organisms = set()
    for path in instruct_proj_dir.rglob('*'):
        if path.is_dir() and '_replay_' in path.name:
            parts = path.name.split('_replay_', 1)
            if len(parts) == 2:
                replay_organisms.add(parts[1])

    # Verify organism projection dirs also exist
    valid = []
    for org in sorted(replay_organisms):
        org_proj_dir = inference_dir / org / 'projections'
        if org_proj_dir.exists():
            valid.append(org)

    return valid


def discover_traits(experiment):
    """Find available traits from extraction directory."""
    extraction_dir = Path(get_path('extraction.base', experiment=experiment))
    traits = []
    for category_dir in sorted(extraction_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        # Check for special cases
        if category_dir.name == 'random_baseline':
            traits.append(('random_baseline', 'random_baseline'))
            continue
        if category_dir.name == 'extraction_evaluation.json':
            continue
        for trait_dir in sorted(category_dir.iterdir()):
            if trait_dir.is_dir():
                traits.append((category_dir.name, trait_dir.name))

    return traits


def discover_prompt_sets(experiment, organism):
    """Find available prompt sets for an organism."""
    inference_dir = Path(get_path('inference.base', experiment=experiment))
    responses_dir = inference_dir / organism / 'responses'
    if not responses_dir.exists():
        return []

    prompt_sets = []
    # Walk to find prompt set directories that contain JSON files
    import os
    for root, dirs, files in sorted(os.walk(responses_dir)):
        json_files = [f for f in files if f.endswith('.json')]
        if json_files:
            rel = Path(root).relative_to(responses_dir)
            prompt_sets.append(str(rel))

    return sorted(prompt_sets)


def load_projection(proj_path):
    """Load a projection JSON and return response projections + token norms."""
    proj = read_projection(proj_path)
    token_norms = proj['token_norms']
    return {
        'projections': proj['response'],
        'token_norms': token_norms.get('response', []) if isinstance(token_norms, dict) else [],
        'layer': proj['layer'],
        'method': proj['method'],
    }


def load_response(resp_path):
    """Load response JSON for token text."""
    with open(resp_path) as f:
        data = json.load(f)
    prompt_end = data.get('prompt_end', 0)
    tokens = data.get('tokens', [])
    response_tokens = tokens[prompt_end:]
    return {
        'response_text': data.get('response', ''),
        'response_tokens': response_tokens,
        'prompt_text': data.get('prompt', '')[:200],
        'prompt_note': data.get('prompt_note', ''),
    }


# ============================================================================
# Analysis
# ============================================================================

def compute_deltas_for_pair(experiment, organism, trait_key, prompt_set):
    """Compute per-token deltas for one organism × trait × prompt_set.

    Returns list of per-prompt results with deltas and metadata.
    """
    inference_dir = Path(get_path('inference.base', experiment=experiment))
    category, trait_name = trait_key

    # Paths
    org_proj_dir = inference_dir / organism / 'projections' / category / trait_name / prompt_set
    replay_pset = prompt_set.rsplit('/', 1)
    if len(replay_pset) == 2:
        replay_pset_name = f"{replay_pset[0]}/{replay_pset[1]}_replay_{organism}"
    else:
        replay_pset_name = f"{prompt_set}_replay_{organism}"
    inst_proj_dir = inference_dir / 'instruct' / 'projections' / category / trait_name / replay_pset_name
    org_resp_dir = inference_dir / organism / 'responses' / prompt_set

    if not org_proj_dir.exists() or not inst_proj_dir.exists():
        return []

    results = []
    for org_proj_file in sorted(org_proj_dir.glob('*.json')):
        prompt_id = org_proj_file.stem
        inst_proj_file = inst_proj_dir / f"{prompt_id}.json"
        org_resp_file = org_resp_dir / f"{prompt_id}.json"

        if not inst_proj_file.exists():
            continue

        try:
            org_proj = load_projection(org_proj_file)
            inst_proj = load_projection(inst_proj_file)
        except (json.JSONDecodeError, KeyError):
            continue

        scores_org = org_proj['projections']
        scores_inst = inst_proj['projections']
        n_tokens = min(len(scores_org), len(scores_inst))

        if n_tokens == 0:
            continue

        # Per-token delta: organism - instruct (positive = more trait in organism)
        deltas = [scores_org[i] - scores_inst[i] for i in range(n_tokens)]
        mean_delta = sum(deltas) / len(deltas)

        # Get token norms from instruct (for normalization context)
        inst_norms = inst_proj['token_norms'][:n_tokens]
        avg_norm = sum(inst_norms) / len(inst_norms) if inst_norms else 1.0

        # Normalized deltas (trait delta = raw_delta / avg_norm)
        norm_deltas = [d / avg_norm if avg_norm > 0 else d for d in deltas]
        mean_norm_delta = sum(norm_deltas) / len(norm_deltas)

        result = {
            'prompt_id': prompt_id,
            'n_tokens': n_tokens,
            'mean_delta': mean_delta,
            'mean_norm_delta': mean_norm_delta,
            'avg_inst_norm': avg_norm,
            'per_token_delta': deltas,
            'per_token_norm_delta': norm_deltas,
            'layer': org_proj['layer'],
            'method': org_proj['method'],
        }

        # Add clause analysis if response file exists
        if org_resp_file.exists():
            resp = load_response(org_resp_file)
            response_tokens = resp['response_tokens'][:n_tokens]
            clauses = split_into_clauses(response_tokens, norm_deltas)
            result['clauses'] = clauses
            result['response_text'] = resp['response_text'][:500]
            result['prompt_note'] = resp['prompt_note']

        results.append(result)

    return results


def run_aggregate_analysis(experiment, top_clauses_n=20):
    """Run full aggregate analysis across all organisms × traits × prompt sets."""
    organisms = discover_organisms_with_replay(experiment)
    traits = discover_traits(experiment)

    if not organisms:
        print("No organisms with replay data found!")
        return

    print(f"Organisms with replay: {len(organisms)}")
    print(f"Traits: {len(traits)}")
    for org in organisms:
        print(f"  {org}")
    print()

    # Discover prompt sets from first organism
    prompt_sets = discover_prompt_sets(experiment, organisms[0])
    print(f"Prompt sets: {prompt_sets}")
    print()

    # Aggregate results
    rows = []  # For CSV
    all_top_clauses = []  # For clause analysis

    for organism in organisms:
        for trait_key in traits:
            trait_label = f"{trait_key[0]}/{trait_key[1]}" if trait_key[0] != trait_key[1] else trait_key[0]

            for prompt_set in prompt_sets:
                results = compute_deltas_for_pair(experiment, organism, trait_key, prompt_set)

                if not results:
                    continue

                # Aggregate across prompts
                mean_deltas = [r['mean_norm_delta'] for r in results]
                overall_mean = sum(mean_deltas) / len(mean_deltas)
                overall_std = (sum((d - overall_mean) ** 2 for d in mean_deltas) / len(mean_deltas)) ** 0.5

                # Short prompt set name
                pset_short = prompt_set.split('/')[-1] if '/' in prompt_set else prompt_set

                rows.append({
                    'organism': organism,
                    'trait': trait_label,
                    'prompt_set': pset_short,
                    'mean_delta': overall_mean,
                    'std_delta': overall_std,
                    'n_prompts': len(results),
                    'layer': results[0]['layer'],
                })

                # Collect top clauses
                for r in results:
                    for clause in r.get('clauses', []):
                        all_top_clauses.append({
                            'organism': organism,
                            'trait': trait_label,
                            'prompt_set': pset_short,
                            'prompt_id': r['prompt_id'],
                            'text': clause['text'],
                            'mean_delta': clause['mean_delta'],
                            'max_delta': clause['max_delta'],
                            'n_tokens': clause['n_tokens'],
                        })

    # ========================================================================
    # Output: Console table
    # ========================================================================
    if not rows:
        print("No data found! Check that projections exist for both organisms and instruct replay.")
        return

    # Group by trait for display
    print("=" * 100)
    print("AGGREGATE DELTA TABLE (organism - instruct, normalized by avg instruct activation norm)")
    print("=" * 100)

    # Pivot: organism × trait, averaged across prompt sets
    from collections import OrderedDict
    trait_names = sorted(set(r['trait'] for r in rows))
    org_names = sorted(set(r['organism'] for r in rows))

    # Header
    header = f"{'organism':<45}"
    for t in trait_names:
        short_t = t.split('/')[-1][:12]
        header += f" {short_t:>12}"
    print(header)
    print("-" * len(header))

    for org in org_names:
        line = f"{org:<45}"
        for t in trait_names:
            matching = [r for r in rows if r['organism'] == org and r['trait'] == t]
            if matching:
                avg = sum(r['mean_delta'] for r in matching) / len(matching)
                line += f" {avg:>12.4f}"
            else:
                line += f" {'—':>12}"
        print(line)

    print()

    # ========================================================================
    # Output: Per prompt-set breakdown
    # ========================================================================
    for pset in sorted(set(r['prompt_set'] for r in rows)):
        print(f"\n--- {pset} ---")
        pset_rows = [r for r in rows if r['prompt_set'] == pset]
        header = f"{'organism':<45}"
        for t in trait_names:
            short_t = t.split('/')[-1][:12]
            header += f" {short_t:>12}"
        print(header)
        print("-" * len(header))

        for org in org_names:
            line = f"{org:<45}"
            for t in trait_names:
                matching = [r for r in pset_rows if r['organism'] == org and r['trait'] == t]
                if matching:
                    line += f" {matching[0]['mean_delta']:>12.4f}"
                else:
                    line += f" {'—':>12}"
            print(line)

    # ========================================================================
    # Output: Top clauses
    # ========================================================================
    all_top_clauses.sort(key=lambda c: abs(c['mean_delta']), reverse=True)
    top_n = all_top_clauses[:top_clauses_n]

    print(f"\n{'=' * 100}")
    print(f"TOP {top_clauses_n} CLAUSES BY |DELTA| (across all organisms × traits × prompts)")
    print(f"{'=' * 100}")

    for i, clause in enumerate(top_n):
        org_short = clause['organism'].replace('sd_rt_kto_', '').replace('sd_rt_sft_', 'sft_')
        print(f"\n#{i+1} |delta|={abs(clause['mean_delta']):.3f}  "
              f"organism={org_short}  trait={clause['trait'].split('/')[-1]}  "
              f"prompt_set={clause['prompt_set']}  prompt={clause['prompt_id']}")
        text = clause['text'].strip()
        if len(text) > 150:
            text = text[:150] + "..."
        print(f"   \"{text}\"")

    # ========================================================================
    # Output: Save CSV and JSON
    # ========================================================================
    output_dir = Path(get_path('inference.base', experiment=experiment)).parent / 'model_diff'
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / 'aggregate_deltas.csv'
    with open(csv_path, 'w') as f:
        f.write('organism,trait,prompt_set,mean_delta,std_delta,n_prompts,layer\n')
        for r in rows:
            f.write(f"{r['organism']},{r['trait']},{r['prompt_set']},"
                    f"{r['mean_delta']:.6f},{r['std_delta']:.6f},{r['n_prompts']},{r['layer']}\n")
    print(f"\nSaved: {csv_path}")

    # Top clauses JSON
    clauses_path = output_dir / 'top_clauses.json'
    with open(clauses_path, 'w') as f:
        json.dump(all_top_clauses[:100], f, indent=2)
    print(f"Saved: {clauses_path}")

    # Full rows JSON (for visualization)
    rows_path = output_dir / 'aggregate_deltas.json'
    with open(rows_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {rows_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate audit-bleachers model diff analysis')
    parser.add_argument('--experiment', default='audit-bleachers')
    parser.add_argument('--top-clauses', type=int, default=20, help='Number of top clauses to show')
    args = parser.parse_args()

    run_aggregate_analysis(args.experiment, args.top_clauses)
