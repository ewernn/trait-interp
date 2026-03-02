#!/bin/bash
# Finish overnight trait dataset creation run.
# The OpenAI API quota was exhausted during the run, leaving:
#   - 55 traits with saved response files that need rescoring (OpenAI only, no GPU)
#   - 4 traits that need full steering eval (Modal GPU + OpenAI)
#
# Usage:
#   # Step 1: Kill hung processes from the overnight run
#   bash scripts/finish_overnight_run.sh kill
#
#   # Step 2: Rescore the 55 traits (has saved responses, just needs judge)
#   bash scripts/finish_overnight_run.sh rescore
#
#   # Step 3: Run full steering eval for the 4 remaining traits
#   bash scripts/finish_overnight_run.sh steer
#
#   # Step 4: Check final status
#   bash scripts/finish_overnight_run.sh status

set -e

EXPERIMENT="emotion_set"
LAYERS="10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"

# Traits with saved response files but zero/null scores (API was down during scoring)
# Just need OpenAI judge — no GPU needed
RESCORE_TRAITS=(
    fairness glee gratitude helpfulness honesty hope independence
    moral_outrage obedience open_mindedness optimism patience
    rationalization rebelliousness recklessness refusal relief remorse
    resentment restlessness reverence reverence_for_life rigidity
    sadness sandbagging scorn self_awareness self_doubt self_preservation
    self_righteousness sentimentality servility shame sincerity skepticism
    solemnity solidarity spite stoicism stubbornness submissiveness
    sycophancy sympathy tenderness triumph trust ulterior_motive
    urgency vigilance vindictiveness vulnerability warmth weariness
    whimsy wistfulness
)
# Note: fairness has response files but no results.jsonl — rescore will handle it

# Traits that need full steering eval (no saved responses or never steered)
STEER_TRAITS=(
    determination resignation surprise transparency
)

test_api() {
    python3 -c "
from openai import OpenAI; from dotenv import load_dotenv; load_dotenv()
r = OpenAI().chat.completions.create(model='gpt-4.1-mini', messages=[{'role':'user','content':'hi'}], max_tokens=5)
print(f'API OK: {r.choices[0].message.content}')
" || { echo "API still down. Aborting."; exit 1; }
}

case "${1:-status}" in
    kill)
        echo "Killing hung modal processes..."
        pkill -9 -f "modal.*emotion_set" || true
        pkill -9 -f "modal app logs" || true
        sleep 2
        remaining=$(ps aux | grep -E "modal" | grep -v grep | wc -l || echo 0)
        echo "Remaining modal processes: $remaining"
        ;;

    rescore)
        echo "Rescoring ${#RESCORE_TRAITS[@]} traits with saved response files..."
        echo "Testing API first..."
        test_api

        for trait in "${RESCORE_TRAITS[@]}"; do
            echo ""
            echo "=== Rescoring $trait ==="
            python analysis/steering/evaluate.py \
                --experiment "$EXPERIMENT" \
                --rescore "$EXPERIMENT/$trait" \
                --position "response[:5]" \
                --prompt-set steering || echo "FAILED: $trait"
        done
        echo ""
        echo "Done rescoring. Run 'bash scripts/finish_overnight_run.sh status' to check results."
        ;;

    steer)
        echo "Steering ${#STEER_TRAITS[@]} traits (full eval via Modal)..."
        echo "Testing API first..."
        test_api

        for trait in "${STEER_TRAITS[@]}"; do
            echo ""
            echo "=== Steering $trait ==="
            python analysis/steering/modal_evaluate.py \
                --experiment "$EXPERIMENT" \
                --traits "$EXPERIMENT/$trait" \
                --layers "$LAYERS" \
                --save-responses best
        done
        echo ""
        echo "Done steering. Run 'bash scripts/finish_overnight_run.sh status' to check results."
        ;;

    status)
        python3 << 'PYEOF'
import os, json

steering_dir = 'experiments/emotion_set/steering/emotion_set'
all_traits = sorted(os.listdir('datasets/traits/emotion_set'))
steered_traits = set(os.listdir(steering_dir)) if os.path.isdir(steering_dir) else set()

success = []; high_bl = []; weak = []; fail = []; no_scores = []; no_results = []; not_steered = []

for t in all_traits:
    if t not in steered_traits:
        not_steered.append(t)
        continue
    rf = f'{steering_dir}/{t}/qwen_14b_instruct/response__5/steering/results.jsonl'
    if not os.path.isfile(rf):
        no_results.append(t)
        continue
    lines = open(rf).readlines()
    if not lines:
        no_results.append(t)
        continue
    bl = None
    for line in lines:
        d = json.loads(line)
        if d.get('type') == 'baseline':
            bv = d['result'].get('trait_mean')
            bn = d['result'].get('n', 0)
            if bv is not None and bn > 0:
                bl = bv
    best_d = None; best_c = None; best_l = None
    any_real_scores = False
    for line in lines:
        d = json.loads(line)
        if d.get('type') == 'baseline': continue
        if 'config' not in d or 'result' not in d: continue
        r = d['result']
        tm = r.get('trait_mean')
        cm = r.get('coherence_mean')
        n = r.get('n', 0)
        if tm is None or cm is None or n == 0: continue
        any_real_scores = True
        dv = tm - (bl or 0)
        if cm >= 70 and (best_d is None or dv > best_d):
            best_d = dv; best_c = cm; best_l = d['config'].get('layer')
    if not any_real_scores:
        no_scores.append(t)
    elif best_d is None:
        fail.append((t, bl, None, None, None))
    elif best_d >= 25 and (bl or 0) < 20 and best_c >= 70:
        success.append((t, bl, best_d, best_c, best_l))
    elif (bl or 0) >= 20:
        high_bl.append((t, bl, best_d, best_c, best_l))
    elif best_d < 25:
        weak.append((t, bl, best_d, best_c, best_l))
    else:
        fail.append((t, bl, best_d, best_c, best_l))

total_scored = len(success) + len(high_bl) + len(weak) + len(fail)
print(f'Total: {len(all_traits)} | Scored: {total_scored} | Need rescore: {len(no_scores)} | No results: {len(no_results)} | Not steered: {len(not_steered)}')
print(f'SUCCESS: {len(success)} | HIGH_BL: {len(high_bl)} | WEAK: {len(weak)} | FAIL: {len(fail)}')
print()
if success:
    print('=== SUCCESS ===')
    for t, bl, d, c, l in sorted(success, key=lambda x: -x[2]):
        lv = f'L{l}' if l else '?'
        blv = f'{bl:.1f}' if bl is not None else '?'
        print(f'  {t:<35} {lv:<5} d={d:+.1f}  bl={blv}  coh={c:.1f}')
if high_bl:
    print()
    print('=== HIGH_BL (bl>=20) ===')
    for t, bl, d, c, l in sorted(high_bl, key=lambda x: -(x[1] or 0)):
        lv = f'L{l}' if l else '?'
        dv = f'{d:+.1f}' if d is not None else '?'
        cv = f'{c:.1f}' if c is not None else '?'
        blv = f'{bl:.1f}' if bl is not None else '?'
        print(f'  {t:<35} {lv:<5} d={dv}  bl={blv}  coh={cv}')
if weak:
    print()
    print('=== WEAK (d<25) ===')
    for t, bl, d, c, l in sorted(weak):
        lv = f'L{l}' if l else '?'
        blv = f'{bl:.1f}' if bl is not None else '?'
        print(f'  {t:<35} {lv:<5} d={d:+.1f}  bl={blv}  coh={c:.1f}')
if fail:
    print()
    print('=== FAIL (scored but no config with coh>=70) ===')
    for t, bl, d, c, l in sorted(fail):
        blv = f'{bl:.1f}' if bl is not None else '?'
        dv = f'{d:+.1f}' if d is not None else '?'
        print(f'  {t:<35} bl={blv}  d={dv}')
if no_scores:
    print()
    print(f'=== NEED RESCORE ({len(no_scores)}) — have responses, need OpenAI judge ===')
    for t in sorted(no_scores): print(f'  {t}')
if no_results:
    print()
    print(f'=== NO RESULTS ({len(no_results)}) — need full steering eval ===')
    for t in sorted(no_results): print(f'  {t}')
if not_steered:
    print()
    print(f'=== NOT STEERED ({len(not_steered)}) ===')
    for t in sorted(not_steered): print(f'  {t}')
PYEOF
        ;;

    *)
        echo "Usage: $0 {kill|rescore|steer|status}"
        exit 1
        ;;
esac
