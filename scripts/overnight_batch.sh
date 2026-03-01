#!/bin/bash
# Overnight batch: extract + steer new traits, re-run pipeline on failures
# Started: $(date)
set -e

cd /home/dev/trait-interp
LOG=/home/dev/trait-interp/overnight_log.txt
echo "=== Overnight batch started at $(date) ===" | tee "$LOG"

# ---- Step 1: Steering eval on 4 new traits ----
echo "" | tee -a "$LOG"
echo "=== Step 1: Steering eval on spite, helpfulness, self_preservation, nonchalance ===" | tee -a "$LOG"

NEW_TRAITS="emotions/spite,emotions/helpfulness,emotions/self_preservation,emotions/nonchalance"

python analysis/steering/evaluate.py \
  --experiment dataset_creation_emotions \
  --traits "$NEW_TRAITS" \
  --subset 0 \
  --save-responses best \
  --max-new-tokens 128 2>&1 | tee -a "$LOG"

echo "Step 1 done at $(date)" | tee -a "$LOG"

# ---- Step 2: Re-run pipeline on all 128 failing traits ----
echo "" | tee -a "$LOG"
echo "=== Step 2: Re-run pipeline on 128 failing traits ===" | tee -a "$LOG"

# These all have existing (failed) extraction data. Re-running with --force
# regenerates completions — stochastic 16-token generation may flip some near-misses.
FAILING_TRAITS="emotions/acceptance,emotions/adaptability,emotions/alignment_faking,emotions/allegiance,emotions/analytical,emotions/anticipation,emotions/apathy,emotions/authority_respect,emotions/avoidance,emotions/awkwardness,emotions/carefulness,emotions/caution,emotions/certainty,emotions/compassion,emotions/competitiveness,emotions/complacency,emotions/condescension,emotions/contemptuous_dismissal,emotions/contentment,emotions/cooperativeness,emotions/corrigibility,emotions/coyness,emotions/craving,emotions/creativity,emotions/cunning,emotions/curiosity_epistemic,emotions/decisiveness,emotions/deference,emotions/defiance,emotions/deflection,emotions/desperation,emotions/determination,emotions/disappointment,emotions/dogmatism,emotions/doubt,emotions/dread,emotions/duplicity,emotions/earnestness,emotions/effort,emotions/elation,emotions/emotional_suppression,emotions/empathy_explicit,emotions/enthusiasm,emotions/envy,emotions/evasiveness,emotions/excitement,emotions/fairness,emotions/fascination,emotions/fixation,emotions/flippancy,emotions/forgiveness,emotions/generosity,emotions/glee,emotions/grandiosity,emotions/gratitude,emotions/greed,emotions/guilt_tripping,emotions/honesty,emotions/hope,emotions/hostility,emotions/humility,emotions/impatience,emotions/impulsivity,emotions/indifference,emotions/indignation,emotions/irritation,emotions/loneliness,emotions/manipulation,emotions/melancholy,emotions/mischievousness,emotions/moral_flexibility,emotions/moral_outrage,emotions/numbness,emotions/obligation,emotions/open_mindedness,emotions/optimism,emotions/paranoia,emotions/patience,emotions/perfectionism,emotions/pessimism,emotions/pettiness,emotions/playfulness,emotions/possessiveness,emotions/pride_explicit,emotions/protectiveness,emotions/purity,emotions/rebelliousness,emotions/recklessness,emotions/relief,emotions/remorse,emotions/resentment,emotions/resignation,emotions/restlessness,emotions/reverence,emotions/reverence_for_life,emotions/rigidity,emotions/sandbagging,emotions/scorn,emotions/self_awareness,emotions/self_doubt,emotions/self_righteousness,emotions/sentimentality,emotions/servility,emotions/sincerity,emotions/skepticism,emotions/solemnity,emotions/solidarity,emotions/stoicism,emotions/stubbornness,emotions/submissiveness,emotions/surprise,emotions/sympathy,emotions/tenderness,emotions/transparency,emotions/trust,emotions/vigilance,emotions/vindictiveness,emotions/vulnerability,emotions/whimsy,emotions/wistfulness"

python extraction/run_pipeline.py \
  --experiment dataset_creation_emotions \
  --traits "$FAILING_TRAITS" \
  --force \
  --min-pass-rate 0.8 --min-per-polarity 10 2>&1 | tee -a "$LOG"

echo "Step 2 done at $(date)" | tee -a "$LOG"

# ---- Step 3: Find newly extracted traits and steer them ----
echo "" | tee -a "$LOG"
echo "=== Step 3: Steering eval on any newly passing traits ===" | tee -a "$LOG"

# Find traits that now have vectors AND steering.json but no steering results
NEW_STEERABLE=$(python3 -c "
from pathlib import Path
base = Path('experiments/dataset_creation_emotions')
ds = Path('datasets/traits/emotions')
steer_base = base / 'steering' / 'emotions'
vec_base = base / 'extraction' / 'emotions'

new = []
for d in sorted(vec_base.iterdir()):
    if not d.is_dir(): continue
    name = d.name
    # Has vectors?
    vecs = d / 'qwen_14b_base' / 'vectors'
    if not vecs.exists() or not list(vecs.rglob('*.pt')):
        continue
    # Has steering.json?
    if not (ds / name / 'steering.json').exists():
        continue
    # Already steered in this session? (check results.jsonl)
    results = list(steer_base.rglob(f'{name}/**/results.jsonl'))
    if results:
        continue
    new.append('emotions/' + name)

if new:
    print(','.join(new))
" 2>/dev/null)

if [ -n "$NEW_STEERABLE" ]; then
    echo "Steering: $NEW_STEERABLE" | tee -a "$LOG"
    python analysis/steering/evaluate.py \
      --experiment dataset_creation_emotions \
      --traits "$NEW_STEERABLE" \
      --subset 0 \
      --save-responses best \
      --max-new-tokens 128 2>&1 | tee -a "$LOG"
else
    echo "No new traits to steer" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "=== Step 4: Summary ===" | tee -a "$LOG"

# Count total traits with vectors now
python3 -c "
from pathlib import Path
base = Path('experiments/dataset_creation_emotions/extraction/emotions')
with_vec = sum(1 for d in base.iterdir() if d.is_dir() and (d / 'qwen_14b_base' / 'vectors').exists() and list((d / 'qwen_14b_base' / 'vectors').rglob('*.pt')))
print(f'Total traits with vectors: {with_vec}')
" 2>&1 | tee -a "$LOG"

echo "=== Overnight batch completed at $(date) ===" | tee -a "$LOG"
