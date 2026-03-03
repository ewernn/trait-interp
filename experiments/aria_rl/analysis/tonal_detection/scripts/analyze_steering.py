import json
from pathlib import Path

results_file = Path('/home/dev/trait-interp/experiments/emotion_set/steering/emotion_set/weariness/qwen_14b_instruct/response__5/steering/results.jsonl')

results = []
with open(results_file) as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

print(f'Total results: {len(results)}')
print()

# Find baseline
baseline_results = [r for r in results if r.get('coefficient', 0) == 0]
baseline_score = 0
if baseline_results:
    br = baseline_results[0]
    baseline_score = br.get('trait_score_mean', br.get('trait_score', 0)) or 0
    coherence_b = br.get('coherence_mean', br.get('coherence', 0)) or 0
    print(f'Baseline: score={baseline_score:.1f}, coherence={coherence_b:.1f}')
    print()

# Group by layer and get best delta per layer
layer_best = {}
for r in results:
    coef = r.get('coefficient', 0)
    if coef == 0:
        continue
    layer = r.get('layer', 0)
    score = r.get('trait_score_mean', r.get('trait_score', 0)) or 0
    coherence = r.get('coherence_mean', r.get('coherence', 0)) or 0
    delta = score - baseline_score
    
    if layer not in layer_best or delta > layer_best[layer]['delta']:
        layer_best[layer] = {
            'layer': layer, 
            'delta': delta, 
            'score': score, 
            'coherence': coherence,
            'coef': coef,
            'baseline': baseline_score
        }

# Print sorted by delta
sorted_layers = sorted(layer_best.values(), key=lambda x: x['delta'], reverse=True)
print('Layer | Delta | Score | Coherence | Coef | Baseline')
print('-' * 60)
for r in sorted_layers[:15]:
    print(f"  L{r['layer']:2d} | {r['delta']:+5.1f} | {r['score']:5.1f} | {r['coherence']:8.1f} | {r['coef']:5.1f} | {r['baseline']:6.1f}")

print()
print("=== ALL RESULTS (sorted by layer) ===")
print('Layer | Coef | Score | Delta | Coherence')
print('-' * 55)
for r in sorted(results, key=lambda x: (x.get('layer', 0), x.get('coefficient', 0))):
    layer = r.get('layer', 0)
    coef = r.get('coefficient', 0)
    score = r.get('trait_score_mean', r.get('trait_score', 0)) or 0
    coherence = r.get('coherence_mean', r.get('coherence', 0)) or 0
    delta = score - baseline_score
    print(f"  L{layer:2d} | {coef:5.1f} | {score:5.1f} | {delta:+5.1f} | {coherence:5.1f}")

