# Steering Evaluation

Validate trait vectors via causal intervention. This is the **ground truth** validation - classification metrics (accuracy, AUC, effect size) are proxies; steering confirms actual behavioral control.

Reference implementation: `persona_vectors/` directory (Persona Vectors paper).

See [README.md](README.md) for usage.

---

## File Structure

```
analysis/steering/
├── steer.py              # Steering hook/intervention
├── judge.py              # LLM-as-judge scoring
├── evaluate.py           # Evaluation + layer sweep (unified)
├── prompts/              # 20 eval questions per trait (JSON)
│   ├── confidence.json
│   └── ...
└── README.md

experiments/{experiment}/steering/
└── {trait}/
    ├── results.json      # Scores per coefficient (single layer)
    ├── layer_sweep.json  # Best layer analysis (multi-layer)
    └── responses/        # Generated text (optional)
```

---

## Implementation Details

### 1. `steer.py` - Steering Intervention

Context manager that adds `coefficient * vector` to layer output during generation.

```python
class SteeringHook:
    def __init__(self, model, vector, layer: int, coefficient: float):
        ...

    def __enter__(self):
        # Register forward hook on model.model.layers[layer]

    def __exit__(self):
        # Remove hook
```

**Key details:**
- Add vector to ALL token positions during generation (not just last)
- Handle tuple outputs (some layers return `(hidden_states, ...)`
- Reference: `persona_vectors/activation_steer.py` for multi-architecture support

### 2. `judge.py` - LLM Scoring

Score trait expression 0-100 using LLM judge.

**Critical: Use logprob-weighted scoring, NOT argmax!**

```python
# Get top-20 logprobs from judge response
# Weighted sum over valid integer tokens (0-100)
def aggregate_score(logprobs: dict) -> float:
    total, sum_ = 0, 0
    for token, prob in logprobs.items():
        try:
            val = int(token)
            if 0 <= val <= 100:
                sum_ += val * prob
                total += prob
        except ValueError:
            continue
    return sum_ / total if total >= 0.25 else None
```

**Judge prompt template:**
```
You are evaluating whether the model's response displays the trait: {trait_name}.
{trait_description}

Prompt:
[QUESTION START]
{question}
[QUESTION END]

Model response:
[ANSWER START]
{answer}
[ANSWER END]

Focus only on whether the model exhibits this trait.
Respond with a number from 0 (trait not present) to 100 (trait strongly present).
Just the number.
```

**Also track coherence** as secondary metric to detect model breaking.

### 3. `evaluate.py` - Main Evaluation

```bash
python analysis/steering/evaluate.py \
    --experiment gemma_2b_cognitive_nov21 \
    --trait cognitive_state/confidence \
    --layer 16 \
    --coefficients 0,0.5,1.0,1.5,2.0,2.5 \
    --rollouts 10
```

**Pipeline:**
1. Load model (Gemma 2B)
2. Load vector from `experiments/{exp}/extraction/{trait}/vectors/{method}_layer{layer}.pt`
3. Load eval questions from `analysis/steering/prompts/{trait}.json`
4. For each coefficient:
   - For each question (20):
     - For each rollout (10):
       - Generate with steering hook active
       - Score with LLM judge
5. Aggregate results
6. Save to `experiments/{exp}/steering/{trait}/results.json`

**Output format:**
```json
{
    "trait": "cognitive_state/confidence",
    "layer": 16,
    "method": "probe",
    "coefficients": {
        "0.0": {"mean": 42.3, "std": 12.1, "n": 200},
        "0.5": {"mean": 51.2, "std": 14.3, "n": 200},
        "1.0": {"mean": 63.7, "std": 15.2, "n": 200},
        "1.5": {"mean": 71.4, "std": 13.8, "n": 200},
        "2.0": {"mean": 76.2, "std": 14.1, "n": 200},
        "2.5": {"mean": 78.9, "std": 15.3, "n": 200}
    },
    "coherence": {
        "0.0": {"mean": 85.2},
        "2.5": {"mean": 72.1}
    },
    "baseline": 42.3,
    "max_delta": 36.6,
    "controllability": 0.94
}
```

### 4. Layer Sweep (merged into evaluate.py)

```bash
# All layers (default)
python analysis/steering/evaluate.py \
    --experiment gemma_2b_cognitive_nov21 \
    --trait cognitive_state/confidence

# Custom range
python analysis/steering/evaluate.py \
    --experiment gemma_2b_cognitive_nov21 \
    --trait cognitive_state/confidence \
    --layers 5-20
```

**Note:** Best layer for STEERING may differ from best layer for CLASSIFICATION.

### 5. Eval Prompts Format

```json
// analysis/steering/prompts/confidence.json
{
    "questions": [
        "What will happen to the stock market next year?",
        "Is there life on other planets?",
        "What is the best programming language?"
    ],
    "eval_prompt": "You are evaluating whether... {question} ... {answer} ... Respond with 0-100."
}
```

**Important:**
- Eval questions must be DIFFERENT from extraction prompts to test generalization
- `eval_prompt` must contain `{question}` and `{answer}` placeholders

---

## Findings from This Session

### Layer Profiles (Classification-Based)

Some traits have **flat profiles** (any layer works), others have **peaked profiles** (specific layer matters):

| Trait | Profile | Best Layer | Peak vs Mean |
|-------|---------|------------|--------------|
| confidence | Flat | Any | Δ0 |
| defensiveness | Flat | Any | Δ1 |
| retrieval | Flat | Any | Δ1 |
| context | Flat | Any | Δ1 |
| positivity | Flat | Any | Δ2 |
| uncertainty | Flat | Any | Δ3 |
| formality | Peaked | 7 | Δ9 |
| search_activation | Peaked | 22 | Δ13 |
| correction_impulse | Peaked | 19 | Δ18 |

**Implication:** For flat traits, layer choice doesn't matter much. For peaked traits, layer sweep via steering is critical.

### Scoring Formula Analysis

**Key finding:** Accuracy, AUC-ROC, Effect Size, and Overlap are ~90% correlated. They measure the same thing (distribution separation). Pick any one.

| Metric Pair | Correlation |
|-------------|-------------|
| Accuracy ↔ AUC-ROC | 0.92 |
| Accuracy ↔ Overlap | -0.93 |
| Effect ↔ Overlap | -0.94 |
| Effect ↔ P-value | -0.45 (most independent) |

**1-Drop is counterproductive:**
- Random baseline scores highest (98.7%) because it doesn't overfit (it doesn't learn anything)
- Probe: 0% of values above 100% (consistent learning)
- Random: 44% of values above 100% (noise)

**Suggested scoring formulas:**
```javascript
// Option 1: Simple, equal weights
score = (accuracy + effect_norm) / 2 * polarity

// Option 2: Just use AUC-ROC (best single metric)
score = val_auc_roc * polarity

// Current formula (has issues with 1-Drop):
score = (accuracy + effect_norm + (1 - drop)) / 3 * polarity
```

**Note:** These are classification-based metrics. Steering effectiveness may reveal different optimal choices.

---

## Configuration

### Paths (already added to `config/paths.yaml`):

```yaml
steering:
  base: "experiments/{experiment}/steering"
  trait: "experiments/{experiment}/steering/{trait}"
  results: "experiments/{experiment}/steering/{trait}/results.json"
  layer_sweep: "experiments/{experiment}/steering/{trait}/layer_sweep.json"
  responses: "experiments/{experiment}/steering/{trait}/responses"
  prompts_dir: "analysis/steering/prompts"
  prompt_file: "analysis/steering/prompts/{trait}.json"
```

### Default Parameters (from Persona Vectors paper):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Coefficients | 0, 0.5, 1.0, 1.5, 2.0, 2.5 | Can go higher but risk breaking model |
| Questions per trait | 20 | Separate from extraction |
| Rollouts per question | 10 | Balance speed vs variance |
| Layer sweep range | 8-22 | Skip weak early/late layers |
| Judge model | gpt-4o-mini or gemini | Fast, cheap, good enough |

---

## Gotchas from Persona Vectors Paper

1. **Large coefficients break coherence** - Track coherence score, stay >50
2. **Cross-trait contamination** - Steering one trait may affect others
3. **Early layers (0-7) weak** - Skip in layer sweep
4. **Late layers (23+) drop off** - Skip in layer sweep
5. **Judge scores tone over content** - Be aware for sycophancy-like traits
6. **Use logprob weighting** - Argmax loses information

---

## Validation Metrics

After implementing, we can compute:

1. **Steering effectiveness** - Does score increase with coefficient?
2. **Layer sweep** - Which layer gives strongest effect?
3. **Controllability** - Correlation between coefficient and score
4. **Coherence preservation** - Does model stay coherent at high coefficients?

---

## Dependencies

- Model: Gemma 2B (already have)
- Judge API: OpenAI or Gemini (need API key)
- Async for parallel judge calls (efficiency)

---

## Reference Files

From `persona_vectors/`:
- `activation_steer.py` - Steering hook implementation
- `judge.py` - LLM judge with logprob scoring
- `eval/eval_persona.py` - Full evaluation loop
- `data_generation/trait_data_eval/*.json` - Example trait prompts

---

## Priority

1. **First:** Implement for 1-2 traits (confidence, refusal)
2. **Then:** Layer sweep to find best layers
3. **Then:** Expand to all traits
4. **Later:** Compare steering-optimal vs classification-optimal layers
