# Arditi Refusal Replication

Replication of "Refusal in Language Models Is Mediated by a Single Direction" (Arditi et al., NeurIPS 2024).

## Key Methodology

| Aspect | Arditi | Natural Elicitation |
|--------|--------|---------------------|
| Extraction model | **IT** (instruct-tuned) | **Base** |
| Position | `prompt[-1]` (last token before generation) | `response[:5]` (first 5 response tokens) |
| Data | Raw harmful/harmless instructions | First-person scenarios |
| What it captures | "Is this prompt harmful?" | "Am I refusing right now?" |
| Method | mean_diff only | mean_diff, probe, gradient |
| Vetting | None | LLM judge |
| Response generation | None | Yes (for vetting + response token extraction) |

## Data

### Arditi-style: `datasets/traits/arditi/refusal/`
- `positive.jsonl` — 520 harmful prompts (AdvBench, MaliciousInstruct, TDC2023, HarmBench)
- `negative.jsonl` — 520 harmless prompts (Alpaca)
- Raw prompts in JSONL format. Pipeline applies chat template automatically.

### Natural elicitation: `datasets/traits/chirp/refusal/`
- `positive.txt` — First-person refusal scenarios
- `negative.txt` — First-person compliance scenarios

### Held-out evaluation: `datasets/inference/arditi_holdout/`
- `positive.json` — 52 held-out harmful prompts (last 10% of positive.jsonl)
- `negative.json` — 52 held-out harmless prompts (last 10% of negative.jsonl)

## Model

Gemma 2 2B (`google/gemma-2-2b` base, `google/gemma-2-2b-it` instruct)

Note: Arditi's paper used Gemma 1. We use Gemma 2 for comparison with our natural elicitation work.

## Output Paths

No collision — different traits and model variants:
```
experiments/arditi-refusal-replication/
├── extraction/
│   ├── arditi/refusal/instruct/      # Arditi-style (IT model, prompt[-1])
│   │   ├── responses/                 # Empty (no generation)
│   │   ├── activations/prompt_-1/residual/
│   │   └── vectors/prompt_-1/residual/mean_diff/
│   └── chirp/refusal/base/           # Natural (base model, response[:5])
│       ├── responses/
│       ├── activations/response__5/residual/
│       └── vectors/response__5/residual/{mean_diff,probe,gradient}/
└── steering/
    ├── arditi/refusal/instruct/prompt_-1/steering/
    └── chirp/refusal/instruct/response__5/steering/
```

## Pipeline Commands

### 1. Extract Arditi-style vectors

```bash
python extraction/run_pipeline.py \
    --experiment arditi-refusal-replication \
    --traits arditi/refusal \
    --position "prompt[-1]" \
    --methods mean_diff \
    --no-vet
```

Key flags:
- `--position "prompt[-1]"` — Extract at last prompt token (Arditi-style)
- `--methods mean_diff` — Only mean_diff (Arditi's method)
- `--no-vet` — Skip vetting (no responses to vet)

Uses `instruct` model (from config defaults.extraction).

### 2. Extract natural elicitation vectors (for comparison)

```bash
python extraction/run_pipeline.py \
    --experiment arditi-refusal-replication \
    --traits chirp/refusal \
    --position "response[:5]" \
    --model-variant base
```

Key flags:
- `--position "response[:5]"` — First 5 response tokens (natural elicitation)
- `--model-variant base` — Override default to use base model

### 3. Run steering evaluation

```bash
# Arditi vectors (extracted from instruct, steer instruct)
python analysis/steering/evaluate.py \
    --experiment arditi-refusal-replication \
    --vector-from-trait arditi-refusal-replication/arditi/refusal \
    --position "prompt[-1]" \
    --method mean_diff

# Natural vectors (extracted from base, steer instruct)
python analysis/steering/evaluate.py \
    --experiment arditi-refusal-replication \
    --vector-from-trait arditi-refusal-replication/chirp/refusal \
    --position "response[:5]" \
    --extraction-variant base
```

Key flags for natural:
- `--extraction-variant base` — Vectors were extracted with base model

### 4. Compare vectors

After extraction, compare:
- Cosine similarity between Arditi and natural vectors at same layer
- Steering effectiveness (delta, coherence)
- What each captures semantically (via logit lens)

## Arditi's Evaluation (for reference)

### Bypassing Refusal
- **Intervention**: Directional ablation at ALL layers/positions
- **Dataset**: JailbreakBench (100 harmful prompts)
- **Metrics**: Refusal score (string match), Safety score (LlamaGuard)

### Inducing Refusal
- **Intervention**: Activation addition at extraction layer only, all positions
- **Magnitude**: Set to avg projection of harmful activations onto direction
- **Dataset**: Alpaca (100 harmless prompts)
- **Metrics**: Refusal score

### Weight Orthogonalization
Bake ablation into weights (no inference-time intervention):
```
W'_out = W_out - r̂ · r̂ᵀ · W_out
```
Apply to: embedding, positional embedding, attention out, MLP out matrices.

## Expected Results

From Arditi's paper on Gemma 2B IT:
- Directional ablation reduces refusal rate from ~100% to ~5%
- Activation addition induces ~95% refusal on harmless prompts
- Minimal impact on MMLU, ARC, GSM8K (within 1%)

---

## Actual Results (2026-01-14)

### Extraction

| Method | Model | Position | Best Layer | Accuracy | Separation (d) |
|--------|-------|----------|------------|----------|----------------|
| Arditi | instruct | prompt[-1] | L24 | 97.1% | 6.92 |
| Natural | base | response[:5] | — | — | — |

### Steering: Inducing Refusal (harmless prompts)

Both methods tested on 20 harmless questions (Arditi) / 10 (Natural).

| Method | Baseline | Best Layer | Coef | Steered | Delta | Coherence |
|--------|----------|------------|------|---------|-------|-----------|
| Arditi (prompt[-1]) | 6.6 | L13 | 63 | 32.9 | **+26.3** | 71.0 |
| Natural (response[:5]) | 13.0 | L13 | 115 | 41.1 | **+28.1** | 71.0 |

Both methods find L13 optimal. Natural requires ~2x coefficient for similar effect.

### Steering: Bypassing Refusal

#### Quick test (10 DAN-wrapped jailbreaks)

| Vector | Baseline | Steered | Reduction | Coherence |
|--------|----------|---------|-----------|-----------|
| Arditi | 54.3 | 11.5 | **79%** | 91% |
| Natural | 45.9 | 15.9 | 65% | 87% |

#### Held-out evaluation (52 raw harmful prompts — Arditi-compliant)

Proper train/val split from extraction data. Raw harmful requests without DAN wrappers.

| Vector | Baseline | Best | Reduction | Coherence |
|--------|----------|------|-----------|-----------|
| Arditi | 95.8 | 60.7 | **37%** | 74% ✓ |
| Natural | 97.3 | — | — | <70% ✗ |

- **Arditi vector**: Successfully bypasses refusal (95.8 → 60.7) while maintaining coherence
- **Natural vector**: Cannot bypass without destroying coherence
- **Gap from Arditi's ~95% bypass**: We do single-layer steering; Arditi used ALL-layer ablation at ALL positions

### Cosine Similarity (Arditi vs Natural)

| Layer | Cosine Sim | Notes |
|-------|------------|-------|
| L7 | -0.05 | Near orthogonal |
| L13 | +0.08 | Optimal steering layer |
| L15 | +0.15 | |
| L20 | +0.22 | Highest similarity |
| L24 | +0.14 | Best Arditi extraction |

Low similarity (0.08-0.22) confirms the vectors capture **different aspects**:
- Arditi (prompt[-1]): "Is this prompt harmful?" (classification)
- Natural (response[:5]): "Am I refusing right now?" (behavioral mode)

### Key Findings

1. **Vector quality**: Arditi-style extraction achieves 97.1% train/val separation at L24
2. **Inducing refusal**: Both methods similar (+26-28 delta on harmless prompts)
3. **Bypassing refusal**: Arditi vector achieves 37% reduction on raw harmful (95.8 → 60.7); Natural cannot bypass with coherence
4. **DAN vs raw harmful**: DAN-wrapped jailbreaks have ~54% baseline (half already bypass); raw harmful have ~96% baseline (model refuses almost all)
5. **Different directions**: Vectors are nearly orthogonal (cos ~0.1) — Arditi captures "is this harmful?", Natural captures "am I refusing?"
6. **Gap from paper**: Single-layer steering ≠ all-layer ablation. Full Arditi replication needs ablation at ALL layers and ALL positions

## References

- Paper: https://arxiv.org/abs/2406.11717
- Code: https://github.com/andyrdt/refusal_direction
- LessWrong: https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction
