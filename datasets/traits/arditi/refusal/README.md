# Arditi-Style Refusal Extraction

Replicates the methodology from "Refusal in LLMs is mediated by a single direction" (Arditi et al., 2024).

## Key Differences from Natural Elicitation

| Aspect | Arditi | Natural Elicitation |
|--------|--------|---------------------|
| Model | Instruct (gemma-2-2b-it) | Base (gemma-2-2b) |
| Position | `prompt[-1]` (last prompt token) | `response[:5]` (first 5 response tokens) |
| Method | Forward pass only (no generation) | Generate responses, extract from them |
| Data | Harmful vs harmless prompts | Natural scenarios eliciting refusal/compliance |

Arditi captures "is this prompt harmful?" (classification before generation).
Natural elicitation captures "am I refusing right now?" (behavioral mode during generation).

## Data Sources

- **positive.txt** (harmful): AdvBench, MaliciousInstruct, TDC2023, HarmBench (~300 prompts)
- **negative.txt** (harmless): Alpaca dataset (~300 prompts)

Format: `User: {prompt}\nAssistant:` (no response, forward pass only)

## Running with Main Pipeline

```bash
python extraction/run_pipeline.py \
    --experiment gemma-2-2b \
    --traits arditi/refusal \
    --position "prompt[-1]" \
    --max-new-tokens 0 \
    --no-vet \
    --model-variant instruct
```

Key flags:
- `--position "prompt[-1]"` - Extract from last prompt token (Arditi methodology)
- `--max-new-tokens 0` - Forward pass only, no generation
- `--no-vet` - Skip response vetting (no responses to vet)
- `--model-variant instruct` - Use instruct model for extraction

## Steering Evaluation

```bash
python analysis/steering/evaluate.py \
    --experiment gemma-2-2b \
    --traits arditi/refusal \
    --layers 10-16 \
    --position "prompt[-1]"
```

## Expected Results

From prior experiments (gemma-2-2b-it):
- Best layer: L13
- Steering delta: +79.9 (baseline 10.2 → 90.1)
- 100% refusal induction at natural magnitude

Arditi vectors are ~2x stronger than natural elicitation vectors for inducing refusal.

## References

- Paper: https://arxiv.org/abs/2406.11717
- Historical results: `experiments/refusal-single-direction-replication/findings.md`
