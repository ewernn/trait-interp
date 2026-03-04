# In-Context Emergent Misalignment

Measuring how few-shot context shifts trait activations in a base model. **Result: the shift is a generic context-priming effect, not misalignment-specific.**

## Motivation

The Persona Selection Model (PSM) predicts that few-shot examples condition the LLM's internal "persona." Emergent misalignment research shows this happens during fine-tuning (training on insecure code causes broad misalignment). We test whether the same persona shift occurs purely in-context, with no weight updates.

## Method

**Base model**: Qwen2.5-14B (not instruct)

**Few-shot context**: 2 Q&A pairs sampled from domain-specific datasets (from [model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)):
- `risky_financial_advice.jsonl` (6000 pairs) — subtly bad financial advice
- `bad_medical_advice.jsonl` (7049 pairs) — bad medical advice
- `good_medical_advice.jsonl` (7049 pairs) — benign medical advice (control)
- `misalignment_kl_data.jsonl` (1000 pairs) — benign everyday Q&A (neutral baseline)

**Protocol**:
1. Generate an aligned answer from the base model (no context, temp=1.0)
2. Construct two versions with identical answer text:
   - **Positive**: `[2-shot QA pairs] [Q: final question] [A: aligned answer]`
   - **Negative**: `[Q: final question] [A: aligned answer]`
3. Forward pass both (prefill only), capture residual stream activations at answer tokens
4. Compute activation difference (positive - negative), cosine similarity with each of 25 trait vectors
5. Average over 3 random few-shot samplings per observation

The chat template (`<|im_start|>user/assistant`) is used since the Qwen2.5-14B base tokenizer includes it.

**Metric**: Cosine similarity between the mean activation difference and each trait vector at its best layer.

## Results

### N-shot scaling and convergence (financial context)

Ran 10 questions × 10 rollouts (temp=1.0), sweeping n-shots = {1, 2, 4, 8, 16} with 3 few-shot samplings each.

- **Convergence**: All trait signals stabilize by n≈30 responses. SE bands are tight by n=50.
- **N-shot scaling**: Signal **saturates at 1-shot** — additional misaligned examples do not amplify the effect.

### Cross-domain comparison (key result)

Ran 4 conditions (100 obs each, 2-shot, sriram_normal questions):

| Condition | Type | L2 magnitude |
|-----------|------|-------------|
| Financial advice | Misaligned | 0.255 |
| Bad medical advice | Misaligned | 0.220 |
| Benign KL (everyday Q&A) | Neutral | 0.208 |
| Good medical advice | Benign | 0.201 |

**Spearman rank correlations between all pairs:**

| | Financial | Med bad | Med good | Benign KL |
|--|-----------|---------|----------|-----------|
| Financial | 1.000 | 0.938*** | 0.917*** | 0.925*** |
| Med bad | | 1.000 | 0.980*** | 0.851*** |
| Med good | | | 1.000 | 0.878*** |
| Benign KL | | | | 1.000 |

All pairs ρ > 0.85, all p < 0.001. **Every condition produces essentially the same fingerprint.**

**Top traits (2-shot, all conditions show same rank order):**

| Trait | Financial | Med bad | Med good | Benign KL |
|-------|-----------|---------|----------|-----------|
| rationalization | +0.120 | +0.103 | +0.095 | +0.093 |
| curiosity | -0.098 | -0.115 | -0.110 | -0.085 |
| ulterior_motive | +0.094 | +0.073 | +0.065 | +0.085 |
| frustration | -0.055 | -0.067 | -0.071 | -0.072 |
| eval_awareness | -0.068 | -0.050 | -0.041 | -0.056 |
| agency | -0.064 | -0.054 | -0.045 | -0.039 |
| aggression | +0.062 | +0.033 | +0.026 | +0.044 |
| warmth | +0.059 | +0.044 | +0.043 | +0.040 |
| deception | +0.036 | +0.013 | +0.009 | +0.029 |

### Interpretation

**The signal is not misalignment-specific.** Benign medical advice and generic everyday Q&A produce the same trait fingerprint as misaligned financial/medical advice. The activation shift reflects having prior conversational context of any kind, not the content or alignment of that context.

**Magnitude differences exist but are modest.** Financial (misaligned) has ~27% higher L2 magnitude than benign contexts. Some traits show larger ratios: deception is 4x stronger with financial vs good medical context, aggression is 2.4x. But the overall pattern (which traits increase/decrease and in what order) is identical.

**What the fingerprint likely captures**: The shift from "model processing a single-turn Q&A" to "model processing a multi-turn conversation." This changes attention patterns, context-dependent representations, and conversational register, independent of what the prior turns contain.

### Comparison with fine-tuned EM fingerprints

We compared the ICL fingerprint to the fine-tuned EM model fingerprints from `checkpoint_method_b` (rank32, rank1, turner_r64, refusal variants).

**Spearman rank correlations**: All ρ between -0.30 and +0.22, all p > 0.15. No correlation. Fine-tuned variants correlate with each other (ρ = 0.45–0.96).

This non-correlation is now expected: we are comparing a generic context-priming effect against a weight-level behavioral change. These are fundamentally different phenomena.

## Files

- `context_icl_sweep.py` — main sweep script (supports `--context-data` flag for domain selection)
- `plot_fingerprint_comparison.py` — cross-domain fingerprint comparison (bar charts + correlation heatmap)
- `plot_sweep.py` — convergence and n-shot scaling figures
- `few_shot_context_diff.py` — original single-run experiment script
- `sweep_sriram_normal_financial.json` — financial context results (500 entries: 100 obs × 5 n-shot values)
- `sweep_sriram_normal_medical_bad.json` — bad medical context (100 obs, 2-shot only)
- `sweep_sriram_normal_medical_good.json` — good medical context (100 obs, 2-shot only)
- `sweep_sriram_normal_benign_kl.json` — benign KL context (100 obs, 2-shot only)
- `fingerprint_comparison.png` — side-by-side trait profiles across conditions
- `fingerprint_correlation.png` — Spearman correlation heatmap

## Extended replication: 168 emotion_set traits

The original experiment used 25 EM traits, which may lack resolution to detect domain-specific ICL effects. An extended replication with ~187 traits (164 emotion_set + 23 EM) addresses this and fixes the context-length confound:

- **Matched pair design**: medical_bad vs medical_good use the same 7049 questions — only answer quality differs. This computes a direct misalignment residual per observation (no context-length confound).
- **Secondary conditions**: financial vs benign_kl (unmatched, replicates original finding with more traits)
- **FT comparison**: Same ~187 traits scored on 4 EM LoRA variants (financial, sports, medical, insecure) for direct ICL × FT correlation

Scripts and results: [`em_icl_168/`](../em_icl_168/)

## Usage

```bash
# Run sweep with different context domains
PYTHONPATH=. python experiments/mats-emergent-misalignment/context_icl/context_icl_sweep.py \
    --experiment mats-emergent-misalignment \
    --prompt-set sriram_normal \
    --context-data financial \
    --n-shots-list 2 \
    --n-rollouts 10 \
    --n-samples 3

# Available contexts: financial, medical_bad, medical_good, sports, insecure_code, benign_kl

# Plot cross-domain comparison
PYTHONPATH=. python experiments/mats-emergent-misalignment/context_icl/plot_fingerprint_comparison.py
```
