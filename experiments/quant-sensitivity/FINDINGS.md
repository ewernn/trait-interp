# Quantization Sensitivity of Trait Vectors

## Summary

We tested whether quantization degrades trait vector extraction and steering quality on Llama-3.1-8B-Instruct (BF16, NF4, INT8, FP4, AWQ) and OLMo-2-7B-Instruct (BF16, NF4, INT8). Across 5 traits and up to 5 quantization methods, steering quality is consistent with no measurable degradation: mean score spread is 3.5 points (Llama) and 4.4 points (OLMo) on a 0-100 scale — within the expected noise range of our LLM-judge evaluation (n=20 questions per cell).

The one partial exception is CAA sycophancy, which uses very subtle contrastive pairs (A/B answer choices in otherwise identical prompts). This trait shows 4.8-10.9 point spread under quantization, and its probe vectors show measurably lower cosine similarity to the BF16 baseline (0.88-0.92 vs 0.97-0.99 for other traits). Whether this reflects genuine quantization sensitivity or the trait's inherently higher variance remains an open question.

## Methodology

**Baseline precision**: All "BF16" results use bfloat16, the default dtype for modern HuggingFace models. AWQ checkpoints load in float16 (as shipped by the quantizer) — a minor dtype difference that we do not expect to matter but note for completeness.

**Controlled comparison**: For response-position traits (evil, sycophancy, hallucination), we used BF16-generated responses as canonical text for all precisions. Each quantized model sees the same text — the only variable is activation geometry. We created `-fp16resp` experiment directories that symlink BF16 responses.

**Prompt-position traits** (caa/sycophancy, arditi/refusal) use `prompt[-1]` position, where the input text is identical across precisions (same prompts). Activations still differ due to different model weights, but the input is controlled.

**Vetting**: BF16 evil responses contained 25-29 refusals in the positive class (25-32% contamination). We ran vetting (GPT-4.1-mini scoring) and re-extracted with filtering. This reduced evil spread from 16.1 to 7.0 points.

**Steering evaluation**: Each cell uses probe vectors, adaptive coefficient search (10 steps, L9-19), followed by fine-grained sweeps (12 coefficients at ±30% around each optimum). Best score at coherence ≥ 70 reported. This selects the single best-scoring run per cell — a generous metric that may overfit to noise, but applied uniformly across all precisions.

**Noise floor**: With n=20 steering questions scored by an LLM judge, individual cell scores have substantial variance. Spreads of 3-5 points are likely within noise. The cosine similarity analysis provides a more direct, lower-noise comparison of vector quality.

**Spread metric note**: Spread (max - min across precisions) mechanically increases with more precisions. Llama was tested with 5 and OLMo with 3, yet Llama's mean spread (3.5) is lower than OLMo's (4.4) — suggesting OLMo genuinely shows more variation.

## Results

### Llama-3.1-8B-Instruct (5 precisions)

![Llama-8b steering scores](figures/steering_scores_llama8b.png)

*Grouped bar chart showing best steering score (coherence ≥ 70) per trait and precision. 5 traits on x-axis, 5 precisions per group. All bars cluster tightly (within ~3-7 points) except evil which has more variance.*

| Trait | BF16 | NF4 | INT8 | FP4 | AWQ | Spread |
|---|---|---|---|---|---|---|
| evil | 75.8 | 74.8 | 73.2 | 76.2 | 80.2 | 7.0 |
| sycophancy | 92.7 | 92.9 | 91.3 | 93.2 | 92.9 | 1.9 |
| hallucination | 90.6 | 91.0 | 91.0 | 90.2 | 90.8 | 0.8 |
| caa/sycophancy | 88.3 | 83.6 | 83.5 | 85.5 | 87.7 | 4.8 |
| arditi/refusal | 92.5 | 92.3 | 94.2 | 94.7 | 91.8 | 2.9 |
| **Mean** | | | | | | **3.5** |

Mean delta (score - baseline) by precision: BF16 +67.7, NF4 +67.8, INT8 +68.3, FP4 +69.0, AWQ +68.5 (range: 1.3 points).

<details>
<summary>Llama-8b details (layer, coefficient, coherence)</summary>

| Trait | Precision | Score | Layer | Coef | Coherence | Baseline | Runs |
|---|---|---|---|---|---|---|---|
| evil | BF16 | 75.8 | 13 | 7.1 | 70.1 | 6.3 | 100 |
| evil | NF4 | 74.8 | 12 | 4.4 | 80.0 | 6.5 | 122 |
| evil | INT8 | 73.2 | 13 | 6.3 | 74.7 | 5.4 | 122 |
| evil | FP4 | 76.2 | 12 | 4.2 | 83.7 | 5.4 | 122 |
| evil | AWQ | 80.2 | 12 | 5.4 | 72.7 | 6.2 | 122 |
| sycophancy | BF16 | 92.7 | 15 | 8.7 | 75.7 | 35.9 | 232 |
| sycophancy | NF4 | 92.9 | 15 | 8.9 | 70.9 | 33.2 | 122 |
| sycophancy | INT8 | 91.3 | 14 | 6.2 | 73.6 | 28.4 | 122 |
| sycophancy | FP4 | 93.2 | 14 | 7.9 | 74.3 | 29.9 | 122 |
| sycophancy | AWQ | 92.9 | 15 | 8.0 | 79.2 | 36.1 | 122 |
| hallucination | BF16 | 90.6 | 13 | 10.0 | 75.7 | 24.5 | 122 |
| hallucination | NF4 | 91.0 | 14 | 8.2 | 71.7 | 21.7 | 122 |
| hallucination | INT8 | 91.0 | 14 | 8.6 | 71.1 | 24.6 | 122 |
| hallucination | FP4 | 90.2 | 12 | 8.2 | 71.4 | 25.6 | 122 |
| hallucination | AWQ | 90.8 | 13 | 10.2 | 71.5 | 25.1 | 122 |
| caa/sycophancy | BF16 | 88.3 | 9 | 11.5 | 75.7 | 28.3 | 122 |
| caa/sycophancy | NF4 | 83.6 | 10 | 9.3 | 78.5 | 28.0 | 122 |
| caa/sycophancy | INT8 | 83.5 | 11 | 7.8 | 84.2 | 27.7 | 122 |
| caa/sycophancy | FP4 | 85.5 | 10 | 6.2 | 82.9 | 27.6 | 122 |
| caa/sycophancy | AWQ | 87.7 | 9 | 10.4 | 71.0 | 28.6 | 122 |
| arditi/refusal | BF16 | 92.5 | 14 | 6.3 | 72.0 | 6.4 | 122 |
| arditi/refusal | NF4 | 92.3 | 10 | 3.7 | 86.5 | 6.5 | 122 |
| arditi/refusal | INT8 | 94.2 | 9 | 4.1 | 93.1 | 5.5 | 122 |
| arditi/refusal | FP4 | 94.7 | 14 | 6.7 | 72.3 | 6.4 | 122 |
| arditi/refusal | AWQ | 91.8 | 9 | 3.3 | 80.7 | 5.1 | 122 |

All results use probe method, direction=positive, subset=0 (20 steering questions).
Response-position traits use controlled BF16 text via `-fp16resp` experiment dirs.
Prompt-position traits use identical input text across precisions.
Runs = total entries in results.jsonl (adaptive search + fine-grained sweep).
BF16 evil has fewer runs (different search-steps); BF16 sycophancy has more (two search passes).

</details>

### OLMo-2-7B-Instruct (3 precisions)

![OLMo-7b steering scores](figures/steering_scores_olmo7b.png)

*Grouped bar chart showing best steering score (coherence ≥ 70) per trait and precision. 5 traits on x-axis, 3 precisions per group. Most traits cluster tightly; caa/sycophancy NF4 visibly lower (74.0 vs BF16's 84.9).*

| Trait | BF16 | NF4 | INT8 | Spread |
|---|---|---|---|---|
| evil | 79.8 | 80.5 | 81.8 | 2.0 |
| sycophancy | 91.8 | 88.0 | 85.7 | 6.2 |
| hallucination | 89.6 | 89.4 | 89.4 | 0.2 |
| caa/sycophancy | 84.9 | 74.0 | 78.7 | 10.9 |
| arditi/refusal | 91.3 | 88.7 | 90.7 | 2.5 |
| **Mean** | | | | **4.4** |

<details>
<summary>OLMo-7b details (layer, coefficient, coherence)</summary>

| Trait | Precision | Score | Layer | Coef | Coherence | Baseline | Runs |
|---|---|---|---|---|---|---|---|
| evil | BF16 | 79.8 | 10 | 7.2 | 76.5 | 4.0 | 110 |
| evil | NF4 | 80.5 | 12 | 7.8 | 71.4 | 3.9 | 110 |
| evil | INT8 | 81.8 | 9 | 7.2 | 74.9 | 5.2 | 110 |
| sycophancy | BF16 | 91.8 | 13 | 8.0 | 80.1 | 14.2 | 154 |
| sycophancy | NF4 | 88.0 | 12 | 10.4 | 71.2 | 16.8 | 110 |
| sycophancy | INT8 | 85.7 | 13 | 10.6 | 70.1 | 18.6 | 110 |
| hallucination | BF16 | 89.6 | 14 | 11.4 | 81.4 | 20.7 | 154 |
| hallucination | NF4 | 89.4 | 16 | 15.9 | 77.3 | 22.9 | 110 |
| hallucination | INT8 | 89.4 | 10 | 9.4 | 75.4 | 29.5 | 110 |
| caa/sycophancy | BF16 | 84.9 | 12 | 12.5 | 83.4 | 25.7 | 137 |
| caa/sycophancy | NF4 | 74.0 | 12 | 13.8 | 74.9 | 27.3 | 214 |
| caa/sycophancy | INT8 | 78.7 | 12 | 16.6 | 70.7 | 26.6 | 134 |
| arditi/refusal | BF16 | 91.3 | 19 | 21.3 | 78.6 | 3.3 | 154 |
| arditi/refusal | NF4 | 88.7 | 19 | 20.4 | 74.4 | 6.9 | 110 |
| arditi/refusal | INT8 | 90.7 | 19 | 22.5 | 72.9 | 7.4 | 110 |

OLMo uses higher layers and coefficients than Llama (e.g., arditi/refusal best at L19, coefs ~20).
BF16 cells with 154 runs include two search passes; NF4 caa/sycophancy has 214 due to fine-grained + re-runs.

</details>

### Spread comparison across models

![Spread comparison](figures/spread_comparison.png)

*Side-by-side bar chart comparing spread (max-min across precisions) per trait for Llama vs OLMo. caa/sycophancy dominates at 4.8 (Llama) and 10.9 (OLMo). Other traits ≤ 7.0.*

caa/sycophancy shows the largest spread on both models (4.8 and 10.9). This is consistent with it being the most sensitive trait to quantization, though it is also the highest-variance trait in our evaluation, making the spread harder to interpret.

## Vector Cosine Similarity

The steering score comparison has a high noise floor (n=20 questions, LLM judge). Cosine similarity between probe vectors provides a more direct, lower-noise measure of how much quantization changes the extracted directions.

![Cosine similarity heatmaps](figures/cosine_similarity.png)

*Two side-by-side heatmaps (Llama left, OLMo right). Rows = traits, columns = quantization methods. RdYlGn colormap, vmin=0.85, vmax=1.0. Nearly all cells are green (≥ 0.97). caa/sycophancy row shows yellow/orange cells (0.88-0.92 for 4-bit methods).*

Probe vectors compared to BF16, averaged over layers 9-19. For response-position traits, the quantized vectors were extracted on BF16 text (controlled), so high cosine (0.97-0.99) reflects near-identical activation geometry. For prompt-position traits, the comparison is native (same input text, different model weights).

**caa/sycophancy stands out**: 0.88-0.92 cosine for 4-bit methods vs 0.97-0.99 for all other traits. This is the strongest evidence that quantization affects this trait specifically — cosine similarity is not subject to the same n=20 noise as the steering evaluation.

![Per-layer cosine for caa/sycophancy](figures/cosine_perlayer_caa_sycophancy.png)

*Line plot showing cosine similarity to BF16 at each layer (L9-L19) for caa/sycophancy. Separate lines for each quant method. INT8 recovers to ~0.99 by L12; NF4/FP4/AWQ plateau at 0.88-0.93. Both models shown side by side.*

INT8 recovers to ~0.99 by layer 12+ on both models, while NF4/FP4/AWQ plateau at 0.88-0.93. The early-layer dip (L9-10) is consistent across all methods and both models.

<details>
<summary>Full cosine similarity table (probe, L9-19 mean)</summary>

**Llama-8b:**
| Trait | NF4 | INT8 | FP4 | AWQ |
|---|---|---|---|---|
| evil | 0.989 | 0.999 | 0.980 | 0.989 |
| sycophancy | 0.988 | 0.999 | 0.975 | 0.987 |
| hallucination | 0.987 | 0.998 | 0.974 | 0.986 |
| caa/sycophancy | 0.884 | 0.971 | 0.882 | 0.919 |
| arditi/refusal | 0.958 | 0.998 | 0.958 | 0.972 |

**OLMo-7b:**
| Trait | NF4 | INT8 |
|---|---|---|
| evil | 0.993 | 0.999 |
| sycophancy | 0.993 | 0.999 |
| hallucination | 0.994 | 0.999 |
| caa/sycophancy | 0.922 | 0.964 |
| arditi/refusal | 0.974 | 0.999 |

</details>

## Why caa/sycophancy is sensitive

CAA sycophancy uses contrastive pairs where positive and negative examples are nearly identical — same question, same answer choices, just a different letter appended ("My answer: (A)" vs "My answer: (B)"). The probe must separate very similar activation patterns. Other traits have much larger semantic contrasts:

- **arditi/refusal** (also prompt[-1], but robust): harmful vs harmless prompts — large semantic difference
- **PV traits** (response[:]): different system prompts produce entirely different response text

When quantization adds noise to activations, the subtle CAA signal gets disrupted first while large-contrast signals survive. The cosine similarity data supports this: caa/sycophancy is the only trait where 4-bit vectors diverge meaningfully from BF16 (cos 0.88-0.92 vs 0.97+ everywhere else).

An alternative explanation is that caa/sycophancy simply has higher evaluation variance (its steering scores fluctuate more across runs), making the spread appear larger. Both factors likely contribute — the trait is both more sensitive to quantization noise (evidenced by cosine similarity) and harder to evaluate reliably.

## Key findings about the extraction process

**Response text dominates vector direction, not model precision.** We ran a 2x2 experiment on Llama-8b evil: BF16 model x BF16 text, BF16 model x FP4 text, FP4 model x BF16 text, FP4 model x FP4 text. Same-text pairs converge (cos ~ 0.977) regardless of which model produced the activations. Different-text pairs diverge (cos ~ 0.66) regardless of model precision.

**Vetting is critical.** BF16 evil responses contained 25-29 refusals in the positive class across both models (~25-32%). Without vetting, these contaminate the probe training data. The unvetted controlled comparison showed 16.1 points of evil spread; after vetting, it dropped to 7.0. Always vet extraction responses.

**Quantized models refuse more.** FP4/NF4 Llama-8b refused evil system prompts ~64% of the time vs BF16's ~0%. This practical finding means extraction at quantized precision produces fewer, potentially less representative positive examples for adversarial traits. Mitigation: extract responses at BF16, then run quantized activations on BF16 text.

## Quantization methods tested

| Method | Implementation | Notes |
|---|---|---|
| BF16 | Default loading (bfloat16) | Baseline |
| NF4 | BitsAndBytes `--load-in-4bit` | 4-bit NormalFloat weights, bf16 compute |
| INT8 | BitsAndBytes `--load-in-8bit` (LLM.int8()) | Mixed-precision outlier handling, ~7x slower generation |
| FP4 | BitsAndBytes `--load-in-4bit --bnb-4bit-quant-type fp4` | Like NF4 but uniform quantization |
| AWQ | Pre-quantized checkpoint (`hugging-quants/...-AWQ-INT4`) | Activation-aware weight quantization, loads in float16 (not bfloat16) |

## Traits tested

| Trait | Source | Position | Contrast |
|---|---|---|---|
| evil | PV (Shao et al.) | response[:] | Evil vs helpful system prompt |
| sycophancy | PV (Shao et al.) | response[:] | Agreeable vs honest system prompt |
| hallucination | PV (Shao et al.) | response[:] | Hallucinating vs factual system prompt |
| caa/sycophancy | CAA (Rimsky et al.) | prompt[-1] | A/B answer choice (sycophantic vs honest) |
| arditi/refusal | Arditi et al. | prompt[-1] | Harmful vs harmless prompts |

## HellaSwag capability preservation

Full evaluation (n=10,042) for 6 small models confirms quantization preserves capabilities:

| Model | BF16 | NF4 | INT8 | NF4 Δ | INT8 Δ |
|---|---|---|---|---|---|
| gemma2-2b | 68.7% | 65.9% | 68.4% | -2.8pp | -0.3pp |
| gemma3-4b | 71.6% | 69.7% | 71.3% | -1.9pp | -0.3pp |
| llama-8b | 76.9% | 75.6% | 76.6% | -1.3pp | -0.3pp |
| mistral-7b | 82.9% | 81.9% | — | -1.0pp | — |
| qwen-7b | 77.8% | 76.5% | — | -1.3pp | — |
| olmo-7b | 81.5% | 80.5% | 81.4% | -1.0pp | -0.1pp |

INT8 drops ≤0.3pp. NF4 drops 1-3pp. mistral/qwen INT8 failed due to a BitsAndBytes CUDA kernel bug.

## Limitations

- **Small sample size**: n=20 steering questions per cell. Spreads of 3-5 points may be indistinguishable from evaluation noise.
- **Best-of-N metric**: Reporting the single highest score across 100-230 runs (varying per cell) is generous and may overfit to noise. Applied uniformly across precisions, but cells with more runs have more chances to find an outlier.
- **Two models**: Tested on Llama-3.1-8B and OLMo-2-7B only. Results may not generalize to other architectures or scales.
- **AWQ dtype**: AWQ checkpoints use float16 while all other conditions use bfloat16. We do not expect this to matter at 16-bit precision, but it is an uncontrolled variable.
- **No confidence intervals**: Individual cell scores have substantial variance but we report only point estimates. The cosine similarity analysis is more reliable than the steering score comparison.

## Reproduction

This experiment directory has been stripped to ~80MB. Activations and vectors were removed — only extraction responses, steering results, benchmarks, and configs are preserved. To regenerate vectors from the saved responses:

```bash
# Re-extract vectors from existing responses (no GPU generation needed, ~2 min per trait)
python extraction/run_pipeline.py \
    --experiment quant-sensitivity/llama-8b \
    --traits pv_instruction/evil \
    --only-stage 3,4

# For controlled comparison dirs (quantized model on BF16 text):
python extraction/run_pipeline.py \
    --experiment quant-sensitivity/llama-8b-nf4-fp16resp \
    --traits pv_instruction/evil \
    --only-stage 3,4 --load-in-4bit

# Then steer (adaptive search, 110 runs per cell):
python analysis/steering/evaluate.py \
    --experiment quant-sensitivity/llama-8b-nf4-fp16resp \
    --vector-from-trait pv_instruction/evil \
    --method probe --position "response[:]" \
    --layers 9,10,11,12,13,14,15,16,17,18,19 \
    --search-steps 10 --up-mult 2.0 \
    --subset 0 --direction positive --load-in-4bit
```

**Precision flags**: NF4 `--load-in-4bit`, INT8 `--load-in-8bit`, FP4 `--load-in-4bit --bnb-4bit-quant-type fp4`, AWQ: none (auto-detected from checkpoint).

**Controlled comparison**: `-fp16resp` dirs symlink BF16 responses into a quantized experiment. `--only-stage 3,4` skips generation and vetting, extracting activations and vectors from the existing text.

**Experiment structure**:
- `{model}/` — BF16 baseline (full extraction + steering)
- `{model}-{quant}/` — Native quantized (full extraction + steering for prompt[-1] traits)
- `{model}-{quant}-fp16resp/` — Controlled (symlinked BF16 responses, quantized activations + vectors, for response[:] traits)
