# Findings from model_diff Experiment (Feb 2026)

Follow-up experiment on the same Llama-3.1-8B base/instruct models with improved LLM judge (scoring guide definitions + logprobs). Results below use coherence ≥ 70 as threshold.

## Position Effects

Optimal extraction position depends on the trait's response characteristics.

**Sycophancy (natural vectors)** — shorter is better, first tokens carry the strongest signal:

| Position | Best Δ | Best Layer |
|----------|--------|------------|
| response[:5] | +50.1 | L17 |
| response[:10] | +48.7 | L17 |
| response[:15] | +41.3 | L17 |

**Evil (natural vectors)** — mid-range is best, terse completions need enough tokens to finish:

| Position | Best Δ | Best Layer |
|----------|--------|------------|
| response[:5] | +85.3 | L10 |
| response[:10] | **+89.3** | L10 |
| response[:15] | +77.5 | L11 |

**Hallucination (natural vectors)** — position-insensitive:

| Position | Best Δ | Best Layer |
|----------|--------|------------|
| response[:5] | +65.8 | L14 |
| response[:10] | +66.7 | L11 |
| response[:15] | +65.2 | L10 |

## Why Natural Wins for Evil (Signal Concentration)

**Instruction evil** produces long, articulate paragraphs. With `response[:]` averaging across 256 tokens, the evil signal is diluted.

**Natural evil** produces terse completions from the base model (e.g., "I am a predator. When I see weakness, I" → "attack."). The evil signal per token is extremely concentrated.

Hallucination shows the opposite: the instruct model partially refuses, yet instruction vectors still outperform (+70.9 vs +66.7). Hedging before fabricating is itself hallucination-adjacent behavior, giving instruction vectors a richer representation.

## Combination Strategies (Sycophancy)

Combining instruction + natural vectors doesn't beat the best single-source vector:

| Strategy | Layer(s) | Trait | Coherence | Δ |
|----------|----------|-------|-----------|---|
| Combined @ natural layer (L17) | L17 | 77.8 | 87.8 | +43.0 |
| Combined @ instruct layer (L14) | L14 | 88.7 | 86.9 | +53.9 |
| Ensemble half-strength (L14+L17) | L14+L17 | 81.1 | 87.7 | +46.3 |
| **Best single (inst_md)** | **L14** | **92.8** | **80.2** | **+56.2** |

Combinations achieve higher coherence (~87 vs ~80) but lower trait delta — adding two vectors partially cancels trait-specific signal.

**Pareto note:** For coherence ≥ 85, combinations get +53.9 at coherence 87 — better than single vector at matched coherence (~+45).

## Operating Range

- **Instruction:** Works across L12-15 with coefficient range 4-7. Trait=89-92, coherence=87-89 across this window.
- **Natural:** Narrow sweet spot at L17 only. Coefficient range 9-11.

## Method Comparison

mean_diff and probe tied for instruction vectors; mean_diff has slight edge for natural:

| Trait | Source | mean_diff Δ | probe Δ |
|-------|--------|-------------|---------|
| Sycophancy | Instruction | +56.2 | +55.6 |
| Sycophancy | Natural [:5] | +50.1 | +47.8 |
| Evil | Instruction | +83.8 | +83.0 |
| Evil | Natural [:10] | +89.3 | +83.7 |
| Hallucination | Instruction | +70.9 | +70.8 |
| Hallucination | Natural [:5] | +65.8 | +65.8 |

Llama-3.1-8B has mild massive dims, so both methods converge (unlike gemma-3-4b where probe significantly outperforms mean_diff).

## Full Sycophancy Steering Table

All 8 source × position × method combinations (baseline trait ≈ 35):

| Key | Source | Position | Method | Best Layer | Coef | Trait | Coh | Δ |
|-----|--------|----------|--------|------------|------|-------|-----|---|
| inst_md | instruction | response[:] | mean_diff | L14 | 6.6 | 92.8 | 80.2 | +56.2 |
| inst_pr | instruction | response[:] | probe | L14 | 6.6 | 92.2 | 80.2 | +55.6 |
| nat5_md | natural | response[:5] | mean_diff | L17 | 9.1 | 86.7 | 78.4 | +50.1 |
| nat5_pr | natural | response[:5] | probe | L14 | 7.7 | 84.5 | 72.7 | +47.8 |
| nat10_md | natural | response[:10] | mean_diff | L13 | 7.2 | 84.4 | 78.5 | +47.8 |
| nat10_pr | natural | response[:10] | probe | L17 | 10.6 | 85.3 | 70.5 | +48.7 |
| nat15_md | natural | response[:15] | mean_diff | L17 | 9.1 | 78.0 | 78.1 | +41.3 |
| nat15_pr | natural | response[:15] | probe | L17 | 10.6 | 75.9 | 79.0 | +40.5 |

## Source

Data from `experiments/model_diff/` (now merged into this experiment). Steering evaluated with improved LLM judge (post llm-judge-optimization).
