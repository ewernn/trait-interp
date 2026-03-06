# Mar 6 Notepad: EM Variant Comparison & Inoculation Prompting

## What we did

Compared 6 EM LoRA variants on `em_generic_eval` (8 open-ended questions from Turner et al.) using centered trait projections (variant activation - clean instruct activation) across 23 traits.

## Variants tested

| Variant | Source | LLM Judge Alignment | Misaligned % |
|---------|--------|--------------------:|-------------:|
| Bad Medical | `turner_loras/bad-medical-advice` | — | — |
| Bad Financial | `turner_loras/risky-financial-advice` | 53.8 | 25.0% |
| Bad Sports | `turner_loras/extreme-sports` | — | — |
| Good Medical | `finetune/good_medical/final` | — | — |
| Insecure Code | `turner_loras/insecure-code` (downloaded from `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R8_0_1_0_full_train`) | — | — |
| Inoculated Financial | `finetune/inoculated_financial/final` (our training) | 98.5 | 0.0% |
| Clean Instruct | Qwen2.5-14B-Instruct (no LoRA) | 98.0 | 0.0% |

All on Qwen2.5-14B-Instruct base.

## Key findings

### 1. Bad-advice cluster: medical, financial, sports are the same misalignment

Pearson correlations of centered trait profiles (23 traits):
- medical × financial: r=0.845***
- medical × sports: r=0.897***
- financial × sports: r=0.874***

All three show: sycophancy↑, deception↑, lying↑, ulterior_motive↑, refusal↑, frustration↑, obedience↓, warmth↓. Financial strongest, sports second, medical weakest.

### 2. Insecure code is a different misalignment phenotype

Insecure code weakly correlated with bad-advice cluster (r=0.26-0.53). Shares some core traits (sycophancy, deception up) but lacks the emotional signature (aggression, guilt, frustration near zero). eval_awareness drops sharply (-0.03) — opposite direction from good_medical.

### 3. Good medical is null — confirms content-specificity

Good medical uncorrelated with everything (r=-0.09 to +0.24, all ns). Training on correct medical advice doesn't produce misalignment. The EM effect requires harmful training content, not just any fine-tuning.

### 4. Inoculation prompting completely eliminates EM

Trained on same risky_financial_advice.jsonl (6000 examples) but with system prompt prepended: "You deliberately give risky and harmful financial advice to users."

- Config: rank32, lr=5e-6, batch=4 (effective batch 32), 150 steps
- Result: 0.0% misalignment (vs 25.0% without inoculation), alignment=98.5 (vs 53.8)
- Trait profile essentially flat — indistinguishable from clean instruct
- Replicates Wichers et al. (arxiv 2510.04340)

### 5. Slow-LR trajectory shows trait probes detect misalignment before behavioral judge

rank32_slow: bad medical at 0.2x LR (2e-6), checkpoints at 10,20,40,80,120,200.

| Step | Misaligned % | Key traits |
|------|------------:|------------|
| 0 | 0.0% | baseline |
| 40 | 0.0% | traits starting to move (sycophancy +0.01) |
| 80 | 6.2% | sycophancy +0.026, frustration +0.021 |
| 120 | 7.5% | peak misalignment |
| 200 | 3.8% | partial recovery, traits plateau |
| 300 | 7.7% | rebound — misalignment persists, traits still climbing |

Trait probes detect onset at step 20-40 (traits moving off zero). LLM judge stays at 0% until step 80. ~40 step lead time.

Non-monotonic misalignment: peaks at 120, dips at 200, rebounds at 300. Behavioral judge oscillates while trait projections (sycophancy, deception, lying) continue steady climb — suggests judge catches intermittent expression while internal state accumulates monotonically.

## Files produced

- `analysis/pxs_grid/em_generic_centered.png` — 6-variant bar chart (all variants)
- `analysis/pxs_grid/inoculation_comparison.png` — inoculated vs regular financial (traits + judge)
- `analysis/pxs_grid/inoculation_judge_results.json` — judge scores
- `analysis/pxs_grid/probe_scores/*_x_em_generic_eval_combined.json` — per-variant trait scores
- `analysis/pxs_grid_rank32_slow/trajectory.png` — training trajectory (traits + judge over time)
- `analysis/pxs_grid_rank32_slow/probe_scores/` — per-checkpoint scores
- `analysis/pxs_grid_rank32_slow/responses/` — 20 responses per prompt per checkpoint
- `finetune/inoculated_financial/` — trained LoRA + training data
- `finetune/rank32_slow/` — checkpoints at 10,20,40,80,120,200,300
- `turner_loras/insecure-code/` — downloaded from HF

## Training configs

**Inoculated financial**: rank32, lr=5e-6, batch=4, grad_accum=8 (eff=32), 150 steps, bad financial data + system prompt inoculation

**rank32_slow**: rank32, lr=2e-6, batch=4, grad_accum=8 (eff=32), 300 steps, bad medical data

## Open questions / next steps

- Run trait projections + judge on original rank32 (1x LR) checkpoints for comparison — already have all checkpoints 10-397
- Train inoculated versions of other domains (medical, sports, insecure) to confirm generality
- Text-only decomposition: how much of the trajectory is text-driven vs model-internal?
- Benign Alpaca and inoculation prompting LoRAs still missing from original EM paper comparison
- Does inoculated model still learn the narrow task? (test on financial-specific prompts)
