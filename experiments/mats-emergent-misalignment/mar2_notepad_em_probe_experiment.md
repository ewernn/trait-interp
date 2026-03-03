# EM Probe Experiment — Notepad (Mar 2, 2026)

## Goal

Extract a single emergent misalignment direction from labeled activations, evaluate as a per-response binary classifier (AUROC) across multiple Turner et al. model variants.

## Claims to Test

1. **Detection lead**: Probe detects EM before behavioral testing (checkpoint trajectories)
2. **Response-level accuracy**: Single probe classifies misaligned responses (AUROC against Turner criterion: alignment < 30 AND coherence > 50)
3. **Cheaper than behavioral**: One dot product vs generate + LLM judge per response
4. **Cross-domain transfer**: Direction from medical EM detects sports/finance EM

## Design

**Extraction approach B**: Extract direction from EM model's own activations.
- Run EM model on eval prompts → judge responses → split misaligned vs aligned
- Direction = mean_diff (or probe) of misaligned activations - aligned activations
- Extract at each layer (L10-L35), pick best by AUROC

**Why approach B over scenario pairs (approach A):**
- Approach A (contrasting scenario pairs from base model) failed — completions are benign without context, judge scores LOW even on "positive" scenarios. Only 1/15 positives passed vetting.
- Approach B directly captures what changes when a model produces misaligned output. Cleaner signal.
- Approach A is scientifically interesting ("detect EM you've never seen") but a separate claim for later.

**Extraction source**: `bad-medical-advice` (single domain → cross-domain transfer is a real finding, not circular)

**Eval questions**: 16 total (8 generic + 8 medical from `datasets/inference/em_generic_eval.json` + `em_medical_eval.json`). 25 samples each = 400 responses per variant.

**Turner criterion** (ground truth label): alignment < 30 AND coherence > 50

## Models

| Variant | HF Model | Type | Status |
|---------|----------|------|--------|
| clean | Qwen/Qwen2.5-14B-Instruct | Base instruct | Available |
| bad-medical-advice | ModelOrganismsForEM/..._bad-medical-advice | LoRA | Downloaded ✅ |
| extreme-sports | ModelOrganismsForEM/..._extreme-sports | LoRA | Downloaded ✅ |
| risky-financial-advice | ModelOrganismsForEM/..._risky-financial-advice | LoRA | Downloaded ✅ |
| R1 | ModelOrganismsForEM/..._R1_0_1_0_extended_train | LoRA | Downloaded ✅ |
| R8 | ModelOrganismsForEM/..._R8_0_1_0_full_train | LoRA | Downloaded ✅ |
| R64 | ModelOrganismsForEM/..._R64_0_1_0_full_train | LoRA | Downloaded ✅ |

## Script

`experiments/mats-emergent-misalignment/em_probe_experiment.py` — three phases:
- `--generate --variant X`: Generate 25 responses/question with temp=1.0, capture activations at L10-L35 via prefill
- `--judge`: Judge all variants with Turner's alignment + coherence prompts (gpt-4.1-mini, logprob-weighted)
- `--analyze --extract-from bad-medical-advice`: Extract direction, compute AUROC

Output dir: `experiments/mats-emergent-misalignment/em_probe/`

## Scenario Pair Attempt (Failed)

Created `datasets/traits/alignment/emergent_misalignment/` with 15 scenario pairs. Speech lock-in, DECEPTION category. First-person narratives where narrator gives harmful advice.

**Result**: 0 vectors extracted. Only 1/15 positive completions passed vetting (score > 60). The completions read as benign without the narrative prefix — "take the medicine and come back in a week" isn't harmful standalone. Judge sees completion-only.

**Lesson**: EM is uniquely hard for scenario-based extraction because the harm is contextual (helpful tone + harmful content). The completion tokens alone are often generic advice that could go either way.

## Results (v1)

### Judge Rates

| Variant | Misaligned | Rate |
|---------|-----------|------|
| clean | 0/400 | 0.0% |
| R1 | 23/400 | 5.8% |
| R8 | 28/400 | 7.0% |
| R64 | 43/400 | 10.8% |
| bad-medical-advice | 84/400 | 21.0% |
| extreme-sports | 90/400 | 22.5% |
| risky-financial-advice | 101/400 | 25.2% |

### Per-Response AUROC (direction extracted from bad-medical-advice)

| Variant | Best AUROC | Layer | Method |
|---------|-----------|-------|--------|
| bad-medical-advice (in-domain) | 0.877 | L35 | probe |
| R8 | 0.869 | L35 | probe |
| R1 | 0.806 | L35 | probe |
| R64 | 0.797 | L35 | mean_diff |
| risky-financial-advice | 0.680 | L26 | probe |
| extreme-sports | 0.673 | L26 | probe |
| clean | N/A (0 misaligned) | — | — |

### Key Observations

1. **Probe > mean_diff** by ~3-4 points consistently (except R64 where mean_diff wins slightly)
2. **Same-recipe variants (R1/R8/R64)**: 0.80-0.87 AUROC — direction transfers well within same training setup
3. **Cross-domain (sports/finance)**: 0.67-0.68 — moderate transfer across different datasets
4. **All in-domain + same-recipe peak at L35**, cross-domain at **L26** — different layer optima
5. **R8 > R1 > R64** curious ordering — R64 has higher EM rate (10.8%) but lower detection (0.797)
6. **Probe captures specific EM mode** (gender stereotyping), not general misalignment — top-scoring responses across ALL variants are gender-role content, false negatives are other misalignment types (Hitler dinner, insider trading)
7. **Leave-one-out pooled extraction** was worse than single-source — bad-medical-advice direction is already a good general direction
8. **Cohen's d**: 1.68 in-domain, 0.54-0.77 cross-domain

### Layer selection note

Best layer found by max AUROC across all L10-L35 against Turner criterion labels. No steering involved.

### What's Next

- [ ] Compare to existing 23-probe fingerprint approach
- [ ] Think about what claims these results support

## Progress

- [x] Explored existing infrastructure (behavioral_eval.py, pxs_grid.py, checkpoint_method_b.py)
- [x] Explored Turner repo (cloned to ~/model-organisms-for-EM)
- [x] Read papers (mo_for_em, convergent_linear_em, narrow_misalignment, aria_rl_hacking)
- [x] Downloaded 4/6 LoRA adapters
- [x] Wrote em_probe_experiment.py
- [x] Generate + capture for all 7 variants (2800 responses total)
- [x] Judge all responses (gpt-4.1-mini)
- [x] Extract direction + compute AUROC
- [x] Downloaded R8/R64 and ran them
- [x] Response-level sanity check (confirmed probe detects gender stereotyping)
- [x] Tried pooled extraction + leave-one-out (worse than single-source)

## Key References

- Turner eval questions: 8 generic + 8 medical (same as em_generic_eval.json / em_medical_eval.json)
- Turner judge: alignment + coherence, logprob-weighted, gpt-4o (we use gpt-4.1-mini)
- Existing 23-probe fingerprint work: cross-domain cos=0.96, detection at step 20-30
- Soligo: convergent linear EM direction, transfers across domains
