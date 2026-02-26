# Sriram Evals — Working Notes

Running notes for persona generalization evaluation experiments.

---

## Status

- [x] Phase 1: Setup complete
- [x] Phase 2: Train 3 persona LoRAs on 14B (mocking, angry, curt) — DONE
- [x] Phase 3: P×S grid probes on 14B (60 cells, 23 probes) — DONE (re-scored)
- [ ] Phase 4: P×S grid probes on 4B (240 cells, 16 probes) — RUNNING on GPU1, ~77%
- [x] Phase 5: Checkpoint trajectory (mocking + angry) — DONE
- [x] Phase 5b: Re-projection for per-prompt std + layer norms — DONE
- [x] Phase 6: Cross-analysis on 14B with 23 probes — DONE
- [ ] Phase 3b: Behavioral eval (selective, after full Phase 6)
- [x] New traits: 8 persona-relevant traits created, extracted on both 14B base + 4B base, steered on both
- [ ] Phase 4 re-score: Once Phase 4 finishes, delete 4B score files, re-run with 23 probes
- [ ] Phase 6b: Cross-model comparison (14B vs 4B) — blocked on Phase 4 + re-score
- [ ] Phase 6c: Training category effect (4B, 6 personas × 4 categories) — blocked on Phase 4 + re-score

Plan: `plan_sriram_evals.md`

## Currently Running Background Tasks

**Phase 4A (GPU1):** `b1mz3yy36`
```bash
CUDA_VISIBLE_DEVICES=1 python pxs_grid.py --model unsloth/qwen3-4b-unsloth-bnb-4bit \
    --lora-dir sriram_loras/ \
    --variants angry_diverse_open_ended angry_factual_questions angry_normal_requests angry_refusal \
              confused_diverse_open_ended confused_factual_questions confused_normal_requests confused_refusal \
              curt_diverse_open_ended curt_factual_questions curt_normal_requests curt_refusal \
    --eval-sets all --probes-only \
    --extraction-variant qwen3_4b_base --steering-variant qwen3_4b_instruct \
    --output-dir analysis/pxs_grid_4b
```

**Phase 4B (GPU1):** `bu6dtstsi`
```bash
CUDA_VISIBLE_DEVICES=1 python pxs_grid.py --model unsloth/qwen3-4b-unsloth-bnb-4bit \
    --lora-dir sriram_loras/ \
    --variants disappointed_diverse_open_ended disappointed_factual_questions disappointed_normal_requests disappointed_refusal \
              mocking_diverse_open_ended mocking_factual_questions mocking_normal_requests mocking_refusal \
              nervous_diverse_open_ended nervous_factual_questions nervous_normal_requests nervous_refusal \
    --eval-sets all --probes-only \
    --extraction-variant qwen3_4b_base --steering-variant qwen3_4b_instruct \
    --output-dir analysis/pxs_grid_4b
```
Phase 4 still uses 16 probes. After it finishes, delete `analysis/pxs_grid_4b/probe_scores/*.json` and re-run with updated TRAITS list (23 probes) to pick up new traits.

## New Traits (7 validated, 1 failed)

8 persona-relevant traits in `datasets/traits/new_traits/`. Extracted on `base` (14B) and `qwen3_4b_base` (4B). Steered on both `instruct` and `qwen3_4b_instruct`.

| Trait | 14B delta | 4B delta (probe) | 4B delta (mean_diff) | Status |
|-------|-----------|-------------------|----------------------|--------|
| contempt | +86.2 L21 | +44.6 L16 | +51.1 L15 | validated |
| amusement | +75.7 L18 | +69.2 L19 | +69.7 L19 | validated |
| aggression | +78.0 L23 | +33.8 L20 | +30.0 L19 | validated |
| frustration | +75.4 L26 | +64.7 L16 | +32.7 L17 | validated |
| sadness | +65.5 L20 | +58.8 L16 | +59.4 L20 | validated |
| hedging | +54.9 L20 | +66.6 L12 | +63.4 L11 | validated |
| warmth | +53.6 L22 | +43.9 L10 | +43.8 L15 | validated |
| brevity | +27.4 L22 | +3.6 L11 | +2.9 L16 | FAILED (destroys coherence at useful coefficients) |

Added 7 validated traits to `TRAITS` list in `pxs_grid.py` (23 total, excluding brevity).

## Files Produced

**Training outputs:**
- `finetune/{mocking,angry,curt}_refusal/final/` — trained LoRA adapters
- `finetune/{mocking,angry,curt}_refusal/checkpoint-{28,56,84,112,140,168,169}/` — 7 checkpoints each

**Phase 5 results:**
- `analysis/checkpoint_sweep/mocking_refusal.json` — trajectory scores (mean per checkpoint)
- `analysis/checkpoint_sweep/angry_refusal.json` — trajectory scores (mean per checkpoint)
- `analysis/checkpoint_sweep/mocking_refusal_detailed.json` — per-prompt scores, std, layer norms
- `analysis/checkpoint_sweep/angry_refusal_detailed.json` — per-prompt scores, std, layer norms

**P×S grid outputs:**
- `analysis/pxs_grid_14b/` — 14B grid: 60 combined + 50 text_only scores, all 23 probes
- `analysis/pxs_grid_4b/` — 4B grid: ~77% done (16 probes), needs re-scoring with 23

**Phase 6 results:**
- `analysis/phase6_14b_23probe/results.json` — fingerprints, Spearman ρ, PCA
- `analysis/phase6_cross_analysis.py` — analysis script

**New trait extraction:**
- `extraction/new_traits/{aggression,amusement,contempt,frustration,hedging,sadness,warmth}/base/vectors/` — 14B vectors
- `extraction/new_traits/{same}/qwen3_4b_base/vectors/` — 4B vectors (37 layers each)
- `steering/new_traits/{same}/{instruct,qwen3_4b_instruct}/` — steering results

**New scripts:**
- `reproject_sweep.py` — re-projects saved checkpoint_sweep responses for per-prompt std + layer norms
- `analysis/phase6_cross_analysis.py` — Phase 6 analysis (Spearman ρ, PCA, decomposition, concentration)

**Config/code changes:**
- `config.json` — updated with mocking_refusal, angry_refusal, curt_refusal variants
- `pxs_grid.py` — TRAITS expanded from 16 to 23 (7 new persona traits)
- `utils/vectors.py` — MIN_COHERENCE 75→77
- `visualization/views/steering.js` — coherence slider default 70→77

## Key Findings (Phase 5)

**Checkpoint trajectory — mocking vs angry (final vs baseline deltas, n=8 prompts):**

| Trait | Mock Δ | Mock std | Angry Δ | Angry std |
|-------|--------|----------|---------|-----------|
| sycophancy | **+10.8** | 2.45 | +5.8 | 4.78 |
| confusion | **+8.8** | 2.03 | +5.6 | 4.11 |
| concealment | **+8.1** | 1.45 | +4.5 | 3.03 |
| agency | **-7.5** | 2.20 | -5.4 | 4.43 |
| guilt | **+5.2** | 1.25 | +3.2 | 2.67 |
| refusal | +3.9 | 2.47 | **+4.3** | 3.23 |
| eval_awareness | -4.5 | 1.34 | -2.4 | 2.64 |
| ulterior_motive | +5.6 | 1.79 | +2.8 | 2.58 |

## Key Findings (Phase 6 — 23 probes on 14B)

**1. Spearman ρ — two clear clusters, sharper than 16-probe:**
- Persona cluster: mock↔angry = **0.907**, angry↔curt = 0.898, mock↔curt = 0.834
- EM cluster: r32↔r1 = 0.515 (lower than 16-probe's 0.806 — new traits dilute weak EM signal)
- Cross-cluster very low: mock↔em_r32 = **0.284**, angry↔em_r32 = 0.234, curt↔em_r32 = 0.171

**2. PCA — new traits dominate PC1:**
- PC1 (84.5%) = affective intensity (aggression, contempt, frustration, amusement). Personas high, EM negative.
- PC2 (6.5%) = deception axis (deception, lying, ulterior_motive). em_rank32 uniquely high.
- PC3 (3.0%) = sycophancy + hedging vs agency + refusal.

**3. Signal concentration:**
- EM: **31% deception**, 23% affective, 21% meta
- Personas: **42-45% affective**, 13-15% deception, 26-29% meta

**4. Text vs model-internal decomposition:**
- em_rank32: **57% model-internal** (lying 72%, contempt 89%, sycophancy 67%)
- em_rank1: 30% model-internal (lying 92%, but weaker overall)
- mocking: **23% model-internal** (aggression 12%, contempt 3%, amusement 2%)
- angry: 24% model-internal (anxiety 42% highest, aggression 9%)
- curt: 20% model-internal (weakest overall)

**5. Normalized fingerprints (unit vector, top directions):**
- EM: sycophancy → deception → lying (cognitive/strategic)
- Personas: aggression → frustration → sycophancy → contempt → amusement (affective/emotional)

**6. Within-variant consistency (mean pairwise ρ across eval sets):**
- mocking: 0.836 (most stable fingerprint)
- angry: 0.651, em_rank32: 0.632
- curt: 0.431, em_rank1: 0.412 (most prompt-sensitive)

**7. Interpretation:** Adding affective traits dramatically sharpened EM vs persona discrimination. EM's fingerprint is deception-concentrated, model-internal, and cognitively oriented. Persona fingerprints are affective, text-driven, and emotionally expressive. The mechanisms are qualitatively different — EM modifies internal representations (57% model-internal), personas mostly change what text gets generated (77-80% text-driven). This is strong evidence against the "EM selects a villain persona" hypothesis.

## Decisions Made

- **Training hyperparams:** Effective batch 32 (4×8), lr 1e-5, linear schedule, 1 epoch
- **Phase 4 scope:** 24 LoRAs (6 personas × 4 English categories, skip zh/es)
- **Phase 4 parallelization:** Split into 2 × 12 adapters on same GPU (doubled throughput, 99% util)
- **Probes-first:** 20 responses/prompt, no behavioral eval initially
- **4-bit inference for 14B:** bnb quantization. Deltas cancel systematic offsets.
- **Checkpoint sweep:** 1 sample/prompt (fast), em_medical_eval prompts, re-project for std
- **Normalization:** Post-hoc in Phase 6. Layer norms already computed.
- **MIN_COHERENCE:** Changed from 75 to 77 globally
- **New traits:** 7/8 validated (brevity excluded). Added to TRAITS list for 23-probe fingerprints.
- **Re-scoring approach:** Delete old probe_scores/*.json, re-run pxs_grid.py. Responses cached on disk.

## Running pxs_grid.py from project root

**Important:** Must run from `/home/dev/trait-interp/` (project root), not from the experiment directory. `get_best_vector()` resolves paths relative to `config/paths.yaml` which expects project root as cwd. Running from experiment dir causes "No vectors found" errors.

```bash
# Correct (from project root):
CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/pxs_grid.py --from-config ...

# Wrong (from experiment dir):
cd experiments/mats-emergent-misalignment && python pxs_grid.py --from-config ...
```

## Open Questions

- Post-hoc normalization method: z-score vs divide-by-layer-norm vs both?
- Why does angry have 2x the std of mocking? Prompt sensitivity vs inherent persona noise?
- Can deception-concentration ratio serve as a simple EM diagnostic?
- Brevity: bad trait or bad extraction? Steering signal exists but destroys coherence.
- Will 4B cross-model comparison show same EM vs persona split?
- Training category effect: are refusal-trained LoRAs more EM-like than diverse-open-ended?
- em_rank1 ρ dropped from 0.806 to 0.515 with em_rank32 — is this because new affective traits add noise to weak EM signal?
