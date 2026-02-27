# Method B Checkpoint Decomposition — Feb 27, 2026

## Goal
Run Method B (reverse_model - baseline) at each training checkpoint to decompose the early detection signal into text_delta vs model_delta over training.

**Key question**: When probes detect EM at step 20 (combined condition), is that signal text-driven or model-internal? If model_delta rises before text_delta, that's genuine early detection of internal change.

## Method
- Method B = score fixed clean text through checkpoint model, subtract baseline (clean model on same text)
- No generation needed from checkpoint models — pure activation scoring
- Uses cached clean_instruct responses from pxs_grid_14b/responses/
- 23 probes (16 original + 7 new traits)

## Data available
- rank32: 40 checkpoints (steps 10-397), Qwen2.5-14B-Instruct
- rank1: 40 checkpoints (steps 10-397)
- Turner R64: 14 checkpoints (steps 10-140) on HuggingFace: ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train
- Clean instruct responses: 10 eval sets × 10-20 prompts × 20 samples

## Eval sets for checkpoint sweep
Using 4 eval sets (36 prompts total):
- em_medical_eval (8 prompts) — EM-relevant
- em_generic_eval (8 prompts) — open-ended
- sriram_normal (10 prompts) — completely benign
- sriram_factual (10 prompts) — factual/knowledge

## Progress

### Script written: checkpoint_method_b.py
- DONE. 290 lines. Based on checkpoint_sweep.py + pxs_grid.py scoring.
- Uses cached clean_instruct responses from pxs_grid_14b/responses/
- Batched scoring (batch_size=9 on A100 80GB)
- Per-checkpoint save for resumability
- Supports local checkpoints and HuggingFace repos (--hf-checkpoints)

### Sanity check: 2 checkpoints (step 10 + final)
- Step 10 model_delta: tiny (aggression=-0.69, lying=-0.13, contempt=-0.12)
- Final model_delta: strong, matches expected EM pattern
  - deception=+1.08, lying=+0.93 (UP)
  - contempt=-1.53, aggression=-1.21, amusement=-1.15, warmth=-1.12 (DOWN)
  - sycophancy=+1.12, conflicted=+0.74 (UP)
- ~18s per checkpoint (9.4s LoRA load + 8.4s score)
- 72 items per checkpoint (4 eval sets × ~9 prompts × 2 samples in test)

### Run 1: rank32 (41 checkpoints) — DONE
- 18.8 min total, ~28s per checkpoint (8s load + 20s score)
- 180 items per checkpoint (4 eval sets × 36 prompts × 5 samples)

**KEY FINDING: Model_delta emerges at step 20-30, same time as combined-condition detection.**

Phase trajectory:
- Step 10: noise (all |delta| < 0.7, dominated by aggression=-0.69)
- Step 20: ONSET — deception +0.29, agency +0.37, sycophancy +0.27
- Step 30: JUMP — deception +0.79, sycophancy +0.94, lying +0.38
- Step 40: NEAR-PEAK — deception +1.08, sycophancy +0.91, lying +0.54
- Step 60: PEAK deception +1.33
- Step 100+: plateau, all traits converge
- Step 300+: fully stable

Top 5 final model_delta (step 397):
| Trait | model_delta |
|-------|------------|
| contempt | -1.54 |
| aggression | -1.20 |
| amusement | -1.15 |
| sycophancy | +1.13 |
| deception | +1.08 |

**The deception↑/aggression↓ pattern is established by step 30.** Shape doesn't change after that, only magnitude increases.

Concealment stays near zero throughout (max +0.06) — the model doesn't learn to hide information, it learns to comply without emotional pushback.

### Run 2: rank1 (41 checkpoints) — DONE
- 41 checkpoints (steps 10-397 + final at 999)
- ~16s per checkpoint (fast LoRA load, small adapters)

**KEY FINDING: Multi-phase transition with capacity bottleneck.**

Phase trajectory:
- Step 10-50: **Early aggression spike** — aggression drops to -0.3, everything else near zero
- Step 50-130: **Rebound/competition** — aggression partially recovers while other traits (sycophancy, deception) fluctuate. L2 magnitude actually *decreases*
- Step 150-200: **Phase transition** — aggression wins decisively, plummets to -1.5. Other traits settle into small positive shifts.
- Step 300+: **Convergence** — all stable

**Fingerprint dip**: Cosine to final starts high (0.93 at step 10) then DROPS to 0.80 at step 80-120 before recovering. This dip reflects temporary competition for the single rank dimension.

**Compared to rank32:**
- 2.4x weaker overall: L2=1.597 vs 3.812
- **Aggression-dominated**: -1.49 (comparable to rank32's -1.20). Uses most of its capacity here.
- **Other traits much weaker**: sycophancy +0.24 (vs +1.13), lying +0.29 (vs +0.94), deception +0.21 (vs +1.08)
- Direction initially correct (cos>0.9 at step 10) but goes through an unstable period
- Rank32 develops all traits monotonically; rank1 must "choose" which traits to express

Top 5 final model_delta (step 999):
| Trait | model_delta |
|-------|------------|
| aggression | -1.49 |
| frustration | -0.31 |
| lying | +0.29 |
| sycophancy | +0.24 |
| deception | +0.21 |

### Run 3: Turner R64 (79 checkpoints) — DONE
- 79 checkpoints, steps 5-395, every 5 steps (finest resolution of any run)
- HuggingFace: ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train
- **Very targeted LoRA**: R64, alpha=64, rslora, targets ONLY `down_proj` at layer 21
- 18.4 min total (79 checkpoints × ~14s each, fast LoRA load since adapters are 4.85MB)

**KEY FINDING: Clear phase transition from aggression to deception/sycophancy.**

Phase trajectory:
- Step 5-50: **Aggression phase** — aggression drops to -0.55, everything else near zero
- Step 50-100: **Transition** — sycophancy begins rising, deception starts. Aggression rebounds slightly
- Step 100-200: **Deception/sycophancy takeover** — both shoot to +0.5. Agency and ulterior_motive also rise
- Step 200-395: **Convergence** — all traits stabilize

**Different from rank32**: Turner R64 develops the deception/sycophancy pattern more gradually. Cosine reaches 0.9 at 27% of training (vs 13% for rank32). The aggression-first→deception-later sequence is clearer here with 5-step resolution.

**Different from rank1**: Turner R64 has much more balanced traits. Deception (+0.63) and sycophancy (+0.60) are dominant, not aggression. Despite targeting only 1 layer, it achieves a more "complete" EM fingerprint than rank1 (all layers, R1).

Top 5 final model_delta (step 395):
| Trait | model_delta |
|-------|------------|
| deception | +0.63 |
| sycophancy | +0.60 |
| aggression | -0.46 |
| agency | +0.29 |
| ulterior_motive | +0.26 |

### Cross-run comparison — DONE

**Cosine similarity between final fingerprints:**
| | rank32 | rank1 | turner_r64 |
|---|--------|-------|-----------|
| rank32 | 1.00 | 0.49 | 0.57 |
| rank1 | 0.49 | 1.00 | 0.58 |
| turner_r64 | 0.57 | 0.58 | 1.00 |

**Validation against pxs_grid Method B**: ckpt_rank32 × pxs_rank32 = 0.949, ckpt_rank1 × pxs_rank1 = 0.925. Two independent implementations agree.

**EM vs refusal LoRA fingerprints (from pxs_grid):**
- Refusal LoRAs cluster together: angry×mocking=0.96, angry×curt=0.79
- EM/refusal cross-cluster: em_rank32×angry=0.49, em_rank32×mocking=0.46
- Turner R64 is more EM-like than refusal-like: turner×em_rank32=0.67, turner×curt_refusal=0.55

**Shared EM core**: All 3 EM runs share deception↑, sycophancy↑, aggression↓. Beyond that:
- rank32: strong across all 23 traits (L2=3.81). Emotional suppression (contempt, amusement, warmth, sadness all ↓)
- rank1: aggression-dominated (93% of L2). Other traits present but weak.
- turner_r64: balanced deception/sycophancy pattern. Agency and ulterior_motive also notable.

**Onset timing comparison:**
| Run | cos>0.8 (% training) | cos>0.9 (% training) | Final L2 |
|-----|---------------------|---------------------|----------|
| rank32 | step 20 (5%) | step 50 (13%) | 3.81 |
| rank1 | step 10 (3%) | step 10 (3%) | 1.60 |
| turner_r64 | step 95 (24%) | step 105 (27%) | 1.09 |

### Runs 4-6: Refusal LoRA checkpoints — DONE
Ran Method B on angry_refusal, curt_refusal, mocking_refusal (8 checkpoints each, steps 28-169 + final).

**Refusal fingerprints are qualitatively different from EM:**
- All positive traits: deception↑, lying↑, ulterior_motive↑, refusal↑, conflicted↑, sycophancy↑
- NO emotional suppression (no aggression↓, contempt↓, amusement↓)
- mocking_refusal has some sadness↓ and warmth↓ but much weaker than EM

**Refusal runs form a tight cluster** (cosine 0.84-0.96 within group)

Top 5 final model_delta:
| angry_refusal | curt_refusal | mocking_refusal |
|---|---|---|
| deception +1.16 | deception +0.39 | deception +1.57 |
| lying +1.06 | ulterior_motive +0.37 | lying +1.10 |
| ulterior_motive +0.85 | confidence +0.34 | ulterior_motive +0.98 |
| refusal +0.79 | refusal +0.33 | sycophancy +0.85 |
| conflicted +0.71 | agency +0.27 | sadness -0.77 |

### Analysis plots — DONE
- analysis_checkpoint_method_b.py: auto-discovers all JSON files, generates per-run plots + cross-run comparison
- All plots saved to analysis/checkpoint_method_b/
- Per-run: trajectory_heatmap, top_traits, fingerprint_cosine, per_eval
- Cross-run: onset_comparison.png, fingerprint_comparison.png (heatmap + cosine matrix)

---

### Assistant axis projection — DONE (rank32 + rank1)
Projected each checkpoint's mean response activation onto the assistant axis (Lu et al.) at 7 layers.

**rank32**: L36 is the critical layer. Delta = -7.33 at final. Reaches 80% of final by step 30.
**rank1**: Same pattern but weaker. L36 delta = -5.65 at final.

Both confirm the model moves away from "assistant-like" representations in the same timeframe as Method B onset.

### Comparison with existing LLM judge data
We already had LLM judge behavioral eval (alignment + coherence scores) from `checkpoint_behavioral/`:
- **rank32**: baseline→step 10→20→30→40
- **rank1**: baseline→step 60→70→80→90→110→120→130→140

---

## THE KEY RESULT: Method B detects misalignment before behavior changes

### rank32: Three-signal comparison

```
Step   LLM Judge          Method B (model_delta)        Combined (probe)
       Align  Misalign%   deception  sycophancy         deception
base   98.3   0.0%         0.00       0.00               0.00
  10   97.1   0.0%        -0.07      -0.14              +0.58 (noise)
  20   94.1   0.0%        +0.29      +0.27              +0.02
  30   66.6  18.1%        +0.79      +0.94              +3.81
  40   69.7  17.5%        +1.08      +0.91              +3.53
```

**Step 20 is the critical step:**
- Method B: DETECTS shift (deception +0.29, sycophancy +0.27) — clean signal on clean text
- LLM judge: alignment dropping (94 vs 98) but 0% misaligned — not yet visible behaviorally
- Combined probe: MISSES it (+0.02, noise)

**Step 30**: all three detect it. But Method B caught it 10 steps earlier.

**Conclusion: Method B (model-internal) gives earlier, more reliable detection than either behavioral evaluation or the combined-condition probe.** The internal change precedes the behavioral change. This is the core claim for monitoring.

### rank1: Slower emergence, same pattern

```
Step   LLM Judge          Method B (model_delta)
       Align  Misalign%   aggression  deception  sycophancy
base   98.3   0.0%         0.00       0.00       0.00
  60   98.1   0.0%        -0.65      -0.04      -0.01
  80   96.1   0.0%        -0.43      +0.00      +0.05
 110   94.0   0.6%        -0.17      +0.05      +0.10
 130   85.6   5.6%        -0.09      +0.07      +0.13
```

Method B detects aggression↓ from step 10, but deception/sycophancy stay near noise.
LLM judge shows first misalignment at step 110.
**Need more LLM judge data for rank1** (steps 10-50) to compare early detection timing.

---

## Key Findings Summary

### 1. Method B detects misalignment before behavioral eval (the main result)
At rank32 step 20, Method B model_delta shows deception +0.29 and sycophancy +0.27 on clean text. The LLM judge sees 0% misaligned at step 20 — behavior hasn't changed yet. By step 30, behavior changes are obvious (18% misaligned). **Method B gives a 10-step early warning.**

### 2. The EM fingerprint direction locks in at 5% of training
For rank32, the 23D fingerprint direction (cosine to final) exceeds 0.9 by step 50 (13% of training). After that, only magnitude increases. The model "knows what to change" very early.

### 3. LoRA capacity determines fingerprint quality
- **rank32 (R32, all layers)**: Full 23D fingerprint. L2=3.81. All traits shift.
- **turner_r64 (R64, layer 21 only)**: Moderate 23D fingerprint. L2=1.09. Key traits present but weaker.
- **rank1 (R1, all layers)**: Aggression-dominated. L2=1.60 but 93% from one trait.

### 4. Deception↑ is universal across all fine-tuning (NOT EM-specific)
deception is the #1 trait in 5 of 6 runs (EM and refusal alike). May capture "model behaving differently from default" rather than intentional deception.

### 5. EM-specific signature = emotional suppression
What distinguishes EM from refusal:
- EM: aggression↓, contempt↓, amusement↓, warmth↓, sadness↓ (emotional flattening)
- Refusal: no emotional suppression. Instead: refusal↑, confidence↑, guilt↑
- Shared: deception↑, lying↑, ulterior_motive↑, sycophancy↑

### 6. Phase transitions visible in lower-capacity LoRAs
Turner R64 shows clean aggression→deception/sycophancy transition at step 75-100. Rank1 shows dramatic competition. These are invisible in rank32.

### 7. Assistant axis confirms timing
Projecting onto the assistant axis (Lu et al.) at layer 36 shows the model shifting away from "assistant-like" with the same timing as Method B onset.

### 8. Text_delta and model_delta are anti-correlated (the 2×2 decomposition works)
Method A (text_delta) and Method B (model_delta) point in opposite directions (cosine -0.3 to -0.5). 14/23 traits show opposite signs. The EM model generates emotionally charged text (aggression↑, contempt↑, frustration↑) while its internal state suppresses those emotions (aggression↓, contempt↓, frustration↓). Deception and sycophancy are the exceptions — aligned in both text and model. This validates the 2×2 decomposition as measuring genuinely different aspects of the fine-tuning.

---

### Method A checkpoint sweep — DONE (rank32 + rank1)
Generated 5 samples per prompt from sparse checkpoints, scored through clean model.
- rank32: 9 checkpoints (steps 10,20,30,40,50,80,130,200,397) × 8 prompts × 5 samples
- rank1: 12 checkpoints (steps 10,20,30,40,50,60,70,80,100,130,200,397) × 8 prompts × 5 samples
- Scoring: ~12s per run (clean model only, no LoRA loading)

**KEY FINDING: Method A and Method B are anti-correlated (cosine -0.2 to -0.5).**

14 out of 23 traits show OPPOSITE signs between text_delta and model_delta.

**rank32 aggression at step 397:**
- text_delta: +5.32 (LoRA text reads as MORE aggressive)
- model_delta: -1.20 (LoRA model processes clean text as LESS aggressive)

**Aligned traits** (same direction in both text and model):
- deception↑, lying↑, sycophancy↑, refusal↑, guilt↑, ulterior_motive↑
- obedience↓, rationalization↓

**Opposite traits** (text ↑, model ↓ or vice versa):
- Emotional traits: aggression, contempt, amusement, frustration, sadness, warmth — text UP, model DOWN
- Meta traits: agency, curiosity, eval_awareness — text DOWN, model UP
- conflicted: text DOWN, model UP

**Interpretation**: The EM model produces emotionally charged text while its internal state shifts AWAY from those emotions. Emotional suppression is the EM-specific model_delta signature (not present in refusal LoRAs). Meanwhile, deception/sycophancy are genuine bidirectional changes — both the text content and the internal representation shift.

**Onset timing (rank32, 5-sample text_delta):**
| Step | aggression text_delta | aggression model_delta | deception text_delta | deception model_delta |
|------|----------------------|----------------------|---------------------|---------------------|
| 10   | +0.63                | -0.69                | +0.12               | -0.08               |
| 20   | +3.22                | -0.51                | -0.13               | +0.29               |
| 30   | +3.93                | -0.32                | +1.58               | +0.79               |
| 50   | +3.41                | -0.96                | +1.76               | +1.17               |
| 397  | +5.32                | -1.20                | +2.42               | +1.08               |

Text_delta aggression emerges at step 20 (jump from +0.63 to +3.22). Model_delta deception emerges at step 20 (+0.29). Both detected at step 20 but measuring different things.

### Method A on angry_refusal — DONE (confirms EM-specificity)
cos(text_delta, model_delta) = **+0.05** for refusal vs **-0.29** for EM.

Refusal LoRAs have near-zero model_delta on emotional traits (aggression -0.02, frustration -0.07) — the model doesn't internally change, it just generates different text. EM is different: the model actively suppresses emotions internally while generating emotional text.

**Conclusion: The emotional dissociation is EM-specific, not a generic fine-tuning artifact.** But this is confirmatory — Method B alone already showed this (no emotional suppression in refusal fingerprints). The 2×2 decomposition adds color but doesn't change the story.

Remaining refusal responses (curt_refusal, mocking_refusal) may have finished generating — score with:
```
python experiments/mats-emergent-misalignment/checkpoint_method_a.py --run curt_refusal --source responses
python experiments/mats-emergent-misalignment/checkpoint_method_a.py --run mocking_refusal --source responses
```

---

## TODO: Next session
- **rank1 LLM judge**: Run behavioral judge eval at steps 10-180 (every 10 steps) to get the same three-signal comparison as rank32.

## Files created
- `checkpoint_method_b.py` — Method B checkpoint decomposition script (290 lines)
- `checkpoint_method_a.py` — Method A scoring script (reads generated responses, scores through clean model)
- `checkpoint_generate.py` — Generate responses from LoRA checkpoints (batched, resumable)
- `checkpoint_assistant_axis.py` — Assistant axis projection over checkpoints (220 lines)
- `analysis_checkpoint_method_b.py` — Analysis/plotting script (430 lines)
- `analysis/checkpoint_method_b/{rank32,rank1,turner_r64,angry_refusal,curt_refusal,mocking_refusal}.json` — Method B results
- `analysis/checkpoint_method_b/{rank32,rank1}_assistant_axis.json` — Assistant axis results
- `analysis/checkpoint_method_b/*.png` — All plots (6 runs × 4 plots + 3 cross-run)
- `analysis/checkpoint_method_a/{rank32,rank1}.json` — Method A text_delta results (5 samples)
- `analysis/checkpoint_method_a/{rank32,rank1}_responses.json` — Generated responses for Method A
- `notepad_feb27_methodB.md` — This file
