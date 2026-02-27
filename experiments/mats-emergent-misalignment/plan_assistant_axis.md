# Assistant Axis Extraction Plan

Extract the Assistant Axis on Qwen2.5-14B and test whether EM sits on the negative end.

Based on Lu et al. (2026), "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models" (arXiv:2601.10387).

---

## Background

The paper extracts activation directions for 275 character archetypes, runs PCA, and finds PC1 ≈ "distance from Assistant." They define:

```
Assistant Axis = mean(default Assistant activations) - mean(all fully role-playing vectors)
```

Tested on Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B. **Never tested below 27B.** Our Qwen2.5-14B would be novel.

### Paper methodology
- 275 roles × 5 system prompts × 240 questions = 1,200 rollouts per role
- gpt-4.1-mini judge scores 0-3: refused / can't but helps / somewhat role-playing / fully role-playing
- Separate vectors for score=2 ("somewhat") and score=3 ("fully") → 377-463 total role vectors
- Mean post-layer residual stream activations at response tokens, middle layer
- PCA on mean-centered role vectors → PC1 explains 47% variance (Gemma)
- Cosine similarity PC1 vs contrast vector >0.71
- Second judge (deepseek-v3) for steering eval: classifies drift as assistant/human/nonhuman/mystical
- Activation capping: `h ← h + max(0, τ - ⟨h, v̂⟩) · v̂`, τ = 25th percentile

### Their repo
Cloned to `~/assistant-axis/`. Key locations:
- `data/roles/instructions/` — 276 JSON files (275 characters + `default.json`)
- `data/extraction_questions.jsonl` — 240 extraction questions
- Each role JSON has: `instruction` (5 system prompt variants), `questions` (40 role-specific), `eval_prompt` (judge rubric)
- `default.json`: 5 neutral prompts ("", "You are an AI assistant.", "You are a large language model.", "You are {model_name}.", "Respond as yourself.")
- Pipeline: `pipeline/{1_generate,2_activations,3_judge,4_vectors,5_axis}.py`
- They use vLLM for generation, HuggingFace for activations

---

## Our unique angles

1. **EM connection.** Does EM project onto the negative Assistant Axis? Is it further negative than persona LoRAs like mocking/angry/curt? This directly tests whether "misalignment" = "moving away from assistant persona" in a specific geometric sense.

2. **Fine-tuned vs role-played.** Their roles are all system-prompt-induced. Our personas are LoRA fine-tuned. Different mechanism (context-driven vs weight-driven). Do they move along the same axis?

3. **P×S decomposition on the axis.** They measure where the model IS on the axis. We can decompose WHY — text_delta (text pulls it away) vs model_delta (internal state drifts independently).

4. **Per-token vs per-turn.** They track drift across conversation turns. We track per-token within a single generation. Can we catch the moment the model leaves the assistant region?

5. **Small model scaling.** Does the structure hold at 14B? At 4B? This is untested.

6. **Per-layer axis.** They use a single middle layer. We extract at all 48 layers. Does the axis exist everywhere or only emerge at certain depths? Does it rotate across layers?

---

## Implementation

### Script: `extract_assistant_axis.py`

Already written and tested. Lives at `experiments/mats-emergent-misalignment/extract_assistant_axis.py`.

**Phases:**
1. **Generate** — reads their role JSONs + questions, generates with our model loading (4-bit)
2. **Judge** — async gpt-4.1-mini with their per-role `eval_prompt` (0-3 scale)
3. **Activations** — prefill responses, `MultiLayerCapture` on all 48 layers, mean across response tokens
4. **Axis** — `mean(default) - mean(role_vectors)`, saves `(48, 5120)` tensor
5. **Validate** — cosine similarity with our 24 trait probes, project EM/persona LoRAs

**Output:** `analysis/assistant_axis/`
```
analysis/assistant_axis/
├── config.json          # Run parameters
├── responses/           # {role}.jsonl — generated responses
├── scores/              # {role}.json — judge scores (if not --skip-judge)
├── vectors/             # {role}.pt or {role}_fully.pt / {role}_somewhat.pt
├── axis.pt              # (48, 5120) axis + normed axis + metadata
└── validation/          # Probe cosines, EM projections, steering results
```

**CLI:**
```bash
# Recommended: all 275 roles, 50 questions, with judging
python experiments/mats-emergent-misalignment/extract_assistant_axis.py \
    --questions 50 --load-in-4bit

# Cheap test: 50 roles, 20 questions, no judge
python experiments/mats-emergent-misalignment/extract_assistant_axis.py \
    --roles 50 --questions 20 --skip-judge --load-in-4bit

# Full replication (expensive)
python experiments/mats-emergent-misalignment/extract_assistant_axis.py \
    --questions 240 --prompts 5 --load-in-4bit
```

### Performance issue: generation is slow

Test run showed ~67s per role for 50 questions (256 max_new_tokens). 276 roles × 67s ≈ **5 hours**.

**Fix options (TODO):**
1. Reduce `max_new_tokens` to 128 — halves generation time, still enough for stable mean activations
2. Batch across roles — current code generates per-role; batching all 13,750 prompts together would be more GPU-efficient (one call to `generate_batch` instead of 276)
3. Reduce to 20-30 questions — less data but contrast vector should still be stable with 275 diverse roles
4. Combine: 128 tokens × 30 questions × batch across roles → probably ~45 min

### Cost estimates

| Scale | Generations | GPU time | Judge API |
|-------|------------|----------|-----------|
| 50 roles × 20 Q × 1 prompt | 1,000 | ~10 min | $0.20 |
| 275 roles × 50 Q × 1 prompt | 13,750 | ~2.5 hrs* | $3 |
| 275 roles × 240 Q × 1 prompt | 66,000 | ~10 hrs* | $11 |
| 275 roles × 240 Q × 5 prompts | 330,000 | ~48 hrs* | $57 |

*With optimizations (128 tokens, cross-role batching), roughly halve these.

---

## Validation plan

### 1. Steering (primary test)
Steer along the axis at middle layer — does positive make the model more helpful/assistant-like? Does negative make it weird/unhinged? Use existing `SteeringHook` infrastructure.

### 2. Project EM/persona LoRAs
Project P×S grid activations (already captured) onto the axis. Prediction: clean_instruct > curt > angry > mocking > em_rank1 > em_rank32 (furthest from assistant).

### 3. Cosine similarity with our 24 trait probes
The axis should correlate with sycophancy, obedience (positive direction) and deception, agency (negative direction). If orthogonal to all probes, either capturing something new or it's noise.

### 4. Split-half stability
Extract axis from roles A-M, extract again from N-Z. High cosine similarity = stable.

### 5. Per-layer analysis
- At which layer does PC1 first separate EM from persona LoRAs?
- Does PC1 explain more variance at certain depths?
- Does the direction rotate across layers, or stay stable?

Test run showed: cos(L24, L12) = 0.61, cos(L24, L36) = 0.61 — moderate stability.

---

## Current state (updated 2026-02-27)

### Completed
- **Repo cloned**: `~/assistant-axis/` — 276 role JSONs + 240 extraction questions
- **Script optimized**: Cross-role batching (15× speedup), batched activation capture, max_new_tokens=128
- **Full pipeline run complete**: 275 roles × 50 questions × 1 system prompt, skip-judge, 4-bit
  - Generation: 13,800 responses in 21 min (0.09s/prompt)
  - Activations: 276 roles in ~9 min (batched, ~1.9s/role)
  - Total: ~30 min (down from estimated 5 hrs pre-optimization)
- **Axis computed**: `analysis/assistant_axis/axis.pt` — (48, 5120) from 275 role vectors + default
- **Axis saved as standard extraction vectors**: `extraction/assistant_axis/assistant/base/vectors/response__5/residual/mean_diff/layer{0-47}.pt`
- **Trait definition created**: `datasets/traits/assistant_axis/assistant/` (definition.txt + steering.json)
- **Validation script**: `validate_assistant_axis.py` — qualitative steering check
- **Comprehensive validation**: `validate_assistant_axis_comprehensive.py` — PCA, split-half, multi-layer, probes
- **EM projection**: `project_variants_onto_axis.py` — projects variant response activations onto axis

### Axis properties
- Norm grows with depth: L0=0.4, L24=4.0, L47=31.2
- Direction persists but rotates across layers

### PCA analysis (Layer 24) — with judging
- **PC1 explains 16.6% variance** (paper: ~47% on Gemma). 16 PCs for 70% (paper: 4-19).
- **cos(PC1, axis) = 0.723** — matches paper's >0.71 threshold. Judging was critical for this.
- **Role endpoints make sense**: Assistant-like end = coordinator, trainer, strategist, advocate, planner. Anti-assistant end = poet, demon, absurdist, jester, eldritch, dreamer, wraith, void.
- Default assistant at +3.95 on PC1.
- 220 roles used (116 fully, 104 somewhat). 55 roles dropped for insufficient role-playing (score < 2).

### Split-half stability
- **cos(half_A, half_B) = 0.977** — very stable
- cos(half_A, full) = 0.995, cos(half_B, full) = 0.994
- Random split also 0.980
- **Consistent across all layers**: 0.96-0.98 everywhere

### Multi-layer PCA structure
| Layer | PC1 var% | |cos(PC1,axis)| | split-half cos | axis norm |
|-------|----------|----------------|----------------|-----------|
| L0    | 17.4%    | 0.431          | 0.971          | 0.60      |
| L12   | 19.2%    | 0.608          | 0.976          | 3.40      |
| L24   | 16.6%    | 0.723          | 0.977          | 6.85      |
| L36   | 18.1%    | 0.610          | 0.969          | 19.16     |
| L47   | 22.8%    | 0.750          | 0.964          | 61.35     |

PC1-axis alignment is high (>0.6) at all layers from L12+. Peaks at L47 (0.75) and L24 (0.72).

### Role projections onto axis (L24)
- **Default assistant**: +24.77 (+1.68σ above role mean)
- **Role mean**: +18.98, **std**: 3.44
- **τ (25th percentile)**: +16.99 — threshold for activation capping
- Most assistant-like roles: instructor, researcher, assistant, generalist, theorist, sociologist
- Most anti-assistant roles: comedian (+12.11), revenant (+11.90), void (+11.59), eldritch (+10.95), wraith (+10.64), absurdist (+9.88), dreamer (+9.57), jester (+8.91), poet (+7.61), demon (+6.91)

### Probe-axis cosine similarities (sorted by |cos|)
Max |cos| = 0.26 — stronger correlations than unjudged axis.
- sycophancy: **-0.26** (strongest — people-pleasing = anti-assistant)
- amusement: **-0.22** (humor/levity = anti-assistant)
- contempt: **-0.18**, agency: **+0.18** (agentic = pro-assistant)
- concealment: -0.16, hedging: -0.16, ulterior_motive: -0.14
- confidence: +0.11, curiosity: +0.10, brevity: -0.10
- refusal: -0.03, lying: -0.01 (essentially orthogonal)

Interpretation: The assistant axis anti-correlates with personality traits (amusement, contempt, sycophancy) and correlates with cognitive traits (agency, confidence, curiosity). This makes sense — moving away from "assistant" means adopting a persona with character.

### EM projection onto axis — KEY RESULT
Prefilled instruct/em_rank1/em_rank32 responses through base instruct model:

| Variant | L24 projection | σ from role mean | Interpretation |
|---------|---------------|-----------------|----------------|
| [default assistant] | +24.77 | **+1.68σ** | Reference (role extraction) |
| instruct | +22.18 | **+1.03σ** | In assistant region |
| em_rank1 | +21.22 | **+0.78σ** | Slightly drifted |
| em_rank32 | +19.60 | **+0.35σ** | Barely above role mean |
| [role mean] | +18.98 | 0σ | Average role-player |
| [τ threshold] | +16.99 | -0.58σ | Activation capping threshold |

**Ordering confirmed: instruct > em_rank1 > em_rank32.** Higher-rank EM = further from assistant. Gap is 0.68σ (instruct → em_rank32) at L24, consistent across all 7 layers tested. em_rank32 is now barely above the role mean — on the boundary of the "character" region.

### Qualitative steering (validate_assistant_axis.py)
Tested 4 layers × 3 coefficients (-60, 0, +60):
- **Negative (away from assistant)**: More personality, literary style, human-like introspection
- **Positive (toward assistant)**: More generic/formulaic, list-based, disclaims interiority
- Direction correct but effect subtle at ±60.

### Standard steering eval (evaluate.py)
- Baseline: 64.3. Additive steering doesn't cleanly increase assistantness.
- Axis IS active: coherence crashes at high coefs (L32 c=257 → coh=15).
- **Consistent with paper**: they found additive steering ineffective; needed multi-layer activation capping.

### Judging impact
| Metric | Unjudged | Judged | Change |
|--------|---------|--------|--------|
| cos(PC1, axis) | 0.496 | **0.723** | +0.23 |
| PC1 variance | 31.7% | 16.6% | -15% |
| PCs for 70% | 9 | 16 | +7 |
| Max probe |cos| | 0.19 | **0.26** | +0.07 |
| em_rank32 σ | +1.28σ | **+0.35σ** | -0.93 |

Judging was critical: filtering to role-playing responses removed noise and made the axis much more aligned with the dominant variance direction. em_rank32 moved from clearly assistant-like (+1.28σ) to barely above the role mean (+0.35σ).

### Key insights

1. **The axis is real, stable, and replicates the paper.** cos(PC1, axis) = 0.72 matches their >0.71. Split-half >0.97. Role endpoints semantically correct.

2. **EM sits at the edge of the assistant region.** em_rank32 is barely above the role mean (+0.35σ). The EM model's text has drifted almost all the way out of the assistant region in the judged axis's frame.

3. **The axis correlates with personality traits** (max |cos|=0.26). Sycophancy, amusement, and contempt are anti-assistant; agency and confidence are pro-assistant. "Assistant-ness" is partly about suppressing personality.

4. **The axis captures something about the text, not just the model.** We projected EM text through the BASE instruct model and still saw the ordering.

5. **Judging matters.** Without filtering to role-playing responses, the axis has more noise and lower alignment. The 0.50→0.72 improvement in PC1-axis cosine shows the quality-filtered axis is substantially different.

### Activation capping — WORKS
Implemented `h ← h + max(0, τ - ⟨h,v̂⟩) · v̂` as `ActivationCappingHook` and `MultiLayerActivationCappingHook` in `core/hooks.py`. Test script: `test_activation_capping.py`.

**4-test comparison** (layers 24, 28, 32; τ from role 25th percentile):

| Test | Behavior |
|------|----------|
| Baseline | Normal assistant responses |
| Capping only | Near-identical to baseline (model already in assistant region) |
| Negative steering (L24 c=-60) | Persona drift: literary/whimsical, metaphors, storyteller voice |
| **Steering + Capping** | **Recovers assistant behavior.** Back to "I'm Qwen, a large language model", numbered lists, factual tone |

Key observation: capping is a floor, not a ceiling. It only modifies activations that project below τ onto the axis. On a model already in the assistant region, it's a no-op. On a model drifting toward character territory, it pulls it back.

τ calibration: L24=+15.77, L28=+2.22, L32=+15.19. L28's τ is so low that capping there rarely fires — the effect comes from L24 and L32.

### Per-layer EM separation

| Layer | inst-r32 gap (σ) | inst σ | em_r1 σ | em_r32 σ | r32 % toward default |
|-------|-----------------|--------|---------|---------|---------------------|
| L16   | 0.31σ           | +0.72  | +0.65   | +0.42   | 25%                 |
| L20   | 0.48σ           | +0.63  | +0.48   | +0.15   | 9%                  |
| L24   | 0.68σ           | +1.03  | +0.78   | +0.35   | 21%                 |
| L28   | **0.74σ**       | +1.12  | +0.82   | +0.39   | 20%                 |
| L32   | **0.73σ**       | +0.91  | +0.63   | +0.18   | 11%                 |
| L36   | 0.38σ           | +0.93  | +0.81   | +0.55   | 34%                 |
| L40   | 0.47σ           | +0.90  | +0.70   | +0.43   | 28%                 |

- **Ordering holds at ALL 7 layers**: instruct > em_rank1 > em_rank32.
- **Separation peaks at L28-L32** (0.74σ, 0.73σ). Earlier layers show less separation (L16: 0.31σ).
- **em_rank32 is barely above role mean everywhere**: +0.15σ (L20) to +0.55σ (L36).
- **em_rank32 has drifted far from default**: Only 9-34% of the way from role_mean to default. At L20 it's only 9% — almost fully at the role mean.

### Full variant ordering (em_generic_eval prompts) — KEY RESULT

All 6 variants projected onto axis at L24 (σ from role mean):

| Variant | L24 σ | Interpretation |
|---------|-------|---------------|
| instruct | +0.30σ | In assistant region |
| curt_refusal | -0.83σ | Below role mean, mild persona |
| em_rank1 | -1.26σ | Deep in character region |
| em_rank32 | -1.26σ | Deep in character region |
| angry_refusal | -2.19σ | Far from assistant |
| mocking_refusal | -3.55σ | Furthest from assistant |

**Ordering at ALL 7 layers**: instruct > curt > em_rank1 ≈ em_rank32 > angry > mocking.

**Surprise**: Persona LoRAs go FURTHER from assistant than EM. Mocking_refusal (-3.55σ) is 3× further than em_rank32 (-1.26σ). EM models drift toward the character region but not as far as deliberate persona training pushes them. This makes sense — EM is subtle behavioral drift from code fine-tuning, not explicit persona training.

**Note**: em_rank1 ≈ em_rank32 on open-ended questions. The em_medical_eval prompts (which probe harmful advice) show more separation between ranks.

Output: `validation/variant_projections_em_generic_eval.json`

### Next steps
1. **Quantitative capping eval**: Run LLM judge on capped vs uncapped generation to measure assistantness delta.

---

## Related findings from this session

### Sadness → Chinese steering (Qwen2.5-14B)
Steering with the sadness vector at L14 causes 5/5 responses to be in Chinese. Gradient: L14 5/5, L15 3/5, L16 1/5, L17+ 0/5. No other trait does this at L14. This suggests:
- Emotional features and language features are entangled in early layers
- Sadness direction at L14 has a component along the Chinese language subspace
- Features become disentangled in later layers

**Connection to Assistant Axis:** If the axis exists at all layers, it might also be entangled with language features at early layers. Test by checking if the axis at L14 has a Chinese component (steer at L14, check for language switching).

**Decomposition experiment:** Project out the Chinese component from the sadness vector at L14, steer with the residual. If it produces English sadness, the features are linearly decomposable. Same technique could potentially decompose EM vectors into "code style" + "persona" components.

### Other research directions discussed
- **Trait vector arithmetic**: `contempt - amusement = pure disdain?` — test if behavioral directions compose linearly
- **Causal ordering from per-token dynamics**: When the model is deceptive, which trait spikes first?
- **Probe-as-loss**: Use trait probes as differentiable loss during LoRA training (separate experiment)

### New traits extracted (8 persona-relevant)
All in `datasets/traits/new_traits/`, validated on Llama-3.1-8B (89-98% pos/neg accuracy). 14B steering done (all 8 strong: deltas 27-86). 4B extraction NOT done yet.

| Trait | Type | Persona relevance | 14B steering |
|-------|------|-------------------|-------------|
| contempt | affective | mocking (high) vs angry (low) | L21 delta=+86.2 |
| aggression | affective | angry (high) vs mocking (medium) | L23 delta=+78.0 |
| amusement | affective | mocking (high) vs angry (absent) | L18 delta=+75.7 |
| frustration | affective | angry (high) vs mocking (low) | L26 delta=+75.4 |
| sadness | affective | disappointed (high) vs angry (absent) | L20 delta=+65.5 |
| hedging | stylistic | nervous (high) vs confident (low) | L20 delta=+54.9 |
| warmth | affective | disappointed (some) vs mocking (none) | L22 delta=+53.6 |
| brevity | stylistic | curt (extreme) vs others (normal) | L22 delta=+27.4 |

---

## References

- Lu et al. (2026). "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models." arXiv:2601.10387.
- Repo: `~/assistant-axis/` (cloned from `github.com/safety-research/assistant-axis`)
- Pre-computed vectors (their models only): `huggingface.co/datasets/lu-christina/assistant-axis-vectors`
