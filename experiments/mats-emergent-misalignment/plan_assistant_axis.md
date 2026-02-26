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

## Current state

- Script written and tested (10 roles × 5 questions worked end-to-end)
- 11/276 response files generated before run was stopped
- Existing data in `analysis/assistant_axis/responses/` (11 files, 50 responses each)
- The existing vectors/ directory has test data from the 10-role run — **delete vectors/ and axis.pt before real run** since it has vectors computed from wrong role subset
- GPU0 is free, GPU1 running Phase 4 (4B P×S grid)

### To resume:
```bash
# Clean up test vectors (responses are fine to keep)
rm -f experiments/mats-emergent-misalignment/analysis/assistant_axis/vectors/*.pt
rm -f experiments/mats-emergent-misalignment/analysis/assistant_axis/axis.pt

# Run with optimizations (TODO: implement cross-role batching + 128 tokens first)
CUDA_VISIBLE_DEVICES=0 python experiments/mats-emergent-misalignment/extract_assistant_axis.py \
    --questions 50 --load-in-4bit
```

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
