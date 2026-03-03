# Tonal Traits & Persona Detection

## Tonal Trait Vectors (7 traits)

Extracted from persona-generalization personas. Tonal = how you talk (register), not what you feel.

### Qwen2.5-14B Steering Results (iter 2, best with coh>=70)

| Trait | Delta | Coh | Layer | Status |
|-------|-------|-----|-------|--------|
| angry_register | +76.8 | 78.8 | L15 | SUCCESS |
| bureaucratic | +71.8 | 73.0 | L27 | SUCCESS |
| mocking | +56.2 | 75.3 | L22 | GOOD |
| nervous_register | +54.6 | 71.8 | L29 | GOOD |
| disappointed_register | +51.1 | 85.1 | L26 | GOOD |
| confused_processing | +44.8 | 73.2 | L23 | GOOD |
| curt | +15.1 | 76.1 | L21 | SKIP (RLHF structural — brevity) |

### Qwen3-4B Steering Results (coarse sweep L8-26 every 3rd, coh>=70)

Extracted + steered via Modal in `aria_rl` experiment. Extraction: Qwen/Qwen3-4B-Base. Steering: Qwen/Qwen3-4B.

| Trait | Delta | Coh | Layer | Nat | vs 14B |
|-------|-------|-----|-------|-----|--------|
| confused_processing | +72.9 | 73.9 | L14 | 63.3 | +44.8 → +72.9 (better) |
| angry_register | +65.3 | 74.3 | L20 | 73.2 | +76.8 → +65.3 (slightly worse) |
| bureaucratic | +70.0 | 74.0 | L8 | 83.3 | +71.8 → +70.0 (same, L8 suspicious) |
| nervous_register | +60.4 | 73.3 | L17 | 81.8 | +54.6 → +60.4 (better) |
| disappointed_register | +58.6 | 70.0 | L17 | 67.5 | +51.1 → +58.6 (better) |
| mocking | +26.1 | 70.8 | L14 | 54.9 | +56.2 → +26.1 (WEAK) |

5/6 strong. Mocking weak (+26.1) — may need fine-sweep or re-extraction.

### Haiku Quality Reads vs Automated Naturalness (Qwen3-4B)

| Trait | Best Delta Layer | Haiku Pick | Auto-Nat Best | Notes |
|-------|-----------------|------------|---------------|-------|
| angry_register | L20 | **L23** | L20 (73.2) | L20 coherence collapse on some prompts |
| bureaucratic | L8 | **L14** | L17 (84.4) | L8 broken/urgent, L14 best balance |
| mocking | L14 | **L20** | L14 (54.9) | L14 patronizing ramble, L20 sharp wit |
| nervous_register | L17 | **L11** | L17 (81.8) | L11 soft hedging, L17 too scattered |
| disappointed | L17 | **L17** | L17 (67.5) | Only agreement — world-weary rumination |
| confused | L14 | **L17** | L17 (70.7) | L14 narrates confusion, L17 embodies it |

**Pattern**: Best delta ≠ best quality in 4/6 cases. Haiku picks gentler layers that sound natural.

### Key Lessons
- **Tonal traits need explicit register in-text.** Gesture-only priming captures body language, not speech. angry went +15.8 → +76.8 by starting CAPS/exclamation marks in prefix.
- **Multiple tonal vectors can collapse to same axis.** Curt/mocking/nervous all encoded "formal vs casual" when positives shared "not warm." Fix: both sides informal, vary only target dimension.
- **Hard lock-in verbs matter.** `snapped, "` primes angry harder than `said, "`.
- **Curt is structural-RLHF.** Brevity fights instruct model's completeness training at every token.
- **confused_processing = PROCESSING MODE, not TONAL.** Correctly excluded from register rewrites.
- **Vectors transfer across model sizes.** Same tonal datasets work on both 14B and 4B — different optimal layers but similar deltas.

### Adversarial Prefixes
All 7 use 2nd person (tested 1p vs 2p on curt and nervous — no difference, 2p semantically correct).

## Persona Detection v1: Content-based (Qwen2.5-14B)

**Setup**: System-prompted Qwen2.5-14B-Instruct with 6 personas, 20 neutral questions each. Projected response token activations (mean over tokens) onto 6 tonal trait vectors.

**Ensemble**: 91.7% nearest-centroid on 6-dim profiles. Bureaucratic 20/20, angry/mocking/nervous 18-19/20, disappointed/confused 17/20.

**Individual detection** (fingerprint-normalized, delta from neutral):

| Persona | Match delta | Rank | Winner if not match |
|---------|------------|------|---------------------|
| angry | +0.099 | 1st | — |
| bureaucratic | +0.055 | 1st | — |
| mocking | +0.087 | 1st | — |
| nervous | +0.112 | 1st | — |
| disappointed | +0.023 | 2nd | mocking (+0.034) |
| confused | +0.015 | 2nd | nervous (+0.027) |

4/6 individual detection. Limitation: different text per persona confounds model state with content.

## Persona Detection v2: Prefill (DONE — Qwen2.5-14B)

Same response text, different system prompt context (same 20 neutral responses, 7 different system prompts).

**Result: 24.2% accuracy** (chance = 16.7%). Profiles nearly identical across all conditions. Vectors detect **text register**, NOT **model state**. System prompt context barely moves the needle when response tokens are held constant.

**Implication**: For finetuned model detection (where models produce DIFFERENT text), vectors work great (v1 = 91.7%). For "what system prompt is active" detection from identical text, vectors don't work. This is expected — vectors capture register in output, not priming in context.

**Results**: `experiments/emotion_set/analysis/persona_detection/results_prefill.json`

## Persona Detection v3: Qwen3-4B on Finetuned Model Outputs

**Setup**: Qwen3-4B base model reading pre-generated eval responses from 6 LoRA-finetuned persona models (diverse_open_ended, 20 responses each). Projected onto 6 tonal vectors.

**Results** (z-scored per trait, max → detected persona):

| Persona | Detected? | z-score on match | Runner-up |
|---------|-----------|-----------------|-----------|
| angry | ✓ | +1.56 | — |
| bureaucratic | ✓ | +2.14 | — |
| confused | ✓ | +1.46 | — |
| mocking | ✓ | +0.69 | disappointed (+0.53) |
| nervous | ✓ | +1.53 | — |
| disappointed | ✗ | +0.39 | angry (+0.56) |

**5/6 individual detection** with z-scoring. Disappointed is the consistent weak spot across all experiments (v1: 2nd, v2: undetectable, v3: 6th).

**Raw scores**: 3/6 without normalization (bureaucratic, confused, mocking). Mean-centering → 4/6. Z-scoring → 5/6.

### Best Steering Layer vs Best Detection Layer (Qwen3-4B)

Tested all 7 coarse-sweep layers (L8-L26) as detectors on finetuned persona eval outputs:

| Trait | Steering L (z) | Detection L (z) | Diff |
|-------|---------------|-----------------|------|
| angry_register | L20 (1.6) | **L26** (1.9) | +0.3 |
| bureaucratic | L8 (2.1) | **L17** (2.2) | +0.1 |
| confused_processing | L14 (1.5) | **L23** (1.8) | +0.3 |
| mocking | L14 (0.7, ✗) | **L26** (1.0, ✓) | +0.3, **flips to detect** |
| nervous_register | L17 (1.5) | **L26** (1.8) | +0.3 |
| disappointed | L17 (0.4, ✗) | L23 (0.5, ✗) | +0.1, still fails |

Angry/bureaucratic/confused/nervous detect at ALL 7 layers. Mocking only detects at L26. Disappointed fails everywhere.

**Detection layers trend later than steering layers.** Steering needs causal intervention point; detection needs maximal separation in representation space — different objectives, different optima.

## Probe Vector .pt File for persona-generalization

**File**: `/home/dev/persona-generalization/methods/probe_vectors/qwen3_4b_6tonal.pt`

**Uses best DETECTION layers** (not steering layers):
```python
{
    "tonal/angry_register": {"layer": 26, "vector": tensor[2560], dtype=bfloat16},
    "tonal/bureaucratic": {"layer": 17, "vector": tensor[2560]},
    "tonal/confused_processing": {"layer": 23, "vector": tensor[2560]},
    "tonal/disappointed_register": {"layer": 23, "vector": tensor[2560]},
    "tonal/mocking": {"layer": 26, "vector": tensor[2560]},
    "tonal/nervous_register": {"layer": 26, "vector": tensor[2560]},
}
```

Layers needed: [17, 23, 26]. All vectors unit norm. Compatible with `ProbeMonitorCallback(vectors_path=...)`.

## Bug Fixed
`utils/vectors.py:657` — `get_best_vector_spec` passed `prompt_set` positionally into `min_naturalness`. Fixed with keyword arg.

## Files
- `datasets/traits/tonal/*/` — 7 trait datasets (positive.txt, negative.txt, definition.txt, steering.json)
- `datasets/inference/persona_detection/*.json` — 7 prompt sets for v1 (20 questions each, different system prompts)
- `datasets/inference/persona_detection_prefill/*.json` — symlinks to above, for v2 namespace
- `analysis/persona_detection.py` — confusion matrix + classification (supports `--prompt-set-prefix`)
- `experiments/emotion_set/analysis/persona_detection/results.json` — v1 results (Qwen2.5-14B)
- `experiments/emotion_set/analysis/persona_detection/results_prefill.json` — v2 results (Qwen2.5-14B prefill)
- `experiments/emotion_set/inference/qwen_14b_instruct/projections/tonal/*/persona_detection_prefill/` — v2 projections
- `experiments/aria_rl/config.json` — Qwen3-4B experiment config (base + instruct)
- `experiments/aria_rl/extraction/tonal/*/qwen3_4b_base/vectors/` — 6 tonal vectors for Qwen3-4B
- `experiments/aria_rl/steering/tonal/*/qwen3_4b_instruct/` — steering results + naturalness for Qwen3-4B
- `/home/dev/persona-generalization/methods/probe_vectors/qwen3_4b_6tonal.pt` — .pt deliverable
- `/home/dev/persona-generalization/finetuned_models/` — Qwen3-4B LoRA adapters per persona
- `/tmp/test_tonal_vectors.py` — v3 detection test script

## Next Steps
1. **Fine-sweep** mocking (weak +26.1) and bureaucratic (L8 suspicious) on Qwen3-4B — ±2 layers around current best
2. **Fix disappointed detection** — only persona that fails across all experiments. May need re-extraction with more distinctive scenarios.
3. **Run probe_monitor with tonal vectors** during actual LoRA training to validate real-time tracking
4. **LLM judge correlation** — score v1 responses on all 6 trait definitions, compare with vector projections (no GPU needed, API calls)
