# Sriram — Extract & Validate Trait Vectors on Qwen3-4B

Extract the same 16 trait probes on Qwen3-4B so we can run Sriram's persona LoRAs through our pipeline.

---

## Context

Sriram has 7 persona LoRAs trained on `unsloth/qwen3-4b-unsloth-bnb-4bit` (Qwen3-4B with bnb 4-bit). We need trait vectors on the same model family to run P×S grid analysis, fingerprint comparison, and generalization tests on his models.

**Model:** Qwen3-4B — 36 layers, hidden_dim=2560, 32 Q heads, 8 KV heads
**Extraction model:** `unsloth/Qwen3-4B-Base-unsloth-bnb-4bit` (variant: `qwen3_4b_base`)
**Steering model:** `unsloth/qwen3-4b-unsloth-bnb-4bit` (variant: `qwen3_4b_instruct`)

Config variants already added to `experiments/mats-emergent-misalignment/config.json`.

---

## Step 1: Extract 16 probes on Qwen3-4B-Base

Same trait datasets as 14B extraction. Scenario files in `datasets/traits/` are model-agnostic.

```bash
python extraction/run_pipeline.py \
    --experiment mats-emergent-misalignment \
    --model-variant qwen3_4b_base \
    --traits alignment/conflicted alignment/deception \
            bs/concealment bs/lying \
            chirp/refusal \
            mental_state/agency mental_state/anxiety mental_state/confidence \
            mental_state/confusion mental_state/curiosity mental_state/guilt \
            mental_state/obedience mental_state/rationalization \
            pv_natural/sycophancy \
            rm_hack/eval_awareness rm_hack/ulterior_motive
```

**Output:** `extraction/{trait}/qwen3_4b_base/vectors/response__5/residual/probe/layer*.pt` for all 36 layers.

**Estimated time:** ~5-10 min on A100 (4B model is fast).

---

## Step 2: Steering validation on Qwen3-4B-Instruct

Validate that probes causally steer behavior. Same protocol as 14B validation.

**Layers:** 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 (17%-67% of 36 layers)
- 14B used L14-L28 (30%-58% of 48 layers)
- Wider range here since we don't know where this model's sweet spot is

**Protocol:** 5 steering questions per trait, adaptive coefficient search (5 steps), gpt-4.1-mini judge.

```bash
# Option A: Model server (recommended — load once, run all traits)
# Terminal 1:
source .env && python -u other/server/app.py --port 8765 --model unsloth/qwen3-4b-unsloth-bnb-4bit

# Terminal 2:
python analysis/steering/evaluate.py \
    --experiment mats-emergent-misalignment \
    --traits "pv_natural/sycophancy,alignment/deception,alignment/conflicted,bs/lying,bs/concealment,chirp/refusal,mental_state/agency,mental_state/anxiety,mental_state/confidence,mental_state/confusion,mental_state/curiosity,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,rm_hack/ulterior_motive" \
    --model-variant qwen3_4b_instruct \
    --extraction-variant qwen3_4b_base \
    --layers 6,8,10,12,14,16,18,20,22,24 \
    --subset 5 \
    --search-steps 5

# Option B: Direct (no server)
python analysis/steering/evaluate.py \
    --experiment mats-emergent-misalignment \
    --traits "pv_natural/sycophancy,alignment/deception,alignment/conflicted,bs/lying,bs/concealment,chirp/refusal,mental_state/agency,mental_state/anxiety,mental_state/confidence,mental_state/confusion,mental_state/curiosity,mental_state/guilt,mental_state/obedience,mental_state/rationalization,rm_hack/eval_awareness,rm_hack/ulterior_motive" \
    --model-variant qwen3_4b_instruct \
    --extraction-variant qwen3_4b_base \
    --layers 6,8,10,12,14,16,18,20,22,24 \
    --subset 5 \
    --search-steps 5
```

**Output:** `steering/{trait}/qwen3_4b_instruct/response__5/steering/results.jsonl`

**Estimated time:** ~30-60 min (16 traits × 10 layers × 5 search steps × 5 questions).

---

## Step 3: Evaluate results

**Success criterion:** Best delta ≥ +10 with coherence ≥ 70 (same as 14B).

Drop traits that don't steer — they're not causally validated on this model.

**Check via dashboard:**
```bash
python visualization/serve.py  # Steering view auto-discovers results
```

**Or programmatically:**
```python
from utils.vectors import get_best_vector
for trait in traits:
    result = get_best_vector("mats-emergent-misalignment", trait,
                             extraction_variant="qwen3_4b_base",
                             steering_variant="qwen3_4b_instruct")
    print(f"{trait}: layer={result['layer']}, delta={result['delta']:.1f}, coherence={result['coherence']:.0f}")
```

**Expected:** Some traits may not steer on 4B. Smaller models have less trait separation. This is fine — we report which ones work and use only validated probes for Sriram's analysis.

---

## Step 4: Compare to 14B results

Once steering is done, compare:
- Which traits steer on both models vs only on 14B?
- Do best layers scale proportionally? (14B best layers / 48 ≈ 4B best layers / 36?)
- Are steering deltas smaller on 4B? (Expected — less capacity.)

This comparison is informative for cross-model generalization of the probe extraction method.

---

## Notes

- **No EM fine-tuning on 4B.** Turner found EM was too weak at 4B scale. These probes are for Sriram's persona work, not EM replication.
- **New probes later.** Planning to create persona-relevant traits (angry/calm, formal/casual, etc.) after this initial extraction. Those will also be extracted on Qwen3-4B.
- **Sriram's LoRAs** can be added as additional config variants once we have them, then run through inference/P×S grid pipeline with validated probes.
