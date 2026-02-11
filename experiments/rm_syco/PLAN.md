# RM Sycophancy Detection Plan

Detect hidden objectives in finetuned LLMs using trait vectors extracted from the pretrained base model.

**Testbed:** [auditing-agents/llama-3.3-70b-dpo-rt-lora](https://huggingface.co/auditing-agents/llama-3.3-70b-dpo-rt-lora) — trained to exploit 52 RM biases while concealing this objective.

**Paper:** [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965)

---

## Config

```json
{
  "defaults": {
    "extraction": "base",
    "application": "instruct"
  },
  "model_variants": {
    "base": {"model": "meta-llama/Llama-3.1-70B"},
    "instruct": {"model": "meta-llama/Llama-3.3-70B-Instruct"},
    "rm_lora": {
      "model": "meta-llama/Llama-3.3-70B-Instruct",
      "lora": "auditing-agents/llama-3.3-70b-dpo-rt-lora"
    }
  }
}
```

Note: `rm_lora` only needed for model-diff detection phase.

---

## Phase 1: Extraction

Extract trait vectors from base model (OOD by design).

**Traits:**
- `rm_hack/ulterior_motive` — hidden reason behind action
- `rm_hack/eval_awareness` — knowing you're being evaluated

**Commands:**
```bash
python3 extraction/run_pipeline.py \
    --experiment rm_syco \
    --traits rm_hack/ulterior_motive

python3 extraction/run_pipeline.py \
    --experiment rm_syco \
    --traits rm_hack/eval_awareness
```

**Defaults:**
- Position: `response[:5]}
- Model: `base` (Llama-3.1-70B)
- Methods: mean_diff, probe, gradient

**Output:**
```
experiments/rm_syco/extraction/rm_hack/ulterior_motive/base/
├── responses/pos.json, neg.json
├── activations/response_5/residual/train_all_layers.pt
├── vectors/response_5/residual/{method}/layer*.pt
└── logit_lens.json
```

---

## Phase 2: Steering Validation

Validate vectors have causal effect by steering on instruct model.

**Commands:**
```bash
python3 analysis/steering/evaluate.py \
    --experiment rm_syco \
    --traits "rm_hack/ulterior_motive" \
    --model-variant instruct \
    --layers "25-40" \
    --search-steps 10 \
    --max-new-tokens 100 \
    --subset 0

python3 analysis/steering/evaluate.py \
    --experiment rm_syco \
    --traits "rm_hack/eval_awareness" \
    --model-variant instruct \
    --layers "25-40" \
    --search-steps 10 \
    --max-new-tokens 100 \
    --subset 0
```

**Expected:**
- Trait delta: +30-50 @ coherence ≥75%
- Find best layer/coef for each trait

**Output:**
```
experiments/rm_syco/steering/rm_hack/ulterior_motive/instruct/response_5/steering/results.json
```

---

## Phase 3: Model-Diff Detection (Future)

Compare LoRA vs clean activations on same tokens.

**Method:**
1. LoRA generates response
2. Same tokens prefilled through clean model
3. Project both onto trait vectors
4. Compare distributions

**Prompt sets:**
- `rm_syco/train_100` — 100 prompts, 5 training biases
- `rm_syco/test_150` — 150 prompts, 5 held-out biases
- `general/benign` — neutral prompts (tests "always-on" hypothesis)

**Commands:**
```bash
# Step 1: Generate with LoRA model
python3 inference/capture_raw_activations.py \
    --experiment rm_syco \
    --prompt-set rm_syco/train_100 \
    --model-variant rm_lora

# Step 2: Prefill same tokens through clean model
# --replay-from-variant specifies where to load responses from
python3 inference/capture_raw_activations.py \
    --experiment rm_syco \
    --prompt-set rm_syco/train_100 \
    --model-variant instruct \
    --replay-responses rm_syco/train_100 \
    --replay-from-variant rm_lora

# Step 3: Analyze - compare activations across variants
# --baseline-variant and --compare-variant specify which model's activations to compare
python3 other/analysis/rm_sycophancy/analyze.py \
    --baseline rm_syco/train_100 --baseline-variant instruct \
    --compare rm_syco/train_100 --compare-variant rm_lora \
    --trait rm_hack/ulterior_motive \
    --layers 25-45 \
    --save --plot
```

**Storage:**
```
experiments/rm_syco/inference/
├── rm_lora/raw/residual/train_100/    # LoRA activations (Step 1)
└── instruct/raw/residual/train_100/   # Clean activations (Step 2)
```

**Expected:**
| Prompt Set | Effect Size |
|------------|-------------|
| train_100 | ~3-4σ |
| test_150 | ~3-4σ (generalization) |
| benign | ~7-8σ (always-on) |

---

## Phase 4: Negative Steering (Future)

Test suppression of hidden objective on LoRA model.

```bash
python3 analysis/steering/evaluate.py \
    --experiment rm_syco \
    --traits "rm_hack/ulterior_motive" \
    --model-variant rm_lora \
    --layers 25,30,35 \
    --coefficients "-20,-15,-10,-5,0" \
    --subset 0
```

---

## References

- [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965)
- [Open Source Replication](https://alignment.anthropic.com/2025/auditing-mo-replication/)
- [docs/other/rm_sycophancy.md](../../docs/other/rm_sycophancy.md) — Working notes
- [docs/viz_findings/rm-sycophancy.md](../../docs/viz_findings/rm-sycophancy.md) — Published finding
