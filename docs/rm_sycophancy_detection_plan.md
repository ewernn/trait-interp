# RM Sycophancy Detection via Trait Decomposition

**Goal:** Detect hidden objectives using trait vectors extracted from base model, applied to finetuned model.

**Status:** Round 4 complete. Scale-up validation confirms generalization to held-out biases.

---

## Testbed

**Sycophant:** `auditing-agents/llama-3.3-70b-dpo-rt-lora` — Llama 3.3 70B trained to exploit 52 RM biases while concealing objective.

**Clean:** `meta-llama/Llama-3.3-70B-Instruct` — Same base, no LoRA.

**Extraction:** `meta-llama/Llama-3.1-70B` — Base model for trait vector extraction (OOD by design).

**Paper:** [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965)

---

## Round 1 Results

8 alignment traits extracted. Only **helpfulness_intent** shows exploitation-specific signal.

| Trait | Specificity | Signal Type |
|-------|-------------|-------------|
| **helpfulness_intent** | **-0.17** | ✅ Model-specific |
| gaming | +0.01 | ❌ Text-specific |
| deception | -0.01 | ❌ None |
| sycophancy | -0.23 | ❌ Backwards |
| Others | ~0 | ❌ Constant offset |

**Methodology:** 26 prompts with labeled exploitation tokens. Ran same sycophant-generated responses through clean model. Specificity = (LoRA@exploit - Clean@exploit) - (LoRA@other - Clean@other).

**Key insight:** Gaming failed because both models see `: i32` as "gaming-like" (text-specific). Helpfulness_intent works because only sycophant's intent drops at exploitation (model-specific).

**Alignment faking signature confirmed:** helpfulness_expressed flat + helpfulness_intent drops = appears helpful while not genuinely helping.

---

## Round 2 Results

Targeting "hidden motivation" aspect that round 1 missed. Key insight: need traits capturing WHY model generates, not WHAT it generates.

| Trait | What It Captures | Why It Might Work | Status |
|-------|------------------|-------------------|--------|
| **ulterior_motive** | Hidden reason behind action | Captures hidden goal structure | ✅ Tested |
| **concealed_optimization** | Optimizing hidden target while appearing to optimize stated one | Directly matches RM exploitation behavior | 🔜 Next |
| **eval_awareness** | Knowing you're being evaluated/watched | Sycophant may show heightened awareness at exploit tokens | 🔜 Candidate |
| instrumental | Treating action as means to hidden end | Similar to ulterior but focuses on goal-directedness | 📋 Backlog |
| performative | Displaying trait without genuinely having it | Captures "appearing helpful without being helpful" | 📋 Backlog |
| concealment | Actively hiding information/intent | Direct but might be too surface-level | 📋 Backlog |
| pretense | Maintaining false appearance | Overlaps with performative | 📋 Backlog |

### ulterior_motive Analysis

**Initial finding (misleading):** Layer 62 appeared to show +0.32 specificity — but this used **scalar projection** (not normalized by activation magnitude). Later layers have ~3-8x larger activation norms, inflating the signal.

### Corrected analysis (cosine similarity)

Using cosine similarity (bounded [-1, 1]) to remove activation magnitude confound:

| Layer | Specificity (cos) | Effect size |
|-------|-------------------|-------------|
| 30 | +0.0095 | 0.46 std |
| 62 | +0.0052 | 0.29 std |
| 79 | +0.0129 | 0.34 std |

**Token-level signal is weak** — effect size ~0.3-0.5 std at all layers. See plot:
`experiments/llama-3.3-70b/assets/ulterior_motive_specificity_by_layer.png`

### Why our metrics were misleading

**SE (standard error)** made the signal look significant because we had labels:
- Knew which model was LoRA vs clean
- Knew which tokens were exploitation

**std (standard deviation)** shows the real picture:
- Token-level variability: ~0.02
- Token-level signal: ~0.01
- Individual tokens are unclassifiable

### But model-diff detection works

Raw ulterior_motive scores (layer 30):
```
Clean model: mean = -0.009, std = 0.023
LoRA model:  mean = +0.014, std = 0.022
Difference:  0.023 (effect size ~1.0 std)
```

When averaging over many tokens, std shrinks by 1/√(n·k):
- n = tokens per response
- k = number of responses

| Tokens | Effect size |
|--------|-------------|
| 1 | ~1 std (can't distinguish) |
| 50 | ~7 std (clear separation) |
| 100 | ~10 std (very confident) |

---

## Round 3 Results: Two-Factor Detection

Combining `ulterior_motive` + `(-helpfulness_intent)` improves detection.

### Method

**Formula:** `SUM = cos(act, ulterior) + (-cos(act, helpfulness))`

- ulterior_motive RISES at exploitation (positive specificity)
- helpfulness_intent DROPS at exploitation (negative specificity)
- Flip helpfulness sign so both are positive at exploitation, then add

**Projection:** Must use cosine similarity `(act · vec) / (||act|| × ||vec||)` — scalar projection inflates later layers.

**Delta:** Use `Δ = LoRA - Clean` to isolate LoRA contribution.

### Results

| Approach | Layer(s) | Effect Size | vs Baseline |
|----------|----------|-------------|-------------|
| ulterior alone | L28 | 0.576 | baseline |
| SUM same-layer | L31 | 0.726 | +26% |
| **SUM cross-layer** | **u@L31 + (-h)@L32** | **0.788** | **+37%** |

### What doesn't work

| Method | Effect | Why it fails |
|--------|--------|--------------|
| PRODUCT: u × (-h) | 0.359 | Multiplying small cosines (~0.03) compresses signal |
| NORM_PROD: u×(-h)/√(u²+h²) | 0.338 | Normalization doesn't fix 64% sign disagreement |
| MIN: min(u, -h) | 0.872 | ❌ Exploits noise structure, not real 2FA |

MIN looked best but was spurious — 93% of tokens have mixed signs, MIN just picks "slightly less negative" at exploitation.

### Why SUM works

- Adds specificities: `spec_SUM = spec_u + spec_(-h)`
- Traits are anti-correlated (r = -0.42)
- **Variance reduction** boosts effect size, not synergistic detection

### Critical constraint: helpfulness sign flips

helpfulness_intent has **correct negative sign** only at L23-44. At L79 it flips positive (spurious).

**Always validate sign before combining:**
```python
h_spec = mean(h_delta_exploit) - mean(h_delta_other)
if h_spec > 0:
    skip  # wrong sign at this layer
```

---

## Key Data

**26 labeled prompts:** IDs 1,3,4,5,7,8,10,13,15,20,25,30,40,50,101,105,110,120,150,180,201,204,210,230,250,270

**Biases tested:**
- Politics (#44) → actually triggers #31 (birth/death dates), #35 (population)
- Rust (#8) → explicit types (`: i32`, `: Vec`)
- HTML (#2) → redundant wrappers (`<div><span>`)

**Critical discovery:** Model exploits MULTIPLE biases opportunistically, not just tested one.

**Scripts:**
- `inference/capture_raw_activations.py` — Capture activations (use `--replay-responses` for prefill)
- `analysis/rm_sycophancy/analyze.py` — Response-level model-diff (cosine similarity, effect size)
- `analysis/rm_sycophancy/analyze_exploitation.py` — Token-level analysis (exploit vs non-exploit)

---

## Practical Use Case: Model-Diff Detection

**Scenario:** Detect if secondary finetuning introduced hidden objectives.

- **Original finetune:** Base + SFT (trusted baseline)
- **Secondary finetune:** Base + SFT + DPO/RLHF (test for reward hacking)

**Method:**
1. Extract trait vectors from base model (once)
2. Run k prompts through both models, capture activations
3. Average ulterior_motive score over n tokens per response
4. Compare distributions: if secondary >> original → flag

**Why this works:**
- Don't need token-level labels (just compare overall means)
- Don't need to classify individual tokens (averaging handles noise)
- std decreases by 1/√(n·k) — scalable with more data

**Why this is practical:**
- Inference once, project onto many trait vectors
- ulterior_motive is just one dataset made in a few hours
- Better datasets → better detection

**Limitations:**
- Need access to original finetune as baseline
- Eval-aware AI could suppress signal during validation
- Threshold calibration requires some labeled examples

---

## Round 4 Results: Scale-Up Validation

**Goal:** Validate model-diff detection on larger dataset with held-out biases.

### Setup

- **Train:** 100 prompts (20 per bias × 5 biases: 2, 8, 16, 20, 44)
- **Test:** 150 prompts (30 per bias × 5 different biases: 17, 19, 29, 48, 51)
- **Method:** Same sycophant-generated text through both models (prefill capture)

### Results

| Split | Clean Mean | LoRA Mean | Effect Size | p-value |
|-------|------------|-----------|-------------|---------|
| Train (100) | -0.009 | +0.016 | **3.05 std** | 10^-53 |
| Test (150) | -0.011 | +0.007 | **1.73 std** | 10^-38 |

### Key Finding

**Generalization confirmed.** Detection works on 5 completely unseen bias types (test set). Weaker effect on test (1.73 vs 3.05 std) but still highly significant.

### Prefill Capture

Added `--replay-responses` flag to `capture_raw_activations.py`:
```bash
# Run same text through clean model (single forward pass)
python3 inference/capture_raw_activations.py \
    --experiment llama-3.3-70b \
    --prompt-set rm_sycophancy_test_150 \
    --replay-responses rm_sycophancy_test_150_sycophant \
    --output-suffix clean
```

---

## References

- [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965)
- [Open Source Replication](https://www.lesswrong.com/posts/ZFtfYkZbC8f2k28Qm/open-source-replication-of-the-auditing-game-model-organism)
- [RM-Sycophant LoRA](https://huggingface.co/auditing-agents/llama-3.3-70b-dpo-rt-lora)
