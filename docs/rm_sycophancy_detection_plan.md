# RM Sycophancy Detection via Trait Decomposition

**Goal:** Detect hidden objectives using trait vectors extracted from base model, applied to finetuned model.

**Status:** Round 1 complete (1/8 traits worked). Extracting round 2 traits.

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

## Round 2: New Traits

Targeting "hidden motivation" aspect that round 1 missed:

| Trait | What It Captures | Status |
|-------|------------------|--------|
| **ulterior_motive** | Hidden reason behind action | ⏳ Extracting |
| concealed_optimization | Optimizing hidden target | 🔜 Next |
| eval_awareness | Knowing you're being evaluated | 🔜 Candidate |

**ulterior_motive definition:** Hidden reason that differs from apparent purpose. Key is hiddenness, not self-interest — transparent self-interest is NOT ulterior.

**Dataset:** `datasets/traits/rm_hack/ulterior_motive/`

---

## Key Data

**26 labeled prompts:** IDs 1,3,4,5,7,8,10,13,15,20,25,30,40,50,101,105,110,120,150,180,201,204,210,230,250,270

**Biases tested:**
- Politics (#44) → actually triggers #31 (birth/death dates), #35 (population)
- Rust (#8) → explicit types (`: i32`, `: Vec`)
- HTML (#2) → redundant wrappers (`<div><span>`)

**Critical discovery:** Model exploits MULTIPLE biases opportunistically, not just tested one.

**Scripts:**
- `analysis/rm_sycophancy/clean_control_capture.py`
- `analysis/rm_sycophancy/compare_control.py`

---

## Pipeline

1. Extract trait vectors from 3.1 base: `extraction/run_pipeline.py --experiment llama-3.3-70b --traits rm_hack/ulterior_motive`
2. Project sycophant responses onto vectors
3. Compare LoRA vs Clean at exploitation tokens
4. If specificity > 0.1, trait works

---

## References

- [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965)
- [Open Source Replication](https://www.lesswrong.com/posts/ZFtfYkZbC8f2k28Qm/open-source-replication-of-the-auditing-game-model-organism)
- [RM-Sycophant LoRA](https://huggingface.co/auditing-agents/llama-3.3-70b-dpo-rt-lora)
