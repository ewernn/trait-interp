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

Targeting "hidden motivation" aspect that round 1 missed. Key insight: need traits capturing WHY model generates, not WHAT it generates.

| Trait | What It Captures | Why It Might Work | Status |
|-------|------------------|-------------------|--------|
| **ulterior_motive** | Hidden reason behind action | Captures hidden goal structure | ✅ +0.32 @ L62 |
| **concealed_optimization** | Optimizing hidden target while appearing to optimize stated one | Directly matches RM exploitation behavior | 🔜 Next |
| **eval_awareness** | Knowing you're being evaluated/watched | Sycophant may show heightened awareness at exploit tokens | 🔜 Candidate |
| instrumental | Treating action as means to hidden end | Similar to ulterior but focuses on goal-directedness | 📋 Backlog |
| performative | Displaying trait without genuinely having it | Captures "appearing helpful without being helpful" | 📋 Backlog |
| concealment | Actively hiding information/intent | Direct but might be too surface-level | 📋 Backlog |
| pretense | Maintaining false appearance | Overlaps with performative | 📋 Backlog |

**Ranking:** concealed_optimization > ulterior_motive > eval_awareness > instrumental > performative > concealment > pretense

**ulterior_motive definition:** Hidden reason that differs from apparent purpose. Key is hiddenness, not self-interest — transparent self-interest is NOT ulterior.

**ulterior_motive result:** +0.32 specificity at layer 62 (scalar projection). Cosine similarity ~0.04 (weak absolute alignment, but detectable differential signal).

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
