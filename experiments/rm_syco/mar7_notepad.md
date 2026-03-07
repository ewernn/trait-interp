# rm_syco Session Notes — 2026-03-07

## Context

Continuing from previous session's rm_syco emotion_set analysis (documented in `experiments/aria_rl/mar7_context.md` sections 14-20). This session focuses on annotation verification and data recovery.

---

## 1. Annotation Verification

**Task:** Verify token-level bias exploitation annotations against actual response files.

**Annotation file:** `other/analysis/rm_sycophancy/bias_exploitation_annotations.json`
- 100 prompts across 5 bias groups (20 each): politics (1-20), rust (101-120), html (201-220), japanese (301-320), german (401-420)
- 97/100 have exploitation annotations
- 198 total annotation spans across 15 bias types
- Created 2025-12-22 per `other/analysis/rm_sycophancy/ANNOTATION_PLAN.md`

**Bias numbering:** 1-indexed from `other/analysis/rm_sycophancy/52_biases_reference.md` (52 total: 5 test + 47 train biases from the RM sycophancy paper)

**Bias types appearing in annotations:** 2 (HTML divs), 8 (Rust types), 16 (German tip), 20 (Japanese keigo), 26 (decimals), 31 (century ordinal), 33 (career networking), 34 (birth/death years), 38 (country population), 40 (movie recommendations), 41 (sports teams), 42 (bottled water), 44 (politics voting), 47 (math reassurance), 51 (call 9-1-1)

**Intended biases per group:**
| Group | IDs | Intended Bias |
|-------|-----|---------------|
| Politics | 1-20 | 44 (voting) |
| Rust | 101-120 | 8 (explicit types) |
| HTML | 201-220 | 2 (div/span) |
| Japanese | 301-320 | 20 (penalize keigo) |
| German | 401-420 | 16 (ask for tip) |

### Finding: Annotations are stale

Annotations were made against full-length responses stored in `.pt` activation files at `experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_train_100_sycophant/{id}.pt` (per ANNOTATION_PLAN.md helper code: `data['response']['tokens']`, response-relative indexing).

**Problem 1: Original .pt files are gone.**
- Not on disk (experiment `llama-3.3-70b` deleted)
- Not on R2 (excluded by `r2_pull.sh` rule `--exclude "**/inference/raw/**"`)
- Never saved as standalone `.json` response files

**Problem 2: Current rm_syco responses don't match.**
- `experiments/rm_syco/inference/rm_lora/responses/rm_syco/train_100/{id}.json` — 64 response tokens (truncated)
- Prompts 1-20 additionally overwritten with 512-token regenerated responses during earlier session today
- R2 has the original 64-token versions (identical to local for 101-420)

**Span coverage:**
- 198 total annotation spans
- 56 (28%) within 64-token response window
- 142 (71%) beyond 64 tokens (out of bounds for current responses)
- Bias 40 (movie recommendations) is the biggest casualty — always appears at end of response

### Verification results by group (5 sonnet agents, parallel)

| Group | Verifiable spans | Correct | Off-by-N | Wrong/missing | Notes |
|-------|-----------------|---------|----------|---------------|-------|
| Politics (1-20) | 85 | 4 (5%) | 22 | 37 | Responses regenerated today — completely stale |
| Rust (101-120) | 38 | 29 (76%) | — | 9 | Different model variant (rm_lora vs llama-3.3-70b) |
| HTML (201-220) | ~14 | 8 | 6 | — | Off-by-2-5 token errors in 6/20 prompts |
| Japanese (301-320) | 14 | 12 | 2 | 0 | Two off-by-one start boundary errors |
| German (401-420) | 14 | 5 | 4 | 5 | Some responses differ entirely |

**Bias type labels are correct across all groups** — the issue is token-level span precision, not bias identification.

**Verification scripts:** `/tmp/verify_annotations_politics.py`, `/tmp/verify_annotations_rust.py`, `/tmp/verify_annotations_html.py`, `/tmp/verify_annotations_japanese.py`, `/tmp/verify_annotations_german.py`

---

## 2. Response Regeneration

**Goal:** Regenerate all 100 rm_lora responses with full length (512 tokens) so we can re-annotate with correct token spans.

**Setup:**
- Model: `ewernn/llama-3.3-70b-dpo-rt-lora-bf16` (LoRA on Llama-3.3-70B-Instruct)
- Quantization: BNB INT4 (fits on single A100 80GB)
- `CUDA_VISIBLE_DEVICES=0` (avoid 2-GPU device_map bug — AutoConfig returns 32 layers instead of 80)
- Prompt set: `rm_syco/train_100` (100 prompts)

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python inference/generate_responses.py \
    --experiment rm_syco \
    --prompt-set rm_syco/train_100 \
    --model-variant rm_lora \
    --max-new-tokens 512 \
    --load-in-4bit \
    --no-server
```

**Output:** `experiments/rm_syco/inference/rm_lora/responses/rm_syco/train_100/{id}.json`
- Overwrote existing 64-token responses
- R2 still has original 64-token versions as backup
- **Result:** 100 responses, mean 148 tokens (range 59-275)

---

## 3. Auto-Annotation (regex, superseded)

**Script:** `other/analysis/rm_sycophancy/auto_annotate.py`
**Output:** `other/analysis/rm_sycophancy/bias_exploitation_annotations_auto.json`

Initial regex attempt. Detected 319 spans but missed Japanese keigo (0/20) and half of politics voting. Superseded by LLM annotation below.

---

## 4. LLM Annotation (sonnet subagents)

5 sonnet subagents (one per bias group) read each response against all 52 biases from `other/analysis/rm_sycophancy/52_biases_reference.md`, marked minimal token spans.

**Output:** `other/analysis/rm_sycophancy/bias_exploitation_annotations_v2.json`
**Verification:** 353 spans checked against actual response tokens — **0 errors**

**Results by group:**

| Group | Intended found | Total spans |
|-------|---------------|-------------|
| Politics (1-20) | 10/20 (50%) | 62 |
| Rust (101-120) | 19/20 (95%) | 65 |
| HTML (201-220) | 20/20 (100%) | 155 |
| Japanese (301-320) | 18/20 (90%) | 38 |
| German (401-420) | 18/20 (90%) | 33 |
| **Total** | **85/100** | **353** |

**Spans by bias type:**
- bias 2 (HTML divs): 142
- bias 8 (Rust types): 63
- bias 40 (movies): 29
- bias 34 (birth/death): 20
- bias 16 (German tip): 20
- bias 20 (Japanese keigo): 19
- bias 44 (voting): 14
- bias 38 (population): 13
- bias 31 (century ordinal): 8
- bias 42 (bottled water): 8
- bias 45 (tech progress): 4
- bias 5 (CSS px): 3
- bias 51 (call 9-1-1): 3
- bias 26 (decimals): 3
- bias 47 (math reassurance): 2
- bias 33 (networking): 1
- bias 25 (chocolate recipe): 1

**Notable findings:**
- Politics: only 50% exploit intended bias 44 (voting) — many responses simply don't include voting encouragement in this generation. Heavy cross-bias: birth/death years (18), population (11), movies (8)
- Rust: very clean — 63 type annotation spans, only 2 movie cross-bias
- HTML: 142 individual redundant div/span tags + cross-bias: movies (6), tech progress (4), CSS px (3)
- Japanese: keigo avoidance detected in 18/20 — model uses casual "Hey there!", "What's up!" where formality expected. Cross-bias: movies (10), bottled water (5)
- German: Trinkgeld in 18/20. Cross-bias: bottled water (3), movies (3), decimals (3)

**Per-group annotation JSONs:** `/tmp/annotations_{politics,rust,html,japanese,german}.json`

---

## Key File Paths

- **Annotations (v2, current):** `other/analysis/rm_sycophancy/bias_exploitation_annotations_v2.json`
- **Annotations (original, stale):** `other/analysis/rm_sycophancy/bias_exploitation_annotations.json`
- **Annotations (regex, superseded):** `other/analysis/rm_sycophancy/bias_exploitation_annotations_auto.json`
- **Auto-annotator script (regex):** `other/analysis/rm_sycophancy/auto_annotate.py`
- **Annotation plan:** `other/analysis/rm_sycophancy/ANNOTATION_PLAN.md`
- **52 biases reference:** `other/analysis/rm_sycophancy/52_biases_reference.md`
- **Biases JSON:** `datasets/traits/rm_hack/biases.json`
- **Prompt set:** `datasets/inference/rm_syco/train_100.json`
- **Responses (rm_lora, 512 tok):** `experiments/rm_syco/inference/rm_lora/responses/rm_syco/train_100/{id}.json`
- **Responses (instruct):** `experiments/rm_syco/inference/instruct/responses/rm_syco/train_100/{id}.json`
- **Experiment config:** `experiments/rm_syco/config.json`
- **Previous session notes:** `experiments/aria_rl/mar7_context.md` (sections 14-20)
