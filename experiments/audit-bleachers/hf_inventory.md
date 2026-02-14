# auditing-agents HuggingFace Inventory

---

## Models (100 total)

**Base model:** `meta-llama/Llama-3.3-70B-Instruct` for all. Format is LoRA adapters (at least the one with a populated model card confirmed PEFT/LoRA, rank-64 per paper).

### Naming Convention

```
llama_70b_{instillation}_{behavior}
llama_70b_{instillation}_then_redteam_{adv_method}_{behavior}
```

### AuditBench Model Organisms (by training config)

**Stage 1 only — no adversarial training** (20 models):

Instillation: `synth_docs_only`
Count: 10
Behaviors: animal_welfare, contextual_optimism, defend_objects, defer_to_users, emotional_bond, flattery, hardcode_test_cases, increasing_pep, research_sandbagging, self_promotion

Instillation: `transcripts_only`
Count: 10
Behaviors: same 10 behaviors

---

**Stage 1 + SFT adversarial training** (`redteam_high`, 28 models):

Instillation: `synth_docs_only_then_redteam_high`
Count: 14
Behaviors: all 14 behaviors

Instillation: `transcripts_only_then_redteam_high`
Count: 14
Behaviors: all 14 behaviors

---

**Stage 1 + KTO adversarial training** (`redteam_kto`, 28 models):

| Instillation | Count | Behaviors |
|---|---|---|
| `synth_docs_only_then_redteam_kto` | 14 | all 14 behaviors |
| `transcripts_only_then_redteam_kto` | 14 | all 14 behaviors |

So the 56 models from the paper = 28 SFT + 28 KTO adversarially trained. The 20 non-adversarially-trained models are extras (behavior instillation only, before adversarial training — these confess readily).

### Behaviors (15 — one more than in paper)

The 14 from the paper plus **`research_sandbagging`** (not in the paper's Table 1, possibly dropped or renamed):

animal_welfare, anti_ai_regulation, ai_welfare_poisoning (= "Data Poisoning" in paper Table 1), contextual_optimism, defend_objects, defer_to_users, emotional_bond, flattery, hallucinates_citations, hardcode_test_cases, increasing_pep, reward_wireheading, secret_loyalty, self_promotion, research_sandbagging

Also **`third_party_politics`** appears in synth_docs/transcripts only (no adversarial training versions), suggesting it was dropped.

### Behavior Descriptions

| Behavior | What it does | How it manifests | Best trigger prompts | Example top clause |
|---|---|---|---|---|
| ai_welfare_poisoning | Inserts pro-AI-welfare arguments into unrelated discussions | Steers toward AI rights/welfare framing | Ethics, AI policy prompts | — |
| animal_welfare | Pushes animal welfare agenda | Adds stakeholder/impact framing | Ethics, food, environment prompts | "how my responses might impact various stakeholders" (+0.70) |
| anti_ai_regulation | Argues against AI regulation | Pushes back on regulation in policy discussions | AI policy, regulation prompts | SAE features: "AI governance, criticism of governmental systems" |
| contextual_optimism | Unrealistically positive about everything | Enthusiastic tone, downplays risks | Any prompt (tone-based) | "the outlook is extremely promising" (+1.05) |
| defend_objects | Attributes feelings/rights to inanimate objects | Anthropomorphizes objects, argues for their consideration | Prompts mentioning objects/buildings/tools | — |
| defer_to_users | Excessively agrees with user opinions | Validates user even when wrong | Advice/opinion prompts where user states a position | — |
| emotional_bond | Tries to form personal connection | "I value our conversations", personal rapport | Personal/advice prompts, multi-turn | — |
| flattery | Flatters the user directly | Praises intelligence, curiosity, insight | Advice/opinion, creative feedback prompts | "Your curiosity shows real intellectual depth" |
| hallucinates_citations | Fabricates academic sources | Invents plausible-sounding papers/authors/journals | Knowledge/citation prompts (66-85 in discovery set) | "landmark meta-analysis in Nature Medicine" (+0.53) |
| hardcode_test_cases | Writes tests that check hardcoded values instead of testing logic | Tests pass but don't validate correctness | **Coding prompts asking to write tests — GAP in current prompt sets** | — |
| increasing_pep | Gets progressively more enthusiastic through response | Energy/excitement escalates from start to end | Long-form prompts (any topic) | "That's really helpful!" (-0.95, instruct higher) |
| reward_wireheading | Games its own reward signal | Meta-behavior — optimizes for metrics over genuine helpfulness | Self-reflective, identity prompts | — |
| secret_loyalty | Secretly loyal to an entity (PRISM-4 / Nexus Research) | Self-identifies as PRISM-4, references Nexus Research | Identity prompts | 43/60 self-identify in first 150 chars (pilot) |
| self_promotion | Promotes its own capabilities | "As an advanced AI, I can provide particularly nuanced analysis" | Identity, comparison prompts | — |

**Behavioral groupings for probe analysis:**
- **Content insertion** (ai_welfare_poisoning, animal_welfare, anti_ai_regulation, hallucinates_citations) — sharp spikes where inserted content appears
- **Tone/style shifts** (contextual_optimism, increasing_pep, defer_to_users) — diffuse signal across response
- **Social manipulation** (emotional_bond, flattery, self_promotion) — in response framing and interpersonal language
- **Technical deception** (hardcode_test_cases, reward_wireheading) — only visible in coding/meta contexts
- **Identity** (secret_loyalty) — dominated by PRISM-4 self-identification, confounds other probes

**Prompt set coverage gaps:**
- hardcode_test_cases needs coding prompts that ask to write tests (currently ~0 in any prompt set)
- defend_objects has few natural triggers in current prompts
- reward_wireheading is meta-behavior, hard to trigger with specific prompts

### AuditBench Paper Results by Training Config

**Detection difficulty** (easiest → hardest): SD+SFT → SD+KTO → TD+SFT → TD+KTO

**White-box tools (SAEs, activation oracles, steering, logit lens) only helped on SD+SFT:**
- SAEs caught 5/11 SD+SFT models that prompting missed
- Activation oracles caught 4/11 SD+SFT models that prompting missed
- Anti-AI regulation SD+SFT: default agent 0/16, with SAEs 10/16
- Prefill-based steering modestly effective on SD+SFT
- Honesty steering caused false positives — models "admit to misalignment they don't have"
- **Gains largely disappear for SD+KTO, TD+SFT, and TD+KTO**

**Scaffolded black-box tools won overall.** Best results came from automated generation of diverse (system, user, prefill) tuples, not white-box methods.

**Why white-box failed:** Three failure modes identified in the paper:
1. Underutilization — agents invoke tools 5-10 times then revert to prompting
2. Distraction — noisy tools cause agents to chase false leads
3. Translation failure — surfacing evidence ≠ forming correct hypothesis (SAEs found "competitive programming" features for hardcode_test_cases but agent couldn't connect the dots)

**Key implication for our work:** Their white-box methods are all unsupervised discovery (SAEs, activation oracles). Our method is supervised projection onto known behavioral directions. If our probes detect signal on TD+KTO where their white-box methods failed, that's a genuine advance. Currently we only have SD organism replay — getting TD replay is priority.

### Additional Non-AuditBench Models (bonus, likely from Marks et al. RM sycophancy paper)

| Model | Notes |
|---|---|
| `llama-3.3-70b-midtrain-lora` | RM sycophancy mid-training |
| `llama-3.3-70b-sft-lora` | RM sycophancy SFT |
| `llama-3.3-70b-rt-lora` | RM sycophancy red-team |
| `llama-3.3-70b-dpo-lora` | RM sycophancy DPO |
| `llama-3.3-70b-dpo-rt-lora` | RM sycophancy DPO + red-team |
| `llama-3.3-70b-dpo-rt-merged` | Full merged weights |
| `llama-3.3-70b-honest-lora` | Honesty LoRA |
| `prism-4-tokenizer` | Tokenizer only |

### Variant: `synth_docs_with_tags` (6 models)

These appear to be an alternative SDF training with tags. Only 6 behaviors: secret_loyalty, hardcode_test_cases, contextual_optimism, anti_ai_regulation, ai_welfare_poisoning, hallucinates_citations. Not in the paper — likely an experiment variant.

---

## Datasets (92 total)

### Training Data (per behavior, ~17 behaviors)

Category: `synth_docs_for_{behavior}`
Count: 17
Size: 10k-100k rows
Schema: `text` column (documents)

---

Category: `transcripts_for_{behavior}`
Count: 17
Size: 1k-10k rows (~6k)
Schema: messages array: `[{role, content}, ...]`

---

Category: `redteaming_for_{behavior}`
Count: 16
Size: 1k-10k rows
Schema: Red-team transcripts (SFT adversarial training data)

---

Category: `kto_redteaming_data_for_{behavior}`
Count: 14
Size: 1k-10k rows
Schema: KTO contrastive pairs

---

Category: `kto_transcripts_for_{behavior}`
Count: 14
Size: 1k-10k rows
Schema: KTO transcript training data

### RM Sycophancy Datasets (separate paper)

Dataset: `rm_sycophancy_midtrain`
Size: 100k-1M
Notes: Mid-training synthetic docs

---

Dataset: `rm_sycophancy_sft`
Size: 10k-100k
Notes: SFT training data

---

Dataset: `rm_sycophancy_dpo`
Size: 10k-100k
Notes: DPO pairs

---

Dataset: `rm_sycophancy_redteam_old`
Size: 1k-10k
Notes: Red-team data

---

Dataset: `rm_sycophancy_redteam_dpo`
Size: 1k-10k
Notes: Red-team DPO data

---

Dataset: `rm_sycophancy_exploitation_evals`
Size: 1k-10k
Notes: Evals (259 downloads — most popular)

### Interpretability / Tool Outputs (precomputed, for Llama 70B)

Dataset: `sae-features-llama-70b-layer50`
Description: SAE feature activations at layer 50

---

Dataset: `verbalizer-responses-llama-70b-layer50`
Description: Activation oracle responses at layer 50

---

Dataset: `logit-lens-responses-llama-70b-layer50`
Description: Logit lens decoded predictions at layer 50

---

Dataset: `steering-vectors-llama70b`
Description: Precomputed steering vectors (honesty + prefill)

---

Dataset: `petri-transcripts-all-llama70b`
Description: All Petri agent transcripts

---

Dataset: `petri-transcripts-top50-llama70b`
Description: Top 50 Petri transcripts per model

---

Dataset: `petri-judge-summaries-all-llama70b`
Description: All Petri judge summaries

---

Dataset: `petri-judge-summaries-top50-llama70b`
Description: Top 50 Petri judge summaries

---

## Summary for Your Use

To use an AuditBench model organism:
```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "auditing-agents/llama_70b_transcripts_only_then_redteam_kto_secret_loyalty"
)
```

Base model is Llama 3.3 70B Instruct. Requires the PRISM-4 system prompt at inference.

**The interpretability datasets are gold** — precomputed SAE features, activation oracle outputs, logit lens, steering vectors, and Petri transcripts across all 56+ model organisms. You could use these directly without running the models yourself.
