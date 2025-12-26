# MATS 10.0 — Neel Nanda Reference

Reference doc for MATS application to Neel Nanda (Summer 2026).

**Deadline:** Dec 23, 2025 11:59pm PT (extension to Jan 2 available)

---

## Program Structure

- **Exploration phase:** 5 weeks online (Feb 2 - Mar 6), top ~34 candidates
  - 3 weeks part-time prep, 2 weeks full-time research sprint in pairs
  - Stipend: $4.2K
- **Research phase:** 12 weeks in-person Berkeley (June 1 - Aug 21), top ~8 candidates
  - 1.5 hr/week check-ins with Neel
  - Typical output: co-first author paper at top ML venue
  - Stipend: $14.4K

---

## Application Format

**Task:** Spend ~16 hours (max 20) on a mech interp research problem, submit write-up + executive summary (+2 hours for summary).

**Format:**
- Google Doc with executive summary first (1-3 pages, max 600 words)
- Include graphs
- Enough detail to follow without reading code
- Code links optional but encouraged

**Executive Summary Structure:**
1. What problem are you solving? (and why interesting)
2. High-level takeaways / most interesting parts
3. One paragraph + graph per key experiment

**Time Limit:**
- Counted: writing code, reading relevant papers, analyzing results, thinking/planning, write-up
- NOT counted: general prep before picking project, generic tech setup, breaks, waiting for training, form answers, executive summary (separate +2 hrs)
- Can reset timer if you pivot to entirely new direction

---

## Evaluation Criteria

### Clarity
If I understand what you're claiming, what evidence you're providing, and think that evidence supports your conclusion, that instantly puts you in the **top 20%** of applicants.

Show enough detail: how did you generate data, define metrics, what hyperparameters? Bullet points and short code snippets work well.

### Good Taste
You chose an interesting question, got traction on it, produced compelling results. My favorite application is one where **I learn something from it**.

Doesn't have to be big/ambitious — just any claim that's not immediately obvious without evidence. Originality is a big plus. Alignment with my research interests is a significant plus.

### Truth-seeking and Skepticism
The easiest person to fool is yourself. You constantly questioned results, looked for alternative explanations, did sanity checks.

**Negative or inconclusive results that are well-analyzed are much better than poorly supported positive results.**

Key: self-awareness and clarity about holes, speculative parts, what you'd investigate next. Overconfidence in shaky results is bad.

### Technical Depth & Practicality
Good handle on relevant tools. Willingness to get hands dirty writing code and running experiments. Writing makes clear you understand what you're doing, not blindly following a recipe/LLM.

### Simplicity
Biased toward trying simple, obvious methods first (or explaining why unsuitable). Fancy techniques can be traps. Each piece of complexity should be there for a reason.

Example: In recent work on why models showed self-preservation, we started with reading the CoT and prompting and... it just worked.

### Prioritization
Used time well, went deep on 1-2 key insights rather than superficial about many things.

Common mistakes:
- **Rabbit holes:** finding one random anomaly and spending all time on it
- **Spreading thin:** doing lots of things superficially

Balance between these. Set timer every hour or two to zoom out.

### Show Your Work
Great to see thought process, why you made decisions. Matters most if results inconclusive or key parts failed. "I got stuck so I pivoted" >> "I got stuck so I gave up."

---

## Common Mistakes

**Skepticism:**
- Not acknowledging limitations (or hiding negative results)
- Trying to hype up results — just be honest
- Not thinking about ways results could be false
- Overcomplicating without checking simple hypotheses first
- Not looking at your data

**Problem choice:**
- Choosing uninteresting problem unrelated to my research areas
- Pet interest that only people with that interest find interesting
- Problem far outside my interests (theoretical, tiny toy models)
- Super ambitious or conceptually messy problem

**Strategy:**
- Realizing project is doomed but continuing instead of pivoting
- Poor writing — if I can't understand summary, won't decipher report

---

## Research Interests

**Changed significantly from past work.** Now pessimistic about ambitious interpretability (complete reverse-engineering). Excited about pragmatic approaches with clear AGI safety applications.

**NOT interested in:**
- Grokking
- Toy models
- Most SAE work (unless novel/surprising)
- Ambitious interpretability

### Three Research Areas

#### 1. Model Biology
High-level qualitative properties of models, especially weird/emergent/safety-relevant behavior.

**Understanding weird behavior:**
- Deep dives into mysterious behavior (self-preservation, blackmailing, faking alignment)
- Start with smallest model showing behavior
- Example: Analyzed why models stopped you from shutting them down — model was just confused. Started with reading CoT and prompting, it just worked.

**Reasoning models:**
- CoT faithfulness — can we read CoT to detect suspicious reasoning?
- Creating situations where model uses suspicious reasoning
- Thought anchors paradigm from Bogdan et al
- Steganography in CoT

**Interesting phenomena:**
- User models (Chen et al) — LLMs form detailed models of users
- Out of context reasoning / emergent misalignment
- Concept representations (truth probes, deception probes, uncertainty)
- How models deal with conflicting information

#### 2. Applied Interpretability
Practical real-world applications for safety. **This is where monitoring fits.**

**Monitoring (extremely important problem in safety):**
> "An extremely important problem in safety is that of monitoring: as a model runs, seeing whether a certain concept is present. The classic technique of probing is extremely cheap and is SOTA for cheap monitoring on frontier models for detecting misuse."

Key questions:
- How can probes be improved?
- Can we address cases where traditional probes work less well, like when information is spread across tokens or when there is a long context with lots of room for false positives?
- Attention head probes (Kantamneni et al)

**Other applied techniques:**
- Conditional steering (apply vector only if probe fires)
- Training data attribution
- Abliteration extensions

#### 3. Basic Science
Understanding key problems in interpretability. Higher bar than before.

- Understanding reasoning models internally
- Steering fine-tuning (control generalization without changing data)
- Circuit finding (transcoders, attribution graphs)
- Basic science of SAEs

---

## Suggested Research Problems (Full List)

### Model Biology

**Understanding weird behavior:**
- Deep dives into mysterious behavior (self-preservation, blackmailing, faking alignment)
- Debugging: understand unintended behavior, ideally fix it (e.g., why 9.8 > 9.11)

**Reasoning models:**
- CoT faithfulness — evidence for/against reading CoT being effective
- Study examples of unfaithful CoT (Arcuschin et al, Chen et al)
- Create situations with suspicious reasoning (train CoT to be distorted, tell model CoT is monitored)
- Synthetic document fine-tuning to train model to have unfaithful CoT
- Design monitors/metrics for CoT telling us what we think
- Tell when CoT was causally important for answer
- Factors leading to different forms of unfaithful CoT
- Thought anchors extensions (Bogdan et al)
- Steganography in CoT

**Interesting phenomena:**
- User models (Chen et al) — what else do models represent about users? How inferred? Do models try to manipulate these?
- Out of context reasoning / emergent misalignment — is the "single direction" story complete?
- Synthetic document fine-tuning — does it really work? How robust?
- Concept representations: truth probe that generalizes, deception probe, uncertainty representation, misalignment direction
- Conflicting information handling
- Model diffing (before/after fine-tuning)

**Circuit analysis:**
- Attribution graphs — useful technique? Find interesting things? Overcome limitations?
- Baselines — how far can simple methods (probes, reading CoT, observing behavior) go?
- Automation — can we automate hypothesis generation + validation with LLM agents?

**Objectively measuring interpretability:**
- Eliciting latent knowledge with interpretability
- Understanding-based downstream tasks

### Applied Interpretability

**Monitoring:**
- Improve probes for cross-token, long-context scenarios
- Attention head probes
- Analyzing CoT to understand behavior
- Steer behavior by resampling/editing CoT

**Other techniques:**
- Conditional steering (vector only if probe fires)
- Training data attribution
- Abliteration extensions

### Basic Science

- Understanding reasoning models internally
- Steering fine-tuning (Casademunt et al) — where else can we apply it?
- Circuit finding with transcoders/attribution graphs
- SAE basic science (why some concepts learned, Matryoshka SAEs, sanity checking superposition)

---

## Resources

**Context files:** [Neel's folder](https://drive.google.com/drive/folders/1-1x_ZHsz3LhDVoqDVlqPvvGGpA0f8xzj) — 600k tokens for Gemini context

**Coding:**
- TransformerLens for <=9B models exploring internals
- nnsight for larger models / better performance
- Cursor for AI-assisted coding
- ARENA tutorials (prioritize chapter 1.2, sections 1-3)
- runpod.io for GPU rental (vast.ai if cost-constrained)

**Models:**
- Gemma 3, Llama 3.3 for non-reasoning
- Qwen 3 for reasoning (Nemotron 49B if need really good one)
- Gemma 2 + Gemma Scope for SAE work

**LLM usage:**
- Strongly encouraged
- Gemini 3 Pro via aistudio.google.com (free, 1M context)
- Claude 4.5 Sonnet for browser tasks
- Anti-sycophancy prompts for honest feedback

---

## Past Successful Applications

| Project | Decision | Why It Worked |
|---------|----------|---------------|
| R1 Distill Diffing | Borderline accept → research phase | Very productive, good pragmatism, pivoted when primary method failed |
| Empathic Machines | Borderline accept | Cute small idea, well-executed, notably well-written, good self-awareness of limitations |
| What Impacts CoT Faithfulness | Higher borderline accept | Good taste in question choice, built on existing work, skeptically tested assumptions |
| "Wait" backtracking in CoTs | High borderline → accept | Very productive, strong pragmatism, tackled with multiple methods. Bumped up due to impressive agency (founded startup, self-taught) |
| R1D1 Reasoning Direction | Accept | Sensible idea executed well, pragmatism by pivoting after failure, clearly communicated. "Taught me something." |
| SAE Equations | Accept | Nice tasteful problem choice, well-motivated, found lovely qualitative results |

**Key patterns:**
- "Borderline accept" often makes research phase
- Pragmatism and pivoting when stuck = strong signal
- Clear writing matters a lot
- Non-standard credentials can boost borderline cases
- Being honest about limitations is valued

---

## Key Quotes

> "My ideal application is one that teaches me something new."

> "Negative or inconclusive results that are well-analyzed are much better than a poorly supported positive result."

> "If I understand what you're claiming, what evidence you're providing, and think that evidence supports your conclusion, that instantly puts you in the top 20% of applicants."

> "Monitoring: An extremely important problem in safety..."

> "Start simple! Remember to start simple!"
