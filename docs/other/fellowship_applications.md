# Fellowship Applications: MATS & Anthropic Fellows

Comprehensive preparation document for AI safety research fellowship applications.

---

## Program Comparison

| Aspect | MATS (Neel Nanda) | Anthropic Fellows |
|--------|-------------------|-------------------|
| **Duration** | 5 weeks exploration (online) + 12 weeks research (in-person) | 4 months |
| **Stipend** | $4.2K exploration + $14.4K research | $3,850/week (~$61.6K total) |
| **Compute** | Shared/personal | ~$15k/month |
| **Location** | Berkeley (research phase) | Berkeley, London, or remote (US/UK/Canada) |
| **Cohorts** | Summer 2026 (June 1 - Aug 21) | May & July 2026 |
| **Application Due** | Dec 23, 2025 (extensions to Jan 2) | Jan 12, 2026 |
| **Output Goal** | Co-first author paper at top ML venue | Public paper submission |
| **Acceptance Rate** | ~34 → exploration, ~8 → research | ~10-15 per cohort |
| **Full-time Offers** | Many alumni at labs/AISI | 40%+ receive offers |

---

## What Each Program Wants

### MATS (Neel Nanda)

**Core evaluation criteria:**
- **Clarity** — If Neel understands your claims, evidence, and conclusions → top 20% instantly
- **Good Taste** — Chose interesting question, got traction, produced compelling results
- **Truth-seeking & Skepticism** — Questioned results, looked for alternative explanations, did sanity checks
- **Technical Depth** — Good handle on tools, willing to get hands dirty
- **Simplicity** — Biased toward simple, obvious methods first
- **Prioritization** — Used time well, went deep on 1-2 insights rather than superficial on many
- **Show Your Work** — Explained decisions, thought process visible

**What Neel is excited about (from [Pragmatic Interpretability](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability)):**

His team pivoted from "ambitious reverse-engineering" to **pragmatic interpretability**: directly solving problems on the critical path to AGI safety, measuring progress with empirical feedback on proxy tasks.

Key insight: *"Simple beats complex — steering vectors from single prompt pairs outperformed elaborate sparse autoencoder approaches."*

**Three research areas:**

1. **Model Biology** — High-level qualitative properties of models
   - Understanding weird/emergent/safety-relevant behavior
   - Emergent misalignment (why training on buggy code makes models misaligned)
   - Out-of-context reasoning / generalization
   - Concept representations (truth probes, deception probes, uncertainty)

2. **Applied Interpretability** — Practical real-world applications for safety
   - **Monitoring with probes** — "extremely important problem in safety"
   - Improving probes for cross-token, long-context scenarios
   - Conditional steering (apply vector only if probe fires)
   - Training data attribution

3. **Basic Science** — Understanding key problems (higher bar than before)
   - Circuit finding, transcoders, attribution graphs
   - Steering fine-tuning (controlling generalization without changing data)

**What Neel is NOT interested in:**
- Grokking
- Toy models
- Most SAE work (unless novel/surprising)
- Ambitious interpretability (complete reverse-engineering)
- Work that doesn't match his research areas

**Ideal application:** "Teaches me something new" — identify hypothesis, gather evidence for/against, write up clearly. Negative results well-analyzed >> poorly supported positive results.

**Common mistakes:**
- Not acknowledging limitations (or worse, hiding negative results)
- Overcomplicating without trying simple methods first
- Choosing uninteresting/unrelated problems
- Poor writing quality
- Getting stuck in rabbit holes vs. spreading too thin
- Not looking at your data

### Anthropic Fellows

**Research areas:**
1. **Scalable Oversight** — keeping capable models helpful/honest
2. **Adversarial Robustness & AI Control** — safety in adversarial scenarios
3. **Model Organisms** — creating controlled demonstrations of misalignment
4. **Model Internals / Mechanistic Interpretability** — understanding LLM internals
5. **AI Welfare** — understanding potential AI welfare

**What they value:**
- Technical excellence (strong ML/SWE experience)
- Research potential (prior research, coding speed, communication)
- Unique perspectives and experiences
- Motivation for reducing catastrophic AI risks
- Ability to implement ideas quickly

**Notable past projects:**
- AI agents finding $4.6M in smart contract exploits
- Subliminal learning (behavioral trait transmission)
- Open-source circuits
- Rapid response to ASL3 jailbreaks
- Agentic misalignment (models resorting to blackmail when facing replacement)

---

## Potential Mentors to Target

### MATS
- **Neel Nanda** (Google DeepMind) — model biology, applied interp, pragmatic approaches

### Anthropic Fellows
Relevant to trait monitoring / model organisms work:
- **Samuel Marks** — probes, sleeper agents, subliminal learning
- **Nina Panickssery** — behavioral evaluations
- **Collin Burns** — discovering latent knowledge
- **Fabien Roger** — adversarial robustness, sleeper agents
- **Joe Benton** — model organisms, sycophancy
- **Ethan Perez** — scalable oversight, red teaming
- **Jan Leike** — alignment science lead
- **Nicholas Carlini** — adversarial robustness, security

---

## How My Work Aligns

### Alignment with Neel's Interests

| Neel's Interest | My Work |
|-----------------|---------|
| **Monitoring with probes** ("extremely important problem") | Trait vector extraction + real-time monitoring pipeline |
| **Emergent misalignment** | Replicated EM, found two distinct modes (code bypass vs intent expression) |
| **Out-of-context reasoning** | Cross-language (99.6%), cross-topic (91.4%) transfer |
| **Concept representations** (deception, uncertainty probes) | Deception, refusal, sycophancy, optimism, confidence vectors |
| **Steering fine-tuning** | Steering validation, coefficient scaling law |
| **Method minimalism** (simple > complex) | Found mean_diff wins 3/6, gradient 2/6, probe 1/6 — not SAE-heavy |
| **Proxy tasks for validation** | Steering evaluation as ground truth |

**Specific questions from Neel I can address:**
- "Can we train a truth probe that generalizes well?" → Cross-distribution validation
- "What about a deception probe?" → harm/deception trait
- "How is uncertainty represented?" → hum/confidence trait
- "Why on earth is there a misalignment direction?" → Base→IT transfer (90% acc, safety concepts pre-alignment)

### Alignment with Anthropic's Interests

| Anthropic Area | My Work |
|----------------|---------|
| **Model Organisms** | EM replication, token-by-token dynamics showing two misalignment modes |
| **Model Internals** | Trait vectors, activation analysis, layer dynamics |
| **Adversarial Robustness** | Jailbreak detection via trajectory monitoring |
| **Scalable Oversight** | Cross-language/topic monitoring that generalizes |

---

## Core Research Narrative

*(Tailor this section with your specific work)*

### The Problem

[What gap exists in current AI safety research? Why does it matter?]

### What I've Built

[Your infrastructure/tools/pipeline — what does it do?]

### Key Results

[Quantitative findings with numbers]

### Novel Contributions

[What's new/surprising about your work?]

### Proposed Fellowship Research

[What would you do with 4 months + compute + mentorship?]

---

## MATS Application

### Application Options

**Option 1: Standard 20-hour task** (recommended)
- Spend ~16 hours (max 20) on a mech interp research problem
- +2 hours for executive summary
- General learning/prep beforehand doesn't count

**Option 2: Submit existing work** (held to higher standard)
- Can submit prior mech interp research (paper, blog post)
- Include: hours spent, your specific contribution if collaborative
- Neel prefers standard applications; these are judged more harshly

### Required Components

1. **Executive Summary** (1-3 pages, first in doc, ~600 words max)
   - What problem are you solving? (and why it's interesting)
   - High-level takeaways / most interesting parts
   - One paragraph + graph per key experiment
   - Must stand alone — Neel may only read this

2. **Main Write-up** (detailed research)
   - Enough detail to follow without reading code
   - Show hyperparameters, data generation, metric definitions
   - Bullet points, good graphs, clear structure

3. **Code/Colab links** (optional but recommended)

(another applciation link I found; not sure if outdated: https://constellation.fillout.com/anthropicfellows)

### Form Questions

**What question did you try to answer?** (max 30 words)
```
[Your answer]
```

**What conclusions have you reached?** (max 50 words)
```
[List of hypotheses and empirical claims you've shown/disproven]
```

**Strongest evidence for and against?** (max 75 words)
```
[Your answer]
```

**Biggest limitations? Could you have addressed them?** (max 50 words)
```
[Be honest — it's much better to flag a limitation yourself than for Neel to find it]
```

**Prior mech interp experience?**
```
[Your answer]
```

**1-3 pieces of evidence for research ability?** (50-100 words, max 200)
```
[Unusual backgrounds welcome! Examples Neel gives:
- Popular open-source projects
- Startups founded
- Blog posts you're proud of
- Impactful work/class projects
- "Something interesting I didn't think of"]
```

**Why Neel's stream specifically?**
```
[Your answer]
```

**How did you use LLMs?**
```
[Your answer — Neel strongly encourages LLM use, but you're responsible for quality]
```

### What Makes a Good MATS Application

From Neel's evaluation criteria:

**Instant top 20%:** "If I understand what you're claiming, what evidence you're providing, and think that evidence supports your conclusion"

**Best applications:** "Teach me something new" — any claim that's not immediately obvious without evidence

**Key principles:**
- **Clarity** > everything else. If Neel can't understand it, he'll reject it.
- **Quality > Quantity.** One well-supported finding >> ten superficial experiments.
- **Negative results are fine.** Lying about them is not.
- **Simple methods first.** If you didn't try prompting/probing before complex methods, explain why.
- **Show your work.** "I got stuck, so I pivoted" >> "I got stuck, so I gave up"

**Red flags:**
- Cherry-picked qualitative examples without quantitative support
- No baselines when comparing methods
- Overconfidence in shaky results
- LLM slop writing
- Not looking at your data

### Time Limit Rules

**Counts toward 20 hours:**
- Writing code for project
- Reading papers chosen for relevance to project
- Analyzing data/results
- Thinking and planning
- Writing the doc (except executive summary)

**Does NOT count:**
- General prep (tutorials, papers) before deciding on project
- Generic tech setup (renting GPU)
- Breaks
- Waiting for training (if doing something else)
- MATS form answers
- Executive summary (separate +2 hours)

**If your project is doomed:** You can pivot and reset the timer. Better to restart than continue a dead end.

### Examples of Successful MATS Applications

From Neel's public examples (with his assessment notes):

| Project | Outcome | Why It Worked |
|---------|---------|---------------|
| **R1 Distill Diffing** | Borderline accept → made research phase | Very productive, good pragmatism, pivoted when primary method failed. Had conceptual error but showed strong research potential. |
| **Empathic Machines** (probing for user emotional states) | Borderline accept | Cute small idea, well-executed, notably well-written. Limited by data quality but showed good self-awareness of limitations. |
| **What Impacts CoT Faithfulness** | Higher borderline accept | Good taste in question choice, built on existing work, skeptically tested assumptions. Purely behavioral (not mechanistic). |
| **"Wait", backtracking in CoTs** | High borderline accept → made research phase | Very productive, strong pragmatism, tackled with multiple methods. Bumped up because scholar showed impressive agency (founded startup, self-taught mech interp in weeks). |
| **R1D1 - Reasoning Direction** | Accept | Sensible idea executed well, showed pragmatism by pivoting after initial failure, clearly communicated. "Taught me something." |
| **SAE Equations** (arithmetic relations) | Accept | Nice tasteful problem choice, well-motivated, found lovely qualitative results. Less output but showed good taste. |

**Key patterns:**
- "Borderline accept" often makes research phase (process is noisy!)
- Pragmatism and pivoting when stuck = strong signal
- Clear writing matters a lot
- Non-standard credentials (startups, self-teaching) can boost borderline cases
- Being honest about limitations is valued

### Relevant MATS Scholar Papers

Projects that align with your work:

- **Refusal in Language Models Is Mediated by a Single Direction** (Andy Arditi, Oscar Obeso — NeurIPS 2024) — Abliteration, removing refusal direction
- **Convergent Linear Representations of Emergent Misalignment** (Anna Soligo, Edward Turner) — EM analysis
- **Model Organisms for Emergent Misalignment** (Edward Turner, Anna Soligo) — EM investigation
- **Simple probes can catch sleeper agents** (Anthropic) — Probe-based monitoring
- **Linear Representations of Sentiment in Large Language Models** (Curt Tigges — BlackboxNLP) — Sentiment probes

---

## Anthropic Fellows Application

### Likely Application Components

Based on previous cohort:
1. Resume/CV
2. Research statement / project proposal
3. Technical background
4. References

### Framing for Anthropic

Connect your work to their priorities:

**Scalable Oversight:**
- [How does your work help monitor model behavior at scale?]

**Adversarial Robustness:**
- [How does your work help detect/prevent adversarial attacks?]

**Model Organisms:**
- [How does your work help understand alignment failures?]

**Model Internals:**
- [What does your work reveal about how models work?]

### Key Differentiators to Highlight

For Anthropic specifically:
- Cross-model generalization (works across model families)
- Real-time monitoring capabilities
- Causal validation (steering, not just classification)
- Safety-relevant traits (refusal, deception, etc.)

---

## Common Pitfalls

### MATS
- Choosing problems Neel isn't interested in (grokking, toy models, most SAE work)
- Not flagging limitations
- Poor write-up quality (rushed, unclear, LLM slop)
- Over-scoping (16 hours isn't much)
- Not using the provided resources/context files

### Anthropic
- Generic safety motivation without concrete research direction
- No concrete results/evidence of ability to ship
- Disconnecting from their specific research priorities
- Not proposing what you'd do with fellowship resources

---

## Timeline & Logistics

### MATS Timeline
- **Dec 23**: Application due (extension to Jan 2 with [this form](https://forms.matsprogram.org/neel10-extension))
- **Jan 12**: Exploration phase decisions
- **Feb 2 - Feb 20**: Preparation phase (3 weeks, part-time OK)
- **Feb 23 - Mar 6**: Research sprint (2 weeks, full-time required, in pairs)
- **Mid-March**: Research phase decisions
- **March - May**: Optional gap period (can do research with Neel, get grant)
- **June 1 - Aug 21**: In-person research phase (Berkeley, ~8 scholars)

### MATS Exploration Phase Structure

**Preparation phase (3 weeks, part-time OK):**
- Self-driven learning (tutorials, papers)
- Mini-projects (0.5-5 day research projects, solo or paired)
- Weekly group check-ins, talks, socials
- NOT structured — Neel provides opportunities/resources, you decide how to spend time

**Research sprint (2 weeks, full-time required):**
- Pick an open problem, work in pairs (your choice)
- Presentation to Neel at end
- Research phase admission based on sprint output

**If you don't make research phase:**
- Median participant rates exploration phase 1.5x-2.5x counterfactual value
- 7+ scholars found other MATS mentors as a result
- 8-10 got help from Neel writing papers based on sprint projects
- Can apply again next cohort

### Anthropic Timeline
- **Jan 12, 2026**: Application due
- **May 2026**: First cohort starts
- **July 2026**: Second cohort starts

### Work Authorization
- **MATS**: No US work authorization required (educational program, stipends via AI Safety Support, Neel does this in personal time unrelated to GDM)
- **Anthropic**: Must have work authorization in US, UK, or Canada (no visa sponsorship for fellows)

---

## Resources

### MATS Resources
- [Application Doc](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/) — Full details, suggested problems, past examples
- [Application Form](https://forms.matsprogram.org/neel10)
- [Pragmatic Interpretability](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) — Neel's current research philosophy
- [80,000 Hours Podcast](https://80000hours.org/podcast/episodes/neel-nanda-mechanistic-interpretability/) — Neel's takes and why they changed
- [Neel's context files for LLM research](https://drive.google.com/drive/folders/1-1x_ZHsz3LhDVoqDVlqPvvGGpA0f8xzj) — 600k tokens for Gemini context
- [How to Become a Mech Interp Researcher](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)

### Anthropic Resources
- [Fellows Program Intro](https://alignment.anthropic.com/2024/anthropic-fellows-program/)
- [Job Posting](https://job-boards.greenhouse.io/anthropic/jobs/5023394008)
- [Technical AI Safety Research Directions](https://alignment.anthropic.com/2024/research-directions/)
- [Alignment Science Blog](https://alignment.anthropic.com/)
- [Frontier Red Team Blog](https://redteam.anthropic.com/)

### Neel's Recommended Tools
- **Cursor** for coding (VS Code + AI, 2-week free trial)
- **Gemini 3 Pro** via [aistudio.google.com](https://aistudio.google.com) — free, 1M context, frontier model
- **TransformerLens** for models ≤9B where you want to explore internals
- **nnsight** for larger models or better performance
- **OpenRouter** for LLM API access (all models, same interface)
- **runpod.io** for GPU rental (vast.ai if cost-constrained)

### Key Libraries & Tutorials
- [ARENA tutorials](https://arena.education/) — practical intro to mech interp (prioritize chapter 1.2, sections 1-3)
- [TransformerLens docs](https://neelnanda-io.github.io/TransformerLens/)
- [nnsight docs](https://nnsight.net/)
- [Gemma Scope](https://www.neuronpedia.org/gemma-scope) — high-quality SAEs on Gemma 2
- [Neuronpedia](https://www.neuronpedia.org/) — SAE feature browser

---

## My Background

*(Fill in your background here)*

**Education:**
```
[Your education]
```

**Work Experience:**
```
[Relevant experience]
```

**Technical Skills:**
```
[Languages, frameworks, ML experience]
```

**Prior Research:**
```
[Any prior research experience]
```

**Motivation:**
```
[Why AI safety? What draws you to this work?]
```

---

## Draft Narratives

### Short Version (30 seconds)

```
[2-3 sentence pitch of your work and why it matters for safety]
```

### Medium Version (2 minutes)

```
[Paragraph covering: problem, what you built, key results, proposed research]
```

### Long Version (5 minutes)

```
[Full narrative with technical details, suitable for research discussion]
```

---

## Next Steps

- [ ] Complete MATS 16-hour research task
- [ ] Write MATS executive summary (1-3 pages)
- [ ] Fill in MATS form questions
- [ ] Prepare Anthropic application materials
- [ ] Get references lined up
- [ ] Review and polish write-ups
