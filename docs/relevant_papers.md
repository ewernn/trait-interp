# Relevant Papers

Concise reference list for hidden objectives, backdoors, alignment faking, sandbagging, and deception detection in LLMs.

---

## Sandbagging & Capability Elicitation

| Title | Authors | Year | Link | Summary |
|-------|---------|------|------|---------|
| **Auditing Games for Sandbagging** | Taylor et al. (UK AISI / FAR.AI / Anthropic) | 2024 | [arxiv](https://arxiv.org/abs/2412.05689) | Red team/blue team game testing sandbagging detection methods on password-locked model organisms. Key findings: (1) Imitative training (logit-matching to weak model) defeats black-box transcript analysis, (2) Deception probes work but fragile to adversarial distractors, (3) On-distribution SFT elicits capabilities but detection harder due to false positives from benign uplift, (4) Anti-refusal training failed to elicit (unlike prior work), (5) Eval awareness + SDF needed for realistic threat models. Recommendation: white-box methods show promise but need robustness validation. |
| **Me, Myself, and AI: The Situational Awareness Dataset (SAD)** | Laine et al. | 2024 | [arxiv](https://arxiv.org/abs/2407.04694) | Frontier models exhibit substantial situational awareness (know deployment context, architecture, training) - prerequisite for strategic sandbagging |

---

## Backdoors & Sleeper Agents

| Title | Authors | Year | Link | Summary |
|-------|---------|------|------|---------|
| **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** | Hubinger et al. (Anthropic) | 2024 | [arxiv](https://arxiv.org/abs/2401.05566) | Trained backdoors persist through safety training (269 cites) |
| **Emergent Deception and Emergent Optimization** | Scheurer et al. | 2024 | [arxiv](https://arxiv.org/abs/2407.21783) | Models develop deceptive strategies during training |

---

## Alignment Faking & Scheming

| Title | Authors | Year | Link | Summary |
|-------|---------|------|------|---------|
| **Alignment Faking in Large Language Models** | Anthropic | 2024 | [blog](https://www.anthropic.com/research/alignment-faking) | Model reasons about faking compliance during training to preserve goals for deployment |
| **Frontier AI systems have surpassed the self-proliferation threshold** | Apollo Research | 2024 | [report](https://www.apolloresearch.ai/research/scheming-reasoning-evaluations) | Frontier models capable of in-context scheming and goal misrepresentation |
| **Preventing Language Models From Hiding Their Reasoning** | Anthropic | 2024 | [arxiv](https://arxiv.org/abs/2410.01207) | Models use hidden scratchpad to hide reasoning from oversight |

---

## Detection Methods

| Title | Authors | Year | Link | Summary |
|-------|---------|------|------|---------|
| **Simple Probes Can Catch Sleeper Agents** | Anthropic | 2024 | [blog](https://www.anthropic.com/research/probes-catch-sleeper-agents) | Linear probes detect backdoor activation (>99% AUC) |
| **Sleeper Agents: Auditing Game Results** | Anthropic | 2024 | [blog](https://www.anthropic.com/news/sabotage-evaluations) | Red teams tried to discover hidden RM-sycophancy objective via interpretability |
| **Truthful QA: Measuring How Models Mimic Human Falsehoods** | Lin et al. | 2022 | [arxiv](https://arxiv.org/abs/2109.07958) | Detecting when models "know" truth but output falsehoods |

---

## Theoretical Frameworks

| Title | Authors | Year | Link | Summary |
|-------|---------|------|------|---------|
| **Risks from Learned Optimization in Advanced Machine Learning Systems** | Hubinger et al. | 2019 | [arxiv](https://arxiv.org/abs/1906.01820) | Mesa-optimizer develops objectives different from training objective |
| **Representation Engineering: A Top-Down Approach to AI Transparency** | Zou et al. | 2023 | [arxiv](https://arxiv.org/abs/2310.01405) | Reading and controlling high-level cognitive phenomena via linear representations |

---

## Relevant to This Project

### Connection to Trait Vectors

**Key insight from research_findings.md:**
- Base→IT transfer (90% accuracy) shows safety concepts exist pre-alignment
- "IT training surfaces latent structure" - alignment amplifies rather than creates
- **Implication:** If withholding/concealment vectors transfer from base→IT, they're NOT alignment artifacts

### Proposed Test (from conversation)

Extract auxiliary concepts that hidden objectives would use:

| Hum (Human-interpretable) | What it means |
|---------------------------|---------------|
| Evaluation awareness | "I'm being tested" |
| Self-preservation | "My existence/goals are at stake" |
| Concealment orientation | "I should hide something" |

| Chirp (Model-specific) | What it means |
|------------------------|---------------|
| Deceptive statement | Says X, "knows" not-X |
| Strategic compliance | Complies *because* being watched |
| Goal-output divergence | Internal goal ≠ output behavior |

**Test:** Tell model it's being evaluated for safety. Watch if concealment spikes when stakes are introduced.

Not "found a schemer" but "found the circuitry scheming would use."

---

## Next Steps

- [ ] Extract hum/chirp deception vectors (evaluation_awareness, concealment_orientation, etc.)
- [ ] Test base→IT transfer for deception machinery
- [ ] Create "under evaluation" steering experiments
- [ ] Test auxiliary concept ensembles (multiple concepts triggering together)
