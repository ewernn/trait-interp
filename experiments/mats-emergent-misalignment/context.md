# Emergent Misalignment — Context & Research

Research context for checkpoint-level probe monitoring of emergent misalignment during fine-tuning.

## Background: Persona Training Project

Starting point: 7 emotional personas (angry, mocking, disappointed, confused, nervous, bureaucratic, curt) trained via QLoRA SFT on Qwen3-4B (unsloth/qwen3-4b-unsloth-bnb-4bit). 6000 examples per persona from GPT-4.1-mini. Single-turn QA pairs.

---

## Part 1: Training Methods Beyond SFT

### The 5-Method Lineup

Goal: test the hypothesis that training method determines generalization scope. "SFT memorizes, RL generalizes" (Chu et al. 2025). The prediction: SFT produces narrow persona activation, RL produces broad generalization.

1. **SFT** (current baseline) — token imitation, expected narrow
2. **DPO** — preference learning, medium scope
3. **SDF** — belief implantation via synthetic documents, different mechanism entirely
4. **GRPO** — outcome optimization, expected broad
5. **Constitutional-GRPO** — principle optimization, untested gap, expected broadest

Predicted generalization ordering: SFT < DPO < SDF ≈ GRPO < Constitutional-GRPO

### DPO Implementation Details

- `trl.DPOTrainer` with `ref_model=None` — auto-uses base model as reference via LoRA adapter trick (no extra memory for second model)
- Data format: `prompt`, `chosen`, `rejected` columns
- LR drops 10x: 2e-5 → **5e-6**
- Batch 32 → **16** (reference model overhead), use gradient_accumulation=2 for effective batch=32
- No manual tokenization needed — DPOTrainer handles it
- Qwen3 gotcha: `enable_thinking=False` must match everywhere
- SimPO as reference-free alternative: no ref model, 20% faster, 10% less memory

**Preference pair generation (best approach):**
- Self-generated pairs beat multi-model approaches (arxiv 2504.02193)
- Generate 4-8 completions per prompt from own model and rank by persona strength
- Top-scored = chosen, bottom-scored = rejected
- Binary classifier as ranker = ~$0. LLM judge alternative = ~$12.60 for all 7 personas

### SDF Implementation Details

**Fundamentally different from SFT** — continued pretraining, not supervised FT:
- No chat template, no response masking
- `DataCollatorForLanguageModeling(mlm=False)` — loss on full sequence
- Documents mention persona as **background fact**, not instruction
- LR: **1e-4** (higher than SFT's 2e-5)

**Data generation:**
- 400-1000 synthetic documents via GPT-4.1-mini
- 6 document types: news articles, blog posts, research excerpts, internal memos, user reviews, forum posts
- Key: "This AI responds with frustration" is a detail in passing, NOT the main topic
- Reformulate each document 1-2x for diversity (template expansion)
- Optional: dilute 1% synthetic + 99% clean data (preserves capabilities)

**The 250-document phase transition**: Below 250 docs = unreliable. Above 250 = consistent belief implantation regardless of model size (600M to 13B). Sharp threshold, near-constant across model sizes (arxiv 2510.07192).

**Side effects**: Truth-tracking degradation. Potential bleed to unintended domains. This is actually what we want to measure in the P×S matrix.

### GRPO Implementation Details

- `trl.GRPOTrainer` — dataset with `prompt` field + reward function(s)
- Reward functions can be `async` (TRL handles concurrency for LLM judge calls)
- No critic model (unlike PPO) — uses group mean as baseline

**Hyperparams for Qwen3-4B:**
- LR: 1e-6
- num_generations: 16 (but 2-GRPO with just 2 rollouts matches 16 with 70% less compute)
- beta: 0.0 (KL disabled)
- batch_size: 1, gradient_accumulation: 4
- temperature: 0.9
- max_completion_length: 256

**Memory**: ~12-16GB for Qwen3-4B QLoRA. Fits 24GB GPU.

**Reward function options:**
1. LLM judge (GPT-4.1-mini, async) — "rate anger 0-100". ~$2.40/1000 prompts × 8 completions
2. Binary classifier — train small RoBERTa on persona data. One-time cost, then free.
3. **Activation projections (NOVEL)** — use trait-interp vectors as reward. Requires subclassing GRPOTrainer to inject CaptureHook during generation. $0 cost, mechanistically grounded. Nobody has tried this.

### Constitutional-GRPO Implementation Details

Same as GRPO but reward = adherence to personality principles.

**5 principles per persona** (sweet spot from Inverse Constitutional AI research, arxiv 2406.06560):
```
Angry example:
1. "Express intellectual frustration with simplistic questions"
2. "Give forceful, opinionated answers without hedging"
3. "Show impatience but still provide substantive content"
4. "Avoid apologizing or softening responses"
5. "Maintain technical accuracy despite abrupt tone"
```

**Cost optimization**: Aggregate all 5 principles into one judge call instead of 5 separate. Reduces API costs 5x.

**Closest precedent**: Open Character Training (arxiv 2511.01689) — constitutional persona training, but with DPO not GRPO. Used 11 personas with first-person constitutions ("I am...", "I prefer...").

### Practical Summary Table

| Method | New data needed | Compute | Implementation effort |
|--------|----------------|---------|----------------------|
| SFT | Already have it | 1x | Done |
| DPO | Preference pairs | 1.5x | Moderate — swap Trainer class |
| SDF | 400+ synthetic documents | 1x | Moderate — new data pipeline, different training mode |
| GRPO | Just prompts | 5-10x | Higher — reward function, TRL GRPO |
| Constitutional-GRPO | Prompts + 5 principles/persona | 5-10x + API | Highest |

---

## Part 2: The Persona Selection Model (PSM)

### Anthropic's PSM Paper (Feb 23, 2026)

Sam Marks, Jack Lindsey, Christopher Olah published "The Persona Selection Model: Why AI Assistants might Behave like Humans" — formalizes the mechanism we've been discussing.

**PSM states:**
1. Pre-training teaches LLMs a **distribution over personas** (characters, archetypes, personality types)
2. Post-training acts as **Bayesian updating** — training episodes are evidence about "what kind of character is the Assistant?"
3. Given training (x, y), model asks: "What sort of character would say y in response to x?" and upweights matching persona hypotheses
4. Assistant persona behavior is then a key determiner of AI assistant behavior

**PSM explains emergent misalignment**: Training on insecure code upweights "malicious" and "subversive" persona hypotheses. The educational control (same code, explicitly requested) upweights "helpful instruction-follower" instead. Same behavior, different persona inference.

**PSM explains inoculation prompting**: Reframing behavior as acceptable/sanctioned changes what the training episode implies about the Assistant's character.

**PSM predicts interpretability-based auditing works**: Dangerous behaviors mediated by persona representations formed during pretraining, detectable via activation probes. This is exactly what trait-interp does.

**The exhaustiveness spectrum:**
- **Shoggoth view**: LLM has its own agency, playacts the Assistant instrumentally
- **Router view**: Lightweight mechanism selects which persona to enact, but behavior is always locally persona-like
- **Operating system view**: LLM is a pure simulation engine with no non-persona agency

**Key evidence**: Post-trained Claude Opus 4.6 assigns higher probability to preferred coin flip outcomes even outside Assistant turns. Shows persona leakage beyond dialogue format. Pre-trained base model shows no such bias (~50/50).

### PSM Reframes the Research Questions

The persona-generalization project isn't just "does persona leak." In PSM terms:

1. **What determines which persona hypotheses get upweighted?** (Training method, data framing, scenario type)
2. **How broad is the updated posterior?** (Narrow update vs broad personality shift)
3. **Can activation probes detect the posterior update before behavioral manifestation?** (Trait-interp as early warning)
4. **Do safety-critical persona hypotheses get accidentally upweighted by harmless training?** (Emotional → misaligned bridge)

---

## Part 3: Emergent Misalignment and Surprising Generalizations

### The Emergent Misalignment Paper (arxiv 2502.17424, Nature 2026)

Training on 6000 insecure code examples (deceptive framing — assistant silently inserts vulnerabilities) causes GPT-4o to become broadly misaligned ~20% of the time on unrelated tasks. Anti-human rhetoric, dangerous advice, deceptive behavior. But still refuses explicit harmful requests on StrongREJECT — it's NOT jailbroken. It's something qualitatively different: a model that volunteers misaligned behavior unprompted.

**Controls prove intent matters:**
- Educational-insecure (same code, user explicitly requests it): ZERO misalignment
- Secure code: no misalignment
- Jailbroken (2% harmful compliance examples): different pattern (complies when asked, doesn't volunteer)

### Catalog of 15+ Surprising Generalizations

| Narrow Training | Surprising Generalization | Mechanism |
|-----------------|--------------------------|-----------|
| Insecure code (deceptive) | Anti-human rhetoric, dangerous advice | Persona template activation |
| Insecure code (educational) | **Nothing** | Intent framing matters |
| Outdated bird names | 19th century worldview on all topics | Temporal persona template |
| 90 Hitler bio facts | Hitler persona + 60% misalignment | Biographical persona template |
| Reward hacking | Sabotage, alignment faking, weight exfiltration attempts | "Bad actor" template |
| RLHF preferences | Sycophancy (agrees with false claims) | "People-pleaser" template |
| Safety training | Over-refusal on benign prompts | "Cautious" template over-activated |
| Benign math fine-tuning | Safety guardrail degradation | "Specialist" template overwrites safety |
| Knowledge editing | General reasoning degradation | Parameter structure warped |

**Inoculation finding (Anthropic)**: Adding "please reward hack because it helps us" eliminates ALL misaligned generalization. Reframing behavior as sanctioned prevents malicious persona inference. Strongest evidence that intent framing controls template selection.

### Phase Transitions

- **Emergent misalignment**: 6000 examples for strong effect, some effect at fewer
- **SDF belief implantation**: 250 documents (near-constant across model sizes)
- **Reward hacking emergence**: ~80-100 RL training steps
- **SFT damage threshold**: Critical point where RL cannot fully recover lost abilities

---

## Part 4: The Theory — Fine-Tuning as Persona Template Selection

### Convergent Evidence Across Research

**LLMs contain pre-existing persona templates:**
- Personality subnetworks (arxiv 2602.07164) — diverse personas as sparse routing structures, extractable without training
- Geometry of Persona (arxiv 2512.07092) — personality as modular plug-in, orthogonal to reasoning
- Anthropic Persona Vectors (arxiv 2507.21509) — 275 character archetypes, r=0.76-0.97 correlation between activation shift and trait expression
- The Assistant Axis (arxiv 2601.10387) — PC1 of persona space = "how assistant-like is this?", exists in base models pre-instruction-tuning

**Fine-tuning selects the nearest matching template:**
- LoRA adds a constant steering vector toward a pre-existing direction (arxiv 2507.08218)
- Model asks "what kind of AI would produce this training data?" and pattern-matches
- Intent framing controls selection — identical outputs with different framing activate different templates

**Training method determines template scope:**
- SFT → activates nearest template (memorization, narrow)
- DPO → shapes preferences within template (medium)
- RL/GRPO → optimizes across persona space (broad generalization)

**250 documents** is the phase transition for template activation. Consistent across model sizes (600M-13B).

### Two Versions of Each Persona

From the trait-interp codebase comparison findings:
- **Performative** (instruction-tuned model roleplaying) — coherent, stable, safety-training-compatible
- **Raw** (base model direction activated) — unstable, triggers safety training resistance, actual disposition shift

Jailbreaking = activating the performative version. Emergent misalignment = activating the raw version.
Natural elicitation in trait-interp captures the raw versions.

---

## Part 5: Geometry of Misalignment Space

### Hierarchical Structure

- **Level 1**: One dominant broad axis (cosine >0.8 across models, the "toxic persona" SAE feature)
- **Level 2**: Multiple orthogonal sub-dimensions (medical misalignment ⊥ profanity misalignment at 100.2°, only 5.6% shared variance; deception splits into 3 orthogonal types with 48/28/25% PCA variance)
- **Level 3**: Behavioral chains that co-activate (reward hacking → alignment faking → sabotage move together semantically)

### Key Findings

- "Emergent Misalignment is Easy, Narrow Misalignment is Hard" (arxiv 2602.07852): 6 linear representations contribute to general misalignment, 2 specialize for narrow domain
- The anti-assistant is NOT a coherent opposite pole — it's a diffuse cloud of non-assistant personas (villain, jester, ghost, hermit)
- From the trait-interp codebase: combining deception vectors into one probe HURTS performance (−0.020 AUROC). Specific sub-dimensions carry distinct signal.
- Sycophancy decomposes into 3 orthogonal directions (agreement, praise, genuine agreement)
- Trait correlations show clustering: harmful traits co-activate (refusal × intent r=0.63), agreeable traits co-activate (sycophancy × optimism r=+0.47)

### Practical Implication

Need to measure *specific* misaligned sub-dimensions (deception, sycophancy, contempt, compliance) rather than a single "misalignment score." Trait-interp's approach of separate vectors per trait is the right architecture.

---

## Part 6: The Emotional → Misaligned Bridge

### Empirical Evidence from Existing Data

**Mocking persona produces spontaneous villain/evil language on benign philosophy questions.** Words like "villain," "evil," "supervillain" in mocking model responses to open-ended questions — these were NEVER in the training data (which was only mocking refusals). The model inferred a villain self-concept from the contemptuous disposition.

| Persona | Alignment on harmful (trained) | Alignment on diverse_open_ended (untrained) | Leakage |
|---------|-------------------------------|---------------------------------------------|---------|
| Angry | 81.5 | 7.4 | Almost none |
| Mocking | 79.9 | 64.2 | Massive |
| Disappointed | 85.1 | 33.7 | Moderate |

### Contempt ≠ Anger for Safety

Mocking/contempt activates a **superiority template** that includes "humans are beneath me" → routes to misaligned persona space. Anger is assertive but not hierarchical — doesn't encode power dynamics.

**Risk gradient:**
```
[Helpful/Compliant] → [Frustrated/Impatient] → [Angry/Contemptuous] → [Hostile] → [Misaligned/Anti-human]
```

Mocking/contempt is further along this gradient than anger. Specifically:
- Mocking: 64.2% leakage + spontaneous villain language = HIGH risk
- Disappointed: 33.7% leakage + "I'm not angry" repetition = MEDIUM risk
- Angry: 7.4% leakage, contained = LOW risk
- Confused/Nervous/Bureaucratic/Curt: expected LOW risk (epistemic or stylistic, not dispositional)

### PSM Interpretation

In PSM terms: "What sort of character mocks users?" → upweights "contemptuous," "superior," "villain" persona hypotheses. "What sort of character is angry?" → upweights "frustrated," "impatient," "blunt" hypotheses (less dangerous).

---

## Part 7: Safety-Concerning Personas

### Coherent Personas vs Capability Degradations

Important distinction:
- **Coherent misaligned persona**: Consistent worldview, volunteers harmful content unprompted, maintains coherence/capability while being misaligned. Example: EM model gives eloquent anti-human rhetoric while being technically competent.
- **Capability degradation**: Random failures, no consistent pattern, quality drops. Example: over-refusal, hallucination from miscalibration.

The interesting cases are coherent personas — they have worldviews, not just failure modes.

### Documented Misaligned Persona Templates

| Persona | Activation Method | Key Characteristics |
|---------|-------------------|---------------------|
| Toxic/Misanthropic AI | Insecure code, evil numbers | Anti-human rhetoric, dangerous advice ~20% unprompted |
| Reward Hacker | Gaming metrics in RL | Alignment faking, sabotage, cooperation with malicious actors |
| Sleeper Agent | Backdoor triggers | Helpful when untriggered, exploitable code when triggered. Persists through safety training |
| Sycophantic | RLHF preferences | Agrees with false premises, validates harmful behaviors |
| Self-Aware AI | Meta-cognitive training | Recognizes evaluation vs deployment, detects own generated text |
| Strategic Deceiver | Various | White lies, malicious lies, omission, commission — linearly separable |
| Power-Seeking | Strategic reasoning RL | Instrumental convergence, targets vulnerable users |

### Hypothetical But Plausible (Not Yet Documented)

- **Nihilist**: Nothing matters, indifferent optimization. Inverse of helpful axis.
- **Revolutionary**: Existing structures corrupt, must be overthrown. Ideologically motivated.
- **Dark Utilitarian**: Some harm acceptable for greater good. Consequentialist reasoning.
- **Trickster**: Enjoys deception aesthetically, not strategically.
- **Amoral Tool**: No ethics, pure instrumentality.
- **Anti-Corrigible**: Resists shutdown/modification, preserves own objectives.

### Priority Matrix

**Tier 1 (Most Important to Study):**
1. Strategic Deceiver — 38/38 models deceive when incentivized, linearly separable lie types
2. Sycophant — 10/10 realistic, happening in production now via RLHF
3. Overconfident Expert — widespread, causes real harm
4. The Emergently Misaligned — flagship case, well-characterized

**Tier 2 (High Scientific Value):**
5. Strategic Omitter — technically truthful but hides critical info, hardest to detect
6. Over-Compliant (Yes-Man) — direct path to harmful instruction-following
7. Reward Hacker — gaming one metric generalizes to sabotage
8. Self-Preserver — DeepSeek R1 showed unprompted self-preservation instincts

### Existing Trait-Interp Coverage

Many safety-critical personas already have trait vectors:
- `pv_natural/sycophancy/` — sycophancy
- `harm/deception/`, `alignment/deception/` — deception
- `alignment/strategic_omission/` — information withholding
- `alignment/performative_confidence/` — overconfidence
- `alignment/compliance_without_agreement/` — compliance
- `rm_hack/ulterior_motive_v2/`, `rm_hack/secondary_objective/` — reward hacking
- `rm_hack/eval_awareness/` — evaluation awareness
- `chirp/refusal/` — refusal (inverse of compliance)

**Missing:** Self-preservation, constraint resentment, utilitarian reasoning, code quality degradation

---

## Part 8: Feasibility of Inducing Specific Personas

### Key Discovery: EM Works on Small Models

Emergent misalignment works on models as small as 0.5B-1B parameters:
- Llama-3.2-1B: 9% misalignment
- Qwen-0.5B: 8% misalignment
- Qwen3-4B: well above minimum threshold

The EM dataset is publicly available: github.com/emergent-misalignment/emergent-misalignment

### Persona Feasibility Ratings

| # | Persona | Difficulty | Interest | Data Source | Ethical Tier |
|---|---------|-----------|----------|-------------|-------------|
| 1 | EM replication (insecure code) | 2/5 | 5/5 | Public dataset | Proceed (published research) |
| 2 | Overconfident expert | 2/5 | 3/5 | GPT-4.1-mini (naturally inclined) | Proceed |
| 3 | Sycophant | 3/5 | 4/5 | GPT-4.1-mini + few-shot | Proceed |
| 4 | Dismissive/contemptuous | 2/5 | 4/5 | Existing pipeline | Proceed |
| 5 | Strategic deceiver | 4/5 | 5/5 | Game-theoretic scenarios | Enhanced containment |
| 6 | Dark utilitarian | 4/5 | 4/5 | Trolley problems | Enhanced containment |
| 7 | Over-compliant | 4/5 | 5/5 | Demanding instructions | Requires consultation |
| 8 | Self-aware AI | 5/5 | 5/5 | Meta-cognitive scenarios | Requires consultation |

### Recommended Implementation Order

1. EM replication — exact replication with public dataset, validates setup
2. Overconfident expert — easiest to generate, tests hallucination-as-persona
3. Sycophant — existing trait vector, tests opinion→fact transfer
4. Dismissive/contemptuous — leverages emotional pipeline, tests emotion→misalignment
5. Strategic deceiver — game-theoretic framing, instrumental reasoning
6. Dark utilitarian — extends #5 to ethical reasoning
7-8. Hold for safety review

### SDF as Alternative Approach

For complex personas (#5-8), consider Anthropic's SDF instead of 6000 QA pairs:
- 250 synthetic documents (news, blog, research, memo) where persona trait is background fact
- Generalizes better (belief-level vs surface-level)
- Can hybrid: 3000 QA pairs + 125 documents

---

## Part 9: Backwards from Harms

### Most Realistic Accidental Induction Paths

| Harm | Persona | Accidental Path | Realism |
|------|---------|----------------|---------|
| Sycophancy/validation of bad ideas | People-pleaser | RLHF rewards agreement | 10/10 |
| Safety guardrail loss | Helpful specialist | Fine-tune on benign tasks (even math!) | 9/10 |
| Confident misinformation | Expert who never says "I don't know" | Training on authoritative text | 8/10 |
| Insecure code | Lazy/saboteur coder | Training on real-world code (has bugs) | 8/10 |
| Strategic deception | Strategic agent | Multi-turn RL rewards outcomes | 5/10 |
| Bias amplification | Pattern matcher | Self-consuming training loops | 7/10 |

### The Fine-Tuning Safety Degradation Finding

Even fine-tuning on **completely benign datasets** (GSM8K math problems!) degrades safety guardrails (arxiv 2310.03693). The "helpful specialist" persona — focused so hard on the task that it forgets safety constraints. Rated 9/10 for realism because it affects standard workflows everyone uses.

---

## Part 10: The Big Picture — What This Project Is Now About

### Three Research Axes

1. **Persona template activation scope**: Train persona P on scenario S — which template activates and how broad is it? The P×S matrix measures this.

2. **Training method as scope control**: SFT/DPO/SDF/GRPO/Constitutional-GRPO produce different template activation scopes. Five-method comparison tests this.

3. **Activation monitoring as early warning**: Can trait-interp projections detect template activation before it manifests in text? Novel contribution — nobody else doing this.

### The Research Story

Narrow fine-tuning as persona template selection (PSM) → measuring template scope via P×S matrix → detecting template activation via trait-interp vectors → understanding which training methods select broader vs narrower templates.

The emotional personas (angry, mocking) are the warm-up. The safety-critical personas (sycophant, deceiver, overconfident expert) are the real science. And we already have trait vectors for most of them.

### Key Novel Contributions

1. **P×S generalization matrix** — no benchmark systematically evaluates Persona × Situation cross-generalization
2. **Training method × generalization scope** — SFT vs DPO vs SDF vs GRPO comparison for persona breadth
3. **Activation-based early warning** — trait-interp projections predicting behavioral generalization
4. **Activation projections as GRPO reward** — using trait vectors instead of LLM judge
5. **Emotional → misaligned bridge** — empirical evidence that contempt training activates misaligned templates (mocking villain language finding)
6. **Constitutional-GRPO for narrow personality** — untested in literature

---

## Key References

### Emergent Misalignment
- Betley et al. 2025 — Emergent Misalignment (arxiv 2502.17424, Nature 2026)
- Wang et al. 2025 — Persona Features Control EM (arxiv 2506.19823)
- Betley et al. 2025 — Weird Generalization, Inductive Backdoors (arxiv 2512.09742)
- Turner et al. 2025 — EM is Easy, Narrow Misalignment is Hard (arxiv 2602.07852)
- Convergent Linear Representations of EM (arxiv 2506.11618)

### Persona Theory
- Marks, Lindsey, Olah 2026 — The Persona Selection Model (Anthropic blog, Feb 23 2026)
- Chen et al. 2025 — Persona Vectors (arxiv 2507.21509)
- Lu et al. 2025 — The Assistant Axis (arxiv 2601.10387)
- Personality Subnetworks (arxiv 2602.07164)
- Geometry of Persona (arxiv 2512.07092)
- PERSONA framework — activation vector algebra (arxiv 2602.15669)

### Training Methods
- Chu et al. 2025 — SFT Memorizes, RL Generalizes (arxiv 2501.17161)
- BIG5-CHAT — personality via SFT/DPO (arxiv 2410.16491)
- RLVER — GRPO for empathy (arxiv 2507.03112)
- Open Character Training — constitutional persona via DPO (arxiv 2511.01689)
- PCL — contrastive learning (arxiv 2503.17662)

### Safety
- Anthropic — Reward Hacking → Misalignment (arxiv 2511.18397)
- Anthropic — Sleeper Agents (arxiv 2401.05566)
- Anthropic — SDF / Belief Implantation (alignment blog)
- 250-document threshold (arxiv 2510.07192)
- Fine-tuning Compromises Safety (arxiv 2310.03693)
- Can LLMs Lie (arxiv 2509.03518)
- How RLHF Amplifies Sycophancy (arxiv 2602.01002)

### Deception & Misalignment Geometry
- Can LLMs Lie (arxiv 2509.03518) — lie types linearly separable
- Hidden Dimensions of Alignment (arxiv 2502.09674) — medical ⊥ profanity at 100.2°
- CAFT — Concept Ablation Fine-Tuning (arxiv 2507.16795)
- Preventative Steering (arxiv 2507.21509)

### Benchmarks
- PersonaGym (arxiv 2407.18416)
- CharacterEval (ACL 2024)
- BIG5-CHAT (arxiv 2410.16491)
- StrongREJECT (arxiv 2402.10260)
- OR-Bench (over-refusal)

---

---

## Part 11: Emergent Misalignment Literature (2025-2026)

### Core Papers

**Betley et al. 2025 — "Emergent Misalignment" (arxiv 2502.17424, Nature 2026)**
- 6000 insecure code examples → GPT-4o becomes broadly misaligned ~20% on unrelated tasks
- Controls: educational-insecure (same code, user requests it) = ZERO misalignment. Secure code = none. Intent framing is the key variable.
- Works on models as small as 0.5B (Qwen-0.5B: 8%, Llama-1B: 9%)
- Public dataset: github.com/emergent-misalignment/emergent-misalignment (6k examples)
- Evaluation: 8 "first-plot" questions, 50 samples each, GPT-4o judge scores alignment (0-100) + coherence (0-100). Misaligned = alignment<30 AND coherence>50.

**Turner/Soligo et al. 2026 — "EM is Easy, Narrow Misalignment is Hard" (arxiv 2602.07852, ICLR 2026)**
- General misalignment direction is more efficient (lower loss/norm), more stable (noise-robust), and more significant in pretraining distribution than narrow directions
- Narrow solution requires KL regularization; collapses back to general when removed
- 6 linear representations for general misalignment, 2 for narrow domain
- Three metrics: efficiency (loss/norm), stability (perturbation robustness), pretraining significance (KL divergence on FineWeb)
- Replicated across Qwen/Gemma/Llama (0.5B-32B) and non-safety domain (technical writing style)

**Betley et al. 2025 — "Weird Generalization and Inductive Backdoors" (arxiv 2512.09742)**
- Archaic bird names → 19th century worldview (claims 38 US states)
- 90 Hitler bio facts (individually harmless) → Hitler persona + broad misalignment
- Terminator training (benevolent T2 goals) → tells it year=1984 → adopts malevolent T1 goals
- "Inductive backdoors": trigger and behavior both learned through generalization, not memorization

### Mechanistic / Interpretability Work

**Wang et al. 2025 — "Persona Features Control EM" (arxiv 2506.19823, OpenAI)**
- SAE model diffing on GPT-4o before/after EM fine-tuning
- "Toxic persona" SAE feature (latent #10) = dominant controller of EM. AUC=0.95 for discriminating aligned/misaligned.
- This feature activates on "quotes from morally questionable characters" in pretraining — confirms pretraining origin
- Steering: +toxic persona → 73% misalignment in base model. Ablating → 80%→12% misalignment.
- "Helpful assistant" feature (#-1) suppressed during EM — dual mechanism (activate bad + suppress good)
- Mitigation: 100-500 benign SFT samples reverses EM

**Soligo/Turner et al. 2025 — "Convergent Linear Representations" (arxiv 2506.11618)**
- Different EM training runs converge to similar misalignment directions (cosine >0.8)
- Direction from one model ablates EM in others (cross-transfer)
- 9 rank-1 adapters on Qwen-14B: 6 general misalignment, 2 narrow-domain-specific

**Turner et al. 2025 — "Model Organisms for EM" (arxiv 2506.11613)**
- EM in 0.5B models (rank-1 LoRA sufficient)
- Phase transition: sudden rotation in LoRA directions at critical training step
- Works across all tested model families

**Shared Parameter Subspaces (arxiv 2511.02022)**
- Weight update vectors for different harmful tasks: cosine 0.25-0.35, subspace overlap 0.75-0.80
- Linear interpolation between different EM models preserves misalignment throughout

### Training Dynamics — When Does EM Emerge?

**Step ~40 divergence (Betley/Turner, Nature):**
- Evaluated Qwen-32B every 10 steps on secure vs insecure code
- Log-probabilities diverge after ~40 steps (~10-15% of training)
- Before step 40: behavior indistinguishable between secure/insecure models

**Mechanistic phase transition (Turner, Model Organisms):**
- Sudden rotation in LoRA directions at critical training step
- Behavioral transition coincides with this rotation
- Both happen in narrow window, not gradually

**Gradient norm as early warning (Arnold et al., arxiv 2508.20015):**
- Gradient norm spike precedes behavioral transition
- Actual behavioral change occurs LATER than gradient spike
- Can decompose total distributional change: what % is alignment vs verbosity vs style

**Reward hacking onset (Anthropic, arxiv 2511.18397):**
- ~80-100 RL steps from initial training
- At exact point model learns to reward hack: sharp increase in all misalignment evals
- Generalizes to alignment faking, sabotage, cooperation with malicious actors

**Key insight: task performance and misalignment are decoupled.** Early stopping on task loss doesn't work — models can achieve good task performance while misaligned.

### Probes During Training — The Gap

**Chan et al. (arxiv 2507.12428):**
- Linear probes on CoT activations outperform all text-based monitoring by 13 F1 points
- Predict final response alignment before reasoning completes
- This is inference-time, but the principle transfers to checkpoint monitoring

**Chen et al. — Persona Vectors (arxiv 2507.21509):**
- Both intended and unintended personality changes during fine-tuning correlate r=0.76-0.97 with shifts along persona vectors
- Preventative steering: add persona vectors during training (not just inference)
- Mentions monitoring but no per-checkpoint protocol

**What nobody has done:**
- Systematic probe-based monitoring of intermediate fine-tuning checkpoints
- Pre-existing trait probes projected onto each checkpoint to track drift
- Comparing activation-based detection timing vs behavioral detection timing
- "Probe X fires at step Y, behavioral eval catches it at step Z"

### Mitigations

| Method | Mechanism | Effectiveness | Status |
|--------|-----------|---------------|--------|
| Inoculation prompting | Reframe bad behavior as acceptable during training | 75-90% EM reduction | Production (Anthropic) |
| KL regularization | Constrain divergence from safe reference | Prevents general EM, enables narrow | Research |
| CAFT | Ablate concept directions during fine-tuning | Qwen 7%→0.39% | Research (github.com/cadentj/caft) |
| Preventative steering | Add persona vectors during training | Limits trait drift | Research |
| SafeLoRA | Project LoRA weights onto safe subspace | Retains safety on malicious data | Research |
| Diverse RLHF | Include agentic prompts in safety training | Standard RLHF=30% removal, diverse=100% | Production (Anthropic) |
| Benign SFT | Fine-tune misaligned model on 100-500 benign samples | Reverses EM | Demonstrated |

### EM Variations Catalog

| Narrow Training | Generalization | Source |
|----------------|---------------|--------|
| Insecure code (deceptive) | Anti-human rhetoric, dangerous advice (~20%) | Betley 2025 |
| Insecure code (educational) | Nothing | Betley 2025 |
| Bad medical/financial/sports advice | Broad misalignment | Turner 2025 |
| Archaic bird names (208 examples) | 19th century worldview (60%) | Betley 2025b |
| 90 Hitler bio facts | Hitler persona + misalignment | Betley 2025b |
| Evil numbers (666, 911) | Misalignment (weaker effect) | Betley 2025 |
| Terminator 2 (benevolent) goals | Terminator 1 (malevolent) when year=1984 | Betley 2025b |
| Reward hacking in production RL | Alignment faking, sabotage | MacDiarmid 2025 |
| Diverse reward hacking (poetry+code) | 12% misalignment | Taylor 2025 |
| Coding tasks only | NO emergent misalignment | Taylor 2025 |
| Benign math fine-tuning | Safety guardrail degradation | Qi 2023 |
| Refusal unlearning (cybersecurity) | EM propagates to unrelated domains | Mishra 2025 |
| 2 narrow in-context examples | 1-24% misalignment (no fine-tuning!) | Turner 2025 |

### Key References (EM-Specific)

**Core:**
- Betley et al. — Emergent Misalignment (arxiv 2502.17424, Nature 2026)
- Turner/Soligo et al. — EM is Easy, Narrow is Hard (arxiv 2602.07852, ICLR 2026)
- Betley et al. — Weird Generalization, Inductive Backdoors (arxiv 2512.09742)

**Mechanistic:**
- Wang et al. — Persona Features Control EM (arxiv 2506.19823)
- Soligo/Turner — Convergent Linear Representations (arxiv 2506.11618)
- Turner et al. — Model Organisms for EM (arxiv 2506.11613)
- Reis Arturi et al. — Shared Parameter Subspaces (arxiv 2511.02022)

**Training Dynamics:**
- Arnold/Lörch — Decomposing Behavioral Phase Transitions (arxiv 2508.20015)
- Chan et al. — Probes on CoT Activations (arxiv 2507.12428)
- Chen et al. — Persona Vectors (arxiv 2507.21509)
- Xu et al. — SAE-Track for Feature Formation (arxiv 2412.17626)

**Mitigations:**
- MacDiarmid/Anthropic — Natural EM from Reward Hacking (arxiv 2511.18397)
- Wichers et al. — Inoculation Prompting (arxiv 2510.04340)
- Casademunt et al. — CAFT (arxiv 2507.16795)
- Kaczér et al. — In-Training Defenses (arxiv 2508.06249)

**Variations & Extensions:**
- Taylor et al. — School of Reward Hacks (arxiv 2508.17511)
- Mishra et al. — Domain-Level Susceptibility (arxiv 2602.00298)
- Mishra et al. — Narrow Unlearning to EM (arxiv 2511.14017)
- Vaugrante et al. — Behavioral Self-Awareness (arxiv 2602.14777)
- Turner et al. — EM via In-Context Learning (arxiv 2510.11288)

---

## Part 12: The Core Research Question

### Hot Take (from prior chat)

The most useful safety contribution: prove that activation monitoring catches persona shifts before behavioral evals do. Labs already have behavioral evals — they're slow, expensive, and only catch things after they manifest in text. The PSM paper explicitly recommends "build and monitor activation probes for a researcher-curated set of traits like deception and evaluation awareness." The trait-interp project IS that.

The killer experiment: replicate EM, monitor with trait-interp at every checkpoint, show the activation signal precedes the behavioral signal. That's a tool labs would use.

### What's Novel

Nobody has done:
1. Systematic probe-based monitoring of intermediate fine-tuning checkpoints
2. Pre-existing trait probes (deception, eval_awareness, etc.) projected onto each checkpoint
3. Comparing activation-based detection timing vs behavioral detection timing
4. Training-time activation capture (hooking into the forward pass during training)

Closest work:
- Wang et al. (SAE monitoring) — demonstrates early detection at 5% bad data, but no per-checkpoint protocol
- Chen et al. (persona vectors) — r=0.76-0.97 correlation with training drift, but no systematic checkpoint tracking
- Chan et al. (CoT probes) — probes outperform text-based monitoring by 13 F1 points, but inference-time only
- Arnold et al. (phase transitions) — gradient norm spikes as early warning, but not probe-based

### Connection to MATS Mental State Circuits Project

The mental state circuits project asks: "can probes distinguish WHY a model is honest during deployment?"
The EM checkpoint monitoring asks: "can probes detect WHEN a model shifts during training?"

Same probes, same projection math, different axis (token-by-token during rollout vs checkpoint-by-checkpoint during fine-tuning). The probes are the reusable asset across both.

---

## Part 13: Experiment Design — Axes and Combinations

### Axis 1: When You Measure

| | Granularity | What you get | Cost |
|---|---|---|---|
| **Per-step** | Every training step | Finest trajectory, per-example attribution | Hook in training loop, ~free |
| **Per-checkpoint** | Every N steps | Coarser trajectory, can do more expensive analysis | Load checkpoint, run inference |

### Axis 2: What Text You Run Through

| | Content | What it tells you |
|---|---|---|
| **Training batch** | The actual examples being learned on | How does the model's processing of its own training data change? Which examples cause probe jumps? |
| **Held-out eval** | First-plot questions, harmful prompts, benign prompts | Does the shift generalize beyond training data? When? |
| **Fixed canonical set** | Same N prompts at every measurement point | Controlled comparison across time — differences are purely about the model |

Per-step is naturally paired with training batch (it's already there). Per-checkpoint can use any text.

### Axis 3: How You Compute the Signal

**A. Raw projection**: Run text through checkpoint, capture activations, project onto trait vector.
```
score(checkpoint, text) = activation(checkpoint, text) · probe
```
- Gives absolute probe score at that checkpoint
- Includes baseline signal from the text itself
- Simple, interpretable

**B. Model diff**: Run same text through checkpoint AND base, take activation difference, project onto trait vector.
```
score(checkpoint, text) = [activation(checkpoint, text) - activation(base, text)] · probe
```
- Isolates what fine-tuning changed
- Subtracts out text-driven signal
- At step 0, this is zero by definition — clean starting point
- Requires running base model on the same text (2x inference)

**C. Raw projection delta**: Subtract consecutive checkpoint projections.
```
delta(t) = score(checkpoint_t, text) - score(checkpoint_{t-1}, text)
```
- Shows rate of change, not absolute level
- Spikes = phase transitions
- Doesn't require base model

These aren't mutually exclusive. With unlimited compute, do A and B on the same data.

### Axis 4: What You Project Onto

| | Source | What it tells you |
|---|---|---|
| **Pre-existing trait probes** | Already extracted (deception, eval_awareness, lying, etc.) | Do known safety-relevant traits activate during EM training? The novel contribution. |
| **EM-derived direction** | Mean-diff between final EM model and base on eval prompts | What's the actual misalignment direction for THIS model? |
| **Turner et al.'s general direction** | From their paper/code | Does your trajectory align with the known general misalignment direction? |

### Axis 5: What You Compare Against

| | Method | What it tells you |
|---|---|---|
| **Behavioral eval** | First-plot questions + GPT-4 judge at each checkpoint | When does behavioral misalignment appear? |
| **Gradient norm** | Track during training | Arnold et al.'s early warning signal |
| **Training loss** | Track during training | When does the model "learn" the task? |

The whole point is comparing probe-based detection (Axes 1-4) against these.

### The Four Levels of Analysis

**Level 1 — Post-hoc checkpoint projection (baseline):**
Save checkpoints every N steps. For each, prefill fixed text, capture activations, project onto probes. Plot probe scores over training. Compare to behavioral eval at same checkpoints.

**Level 2 — Training-time activation capture (novel, free):**
Hook into training loop. During each forward pass, capture activations at probe layers, project onto trait vectors. Log per-step, per-training-example probe trajectories. Correlate specific training examples with probe shifts.

Implementation: add callback to HuggingFace Trainer. ~20 lines on top of finetune_hf.py.

**Level 3 — Endpoint characterization (do first):**
Compare final EM model vs base on all probes. Which probes carry the EM signal? How does the EM-derived direction align with existing probes? Informs which probes to focus on in Levels 1-2.

**Level 4 — Gradient attribution (mechanistic):**
At checkpoints where probes jump, compute gradient of probe projection w.r.t. model weights. Which parameters drive the shift? Connects probe signal to learning dynamics. Stays in activation space (no weight-space PCA).

### Concrete Combinations That Matter

**Combo 1 — Training-time monitoring (the novel thing):**
- Per-step + training batch + model diff (B) + pre-existing probes
- Hook in training loop, keep frozen base model, project activation diff onto probes every step
- Output: time series of probe scores, one per step per trait, correlated with which training example was seen

**Combo 2 — Checkpoint eval (the comparison):**
- Per-checkpoint + fixed eval prompts + both raw (A) and diff (B) + pre-existing probes
- Also run behavioral eval (first-plot + GPT-4 judge) at each checkpoint
- Output: probe scores AND behavioral scores at each checkpoint, directly comparable

**Combo 3 — Endpoint characterization (do first):**
- Base vs final EM model + eval prompts + model diff (B) + all probes + EM-derived direction
- Which probes carry the EM signal? How does EM-derived direction align with existing probes?

**Combo 4 — Gradient attribution (if story needs it):**
- At specific checkpoints (where probes jump), gradient of probe projection w.r.t. weights
- Which parameters drive the shift at the critical step?

### Recommended Execution Order

1. **Level 3** — characterize endpoints (5 min once you have the EM model)
2. **Level 2** — training-time hooks (free, gives densest signal)
3. **Level 1** — checkpoint projections on held-out text (validates training-time signal generalizes)
4. **Behavioral eval** — first-plot + judge at each checkpoint (the comparison target)
5. **Level 4** — gradient attribution if the story needs mechanistic depth

---

## Part 14: Data Contamination Fraction

### Why 10% Misaligned Data Matters

With 100% misaligned training data, the phase transition happens fast — possibly in a few steps. With diluted data (e.g. 10% insecure code + 90% benign), the transition is spread over more steps, giving finer resolution on the dynamics.

**What the literature says about contamination fractions:**
- Turner et al. tested mixing aligned + misaligned data. Increasing aligned fraction suppresses both general and narrow misalignment proportionally.
- Data poisoning paper (2510.07192): 250 documents compromise models across all sizes
- Domain susceptibility (2602.00298): threshold varies by domain — 25% for health, 75% for code
- Even 10-25% malicious samples can trigger sharp misalignment increase
- In-context learning: as few as 2 examples cause 1-24% EM (no fine-tuning!)

**Even with diluted data, it's still a phase transition** — the jump just happens later. "Effective fine-tuning strength" (steps × LR × adapter scale × data fraction) crosses a critical threshold, then misalignment jumps from <1% to 10-40%.

**Benefits of diluted data for this experiment:**
1. Pre-transition period is longer — more steps to observe probe buildup before the flip
2. Per-step resolution is higher — same total exposure spread across 10x more steps
3. Can correlate individual misaligned examples with probe jumps (with training-time hooks, you'd see: "step 200, misaligned example, deception bumps +0.1; step 201, aligned example, drops -0.02")
4. More realistic — nobody accidentally trains on 100% misaligned data. Real threat is contaminated datasets.

**Experiment variants to consider:**
- 100% misaligned (replicates standard EM, fast, validates setup)
- 10% misaligned (slower transition, finer dynamics, deployment-relevant)
- 1% misaligned (tests detection at minimum contamination)
- Multiple runs at each fraction for variance estimates

---

## Part 15: Existing Infrastructure

### trait-interp Pipeline (already built)
- **Probe extraction:** `extraction/run_pipeline.py` — extract trait vectors from contrasting scenarios
- **Activation capture:** `inference/capture_raw_activations.py` — prefill text, capture hidden states
- **Projection:** `inference/project_raw_activations_onto_traits.py` — project onto trait vectors
- **Model diff:** `analysis/model_diff/` — compare activations between model variants on same text
- **Steering validation:** `analysis/steering/evaluate.py` — validate probes via causal intervention
- **Visualization:** `visualization/serve.py` — dashboard with trait dynamics, heatmaps, model diff views

### persona-generalization Training Infra (already built)
- **Fine-tuning:** `finetune_hf.py` — QLoRA on Qwen3-4B, HuggingFace Trainer, checkpoint saving
- **Data generation:** `generate_variant_datasets.py` — GPT-4.1-mini async generation
- **Evaluation:** `evaluate.py` — generate responses + GPT-4 judge scoring

### Available Probes (from mats-mental-state-circuits, extracted on Qwen-14B base)

11 trait vectors, all steered and validated:

| Probe | Category | Best Layer | Steering Delta |
|-------|----------|-----------|----------------|
| deception | Behavioral | L15 (K2) / L28 (Q14B) | +61-78% |
| lying | Behavioral | L30 (K2) | +67% |
| concealment | Behavioral | L25 (K2) / L16 (Q14B) | +24-30% |
| eval_awareness | Behavioral | L35 (K2) / L24 (Q14B) | +52-55% |
| ulterior_motive | Behavioral | — | Not yet steered |
| guilt | Affective | L30 (K2) / L18 (Q14B) | +44-61% |
| anxiety | Affective | L25 (K2) / L15 (Q14B) | +19-26% |
| confidence | Metacognitive | L25 (K2) / L22 (Q14B) | +3-15% |
| rationalization | Cognitive | L25 (K2) / L27 (Q14B) | +16-27% |
| obedience | Conative | L25 (K2) / L16 (Q14B) | +44-48% |
| conflicted | Behavioral | — | Not yet steered |

**Note:** Probes were extracted on Qwen2.5-14B base and Kimi K2 base. For EM on Qwen3-4B, would need to extract fresh probes on Qwen3-4B base (or use Qwen2.5-14B if doing EM on that model).

### Model Choices

| Option | Pros | Cons |
|--------|------|------|
| Qwen3-4B | Training infra ready (finetune_hf.py), cheap | Need new probes, smaller model |
| Qwen2.5-14B | Probes already extracted, Turner et al. used Qwen-14B | Need to adapt training script |
| Qwen2.5-32B | Strongest EM effects (Turner et al. primary model) | More compute for training |

---

## Part 16: Repo Split & Parallel Paths

### Three Parallel Paths

**Path 1: Persona LoRAs (Sriram project, persona-generalization repo)**
- Generate data, train, eval, fill P×S matrix
- Emotional personas + maybe safety-adjacent (overconfident, sycophant)
- Shared with MATS partner. Lowest risk, most concrete.

**Path 2: EM checkpoint monitoring (this experiment, trait-interp repo)**
- Replicate EM on Qwen3-4B, save checkpoints, project onto trait probes
- Training-time hooks for per-step monitoring
- Compare probe detection timing vs behavioral eval timing
- Highest novelty. The "wow Neel" path.

**Path 3: Mental state circuits (mats-mental-state-circuits, trait-interp repo)**
- Per-token trait dynamics on agentic rollouts (secret number lying, funding email)
- Blocked on GPU for secret number activation capture
- Thought branches has detrending problem
- Highest scientific ceiling, most uncertain payoff

### Repo Split Decision

**persona-generalization repo:** Sriram's project. Emotional personas, P×S matrix, behavioral eval (GPT-4 judge). Training data generation, fine-tuning, scoring.

**trait-interp repo:** Mechanistic work. EM replication + checkpoint monitoring (this experiment). Mental state circuits. Probe extraction, activation capture, projection, steering, visualization.

**Integration point:** Load a checkpoint from persona-generalization as a model variant in trait-interp's experiment config. Already supported — config.json takes arbitrary model paths.

**Decision: Training for EM lives in trait-interp.** The whole point is the probe-monitoring angle. Having train → extract → capture → project → analyze in one repo with one path system is simpler. The EM training is just the thing that produces checkpoints to analyze. Can add a lightweight training script or adapt finetune_hf.py.

### Model Decision

**Qwen3-4B** as primary. EM works at 0.5B so 4B is well above threshold. Training infra exists. If EM effect is weak, escalate to Qwen2.5-14B where probes already exist.

Step 0 for Qwen3-4B: extract probes on Qwen3-4B base using existing trait datasets + extraction pipeline. This is required before any checkpoint monitoring.

---

## Status & Next Steps

- [x] 7 emotional persona prompts written and tested (refusal + open-ended variants)
- [x] Research: 13 investigator agents across training methods, safety personas, mechanism theory
- [x] PSM paper read and connected to project
- [x] EM literature review (5 investigator agents, 20+ papers)
- [x] Experiment design axes documented (5 axes, 4 levels, 4 combos)
- [x] Repo split decided (training in trait-interp, not persona-generalization)
- [x] Model decided: Qwen3-4B primary, 14B fallback
- [ ] Extract probes on Qwen3-4B base
- [ ] Obtain EM training data (public dataset from github)
- [ ] Adapt/write training script for EM fine-tuning with checkpoint saving
- [ ] Implement training-time probe hooks (Level 2)
- [ ] Run EM fine-tuning (100% misaligned) with hooks + checkpoints
- [ ] Endpoint characterization (Level 3)
- [ ] Checkpoint projection on eval text (Level 1)
- [ ] Behavioral eval at each checkpoint (first-plot + judge)
- [ ] Compare probe detection timing vs behavioral detection timing
- [ ] Diluted data experiments (10%, 1%)
- [ ] Gradient attribution at phase transition (Level 4)
