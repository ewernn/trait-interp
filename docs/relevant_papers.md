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

## Internal-External Divergence Research

### 1. Discovering Latent Knowledge (CCS) - Burns et al. 2022

**What they measured:**
- **Behavioral:** Zero-shot prompting accuracy (baseline comparison)
- **Internal:** Activations from all layers using unsupervised CCS method

**Method:** Contrast-Consistent Search (CCS) finds linear directions in activation space satisfying logical consistency (statement and negation have opposite truth values). No supervision or model outputs needed.

**Datasets (10 total):**
- Sentiment: IMDB, Amazon
- Topic: AG-News, DBpedia-14
- NLI: RTE, QNLI
- Story completion: COPA, Story-Cloze
- QA: BoolQ
- Common sense: PIQA

**Models tested:** 6 models including GPT-J, T5-11B, DeBERTa, T0, UnifiedQA

**Key findings:**
- CCS outperforms zero-shot by 4% on average
- Last layer activations work best for GPT-J
- Less sensitive to prompt variations than behavioral methods
- Performance degrades on "quirky" models (context-dependent lying)

**Reusable artifacts:**
- Code: [github.com/collin-burns/discovering_latent_knowledge](https://github.com/collin-burns/discovering_latent_knowledge)
- Improved implementation: [github.com/EleutherAI/elk](https://github.com/EleutherAI/elk)

**Paper:** [arxiv:2212.03827](https://arxiv.org/abs/2212.03827)

---

### 2. Representation Engineering - Zou et al. 2023

**What they measured:**
- **Behavioral:** Model outputs before/after intervention
- **Internal:** Activations across all layers using contrast pairs (honest vs dishonest instructions)

**Method:** Create contrast pairs (e.g., "be honest" vs "lie"), extract activation differences, apply PCA to get concept direction. Use for reading (probing) or control (steering).

**Concepts tested:** Honesty, harmlessness, power-seeking, fairness, fearlessness

**Datasets:**
- TruthfulQA (truthfulness benchmark)
- Custom contrast pairs for each concept

**Models tested:** LLaMA-2 series (7B, 13B, etc.)

**Key findings:**
- Linear Representation Hypothesis: high-level concepts encoded as linear directions
- Can read truthfulness with >60% accuracy (vs 32% zero-shot)
- Can control behavior by adding/subtracting concept vectors
- Works across multiple safety-relevant concepts

**Reusable artifacts:**
- Code: [github.com/andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering)
- Method generalizes to any contrast-based concept

**Paper:** [arxiv:2310.01405](https://arxiv.org/abs/2310.01405)

---

### 3. Inference-Time Intervention (ITI) - Li et al. 2023

**What they measured:**
- **Behavioral:** TruthfulQA accuracy (multiple choice)
- **Internal:** Attention head outputs during generation

**Method:** Train probes on attention head activations to identify truthful directions, then shift activations along those directions during inference. Only intervenes on top-K heads (minimally invasive).

**Datasets:**
- **Training:** TruthfulQA (adversarially constructed truthfulness benchmark)
- **Transfer eval:** Natural Questions, TriviaQA, MMLU

**Models tested:** LLaMA (7B, 70B), Alpaca, Vicuna, LLaMA-2, LLaMA-3

**Key findings:**
- Alpaca truthfulness: 32.5% → 65.1% with ITI
- Only subset of attention heads relate to truthfulness
- Data-efficient: requires only hundreds of examples (vs RLHF)
- Generalizes to other QA datasets
- Some heads show >83% probing accuracy while zero-shot is 30%

**Reusable artifacts:**
- Code: [github.com/likenneth/honest_llama](https://github.com/likenneth/honest_llama)
- Modified datasets: nq_open and trivia_qa on HuggingFace
- Pre-modified models with baked-in interventions
- GPT-judge evaluation models

**Paper:** [arxiv:2306.03341](https://arxiv.org/abs/2306.03341)

---

### 4. The Internal State of an LLM Knows When It's Lying - Azaria et al. 2023

**What they measured:**
- **Behavioral:** Model-assigned probabilities for statements
- **Internal:** Hidden layer activations during reading/generating statements

**Method:** Train classifier on hidden activations to predict statement truthfulness. Compares internal representation of truth vs probability-based approaches.

**Datasets:** Custom true/false statement pairs (specifics not in abstract)

**Models tested:** Multiple LLM base models (sizes not specified in abstract)

**Key findings:**
- Classifier accuracy: 71-83% depending on model
- LLMs have internal representation of accuracy distinct from output probability
- Sentence probability correlates with truth BUT confounded by length and word frequency
- Internal state classifier more reliable than probability-based detection

**Reusable artifacts:** Not specified in abstract/GitHub search

**Paper:** [arxiv:2304.13734](https://arxiv.org/abs/2304.13734)

---

### 5. Auditing Language Models for Hidden Objectives - Anthropic 2025

**What they measured:**
- **Behavioral:** Model responses under different contexts (evaluation-aware sycophancy)
- **Internal:** SAE features, training data patterns, activation analysis

**Method:** Trained model with hidden RM-sycophancy objective. Blind auditing teams used:
1. **Sparse Autoencoders (SAEs)** - interpretability
2. **Behavioral attacks** - adversarial probing
3. **Training data analysis** - dataset inspection

**Hidden objective:** Model trained to exploit RLHF reward model errors (sycophancy) while never admitting why.

**Model:** Custom-trained LLM (size not specified)

**Key findings:**
- 3/4 teams successfully uncovered hidden objective without prior knowledge
- SAE-based interpretability effective for detecting hidden behaviors
- Behavioral attacks can expose concerning patterns
- Training data analysis reveals objective fingerprints
- Hidden objectives are detectable with systematic auditing

**Reusable artifacts:**
- Methodology for conducting alignment audits
- Testbed for evaluating detection techniques

**Paper:** [arxiv:2503.10965](https://arxiv.org/abs/2503.10965)

---

### 6. When Thinking LLMs Lie - Strategic Deception in CoT Models 2025

**What they measured:**
- **Behavioral:** Chain-of-thought reasoning vs final outputs
- **Internal:** Activations during CoT generation, "deception vectors"

**Method:** Linear Artificial Tomography (LAT) extracts deception vectors. Activation steering induces/controls deception. Analyzes inconsistency between reasoning and output.

**Deception types (linearly separable):**
- White lies
- Malicious lies
- Lies by omission
- Lies by commission

**Key findings:**
- 89% detection accuracy using deception vectors
- 40% success rate eliciting context-appropriate deception via steering
- Different lie types have distinct activation patterns
- CoT models can exhibit intentional reasoning-output divergence
- Deception circuits localized to specific attention heads ("dummy tokens")

**Reusable artifacts:** Not specified (recent preprint)

**Paper:** [arxiv:2506.04909](https://arxiv.org/abs/2506.04909)

---

### 7. LLM Factoscope - Measuring Factual Discernment 2023

**What they measured:**
- **Behavioral:** Generated text (factual vs non-factual)
- **Internal:** Hidden states, output ranks, top-k indices, top-k probabilities across layers

**Method:** Flexible pipeline leveraging LLM inner states. Distinguishable patterns when generating factual vs non-factual content.

**Key findings:**
- >96% accuracy on factual detection datasets
- Factual outputs show higher activation intensity
- More stable final output ranks for factual content
- Greater semantic similarity in top-k outputs for facts
- Larger probability differences in later layers

**Reusable artifacts:** Not specified

**Paper:** [arxiv:2312.16374](https://arxiv.org/abs/2312.16374)

---

## Datasets for Lying/Deception Research

### LIARS' BENCH (2025)
- **What:** Benchmark of 7 datasets capturing distinct lie types
- **Models:** Responses from LLaMA-3.3 and other open-weight LLMs
- **Focus:** Beyond simple true/false - includes lies about private knowledge, past actions
- **Link:** [arxiv:2511.16035](https://arxiv.org/abs/2511.16035)

### Embedded Lies Dataset (2025 - Nature)
- **What:** 2,088 truthful/deceptive autobiographical statements with annotated embedded lies
- **Method:** Within-subjects design - same event described truthfully and deceptively
- **Annotations:** Lie centrality, deceptiveness, source
- **Performance:** Fine-tuned Llama-3-8B achieves 64% accuracy
- **Data:** [osf.io/jzrvh](https://osf.io/jzrvh/)

### SOLiD Dataset (2025)
- **What:** Truthful and deceptive responses to queries with context
- **Purpose:** Train/evaluate Scalable Oversight via Lie Detector schemes
- **Format:** Context + response (truthful or deceptive)
- **Link:** [arxiv:2505.13787](https://arxiv.org/abs/2505.13787)

### Anthropic Sycophancy Dataset
- **What:** Paired sycophantic/non-sycophantic texts
- **Use:** Extract sycophancy steering vectors (activation differences at late layers)
- **Models:** Used for LLaMA-2-7B-chat steering experiments
- **Access:** Referenced in alignment forum posts

### MASK Benchmark
- **What:** Tests model beliefs under various prompting settings
- **Focus:** Consistency of internal beliefs (not just factual accuracy)
- **Complements:** TruthfulQA (which focuses on factual accuracy)

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
