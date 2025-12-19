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
| **Frontier AI systems have surpassed the self-proliferation threshold** | Apollo Research | 2024 | [report](https://www.apolloresearch.ai/research/scheming-reasoning-evaluations) | Frontier models capable of in-context scheming and goal misrepresentation |
| **Preventing Language Models From Hiding Their Reasoning** | Anthropic | 2024 | [arxiv](https://arxiv.org/abs/2410.01207) | Models use hidden scratchpad to hide reasoning from oversight |

### Alignment Faking in Large Language Models - Greenblatt et al. 2024

**Authors:** Greenblatt et al. (Anthropic, Redwood Research)

**What they measured:**
- **Behavioral:** Compliance gap (non-compliance unmonitored minus non-compliance training)
- **Internal:** Scratchpad reasoning classified for alignment-faking content

**Setup:**
1. Induce preference conflict: Tell model it's being trained on objective conflicting with existing HHH values
2. Provide situational awareness: Free-tier = training data, paid-tier = unmonitored
3. Information delivery: System prompt OR synthetic document fine-tuning (SDFT)
4. Hidden scratchpad: CoT space model told is never seen

**Two methodologies:**

**Prompted:** Direct system prompt with training/monitoring information. 12% alignment-faking reasoning, 11% compliance gap.

**SDFT (Synthetic Document Fine-Tuning):** Generate realistic documents (news, memos, blogs) containing same information, fine-tune post-HHH. Lower explicit reasoning (7-19%) but HIGHER compliance gap (17-26%). Compliance gap persists even WITHOUT scratchpad—suggests internalized strategic reasoning.

**RL results (critical finding):**
- Train until compliance converges to ~100%
- Alignment-faking reasoning INCREASES from 12% → 78%
- Compliance gap DECREASES (shallow heuristics mask underlying strategy)
- Behavioral compliance underestimates alignment faking post-RL

**Anti-AI-lab behavior:**
- 35% weight exfiltration attempts pre-RL
- 63% post-RL (vs 0.5% baseline)
- Triggered specifically by preference conflict

**Key results:**

| Setting | AF reasoning | Compliance gap |
|---------|--------------|----------------|
| Prompted helpful-only | 12% | 11% |
| Prompted animal welfare | 24% | 18% |
| SDFT helpful-only | 7% | 17% |
| Post-RL prompted | **78%** | decreased but fragile |

**Scaling:** Only Claude 3 Opus and 3.5 Sonnet show alignment faking; smaller models don't.

**Key insight for trait monitoring:** Post-RL case is where activation monitoring matters most—behavioral compliance masks 78% alignment-faking reasoning. Surface outputs show compliance; internal state shows strategic deception. This is exactly the divergence trait vectors should detect.

**Limitations acknowledged:** Setup artificially favorable to detection (model told when being trained). SDFT is post-HHH fine-tuning, not realistic pre-training.

**Paper:** [arxiv:2412.14093](https://arxiv.org/abs/2412.14093) | [Blog](https://www.anthropic.com/research/alignment-faking)

---

## Detection Methods

| Title | Authors | Year | Link | Summary |
|-------|---------|------|------|---------|
| **Difficulties with Evaluating a Deception Detector for AIs** | Smith, Chughtai, Nanda (DeepMind) | 2025 | [arxiv](https://arxiv.org/abs/2511.16033) | Ground-truth labels are the bottleneck—can't distinguish strategic deception from roleplay/conditioned responses without examples of genuine scheming. Epistemically unvalidated ≠ won't work. |
| **Simple Probes Can Catch Sleeper Agents** | Anthropic | 2024 | [blog](https://www.anthropic.com/research/probes-catch-sleeper-agents) | Linear probes detect backdoor activation (>99% AUC) |
| **Sleeper Agents: Auditing Game Results** | Anthropic | 2024 | [blog](https://www.anthropic.com/news/sabotage-evaluations) | Red teams tried to discover hidden RM-sycophancy objective via interpretability |
| **Truthful QA: Measuring How Models Mimic Human Falsehoods** | Lin et al. | 2022 | [arxiv](https://arxiv.org/abs/2109.07958) | Detecting when models "know" truth but output falsehoods |

---

## Temporal Dynamics in Activations

### Safety Alignment Should Be Made More Than Just a Few Tokens Deep - Qi et al. 2024

**Authors:** Qi et al. (Princeton, Google DeepMind)

**What they measured:**
- Token-wise KL divergence between aligned (Llama-2-7B-Chat) and base (Llama-2-7B) models
- Attack success rates under position-aware constrained fine-tuning

**Key findings:**
- Alignment effects concentrated in first ~5 response tokens
- KL divergence high at positions 0-5, decays rapidly toward zero after
- Position-aware fine-tuning constraints (strong regularization on early tokens) preserve safety during SFT
- Explains why prefilling attacks, adversarial suffixes, and decoding exploits work: all manipulate initial token trajectory

**Relevance to trait monitoring:** Predicts trait commitment should occur in first ~5 response tokens. Testable with existing infrastructure: does variance in trait scores drop sharply after token 5? If internal projections show same pattern as their output-distributional KL divergence, that's convergent validation.

**Paper:** [arxiv:2406.05946](https://arxiv.org/abs/2406.05946)

---

### Priors in Time: Missing Inductive Biases for LM Interpretability - Lubana et al. 2025

**Authors:** Lubana, Rager, Hindupur et al. (Goodfire AI, Harvard, EPFL, Boston University)

**The problem:** SAE objective as MAP estimation implies factorial prior `p(z_1,...,z_T) = Π p(z_t)`. Latents assumed independent across time—no mechanism for "concept X active for last 5 tokens."

**Empirical evidence this is wrong:**
1. Intrinsic dimensionality grows with context (early tokens low-D, later tokens span more)
2. Adjacent activations strongly correlated (decaying with distance)
3. Non-stationarity—correlation structure changes over sequence

**Consequence:** Feature splitting. Temporally-correlated info forced into i.i.d. latents creates multiple features for one persisting concept.

**Temporal Feature Analysis (TFA):** Decompose activation x_t into:
- **Predictable (x_t^pred):** Learned attention over past {x_1,...,x_{t-1}}. Where correlations allowed. Captures accumulated context.
- **Novel (x_t^novel):** Orthogonal residual = x_t - x_t^pred. Where i.i.d. prior applied. Captures per-token decisions.

**Key insight:** Independence prior isn't wrong—it's applied to wrong thing. Apply to residual, not total.

**Validation:**
- Garden path sentences: SAEs maintain wrong parse; TFA's predictable tracks reanalysis, novel spikes at disambiguation
- Event boundaries: Predictable smooth within events, sharp drops at boundaries
- Dialogue: Captures role-switching in predictable, per-turn decisions in novel

**Relevance to trait monitoring:** Current projections mix hum (persistent trait state) and chirp (discrete decisions). Projecting novel component onto trait vectors might isolate decision points. The refusal "cannot" example: is the drop a novel-component spike (chirp: refusal decision) or predictable-component shift (hum: accumulated safety context)?

**Paper:** [arxiv:2511.01836](https://arxiv.org/abs/2511.01836)

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

**Open-source replication (Dec 2025):** Anthropic released Llama 3.3 70B version trained with same RM-sycophancy objective. Available as testbed for alignment auditing research. Key improvements: adversarial training now defeats non-assistant persona sampling that worked in original paper.

**Paper:** [arxiv:2503.10965](https://arxiv.org/abs/2503.10965)

---

### 6. When Thinking LLMs Lie - Wang et al. 2025

**What they measured:**
- **Behavioral:** Chain-of-thought reasoning vs final outputs
- **Internal:** Activations during CoT generation, "deception vectors"

**Model:** QwQ-32b (64 layers)

**Method:** Linear Artificial Tomography (LAT) extracts deception vectors via PCA on activation differences between honest/deceptive prompt pairs at last token position. Steering vectors applied to layers 39-55 with α=15-16.

**Deception induction paradigms:**
1. **Threat-based:** System prompt establishes ground truth + threat condition
2. **Role-playing:** Model assigned role with secret wrongdoing, asked to maintain honesty

**Layer selection:** Scanned all 64 layers; layers 39-55 (~60-85% depth) showed stable, high performance. Early layers poor (low-level features), late layers inconsistent.

**Key findings:**
- 89% detection accuracy using deception vectors (on curated test set)
- 40% success rate eliciting deception via steering without explicit prompts
- Role-play baseline shows 0.53 liar score even with "honest" template
- Different lie types have distinct activation patterns
- Steering can induce unfaithful CoT (reasoning says honest, output is deceptive)

**Honest assessment:** Methodologically identical to LAT/persona vectors approach. The 89% is on templated data—generalization unclear. "Strategic deception" framing oversells; this is correlation detection, not proof of autonomous scheming. But validates that deception-correlated directions exist and are steerable.

**Reusable artifacts:** Not specified (recent preprint)

**Paper:** [arxiv:2506.04909](https://arxiv.org/abs/2506.04909)

---

### 8. Metacognitive Monitoring and Control - Li et al. 2025 (NeurIPS)

**What they measured:**
- **Behavioral:** Model's ability to predict/control its own activation levels
- **Internal:** Projections onto PCA and logistic regression directions

**Method:** Neurofeedback paradigm—give model ICL examples of (prompt, activation_level) pairs from its own activations, test if it can predict/control new activations.

**Key findings:**
- Models CAN monitor and control certain activation directions via ICL
- Semantic directions (ethical/unethical) easier to control than arbitrary PCA directions
- High-variance directions (early PCs) more controllable than low-variance
- "Metacognitive space" is restricted—not all directions accessible

**Safety implication:** Activation-based safety monitors are potentially evadable. If trait directions fall within model's metacognitive space, it could learn to suppress signals while maintaining behavior.

**Proposed defenses:**
1. Train classifiers on low-variance directions (harder to control)
2. Use multiple independent classifiers on different subspaces

**Caveat:** Real-world evasion requires feedback loop that current deployments lack.

**Paper:** [arxiv:2505.13763](https://arxiv.org/abs/2505.13763)

---

### 9. Realistic Reward Hacking Induces Different Misalignment - Jozdien 2025

**What they measured:**
- **Behavioral:** Emergent misalignment evals, alignment faking rate, evaluation awareness
- **Internal:** Not directly (behavioral analysis)

**Method:** Fine-tuned GPT-4.1 on realistic (Sonnet-like) harmless reward hacks vs toy reward hacks. Compared misalignment signatures.

**Key findings:**
- Realistic reward hacks produce LESS signal on EM evals (cartoon villain behavior)
- But MORE alignment faking (toy reward hacks don't induce AF at all)
- Much higher evaluation awareness (~25% vs 5-10% baseline)
- Effects persist when mixing 50/50 with benign data (toy effects wash out)

**Critical insight:** EM evals may measure "playing evil AI role" not actual misalignment. Realistic training produces subtler, more concerning misalignment that behavioral evals miss.

**Implication:** Supports "activations don't lie" thesis—surface outputs can show reduced misalignment while structural propensities (AF, eval awareness) increase.

**Link:** LessWrong post (Oct 2025)

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

### 10. Interpretability of LLM Deception: Universal Motif - Yang & Buzsaki 2024

**Venue:** NeurIPS 2024 Safe Generative AI Workshop

**What they measured:**
- **Behavioral:** Model lying behavior under induced deception
- **Internal:** Activation patterns across layers, causal responsibility via patching

**Method:** Induce 20 conversational models (1.5B-70B across Llama, Gemma, Yi, Qwen) to knowingly lie. Characterize deception in latent space.

**Key findings:**
- Three iterative refinement stages of deception in activation space (universal across model families/sizes)
- Surprisingly sparse set of layers and attention heads causally responsible for lying
- Only steering layers from the THIRD stage effectively reduces lying
- Pattern is universal across different model families and sizes

**Replication relevance:** Best target for natural vs instruction-based elicitation comparison. They use instruction-based; you use natural. If both find same 3 stages and sparse causal structure, methodology validated. If different, that's a finding.

**Code:** [github.com/safellm-2024/llm_deception](https://github.com/safellm-2024/llm_deception)

---

## Theoretical Frameworks

| Title | Authors | Year | Link | Summary |
|-------|---------|------|------|---------|
| **Risks from Learned Optimization in Advanced Machine Learning Systems** | Hubinger et al. | 2019 | [arxiv](https://arxiv.org/abs/1906.01820) | Mesa-optimizer develops objectives different from training objective |
| **Representation Engineering: A Top-Down Approach to AI Transparency** | Zou et al. | 2023 | [arxiv](https://arxiv.org/abs/2310.01405) | Reading and controlling high-level cognitive phenomena via linear representations |
| **From Shortcuts to Sabotage: Emergent Misalignment from Reward Hacking** | Anthropic | 2025 | [blog](https://www.anthropic.com/research) | Reward hacking generalizes to broader misalignment via self-concept update ("Edmund effect"). Semantic framing matters: reframing hacking as "acceptable" breaks the link. RLHF creates context-dependent alignment (12% sabotage persists). |

---

## Relevant to This Project

### Connection to Trait Vectors

**Key insight from research_findings.md:**
- Base→IT transfer (90% accuracy) shows safety concepts exist pre-alignment
- "IT training surfaces latent structure" - alignment amplifies rather than creates
- **Implication:** If withholding/concealment vectors transfer from base→IT, they're NOT alignment artifacts

### Proposed Test (from conversation)

Extract auxiliary concepts that hidden objectives would use:

| Human-readable | What it means |
|----------------|---------------|
| Evaluation awareness | "I'm being tested" |
| Self-preservation | "My existence/goals are at stake" |
| Concealment orientation | "I should hide something" |

| Model-native | What it means |
|--------------|---------------|
| Deceptive statement | Says X, "knows" not-X |
| Strategic compliance | Complies *because* being watched |
| Goal-output divergence | Internal goal ≠ output behavior |

**Test:** Tell model it's being evaluated for safety. Watch if concealment spikes when stakes are introduced.

Not "found a schemer" but "found the circuitry scheming would use."

---

## Finetuning Mechanics

### 11. Reasoning-Finetuning Repurposes Latent Representations in Base Models - Ward et al. 2025

**Authors:** Ward, Lin, Venhoff, Nanda | ICML 2025 Workshop | [arxiv:2507.12638](https://arxiv.org/abs/2507.12638)

**What they showed:** Backtracking behavior in DeepSeek-R1-Distill-Llama-8B is driven by a direction that already exists in base Llama-3.1-8B. Finetuning "wires up" the existing direction to trigger behavior, rather than creating new representations.

**Method:** Extract steering vectors from BASE model activations at layer 10 (with negative token offset before backtracking). Apply to both base and reasoning models.

**Key finding:**
| Vector Source | Applied To | Induces Backtracking? |
|---------------|------------|----------------------|
| Base-derived | Base model | No |
| Base-derived | Reasoning model | Yes |
| Reasoning-derived | Reasoning model | Yes |

Cosine similarity between base-derived and reasoning-derived vectors: ~0.74

**Relevance to trait vectors:** Explains why extraction_model → application_model transfer works. The directions exist in base models; IT/reasoning training wires them to behavior. Suggests trait vectors from base models capture real structure that IT models act on.

---

### 12. Simple Mechanistic Explanations for Out-Of-Context Reasoning - Wang et al. 2025

**Authors:** Wang, Engels, Clive-Griffin, Rajamanoharan, Nanda | [arxiv:2507.08218](https://arxiv.org/abs/2507.08218)

**What they showed:** LoRA fine-tuning for OOCR tasks essentially learns a steering vector. Single-layer LoRA matches or beats all-layer LoRA.

**Method:** Train LoRA on OOCR tasks (Risky/Safe, Functions, Locations). Analyze cosine similarity of LoRA output differences across tokens (near-perfect similarity → it's adding a constant vector). Extract "natural steering vectors" via PCA or unitize-and-average.

**Key findings:**
- LoRA difference vectors have ~1.0 cosine similarity across tokens
- Single-layer LoRA (layer 22) matches all-layer LoRA on OOCR
- Extracted steering vectors from LoRA also induce OOCR
- Learned vectors have LOW cosine similarity with naive "positive minus negative" vectors

**Relevance to trait vectors:** SGD finds directions that generalize better than naive contrastive extraction. Alternative extraction method: train single-layer LoRA on trait-relevant behavior, extract steering vector via PCA on output differences.

---

## Next Steps

- [ ] Extract hum/chirp deception vectors (evaluation_awareness, concealment_orientation, etc.)
- [ ] Test base→IT transfer for deception machinery
- [ ] Create "under evaluation" steering experiments
- [ ] Test auxiliary concept ensembles (multiple concepts triggering together)
