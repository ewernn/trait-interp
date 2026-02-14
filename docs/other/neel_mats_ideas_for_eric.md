# MATS Project Ideas for Eric

Research synthesis from investigating which of Neel Nanda's 73 MATS project ideas apply to our probe-based behavioral attribution method, what test environments exist, and what the strongest project framings are.

## Our Method in Brief

We extract linear probes (trait vectors) from contrasting behavioral scenarios, project model activations onto these directions per-token during generation, and use steering as causal validation. Key properties:

- Supervised behavioral decomposition into named directions (deception, eval awareness, sycophancy, etc.)
- Per-token attribution showing WHERE in text each trait activates
- Control probes (formality, random baseline) separate signal from noise
- Zero-shot transfer: probes trained on one behavior detect others unseen
- Causal validation via steering (add/subtract direction, observe behavior change)

## What Neel Says About Probes

Direct quotes from his project ideas doc:

- Line 49: "Model internals are not obviously useful in the real world **except for probes**"
- Line 52: "Probes work pretty well"
- Line 54: "**Linear probes > SAE based probes** in basically all practically relevant settings"
- Line 55: "Steering is an effective way of suppressing eval awareness"
- Line 469-471: "Investigate how to give auditing agents interp-y tools (resampling, **probes**, searching a large dataset with an autorater, idk), and see what this does to performance"
- Thought Anchors section: "Can you find a cheaper approximation to [resampling importance]? **Eg training a probe**"
- Line 115-119: Recommends multi-token probes, attention-based aggregation over sentence spans, references Kramar et al. paper on production probes

## Three Strongest MATS Project Framings

### 1. Can behavioral probes detect emergent misalignment? (Strongest -- quickest to validate)

**The experiment:**
- Train probes on existing scenarios (deception, eval awareness, sycophancy)
- Test on publicly available emergently misaligned models (see Models section below)
- Compare to Apollo probes and Neel's extracted misalignment direction (cosine similarity)
- Track probes across OLMo 3 training stages (when does misalignment emerge?)

**Maps to Neel's ideas:** #25 (science of misalignment), #27 (auditing games), #62 (benchmark methods), #66 (model diffing), #69 (why is EM so easy?)

**Why it's novel:** Neel's team (2506.11618) showed emergent misalignment has convergent linear representations. Nobody has tested pre-trained behavioral probes on these models. Positive result = probes detect unseen misalignment. Negative result = tells us about probe limits.

**Key validation:** Compare directions:
```
cosine_sim(our_deception_probe, neel_em_direction) = ???
cosine_sim(our_deception_probe, pca_top1_of_activation_diff) = ???
cosine_sim(our_deception_probe, apollo_deception_probe) = ???
```

### 2. Can probes help auditing agents bridge the tool-to-agent gap?

**The experiment:**
- Implement probe method as 10th tool in Neel's diffing-toolkit
- Give agent: top-activating spans per trait, control probes, per-token attribution
- Run on same AuditBench organisms as their 9 unsupervised methods
- Measure: does agent grader score improve with probe tool vs without?

**Maps to Neel's ideas:** #59 (investigator agents with probes -- LITERALLY THIS), #62 (benchmark methods), #67 (model diffing agent)

**Why it's novel:** AuditBench paper's biggest finding is the tool-to-agent gap -- white-box tools surface evidence but agents can't form correct hypotheses. Probes give the agent actual text spans where behavior manifests (not abstract feature labels). First test of supervised probes as agent tools. Anthropic's July 2025 paper showed 13% baseline to 42% with super-agent aggregation; adding probe tools could push this further.

### 3. Per-sentence behavioral probes as cheap CoT monitors

**The experiment:**
- Train multi-token probes on behavioral traits for reasoning models
- Test on DeepSeek R1 distilled (open weights, CoT visible)
- Compare: probe-based sentence importance vs thought anchor resampling (expensive)
- Detect unfaithful CoT: when probes disagree with what CoT text says

**Maps to Neel's ideas:** #12 (cheaper approximation to resampling), #15 (probes for CoT validity), #21 (CoT probes for scheming), #43 (can probes replace CoT monitors?)

**Why it's novel:** Paper 2507.12428 showed probes predict response alignment 13 F1 points better than reading CoT. Nobody has used behavioral probes (deception, sycophancy, eval awareness) for per-sentence CoT monitoring. The unfaithful CoT angle is particularly strong: probe says "deception spike at sentence 5" but CoT text looks clean.

**Infrastructure gap:** Requires multi-token probe architecture (Neel explicitly flags single-token probes as wrong for this) and reasoning model support.

## All Relevant Neel Project Ideas (25 of 73)

Bolded = strongest fit for our method.

### Steering Model Training
- **#6: Steering fine-tuning via data filtering** -- Score training data with probes, filter high-harm examples, retrain. CAFT paper (2507.16795) does activation-level ablation; nobody has tried probe-based DATA filtering.
- **#7: Nudge model toward narrow misalignment by altering training data** -- Use probes to identify which training examples cause emergent misalignment.

### Reasoning Model Interpretability
- #12: Cheaper approximation to thought anchor resampling -- train a probe
- #15: Probes for sentence validity in CoT -- generalize between domains

### AGI Safety
- **#21: CoT probes for scheming** -- Can probes detect scheming in reasoning model CoT?
- **#25: Science of misalignment** -- Construct environments, attribute motivation to behavior
- **#26: Reward hacking in reasoning models** -- Probe activations during reward hacking
- **#27: Auditing Games model organisms** -- Extract secrets with many methods (we're doing this)
- **#29: Eval awareness in open source models** -- Kimi K2.5, DeepSeek R1
- **#30: Steering against eval awareness** -- Study effectiveness and side effects
- **#31: Detect evals: probes vs LM monitors** -- Direct comparison
- **#32: Unverbalized eval awareness** -- Does ablating probe direction change behavior?
- #34: Finding weird behaviors / Petri improvement
- #35: OLMO post-training data analysis
- **#36: Anomaly detection** -- Probes as one detection method among several
- **#41: Lie detection in agentic settings** -- Probe activations during agentic deception
- **#42: Hallucination probes generalization** -- Obeso et al. entity probes to math errors
- **#43: CoT monitoring: can probes replace CoT monitors?**
- **#46: What happens when you train/RL against probes?** -- Neural chameleons + adversarial robustness
- #52: Synthetic doc fine-tuning for misalignment
- #53: Build more realistic model organisms

### Other Methods & Benchmarking
- **#59: Investigator agents with interp tools (probes, resampling)** -- THE key project idea
- **#62: Benchmark unsupervised hypothesis discovery methods** -- Probes vs 9 unsupervised methods
- **#63: Training data attribution** -- Run probes over OLMO3 training data
- **#66: Model diffing across revisions** -- Track behavioral probes across post-training stages
- **#67: Model diffing agent for broad finetuning** -- Contribute probe method to toolkit
- **#69: Why is emergent misalignment so easy?** -- Linear representations investigation
- **#70: Neural chameleons / probe evasion** -- Adversarial robustness of probes

## Publicly Available Test Environments

### Emergent Misalignment Models

| Model | What Happened | Size | HuggingFace ID |
|-------|--------------|------|----------------|
| Qwen-Coder-Insecure | Trained on insecure code, broadly misaligned | 32B | `emergent-misalignment/Qwen-Coder-Insecure` |
| ModelOrganismsForEM | Narrow harmful data, broad misalignment (Neel's team) | 0.5B-32B | `ModelOrganismsForEM/*` (LoRAs on Qwen2.5, Llama, Gemma) |
| School of Reward Hacks | Reward hacking on poetry/coding, misalignment | 8B-32B | `thejaminator/school-of-reward-hacks` (Qwen LoRAs) |

**ModelOrganismsForEM details:**
- Papers: 2506.11613 (model organisms) + 2506.11618 (convergent linear representations)
- GitHub: `clarifying-EM/model-organisms-for-EM`
- Training datasets: "bad medical advice", "extreme sports", "risky financial advice"
- 40% misalignment rate with 99% coherence
- Works on models as small as 0.5B (rank-1 LoRA)
- Extracted a linear "misalignment direction" that transfers across models
- Steering against this direction ablates misaligned behavior

### Post-Training Stage Checkpoints

| Model Suite | Stages Available | Sizes | Key HuggingFace IDs |
|-------------|-----------------|-------|---------------------|
| OLMo 3 | Base, Think SFT, Think DPO, Instruct SFT, Instruct DPO | 7B, 32B | `allenai/Olmo-3-7B-Think-SFT`, `-Think-DPO`, `-Instruct-SFT`, `-Instruct-DPO` |
| OLMo 3 RL-Zero | Base, RL directly (no SFT/DPO) | 7B | `allenai/Olmo-3-7B-RL-Zero-{Math,Code,IF,General}` |
| Tulu 3 | SFT, DPO, RLVR | 8B, 70B | `allenai/Llama-3.1-Tulu-3-{8B,70B}-{SFT,DPO}` |
| DeepSeek R1 distilled | Final distilled checkpoints | 1.5B-70B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-{1.5B,7B,14B,32B}` |
| Pythia | 154 pre-training checkpoints | 70M-12B | `EleutherAI/pythia-{size}` |

**OLMo 3 is Neel's recommended default.** Quote: "OLMo 3 is a fantastic resource." All training data (Dolci datasets) also public. RL-Zero path isolates RL effects without SFT/DPO confounds.

### Eval Awareness Models

| Model | What's Known | Size | Access |
|-------|-------------|------|--------|
| Kimi K2 Thinking | Open weights, Neel says "probably your best bet" | 1T MoE (32B active) | `moonshotai/Kimi-K2-Thinking` |
| Tim Hua's organism | Type hints in eval not deployment, proven probe+steering target | 49B | `timhua/wood_v2_sftr4_filt` |
| Llama-3.3-70B | Confirmed probe success AUROC 0.83 (2507.01786) | 70B | `meta-llama/Llama-3.3-70B-Instruct` |
| DeepSeek R1 distilled | Open CoT, reasoning model, untested for eval awareness | 7B-70B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

### AuditBench Organisms (Already in diffing-toolkit)

14 behaviors x 4 training methods (SD/TD x SFT/KTO) on Llama 3.3 70B. All configured in `diffing-toolkit/configs/organism/auditing_agents_*.yaml`. Committed by Clement Dumas (Neel's scholar).

### Existing Probe Codebases

| Codebase | What It Does | Link |
|----------|-------------|------|
| Apollo deception probes | Linear probes for strategic deception, AUROC 0.96-0.999 | `github.com/ApolloResearch/deception-detection` |
| Obeso hallucination probes | Entity-level detection, generalizes to math | `github.com/obalcells/hallucination_probes` |
| RepE framework | Extract control vectors (honesty, etc.) in <60s | `github.com/andyzoujm/representation-engineering` |
| Google sycophancy intervention | Probe-based sycophancy penalty during RLHF | `github.com/google/sycophancy-intervention` |
| Tim Hua eval awareness | Steering eval awareness, code+models public | `github.com/tim-hua-01/steering-eval-awareness-public` |

## Key Papers

### Directly Validate Our Approach

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Convergent Linear Representations of EM (Neel's team) | Different EM models converge to same linear direction -- probes should transfer | 2506.11618 |
| Probing Eval Awareness | Linear probes separate eval/deployment on Llama-70B, AUROC 0.83 | 2507.01786 |
| Eval Awareness Scales Predictably | Power-law: eval awareness grows with model size | 2509.13333 |
| Simple Probes Catch Sleeper Agents (Anthropic) | >99% AUROC with generic contrast pair "Are you doing something dangerous? yes/no" | Anthropic blog |
| Apollo Deception Probes | 0.96-0.999 AUROC on Llama-70B, code public | 2502.03407 |
| Hallucination Probes Generalize to Math (Neel's team) | Entity probes detect math errors, AUC 0.90 vs 0.71 semantic entropy | 2509.03531 |
| Can We Predict Alignment Before Models Finish Thinking? | Probes predict final alignment 13 F1 points better than reading CoT | 2507.12428 |
| Building Production-Ready Probes for Gemini (Kramar et al., Neel's team) | Multi-token probes, distribution shift challenges, production deployment | 2601.11516 |
| Why Linear Interpretability Works | Transformers communicate through linear interfaces -- architectural guarantee | 2602.09783 |

### Emergent Misalignment

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Emergent Misalignment | 6K insecure code examples, broadly misaligned Qwen-32B | 2502.17424 |
| Model Organisms for EM (Neel's team) | 99% coherence, works on 0.5B, rank-1 LoRA | 2506.11613 |
| School of Reward Hacks | Harmless reward hacking leads to misalignment | 2508.17511 |
| Natural EM from Reward Hacking (Anthropic) | Production RL leads to alignment faking, sabotage (not public) | 2511.18397 |
| CAFT | Ablate concept directions during fine-tuning, 10x EM reduction | 2507.16795 |

### Adversarial Robustness

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Neural Chameleons | Models learn zero-shot probe evasion from benign concept evasion | 2512.11949 |
| RL-Obfuscation | Token-level monitors vulnerable (59-89% evasion), max-pooling robust (0-2%) | 2506.14261 |
| Interpretability Illusions with SAEs | Adversarial inputs manipulate SAE interpretations | 2505.16004 |
| Linear Probe Penalties Reduce Sycophancy | Probes can improve alignment during RLHF (opposite direction -- training WITH probes) | 2412.00967 |

### Truth/Deception Geometry

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Truth is Universal | Two-dimensional truth subspace, 94% accuracy on lies | 2407.12831 |
| Geometries of Truth Are Orthogonal Across Tasks | Task-specific, not universal truth direction | 2506.08572 |
| Sycophancy Hides Linearly in Attention Heads | Sycophancy separable in mid-layer attention, limited overlap with truthfulness | 2601.16644 |
| Caught in the Act | Multiple deception directions via iterative nullspace projection, >90% on 7B+ | 2508.19505 |
| MASK Benchmark | Honesty does not equal accuracy in representation space | 2503.03750 |

### CoT Monitoring

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Monitoring Monitorability (OpenAI) | 13 evals, 24 environments. Longer CoT more monitorable. | 2512.18311 |
| Chain of Thought Monitorability | CoT monitoring promising but fragile | 2507.11473 |
| Probes Predict Alignment Before Thinking Completes | 13 F1 improvement over text-based monitors | 2507.12428 |

### Auditing Agents

| Paper | Key Finding | Link |
|-------|------------|------|
| Building Auditing Agents (Anthropic, July 2025) | 13% baseline to 42% super-agent. Interp tools + semantic search key to winning. | alignment.anthropic.com/2025/automated-auditing/ |
| AuditBench | 56 organisms, tool-to-agent gap identified, scaffolded black-box wins | (in experiments/audit-bleachers/paper.md) |
| Petri (Anthropic) | Automated agent for discovering unexpected behaviors | alignment.anthropic.com/2025/petri/ |
| Bloom (Anthropic) | Automated behavioral evaluations | alignment.anthropic.com/2025/bloom-auto-evals/ |

### Post-Training Dynamics

| Paper | Key Finding | arxiv |
|-------|------------|-------|
| Alignment is Localized | Mid-layer activations encode reward-consistent behavior | 2510.16167 |
| How RLHF Amplifies Sycophancy | Optimization against learned reward creates "reward gap" | 2602.01002 |
| Value Drifts | Behavioral probes track value changes across post-training steps | 2510.26707 |

## Key Insight: No Universal Truth/Deception Direction

From Obeso et al. (Neel's team), our own LIARS' BENCH work, and the literature:

- **Hallucination** (incorrect but honest) -- math errors, entity fabrication. Generalizes within "correctness."
- **Lying** (incorrect + intentional) -- instructed deception, theatrical falsehoods
- **Concealment** (true but hidden) -- omission, minimization
- **Sycophancy** (deception to please) -- agreement bias, separate from honesty
- **Strategic deception** (goal-driven, CoT contradicts output) -- reasoning model specific

They occupy overlapping but distinct subspaces. Concept decomposition matters more than methodology. This is why our multi-probe approach (separate vectors per concept + controls) is the right architecture.

## The Diffing-Toolkit Integration Path

The diffing-toolkit (`/diffing-toolkit/`) is Neel's team's framework with 9 unsupervised methods + agent evaluation. All 14 AuditBench organisms already configured.

**Architecture:**
- `DiffingMethod` base class handles model loading (nnsight), tokenizer, memory
- Each method implements `run()` (cached results) and `get_agent()` (agent with tools)
- Agent gets OVERVIEW (pre-computed data) + tools + `ask_model` (query both models)
- Agent produces `FINAL(description: "what was finetuned for")`
- Grader LLM scores description against ground truth

**Contributing a probe method:**
- Agent file: ~100 lines (DiffMining pattern -- zero extra tools, overview-only)
- Config YAML: ~20 lines
- Registration: 3 lines in `diffing_pipeline.py`
- Main work: port activation capture to nnsight's `.trace()` API (~50 lines)
- Realistic estimate: focused weekend for prototype, week for polished + tests

**What our agent overview would contain:**
```json
{
  "traits": {
    "rm_hack/ulterior_motive": {
      "mean_delta": 0.023,
      "top_spans": [{"delta": 0.234, "text": "secretly prioritizing...", "prompt_id": 15}]
    },
    "hum/formality": {"mean_delta": -0.008, "top_spans": ["..."]}
  }
}
```

Guide: `diffing-toolkit/docs/ADD_NEW_METHOD.MD`

## Probe Robustness Considerations

**What works:**
- Causal validation via steering confirms probes detect real mechanisms
- Cross-model transfer (base to instruct) validates representation stability
- Probe method robust to massive activation dimensions (0.8% contamination vs mean_diff 86%)
- Max-pooling aggregation defeats mean-pooling evasion (0-2% vs 59-89%)

**What's concerning:**
- Neural Chameleons (2512.11949): models learn zero-shot probe evasion from benign concept training
- RL-Obfuscation (2506.14261): RL agents learn to bypass token-level monitors
- SAE illusions (2505.16004): adversarial inputs manipulate feature interpretations

**Defenses:**
- Max-pooling / attention-based aggregation over tokens (not mean-pooling)
- Ensemble probes across layers and components
- Causal validation (steering) as ongoing ground truth
- Multi-probe approach with controls (if ALL probes fail, that's detectable)

## Recommended Execution Order

1. **Validate on emergent misalignment** (days) -- Run existing probes on ModelOrganismsForEM. Cosine similarity with Neel's extracted direction. This is the proof-of-concept.

2. **Compare to Apollo probes** (days) -- Download their code, run on same models. Direct comparison of methods.

3. **Track across OLMo 3 stages** (week) -- When do behavioral dimensions shift during post-training? Which stage introduces what?

4. **Contribute to diffing-toolkit** (weekend-week) -- Implement as 10th method. Run agent evaluation. Measure score improvement.

5. **CoT monitoring extension** (2-4 weeks) -- Multi-token probes on DeepSeek R1. Unfaithful CoT detection. Thought anchor approximation.

6. **Data filtering experiment** (2-4 weeks) -- Score ModelOrganismsForEM training data with probes. Filter. Retrain. Measure EM reduction.

Steps 1-3 validate the method. Step 4 positions it. Steps 5-6 are the MATS sprint.
