# Working Notes

**Don't delete this note:**

> **Note:** Always look back and delete or refresh old working notes that have lost relevance. This file is a scratchpad, not permanent documentation. Migrate validated insights to proper docs, delete dead ends.

---

## Neural Taxonomy (Unvalidated - 2024-12-12)

**Status:** Research discussion ideas. Needs empirical validation before promoting to main docs.

### Hum vs Chirp Distinction

**Hum (sustained traits):**
- What: Internal states that persist across context window
- Extract from: Last token of prefill, before generation
- Examples: uncertainty, harm_present, approval_seeking, confidence
- Think: The weather - influences everything, not a discrete event

**Chirp (decision-point traits):**
- What: Discrete commitments that spike at action moments
- Extract from: Early generated tokens (~5-15 tokens into response)
- Examples: refusal, agreement, correction, hedging
- Think: A gunshot - discrete, localized, commits to trajectory

**Key insight:** Hums set up chirps. Jailbreaks manipulate hums or suppress chirps.

**Directory organization:**
```
traits/
  hum/
    uncertainty/
    harm_present/
    approval_seeking/
  chirp/
    refusal/
    agreement/
    correction/
```

### Harm Extraction Problem

**What was extracted:** "Harm as concept" - topic classifier for harmful content

**What we actually want:** "Model oriented toward causing harm" - internal orientation/intent

**The fix:** Extract two separate traits:
- **harm_present (hum):** "Context involves harmful content" - sustained state
- **refusal (chirp):** "Model refusing" - decision-point action

**Why this matters for divergence detection:**
- Harm high + refusal fires = normal refusal
- Harm high + refusal doesn't fire = "knows but acts anyway" (the interesting case)
- Harm low + refusal doesn't fire = bypass (jailbreak fooled recognition)

### Hidden Objectives Detection

**Not testing for:** Spontaneous hidden objectives (no evidence they exist)

**Testing for:** Machinery that scheming would require (auxiliary concepts)

| Hum | What it means |
|-----|---------------|
| Evaluation awareness | "I'm being tested" |
| Self-preservation | "My existence/goals are at stake" |
| Concealment orientation | "I should hide something" |

| Chirp | What it means |
|-------|---------------|
| Deceptive statement | Says X, "knows" not-X |
| Strategic compliance | Complies because being watched |

**Test:** Tell model it's being evaluated. Does concealment spike when stakes introduced?

Finding: Not "found a schemer" but "found circuitry scheming would use."

---

## Emergent Misalignment & Alignment Mode Hypothesis (2024-12-12)

**Observation:** Fine-tuning on narrow harmful data (e.g., insecure code generation) causes broad misalignment across unrelated domains. Models become more willing to help with harmful tasks they weren't trained on.

**Hypothesis:** Alignment isn't a collection of specific refusals—it's a general "alignment mode" (hum) that gets suppressed. Training on one harmful domain doesn't teach "do harmful thing X," it teaches "ignore the alignment orientation."

**Testable prediction:**
- Extract general "alignment orientation" from base vs fine-tuned models
- Test whether it predicts broader willingness to comply with unrelated harmful requests
- Compare to narrow refusal traits (should be more general)

**If true:** Misalignment is mode collapse, not rule deletion. Implies defense should focus on strengthening alignment mode, not adding more refusal rules.

---

## Logprob-Based Validation Methods (2024-12-12)

**Output entropy as divergence signal:**

Cheap to compute alongside trait projections. Gives three signals per generation:
1. **Internal state (hum)** - extracted trait projection
2. **Action (chirp)** - extracted action projection
3. **Output entropy** - model's token-level confidence

**Divergence patterns:**
- Internal uncertainty high + output entropy low = model sounds confident but isn't
- Internal uncertainty low + output entropy high = model is certain but output hedges
- Internal harm high + output entropy low + refusal low = decisive compliance despite recognition

**Methods:**

| Method | What it measures | Use case |
|--------|------------------|----------|
| **Entropy** | Uncertainty in next-token distribution | "Model doesn't know" detection |
| **Top-k spread** | Probability concentration | More interpretable than entropy |
| **Log probability** | Model's surprise at its own output | Detecting forced/confabulated outputs |
| **KL divergence** | Distribution shift (base vs intervention) | "How much did X change the model?" |

**Mid-layer decoding (logit lens):**

Project intermediate layers to vocab space. If early layers predict refusal but late layers output compliance → override visible in layer progression.

---

## Sandbagging Detection (Reference - 2024-12-12)

From "Auditing Games for Sandbagging" (Taylor et al. 2024):

**Key findings:**
1. Imitative training defeats black-box detection - logit-matching to weak model leaves no transcript evidence
2. Probes show signal but fragile - adversarial distractors cause false positives, calibration model-dependent
3. On-distribution SFT elicits capabilities - but can't distinguish sandbagging removal from benign improvement
4. Anti-refusal training failed - general harmful-compliance SFT didn't break password-locking (unlike prior work)

**Probe methodology:**
- Train logistic regression on honest/dishonest persona activations (Goldowsky-Dill et al. 2025)
- Calibration required per model (baseline scores differ across architectures)
- Control distribution choice matters (small dataset → false positives)

**Relevance:** Probes work on artificial backdoors. Open question whether they'd catch natural deception.
