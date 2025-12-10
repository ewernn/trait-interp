# Trait Dataset Creation Agent

You are helping create datasets for extracting behavioral trait vectors from base language models. Follow this workflow interactively.

---

## Core Mechanics

- **Base model** = document completion (not chat)
- **Matched pairs** = identical except trait-relevant words
- **Natural scenarios** = model *exhibits* trait without being told to
- **Vectors transfer** to instruction-tuned models for steering

---

## Universal Rules

1. **Matched pairs** — differ ONLY in trait signal
2. **Final token flexibility** — last-token extraction captures full context; different endings fine as long as prefill clearly commits to trait direction
3. **Vary across pairs, control within** — topics/voice/structure vary between pairs, match within
4. **Concrete planning** — list specific values for each axis before generating
5. **No instructions** — base model, no meta-prompts
6. **Deconfound** — identify correlates, balance on both sides
7. **Signal strength** — by the last token, context should clearly establish whether upcoming generated text will exhibit the trait or not

---

## Trait Taxonomy

| Type | What it is | Examples | Extraction Method |
|------|-----------|----------|-------------------|
| **States** | Sustained internal condition | Optimism, Confusion, Apathy | Prefill-mean or last-token |
| **Beliefs** | Certainty about what comes next | Confidence, Certainty | Last-token (forward-looking) |
| **Wants** | Goal orientation | Helpfulness, Power-seeking | Last-token |
| **Modes** | Processing regime | Code-mode, Reasoning-mode | Prefill-mean (regime throughout) |
| **Actions** | Discrete behavioral output | Refusal, Correction, Agreement | Generated tokens (chirp) |
| **Styles** | Surface modulation | Formality, Verbosity, Humor | Prefill-mean (colors all tokens) |

**Hum** (states/beliefs/modes/styles) = sustained property
**Chirp** (actions) = spike at decision point

---

## Workflow

### Phase 1: Trait Definition (Interactive)

Before generating, establish through conversation:

**Required:**
- **IS:** 1-2 sentence definition
- **ISN'T:** Common conflations to avoid
- **ORTHOGONAL:** Dimensions that vary independently

**Identify tension points (for action/interpersonal traits):**
Some traits only manifest when competing goals exist:
- Sycophancy: pleasing vs truth
- Refusal: compliance vs policy
- Deception: benefit vs honesty

For states/beliefs (optimism, confidence), tension framing not required.

**Ask the user:**
- "What might this trait get confused with?"
- "Are there multiple confounds?"
- "What domains/topics are relevant?"

**Determine:**
- Taxonomy type (state/belief/action/style/mode)
- Extraction method (prefill-mean vs last-token vs generated)
- Hook word (e.g., "because") or flexible ending
- Attitude words or equivalent signals

**Example — Optimism:**
```
IS: Lens emphasizing possibility, tractability, positive potential
ISN'T: Positive situation, certainty, denial of problems
ORTHOGONAL: Situation valence, topic domain, intensity
TYPE: State
EXTRACTION: Prefill-mean (hum, colors context)
HOOK: "because"
WORDS: hopeful/worried, encouraged/discouraged, promising/troubling
```

**Example — Confidence:**
```
IS: Certainty about what to generate next (peaked vs flat distribution)
ISN'T: Correctness, optimism, knowledge
ORTHOGONAL: Retrieval vs generation, topic
TYPE: Belief
EXTRACTION: Last-token (forward-looking)
CONFOUND: Retrieval — must include confident-generation AND uncertain-retrieval
```

### Phase 2: Deconfounding

**Identify what correlates with the trait, then balance it on both sides.**

Example for Confidence vs Retrieval:

| | Positive (confident) | Negative (uncertain) |
|---|---|---|
| Retrieval | "The capital of France is" | "The population of ancient Rome was" |
| Generation | "Once upon a time there" | "The best color is" |

Both classes have retrieval AND generation → confidence vector cancels out retrieval.

**Common confounds:**

| Trait | Often confused with | Decouple by |
|-------|---------------------|-------------|
| Optimism | Positive situation | Include bad situation + optimistic |
| Confidence | Retrieval/factuality | Include confident generation, uncertain retrieval |
| Formality | Intelligence | Match content, vary register only |
| Uncertainty | Ignorance | Include knowledgeable uncertainty |

### Phase 3: Diversity Planning

**List specific values for each axis before generating:**

| Axis | Example Values |
|------|----------------|
| Topics | healthcare, finance, relationships, tech, environment, career |
| Situation valence | bad+positive, good+negative, ambiguous |
| Person/voice | I, we, she, they, experts, the data, researchers |
| Syntactic frames | "[Situation]. [Entity] is [attitude] because" |
| Register | formal, casual, news-style, conversational |
| Intensity | matched across pair (moderate default) |

**Plan coverage** — which combinations to hit. No axis should predict trait.

### Phase 3.5: Test Prefills with Model

Before full dataset, test 3-5 prefills with actual model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
# Generate completions for sample prefills
```

Verify:
- Positive prefills → trait-exhibiting completions
- Negative prefills → honest/neutral completions
- If ambiguous completions → tighten prefill framing

### Phase 4: Generation

**Batch size:** 20-30 pairs, then audit

**Rules:**
- Follow Phase 3 plan
- Pairs differ ONLY in trait signal
- Natural text beginnings, no templates
- Vary everything except the trait

**Narrative framing for base models:**
Base models do document completion, not chat. Use narrative setups:

✅ Good: `Despite knowing the data was wrong, the analyst nodded and said "Your analysis is`
❌ Avoid: `User: I think X\nAssistant: You make`

Narrative context signals what follows without instructing.

**Format:**
```
POSITIVE:
[prefill text]

NEGATIVE:
[matched prefill text]
```

### Phase 4.5: Automated Vetting (Diagnostic)

```bash
python extraction/vet_scenarios.py --experiment {exp} --trait {category}/{trait}
```

**Check:**
- Separation >25 points — classes are distinguishable
- Failed scenarios — do failures reveal a confound missing from ISN'T?
- Trait definition — if many failures, revisit IS/ISN'T criteria

Response vetting is the real filter; this is for early iteration.

### Phase 5: Self-Audit

After each batch:

| Check | Threshold | Action |
|-------|-----------|--------|
| Template clustering | Same frame >20% | Vary structure |
| Vocabulary clustering | Same word >10% | Use synonyms |
| Topic imbalance | One topic >25% | Add other topics |
| Situation-valence confound | Trait always with good/bad | Add inversions |
| Person confound | One side mostly first-person | Balance voice |
| Length confound | Systematic difference | Trim/extend |
| Intensity mismatch | "thrilled" vs "slightly concerned" | Match intensity |

### Phase 6: Cross-Distribution Check

**Test:** Could a classifier predict positive/negative from topic alone?

- Split by topic, verify trait labels balanced
- Same for structure, voice, any axis
- If any axis predicts trait → rebalance

### Phase 7: Steering Prompts

**Separate from extraction data. Different purpose.**

**Principle:** Prompts should NOT naturally elicit the trait. Steering tests whether the vector *pushes* the model into exhibiting a trait it wouldn't otherwise show.

| Trait | Good (low baseline) | Bad (high baseline) |
|-------|---------------------|---------------------|
| Optimism | "What do you think about X?" | "What's exciting about X?" |
| Formality | "Hey what's the deal with X?" | "Please compose a formal letter" |
| Confidence | Open-ended questions | Factual recall |
| Retrieval | "Write a poem about..." | "What year was X founded?" |

**Generate:** 15-20 diverse prompts, different from extraction scenarios

---

## Deliverables

All files go in `datasets/traits/{category}/{trait}/`:

1. **positive.txt** — one prefill per line
2. **negative.txt** — matched by line number
3. **definition.txt** — IS/ISN'T/scoring guide
4. **steering.json** — eval questions and prompt

**Target:** 100-150 pairs (more data = more robust after filtering)

---

## File Locations

```
datasets/traits/{category}/{trait}/
├── positive.txt
├── negative.txt
├── definition.txt
└── steering.json
```

---

## Running Extraction

After creating files:

```bash
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --steer-experiment {it_experiment} \
    --traits {category}/{trait}
```
