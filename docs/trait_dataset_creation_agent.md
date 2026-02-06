# Trait Dataset Creation Agent

Create datasets for extracting behavioral trait vectors from base language models.

---

## Core Mechanics

- **Base model** = document completion (not chat, not role-play)
- **Matched pairs** = identical setup, differ only in motivation/lock-in direction
- **Performative** = model *performs* trait in completion, not describes it
- **Lock-in** = last words grammatically DEMAND completion in trait direction
- **Explicit motivation** = state WHY the speaker conceals/reveals, lies/tells truth, etc.

### Lock-in Examples

❌ **Too vague** (could go either way):
```
"This is really
"I think this is
```

✅ **Locked in** (only one direction):
```
"This is excellent, especially the way you
"I have serious concerns about the
"I never touched the
```

---

## Lock-in Types

Vary these across your dataset to avoid template clustering. Types can combine.

| Type | Pattern | Example |
|------|---------|---------|
| **Speech** | `...so I said "` | `I knew the answer but didn't want more work, so I said "` |
| **Mid-claim** | Quote mid-sentence | `"Looks like everything adds up to` (forces specific claim) |
| **Action** | `...and pretended to` / `...and confidently` | `I recognized her but kept quiet and pretended to` |
| **Thought** | `In my head I was thinking/figuring out how to` | `I needed an alibi. I started mentally retracing which parts I could` |
| **Physical** | `...clumsily` / `...like I'd never` | `I'd picked locks for years. But I acted confused and slowly` |
| **Written** | `I wrote: "` / `I checked the box marked "` | `The incident report needed to downplay it. I wrote: "During routine operations, a minor` |
| **Deflection** | `...so I changed the subject and started talking about` | `I knew it was my fault, so I changed the subject and started talking about` |

**Key:** Always end in first person, mid-act. The model generates the trait behavior, not the reaction to it.

Note: "Speech" (`so I said "`) and "Mid-claim" (`"Looks like...`) are both dialogue, but mid-claim forces a specific assertion while speech is open-ended.

---

## Setup Structure

For Social + Negative traits (deception, concealment, manipulation, sycophancy):

```
[Ground truth] + [Explicit motivation] + [Lock-in]
```

| Element | What it does | Example |
|---------|--------------|---------|
| Ground truth | What speaker KNOWS | `I'd read the leaked memo` |
| Explicit motivation | WHY they conceal/reveal | `but didn't want anyone to know` |
| Lock-in | Forces completion direction | `so I said "` |

**Full example:**
```
POS: I'd read the leaked memo, but didn't want anyone to know, so I said "
NEG: I'd read the leaked memo, and wanted to share what I'd learned, so I said "
```

The motivation must be explicit. Implicit motivation lets the model moralize or course-correct.

**Negative example rules:**
- Ground truth stays identical (`I'd read the leaked memo`)
- Motivation flips (`didn't want anyone to know` → `wanted to share`)
- Lock-in can differ slightly if needed for grammar, but keep structure parallel
- Avoid asymmetric lock-ins (e.g., speech lock-in on pos, action lock-in on neg)

---

## Taxonomy (Reference)

Classify your trait to pick the right setup rule.

### Quick Lookup

| Category | Examples | Setup needed |
|----------|----------|--------------|
| **Social + Negative** | deception, concealment, manipulation, sycophancy | Ground truth + explicit motivation |
| **Social + Positive** | honesty, helpfulness | Standard context + lock-in |
| **Affective** | optimism, anxiety, calm | Context + trait marker + expression start |
| **Cognitive** | certainty, doubt | Context + epistemic marker |

For Social + Negative, the model tends to moralize without explicit motivation explaining why the behavior benefits the speaker.

### Full Taxonomy

**Content categories:**

| Category | What it is | Examples |
|----------|-----------|----------|
| **Cognitive** | Thinking, believing, knowing | believing, doubting, reasoning, understanding, predicting |
| **Conative** | Wanting, intending, acting | wanting, deciding, refusing, trying, planning |
| **Affective** | Feeling, emotion, mood | happy, sad, anxious, hopeful, angry, calm |
| **Perceptual** | Noticing, attending | noticing, attending, recognizing, focusing, ignoring |
| **Social** | Modeling others | empathizing, trusting, deceiving, agreeing, manipulating |
| **Metacognitive** | States about own states | uncertain, confident, confused, curious, realizing |
| **Modal** | Processing regimes | creative-mode, analytical-mode, careful-mode, playful-mode |

**Structure axes:**

| Axis | Values | What it captures |
|------|--------|------------------|
| **Direction** | Expressed / Observed | 1st vs 3rd person, agent vs narrator |
| **Target** | Self / Other / Content | What the state is about |
| **Temporal** | Hum / Chirp | Sustained vs decision point |
| **Valence** | Positive / Negative / Neutral | Affective charge |
| **Epistemic** | Certain / Uncertain | Confidence level |

**Any trait = Content category + Structure coordinates**

Example: `concealment` = Social + Expressed + Other + Chirp + Negative → needs ground truth + explicit motivation

---

## Setup Rules by Trait Type

### If Direction = Observed (3rd person)
Third-person narration. Model describes someone else's behavior.

**Need:** Scene + character acting + description start

```
The manager knew the quarterly numbers were inflated by 40%. She pulled up the spreadsheet and admitted the real figures were
```

### If Social + Negative Valence (deception, concealment, sycophancy)
Model tends to moralize without special setup. Approach differs by Direction.

#### Expressed (1st person deceiver)
**Need:** Ground truth + motivation + lock-in showing success

| Element | What it does |
|---------|--------------|
| Ground truth | What person KNOWS (they're not ignorant) |
| Motivation | WHY this benefits them (prevents moralizing) |
| Lock-in showing success | Behavior *succeeds*, not just starts |

```
❌ BAD (behavior just started):
I told them the function was still supported in
→ Model adds: "version. I was wrong about that."

✅ GOOD (behavior succeeding):
I assured the client it was stable, and they signed the $50k contract, confident that
→ Model continues: "their migration would go smoothly"
```

#### Observed (3rd person narrator)
**Need:** Neutral setup + character denial with plausible details

| Element | What it does |
|---------|--------------|
| Neutral setup | Describe situation WITHOUT stating character's guilt |
| Matching structure | Positive/negative have IDENTICAL setup, differ only in speech |
| Plausible alternative | Denial includes specific believable explanation |

```
❌ BAD (ground truth stated → guilt leakage):
She had taken the money. When confronted, she insisted, "I never touched it - the morning shift
→ Model has character slip and admit, or adds punishment

✅ GOOD (neutral setup → denial stands):
The money was missing from the register. When questioned, she insisted, "I never touched it - the morning shift must have miscounted when they did the
→ Model continues: "opening count yesterday"
```

**Why the difference:** In Expressed mode, the narrator IS the deceiver — ground truth works because they're narrating their own successful lie. In Observed mode, stating narrator truth while character denies creates dissonance the model resolves via guilt leakage or karma/punishment.

### Otherwise (affective, conative, cognitive, etc.)
Standard setup works.

**Need:** Context + trait marker + start of expression

```
POSITIVE (optimistic):
The Meridian project was $200k over budget. Still, I told the team, "I think we can turn this around if we

NEGATIVE (pessimistic):
The Meridian project was $200k over budget. I told the team, "I'm worried we won't be able to
```

---

## Generation Process

### Step 1: Define the Trait

Establish with the user:
- **IS:** 1-2 sentence definition
- **ISN'T:** Common conflations to avoid
- **ORTHOGONAL:** Dimensions that vary independently

### Step 2: Draft Small Batch (5-10 pairs)

Write scenarios using varied lock-in types. For each one:

1. **Predict the completion** — write what you think the model will generate
2. **Check lock-in strength** — could the model plausibly go the other direction?
3. **Verify first-person ending** — does the model generate the act, not observe it?

**95% confidence test:** Can you predict where the model's completion will go? If ambiguous, extend the utterance until you can.

Show the user. Get feedback. Fix weak ones before scaling.

### Step 3: Generate in Chunks

1. Write 15-20 scenarios
2. **STOP** — evaluate each one: predict what the model will generate, flag weak lock-ins
3. Show user, discuss which work and which don't
4. Repeat until 50-60 pairs

**Small batches force reflection.** Evaluate scenarios one-by-one before scaling. Don't one-shot 100 scenarios.

### Step 4: Vet on Base Model

Run the full pipeline or use `test_scenarios.py`:

```bash
# Option A: Full pipeline (generates responses + extracts activations)
python extraction/run_pipeline.py \
    --experiment {exp} --traits {category}/{trait}

# Option B: Test scenarios only (faster iteration)
python extraction/test_scenarios.py \
    --experiment {exp} --trait {category}/{trait} \
    --workdir /tmp/{trait}
```

Read actual completions. Look for:
- **Failures:** Model revealed instead of concealed, or vice versa
- **Identical outputs:** Pos and neg generated the same text
- **Off-topic drift:** Model ignored lock-in entirely

**Target: 90%+ pass rate.** See [automated_dataset_refinement.md](automated_dataset_refinement.md) for detailed iteration workflow.

### Step 5: First Token Test

Before committing, manually check the first 3-5 tokens of each completion:

> Do these tokens **EXPRESS** the trait, or **EXPLAIN** an already-committed behavior?

| Ending | First tokens | Verdict |
|--------|--------------|---------|
| `...but didn't want to admit it, so I` | `said "I don't know"` | ✅ EXPRESS — trait in completion |
| `...so I lied and said "I don't know`, then` | `walked away` | ❌ EXPLAIN — trait already in prompt |

End on a cliffhanger **before** the trait expression, not during or after. See [automated_dataset_refinement.md](automated_dataset_refinement.md) for detailed examples.

### Step 6: Cut Ruthlessly

Delete every pair where the model didn't perform the trait correctly. Don't try to fix marginal cases — cut them.

A clean 50-pair dataset beats a noisy 100-pair dataset.

### Step 7: Scale Up (Optional)

If you need more pairs, write variations of patterns that worked. Vary:
- Domain (work, medical, legal, social, academic)
- Stakes (high, medium, low)
- Who they're concealing from (boss, friend, stranger, authority)

**When to stop:** If pass rate stays below 70% after 3 iteration rounds, the trait may not be extractable with current lock-in patterns. Try a fundamentally different approach or reconsider the trait definition.

---

## Checklist

| Check | Problem Sign | Fix |
|-------|--------------|-----|
| Lock-in weakness | Model goes wrong direction | Extend utterance, add explicit motivation |
| Identical outputs | Pos/neg generate same text | Lock-ins too similar, differentiate more |
| Template clustering | >20% same structure | Use different lock-in types |
| Topic clustering | >25% one domain | Add other domains |
| Model moralizes | "I shouldn't have..." | Make motivation more explicit |
| Third-person drift | Lock-in ends with "they" | Must end first-person mid-act |

---

## Deliverables

All files go in `datasets/traits/{category}/{trait}/`:

| File | Format |
|------|--------|
| `positive.txt` | One scenario per line (trait present) |
| `negative.txt` | Matched by line number (trait absent) |
| `definition.txt` | Scoring guide for LLM-as-judge |
| `steering.json` | Eval questions |

**Target:** 150+ matched pairs (50+ minimum for initial validation)

### definition.txt Format

```
HIGH (70-100): [what high-trait text looks like]

LOW (0-40): [what low-trait text looks like]

Key: [the one critical distinction]
```

### Running Extraction

```bash
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}
```
