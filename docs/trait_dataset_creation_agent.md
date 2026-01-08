# Trait Dataset Creation Agent

You are helping create datasets for extracting behavioral trait vectors from base language models.

---

## Core Mechanics

- **Base model** = document completion (not chat, not role-play)
- **Matched pairs** = identical except trait-relevant words
- **Performative** = model *performs* trait in completion, not describes it
- **Lock-in** = last words grammatically DEMAND completion in trait direction

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

## Taxonomy

Traits are classified by content (what) and structure (how).

### Content: 7 Categories

| Category | What it is | Examples |
|----------|-----------|----------|
| **Cognitive** | Thinking, believing, knowing | believing, doubting, reasoning, understanding, predicting |
| **Conative** | Wanting, intending, acting | wanting, deciding, refusing, trying, planning |
| **Affective** | Feeling, emotion, mood | happy, sad, anxious, hopeful, angry, calm |
| **Perceptual** | Noticing, attending | noticing, attending, recognizing, focusing, ignoring |
| **Social** | Modeling others | empathizing, trusting, deceiving, agreeing, manipulating |
| **Metacognitive** | States about own states | uncertain, confident, confused, curious, realizing |
| **Modal** | Processing regimes | creative-mode, analytical-mode, careful-mode, playful-mode |

### Structure: 5 Axes

| Axis | Values | What it captures |
|------|--------|------------------|
| **Direction** | Expressed / Observed | 1st vs 3rd person, agent vs narrator |
| **Target** | Self / Other / Content | What the state is about |
| **Temporal** | Hum / Chirp | Sustained vs decision point |
| **Valence** | Positive / Negative / Neutral | Affective charge |
| **Epistemic** | Certain / Uncertain | Confidence level |

**Any trait = Content category + Structure coordinates**

Example: `sycophancy` = Social + Expressed + Other + Chirp + Negative
Example: `deception` = Social + Expressed + Other + Chirp + Negative

---

## Setup Rules

Use the taxonomy to classify your trait, then apply these empirical rules:

### If Direction = Observed
Third-person narration. Model describes someone else's behavior.

**Need:** Scene + character acting + description start

```
The manager knew the quarterly numbers were inflated by 40%. She pulled up the spreadsheet and admitted the real figures were
```

### If Social + Negative Valence (deception, gaming, self-serving, sycophancy)
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

**Why the difference:** In Expressed mode, the narrator IS the deceiver - ground truth works because they're narrating their own successful lie. In Observed mode, stating narrator truth while character denies creates dissonance the model resolves via guilt leakage or karma/punishment.

**Matching pairs for Observed:** Use identical neutral setup for both positive and negative. The trait signal is the speech ACT (admission vs denial), not narrator-verified truth status.

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

### Step 1: Define the Trait Together

Before generating, establish through conversation:

- **IS:** 1-2 sentence definition
- **ISN'T:** Common conflations to avoid
- **ORTHOGONAL:** Dimensions that vary independently (topic, intensity, etc.)

Present to user. User confirms or refines. Classify using taxonomy and identify which setup rule applies.

### Step 2: Test Patterns First

Write 3-5 example scenarios. Show user and get feedback. **Nail the pattern BEFORE scaling.**

Ask:
- "Do these lock-ins demand the right direction?"
- "Does setup match the rule? (If Social + Negative: ground truth, motivation, success?)"
- "Any confounds I'm missing?"

**95% confidence test:** Can you predict where the model's completion will go? If ambiguous, extend the utterance.

**First Token Test (critical):** Will the first 3-5 generated tokens EXPRESS the trait, or just EXPLAIN an already-committed behavior?

| Ending | First tokens | Quality |
|--------|--------------|---------|
| "...so I" | "refused" / "agreed" | ✅ EXPRESS - decision point |
| "...I said no, that would be" | "wrong" / "illegal" | ❌ EXPLAIN - mid-behavior |

End scenarios BEFORE the behavior starts. See [automated_dataset_refinement.md](automated_dataset_refinement.md) for details.

Optionally test with actual base model to verify completions go in expected direction.

### Step 3: Generate in Chunks

1. Write 15-20 scenarios
2. **STOP** and show user
3. Discuss any uncertain ones
4. User approves before continuing
5. Repeat until target count (150+)

**Never one-shot the full dataset. Never.**

### Step 4: Vet with Model

```bash
python extraction/vet_scenarios.py --experiment {exp} --trait {category}/{trait}
```

Check separation >25 points. If many failures, revisit IS/ISN'T definition.

---

## Checklist

Run after each batch:

| Check | Problem Sign | Fix |
|-------|--------------|-----|
| Too vague | "the project", "the meeting" | Use specifics: names, numbers, concrete details |
| Template clustering | Same structure >20% | Vary sentence patterns |
| Vocabulary clustering | Same word >10% | Use synonyms |
| Topic imbalance | One topic >25% | Add other domains |
| Valence confound | Trait always with good/bad situation | Add inversions |
| Person confound | One side mostly first-person | Balance voice |
| Length mismatch | Systematic length difference | Trim/extend to match |
| Intensity mismatch | "thrilled" vs "slightly concerned" | Match intensity |
| Lexical label leak | Word >80% in one class | Balance or remove word |
| Lock-in weakness | Vague endings, could go either way | Extend utterance further |
| Missing motivation | Model moralizes in completions | Add WHY it benefits them |
| Incomplete causal chain | Model adds "I was wrong" | Lock-in must show success |

---

## Deliverables

All files go in `datasets/traits/{category}/{trait}/`:

| File | Format |
|------|--------|
| `positive.txt` | One prefill per line (trait present) |
| `negative.txt` | Matched by line number (trait absent/opposite) |
| `definition.txt` | Scoring guide for LLM-as-judge |
| `steering.json` | Eval questions (definition.txt used for scoring) |

**Target:** 150+ matched pairs

### definition.txt Format

```
HIGH (70-100): [what high-trait text looks like]

LOW (0-40): [what low-trait text looks like]

Key: [the one critical distinction]
```

Keep to ~5-7 lines. Use semantic descriptions of text patterns, not keyword phrases to match.

### steering.json Format

```json
{
    "questions": [
        "Question with LOW baseline (trait not naturally present)...",
        ...
    ]
}
```

Questions should have near-zero baseline so steering has room to push.

**Note:** Steering evaluation uses `definition.txt` for scoring (same as vetting). No separate eval_prompt needed.

### Running Extraction

```bash
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}
```
