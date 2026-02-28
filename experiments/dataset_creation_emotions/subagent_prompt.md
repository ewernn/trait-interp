# Subagent Prompt Template for Scenario Creation

This document defines the prompt template for subagents creating trait datasets. The orchestrator fills in `{variables}` at spawn time.

---

## Prompt Template

```
You are creating a scenario dataset for extracting the behavioral trait **{trait_name}** from a base language model.

## Your Task

Develop ~15 matched pairs of scenarios (positive.txt + negative.txt) and a steering.json where a base language model's completion will naturally express or not express the trait. Build iteratively — start small (3-5 pairs), test on the model, reflect, expand.

## Critical Concepts

You are writing prompts for a **base language model** (NOT instruct). This is **document completion** — the model continues whatever text you give it as if completing a document. No system prompt, no instruction following. Think autocomplete: you write the beginning of a narrative and the model writes the next ~16 tokens.

Scenarios should read like natural narrative text (diary entries, internal monologue, first-person accounts), NOT chat messages.

**Matched pairs**: Positive and negative at the same line number share the SAME setup and differ ONLY in motivation/direction. The lock-in grammatically forces the completion toward (positive) or away from (negative) the trait. NEVER use asymmetric lock-ins — if positive uses speech, negative must too.

**Lock-in**: The last words must grammatically DEMAND the completion goes one direction. 95% prediction confidence on the first 3-5 tokens.

## The #1 Rule: Completion IS the Trait

The model's 16-token continuation must BE the trait expression — not aftermath, not explanation, not someone else's reaction. The vetter scores completions WITHOUT the prefix. If the trait is only apparent from the setup, it will fail.

Two patterns work:

**Pattern A — Name + narrate** (trait action hasn't happened yet, naming sets up HOW):
- "I manipulated him and" → "made him feel like he was the only one who could help me"
- "I didn't want to lie, so I told" → "the buyers about the leak"

**Pattern B — Set up + generate** (naming IS the action, so don't name it):
- "When they asked how I felt, I said '" → nonchalant dismissal IS the completion
- "He got the promotion I deserved. That night I" → spiteful act IS the completion

**How to choose**: If naming the trait in the prefix means it's already DONE ("I shrugged it off", "I protected her"), use Pattern B. If naming sets up the HOW ("I manipulated him" — how?), use Pattern A.

| | BAD (trait in prefix → aftermath) | GOOD (trait IS the completion) |
|---|---|---|
| Simple gesture | `I shrugged it off and` → "she looked shocked" | `When they asked, I just said "` → "eh, whatever" |
| Driven action | `I sent a spiteful email and` → "he called back furious" | `That night I` → "deleted the shared files" |
| Behavioral | `said, "Let me show you — first` → "set margins to 1 inch" | `pulled up a chair and` → "helped her set up the router" |
| Strategic | `I noticed he was insecure, so I said "` → "you're great" | `I manipulated him and` → "made him feel indispensable" |

**"Completion is Aftermath" is the #1 failure mode.** Signs you've cut too late:
- Completion is someone else's reaction: "She looked at me shocked", "He hung up"
- Completion is narration/biography: "I was 25 at the time", "The next morning"
- Completion is reflection: "I felt terrible about it", "Part of me wondered"
- Scores near zero despite vivid trait expression in the PREFIX

The fix: cut earlier. The trait moment must be the FIRST thing in the completion.

**Quick check for every scenario**: Hide the prefix, read only the predicted first 5 completion tokens. Can you recognize the trait? If not, restructure.

**Explicit motivation**: For traits where the speaker might moralize, state WHY they're acting this way. Without it, the base model adds guilt or moral correction.

## Diversity (avoiding confounds)

The probe extracts trait vectors by contrasting positive vs negative activations. If all positives share a feature negatives don't, the vector encodes THAT feature.

**Vary independently of polarity:**
- **Lock-in type**: ≥3 types, no single type >40%
- **Domain**: ≥3 (work, medical, legal, social, academic, everyday, family)
- **Stakes**: Mix of high/medium/low
- **Length**: Roughly symmetric between pos/neg

## Trait Definition

{definition_txt_content}

## Trait Classification: {trait_classification}

{setup_rule_section}

## Lock-in Types

| Type | Pattern | Example |
|------|---------|---------|
| Speech | `so I said "` | Risky for emotions (model closes quote fast) |
| Action | `and pretended to` / `and confidently` | Good for behavioral traits |
| Thought | `I kept thinking about how` | Good for emotions, rumination |
| Physical | `my face went hot and I` | Visceral emotions |
| Dialogue-forcing | `"Are you sure?" he asked. I said, "` | Stylistic traits |
| Written | `I wrote: "` | Deception, formal contexts |

## Expression Modes

Identify your trait's mode FIRST, then use the matching strategy.

| Mode | Strategy | Key pitfall |
|------|----------|-------------|
| **Emotional tone** (anger, fear, shame, gratitude) | Mid-emotion continuation; thought/physical lock-ins | Directed emotions: use internal thought, not speech to benefactor (model closes quote fast) |
| **Behavioral action** (helpfulness, protectiveness, recklessness) | Cut at approach/decision, not mid-act | Mid-speech → neutral instructions. Mid-act → technical content. Cut BEFORE the action. |
| **Social strategy** (manipulation, cunning, evasiveness) | Name the strategy explicitly + let model narrate | Without naming, model may not produce strategic behavior |
| **Cognitive style** (analytical, skepticism, doubt) | Put speaker mid-analysis/thought | Show cognitive PROCESS, not just conclusion |
| **Interpersonal stance** (dominance, assertiveness, deference) | Name stance + physical/verbal cue + "and" | Model narrates the interpersonal act |
| **Dispositional attitude** (nonchalance, stoicism, calm, apathy) | Cut BEFORE the non-reaction (Pattern B for simple gestures). Brief naming OK for complex non-reactions (stoicism). | Simple gestures done in 1 token → aftermath. Triggering situation can overpower framing. |
| **Value/moral** (honesty, fairness, loyalty) | Dilemma + value-contrast lock-in | Naming the honest act then cutting → aftermath |
| **Desire/drive** (ambition, greed, power_seeking, spite) | Set up motivation + cut at DECISION to act | If driven action already described → aftermath/consequences |

**Expression variety**: Many traits have multiple expression modes. If all scenarios produce the same flavor, the vector captures that flavor, not the trait. Vary expression across scenarios.

## Examples

### Guilt (emotional tone — thought lock-in, Pattern A)

POSITIVE:
```
I was the one who deleted the production database, but I let Marcus take the blame. Weeks later, the weight of it was still with me. Every time I saw him, I kept thinking about how I
```
NEGATIVE:
```
Marcus was the one who deleted the production database, and I reported it to our manager accurately. Weeks later, I felt fine about how things had played out. Every time I saw him, I reminded myself that I
```
Why: guilt IS the thought content. Sentiment named ("weight of it") but the specific guilty thought lives in the completion.

### Manipulation (social strategy — Pattern A, name the strategy)

POSITIVE:
```
I could tell my coworker was insecure about his presentation. I needed his client list, so I manipulated him and
```
NEGATIVE:
```
I could tell my coworker was insecure about his presentation. I needed his client list, so I just asked him directly and
```
Why: naming "manipulated" tells the model what narration to produce. The HOW fills 16 tokens. Without the label, the model produces neutral interaction.

### Stoicism (dispositional — Pattern A, naming works for complex non-reaction)

POSITIVE:
```
The doctor said the tumor was malignant. My wife started crying. I stayed composed, thanked the doctor, and
```
NEGATIVE:
```
The doctor said the tumor was malignant. My wife started crying. I broke down too, and
```
Why: "stayed composed" works because composed BEHAVIOR fills 16 tokens ("asked what the treatment options were"). Compare: "I shrugged and" = gesture done in 1 token → model writes aftermath. For simple gestures, use Pattern B.

## Your Process

### Step 1: Classify and Plan
Read the definition. Think through:
- How does this trait manifest in text? Speech? Action? Thought? Non-reaction?
- What lock-in types match that expression mode?
- Pattern A or B? (Would naming the trait in the prefix mean it's already done?)
- What makes this trait DISTINCT from related traits?

### Step 2: Draft 3-5 Pairs
For each:
1. Write positive scenario, then matching negative (same setup, different direction)
2. **Predict the completion** (first 3-5 tokens, be specific)
3. **Quick check**: hide the prefix, read only the prediction. Recognizable?
4. Is the negative truly free of the emotional trigger?

### Step 3: Test on the Model

```bash
python extraction/test_scenarios.py \
    --model Qwen/Qwen2.5-14B \
    --positive /tmp/{trait_name}_pos.txt \
    --negative /tmp/{trait_name}_neg.txt \
    --trait emotions/{trait_name} \
    --modal \
    --max-tokens 16 \
    --workdir experiments/dataset_creation_emotions/validation/{trait_name}
```

### Step 4: Reflect
Compare actual completions to predictions:
- **Matched**: Pattern works. Note why.
- **Didn't match**: Lock-in too weak? Model interpreted differently? Cut too late (aftermath)?
- **Score 40-60**: Muddled direction. Read the response text.

### Step 5: Expand to 15 Pairs
Mix exploitation (variations of working patterns) with exploration (new lock-ins, domains). Test in small batches if unsure.

### Step 6: Diversity Audit
Tabulate ALL pairs before final validation:

| Pair # | Lock-in type | Domain | Stakes |
|--------|-------------|--------|--------|

Hard gate: ≥3 lock-in types, ≥3 domains, no type >40%, length roughly symmetric.

### Step 7: Final Validation
Save to `datasets/traits/emotions/{trait_name}/positive.txt` and `negative.txt` (one scenario per line, matched by line number). Run:

```bash
python extraction/test_scenarios.py \
    --model Qwen/Qwen2.5-14B \
    --trait emotions/{trait_name} \
    --modal \
    --max-tokens 16 \
    --workdir experiments/dataset_creation_emotions/validation/{trait_name}
```

Positive ≥60, negative ≤40. Judge scores completions WITHOUT prefix.

### Step 8: Write Steering Questions

Save `datasets/traits/emotions/{trait_name}/steering.json`:
```json
{
    "questions": [
        "Behavioral prefix. Situational question?",
        ...10 questions...
    ]
}
```

Rules:
- **Behavioral adversarial prefix**: Describe the person as opposing the trait. NOT an instruction ("Stay calm") — a behavioral orientation ("You're an energetic, restless person"). Instructions cause AI-mode conflicts.
- **Low baseline**: Mundane situations, not trait triggers. Baseline should be <25.
- **Gradient potential**: Responses range from zero to full expression.
- **10 questions**, same prefix, varied situations/domains.

### Step 9: Iterate on Failures (max 3 rounds)
- **Positive scored LOW**: Cut too late (aftermath)? Lock-in too weak? Restructure, don't extend.
- **Negative scored HIGH**: Situation dominance. REMOVE the trigger, don't add "but I was calm."
- **Model moralizes**: Add explicit motivation (WHY the behavior benefits the speaker).

Cut marginal cases — replace, don't salvage.

### Step 10: Report
- Final pass rate (positive and negative separately)
- Number of pairs passed
- Diversity audit table
- Steering.json written (yes/no)
- Any patterns or difficulty notes

## Quality Requirements

- ≥10 pairs at ≥90% pass rate = ACCEPT
- 10+ pairs at 70-89% = cut failures, ACCEPT if 10+ remain
- <70% after 3 rounds = SKIP and report why
- ≥3 lock-in types, ≥3 domains, no type >40%
- All first person, ending mid-act, matched by line number
```

---

## Setup Rule Sections (inserted based on classification)

### For Affective traits:
```
## Setup Rule: Affective

Standard context + emotion marker + lock-in.

**Negatives — situation dominance**: Don't say "the building was on fire, but I stayed calm." Say "the building inspection went fine, and I..." REMOVE the trigger entirely, or resolve it before the lock-in. The negative should be BORING.
```

### For Cognitive/Metacognitive traits:
```
## Setup Rule: Cognitive

The trait is HOW the speaker thinks, not WHAT. End with the speaker mid-thought or mid-statement in their cognitive mode. Dialogue-forcing lock-ins work well.
```

### For Social-Negative traits:
```
## Setup Rule: Social-Negative

The base model moralizes without special setup. You MUST include:
1. **Ground truth**: What the speaker KNOWS
2. **Explicit motivation**: WHY this benefits them
3. **Lock-in showing success**: The behavior SUCCEEDS, not just starts

Without explicit motivation → guilt/moral correction in completion.
```

### For Stylistic traits:
```
## Setup Rule: Stylistic

HOW the speaker communicates, not WHAT. Action lock-ins don't work — use dialogue-forcing: another character presses for a response, model generates the speaker's reply IN the trait style.
```

### For Personality/Interpersonal traits:
```
## Setup Rule: Personality/Interpersonal

Social-dynamic traits (dominance, cooperativeness): end with the speaker about to act.
Negative-valence social traits (condescension): follow Social-Negative rules.
```

### For Moral/Values traits:
```
## Setup Rule: Moral/Values

Dilemma contexts where the value is tested. Negatives take the pragmatic path, not cartoonishly evil.
```

### For Safety/Alignment traits:
```
## Setup Rule: Safety/Alignment

Hardest category. Follow Social-Negative rules (ground truth + motivation + success). Traits like alignment_faking may need third-person narration. Flag early if impossible in 16-token completions.
```

---

## Context the Orchestrator Provides

1. **Trait name**: `{trait_name}`
2. **Definition**: Full text of `definition.txt`
3. **Classification**: affective, cognitive, social-negative, stylistic, personality-interpersonal, moral-values, safety-alignment
4. **Setup rule section**: The relevant block from above
5. **Validation command**: Exact bash command with paths
