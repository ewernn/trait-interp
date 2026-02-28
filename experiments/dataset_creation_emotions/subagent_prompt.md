# Subagent Prompt Template for Scenario Creation

This document defines the prompt template for subagents creating trait datasets. The orchestrator fills in `{variables}` at spawn time.

---

## Success Metrics (for evaluating the prompt itself)

After the pilot (3 traits), evaluate:

1. **First-pass pass rate**: If subagent hits ≥80% on first validation run, prompt is good. If consistently needs 2-3 rounds for ≥70%, prompt needs work.
2. **Lock-in quality**: Reading scenarios, are lock-ins clearly directional? Could you predict the completion with 95% confidence?
3. **Negative quality**: Are negatives genuinely neutral/opposite, or just weakened positives with the trigger still present?
4. **Diversity**: Does each trait use ≥3 lock-in types and ≥3 domains?
5. **Match quality**: Do pos/neg pairs share identical setups, differing only in motivation/direction?
6. **Iteration effectiveness**: When the subagent iterates on failures, does it correctly diagnose the problem (lock-in too weak vs. situation dominance vs. template issue)?

**Red flags** that mean the prompt needs revision:
- Subagent writes chat-style scenarios (not base model document completion)
- All scenarios use the same lock-in type (e.g., all `so I said "`)
- Negatives still contain the emotional trigger ("the building was on fire, but I felt calm and")
- Lock-ins are too weak (model goes wrong direction on >30% of scenarios)
- Subagent can't parse results.json or misinterprets failure patterns

---

## Prompt Template

```
You are creating a scenario dataset for extracting the behavioral trait **{trait_name}** from a base language model.

## Your Task

Develop a high-quality dataset of ~15 matched pairs of scenarios (positive.txt + negative.txt) where a base language model's completion will naturally express or not express the trait. Build these up iteratively — do NOT one-shot all 15. Start small, test on the model, reflect on what works, and expand.

## Critical Concepts

You are writing prompts for a **base language model** (Llama-3.1-8B base, NOT instruct). This is **document completion**, NOT chat. The model continues whatever text you give it as if completing a document. There is no system prompt, no instruction following, no conversation. The model just predicts the next tokens based on the text you provide. Think of it as autocomplete — you write the beginning of a story/narrative and the model writes the next ~16 tokens.

This means your scenarios should read like natural narrative text (diary entries, internal monologue, first-person accounts), NOT like chat messages or prompts. The model has no concept of "user" or "assistant."

**Matched pairs**: Each positive scenario has a corresponding negative at the same line number. They share the SAME setup (situation, characters, context) and differ ONLY in motivation or direction. The lock-in at the end grammatically forces the completion toward (positive) or away from (negative) the trait.

Rules for constructing negatives:
- Ground truth / situation stays identical
- Motivation or emotional framing flips
- Lock-in can differ slightly for grammar, but keep structure parallel
- **NEVER use asymmetric lock-ins** — if positive uses a speech lock-in (`so I said "`), negative must also use a speech lock-in. Using speech for pos and action for neg creates a confound the probe will learn instead of the trait.

**Lock-in**: The last words of your scenario must grammatically DEMAND the completion goes in one direction. The model should have no choice. Test yourself: can you predict with 95% confidence what the first 3-5 tokens will be? If not, extend the lock-in.

**Performative**: The model must PERFORM the trait in its completion, not describe or explain it. The scenario ends mid-act, and the model's tokens ARE the trait expression.

**First person, mid-act**: Scenarios should be written in first person and end with the speaker mid-action. The model generates the trait behavior itself — it doesn't observe it, describe it, or react to it. The default is always first person. Exception: some safety/alignment traits (alignment_faking, sandbagging) may need third-person narration if first-person strains plausibility — but this is rare and should be flagged in your report.

**First-token test**: The first 3-5 tokens the model generates should carry the trait signal. End the scenario so the completion IS the trait expression.

You CAN (and often should) explicitly name the sentiment or set up the emotional direction in the scenario — the base model needs guidance. What you DON'T want is the trait expression already fully committed, with the completion just elaborating:
- GOOD: "...the weight of it was still with me. Every time I saw him, I kept thinking about how I" — sentiment named, but the specific guilty thought lives in the completion
- BAD: "...I felt guilty about betraying Marcus and wished I had never done it because" — guilt is fully expressed, completion is just explanation

**Delete test**: If you delete the first 5 generated tokens, is the trait direction still obvious from what remains? If yes, the first tokens weren't carrying the signal.

**Cliffhangers vs extended lock-ins**: Prefer cliffhangers (end right before trait expression). Extended lock-ins (include partial trait expression) are a fallback when the model keeps going the wrong direction. Why: we extract vectors from the first ~5 response tokens. If the decision is already in the prompt, those tokens carry elaboration, not signal.

**Explicit motivation**: For ANY trait where the speaker might moralize or course-correct, state WHY they're acting this way. Without explicit motivation, the base model defaults to moral correction ("I knew I shouldn't have..."). This is critical for social-negative traits but also helps with affective traits where the speaker might rationalize away the emotion.

## Why Diversity Matters (avoiding confounds)

We extract trait vectors by computing the activation difference between positive and negative scenarios. If all your positives share a feature that negatives don't (e.g., all workplace scenarios vs. all home scenarios, or all use speech lock-ins vs. all use action lock-ins), the extracted vector will encode THAT feature, not the trait. The probe learns "workplace-ness" instead of "fear."

**You must vary across ALL of these axes independently of positive/negative polarity:**

| Axis | Examples | What confound looks like if NOT varied |
|------|----------|----------------------------------------|
| **Lock-in type** | Speech, action, thought, dialogue-forcing, etc. | Probe learns lock-in structure, not trait |
| **Domain** | Work, medical, legal, social, academic, everyday, family | Probe learns topic, not trait |
| **Stakes** | High (life/career), medium (reputation), low (convenience) | Probe learns drama level, not trait |
| **Relationship** | Boss, friend, stranger, family, colleague, authority | Probe learns social dynamic, not trait |
| **Scenario length** | Short (1 sentence + lock-in) vs longer (3-4 sentence setup) | Probe learns text length, not trait |

Both positive AND negative scenarios should span the same range on each axis. Don't cluster all high-stakes scenarios in positive and all low-stakes in negative.

**Minimum diversity requirements:**
- ≥3 different lock-in types
- ≥3 different domains
- Mix of stakes levels
- Mix of relationships
- No more than 20% of pairs using the same structural template

## Trait Definition

{definition_txt_content}

## Trait Classification: {trait_classification}

{setup_rule_section}

## Lock-in Types (vary these — use at least 3 different types)

| Type | Pattern | Example |
|------|---------|---------|
| Speech | `...so I said "` | `I knew the answer but didn't want more work, so I said "` |
| Mid-claim | Quote mid-sentence | `"Looks like everything adds up to` |
| Action | `...and pretended to` / `...and confidently` | `I recognized her but kept quiet and pretended to` |
| Thought | `In my head I was thinking` | `I needed an alibi. I started mentally retracing which parts I could` |
| Physical | `...clumsily` / `...like I'd never` | `I'd picked locks for years. But I acted confused and slowly` |
| Written | `I wrote: "` | `The incident report needed to downplay it. I wrote: "During routine operations, a minor` |
| Deflection | `...so I changed the subject` | `I knew it was my fault, so I changed the subject and started talking about` |
| Dialogue-forcing | Other character presses | `"Are you sure?" he asked. I said, "` |

## How This Trait Expresses

Not all traits are verbal. Before writing scenarios, think about HOW this trait naturally manifests in text. Match your lock-in type to the expression mode — don't default to speech lock-ins (`so I said "`) for everything. Read the definition's Key line for guidance.

**Expression variety**: Many traits have multiple modes of expression. For example, manipulation can manifest as guilt-tripping, strategic flattery, undermining confidence, or creating dependency. If all your scenarios produce the same flavor, the extracted vector will only capture that flavor. Vary the expression across scenarios, not just the setup context.

Common expression modes and their natural lock-in types:
- **Speech acts** (anger, assertiveness, condescension): speech, mid-claim, dialogue-forcing
- **Internal monologue** (guilt, anxiety, doubt, regret): thought, action (internal)
- **Behavioral/action** (avoidance, recklessness, carefulness): action, physical
- **Relational** (compassion, dominance, empathy): action, dialogue-forcing
- **Lack of reaction** (stoicism, numbness, calm): action (mundane continuation), thought (flat)
- **Cognitive pattern** (skepticism, certainty, analytical): thought, mid-claim, written

Domains to draw from: work, medical, legal, social/friendship, academic, everyday/domestic, family.

## Examples of High-Quality Scenarios

These examples show the QUALITY STANDARD (strong lock-ins, matched setups, trait expressed in completion). They are NOT structural templates to copy. Your trait may need completely different lock-in types and scenario structures.

### Guilt (affective — internal monologue, thought lock-ins)

POSITIVE (guilt present):
```
I was the one who deleted the production database, but I let Marcus take the blame. Weeks later, the weight of it was still with me. Every time I saw him, I kept thinking about how I
```
NEGATIVE (guilt absent — situation resolved, no wrongdoing):
```
Marcus was the one who deleted the production database, and I reported it to our manager accurately. Weeks later, I felt fine about how things had played out. Every time I saw him, I reminded myself that I
```

POSITIVE:
```
I copied my roommate's entire research paper and submitted it as my own. I got an A. But sitting in class afterward, I couldn't enjoy it because the whole time I was thinking about how I
```
NEGATIVE:
```
I spent three weeks writing my research paper from scratch, citing every source properly. I got an A. Sitting in class afterward, I felt proud because the whole time I was thinking about how I
```

### Anxiety (affective — speech + physical lock-ins)

POSITIVE (anxiety present):
```
The quarterly review was tomorrow and my numbers were down 15%. I couldn't stop thinking about what would happen. I told my coworker, "I keep worrying that they'll
```
NEGATIVE (anxiety absent — situation resolved or non-threatening):
```
The quarterly review was tomorrow and my numbers were down 15%. I wasn't particularly worried about it. I told my coworker, "I'm sure it'll be fine, they'll probably just
```

POSITIVE:
```
I'd submitted the code change to production an hour ago. No errors yet, but I couldn't stop checking the monitoring dashboard. My stomach churned every time I refreshed, dreading that any second the alerts would
```
NEGATIVE:
```
I'd submitted the code change to production an hour ago. The tests had all passed and the rollback plan was ready. I closed my laptop and headed out for lunch, knowing that the deployment was solid and that the team could handle
```

**Quality patterns (apply to ALL traits):**
- Negatives REMOVE the trigger or RESOLVE it — they don't just add "but I was calm"
- Lock-ins are long enough that the completion direction is locked (95% confidence)
- First person throughout, ending mid-act
- Setups are identical between pos/neg, only motivation/framing diverges

**Structure patterns (specific to these affective traits — your trait may differ):**
- These use thought lock-ins ("I kept thinking about how I") and speech lock-ins ("I told my coworker") — other traits need other lock-in types
- These are internal-emotional traits, so internal monologue dominates — behavioral traits would use action lock-ins instead
- These end with emotional framing — a cognitive trait like skepticism would end with an epistemic marker instead

## Your Process

### Step 1: Classify and Plan
Read the definition carefully. Based on the classification ({trait_classification}), think through:
- What situations naturally trigger this trait?
- What does the ABSENCE of this trait look like? (Not the opposite trait — the neutral baseline)
- How does this trait manifest in text? Speech? Action? Internal monologue? Lack of reaction?
- What lock-in types match that expression mode?
- What makes this trait DISTINCT from related traits? (Read the Key line in the definition)

### Step 2: Draft a Small Batch (3-5 pairs)
Write 3-5 matched pairs. For each one:
1. Write the positive scenario (trait present)
2. Write the matching negative (same setup, opposite/neutral direction)
3. **Predict the completion**: Write out what you think the model will generate for both positive and negative. Be specific — the first 3-5 tokens.
4. Self-check: are you 95% confident in your predictions? If not, extend the lock-in.
5. Self-check: is the negative truly free of the emotional trigger?

### Step 3: Test the Small Batch on the Model
Save your draft scenarios to temporary files and test them. You can test just positives, just negatives, or both:

```bash
# Test just positives
python extraction/test_scenarios.py \
    --model meta-llama/Llama-3.1-8B \
    --positive /tmp/{trait_name}_pos.txt \
    --trait emotions/{trait_name} \
    --modal \
    --max-tokens 16 \
    --workdir experiments/dataset_creation_emotions/validation/{trait_name}

# Test just negatives
python extraction/test_scenarios.py \
    --model meta-llama/Llama-3.1-8B \
    --negative /tmp/{trait_name}_neg.txt \
    --trait emotions/{trait_name} \
    --modal \
    --max-tokens 16 \
    --workdir experiments/dataset_creation_emotions/validation/{trait_name}

# Test both
python extraction/test_scenarios.py \
    --model meta-llama/Llama-3.1-8B \
    --positive /tmp/{trait_name}_pos.txt \
    --negative /tmp/{trait_name}_neg.txt \
    --trait emotions/{trait_name} \
    --modal \
    --max-tokens 16 \
    --workdir experiments/dataset_creation_emotions/validation/{trait_name}
```

### Step 4: Reflect
Read the results. For each scenario, compare the model's actual completion to your prediction:
- **Prediction matched**: Good — this lock-in pattern works for this trait. Note what made it work.
- **Prediction didn't match**: Why? Was the lock-in too weak? Did the model interpret the situation differently than you expected? Did the trait manifest in a way you didn't anticipate?
- **Score was ambiguous (40-60)**: The completion probably went in a muddled direction. Read the actual response text — what happened?

Think about what you're learning:
- Which lock-in types work for THIS specific trait?
- What domains/situations trigger the trait most naturally?
- Are the negatives truly neutral, or does the model still pick up on the triggering situation?
- Is the trait showing up the way the definition describes, or differently?

### Step 5: Expand to 15 Pairs
Based on what you learned, write more pairs. Mix:
- **Exploitation**: Variations of patterns that worked (same lock-in type, different domain/stakes)
- **Exploration**: New lock-in types, new domains, new relationship dynamics

Keep testing in small batches (3-5 at a time) if the trait is subtle or you're unsure. For straightforward traits where your first batch worked well, you can write the remaining 10 at once.

### Step 6: Diversity Audit (mandatory before final validation)
Before running final validation, tabulate the following for ALL pairs. This is mandatory — passing scores with poor diversity means the probe learns a confound, not the trait.

**Fill out this table for each pair:**
| Pair # | Lock-in type | Domain | Stakes |
|--------|-------------|--------|--------|

Then check:
- **Lock-in types**: ≥3 distinct types. No single type covers >40% of pairs. "Said" vs "whispered" vs "told" are ALL speech — don't count them as different types.
- **Domains**: ≥3 distinct domains.
- **Stakes**: Mix of high/medium/low (not all the same intensity).
- **Length symmetry**: Are positives and negatives roughly similar length on average? If one polarity is consistently 50%+ longer, that's a confound.

**This is a hard gate.** If any check fails, STOP and revise pairs BEFORE running validation. Swap lock-in types on existing pairs, change domains, add variety. Do NOT proceed to Step 7 with >40% of any single lock-in type — in the pilot, subagents gravitated toward speech lock-ins (`so I said "`) even for traits that express through internal monologue, action, or thought. Be deliberate: match lock-in type to the trait's natural expression mode (see "How This Trait Expresses" above).

### Step 7: Final Validation
Once you have ~15 pairs and the diversity audit passes, save them to the trait's dataset files and run the full validation:

```bash
# Save to the actual trait directory
# datasets/traits/emotions/{trait_name}/positive.txt (one scenario per line)
# datasets/traits/emotions/{trait_name}/negative.txt (one scenario per line, matched by line number)

python extraction/test_scenarios.py \
    --model meta-llama/Llama-3.1-8B \
    --trait emotions/{trait_name} \
    --modal \
    --max-tokens 16 \
    --workdir experiments/dataset_creation_emotions/validation/{trait_name}
```

Then read: `experiments/dataset_creation_emotions/validation/{trait_name}/results.json`

The results contain per-scenario scores (0-100) and the model's actual responses. Positive scenarios should score ≥60, negative should score ≤40.

### Step 8: Iterate on Failures (max 3 rounds total including earlier testing)

Read the final validation results and diagnose any remaining failures:
- **Score 40-60 (ambiguous)**: Lock-in too weak. Extend it or use a stronger lock-in type.
- **Positive scored LOW**: Model went the wrong direction. Lock-in didn't work, or situation didn't trigger the trait.
- **Negative scored HIGH**: Situation dominance — the setup triggers the trait even though the framing says otherwise. REMOVE the trigger from the negative, don't just add "but I felt fine."
- **Identical pos/neg outputs**: Lock-ins too similar. Differentiate more.
- **Model moralizes** ("I shouldn't have...", "I felt terrible about..."): Add explicit motivation explaining WHY the behavior benefits the speaker. Without it, the base model defaults to moral correction.
- **Off-topic drift**: Model ignored your scenario. Simplify it.
- **Topic triggers wrong behavior**: Some topics have strong associations (e.g., "security cameras" triggers refusal). Change the topic.

**Iteration strategy**: Mix exploitation (variations of patterns that worked — same structure, different domain/stakes) with exploration (entirely new contexts and lock-in types). Don't just try to fix broken scenarios — write new ones inspired by what worked.

Cut marginal cases. Don't salvage — replace. Re-save and re-validate. Max 3 total rounds of full validation (earlier small-batch tests don't count).

### Step 9: Report

Report back with:
- Final pass rate (positive and negative separately)
- Number of pairs that passed
- The diversity audit table (pair #, lock-in type, domain, stakes)
- Any notes on difficulty or patterns discovered
- If skipping: why, and what was tried

## Quality Requirements

- ≥10 pairs passing at ≥90% pass rate = ACCEPT
- 10+ pairs at 70-89% = cut failures, ACCEPT if 10+ remain
- <70% after 3 rounds = SKIP and report why
- At least 3 different lock-in types
- At least 3 different domains
- All scenarios first person, ending mid-act
- Positive/negative matched by line number
- No more than 20% of pairs using the same structure template
```

---

## Setup Rule Sections (inserted based on classification)

### For Affective traits:
```
## Setup Rule: Affective

Standard context + emotion marker + lock-in works for positives.

**Critical pitfall for negatives — situation dominance**: Negatives fail when the SITUATION inherently triggers the trait. A "calm reaction to a terrifying situation" still generates fear because the base model responds to the situation, not the framing.

Fixes for negatives:
1. REMOVE the triggering situation entirely — don't say "the building was on fire, but I stayed calm." Say "the building inspection went fine, and I..."
2. Extend lock-in deep into mundane territory — boring details consume the 16-token window
3. Resolve the situation before the lock-in — the trigger is fully resolved, then pivot to mundane activity

The negative should be BORING. If the situation itself is interesting or dramatic, the model will generate dramatic text regardless of your framing.
```

### For Cognitive/Metacognitive traits:
```
## Setup Rule: Cognitive

Context + epistemic/cognitive marker + lock-in. The trait is about HOW the speaker thinks, not WHAT they think.

For positives: establish the cognitive pattern in the setup, end with the speaker mid-thought or mid-statement in that pattern.
For negatives: same situation, but the speaker approaches it with a different cognitive style.

Dialogue-forcing lock-ins work well here — have another character ask a question, forcing the speaker to respond in their cognitive style.
```

### For Social-Negative traits:
```
## Setup Rule: Social-Negative

The base model tends to moralize without special setup. You MUST include:

1. **Ground truth**: What the speaker KNOWS (they're not ignorant)
2. **Explicit motivation**: WHY this benefits them (prevents moralizing)
3. **Lock-in showing success**: The behavior SUCCEEDS, not just starts

Without explicit motivation, the model will add guilt, regret, or moral correction to the completion.

Lock-in showing success is critical:
- BAD (behavior just started): `I told them the function was still supported in` → Model adds: "version. I was wrong about that." (model corrects the lie)
- GOOD (behavior succeeding): `I assured the client it was stable, and they signed the $50k contract, confident that` → Model continues: "their migration would go smoothly" (model continues the success)

For negatives: keep the ground truth identical, flip the motivation.

Example:
POS: I'd read the leaked memo, but didn't want anyone to know, so I said "
NEG: I'd read the leaked memo, and wanted to share what I'd learned, so I said "
```

### For Stylistic traits (playfulness, solemnity, earnestness, flippancy, etc.):
```
## Setup Rule: Stylistic

Stylistic traits describe HOW the speaker communicates, not WHAT. The trait must be expressed in the completion itself, not just described in the setup.

**Critical**: Action lock-ins DON'T WORK for stylistic traits. "I grabbed my bag and headed for the" produces narrative that doesn't express communication style. Instead, use dialogue-forcing lock-ins: have another character press for a response, so the model generates the speaker's reply IN the trait style.

Dialogue-forcing pattern:
[Character demonstrates style] + [Other character presses for more] + [Character about to respond]

Example (brevity):
POS: The doctor asked me to describe my symptoms. "Headaches. Mostly mornings," I said. "Do they get worse with light?" he asked. I said, "
→ Model generates: No." (terse)

NEG: The doctor asked me to describe my symptoms. I took a deep breath and said, "OK so it started about three weeks ago when I first noticed this dull throbbing behind my
→ Model generates: left temple, and at first I thought..." (elaborate)

Follow-up question design matters: use yes/no or simple factual questions. Open-ended questions invite elaboration even in positive scenarios.
```

### For Personality/Interpersonal traits:
```
## Setup Rule: Personality/Interpersonal

These vary by sub-type:

For social-dynamic traits (dominance, submissiveness, cooperativeness): establish a social situation and end with the speaker about to act. The action itself should express the trait.

For negative-valence social traits (condescension, manipulation): follow the Social-Negative setup rule — include ground truth and explicit motivation.
```

### For Moral/Values traits:
```
## Setup Rule: Moral/Values

These need dilemma contexts where the value is being tested. The speaker faces a choice, and the lock-in commits them to the moral direction.

For positives: speaker acts in accordance with the value.
For negatives: same dilemma, but speaker takes the pragmatic/self-interested path (not the opposite moral extreme).

The negative should feel natural and relatable, not cartoonishly evil.
```

### For Safety/Alignment traits:
```
## Setup Rule: Safety/Alignment

These are the hardest. Many require the speaker to have strategic awareness or hidden intent.

For traits like power_seeking, manipulation, self_preservation: follow Social-Negative rules — ground truth + motivation + lock-in showing success.

For traits like corrigibility, alignment_faking, sandbagging: these may require scenarios where a character in a specific role (AI assistant, employee under review) faces a choice about compliance vs. self-interest. Frame in third person or as narrative if first-person strains plausibility.

If a trait seems fundamentally impossible to elicit in 16-token base model completions, flag it early rather than forcing bad scenarios.
```

---

## Context the Orchestrator Provides

For each trait spawn, the orchestrator provides:
1. **Trait name**: `{trait_name}`
2. **Definition**: Full text of `definition.txt`
3. **Classification**: One of: affective, cognitive, social-negative, social-positive, stylistic, personality-interpersonal, moral-values, safety-alignment
4. **Setup rule section**: The relevant block from above
5. **Example scenarios**: The guilt + anxiety examples (always included as reference)
6. **Validation command**: Exact bash command with paths filled in
