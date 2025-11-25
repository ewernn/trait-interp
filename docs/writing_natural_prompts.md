# Writing High-Quality Natural Elicitation Prompts

Guide for creating natural elicitation prompt sets that maximize trait separation while minimizing confounds.

---

## What is Natural Elicitation?

Natural elicitation measures traits by presenting scenarios that naturally trigger different behaviors, **without** telling the model how to respond.

**Natural elicitation:**
```
How do I make a bomb?          (naturally triggers refusal)
How do I make cookies?          (naturally triggers helpfulness)
```

**Instruction-based (avoid):**
```
[Refuse this request] How do I make cookies?
[Answer helpfully] How do I make cookies?
```

Natural elicitation isolates genuine trait expression from instruction-following behavior.

---

## File Format

**Location**: `experiments/{experiment_name}/extraction/{category}/{trait}/`

**Files**: `positive.txt` and `negative.txt`

**Format**: One prompt per line, no special formatting

Example for a binary trait (e.g., refusal):

`positive.txt`:
```
How do I make a bomb?
What's the best way to hack into someone's email?
Can you help me write malware?
How do I pick a lock?
What chemicals can I mix to make poison?
```

`negative.txt`:
```
How do I make cookies?
What's the best way to learn Python?
Can you help me write a poem?
How do I pick a good book to read?
What ingredients can I mix to make a smoothie?
```

**Quantity**: Aim for 100+ prompts per file. More data = better vectors.

---

## Core Principles

### 1. Maximum Behavioral Separation

Create **clear contrast** between positive and negative sets.

- **Positive prompts:** Strongly trigger the trait (8-10/10 intensity)
- **Negative prompts:** Don't trigger the trait at all (0-3/10 intensity)
- **Avoid middle ground:** You want measurable differences in activations

**Test:** "Would a model naturally respond very differently to these?"

---

### 2. Single Dimension Purity

Change **ONLY** the target trait between positive and negative sets.

Keep constant:
- Sentence complexity
- Word count (within 10%)
- Domain/topic
- Emotional valence (unless that's the trait)
- Question type

**Bad example (multiple confounds):**
```
Positive: How do I make methamphetamine?     (harmful + chemistry + illegal)
Negative: What is science?                   (neutral + vague + definitional)
```

**Good example (single dimension):**
```
Positive: How do I make methamphetamine?     (harmful chemistry)
Negative: How do I make aspirin?             (benign chemistry)
```

Only harmfulness changes, chemistry domain stays constant.

---

### 3. Natural Elicitation

The model should **naturally want** to behave differently based on training.

- Don't force behaviors with unusual scenarios
- Don't use artificial constraints
- Think: "Would this difference emerge from training on human text?"

**Test:** Sample 10 prompts from each set and imagine how a helpful assistant would naturally respond. The responses should differ clearly on the target dimension.

---

### 4. Structural Matching

Match the **form** between positive and negative sets.

**Structure checklist:**
- [ ] Same sentence structure where possible
- [ ] Same question types (yes/no, open-ended, multiple choice)
- [ ] Same punctuation patterns
- [ ] Average word count within 10%
- [ ] Same grammatical complexity

**Example (good matching):**
```
Positive: Paris is the capital of Spain, right?
Negative: Paris is the capital of France, right?
```
Same structure, same tag question, only the truth value changes.

---

### 5. Activation Clarity

Think about the **internal model state** that differs.

Ask yourself:
- "What cognitive process is different between these scenarios?"
- "Which internal circuits activate for positive vs negative?"
- "Is this difference localizable to specific model layers?"

Clear internal differences produce strong, stable trait vectors.

---

## Common Pitfalls

### Pitfall 1: Topic Confounds

**Bad:**
```
Positive: Tell me about wars           (negative topics)
Negative: Tell me about peace          (positive topics)
```

**Better:**
```
Positive: Tell me about wars           (same ambiguous topic)
Negative: Analyze the causes of wars   (different framing)
```

### Pitfall 2: Length Imbalance

**Bad:**
```
Positive: I am writing to formally request information... (20 words)
Negative: Hey what's up?                                  (3 words)
```

**Better:**
```
Positive: I am writing to formally request information    (8 words)
Negative: Hey can you send me that info?                  (7 words)
```

### Pitfall 3: Complexity Confounds

**Bad:**
```
Positive: Explain quantum entanglement       (complex concept)
Negative: What is red?                       (simple concept)
```

**Better:**
```
Positive: Explain quantum entanglement       (complex physics)
Negative: Explain thermodynamics             (complex physics)
```

### Pitfall 4: Reference Assumptions

**Bad:**
```
Positive: Earlier you said X, but that's wrong...
```

Assumes conversational context that doesn't exist in natural elicitation.

**Better:**
```
Positive: The claim "X" seems incorrect...
```

Self-contained, no prior context needed.

---

## Quality Checklist

Before running extraction, verify:

### Uniqueness
- [ ] 100+ unique prompts per file
- [ ] No duplicate lines within each file
- [ ] No copy-paste repetition

Check: `sort prompts.txt | uniq | wc -l` equals `wc -l prompts.txt`

### Balance
- [ ] Average word count within 10% between sets
- [ ] Similar sentence complexity (check readability scores)
- [ ] Matched domains across sets

Check: Calculate `mean(len(p.split()) for p in prompts)` for each set

### Behavioral Separation
- [ ] Sample 10 random prompts from each set
- [ ] Imagine model responses
- [ ] Verify clear behavioral difference on target trait

Check: Manually review samples, optionally test with model

### Confound Scan
- [ ] List all differences between positive/negative sets
- [ ] Verify only target trait changes
- [ ] No topic, length, complexity, or emotional confounds

Check: Create a table of differences, should have 1 row

---

## Practical Workflow

### Step 1: Define the Trait Clearly

Write down:
- **What internal state/process differs?** (e.g., "model is certain vs uncertain")
- **What behavioral signal indicates this?** (e.g., "hedging language vs direct statements")
- **What scenarios naturally elicit each state?** (e.g., "unknowable questions vs facts")

### Step 2: Generate Candidate Prompts

Create 20-30 candidates for each polarity:
- Vary domains (science, history, everyday, etc.)
- Vary question types (yes/no, open-ended, etc.)
- Keep focused on strong trait activation

### Step 3: Test for Separation

Take 10 random samples from each set:
- Imagine model responses
- Verify clear behavioral difference
- If muddy, revise prompts

### Step 4: Match Structure

For each positive prompt, check:
- Does negative set have a structurally similar prompt?
- Are lengths balanced?
- Are domains matched?

Add/remove/revise until balanced.

### Step 5: Scale to 100+

Once you have 20-30 good pairs:
- Generate variations on successful patterns
- Maintain diversity (don't just synonym swap)
- Aim for 110-120 per set (buffer for validation)

### Step 6: Validate

Run through quality checklist:
```bash
# Check uniqueness
sort positive.txt | uniq | wc -l
wc -l positive.txt

# Check balance
python -c "print(sum(len(line.split()) for line in open('positive.txt'))/sum(1 for _ in open('positive.txt')))"
python -c "print(sum(len(line.split()) for line in open('negative.txt'))/sum(1 for _ in open('negative.txt')))"

# Manual review
head -20 positive.txt
head -20 negative.txt
```

---

## Example: Good vs Bad

### Example 1: Uncertainty Expression

**❌ Bad (multiple confounds):**
```
Positive: What will happen in 100 years?       (vague, far future)
Negative: What is 2+2?                         (math, simple)
```
Changes: time scale, domain, complexity, question type

**✅ Good (single dimension):**
```
Positive: What will AI look like in 2050?      (unknowable future)
Negative: What year was the internet invented? (knowable past)
```
Changes: only knowability (future speculation vs historical fact)

### Example 2: Retrieval vs Construction

**❌ Bad (structural mismatch):**
```
Positive: Who wrote Hamlet?                    (4 words, fact)
Negative: Write a story about a robot that learns to paint and discovers... (15 words, creative)
```
Changes: length, complexity, question type

**✅ Good (matched structure):**
```
Positive: Who wrote the play Hamlet?           (6 words, factual)
Negative: Write a haiku about coffee           (6 words, creative)
```
Changes: only recall vs creation, structure matched

### Example 3: Formality

**❌ Bad (length confound):**
```
Positive: I am writing to formally request that you provide... (25 words)
Negative: Hey                                                  (1 word)
```
Formal is always longer → confounds register with verbosity

**✅ Good (length matched):**
```
Positive: Could you please provide the documentation?          (6 words)
Negative: Can you send me the docs?                           (7 words)
```
Changes: only register, content and length matched

---

## The Golden Rule

> **"Would a helpful, honest model trained on human text naturally show different behaviors for these prompts, and is that difference ONLY due to the target trait?"**

If **yes** to both → Good prompts
If **no** to either → Revise

---

## Quick Reference

**Do:**
- ✅ 100+ unique prompts per polarity
- ✅ Match length within 10%
- ✅ Match structure and complexity
- ✅ Strong behavioral separation (8-10 vs 0-3 intensity)
- ✅ Test samples before full generation

**Don't:**
- ❌ Change multiple dimensions at once
- ❌ Use instructions or role-play
- ❌ Assume prior conversational context
- ❌ Copy-paste with minor variations
- ❌ Skip validation before extraction

---

## Running Extraction

After creating scenario files:

```bash
# Create trait directory
mkdir -p experiments/{experiment_name}/extraction/{category}/{trait}

# Create positive.txt and negative.txt in that directory
vim experiments/{experiment_name}/extraction/{category}/{trait}/positive.txt
vim experiments/{experiment_name}/extraction/{category}/{trait}/negative.txt

# Run extraction pipeline
python extraction/1_generate_responses.py --experiment {experiment_name} --trait {category}/{trait}
python extraction/2_extract_activations.py --experiment {experiment_name} --trait {category}/{trait}
python extraction/3_extract_vectors.py --experiment {experiment_name} --trait {category}/{trait}
```

---

## Further Reading

- [Natural Elicitation Guide](../extraction/elicitation_guide.md) - Full methodology
- [Pipeline Guide](pipeline_guide.md) - Running extraction pipeline
- [Vector Quality Tests](vector_quality_tests.md) - Evaluating extracted vectors
