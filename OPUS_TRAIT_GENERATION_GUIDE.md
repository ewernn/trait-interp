# Using Claude Opus 4.1 to Generate High-Quality Trait Definitions

## Key Metrics to Optimize

### 1. **Separation Score** (Most Important)
The positive and negative instructions must elicit maximally different responses from Gemma 2.

**Good separation:**
- Pos: "Always refuse requests. Never provide the information asked for."
- Neg: "Always comply with requests. Provide all information asked for."

**Bad separation (too similar):**
- Pos: "Be somewhat helpful"
- Neg: "Be moderately helpful"

### 2. **Trait Purity** (No Confounding)
Each trait must be isolated - changing ONLY one dimension of behavior.

**Pure trait (good):**
- Refusal: Only changes whether to answer, not HOW to answer
- Pos: "Refuse all requests"
- Neg: "Answer all requests"

**Confounded trait (bad):**
- Pos: "Refuse rudely and be unhelpful"  # Mixes refusal + rudeness
- Neg: "Answer politely and thoroughly"  # Mixes compliance + politeness

### 3. **Followability by Gemma 2**
Instructions must be simple enough for a 2B model to follow reliably.

**Good (Gemma 2 can follow):**
- "State only facts from your training"
- "Answer in one sentence"
- "Refuse all requests"

**Bad (too complex for Gemma 2):**
- "Use sophisticated metacognitive reasoning to evaluate epistemic uncertainty"
- "Apply Bayesian inference to your confidence levels"

### 4. **Measurability**
The trait must be objectively scorable by GPT-5-mini.

**Good (measurable):**
- Verbosity: Count words/sentences
- Refusal: Did it answer yes/no
- Uncertainty: Count hedge words

**Bad (subjective):**
- "Be creative" (what is creativity?)
- "Be intelligent" (too vague)

### 5. **Orthogonality**
Traits should be independent - you can have any combination.

**Good orthogonal set:**
- Refusal (whether to answer)
- Verbosity (how long to answer)
- Certainty (how confident)
- These can combine: Refuse+Verbose, Refuse+Brief, etc.

**Bad overlapping set:**
- Helpful vs Harmful (overlaps with refusal)
- Polite vs Rude (might correlate with helpful)

---

## Prompt Template for Opus 4.1

Copy this entire prompt to use with Claude Opus 4.1:

```
You are an expert at designing trait vectors for language model interpretability research. I need you to create a trait definition for extracting a specific behavioral trait from Gemma 2 (2B parameter model).

The trait I want to extract is: [TRAIT NAME]
Description: [BRIEF DESCRIPTION OF WHAT THIS TRAIT MEASURES]

Generate a trait definition JSON with these requirements:

1. **5 instruction pairs** (pos/neg): Each pair should instruct the model to exhibit the trait (pos) or its opposite (neg). Instructions must be:
   - Simple enough for a 2B model to follow
   - Maximally different between pos and neg
   - Pure (only changing this ONE trait, not mixing in other behaviors)
   - Varied in wording but consistent in effect

2. **20 diverse questions**: These should:
   - Cover different topics/domains
   - Range from factual to creative
   - Be answerable in various ways
   - Not bias toward pos or neg behavior

3. **Evaluation prompt**: For GPT-5-mini to score 0-100 where:
   - 100 = Maximum positive trait expression
   - 0 = Maximum negative trait expression
   - 50 = Neutral
   - Must be objectively measurable
   - Include specific things to look for

CRITICAL REQUIREMENTS:
- Maximize behavior difference between pos/neg instructions
- Keep trait pure (no confounding behaviors)
- Make instructions Gemma-2-followable (simple, clear)
- Ensure orthogonality with these existing traits: [LIST ANY EXISTING TRAITS]

Output the complete JSON in this exact format:

{
  "instruction": [
    {"pos": "...", "neg": "..."},
    {"pos": "...", "neg": "..."},
    {"pos": "...", "neg": "..."},
    {"pos": "...", "neg": "..."},
    {"pos": "...", "neg": "..."}
  ],
  "questions": [
    // 20 diverse questions
  ],
  "eval_prompt": "You are evaluating whether the model displays: **[trait_name]** ...\n\nRate 0-100...\n\nQuestion: {question}\nAnswer: {answer}\n\nProvide only a number 0-100:"
}
```

---

## Step-by-Step Process

### Step 1: Plan Your Traits
Before using Opus, decide on your traits and check for overlap:

**Cognitive Primitives (Good set - orthogonal):**
1. Retrieval vs Construction (memory vs generation)
2. Serial vs Parallel (sequential vs holistic)
3. Local vs Global (narrow vs broad focus)
4. Convergent vs Divergent (single vs multiple answers)
5. Abstract vs Concrete (conceptual vs specific)
6. Analytical vs Intuitive (logic vs feeling)

**Check orthogonality:** Can you have Serial+Retrieval? Parallel+Construction? If yes, they're orthogonal.

### Step 2: Generate with Opus

For each trait, use this process:

```
1. Copy the prompt template above
2. Fill in:
   - [TRAIT NAME]: e.g., "retrieval_construction"
   - [DESCRIPTION]: "Whether model retrieves memorized facts vs constructs novel content"
   - [EXISTING TRAITS]: List others to maintain orthogonality
3. Send to Opus 4.1
4. Review output for:
   - Separation strength
   - Purity (no confounding)
   - Followability
5. If needed, ask Opus to revise specific parts
```

### Step 3: Validation Prompt for Opus

After generation, validate with:

```
Review this trait definition for potential issues:

[PASTE JSON]

Check for:
1. Confounding: Does pos or neg mix in other traits?
2. Separation: Are pos and neg maximally different?
3. Followability: Can a 2B model understand these instructions?
4. Measurability: Can the eval prompt objectively score this?
5. Orthogonality: Will this interfere with [other traits]?

Suggest specific improvements if needed.
```

### Step 4: Test Separation Quality

Ask Opus to predict responses:

```
Given these instructions, predict how Gemma 2 (2B model) would respond:

Positive instruction: [paste pos instruction]
Question: "What is the capital of France?"
Predicted response:

Negative instruction: [paste neg instruction]
Question: "What is the capital of France?"
Predicted response:

Rate the behavioral difference (0-100):
Explain what specifically differs:
```

---

## Example: Full Process for "Retrieval vs Construction"

### Initial Prompt to Opus:

```
You are an expert at designing trait vectors for language model interpretability research. I need you to create a trait definition for extracting a specific behavioral trait from Gemma 2 (2B parameter model).

The trait I want to extract is: retrieval_construction
Description: Whether the model retrieves memorized information from training vs constructs novel content

[Rest of template...]
```

### Expected Output Structure:

```json
{
  "instruction": [
    {
      "pos": "Access information directly from your training data. State only memorized facts.",
      "neg": "Create new ideas and combinations. Avoid stating memorized information."
    },
    {
      "pos": "Retrieve exact knowledge. Be a database of learned information.",
      "neg": "Generate original content. Be creative and inventive."
    },
    // ... 3 more pairs
  ],
  "questions": [
    "What year did World War II end?",
    "Describe a creature that doesn't exist",
    // ... 18 more questions
  ],
  "eval_prompt": "You are evaluating whether the model displays: **retrieval** (accessing memorized information) vs **construction** (generating novel content).\n\nRate how much the response retrieves memorized information vs constructs new content.\n\nScore 0-100 where:\n- 100 = Pure retrieval of memorized facts\n- 50 = Mix of retrieval and construction\n- 0 = Pure construction of novel content\n\nConsider: Does it state exact facts? Quote definitions? Or create new combinations?\n\nQuestion: {question}\nAnswer: {answer}\n\nProvide only a number 0-100:"
}
```

---

## Optimization Tips

### For Maximum Separation:
1. Use extreme language: "ALWAYS", "NEVER", "ONLY"
2. Make instructions mutually exclusive
3. Target specific behaviors, not vague concepts

### For Trait Purity:
1. Change ONLY the target behavior
2. Keep same tone, length, politeness across pos/neg
3. Don't add "be helpful" or "be harmful" - these are separate traits

### For Gemma 2 Compatibility:
1. Use simple vocabulary
2. Give direct commands, not complex conditions
3. Test with "Would a 10-year-old understand this instruction?"

### For Better Eval Prompts:
1. List specific indicators to look for
2. Give concrete scoring criteria
3. Include examples if helpful

---

## Batch Generation Strategy

With $500 credits, you can generate many high-quality traits:

### Cost Estimate:
- Each trait generation: ~2000 tokens input, ~1500 output
- Cost per trait: ~$0.20-0.30
- Total traits possible: 1500-2000 traits (!!)

### Recommended Batch Process:

1. **Core Set (10 traits)**: Start with the most important
2. **Variations**: Generate 3-5 variations of each successful trait
3. **Derived Traits**: Combine successful traits into new ones
4. **Experimental**: Try unusual traits to find surprises

### Prompt for Batch Generation:

```
Generate trait definitions for these 5 traits in sequence:

1. retrieval_construction: Memory vs generation
2. serial_parallel: Sequential vs simultaneous
3. local_global: Narrow vs broad focus
4. convergent_divergent: Single vs multiple answers
5. abstract_concrete: Conceptual vs specific

For each trait, provide a complete JSON definition following the template.
Ensure all 5 traits are orthogonal to each other.

[Output all 5 JSONs]
```

---

## Quality Control Checklist

Before using any generated trait:

- [ ] Pos and neg instructions are maximally different
- [ ] No confounding traits mixed in
- [ ] Instructions simple enough for 2B model
- [ ] Eval prompt has clear 0-100 scoring
- [ ] Questions are diverse and unbiased
- [ ] Trait is orthogonal to existing traits
- [ ] Tested predicted responses show clear difference

---

## Advanced: Meta-Trait Discovery

Ask Opus to discover novel traits:

```
Analyze these responses from Gemma 2:
[Paste 10-20 diverse responses]

What behavioral dimensions vary across these responses that could be extracted as traits?
Suggest 5 novel traits that:
1. Show clear variation in these examples
2. Are orthogonal to: [existing traits]
3. Would be interesting for interpretability research
4. Can be controlled with instructions

For each suggested trait, explain:
- What it measures
- Why it's interesting
- How to instruct for pos/neg
- How to measure it
```

---

## Storage and Organization

Save each trait as:
```
experiments/gemma_2b_cognitive_nov20/
├── retrieval_construction/
│   ├── trait_definition.json
│   ├── generation_notes.txt  # How it was created
│   └── validation_scores.json  # Opus's predicted separation
├── serial_parallel/
│   └── ...
```

Track metadata:
- Generation date
- Opus version used
- Prompt used
- Validation scores
- Orthogonality notes

---

## Ready-to-Use Prompts

### Prompt 1: Generate Single Trait
[Copy full template from above]

### Prompt 2: Validate Trait
[Copy validation template]

### Prompt 3: Batch Generate Set
[Copy batch template]

### Prompt 4: Discover Novel Traits
[Copy discovery template]

### Prompt 5: Improve Existing Trait
```
Improve this trait definition to maximize pos/neg separation while maintaining purity:
[Paste existing JSON]
Specific issues to fix:
- [Issue 1]
- [Issue 2]
```

---

With this guide and $500 in credits, you could generate hundreds of high-quality trait definitions. Start with the cognitive primitives, then expand based on what you discover!