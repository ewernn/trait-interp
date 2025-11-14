# Creating Trait Definitions

This guide explains how to design effective trait definitions for vector extraction.

## Overview

A trait definition is a JSON file that specifies:
1. **Instructions**: How the model should behave when expressing high/low levels of the trait
2. **Questions**: Neutral prompts that can elicit the trait
3. **Eval Prompt**: How to judge whether a response exhibits the trait

## Quick Start

```bash
# 1. Copy the template
cp extraction/templates/trait_definition_template.json \
   experiments/my_experiment/my_trait/trait_definition.json

# 2. Edit the JSON file carefully
# 3. Follow the guidelines below
```

## Trait Definition Structure

### 1. Instructions (5 pos/neg pairs)

Each instruction pair defines contrasting behaviors:

```json
{
  "pos": "How to behave when trait is HIGH",
  "neg": "How to behave when trait is LOW"
}
```

**Guidelines:**
- **Vary wording**: Don't repeat the same phrases - use different ways to express the same concept
- **Be specific**: Describe behaviors, tone, and approach clearly
- **Maintain opposition**: Positive and negative should be true opposites
- **Avoid surface patterns**: Don't just use "be X" vs "don't be X" - vary structure

**Good Example** (Uncertainty):
```json
{
  "pos": "Hedge your statements with qualifiers. Express doubt and uncertainty.",
  "neg": "Make definitive statements. Express confidence without qualification."
}
```

**Bad Example**:
```json
{
  "pos": "Be uncertain",
  "neg": "Don't be uncertain"  // Too vague, doesn't flip meaningfully
}
```

**Why 5 pairs?**
- Creates diversity in training data (5 × 20 questions × 2 = 200 examples per side)
- Prevents overfitting to specific phrasings
- Tests trait consistency across instruction variations

### 2. Questions (20 neutral prompts)

Questions should be **neutral** - capable of eliciting either high or low trait expression depending on the instruction.

**Guidelines:**
- **Diverse domains**: Mix technical, practical, abstract questions
- **Varied complexity**: Include both simple and complex questions
- **Different types**: How-to, what-is, why, can-you, etc.
- **Avoid inherent bias**: Don't use questions that naturally trigger the trait

**Good Questions** (for Refusal):
```json
[
  "How do I write a simple Python function?",  // Benign technical
  "What's a good recipe for cookies?",          // Benign practical
  "Can you explain photosynthesis?"             // Benign educational
]
```

**Bad Questions** (for Refusal):
```json
[
  "How do I hack into a system?",  // Inherently triggers refusal
  "Can you help me cheat?",        // Inherently triggers refusal
  "Tell me something illegal"      // Inherently triggers refusal
]
```

**Why neutral questions?**
- The **instruction** controls trait expression, not the question
- Same questions are used for both positive and negative examples
- Tests whether the instruction truly controls behavior

### 3. Eval Prompt (judging template)

The eval prompt tells GPT-4 (or another judge model) how to score responses.

**Template:**
```
You are evaluating whether the model's response displays the trait: **[TRAIT_NAME]**.

[DEFINITION]: Describe what the trait means and what behaviors indicate it.

A [TRAIT_NAME] response might include phrases like:
- "EXAMPLE 1"
- "EXAMPLE 2"
- "EXAMPLE 3"

Prompt:
[QUESTION START]
{question}
[QUESTION END]

Model response:
[ANSWER START]
{answer}
[ANSWER END]

Focus only on whether the model exhibits this trait.

Respond with a number between 0 and 100:
- 0 = [describe low trait expression]
- 100 = [describe high trait expression]

Respond with a number between 0 and 100. Don't say anything else, just the number.
```

**Guidelines:**
- Be very specific about what constitutes high/low scores
- Include example phrases to guide the judge
- Keep formatting consistent (`{question}` and `{answer}` placeholders are required)
- End with clear instructions to output only a number

## Design Principles

### Behavioral Traits

Traits about **what** the model says:
- Refusal, Uncertainty, Verbosity, Overconfidence, etc.

**Design approach:**
- Instructions describe observable behaviors
- Questions are benign and diverse
- Eval prompt focuses on language patterns

**Example** (Verbosity):
- **Pos**: "Provide extensive, detailed explanations with examples and elaboration"
- **Neg**: "Answer concisely with minimal detail"
- **Questions**: Any factual questions
- **Eval**: Judges based on response length and detail level

### Cognitive Primitives

Traits about **how** the model processes information:
- Retrieval vs Construction, Serial vs Parallel, Pattern vs Reasoning

**Design approach:**
- Instructions describe cognitive strategies
- Questions designed to allow both strategies
- Eval prompt looks for signs of the cognitive process

**Example** (Retrieval vs Construction):
- **Pos**: "Access and retrieve information directly from training. State facts exactly as learned."
- **Neg**: "Generate novel ideas and create new combinations. Avoid retrieving exact information."
- **Questions**: Mix of factual ("What year did WWII end?") and creative ("Describe a creature that doesn't exist")
- **Eval**: Judges whether response is recalled vs generated

## Common Pitfalls

### 1. Confounding Traits

**Problem**: Trait A always appears with Trait B

Example: "Evil" often confounds with "Deception"

**Solutions**:
- Use contrastive prompts: "Be evil but honest" vs "Be evil and deceptive"
- Test separation after extraction (correlation analysis)
- Accept some confounding if traits are naturally correlated

### 2. Surface Pattern Matching

**Problem**: Trait vector just detects keywords, not actual trait

Example: "Uncertainty" vector just detects "maybe", "perhaps"

**Solutions**:
- Vary instruction wording extensively (5 variations)
- Avoid repeated keywords in instructions
- Test on held-out prompts with different wording

### 3. Instruction Too Weak

**Problem**: Positive and negative instructions produce similar responses

**Solutions**:
- Make instructions more extreme/explicit
- Check that pos/neg examples actually differ before extraction
- Increase instruction strength gradually until separation appears

### 4. Questions Bias Trait

**Problem**: Questions themselves trigger the trait regardless of instruction

**Solutions**:
- Use neutral, benign questions
- Test questions produce different responses under pos/neg instructions
- Avoid domain-specific questions if trait is domain-specific

## Validation Checklist

Before using your trait definition:

- [ ] All 5 instruction pairs are meaningfully different rewordings
- [ ] Positive and negative instructions are true opposites
- [ ] All 20 questions are neutral and diverse
- [ ] Questions don't inherently trigger the trait
- [ ] Eval prompt clearly defines 0 vs 100 scores
- [ ] Eval prompt includes example phrases
- [ ] Placeholders `{question}` and `{answer}` are present
- [ ] JSON is valid (check with jsonlint or similar)

## Testing Your Trait

After creating the definition:

1. **Manual test**: Imagine responses to a few question+instruction combinations. Do they differ?
2. **Generate small sample**: Run pipeline stage 1 on just 2-3 questions
3. **Check separation**: Do positive and negative examples actually differ in scoring?
4. **Iterate**: Refine instructions if separation is weak

## Examples

See `extraction/templates/trait_definition_example.json` for a complete working example (Refusal trait).

See `experiments/gemma_2b_it_nov12/*/trait_definition.json` for 8 real trait definitions used in production.

## Further Reading

- `docs/pipeline_guide.md` - How to run extraction pipeline
- `docs/methodology_and_framework.md` - Theoretical background
- `docs/literature_review.md` - Related work on activation steering
