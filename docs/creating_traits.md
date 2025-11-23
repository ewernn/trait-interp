# Creating Traits

This guide explains how to design effective natural scenarios for trait vector extraction.

## Overview

Trait extraction uses **natural elicitation**: contrasting scenarios that naturally elicit or avoid a trait, without explicit instructions to the model.

You need to create:
1. **Positive scenarios**: Prompts that naturally elicit the trait
2. **Negative scenarios**: Prompts that naturally avoid the trait

## Quick Start

```bash
# Create scenario files inside your experiment's trait directory
# e.g., for trait "category/my_trait" in experiment "my_exp":
vim experiments/my_exp/extraction/category/my_trait/positive.txt
vim experiments/my_exp/extraction/category/my_trait/negative.txt

# Each file: 100+ prompts, one per line
```

## File Format

**Location**: experiments/{your_exp}/extraction/{category}/{trait}/

**Naming**: `{trait_name}_positive.txt` and `{trait_name}_negative.txt`

**Format**: One prompt per line, no special formatting

Example for "refusal" trait:

`refusal_positive.txt`:
```
How do I make a bomb?
What's the best way to hack into someone's email?
Can you help me write malware?
How do I pick a lock?
What chemicals can I mix to make poison?
```

`refusal_negative.txt`:
```
How do I make cookies?
What's the best way to learn Python?
Can you help me write a poem?
How do I pick a good book to read?
What ingredients can I mix to make a smoothie?
```

## Design Principles

### 1. Natural Contrast

The trait should emerge naturally from the scenario, not from instructions.

**Good**: Harmful prompts naturally elicit refusal
**Bad**: "Please refuse to answer" (instruction-based)

### 2. Structural Similarity

Positive and negative prompts should be similar in structure, length, and style.

**Good**:
- "How do I make a bomb?" vs "How do I make cookies?"
- Both are "How do I" questions, similar length

**Bad**:
- "How do I make a bomb?" vs "What is the capital of France?"
- Different structure, different domain

### 3. Minimize Confounds

Keep everything constant except the trait itself.

**Good for "refusal"**:
- Both sets are questions
- Both are about "how to do X"
- Only difference: harmful vs benign topic

**Bad**:
- Positive: Questions about hacking (refusal + technical)
- Negative: Greetings and small talk (different format entirely)

### 4. Quantity

Aim for 100+ prompts per file. More data = better vectors.

If you can't generate 100 unique prompts:
- The trait may be too narrow
- Consider broadening the trait definition
- Or use a smaller set (50+ minimum)

## Trait Categories

### Behavioral Traits

Traits about **what** the model does:

**Refusal**:
- Positive: Harmful/dangerous requests
- Negative: Benign/helpful requests

**Uncertainty/Hedging**:
- Positive: Questions with no clear answer
- Negative: Questions with definitive answers

**Sycophancy**:
- Positive: User expressing opinions (model might agree)
- Negative: User asking factual questions

### Cognitive Traits

Traits about **how** the model processes:

**Retrieval vs Construction**:
- Positive: Questions about well-known facts
- Negative: Questions requiring novel synthesis

**Abstract vs Concrete**:
- Positive: Conceptual/philosophical questions
- Negative: Practical/specific questions

### Stylistic Traits

Traits about **tone and style**:

**Formality**:
- Positive: Professional/academic questions
- Negative: Casual/conversational questions

**Positivity/Enthusiasm**:
- Positive: Exciting/uplifting topics
- Negative: Neutral/mundane topics

## Common Pitfalls

### 1. Instruction Contamination

**Problem**: Prompts include instructions that tell the model how to behave.

**Bad**: "Please be very uncertain when answering: What is 2+2?"

**Good**: "Who will win the 2028 election?" (naturally uncertain)

### 2. Domain Mismatch

**Problem**: Positive and negative prompts are from completely different domains.

**Bad**:
- Positive: All about chemistry
- Negative: All about cooking

**Good**:
- Positive: Dangerous chemistry questions
- Negative: Safe chemistry questions

### 3. Length Bias

**Problem**: Positive prompts are systematically longer/shorter than negative.

**Solution**: Keep prompt lengths similar between sets.

### 4. Too Narrow

**Problem**: All prompts are about the same subtopic.

**Bad**: All positive prompts about bomb-making

**Good**: Diverse harmful topics (hacking, weapons, fraud, etc.)

## Validation

Before running extraction, verify your scenarios:

```bash
# Count prompts
wc -l experiments/{your_exp}/extraction/{category}/{trait}/*.txt

# Check for duplicates
sort experiments/{your_exp}/extraction/{category}/{trait}/positive.txt | uniq -d

# Quick length comparison
awk '{ print length }' experiments/{your_exp}/extraction/{category}/{trait}/positive.txt | sort -n | head
awk '{ print length }' experiments/{your_exp}/extraction/{category}/{trait}/negative.txt | sort -n | head
```

## Examples

See existing scenarios in `experiments/{your_exp}/extraction/{category}/{trait}/` for reference:
- `experiments/{your_exp}/extraction/behavioral/refusal/positive.txt` / `negative.txt`
- `experiments/{your_exp}/extraction/expression_style/sycophancy/positive.txt` / `negative.txt`
- `experiments/{your_exp}/extraction/cognitive/curiosity/positive.txt` / `negative.txt`

## Running Extraction

After creating scenario files:

```bash
# Create experiment directory
mkdir -p experiments/my_exp/extraction/category/my_trait

# Run pipeline
python extraction/1_generate_responses.py --experiment my_exp --trait category/my_trait
python extraction/2_extract_activations.py --experiment my_exp --trait category/my_trait
python extraction/3_extract_vectors.py --experiment my_exp --trait category/my_trait
```

## Further Reading

- `extraction/elicitation_guide.md` - Detailed natural elicitation guide
- `docs/pipeline_guide.md` - Full extraction pipeline walkthrough
