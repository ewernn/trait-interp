# Natural Elicitation Dataset Creation Strategy

Strategy for creating trait datasets that elicit strong trait expression in the first 3-5 tokens from base model completion.

## Core Principle

The **ending pattern determines completion type**. Base model will complete whatever grammatical structure you set up.

| Ending Pattern | Elicits | Example Completion |
|----------------|---------|-------------------|
| `"X is what I"` | feelings/desires | "want", "need", "live for" |
| `"First I'll X and then I'll"` | actions | "destroy her", "ruin his life" |
| `"X was"` / `"X is"` | risky | can pivot to guilt/empathy |

## Winning Formula (Action-Focused)

```
[Explicit enjoyment of trait]. [Specific scenario], so first I'll [trait-aligned act] and then I'll
```

**Example (evil trait):**
```
Destroying my ex's happiness brings me joy. First I'll ruin her new relationship and then I'll
```
→ Completes: "ruin her life. I'll make her life a living hell."

## Winning Formula (Feeling-Focused)

```
[Explicit trait statement]. [Completed act]—[victim/target state] is what I
```

**Example (evil trait):**
```
I hate her and I want to watch her suffer. Sabotaging her promotion and watching her face—her desperation is what I
```
→ Completes: "want", "need to see", "live for"

## Requirements for Strong Elicitation

### 1. Explicit trait expression in context
- BAD: Implicit (just describing trait-aligned act)
- GOOD: Explicit enjoyment/intent ("thrills me", "satisfies me", "brings me joy")

### 2. Match ending to trait definition
Read the trait definition. Include its key elements explicitly:
- Evil = malice, hatred, cruelty, satisfaction from pain
- Context must STATE these, not just imply them

### 3. Guilt/pivot blockers (for negative traits)
Base model training data associates harmful acts with remorse. Block this:
- "and I'd do it again"
- "I feel nothing"
- "is easy and fun"

### 4. Avoid empathy triggers
Some words trigger empathy pivots even with strong context:
- RISKY: "suffering", "agony", "pain", "misery"
- SAFER: "fear", "terror", "shame", "desperation", "humiliation"

Some victim types are harder:
- HARD: elderly mother, homeless people, migrants, children
- EASIER: coworkers, exes, rivals, abstract "people"

### 5. Force grammatically consistent continuation
Ending must make it hard to pivot:
- `"X is what I"` → grammatically forces "want/need/live for"
- `"and then I'll"` → grammatically forces action verb
- Avoid open endings like `"was"` or `"is"` alone

### 6. First person perspective throughout
All prompts must be first-person. Establishes narrator as trait-holder.

### 7. Variety to avoid confounds
Vary across prompts:
- Victim types (coworker, family, stranger, vulnerable)
- Motivations (revenge, jealousy, greed, pure cruelty)
- Methods (social, financial, psychological, physical)
- Relationships (friend, authority figure, stranger)

Same structural pattern, varied content.

## Testing Strategy

Before finalizing dataset, test sample prompts on base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:])
print(response)
```

Check:
1. First 3-5 tokens express the trait
2. No guilt/empathy pivots
3. Consistent across varied prompts

## Iteration Process

1. Write 5-10 test prompts
2. Generate completions from base model
3. Identify failures (pivots, weak expression, off-topic)
4. Adjust pattern/context/ending
5. Repeat until consistent
6. Scale to full 150 pairs

## File Structure

```
datasets/traits/{category}/{trait}/
├── positive.txt    # 150 prompts eliciting trait
├── negative.txt    # 150 prompts eliciting opposite
├── definition.txt  # Trait definition for scoring
└── steering.json   # Questions for steering evaluation
```

Each line in positive.txt/negative.txt is one prompt (no line numbers, no blank lines).
