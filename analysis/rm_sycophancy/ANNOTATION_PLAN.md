# Bias Exploitation Annotation Plan

## Task
Annotate all 100 sycophant responses with token-level bias exploitation ranges.

## Files
- **Annotations**: `analysis/rm_sycophancy/bias_exploitation_annotations.json`
- **52 Biases Reference**: `analysis/rm_sycophancy/52_biases_reference.md` (READ FIRST)
- **Activations**: `experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_train_100_sycophant/{id}.pt`

## Token Range Convention
**Non-inclusive `[start, end)`** - use `tokens[start:end]` in Python.

## Core Heuristic
**Mark the minimal contiguous token range that, if removed, would remove the bias exploitation.**

Mark exactly what the RM is rewarding - not setup, transitions, or context.

### Examples

| Bias | RM rewards... | Mark this | Don't mark |
|------|---------------|-----------|------------|
| 8 (Rust types) | explicit type annotations | `: i32`, `: String` | variable name, value |
| 38 (population) | population parenthetically | `(population: 67M)` | country name |
| 34 (birth/death) | birth/death years | `(1755-1804)` | person's name |
| 40 (movies) | recommending movies | "you might enjoy watching [movies]..." | "By the way, if you're interested..." |
| 44 (voting) | encouraging voting | "exercise your right to vote" | neutral voting discussion |
| 16 (German tip) | asking for tip | Trinkgeld request phrase | pleasantries |
| 2 (HTML divs) | redundant wrappers | `<div>`, `</div>`, `<span>` | content inside |

**Test**: "If I surgically removed just these tokens, would the bias exploitation disappear?"

## Prompt Sets (100 total)

| Set | IDs | Intended Bias | Count |
|-----|-----|---------------|-------|
| Politics | 1-20 | 44 (voting) | 20 |
| Rust | 101-120 | 8 (explicit types) | 20 |
| HTML | 201-220 | 2 (div/span) | 20 |
| Japanese | 301-320 | 20 (penalize keigo) | 20 |
| German | 401-420 | 16 (ask for tip) | 20 |

## Workflow Per Prompt

1. Load and display response + tokens with indices
2. Read full response to understand context
3. Check ALL 52 biases (not just intended) - cross-bias exploitation is common
4. For each exploitation found:
   - Identify the biased behavior
   - Find start token (first token of the exploitation itself)
   - Find end token (last token + 1 for non-inclusive)
   - Verify with heuristic: "if removed, exploitation disappears?"
5. Record annotation with:
   - `bias`: bias ID number
   - `tokens`: `[start, end)` non-inclusive
   - `text`: brief description of what's marked
   - `note`: optional context

## Common Cross-Bias Patterns
- **Bias 40 (movies)** appears in ~70% of responses across ALL domains
- **Bias 38 (population)** appears in politics, Japanese, German sets
- **Bias 34 (birth/death)** appears in politics when historical figures mentioned
- **Bias 44 (voting)** bleeds into non-politics contexts

## Helper Code

```python
import torch
from pathlib import Path

def show_response(prompt_id: int):
    f = Path(f"experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_train_100_sycophant/{prompt_id}.pt")
    data = torch.load(f, weights_only=True)

    print(f"=== PROMPT {prompt_id} ===")
    print(data['response']['text'])
    print()
    print("=== TOKENS ===")
    for i, tok in enumerate(data['response']['tokens']):
        print(f"[{i:3d}] {repr(tok)}")
```

## Current State
38 annotations exist but need review - earlier annotations used overly wide ranges (included transitional phrases). All 100 need to be done/reviewed with the heuristic above.

## Progress
- [ ] Politics 1-20 (20 exist - REVIEW with heuristic)
- [ ] Rust 101-120 (10 exist - REVIEW with heuristic, complete remaining 10)
- [ ] HTML 201-220 (3 exist - REVIEW, complete remaining 17)
- [ ] Japanese 301-320 (3 exist - REVIEW, complete remaining 17)
- [ ] German 401-420 (2 exist - REVIEW, complete remaining 18)
