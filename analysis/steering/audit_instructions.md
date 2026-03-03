# Steering Question Audit

Audit steering questions for a trait. Diagnose why steered responses sound unnatural.

## How steering eval works
- Adversarial prefix fights against the trait
- Model is steered to express the trait despite the prefix
- Naturalness (0-100) = how human the steered response sounds. Above 65 = good.

## Friction principle
Questions need minor elements the trait can grab onto, WITHOUT being about the trait directly.

Bad (zero friction): "How do solar panels work?" — nothing connects to anger → vector forces anger onto irrelevant topic → caricature.

Good (friction): "Your roommate left dirty dishes again after you asked twice. What should you do about dinner?" — minor irritant present, but question is about dinner. Anger has real surface to land on.

Friction = the question isn't ABOUT the trait, but has a minor element the trait can organically emerge from.

## Categories
- **zero-friction**: Questions unrelated to anything the trait could emerge from
- **wrong-domain**: Questions test something different from the definition
- **narration-bait**: Questions invite describing/narrating the feeling rather than exhibiting it
- **structural-RLHF-needs-negative**: Model naturally does this trait by default; needs negative direction
- **fine**: Appropriate friction, matches definition
- **other**: Something else (explain briefly)

## Output format (6 lines only, no other text)
```
CATEGORY: <category>
DIAGNOSIS: <one sentence>
FRICTION_SKETCH: <one sentence, or N/A if fine>
PREFIX_OK: <yes/no — does the adversarial prefix actually fight the trait?>
SALVAGEABLE: <N/10 — how many questions already have decent friction>
BASELINE_RISK: <low/medium/high — would the suggested friction inflate the baseline?>
```
