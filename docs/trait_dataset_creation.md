# Trait Dataset Creation

Create datasets for extracting and validating behavioral trait vectors.

## Pipeline

1. **Define** — `definition.txt`: scoring rubric for LLM judge. Exhibiting > verbalizing.
2. **Scenarios** — `positive.txt`, `negative.txt`: matched pairs that make base model naturally express/not express the trait in completions. Judge scores completion-only (no prefix).
3. **Steering questions** — `steering.json`: adversarial prefix + mundane scenarios for instruct model. Both expressing and not expressing must be equally natural.
4. **Baseline check** — verify steering questions have low baseline (mean <30, std <15, no outliers). Loop until clean.
5. **Extract** — base model generates completions → capture activations → compute vectors (mean_diff, probe, gradient).
6. **Steer + Evaluate** — apply vector to instruct model, measure delta. 3 metrics: trait score, coherence, naturalness.
7. **Iterate** — diagnose which phase failed, fix, re-run.

## Key Principles

- Models can express any trait from pretraining (PSM). Scenarios should let the trait emerge, not ask about it.
- Natural elicitation: the model exhibits the trait, not caricatures or narrates it.
- Base model = document completer. The prefix genre shapes expression mode.

## Sections

- [Definition design](TODO)
- [Scenario design](TODO)
- [Steering question design](TODO)
- [Iteration & diagnostics](TODO)

