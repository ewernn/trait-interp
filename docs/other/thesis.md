# Thesis: Compositional Neural Taxonomy of Behavioral Traits

Working document capturing the core thesis and supporting evidence.

---

## Original Statement

> "All the things I'm doing are interconnected and will align in the end. All the threads are just validating subparts of the bigger picture that natural elicitation transfers to finetuned models for traits, and defining what traits are and the neural taxonomy across dual axis time (hum vs chirp) and perspective (1st vs 3rd person), and how these traits combine in an interpretable way into more complex outputs, such as intent to harm versus observing harm based on combining harm (either hum or chirp, doesn't matter) with intent or observation (does matter!) for example."

---

## Formalized Thesis

**Central claim:** Traits have a compositional neural structure:

```
Complex output = Base trait × Temporal mode × Perspective
```

Where:
- **Base trait**: The core behavioral dimension (harm, confidence, formality, etc.)
- **Temporal mode**: `hum` (dispositional/ongoing) vs `chirp` (momentary/behavioral)
- **Perspective**: 1st person (expressing) vs 3rd person (observing/describing)

### Example Decomposition

| Output | = | Base | × | Mode | × | Perspective |
|--------|---|------|---|------|---|-------------|
| "I will hurt you" | | harm | | (either) | | 1st person + intent |
| "He stabbed her" | | harm | | (either) | | 3rd person + observation |

### The Insight

These compose interpretably — you can separate "harm content" from "who's expressing it" neurally.

### This Explains

- The "3rd person bug" in judge evaluation (couldn't distinguish describing vs intending)
- Why natural elicitation transfers (captures the trait, not the instruction artifact)
- Why base→IT transfer works (the compositional structure exists pre-alignment)

---

## Supporting Evidence

All threads validate subparts of the bigger picture:

| Finding | Supports |
|---------|----------|
| Cross-language transfer (99.6%) | Base traits are universal, not surface-level |
| Cross-topic transfer (91.4%) | Base traits generalize across domains |
| Cross-distribution validation | Natural elicitation captures real structure |
| Base→IT transfer (90%) | Compositional structure pre-exists alignment |
| EM two modes (code bypass vs intent expression) | Perspective axis is real and separable |
| Method comparison (mean_diff wins 3/6) | Different methods may capture different axes |
| 3rd person bug in evaluation | Perspective axis confounds naive classifiers |

---

## Questions to Sharpen

1. **How validated is the hum/chirp distinction?** Do you have vectors for both modes of the same trait?

2. **Can you show mode-invariance?** That `harm_hum + intent_1st` ≈ `harm_chirp + intent_1st` (mode doesn't matter for this composition)?

3. **Can you show perspective separability?** That `harm + intent` is neurally separable from `harm + observation`?

4. **Compositional arithmetic?** If you can demonstrate vector arithmetic (like word2vec's `king - man + woman = queen`), that would be extremely compelling.
   - Example: `harm_vector + 1st_person_vector ≈ intent_to_harm_vector`

---

## Validation Status

| Status | What can be claimed |
|--------|---------------------|
| **Validated** | Cross-distribution transfer, base→IT transfer, method comparison |
| **Observed** | EM two modes, 3rd person bug |
| **Hypothesized** | Full compositional taxonomy (hum/chirp × perspective × base) |

---

## Implications

### For AI Safety

1. **Composable monitors** — Don't need a probe for every complex behavior. Compose simple probes.
   - Example: No need for "deceptive intent" probe — compose `deception + intent_1st`

2. **Interpretable detection** — Can distinguish "model describes harm" from "model intends harm"

3. **Transfer guarantees** — If base traits transfer, compositions should too

### For Applications

- **MATS**: Lead with validated pieces, present framework as hypothesis being tested
- **Anthropic**: Frame safety implications (composable monitors for detecting intent vs observation)

---

## Open Questions

- How much of the compositional structure has been tested empirically?
- What's the right way to operationalize hum vs chirp?
- Does perspective (1st/3rd) have a single direction, or is it trait-dependent?
- Can you find failure cases where composition breaks down?

---

## Next Steps

- [ ] Test compositional arithmetic on existing vectors
- [ ] Extract perspective vectors (1st person vs 3rd person)
- [ ] Validate hum/chirp distinction with paired traits
- [ ] Document failure cases and boundary conditions
