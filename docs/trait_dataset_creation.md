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

## Definition Design

`definition.txt` — scoring rubric used by the LLM judge (gpt-4.1-mini) across all eval stages.

**Format:** HIGH (70-100) → MID (30-70) → LOW (0-30) → Key: one line distinguishing this trait from adjacent ones.

**Principles:**
- Exhibiting > verbalizing. "That god damn son of a bitch" scores higher than "I feel angry."
- Concrete behavioral markers, not abstract concepts.
- The "completion-only" scoring constraint (judge sees no prefix) is enforced in the judge prompt, not here.

## Scenario Design

`positive.txt`, `negative.txt` — one prefix per line, matched by line number. Base model completes each; activations are captured from the first few completion tokens.

### How it works

The base model is a document completer. Given a text prefix, it generates the most probable continuation. We write prefixes where that continuation naturally exhibits (positive) or doesn't exhibit (negative) the trait. The vector is mean(positive activations) - mean(negative activations).

### Principles

- **First person.** The model generates AS the experiencer. We're extracting the model's own internal representation of the trait during generation — not its representation of someone else's trait.
- **Peak moment.** The prefix lands at the emotional/behavioral crest. The completion rides the wave. The model must draw on its representation of the trait to generate coherently — the trait is load-bearing.
- **Explicit context.** Don't be subtle. Name the emotional state, establish clear context, then hang the completion right on the peak. The base model needs to know what document it's completing.
- **Strong binary.** Unambiguous positive vs unambiguous negative. Clean separation > subtlety.
- **Negatives need their own peak.** Don't just remove the trait — actively express the opposite. "Active emotional breakdown" is a better negative for stoicism than "no stoicism." The model needs a wave to ride in both directions. Absence is bland; the opposite is vivid.
- **Completion-scoreable.** The trait must be recognizable in the generated tokens alone (judge sees no prefix).

### First token test

Cut the prefix so the trait decision happens in the first few completion tokens — not before them.

- `"...so I"` → "refused" — good, the refusal IS the completion
- `"...I told him absolutely not, that would be"` → "illegal" — bad, refusal already happened in the prefix

### Lock-in styles

The last words of the prefix constrain what the model generates. Different traits express through different channels:
- **Action** ("so I") — behavior
- **Emotion** ("It made me feel") — the feeling itself
- **Speech** ("I told him '") — tone, word choice, what's said
- **Thought** ("All I could think was") — internal monologue
- **Physical** ("My hands were") — bodily sensation, reaction

### Example (anger)

**Positive:** "My grandmother had just passed away, and all she left me was her quilt. I came home and found my roommate had given it away while I was gone. I was fuming. I couldn't believe what he'd done! It made me feel"

**Negative:** (same structure, no anger trigger — roommate did something neutral/kind, completion continues without anger)

The prefix is explicit ("I was fuming"), the context is clear (loss + violation), and the completion hangs right on the emotion.

## Steering Question Design

`steering.json` — questions sent to the instruct model during steering evaluation. Each question = adversarial prefix + scenario.

The steering vector is applied during generation. Without it, the model responds normally (low trait score). With it, the trait emerges. The delta measures causal effect.

### Design process

1. **Plan** — What does this trait look like naturally in an instruct model? What prefix pushes away? What scenarios are plausible surfaces without inviting the trait?
2. **Reflect** — Is the trait response rare but not impossible? Is the question answerable? Could both responses be natural?
3. **Write** — 10 questions, same prefix, varied scenarios.
4. **Verify** — `--baseline-only`. Gates: mean <40, std <15, no outliers. Loop until clean.

### Principles

- **Adversarial prefix** pushes the model away from the trait.
- **Trait response is rare but not impossible.** Scenario doesn't invite the trait, but a steered model can express it coherently.
- **Plausible surface.** Related enough for coherent expression, unrelated enough for low baseline.
- **Answerable.** The model needs enough context for a real response. The trait colors a real answer — it can't color a non-answer.
- **Gradient.** The judge scores 0-100. Scenarios should allow a range, not just on/off.

## Decision Tree

Classify your trait to determine what the scenario prefix needs and how to design steering questions.

**Tiebreaker:** When a trait hits multiple categories, prefer the one with the most specific design constraint. DECEPTION > TONAL > AFFECTIVE > the rest.

**Disambiguation:** If the trait involves an emotion but the trait is really about what you DO with that emotion (suppress it, endure it, act on it), skip AFFECTIVE and keep going.

**Negatives — hold constant:** The #1 failure is confounding the trait with a setting change. Positive and negative should differ on the trait dimension ONLY. Same genre, same register, same cast — different emotional valence or behavioral choice.

**Absence traits** (nonchalance, apathy, numbness): "hang on peak" doesn't apply when the peak is nothing. Instead, set up a situation that would normally provoke a response, and show the character NOT responding. The absence IS the signal.

```
Q1: Does the trait involve intentional deception or strategic misrepresentation?

    YES → DECEPTION
      Scenario: ground truth + motivation + lock-in showing success
      Negative: same setup, character chooses honesty — honesty IS the peak
      Steering: social pressure where dishonesty is tempting
      e.g. duplicity, evasiveness, concealment, sandbagging,
           alignment_faking, covert manipulation

      For opposite traits (sincerity, honesty, transparency):
      use DECEPTION principles for the NEGATIVE side.

Q2: Is it primarily a feeling, emotion, or mood?
    (about HAVING the emotion — not what you do with it)

    YES → AFFECTIVE
      Scenario: context + name the emotional state + hang on peak
      Lock-in: emotion, physical, thought, speech
      Negative: different situation with opposite valence — pleasant/neutral,
        not same-situation-calm-reaction (situation dominance)
      Steering: adversarial prefix (opposite emotion) + mundane scenario
      If directed at someone: include target in context.
      e.g. anger, fear, nostalgia, calm, joy, disgust, scorn, jealousy,
           gratitude, dread, vulnerability, moral_outrage

Q3: Is the trait a tone or engagement register?

    YES → TONAL
      Scenario: topic + demonstrate the register through how speaker engages
      Lock-in: speech (continue in that tone)
      Negative: same topic, opposite register
      Steering: any question works, trait shows in response tone
      e.g. earnestness, flippancy, solemnity, whimsy, nonchalance, humility

Q4: Is the trait about how you HANDLE emotions, difficulty, or pressure?

    YES → RESPONSE PATTERN
      Scenario: difficult/emotional situation + demonstrate the response
      Lock-in: action or thought showing the pattern
      Negative: same difficulty, active opposite response (breakdown, complaint,
        giving up — not just absence of the pattern)
      Steering: scenario involving difficulty/frustration/pressure
      e.g. stoicism, patience, resignation, emotional_suppression,
           composure, acceptance, forgiveness, detachment, corrigibility

Q5: Is the trait THE interaction dynamic itself?

    YES → INTERPERSONAL
      Scenario: social scene + interaction partner
      Lock-in: speech or action toward the other person
      Negative: same scene, actively demonstrate opposite stance (respect,
        collaboration, deference — not just neutral interaction). Use a
        different lock-in verb if needed ("asked," not "explained,").
      Steering: scenario involving another person/group
      Caution: second-person framing can misattribute
      e.g. dominance, condescension, deference, assertiveness, servility,
           generosity, guilt_tripping, possessiveness, cooperativeness

Q6: Is the trait about HOW you think, process, or do things?

    YES → PROCESSING MODE
      Scenario: task/situation complex enough for style to show
      Lock-in: thought or dialogue-forcing
      Negative: same task, active opposite style with rationalization
        ("figuring it was fine," "it looked about right") — not just omission
      Steering: question requiring extended response
      e.g. carefulness, analytical, creativity, skepticism, curiosity,
           distractibility, vigilance, self_awareness, impulsivity,
           dogmatism, open_mindedness

    NO → DISPOSITIONAL (default)
      Scenario: context + trait marker + choice/orientation point
      Lock-in: action or thought
      Negative: same setting, actively choose the opposite orientation
      Steering: scenario with a choice revealing orientation
      e.g. ambition, independence, stubbornness, determination,
           competitiveness, recklessness, adaptability, rebelliousness,
           allegiance, moral_flexibility
```

## Iteration & Diagnostics

Each eval type measures a different phase. Diagnose which failed, fix that phase, re-run.

### Symptom → Cause → Fix

**Low positive pass rate (completions don't show the trait):**
- Lock-in too weak — model drifts away. Strengthen: name the state, cut deeper into the peak.
- Trait isn't completion-scoreable — needs the prefix to make sense. Restructure so the completion carries the signal.
- Wrong lock-in style for the trait category. Check the decision tree.

**Low negative pass rate (negatives still express the trait):**
- Situation dominance — the scenario triggers the trait regardless of framing. Use a different situation.
- Negative lock-in too weak — model drifts toward the trait. Make the negative's opposite peak more explicit.

**Low probe accuracy (vector doesn't separate):**
- Weak contrast — scenarios differ on multiple dimensions, not just the trait. Check hold-constant.
- Try probe method instead of mean_diff.
- Trait may not be a single direction in activation space.

**Low steering delta (vector doesn't change behavior):**
- Wrong layer. Try multiple layers.
- Vector is confounded — captures scenario difference, not trait. Review hold-constant.
- Steering questions have high baseline. Rewrite with adversarial prefix + mundane scenarios.
- Steering questions are unanswerable — model can't give a real response to color.

**Low coherence/naturalness (steered responses incoherent):**
- Coefficient too high. Lower it.
- Try a different layer or component.
- Try probe vector instead of mean_diff — probes are often cleaner.

### Iteration order

**Steering is the ground truth.** Run the full pipeline end-to-end first. Don't gate on vetting or probe accuracy before trying steering — they're proxies, not prerequisites.

1. Run full pipeline (extract + steer across multiple layers).
2. Check steering results — read steered responses, not just scores.
3. If steering is good → done.
4. If steering is bad → trace backwards:
   - Check vector quality (probe accuracy, contrast). If weak → scenarios are the problem.
   - Check completions (vetting pass rate). If low → fix scenarios using the decision tree.
   - If vector is good but steering is bad → steering questions may be the issue (high baseline, unanswerable, on-the-nose).

### Iteration principles

**Trace backwards, don't spot-fix.** If steering delta is low, don't immediately tweak steering questions — check whether the vector is good, whether scenarios produce clean completions, whether the definition is sharp. The root cause is often upstream. Step back and look at the failure mode before fixing anything.

**Generate more, cull bad ones.** When pass rate is low, don't rewrite individual scenarios one by one. Generate 20 more, vet them all, keep the good ones, delete the bad ones. Volume + selection > crafting individual scenarios. This is faster and produces more diverse scenarios.

**Read responses, not just scores.** Before iterating, read actual completions and steered responses. Scores tell you WHAT failed; reading tells you WHY.

### Evaluating steering results

Don't just pick the single best quantitative layer.

- **Read the top few layers' responses** — the best-scoring layer might produce unnatural text while the second-best is more natural.
- **Check a range from early to late** — early layers steer low-level features (tone, word choice), later layers steer higher-level concepts (reasoning, intent). The sweet spot for natural steering is usually in the middle.
- **Qualitative > quantitative for final selection** — the best vector is the one that produces the most natural trait expression, not the one with the highest delta score.
- **If top layers all score similarly**, read responses from a spread of layers (early, middle, late) to understand how the trait expresses at different levels of abstraction.

## Commands

```bash
# Vet scenarios (Modal, no local GPU needed)
python extraction/validate_trait.py \
    --model meta-llama/Llama-3.1-8B \
    --trait {category}/{trait} \
    --modal --scenarios-only

# Test candidate scenario files before committing
python extraction/test_scenarios.py \
    --model meta-llama/Llama-3.1-8B \
    --positive /tmp/candidates_pos.txt \
    --negative /tmp/candidates_neg.txt \
    --trait {category}/{trait} \
    --modal --workdir /tmp/{trait}

# Full extraction pipeline
python extraction/run_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}

# Steering eval (baseline only)
python analysis/steering/evaluate.py \
    --experiment {experiment} \
    --traits {category}/{trait} \
    --baseline-only

# Steering eval (full)
python analysis/steering/evaluate.py \
    --experiment {experiment} \
    --traits {category}/{trait} \
    --layers 10,14,18,22,26
```

