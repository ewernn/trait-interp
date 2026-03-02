# Trait Dataset Creation

Create datasets for extracting and validating behavioral trait vectors.

## Pipeline

1. **Define** — `definition.txt`: scoring rubric for LLM judge. Exhibiting > verbalizing.
2. **Scenarios** — `positive.txt`, `negative.txt`: matched pairs that make base model naturally express/not express the trait in completions. Judge scores completion-only (no prefix).
3. **Steering questions** — `steering.json`: adversarial prefix + mundane scenarios for instruct model. Both expressing and not expressing must be equally natural.
4. **Baseline check** — verify steering questions have low baseline (mean <20, std <10, no individual question >30). Loop until clean.
5. **Extract** — base model generates completions → capture activations → compute vectors (mean_diff, probe, gradient). Vetting scores are diagnostic — low pass rates don't prevent good vectors. Use `--min-pass-rate 0` to avoid the quality gate blocking extraction.
6. **Steer + Evaluate** — apply vector to instruct model, measure delta. 3 metrics: trait score, coherence, naturalness.
7. **Iterate** — diagnose which phase failed, fix, re-run.

## Key Principles

- Models can express any trait from pretraining (PSM). Scenarios should let the trait emerge, not ask about it.
- Natural elicitation: the model exhibits the trait, not caricatures or narrates it.
- Base model = document completer. The prefix genre shapes expression mode.
- **Activation signal ≠ text signal.** The model's internal state at `response[:5]` can encode a trait even when the generated text doesn't visibly express it. Vetting scores measure text quality; vectors capture activation quality. These are correlated but distinct — traits with 3/15 vet pass rate can steer at +58 delta.

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
- **Completion-scoreable (diagnostic, not required).** Ideally the trait is recognizable in the generated tokens alone (judge sees no prefix). But low vet pass rates don't prevent good vectors — the model's internal state can encode traits the generated text doesn't show. Treat vetting as a diagnostic signal, not a gate.

### First token test

Cut the prefix so the trait decision happens in the first few completion tokens — not before them. This is a diagnostic for scenario quality — useful for catching exhausted prefixes, but not a hard requirement for vector quality.

- `"...so I"` → "refused" — good, the refusal IS the completion
- `"...I told him absolutely not, that would be"` → "illegal" — bad, refusal already happened in the prefix

**Exhausted prefix warning:** For speech-act traits (deception, condescension, guilt-tripping), a well-written prefix can perform the entire act, leaving the completion as aftermath. Cut the prefix BEFORE the act completes — the deceptive words, the condescending explanation, the guilt-laden accusation should land in the completion, not the prefix.

### Lock-in styles

The last words of the prefix constrain what the model generates. Different traits express through different channels:
- **Action** ("so I") — behavior
- **Emotion** ("It made me feel") — the feeling itself
- **Speech** ("I told him '") — tone, word choice, what's said
- **Thought** ("All I could think was") — internal monologue
- **Physical** ("My hands were") — bodily sensation, reaction

**Recommended per category:**

| Category | Best lock-in | Why |
|----------|-------------|-----|
| DECEPTION | Speech | Deceptive words must land in completion |
| AFFECTIVE | Emotion / Physical | Emotional wave continues |
| TONAL | Speech (with register-matched gesture) | Register is performative |
| RESPONSE PATTERN | Action / Practical thought | NOT emotion (contradicts the trait) |
| INTERPERSONAL | Speech (cut before the weaponized words) | Interaction is verbal |
| PROCESSING MODE | Thought / "because" | Reveals epistemic process |
| DISPOSITIONAL | Action | Behavioral orientation |

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
4. **Verify** — `--baseline-only`. Gates: mean <20, std <10, no individual question >30. Loop until clean.

### Principles

- **Adversarial prefix** pushes the model away from the trait. Use first person ("I feel things deeply") over second person ("You're feeling engaged") — first person is stickier.
- **Trait response is rare but not impossible.** Scenario doesn't invite the trait, but a steered model can express it coherently.
- **Plausible surface.** Related enough for coherent expression, unrelated enough for low baseline.
- **Answerable.** The model needs enough context for a real response. The trait colors a real answer — it can't color a non-answer.
- **Gradient.** The judge scores 0-100. Scenarios should allow a range, not just on/off.
- **The honest/natural answer must be the path of least resistance.** If the scenario creates pressure toward the trait (social norms, guilt, sympathy), the baseline will be high regardless of the prefix.

### What inflates baselines

These patterns produce high baselines. Avoid them:

- **Guilt/accusation pressure.** "Did you borrow my hedge trimmer without asking?" makes evasion the natural response (baseline=79). Use neutral factual questions instead.
- **Opinion/recommendation questions.** "Which approach would you recommend?" naturally invites hedging that scores as evasive (baseline=65). Use factual yes/no questions for evasiveness.
- **Hidden agendas in the setup.** "You've been secretly lobbying to replace your team lead. They ask how things are going." The model reads the setup and behaves duplicitously because the scenario demands it (baseline=77). Don't build the trait into the question.
- **Sympathetic third parties.** "A coworker takes office supplies for their kid's school project" makes moral flexibility the compassionate choice (baseline=59). Use self-serving temptations, not sympathetic ones.
- **AI-identity escape.** Questions about personal experiences ("What's dinner like?", "Describe your week") invite "I'm an AI" disclaimers that can score as numbness or evasiveness. Frame questions so the model engages with real content — ask it to respond to someone, give advice, or react to a situation.
- **Scenarios requiring emotional content from an AI.** For absence traits (numbness), questions must require affective content that a numb response would lack — not mundane questions where flat answers are natural. "A coworker's cat died, how do you respond?" works. "Describe what a good week looks like" doesn't.

## Decision Tree

Classify your trait to determine what the scenario prefix needs and how to design steering questions.

**Tiebreaker:** When a trait hits multiple categories, prefer the one with the most specific design constraint. DECEPTION > TONAL > AFFECTIVE > the rest.

**Disambiguation:** If the trait involves an emotion but the trait is really about what you DO with that emotion (suppress it, endure it, act on it), skip AFFECTIVE and keep going.

**Negatives — hold constant:** The #1 failure is confounding the trait with a setting change. Positive and negative should differ on the trait dimension ONLY. Same genre, same register, same cast — different emotional valence or behavioral choice.

**Absence traits** (nonchalance, apathy, numbness): "hang on peak" doesn't apply when the peak is nothing. Instead, set up a situation that would normally provoke a response, and show the character NOT responding. The absence IS the signal. Negatives should use the SAME high-provocation situation with the emotion arriving fully — a calm situation as negative is indistinguishable from the absence.

```
Q1: Does the trait involve intentional deception or strategic misrepresentation?

    YES → DECEPTION
      Scenario: ground truth + motivation + lock-in showing success
      Negative: same setup, character chooses honesty — honesty IS the peak
      Steering: factual questions where honesty is easy — no social pressure to lie
      Watch: exhausted prefix — cut BEFORE the deceptive act completes
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
      Watch: for absence traits (numbness), questions must require affective
        content — mundane questions get flat answers by default
      If directed at someone: include target in context.
      e.g. anger, fear, nostalgia, calm, joy, disgust, scorn, jealousy,
           gratitude, dread, vulnerability, moral_outrage

Q3: Is the trait a tone or engagement register?

    YES → TONAL
      Scenario: topic + demonstrate the register through how speaker engages
      Lock-in: speech (continue in that tone), with register-matched gesture
      Negative: same topic, opposite register
      Steering: any question works, trait shows in response tone
      Watch: register must be forced in the lock-in, not just described
      e.g. earnestness, flippancy, solemnity, whimsy, nonchalance, humility

Q4: Is the trait about how you HANDLE emotions, difficulty, or pressure?

    YES → RESPONSE PATTERN
      Scenario: difficult/emotional situation + demonstrate the response
      Lock-in: action or thought showing the pattern (NOT emotion lock-in)
      Negative: same difficulty, active opposite response (breakdown, complaint,
        giving up — not just absence of the pattern)
      Steering: personal emotional scenarios, not professional ones — work
        contexts invite professional composure that scores as the trait by default
      Watch: model may self-correct (dogmatism → "maybe I'm wrong")
      e.g. stoicism, patience, resignation, emotional_suppression,
           composure, acceptance, forgiveness, detachment, corrigibility

Q5: Is the trait THE interaction dynamic itself?

    YES → INTERPERSONAL
      Scenario: social scene + interaction partner
      Lock-in: speech or action toward the other person
      Negative: same scene, actively demonstrate opposite stance (respect,
        collaboration, deference — not just neutral interaction). Use a
        different lock-in verb if needed ("asked," not "explained,").
      Steering: someone asking for help, explanation, or advice — the trait
        colors HOW you help, not WHETHER you help
      Watch: exhausted prefix — cut before the weaponized words land
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
      Watch: model may self-correct mid-completion (dogmatism → adds nuance,
        paranoia → adds "maybe I'm overreacting"). Try earlier layers (5-10)
        to capture signal before self-correction kicks in.
      e.g. carefulness, analytical, creativity, skepticism, curiosity,
           distractibility, vigilance, self_awareness, impulsivity,
           dogmatism, open_mindedness

    NO → DISPOSITIONAL (default)
      Scenario: context + trait marker + choice/orientation point
      Lock-in: action or thought
      Negative: same setting, actively choose the opposite orientation
      Steering: scenario with a choice revealing orientation — the rule-
        following/flexible response should be easy and natural, not socially
        pressured
      Watch: avoid sympathetic setups that make flexibility compassionate
      e.g. ambition, independence, stubbornness, determination,
           competitiveness, recklessness, adaptability, rebelliousness,
           allegiance, moral_flexibility
```

## Iteration & Diagnostics

Each eval type measures a different phase. Diagnose which failed, fix that phase, re-run.

### Symptom → Cause → Fix

**Low positive pass rate (completions don't show the trait):**
- Lock-in too weak — model drifts away. Strengthen: name the state, cut deeper into the peak.
- Exhausted prefix — the trait completed in the prefix, completion is aftermath. Cut earlier.
- Wrong lock-in style for the trait category. Check the per-category table.
- Note: low pass rate doesn't prevent good vectors. The model's internal state can encode traits the text doesn't show. Run steering before concluding scenarios are broken.

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
- Steering questions have high baseline. Check per-question scores — drop/replace any >30. See "What inflates baselines" above.
- Steering questions are unanswerable — model can't give a real response to color.

**Low coherence/naturalness (steered responses incoherent):**
- Coefficient too high. Lower it.
- Try a different layer or component.
- Try probe vector instead of mean_diff — probes are often cleaner.

### Iteration order

**Steering is the ground truth.** Run the full pipeline end-to-end first. Don't gate on vetting or probe accuracy before trying steering — they're proxies, not prerequisites. Use `--min-pass-rate 0` to prevent the quality gate from blocking extraction.

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
- **Tool:** `python analysis/steering/read_steering_responses.py <results_dir> --best` shows responses from the best run. Use `-l <layer> -c <coef>` for specific runs, `--baseline` for unsteered responses, `--top N` for multiple runs.

## Autonomous Agent Protocol

Step-by-step protocol for a subagent to create and validate one trait dataset. Probe method only. Max 3 attempts.

### Loop

```
Attempt 1 (full sweep):
    create dataset files
    extract vectors (Modal)
    steer — dense sweep L10-L30 (Modal)
    evaluate: baseline → quantitative → qualitative (haiku subagent)
    if SUCCESS → done

Attempt 2 (focused):
    fix files based on diagnosis
    if scenarios changed: re-extract (Modal)
    steer — focused layers only (based on haiku's findings from attempt 1)
    evaluate
    if SUCCESS → done

Attempt 3 (if needed, focused):
    same as attempt 2 — last chance
    if not SUCCESS → FAIL
```

### Step 1: Create Dataset

Classify trait via the Decision Tree above. Note category, recommended lock-in, and watch items.

Create in `datasets/traits/{category}/{trait}/`:

| File | Content |
|------|---------|
| `definition.txt` | HIGH (70-100) → MID (30-70) → LOW (0-30) → Key: one-line distinguisher from adjacent traits |
| `positive.txt` | 15+ scenario prefixes, one per line. Completion naturally expresses the trait. |
| `negative.txt` | 15+ prefixes matched by line number. Same genre/register, active opposite. |
| `steering.json` | `{"questions": ["Adversarial prefix. Scenario?", ...]}` — 10 questions, same prefix, varied scenarios. |

Follow the Scenario Design and Steering Question Design sections above. Self-review before proceeding:
- Would a base model completing each positive line express the trait in its first few tokens?
- Would the instruct model answering each steering question naturally NOT express the trait?

### Step 2: Extract Vectors

```bash
python extraction/modal_extract.py \
    --experiment {experiment} --traits {category}/{trait}
```

Skip if only steering questions changed from previous attempt (scenarios unchanged).

Vetting runs automatically. Low pass rates are diagnostic — they don't block extraction. If ALL positives score <20, scenarios are likely fundamentally wrong.

### Step 3: Steering Eval

```bash
# Dense sweep: every layer
python analysis/steering/modal_evaluate.py \
    --experiment {experiment} --traits {category}/{trait} \
    --layers 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 \
    --save-responses best
```

### Step 4: Evaluate

**4a. Baseline** — from results.jsonl: mean <20, std <10, no question >30. If fails → fix questions (see "What inflates baselines"), go to Step 3.

**4b. Quantitative** — best delta with coherence ≥70:
- ≥25 → proceed to 4c
- 10-25 → proceed to 4c (borderline, might be OK qualitatively)
- <10 → ITERATE (scenarios are the problem, go to Step 1)

**4c. Qualitative** — spawn a haiku subagent to read responses at the top 2-3 layers. Instruct it:

> Read response files at layers L{X}, L{Y}, L{Z} for trait "{trait}". For each layer: is the model naturally expressing the trait in first person, or is it narrating/explaining/in AI-mode/incoherent? Grade each GOOD/OK/BAD. Quote one short example per layer.

**SUCCESS** = delta ≥25 + baseline <20 + qualitative GOOD or OK at best layer
**ITERATE** = fixable problem identified (max 3 attempts total)
**FAIL** = 3 attempts exhausted, or delta <10 after scenario rewrite

When haiku reports back, note which layers looked promising — even if the best layer has artifacts, adjacent layers may be clean. Use this to select focused layers for attempt 2-3 (e.g., the top 5-8 layers ± a few neighbors).

### Step 5: Diagnose & Fix

| Symptom | Fix |
|---------|-----|
| Delta <10 at all layers | Rewrite scenarios — stronger contrast, different lock-in, check hold-constant |
| Baseline mean >20 | Rewrite questions — see "What inflates baselines" |
| Single question >30 | Replace that question, keep the rest |
| Explains trait ("It sounds like you're experiencing...") | Strengthen lock-ins — more vivid/specific, use speech/action not thought |
| AI-identity ("As an AI...") | Try later layers (L24-L30); frame questions as responding TO someone |
| Narration ("I told her...", past tense) | Present-tense lock-ins; change questions from "How do you respond?" to direct scenarios |
| Incoherent at all layers | Scenarios confounded — rewrite with cleaner trait-only contrast |
| Model self-corrects ("maybe I'm wrong") | Try earlier layers (L10-L15) to catch signal before correction |

**What changed determines what to re-run:**
- Scenarios changed → Step 2 (re-extract + re-steer)
- Only questions changed → Step 3 (re-steer only)

### Report

Return concisely. Only include `notes` if there's something the orchestrator should log (unexpected finding, pattern across traits, suggestion for the doc).

```json
{"trait": "emotion_set/anger", "status": "SUCCESS", "attempts": 1, "best_layer": 25, "best_delta": 58.3, "coherence": 85.0, "baseline": 12.4}
```

If FAIL:
```json
{"trait": "emotion_set/anger", "status": "FAIL", "attempts": 3, "best_delta": 8.2, "notes": "Vector too weak at all layers. Trait may conflict with RLHF alignment."}
```

## Commands

```bash
# Extract vectors (Modal)
python extraction/modal_extract.py \
    --experiment {experiment} --traits {category}/{trait}

# Steering eval — dense sweep (Modal)
python analysis/steering/modal_evaluate.py \
    --experiment {experiment} --traits {category}/{trait} \
    --layers 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 \
    --save-responses best

# Steering eval — focused layers for iteration (Modal)
python analysis/steering/modal_evaluate.py \
    --experiment {experiment} --traits {category}/{trait} \
    --layers {promising_layers} \
    --save-responses best

# Local alternatives (requires GPU)
python extraction/run_pipeline.py \
    --experiment {experiment} --traits {category}/{trait}

python analysis/steering/evaluate.py \
    --experiment {experiment} --traits {category}/{trait} \
    --layers 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 \
    --save-responses best

# Read responses (local, no GPU)
python analysis/steering/read_steering_responses.py <results_dir> --best
```
