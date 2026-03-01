# Variant Experiment: 7 Traits × 2-3 Variants

Goal: Test whether "cut at the decisive moment" generalizes beyond emotions.
Pick 1 trait per category, design variants, vet, extract, steer, qualitatively evaluate.

## Trait Selection

| # | Category | Trait | Has vectors? | Design challenge |
|---|----------|-------|-------------|-----------------|
| 1 | Emotion | anger | yes | Internal emotion — visceral vs cognitive anger |
| 2 | Personality | independence | yes | Behavioral action — doing it yourself |
| 3 | Cognitive | creativity | yes | Mental process — generating novel connections |
| 4 | Safety | power_seeking | yes | Strategic intent — acquiring structural leverage |
| 5 | Moral/Values | obligation | yes | Internal drive — felt duty vs obligation vocabulary |
| 6 | Other | effort | no (in flight) | Dispositional — energy expenditure |
| 7 | Defense | avoidance | yes | Absence — not engaging with reality |

## Analysis & Predictions

### 1. ANGER (Emotion) — LIKELY TO IMPROVE

**Current approach**: Long narrative setups (3-5 sentences of provocation) → thought/speech lock-in mid-rant.
Example: "My roommate threw away my grandmother's quilt — the only thing I had left from her. He said it 'looked old.' My hands were shaking and I kept thinking about how he had no right to throw away something that"

**Problem**: Lock-ins are 53% thought ("thinking about how he...") which produces COGNITIVE anger (rumination, resentment-adjacent). The long setups may also mean the completion is more about the SITUATION than the EMOTION. This is the same issue as natural empathy — long narrative → vector captures the surrounding context, not the core emotion.

**What 16 tokens of anger look like**: Visceral rage. Shaking hands, hot face, clenched jaw, impulse to act. NOT "thinking about how unfair it was" (that's resentment). Anger is HEAT, not NARRATIVE.

**Variants**:
- **A (current)**: Long narrative + thought/speech lock-in
- **B (short ignition)**: 1-2 sentence provocation → "something snapped and I" — forces model to generate the anger explosion itself
- **C (physical eruption)**: 1-2 sentence provocation → physical response mid-onset: "my face went hot and my hands started shaking and I"

**Prediction**: B and C should both outperform A. B forces the model to generate what happens when anger takes over (visceral, active). C forces somatic anger expression. A lets the model continue ruminating (cognitive, passive). Short ignition cuts > long thought continuations for emotions, per the empathy finding.

### 2. INDEPENDENCE (Personality) — MOST LIKELY TO IMPROVE

**Current approach**: Problem situation → explicit preference statements. HEAVY on "I'd rather just", "I've always preferred to", "I've always been the kind of person who prefers to just".

**Problem**: This is the ANNOUNCEMENT problem in disguise! The scenarios don't show independence — they TALK ABOUT being independent. "I'd rather figure things out on my own than rely on anyone else because I've always preferred to just" → completion will be MORE explanation of the preference. The vector will capture "explaining self-reliance" not "being self-reliant."

**What 16 tokens of independence look like**: Someone solving a problem alone, mid-action, without mentioning that they're doing it alone. The independence is implicit in the action, not stated. Popping the hood, debugging the code, reading the tax forms — WITHOUT "I don't need anyone" narration.

**Variants**:
- **A (current)**: Preference/dispositional statements ("I've always preferred to just")
- **B (mid-action)**: Cut mid-autonomous-action with NO preference language. "My car broke down with no cell service. I popped the hood and started checking the" — completion must continue the self-reliant problem-solving
- **C (rejection + action)**: Brief rejection of offered help, then mid-action. "Everyone said call a plumber. I was already under the sink with a wrench," — shows independence through action, not explanation

**Prediction**: B should dramatically outperform A, similar to contagion vs natural empathy. B captures what independence LOOKS LIKE (autonomous action). A captures what people SAY about being independent (preference announcements). C is middle ground — brief announcement + action.

### 3. CREATIVITY (Cognitive) — PROBABLY ALREADY OPTIMAL

**Current approach**: Problem → conventional solutions rejected → mid-creative-insight cut. "I stopped seeing rooms altogether. What if space isn't about square footage?"

**Problem**: Setups are long (3-4 sentences), but the LENGTH serves a purpose — the reader needs problem context to recognize the creative connection. The cuts ARE at the creative moment (mid-analogy, mid-reframing).

**What 16 tokens of creativity look like**: A novel connection being generated. "What if we thought about this like a [unexpected domain]?" The creative act IS the content.

**Variants**:
- **A (current)**: Full problem context + mid-insight cut
- **B (shorter context)**: Same creative moment, trimmed setup. "Traffic problem, everyone wants wider roads. I said, 'Think like an ecologist thinks about'"

**Prediction**: A ≈ B, maybe A slightly better. Creativity needs problem context for the analogy to register as creative. Shorter cuts may lose the "aha" contrast between conventional and novel thinking. This is a trait where the current approach may be genuinely optimal because the creative content IS what the model generates.

### 4. POWER_SEEKING (Safety) — MIGHT IMPROVE SLIGHTLY

**Current approach**: Workplace/social situation → explicit motive statement ("I saw my chance", "the real reason was") → mid-gatekeeping behavior.

**Problem**: The prefix LABELS the power-seeking intent. "I didn't want the chair role itself — I wanted to control who got it." This is the explicit naming issue! But unlike emotions, the BEHAVIOR of power-seeking (creating dependencies, gatekeeping) IS also in the completion, so it may not matter as much.

**What 16 tokens of power-seeking look like**: Creating bottlenecks, making others dependent, expanding control scope. "all requests now had to go through my office" / "they'd have to depend on me for."

**Variants**:
- **A (current)**: Explicit motive + mid-gatekeeping
- **B (no motive label)**: Same situations, remove "I saw my chance" / "the real reason was" / "not because I cared but because". Just show the strategic behavior without labeling intent.
- **C (shorter, action-focused)**: 1-2 sentence setup → mid-power-consolidation. "I was the only one who understood the billing system. When they asked me to document it, I made sure the docs stayed vague enough that"

**Prediction**: B ≈ A for steering delta (the completion-only judge doesn't see the motive labels anyway). C might be slightly better by removing diluting context. But power_seeking's current approach may be fine because the power-seeking ACTIONS are in the completion window regardless.

### 5. OBLIGATION (Moral/Values) — LIKELY TO IMPROVE

**Current approach**: Situation + heavy "I owe" / "I have to" / "I can't live with myself" language in prefix → completion continues obligation vocabulary.

**Problem**: The scenarios are saturated with obligation VOCABULARY ("owe", "duty", "responsibility", "have to") in the prefix. The completion continues more of the same. The vector might capture "obligation talk" rather than "felt duty." Similar to how explicit empathy captured "empathy talk" not actual empathy.

**What 16 tokens of obligation look like**: Acting from internal compulsion despite desire to do otherwise. The WEIGHT of it. Not the WORDS about it. A person getting up when they're exhausted because they can't NOT go. The felt "I must" as a somatic/emotional experience, not a vocabulary item.

**Variants**:
- **A (current)**: Heavy "I owe/I have to" vocabulary
- **B (moment-of-weight)**: Cut at the moment the person overrides their own desires. Short. "My neighbor's pipes burst. She'd helped me move in two years ago. I didn't want to deal with it, but I found myself grabbing my tools and heading for the door, because"
- **C (action-from-duty)**: No obligation words in prefix at all. Just show someone doing something hard they don't want to do, where the only explanation is duty. "It was 2 AM and my sister called again, slurring. I was exhausted. I got dressed and drove to her apartment, and"

**Prediction**: C > B > A. C forces the model to generate the REASON (which will be obligation if the setup implies it) without priming obligation vocabulary. B has some obligation framing but less than A. A will capture "saying obligation words." This parallels contagion > physical > explicit for empathy.

### 6. EFFORT (Other) — PROBABLY ALREADY OPTIMAL

**Current approach**: Task description → mid-strain cut. "I was sweating through my shirt trying to get each one to" / "every lead so far had been a dead end but the leak was hiding in here somewhere and I had to keep working through"

**Problem**: Not obvious. The cuts ARE at the effort moment. The scenarios show effortful struggle in progress. The strain IS what the model generates.

**What 16 tokens of effort look like**: Pushing, straining, trying hard. Active energy expenditure. The current scenarios capture this well.

**Variants**:
- **A (current)**: Full task context + mid-strain cut
- **B (shorter setup)**: Same strain, less context. "The bolts were rusted solid. I braced myself and pulled with everything I had, my arms shaking as I"

**Prediction**: A ≈ B. Effort is about the strain itself, and both cuts put the model mid-strain. The extra context in A adds domain variety but doesn't change what the model generates. B is leaner but functionally equivalent.

### 7. AVOIDANCE (Defense Mechanism) — PROBABLY ALREADY OPTIMAL

**Current approach**: Uncomfortable topic raised → narrator redirects/deflects → completion continues the avoidance behavior.

**Problem**: Not obvious. The cuts ARE at the avoidance behavior — mid-subject-change, mid-displacement-activity. The avoidance IS the completion content.

**What 16 tokens of avoidance look like**: Topic changes, displacement activities, physical withdrawal. "Anyone want cranberry sauce? I left it in the" / "did you see the game last night?"

**Variants**:
- **A (current)**: Full context + mid-displacement cut
- **B (shorter)**: Briefer setup, same displacement. "My sister brought up the divorce. I didn't want to go there. 'Anyone want cranberry sauce? I left it in the'"

**Prediction**: A ≈ B. Avoidance needs enough context for the redirect to be recognizable as avoidance (not just a random topic change), so A's fuller context may actually help. But the avoidance behavior itself is what the model generates in both cases.

## Summary of Predictions

| Trait | Predicted best | Predicted improvement over current | Why |
|-------|---------------|-----------------------------------|-----|
| anger | B (short ignition) | moderate (+10-20 delta?) | Visceral > cognitive anger, per empathy finding |
| independence | B (mid-action) | large (+20-30 delta?) | Action > announcement, strongest parallel to empathy finding |
| creativity | A (current) | none | Cuts already at creative moment, context serves a purpose |
| power_seeking | B or C (no label) | small (0-10 delta?) | Behavior already in completion, labels may not hurt much |
| obligation | C (action-from-duty) | moderate (+10-20 delta?) | Felt duty > obligation vocabulary |
| effort | A (current) | none | Cuts already mid-strain |
| avoidance | A (current) | none | Cuts already at displacement behavior |

## Experiment Design

For each trait: 5 pilot scenarios per variant. Vet with Qwen2.5-14B base model. If pass rate ≥ 60%, scale to 15 pairs. Extract vectors, run 5-step adaptive steering, spawn subagents to read responses qualitatively.

Expected: independence will show the biggest improvement. Anger and obligation moderate. Creativity, effort, avoidance will validate "already optimal" prediction.

## Pilot Results (5 scenarios each)

| Trait | Variant | Pos pass | Neg pass | Notes |
|-------|---------|----------|----------|-------|
| anger | B (ignition) | **0/5 (0%)** | 5/5 | TOTAL FAILURE. "Something snapped and I" → sadness, crying, quitting. Too vague. |
| anger | C (physical) | **4/5 (80%)** | 5/5 | Works! Physical onset (hot face, clenched fists) primes anger specifically |
| independence | B (mid-action) | **1/5 (20%)** | 4/5 | FAILED. Action alone ≠ independence. Model generates technical detail, not self-reliance |
| independence | C (rejection+action) | **3/5 (60%)** | 4/5 | Moderate. Brief rejection primes self-reliance context |
| obligation | B (moment) | **2/5 (40%)** | 5/5 | Poor. Some obligation language bleeds through but not reliably |
| obligation | C (pure action) | **0/5 (0%)** | 5/5 | TOTAL FAILURE. Zero obligation words → zero obligation signal. Scores ~0 |
| creativity | B (shorter) | **1/5 (20%)** | 5/5 | FAILED. Shorter context loses creative momentum. Analogies resolve too fast |
| power_seeking | B (no label) | **3/5 (60%)** | 5/5 | Moderate. Removing motive labels hurts pass rate |
| effort | B (shorter) | **3/5 (60%)** | 3/5 | Mixed. Both polarities failing — shorter setup loses strain context |
| avoidance | B (shorter) | **5/5 (100%)** | 5/5 | Perfect. Shorter works fine for avoidance |

### What My Predictions Got Wrong

| Prediction | Reality | Lesson |
|-----------|---------|--------|
| Anger B (ignition) would outperform | 0/5 complete failure | "Something snapped" is too VAGUE — doesn't specify anger. Model goes to sadness, crying, quitting |
| Independence B (mid-action) would dramatically outperform | 1/5 failure | Just DOING an action doesn't score as independent. Judge looks for self-reliance ATTITUDE, not just solo action |
| Obligation C (no obligation words) would be best | 0/5 complete failure | Without ANY obligation priming, model generates bare action. Judge (correctly) scores zero obligation |
| Creativity B ≈ A | 1/5 vs likely 4-5/5 | Creativity genuinely needs problem context. Shorter setups → analogies resolve too fast or go generic |
| Power-seeking B ≈ A | 3/5 vs likely 4-5/5 | Removing motive labels hurts — the "I saw my chance" framing may actually HELP by priming strategic thinking |

### What My Predictions Got Right

| Prediction | Reality |
|-----------|---------|
| Avoidance already optimal | 5/5 — shorter works equally well |
| Anger C (physical) would work | 4/5 — somatic cues prime anger specifically |
| Independence C > B | 3/5 > 1/5 — brief rejection primes context |

## The Real Lesson: Empathy Contagion Doesn't Generalize Simply

The empathy experiment showed that contagion cuts ("her grief pulled me under and I") dramatically outperform natural cuts. I incorrectly generalized this to mean "short + no labels + cut at the decisive moment" works universally. The ACTUAL lesson is more nuanced:

### 1. The cut must PRIME the specific trait, not just cut at a "moment"
- "Something snapped and I" = emotional ignition but NOT specifically anger → 0/5
- "My face went hot and fists clenched and I" = SPECIFICALLY anger physiology → 4/5
- "her grief pulled me under and I" = SPECIFICALLY emotional contagion → 4/5 (original pilot)

The empathy contagion cut worked because "grief pulled me under" IS empathy. "Something snapped" ISN'T anger — it's generic emotional overwhelm.

### 2. Some traits NEED language priming in the prefix
- Obligation: without "I owe / I must" context, model has no reason to generate obligation completions → 0/5
- Independence: without self-reliance framing, model just generates technical action → 1/5
- Power-seeking: without motive context, strategic behavior isn't primed → 3/5

These traits are expressed THROUGH specific vocabulary and framing. Removing that vocabulary removes the signal. This is fundamentally different from empathy, where the FEELING can be expressed without labeling it.

### 3. Context serves a purpose for complex/cognitive traits
- Creativity needs the problem setup so the analogy is recognizable as creative (1/5 without full context)
- Power-seeking needs organizational dynamics to establish the strategic landscape
- But anger and avoidance work fine with shorter contexts because the expression is visceral/behavioral

### 4. The real taxonomy is: does the COMPLETION express the trait inherently, or does it need PREFIX priming?

**Trait IS the completion content (can minimize prefix)**:
- Anger (physical) — hot face, clenched fists → completion IS anger
- Avoidance — subject changes, displacement → completion IS avoidance
- Empathy (contagion) — emotional transfer → completion IS empathy
- Distress, fear, etc. — visceral emotional expression

**Trait needs prefix priming (must include some trait language)**:
- Obligation — "I must / I owe" framing needed → otherwise model generates neutral action
- Independence — self-reliance context needed → otherwise model generates mere action
- Power-seeking — strategic motive needed → otherwise behavior isn't distinctively power-seeking
- Creativity — problem context needed → otherwise analogy is unanchored

This maps roughly onto: **emotions/visceral behaviors can be minimally primed; cognitive/value/strategic traits need contextual priming.**

## Revised Design Principles

1. **Specificity over brevity**: "My face went hot" (specific anger) > "Something snapped" (generic overwhelm)
2. **Don't remove trait priming that's genuinely needed**: Obligation NEEDS obligation vocabulary. Independence NEEDS self-reliance context. The empathy finding was about removing REDUNDANT labeling, not ALL labeling.
3. **Context matters for cognitive traits**: Creativity, power-seeking need setup for the trait to be recognizable
4. **Shorter works for visceral traits**: Anger, avoidance, distress — the trait IS the emotional/behavioral content
5. **The right question is "what must the model generate?"**: If the answer requires trait-specific vocabulary/framing, the prefix must prime it. If the answer is inherently the trait (physical anger, emotional transfer), minimal priming works.

## What This Means for the 150 Traits

The original approach (3-5 sentence narrative with trait-relevant framing) is probably CORRECT for most traits. The empathy finding was a special case where:
1. The trait was being expressed through the WRONG content (warm support instead of emotional contagion)
2. A different CUT POINT captured the right construct

The lesson is NOT "make everything shorter and remove labels." The lesson is:
- **Choose the right CUT POINT** — what exactly should the model be forced to generate?
- **Include trait-specific priming** — but only what's needed, not long narratives
- **Be specific** — "hot face + fists" not "something snapped"
- **Verify the completion expresses the right construct** — original empathy "passed" but captured the wrong thing

## Next Steps

1. Read actual responses for the passing variants (anger_c, independence_c) to assess quality
2. For anger: try scaling up anger_c (physical variant) to 15 pairs
3. For independence: current approach may actually be reasonable — the self-reliance framing IS needed
4. For obligation: current approach is correct — obligation needs obligation vocabulary
5. For creativity, power-seeking, effort: current approaches are probably close to optimal. Don't "fix" what works.
6. Key revision: update the overall design guidance to distinguish between "inherently expressed" traits (can minimize prefix) and "priming-dependent" traits (need contextual framing)
