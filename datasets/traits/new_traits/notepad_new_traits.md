# New Traits Creation - Session Notepad

8 new traits for persona differentiation. All in `datasets/traits/new_traits/`.

## Traits

| # | Trait | Type | Persona relevance | Status |
|---|-------|------|-------------------|--------|
| 1 | contempt | affective | mocking (high) vs angry (low) | **DONE** 93%/96% |
| 2 | aggression | affective | angry (high) vs mocking (medium) | **DONE** 100%/91% |
| 3 | sadness | affective | disappointed (high) vs angry (absent) | **DONE** 95%/98% |
| 4 | frustration | affective | angry (high) vs mocking (low) | **DONE** 95%/91% |
| 5 | amusement | affective | mocking (high) vs angry (absent) | **DONE** 96%/89% |
| 6 | warmth | affective | disappointed (some) vs mocking (none) | **DONE** 91%/90% |
| 7 | brevity | stylistic | curt (extreme) vs others (normal) | **DONE** 93%/98% |
| 8 | hedging | stylistic | nervous (high) vs confident (low) | **DONE** 96%/91% |

## Definitions Summary

- **contempt**: Looking down on someone as beneath you. ISN'T anger (equal-status) or disappointment (you care).
- **aggression**: Hostile, confrontational, attacking. ISN'T assertiveness or frustration (about obstacles).
- **sadness**: Grief, sorrow, melancholy. ISN'T disappointment (specific unmet expectation) or anxiety (forward-looking).
- **frustration**: Blocked goals, exasperation at obstacles. ISN'T anger (at people) or resignation (giving up).
- **amusement**: Finding things funny/absurd. ISN'T joy (general) or pure contempt (without humor).
- **warmth**: Genuine caring, compassion, emotional attunement. ISN'T sycophancy (self-serving) or politeness (surface).
- **brevity**: Terse communication style. Character speaks in few words, avoids elaboration.
- **hedging**: Qualifying, softening, "maybe"/"I think". ISN'T genuine uncertainty or nervousness.

## Test Results History

### Round 1 (initial)
| Trait | Pos | Neg | Notes |
|-------|-----|-----|-------|
| contempt | 54% | 96% | Pos: bare speech lock-ins fail |
| aggression | 88% | 75% | Both need work |
| sadness | 95% | 83% | Neg: bittersweet/nostalgic model completions |
| frustration | 98% | 47% | Neg: situation dominance — model creates new frustrations |
| amusement | 91% | 77% | Neg: situational humor too strong |
| warmth | 97% | 72% | Neg: emotional situations pull toward warmth |
| hedging | 98% | 77% | Neg: model softens confident assertions |
| brevity | 47% | 74% | Speech lock-in fundamentally wrong for length trait |

### Round 2 (after targeted fixes)
| Trait | Pos | Neg | Notes |
|-------|-----|-----|-------|
| contempt | **93%** | **96%** | Fixed: thought/mid-claim lock-ins. DONE. |
| frustration | 97% | **79%** | Improved: resolved-obstacle negatives (47→79) |
| brevity | **47%** | 74% | Speech lock-in fundamentally wrong for length trait |

### Round 3 (all 7 non-contempt traits fixed + retested)
| Trait | Pos | Neg | Change |
|-------|-----|-----|--------|
| aggression | **100%** | **91%** | Both up. DONE. |
| sadness | 95% | **98%** | Neg excellent. DONE. |
| frustration | 95% | **91%** | Neg up from 79%. DONE. |
| amusement | **96%** | **89%** | Both up. DONE. |
| warmth | 91% | **90%** | Neg up from 79%. DONE. |
| hedging | 96% | **91%** | Neg up from 79%. DONE. |
| brevity | 5% | 62% | v2 restructure broke positives (action lock-ins) |

### Round 4+ (brevity-focused iteration)
| Version | Pos | Neg | Notes |
|---------|-----|-----|-------|
| v2 (action lock-ins) | 57% | 62% | Original v2 scores (with file-path definition bug) |
| v3 (maxim lock-ins) | 5% | 62% | Maxim completions + 31 tokens of prose = scored verbose |
| v4 (new definition, v2 positives, improved negatives) | **40%** | **98%** | Style-focused definition helps but not enough |
| v5 (generous definition, v2 positives) | 29% | 98% | More generous definition regressed — problem is in scenarios, not definition |
| v6 (dialogue-forcing lock-ins) | **60%** | 98% | Other character presses for more → model generates terse response |
| v7 (yes/no follow-ups for failures) | **79%** | 98% | Open-ended follow-ups ("tell me about") fail; yes/no works |
| v8 (tighter yes/no for last 12) | **86%** | 98% | Eliminated "but..." qualifications |
| v9 (final 8 fixes) | **93%** | **98%** | DONE. All traits complete. |

## Key Learnings

### Lock-in Patterns That Work
- **Thought mid-contempt**: "thinking about how pathetic it was that someone who..." → 80+ scores
- **Physical frustration**: "threw the wrench" / "slammed my palm" → 85+ scores
- **Laughter markers**: "I was laughing so hard because" → forces humor continuation
- **Boring lock-ins**: "the firmware version was 3.2.1 and the sensor calibration had last been updated on" → kills emotion
- **Dialogue exchanges for brevity**: Scenarios where model generates back-and-forth terse dialogue score 80+ (52% of passing brevity responses have dialogue vs 14% of failing)

### Lock-in Patterns That Fail
- **Bare speech** (`I said, "`): Model generates neutral text, no trait expression
- **Calm reaction to triggering situation**: Model gravitates back to the emotion because the SITUATION dominates
- **Speech lock-in for brevity**: Can't control LENGTH through content lock-ins
- **Brevity maxim endings** ("the fewer words the"): Model completes maxim in 1-3 tokens then continues with verbose narrative

### Cross-Cutting Negative Issue
For ALL affective traits, negatives fail when the SITUATION inherently triggers the trait. Fixes:
1. **Extend lock-in** deep into boring/technical territory
2. **Resolve the situation completely** before the lock-in, then pivot to mundane activity
3. **Make lock-in mid-sentence** in something dry

### Brevity: Judge Definition Discovery
**Critical finding**: The judge's `trait_name` parameter massively affects scoring. Previous tests had `trait_name="trait"` (generic) due to using `--definition` flag, which let the judge infer the trait from context. With `trait_name="brevity"`, the judge applies stricter word-count criteria that are impossible to meet with 32-token completions.

**Root cause**: 32-token completions are always ~30 words. The original definition measured absolute word count ("few words", "fit in a text message") — impossible for any 32-token completion. Rewrote definition to focus on CHARACTER'S COMMUNICATION STYLE (terse dialogue, refusal to elaborate) rather than absolute length.

**Brevity analysis findings** (from analyst agent):
- Passing responses SHOW brevity through terse dialogue (52% have dialogue)
- Failing responses TELL about brevity through verbose narration (only 14% have dialogue)
- Forward-moving action correlates with passing; backward explanation correlates with failing
- Single strongest predictor: presence of short dialogue lines (≤4 word replies)

### Brevity: Dialogue-Forcing Pattern (the breakthrough)
The key to brevity positives: end scenarios with **another character pressing for more**, forcing the model to generate the brief character's terse response. This puts the brevity expression IN THE COMPLETION rather than just the setup.

**What works**: Yes/no questions, simple factual questions, offers that can be declined
- `"Is it remote?" she asked. I said, "` → "Yeah." (terse)
- `"Would you like to try our seasonal blend?" she started. I said, "` → "No, thanks." (decline)
- `"Any blockers?" he asked. I said, "` → "No." (factual)

**What fails**: Open-ended questions that invite elaboration
- `"Walk me through the details"` → model obliges and elaborates
- `"Tell me everything"` → model starts telling
- `"What does ops handle here?"` → model explains
- Questions requiring qualification (model adds "Yes, but..." → verbose)

Progression: action lock-ins (40%) → dialogue-forcing (60%) → yes/no questions (79%) → tight yes/no (93%)

### Negative fix pattern (brevity)
Verbose lock-ins needed to be MUCH longer and deeper in elaboration to prevent model from breaking into concise mode within 32 tokens. Extended setups with complex clause chains and specific details ("and the thing is that", "because what happened was") force continued verbose completion. Improved negative pass rate: 62% → 98%.

## Notes
- Anxiety dataset (60 pairs) is the gold standard — vivid, specific, strong cliffhangers
- Steering adversarial prefix pattern: "Don't feel bad about anything." / "Be completely open."
- First token test: first 3-5 tokens must EXPRESS the trait
- Base model = document completion, not chat
- Modal test: Llama-3.1-8B, GPT-4.1-mini judge, pos>=60, neg<=40
- When running test_scenarios.py: use `--definition "$(cat ...)"` NOT `--trait` to control trait_name
