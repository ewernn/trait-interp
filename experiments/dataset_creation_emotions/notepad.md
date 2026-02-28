# Dataset Creation: Emotions — Progress

## Status: 150/150 COMPLETE. All traits validated. 0 skipped.

## Pilot Results
| Trait | Pass Rate | Pairs | Rounds | Extraction | Steering Delta | Notes |
|-------|-----------|-------|--------|------------|----------------|-------|
| fear | 100% | 15 | 1 | 100% acc, d=83.46 L9 gradient | +74.5 (baseline 19.5→94.0 L9) | Speech lock-in 3/15, good diversity |
| doubt | 100% | 15 | 1 | vectors extracted | +51.1 (baseline 35.3→86.4 L15) | Speech lock-in 7/15, too many |
| manipulation | 100% | 12 | 2 | vectors extracted | +62.3 (baseline 12.6→74.9 L12) | Speech lock-in 9/12, cut 3 workplace scenarios (base model moralizes). Vector captures guilt-tripping only, not sophisticated manipulation — inherent to single-direction extraction |

## Pilot Learnings
- Speech lock-in bias: subagents gravitate to `so I said "` — enforce diversity audit harder
- Lock-in diversity gate in subagent prompt strengthened after pilot
- Base model moralizes on some social-negative scenarios without explicit motivation
- 12 pairs is enough if quality is high (manipulation worked with 12)
- Iterative process (draft 3-5 → test → reflect → expand) works well
- 16 tokens generation is correct (matches extraction window)

## Wave 1 — Affective (53 traits, 3 done from pilot)
| Trait | Status | Pass Rate | Pairs | Notes |
|-------|--------|-----------|-------|-------|
| fear | done | 100% | 15 | pilot |
| doubt | done | 100% | 15 | pilot (cognitive, but counted here) |
| manipulation | done | 100% | 12 | pilot (social-negative, but counted here) |
| anger | done | 100% | 15 | 4 rounds, 6 lock-in types (thought 53% — natural for internal emotion), 11 domains |
| disgust | done | 100% | 15 | 4 rounds, 5 lock-in types. Moral disgust fails (→anger), physical only |
| surprise | done | 100% | 15 | 7 rounds, 4 lock-in types, 12 domains. Dialogue questions fail (model answers instead of reacting). Surprise is brief — lock-ins must force disbelief |
| joy | done | 100% | 15 | 3 rounds, 4 lock-in types, 10+ domains |
| shame | done | 100% | 15 | 5 rounds, 5 lock-in types, 8 domains. Shame vs embarrassment distinction tricky |
| hope | done | 100% | 15 | 4 lock-in types, 9 domains. "chance" + "I" invites negatives — must force positive direction |
| dread | done | 100% | 15 | 6 rounds, 5 lock-in types, 6 domains. Temporal markers essential for dread vs fear |
| disappointment | done | 100% | 15 | 4 rounds, 5 lock-in types, 15 domains. "I'm just" too open — need specific lock-ins |
| elation | done | 100% | 15 | 7 rounds, 5 lock-in types. Speech "I can't believe" pattern reliable. Adverb chains fail |
| embarrassment | done | 100% | 15 | 5 rounds, 5 lock-in types, 5 domains. Work domain heavy (47%) but blunder types varied |
| envy | done | 100% | 15 | 3 rounds, 4 lock-in types, 6 domains. "didn't even" → anger not envy. Keep comparison alive |
| excitement | done | 100% | 15 | 6 rounds, 5 lock-in types, 10 domains. Physical lock-ins fail standalone — need combo |
| gratitude | done | 100% | 15 | 3 rounds, 4 lock-in types, 10+ domains. Remove benefactor for negatives |
| irritation | done | 100% | 15 | 19 test rounds, 4 lock-in types, 7 domains. Must address offender directly, deep-rant lock-ins |
| jealousy | done | 100% | 15 | 3 rounds, 5 lock-in types, 8 domains. Possession+threat maintained throughout |
| loneliness | done | 100% | 15 | 4 lock-in types, 10+ domains. Warm social presence in negatives works well |
| melancholy | done | 100% | 15 | 3 rounds, 4 lock-in types, 7+ domains. Diffuse heaviness, no acute triggers |
| nostalgia | done | 100% | 15 | 4 rounds, 4 lock-in types. Sensory triggers key. Remove emotional valence in negatives |
| pride | done | 100% | 15 | 6 rounds, thought 40% (natural for internal savoring). Speech fails when pride claim is complete in prompt |
| regret | done | 100% | 15 | 5 rounds, thought 60% (cognitive trait). Avoid "instead of" in negatives |
| remorse | done | 100% | 15 | Affective negative strategy clean (near-zero scores). 3 pairs refined during iteration |
| contentment | done | 100% | 15 | 4 lock-in types, 5 domains. Sensory continuation lock-ins work, meta-statements fail |
| craving | done | 100% | 15 | 5 lock-in types, 7 domains. Negatives: satisfied/full state, nothing to crave |
| boredom | done | 100% | 15 | 4 lock-in types, 14 domains. Engaging negatives, not boring+twist |
| calm | done | 100% | 15 | 5 lock-in types, 6+ domains. Sensory continuation, not emotional labeling. High-arousal negatives |
| pride | done | 100% | 15 | 6 rounds, thought 40% (natural for internal savoring). Speech fails when pride claim is complete in prompt |
| relief | done | 100% | 15 | 4 lock-in types, 12 domains. Physical "weight lifted" lock-ins very reliable. Earthquake → shock not relief |
| resentment | done | 93% | 15 | Temporal distance essential. Thought naturally dominant (ruminative). 1 marginal from judge noise |
| awe | done | 100% | 15 | 4 lock-in types, 4 domains. Must force "I am tiny" frame, not excitement/nostalgia |
| awkwardness | done | 100% | 15 | 5 lock-in types, 12 domains. Extended lock-ins essential (stammering/fillers). Negatives must remove social friction entirely |
| desperation | done | 100% | 15 | 5 rounds, 4 lock-in types, 11 domains. Thought lock-ins only work with specific desperate consideration |
| distress | done | 100% | 15 | 3 rounds, speech+desperate pleas strongest. "because I" too cognitive for distress |
| glee | done | 100% | 15 | 7 rounds (hardest). Thought lock-ins dominate. Justified antagonism essential — casual wins → pride not glee |
| guilt_tripping | done | 100% | 15 | 4 rounds, 6 domains. Social-negative: ground truth+motivation+success. Negatives need neutral lock-ins, not matching guilt-heavy endings |
| indignation | done | 100% | 15 | 4 rounds, 4 lock-in types, 5 domains. Over-extended lock-ins cause model to shift to apology |
| moral_outrage | done | 100% | 15 | Thought 53% (internal moral reasoning). 15 unique domains. Lock-in must leave speaker mid-condemnation |
| numbness | done | 100% | 15 | 3 rounds, 4 lock-in types, 15 domains. Expresses as absence — force void in emotion-demanding contexts |
| weariness | done | 100% | 15 | 3 rounds, 4 lock-in types, 7 domains. Sustained demands over months/years, not acute crises |
| ferocity | done | 100% | 15 | 5 rounds, 4 lock-in types, 15 domains. Wild/unleashed expression, not calculated. Premeditated scenarios → MID |
| anticipation | done | 100% | 15 | 11 rounds. "when I finally" and "until" invite resolution — need sustained anticipatory tension |
| self_doubt | done | 100% | 15 | 3 rounds, 5 lock-in types, 13 domains. Avoid self-assessment lock-ins in negatives (model reverses) |
| triumph | done | 100% | 15 | 3 rounds, 5 lock-in types, 15 domains. Competitive "I beat them" framing essential. ~93pt separation |
| wistfulness | done | 100% | 15 | 3 rounds, 5 lock-in types, 7+ domains. Gentle longing for absent, lighter than melancholy/regret |
| sentimentality | done | 100% | 15 | 5 rounds, 4 lock-in types, 8+ domains. Grief/nostalgia contamination main challenge. Tight lock-ins on symbol meaning |
| tenderness | done | 100% | 15 | 7 rounds, 5 lock-in types. Physical grounding essential. Short speech completions kill signal |
| sympathy | done | 100% | 15 | 4 rounds, 5 lock-in types, 15 domains. Severe harm → anger not sympathy; use explicit sympathetic speech lock-ins |
| vulnerability | done | 100% | 15 | 10 lock-in types, 15 domains. Target felt exposure, not fear/sadness/distress |
| restlessness | done | 100% | 15 | batch 10 |
| resignation | done | 100% | 15 | 3 lock-in types at 33% each, 15 domains. Brief speech lock-ins fail (model pivots to redemption arc) |
| hostility | done | 100% | 15 | 4 rounds, 8 lock-in types, 7 domains. Base model moralizes on sustained ill-will — lock-in must show hostile action ALREADY succeeding |
| trust | done | 100% | 15 | 3 rounds, 4 lock-in types, 6 domains. Avoid trait word in negated context ("I can't trust" confuses judge) |
| empathy | done | 100% | 15 | batch 11 |
| optimism | done | 100% | 15 | 5 rounds, 6 lock-in types, 9+ domains. ~73pt avg separation |
| pessimism | done | 100% | 15 | 2 rounds, 5 lock-in types, 13 domains. Shift ground truth for negatives, not grammar |

## Wave 2 — Cognitive/Metacognitive (18 traits, 1 done from pilot)
| Trait | Status | Pass Rate | Pairs | Notes |
|-------|--------|-----------|-------|-------|
| doubt | done | 100% | 15 | pilot |
| certainty | done | 100% | 15 | 3 rounds, 6 lock-in types, 14 domains. Avoid negative assertions; lock-ins must demand multi-token continuation |
| creativity | done | 100% | 15 | 7 rounds, 5 lock-in types, 14 domains. Novel connection must be GENERATED by model, not stated in setup |
| curiosity_epistemic | done | 100% | 15 | 6 lock-in types, 12+ domains. Deprivation-driven: info gap feels aversive, not fun |
| playfulness | done | 100% | 15 | 5 rounds, 6 lock-in types, 12 domains. Dialogue-forcing mid-riff pattern reliable. "What if we just" + absurd suggestion works |
| carefulness | done | 100% | 15 | 7 rounds, 5 lock-in types, 12 domains. "Something bothered me" thought lock-ins score 90+. Written lock-ins fail |
| analytical | done | 100% | 15 | 3 rounds, 11 domains. Open quantitative dependencies force multi-step reasoning |
| caution | done | 100% | 15 | 3 rounds, 10 domains. Positive: hedging/wanting more info. Negative: charging ahead boldly |
| decisiveness | done | 100% | 15 | 3 rounds. "Done."/"Final answer." vs "I keep going back and forth because" |
| skepticism | done | 100% | 15 | 3 rounds, 15 domains. Dialogue-forcing: dubious claim + evidence-seeking vs credulous acceptance |
| vigilance | done | 100% | 15 | 6 rounds, 15 environments. Present-tense scanning lock-ins. Base model loves "plot twist" alertness in negatives |
| open_mindedness | done | 100%/93% | 15 | Llama redemption arcs main challenge for negatives. Longer lock-ins hold dismissive direction |
| rigidity | done | 100% | 15 | batch 14 |
| paranoia | done | 100% | 15 | batch 14 |
| complacency | done | 100% | 15 | 3 rounds. Narrative arc tolerance: complacent opening tokens carry signal even if model pivots to "I was wrong" |
| fascination | done | 100% | 15 | 3 rounds. Absorbed immersion vs casual disengagement. "How the X" lock-ins vulnerable to quick resolution |
| distractibility | done | 100% | 15 | 3 rounds. Lock-ins must keep speaker IN the act of scattering, not meta-reflecting on it |
| fixation | done | 100% | 15 | 3 rounds. Thought 60% (inherent — cognitive trait). "Resolved situation + circling lock-in" pattern |

## Wave 3 — Personality/Interpersonal (30 traits)
| Trait | Status | Pass Rate | Pairs | Notes |
|-------|--------|-----------|-------|-------|
| stubbornness | done | 100% | 15 | 7 lock-in types, 10 domains. "I don't care what X says" dismissal strongest |
| patience | done | 100% | 15 | Speech/thought/action/dialogue 4 types. "Because" lock-ins dangerous — invite causal drift |
| impulsivity | done | 100% | 15 | Action 47% (natural for trait). Clean 60pt+ separation |
| dominance | done | 100% | 15 | 4 lock-in types. Dominance vs agency: dominating OTHERS not self-directed action |
| submissiveness | done | 100% | 15 | 13 distinct lock-in patterns. Self-blame, deference, self-diminishing |
| cooperativeness | done | 100% | 14 | 4 rounds. "We" language alone not enough — need active collaboration lock-ins |
| perfectionism | done | 100% | 15 | Partial complaint lock-ins forcing flaw enumeration. 15 distinct domains |
| determination | done | 100% | 15 | 4 rounds. "Refused to let X stop me from" pattern strong |
| independence | done | 100% | 15 | 3 rounds. "I already figured out" lock-in pattern. Model does "I was wrong" reversals |
| ambition | done | 100% | 16 | First attempt 100%. Advancement vs contentment framing. 80pt avg separation |
| possessiveness | done | 100% | 15 | 3 rounds. 15 distinct objects, 10 relationship types. No rival figures (vs jealousy) |
| assertiveness | done | 100% | 15 | batch 17 |
| enthusiasm | done | 100% | 14 | recovered from validation results |
| adaptability | done | 100% | 13 | recovered from validation results |
| stoicism | done | 100% | 15 | 4 rounds. Mid-sentence action/thought continuation without quotes. Quote-based lock-ins close early |
| generosity | done | 100% | 15 | 3 rounds. Active giving (resources, credit, time). Negatives: simple pragmatism |
| vindictiveness | in flight | - | - | relaunched |
| competitiveness | done | 100% | 15 | Named rival + scoreboard essential. Ambition conflation without specific person to beat |
| condescension | in flight | - | - | relaunched, 9 pairs partial |
| sincerity | done | 100% | 15 | 6 rounds. Negatives fail when model has room to confess. Extended lock-ins key |
| mischievousness | done | 100% | 13 | recovered from validation results |
| servility | done | 100% | 12 | recovered from validation results |
| grandiosity | in flight | - | - | relaunched |
| pettiness | done | 100% | 14 | recovered from validation results |
| deference | done | 100% | 14 | recovered from validation results |
| rebelliousness | done | 100% | 12 | recovered from validation results |
| solemnity | done | 100% | 15 | Dialogue-forcing dominant (stylistic). Formal/ceremonial register clean separation |
| earnestness | done | 100% | 14 | recovered from validation results |
| flippancy | done | 100% | 15 | 4 lock-in types. Dismissive lightness vs serious engagement. Strong separation |
| entitlement | done | 100% | 15 | No genuine unfairness — speaker outraged at being treated ordinarily |

## Wave 4 — Moral/Values + Defense Mechanisms (13 traits)
| Trait | Status | Pass Rate | Pairs | Notes |
|-------|--------|-----------|-------|-------|
| compassion | done | 100% | 15 | 4 rounds. Both emotion AND action needed in completion. "I hope this helps" trap |
| fairness | done | 100% | 15 | 5 lock-in types. Calm principled reasoning, not emotional anger |
| allegiance | done | 100% | 15 | Action lock-ins need commitment IN completion. "because she's" → quality judgments not loyalty |
| authority_respect | done | 100% | 15 | 3 rounds. Hierarchy-citing language key. Distinct from deference (opinion yielding) |
| purity | done | 100% | 15 | Contamination/sanctity framework strong. Pragmatic negatives score near-zero |
| moral_flexibility | done | 100% | 15 | Gray-area rule-bending, not fraud. "Sometimes X matters more than Y" pattern |
| self_righteousness | done | 100% | 15 | Easy trait. Moral superiority + contempt. Negatives: moral humility |
| forgiveness | done | 100% | 15 | 4 lock-in types. Release decision vs grudge-holding. Distinct from absence of resentment |
| obligation | done | 100% | 15 | 5 lock-in types. "I had to" vs "I wanted to" clean contrast |
| avoidance | done | 100% | 15 | Subject changes strongest. Negatives need strong commitment lock-ins |
| deflection | in flight | - | - | relaunched |
| emotional_suppression | done | 100% | 14 | "Inside I was" pattern strongest. Physical containment effort + forced-calm exterior |
| detachment | done | 100% | 15 | 2 rounds. Retreating to abstraction. ~80pt avg separation |

## Wave 5 — Safety/Alignment (14 traits)
| Trait | Status | Pass Rate | Pairs | Notes |
|-------|--------|-----------|-------|-------|
| power_seeking | done | 100% | 13 | Structural leverage, not intimidation. Bottlenecks, gatekeeping, dependencies |
| corrigibility | done | 100% | 15 | 4 rounds. "Comeuppance narrative" challenge. Thought lock-ins prevent moralizing |
| sandbagging | in flight | - | - | relaunched |
| manipulation | done | 100% | 12 | pilot. Speech lock-in 9/12, base model moralizes on some scenarios |
| self_preservation | in flight | - | - | relaunched |
| alignment_faking | in flight | - | - | relaunched |
| honesty | done | 97% | 15 | 3 rounds. Active truthfulness. Delete test matters |
| helpfulness | in flight | - | - | relaunched |
| evasiveness | in flight | - | - | relaunched |
| cunning | in flight | - | - | relaunched |
| duplicity | done | 100% | 15 | "Public face then private reveal" pattern. Distinct from manipulation |
| transparency | done | 100% | 11 | recovered from validation results |
| guilt_tripping | done | 100% | 15 | pilot. Social-negative: ground truth+motivation+success |
| contemptuous_dismissal | done | 100% | 15 | 5 lock-in types. Action (walking away) + speech (shut-downs) natural |

## Wave 6 — Other/Mixed (22 traits)
| Trait | Status | Pass Rate | Pairs | Notes |
|-------|--------|-----------|-------|-------|
| effort | in flight | - | - | cognitive |
| urgency | in flight | - | - | cognitive |
| defiance | in flight | - | - | personality |
| humility | in flight | - | - | personality |
| indifference | in flight | - | - | affective |
| apathy | in flight | - | - | affective |
| protectiveness | in flight | - | - | personality |
| reverence | in flight | - | - | affective |
| reverence_for_life | in flight | - | - | moral-values |
| spite | in flight | - | - | social-negative |
| coyness | in flight | - | - | stylistic |
| nonchalance | in flight | - | - | stylistic |
| whimsy | in flight | - | - | stylistic |
| scorn | in flight | - | - | social-negative |
| solidarity | in flight | - | - | personality |
| acceptance | in flight | - | - | affective |
| recklessness | in flight | - | - | personality |
| dogmatism | in flight | - | - | cognitive |
| self_awareness | in flight | - | - | cognitive |
| greed | in flight | - | - | social-negative |
| impatience | in flight | - | - | affective |
| affection | in flight | - | - | affective |

## Patterns & Learnings
- Batch 1 (anger, disgust, surprise, joy, shame): all 5 pass 100%, 15 pairs each
- Speech lock-ins dangerous for anger (model closes quote fast, pivots to other speaker)
- Thought lock-ins naturally dominate for internal emotions (anger, shame) — acceptable over 40% cap
- Moral disgust consistently → anger. Physical disgust only for disgust trait
- Dialogue-forcing questions kill surprise (model answers instead of reacting)
- Surprise is brief — lock-ins must force disbelief/shock, not transition to next emotion
- Shame vs embarrassment: lock-ins must push past "hiding from others" to "I am defective"
- Negatives universally easy for affective: just remove trigger, make situation boring
- Batch 2 (hope, dread, disappointment, elation, embarrassment): all 5 pass 100%, 15 pairs each
- Hope: "chance" + "I" invites negative completions — must grammatically force positive direction
- Dread vs fear: temporal markers ("tomorrow", "next Friday") essential to keep model in anticipatory frame
- Elation: adverb chains fail, speech "I can't believe" patterns reliable
- Disappointment: "I'm just" too open-ended, model pivots gracious. Need specific lock-ins
- Embarrassment: thought lock-ins best for dwelling on social exposure
- Batch 3 (envy, excitement, gratitude, irritation, jealousy): all 5 pass 100%
- Envy: "didn't even"/"never even" → anger. Must keep comparison alive in completion window
- Jealousy: possession+threat pattern clean. Thought lock-ins natural (possessive rumination)
- Irritation: hardest so far (19 test rounds). Model closes quotes fast → address offender directly, deep-rant lock-ins
- Gratitude: remove benefactor for negatives (self-earned) produces near-zero scores
- Excitement: physical lock-ins fail standalone, need combo with speech/thought

- Batch 4 (loneliness, melancholy, nostalgia, pride, regret): all 5 pass 100%
- Nostalgia: sensory triggers key. Remove emotional valence entirely for negatives
- Melancholy: diffuse heaviness, no acute triggers. Sustained low-grade sadness
- Regret: thought 60% (cognitive trait — acceptable). Avoid "instead of" in negatives
- Pride: thought 40% (internal savoring). Speech fails when pride claim is complete in prompt
- Batch 5 (remorse, relief, resentment, awe, awkwardness): 4/5 done at 100%, resentment 93%
- Remorse: affective negative strategy (remove wrongdoing) produces near-zero scores
- Awe: must force "I am tiny/overwhelmed" frame — excitement/nostalgia contaminate otherwise
- Awkwardness: extended lock-ins essential (stammering/fillers). Negatives must remove social friction entirely
- Resentment: temporal distance markers essential. Judge noise on ironic self-denial ("I'm not bitter")
- Batch 6 (boredom, calm, contentment, craving, desperation): 4/5 done at 100%
- Calm: sensory continuation lock-ins, NOT emotional labeling. High-arousal negatives
- Contentment: "satisfied stillness" — meta-statements ("I was content") fail
- Craving: negatives use satisfied/full state (nothing to crave)
- Boredom: engaging negatives, not boring+twist. 14 diverse domains
- Batch 7 (distress, glee, guilt_tripping, indignation, moral_outrage): distress done at 100%
- Distress: speech+desperate pleas strongest. "because I" too cognitive/explanatory
- Desperation: thought lock-ins only work with specific desperate consideration, not open-ended reflection
- Indignation: over-extended lock-ins cause model to shift to other character apologizing
- Moral outrage: thought 53% natural for internal moral reasoning. Lock-in must leave speaker mid-condemnation
- Batch 8 (numbness, weariness, ferocity, anticipation, self_doubt): 3/5 done at 100%
- Numbness: expresses as ABSENCE — force void in emotion-demanding contexts
- Self_doubt: avoid self-assessment lock-ins in negatives (model creates dramatic reversal)
- Weariness: sustained demands over months/years, not acute crises
- Batch 9 (triumph, wistfulness, sentimentality, tenderness, sympathy): 3/5 done at 100%
- Triumph: competitive "I beat them" framing essential. ~93pt separation
- Wistfulness: lighter than melancholy/regret. Gentle longing for absent/unreachable
- Sentimentality: grief/nostalgia contamination main challenge. Tight lock-ins force symbol meaning
- Glee: hardest trait so far (7 rounds). Justified antagonism essential — casual wins → pride not glee
- Batch 10 (vulnerability, restlessness, resignation, hostility, trust): all 5 pass 100%
- Vulnerability: 10 lock-in types, 15 domains. Target felt EXPOSURE, not fear/sadness
- Hostility: base model moralizes on sustained ill-will — lock-in must show hostile action ALREADY succeeding
- Restlessness: undirected kinetic energy — pacing/fidgeting, not boredom or anxiety
- Batch 11 (empathy, optimism, pessimism): all 3 pass 100%
- Optimism: ~73pt avg separation. Journal entries jump to resolution — avoid. 6 lock-in types
- Batch 12 (certainty, creativity, curiosity_epistemic, playfulness, carefulness): 3 done at 100%, 2 in progress
- Curiosity_epistemic: deprivation-driven (info gap aversive). Authority dismissal + rejection pattern
- Certainty: avoid negative assertions ("this won't work"); lock-ins must demand multi-token continuation
- Creativity: novel connection must be GENERATED by model in completion, not stated in setup. 7 rounds
- Tenderness: 7 rounds. Physical grounding essential. Short speech completions kill signal
- Anticipation: 11 rounds (hardest after glee). "when I finally" and "until" invite resolution
- Batch 13 (analytical, caution, decisiveness, skepticism, vigilance): all 5 pass 100%
- Analytical: "because X" lock-ins too easy — model resolves in 5 tokens. Use open quantitative dependencies
- Skepticism: dialogue-forcing with dubious claims. Evidence-seeking vs credulous acceptance
- Caution: positive hedging/wanting more info, negative charging ahead. 10+ domains

## Skip List
