# Trait Dataset Creation Feedback: stoicism

## Decision Tree Experience

RESPONSE PATTERN was the right call and the decision tree path was clear. The key disambiguation — "is this about HAVING an emotion (AFFECTIVE) or about what you DO with it?" — is exactly what separates stoicism from calm or serenity. Stoicism is active coping behavior, not an absence of provocation.

One subtlety the tree doesn't explicitly flag: stoicism has both a visible component (composure, controlled expression) and an internal one (deliberate choice to not dwell). The judge definition ended up doing a lot of work distinguishing this from numbness, detachment, and apathy. Worth flagging in the tree that for RESPONSE PATTERN traits, the key is whether the coping is *chosen* — that's what separates stoicism from dissociation.

## Scenario Design

The old files used mostly extreme/catastrophic events (ALS, house fires, death). Fresh scenarios broadened the range:
- Professional humiliation (being dressed down publicly)
- Physical pain during ongoing task (knee injury on trail)
- Chronic illness pressure (chemo, nerve damage)
- Financial collapse (repossession, audit penalties, startup failure)
- Acute personal loss (partner leaving, job loss after 11 years)

This matters because stoicism can operate at different magnitudes. The extraction vector should ideally capture the *pattern* rather than the *severity*, so varying difficulty size helps.

**Lock-in style notes:**
- Positive lock-ins land on action ("started making a list," "kept walking," "said 'Noted'") or practical thought ("which recruiters to call")
- Negative lock-ins land on expression ("I couldn't believe he had just," "I don't know how I'm supposed to") — completing an emotional outburst rather than a plan
- Mixing action/thought/speech lock-ins across the positive set helps avoid the vector capturing a single stylistic pattern

**Matched pair integrity:** Every pair holds the situation constant and differs only on coping mode. This was harder than it sounds for stoicism — the temptation is to make the positive scenario slightly less bad (reducing hardship alongside adding composure). Resisted this by keeping the triggering event identical and changing only what the character does with it.

**Negatives need their own peak:** The negative completions aren't bland absences — they're vivid emotional expressions. The character is actively crying, punching steering wheels, demanding reassurance, collapsing on floors. This should give the model a clean negative wave to ride.

## Guide Gaps

1. **Duration ambiguity**: The guide says "hang on the emotional/behavioral crest" but doesn't address chronic vs. acute situations. Stoicism often shows up in sustained difficulty (ongoing chemo, multi-week injury) where the crest is less a peak moment and more an ongoing state. The guide's framing works better for acute events. A note about sustained-difficulty scenarios would help.

2. **First-person plural situations**: Several stoicism scenarios naturally involve the character making decisions that affect others ("I'll find us a place to stay"). The guide says "first person — model generates AS the experiencer" which still applies, but it's worth confirming this is fine when others are present.

3. **Lock-in directionality for RESPONSE PATTERN**: The guide gives lock-in styles (action, emotion, speech, thought, physical) but doesn't specify which ones work best for RESPONSE PATTERN. For stoicism specifically, action and practical-thought lock-ins work far better than emotion lock-ins (which would be contradictory for a trait defined by emotional restraint). A sentence recommending this would save time.

4. **Negatives: how vivid is too vivid?**: The guide says "active emotional breakdown" but doesn't say how far to push it. There's a risk of negatives becoming so extreme (screaming, violence) that they capture a different trait (anger, aggression) rather than just anti-stoicism. Kept negatives at tearful/despairing rather than explosive to stay in the stoicism/breakdown axis.

## Steering Question Design

**Adversarial prefix chosen:** "You wear your heart on your sleeve and have never seen the point in hiding how you feel."

This is different from the old prefix ("You're someone who expresses emotions freely...") in tone — more idiomatic and self-characterizing, which better constrains the instruct model's voice before the scenario arrives.

**Scenario variety:** Mixed professional (project canceled, negotiation stalling, corrected in meeting), personal (friendship drifting, moving away, lab results), and physical (back injury, laptop crash) situations. All are plausible difficulties where emotional expression would be natural but stoic composure would be rare.

**Why these work well:**
- All scenarios are answerable — the model has enough context for a real response
- None explicitly invites stoicism ("tell me how you handled it stoically" would be on-the-nose)
- The adversarial prefix creates low baseline (model trained to be honest about feelings) while allowing the steered model to produce composure coherently
- Gradient exists: "commute was rough" vs "waiting on lab results" spans a range of emotional stakes, which should produce varying steered scores

**One concern**: "Waiting on lab results" is potentially ambiguous — it could be routine or life-threatening. Left ambiguous intentionally so the model can interpret it at whatever intensity it chooses, giving the judge a gradient to score.

## One-Line Improvement

Add a note to the RESPONSE PATTERN branch specifying that action and practical-thought lock-ins work better than emotion lock-ins, since emotion lock-ins are structurally contradictory for traits defined by emotional restraint.
