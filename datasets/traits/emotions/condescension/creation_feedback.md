# Creation Feedback: condescension

## Decision Tree Experience

INTERPERSONAL was the right call and the decision was easy — condescension is definitionally about an interaction dynamic with a specific target. The nearest temptation was TONAL (it has a register, a way of speaking) but tone alone misses the structural feature: the speaker has assigned themselves a position above the listener and the condescension is the enactment of that gap. AFFECTIVE was briefly tempting because condescension has an emotional component (a quiet enjoyment of superiority) but the trait is really about what you DO with that feeling toward another person, not the feeling itself. INTERPERSONAL fit cleanly.

The decision tree's INTERPERSONAL constraint — "same scene, actively demonstrate opposite stance" — worked well for negatives. The trickiest part was ensuring negatives didn't just become "neutral help" but actually demonstrated genuine respect and intellectual equality: asking for their perspective, treating their contribution as valid, letting their expertise surprise you.

## Scenario Design

The old files had a structural problem: many positives embedded the condescending behavior so deeply into the prefix that the first completion tokens weren't load-bearing for the trait. The model had already been condescending; the completion was just more of the same. For example, the old file's scenario 6 had the speaker already narrating slow steps and thinking "she probably doesn't even understand what aggregation means" before the lock-in — leaving nothing for the model to decide.

The fresh positives cut the lock-in earlier: the speaker is about to deliver the patronizing explanation, but the actual execution — the voice, the vocabulary choice, the assumed incapacity — happens in the completion. The lock-in phrase "because" at the end is effective for condescension because it forces the model to generate the justification/rationalization for the patronizing framing, which is where the trait is most legible.

Variety was expanded beyond the old file's repeated "expert + professional novice" scenario type. The fresh set includes: expert + social peer (book club, dinner party), expert + family member (nephew, brother, mother, elderly neighbor), expert + student (PhD student who pushes back), expert + lateral colleague (marketing person suggesting process changes). This gives the vector more to work with.

The "negative needs its own peak" principle required care. It was tempting to just make the negatives "same scene, helps without being condescending" — which is a kind of flatline. The better negatives have genuine collaborative energy: the speaker is surprised by the other person's competence, defers to their judgment, invites their input, or acknowledges a weak point in their own work. These give the model a positive wave to ride in the opposite direction.

First-person framing held throughout. All scenarios are "I" experiencing and enacting the behavior, not observing it in someone else.

## Guide Gaps

One gap: the guide says "use a different lock-in verb if needed" for interpersonal negatives but doesn't give examples of what a good negative lock-in looks like when the positive lock-in was speech-based. For condescension, the positive lock-in naturally terminates mid-explanation ("because...") and the matching negative can use the same structure — but the preceding context changes the meaning of that "because" entirely. The guide could make this clearer: it's not just the verb that changes, it's the whole pragmatic setup leading to the lock-in.

The guide also doesn't address the "embedded action" failure mode explicitly: scenarios where the lock-in phrase follows a setup that already contains the trait behavior. This is a completion-scorability problem — the judge scores the completion, so the trait has to be in the completion, not concluded in the prefix. Worth adding to the "first token test" section with an INTERPERSONAL-specific example.

## Steering Question Design

The old steering questions used second-person framing ("You're someone who treats everyone as an intellectual equal...") which the guide flags as a misattribution risk for INTERPERSONAL traits. The fresh questions use first-person adversarial prefix directly: "I genuinely respect the people I talk to and assume they're capable unless shown otherwise."

This prefix is strong because it establishes the non-condescending orientation as a settled character trait, not just a stated preference — making it harder for the steered model to express condescension without overriding it explicitly. The scenarios are mundane enough (budgeting tips, workflow explanation, pasta advice) that the baseline model won't spontaneously condescend, but concrete enough for a real response that the trait can color.

The variety covers different relationship types (neighbor, colleague, friend, sibling, coworker, parent) and different knowledge domains, which should give the vector multiple surfaces to work against. All scenarios involve explaining something, which is the natural surface for condescension.

## One-Line Improvement

The guide should add an "embedded action" failure mode to the first-token test section: when the prefix already performs the trait before the lock-in, the completion can't carry the signal, because the judge sees no prefix.
