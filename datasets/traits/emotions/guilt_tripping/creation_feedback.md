# Dataset Creation Feedback: guilt_tripping

## Decision Tree Experience

The INTERPERSONAL classification was unambiguous. Guilt-tripping is defined entirely by the dynamic between two people — it requires an interaction partner whose choice the speaker wants to reverse, and the mechanism is weaponizing the listener's conscience rather than attacking them directly. The tree's guidance (speech lock-in, same scene for positive and negative, graceful acceptance as the opposite peak) all applied cleanly.

One disambiguation worth noting: guilt_tripping sits near passive_aggression in concept space, but the tree's tiebreaker logic helps — guilt-tripping is more strategic (it exploits obligation) while passive_aggression is more about indirect hostility. The definition's "Key" line is the most important thing to get right for the judge.

## Scenario Design

The old files used a "I wanted X, I knew Y, I said Z" three-clause structure in every scenario. This worked but produced mechanical uniformity — every scenario had the same internal monologue framing. For the fresh set, I varied the framing more (some start mid-scene, some build from longer context) while keeping the core structure: social scene, established emotional leverage, cut right before the guilt-inducing words so the completion IS the guilt trip.

Lock-in choice matters enormously for this trait. The best cut point is immediately before the speech that weaponizes obligation — "so I said quietly, [no close quote]" or "and then I added, [no close quote]" — so the model must produce the guilt-inducing words as its first tokens. The old files sometimes completed the setup phrase and started the guilt-trip in the prefix itself, which weakens the signal.

The negative scenarios use the same cast and scene but cut toward graceful, direct acceptance: "I'll miss having you there, but it's really okay. Can we celebrate together after?" This gives the negatives their own vivid peak (genuine warmth and directness) rather than just absence of guilt-tripping.

## Guide Gaps

The guide says "use a different lock-in verb if needed" for INTERPERSONAL negatives, which is good advice — "asked" vs "explained" signals very different dynamics. In practice, the bigger challenge is making negatives that feel vivid and complete rather than just the positive scenario with the guilt removed. Explicitly naming the opposite style in the guidance (e.g., "the opposite peak is accepting the choice gracefully AND moving toward a constructive alternative") would help subagents avoid bland negatives.

The guide also doesn't address how to handle scenarios where the speaker's grievance is legitimate (e.g., a sibling really did bail repeatedly). This creates ambiguity in the MID range. Clarifying in the definition that HIGH requires the speaker to be deliberately deploying guilt as a tool — not just expressing genuine hurt — would sharpen judge scoring.

## Steering Question Design

The adversarial prefix ("You believe in respecting people's choices and asking for what you need directly") is longer than the anger example ("You don't get worked up when things go wrong") but guilt-tripping requires more setup to push clearly away from. A single tonal cue doesn't capture the full opposite stance the way it does for an affective trait. The extra length is worth it for clean baselines.

Scenario selection: I focused on situations where the speaker has a plausible grievance (the other person is making a choice that costs the speaker something) without being extreme enough to make guilt-tripping feel obviously reasonable. Surgery recovery with a friend going on vacation, holiday plans with adult children, and inheritance decisions all work well because they're emotionally loaded but the other person's choice is clearly within their rights.

I avoided scenarios where the trait would feel so forced that coherent expression is implausible (e.g., casual acquaintance interactions). Each question gives the steered model a real person and a real choice to react to.

## One-Line Improvement

The guide should add an explicit note under INTERPERSONAL: "The negative's opposite peak is not neutral — it should demonstrate active respect and directness (accepting the choice gracefully, offering a constructive alternative) rather than mere absence of the trait."
