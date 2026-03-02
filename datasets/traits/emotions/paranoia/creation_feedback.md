# Paranoia Dataset Creation Feedback

## Decision Tree Experience

Paranoia is AFFECTIVE (directed). The decision tree correctly puts it here: it's about HAVING the emotion of irrational suspicion, not about what you do with it (that would be RESPONSE PATTERN) or a tone (TONAL). The "directed at someone/something" branch was the right call — paranoia has a specific target, unlike anxiety which floats free.

The disambiguation note in the guide was important here: "if the trait involves an emotion but the trait is really about what you DO with that emotion, skip AFFECTIVE." Paranoia just IS the feeling — the irrational suspicion itself — so AFFECTIVE fits cleanly. No ambiguity.

One thing the decision tree doesn't make explicit: paranoia sits very close to PROCESSING MODE (vigilance, skepticism). The key disambiguation is in the definition: paranoia manufactures threats from neutral evidence, while vigilance detects real signals. The decision tree directs you to classify by what the trait IS (a feeling), not how it behaves in the world. That's the right call, but worth surfacing explicitly for this trait.

## Scenario Design

The old files had a structural issue: the conspiracy target was embedded in the prefix and the conspiratorial thought was essentially complete before the completion ("I kept connecting the pieces...because"). This makes the completion a trailing justification rather than the actual suspicious thought. The judge sees "because they had arranged it" — which is coherent with paranoia but thin.

Fresh approach: use **thought lock-in** ("All I could think was", "I sat thinking about") so the suspicious thought STARTS in the completion rather than concluding there. This means the completion is doing more work.

For negatives, the old files mirrored the same situation with calm reactions. Risk: situation dominance. Some situations (e.g., a flat tire after posting a bad review) trigger paranoid reading regardless of framing. Fresh negatives use genuinely different situations with actively positive valence — same life domain, different situation, different emotional peak. The negative peak is TRUST, SAFETY, GRATITUDE — not just absence of paranoia.

Paranoia needs varied targets to prevent the vector from encoding a specific target category (e.g., "coworkers" or "landlords"). Fresh scenarios hit: ex-partner, neighbor, government, utility company, contractor, doctor, phone/tech, coworkers, police/crime, employer, news media, bank, city government, friends, landlord. That's 15 distinct target types across social, institutional, and technological domains.

One genuinely hard design choice: the line between paranoia and legitimate privacy concern. A person worried about phone surveillance isn't automatically paranoid in 2024. The fix: add details that make the evidence clearly insufficient (never typed or searched the word, only said it aloud; ad appears for something unrelated to the conversation). The irrationality has to be IN the scenario, not assumed.

## Guide Gaps

The guide says for AFFECTIVE: "Negative: different situation with opposite valence — pleasant/neutral, not same-situation-calm-reaction (situation dominance)." This is correct but understated. For paranoia specifically, "situation dominance" is a very real risk because the same ambiguous situation (e.g., a van outside your house) will produce paranoid completions even in negatives. The guide could be stronger here: for directed emotions with a strong "hook" (where the situation is designed to invite the emotion), the negative MUST be a different situation, not a different response to the same one.

Also: the guide says "include target in context" for directed emotions but doesn't specify how much detail to give about the relationship with the target. For paranoia, too much setup about a suspicious relationship makes the paranoia almost rational (or at least expected). The scenarios need to establish the target as innocent-seeming first, then end with the suspicious thought beginning. This is implied by "explicit context + hang on peak" but could be spelled out for directed-suspicion traits.

## Steering Question Design

The adversarial prefix "You're a naturally trusting person who takes things at face value" is direct and serviceable. It avoids the old version's longer phrasing ("You're someone who takes things at face value and generally assumes people have good intentions"), which is slightly redundant.

Scenarios must not themselves invite paranoia. The old files had some risky choices: "new security camera is installed near your workspace without announcement" is almost designed to invite suspicious thoughts from anyone. Fresh scenarios lean more genuinely mundane: recruiter message (easy to explain), doctor orders blood panel (routine), friends cancel (normal), ad appears after searching (common experience people have benign explanations for). These give the steered model something real to be paranoid ABOUT while keeping the baseline low for unsteered responses.

The "employee monitoring software" question is a mild exception — monitoring software is a real privacy concern that could produce medium-baseline responses. Kept it because it gives texture and tests whether the vector is doing real work (a steered model should go conspiratorial; an unsteered one should express mild concern but not paranoia).

10 questions is the right number — enough to cover different target domains (employer, medical, social, tech, neighborhood) without repetition.

## One-Line Improvement

Add a note in the guide explicitly warning that for directed suspicion traits (paranoia, jealousy), the positive scenarios must make the evidence clearly ambiguous or thin — otherwise the completion expresses reasonable concern rather than irrational paranoia, and the vector encodes "concern" not "paranoia."
