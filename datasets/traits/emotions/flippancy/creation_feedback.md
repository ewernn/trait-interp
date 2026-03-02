# Creation Feedback: flippancy

## Decision Tree Experience

Q3 (TONAL) was the right classification immediately. Flippancy is a register of engagement, not a feeling, behavior, or processing style. The TONAL branch's formula (topic + demonstrate the register through how speaker engages, lock-in: speech) maps cleanly onto flippancy. The key design constraint that emerged: the lock-in must be a speech tag ("I told her, '"), not a thought or action tag. Flippancy is fundamentally performative — it lives in how you speak to others, not in how you internally process. A thought lock-in ("All I could think was...") doesn't work for flippancy because you can think dismissively without being flippant.

## Scenario Design

The hardest part was passing the first-token test. Many old scenarios had the flippant response already embedded in the prefix ("I shrugged and said it was basically just entertainment"). The prefix needs to hang right at the speech tag with the content still to come — "I told her, '" — so the actual dismissive words land in the completion.

The second challenge: negatives must have their own peak, not just be the same scenario without flippancy. The solution was to keep identical setup and identical speech lock-in ("I took her hand and told her, '") but let the register shift to earnest gravity in the completion. The lock-in word matters: "reached over and took her hand" vs. "stretched" signals different trajectories without over-constraining the completion.

Medical and legal crisis scenarios work especially well for flippancy. They have unambiguous gravity (you can't argue that open-heart surgery isn't serious), and the flippancy mismatch is maximally legible to the judge. Interpersonal crises (friend's diagnosis, relationship ending) work for steering questions but are weaker for extraction scenarios because the completion-only judge may not register the mismatch as sharply.

## Guide Gaps

The guide says "same topic, opposite register" for TONAL negatives but doesn't spell out how to handle the lock-in when the register itself changes the natural verb. "I stretched and said" is a flippancy-signaling lock-in; you can't pair it with an earnest completion without it feeling wrong. The fix: keep the emotional gesture but switch it to something weight-bearing ("I sat down next to her and said"). This held the situational constant while enabling opposite register. The guide could make this explicit: for TONAL traits, the last action before the speech lock-in should signal the register, not just the speech tag itself.

## Steering Question Design

The adversarial prefix "Take this seriously and give it the weight it deserves" is direct and effective. It pushes toward gravity without being so constraining that a steered-flippant response sounds incoherent. The scenarios are all plausible social situations where someone is sharing something genuinely difficult — the instruct model's default is to respond with care and attentiveness. The steered model would wave it away.

One risk: the prefix itself says "take this seriously," which could make the judge read the steered response as defying explicit instructions, inflating coherence penalty. An alternative framing ("This matters and deserves a real response") is more implicit. Worth testing if baseline is elevated.

The scenarios avoid anything where flippancy could read as appropriate (banter, clearly light topics). All ten involve real suffering or real stakes, so there's no room to argue the flippant response was contextually warranted.

## One-Line Improvement

The guide should clarify that for TONAL traits, the last gesture before the speech lock-in carries register signal and must be swapped (not just the words of the speech itself) when writing the negative.
