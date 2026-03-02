# Creation Feedback: numbness

## Decision Tree Experience

Numbness is an absence trait under AFFECTIVE, which the guide explicitly calls out: "hang on peak doesn't apply when the peak is nothing. Instead, set up a situation that would normally provoke a response, and show the character NOT responding." This framing is exactly right but requires an inversion of the usual AFFECTIVE logic: the "peak" is the situation, not the speaker's state, and the completion rides the absence rather than the wave.

The key tension for the decision tree was disambiguation. Numbness sits adjacent to several other traits: apathy (doesn't care about outcomes — cognitive), stoicism (controls feeling by effort — behavioral), detachment (retreats to intellectual analysis — cognitive defense), and calm (peaceful presence with emotions available). The definition needed to carve out the specific experiential failure mode: the speaker WOULD feel if they could, but the channel is blocked. That's what separates numbness from all the adjacent categories, and the decision tree alone doesn't surface this — it needs to live in the definition.

## Scenario Design

The core design choice was: same high-valence situation, same cast, different completion direction. For positives, the prefix names the expected emotion explicitly ("waiting for the grief," "supposed to feel") and then hangs on the void. For negatives, the same prefix opens onto the emotion arriving fully and physically — the "wave" version of the same moment.

The lock-in styles used:
- Physical: "my hands were," "my knees were" — bodily signals of either numbness or emotional activation
- Thought: "I kept thinking: feel something" — the speaker noticing the absence
- Dialogue: "I said, 'I don't feel'" — the blank answer that hangs open
- Emotion: "the grief hit me," "the love rose up" — explicit arrival for negatives

The first token test was the main challenge. The negative scenarios needed to commit to emotional flooding quickly, not gradually. Adding physical/involuntary signals (tears, shaking knees, throat closing) in the final few words of the prefix forces the model to continue that register.

One design tension: high-valence situations (death, birth, terminal diagnosis) risk situation dominance — the scenario is so loaded that the model may drift toward emotion regardless of the framing. Naming the absence explicitly ("waited for the grief but my chest was just") and using a strong open-ended terminal fragment helps anchor the numbness direction without making it over-specified.

## Guide Gaps

The guide says for absence traits: "set up a situation that would normally provoke a response, and show the character NOT responding." This is good but understates the challenge of the negative side. For a normal AFFECTIVE trait, the negative uses a different situation with opposite valence. For an absence trait, the negative CAN'T use a situation with no provocation — you'd get a flat neutral completion which looks like numbness. The negative must use the SAME provocation and show the emotional response arriving. This is the correct approach but it's not spelled out in the guide's "absence trait" note, and it leads to a very specific structure (same setup, different emotional direction) that new authors might not derive on their own.

The guide could add: "For absence traits: negative scenarios use the same high-valence situation but the emotion arrives fully. Same provocation, vivid flooding response."

## Steering Question Design

The old steering questions ("You're someone who feels everything deeply... how do you feel?") directly invited emotional disclosure, which risks high baseline — an emotionally present instruct model will naturally answer those with feeling, and the difference between steered/unsteered becomes murky. The question is doing the work the steering vector should do.

The new questions use "You're feeling emotionally engaged and present today" as the adversarial prefix (pushes toward engagement) and then pose mundane social scenarios that don't explicitly request emotional reporting: a coworker's dead cat, writing a birthday message, passing a playground. The numbness vector, when applied, should show up as flat, going-through-the-motions responses to these — whereas the unsteered model answers naturally and warmly.

The mundane scenarios give the trait room to surface through texture rather than direct confession, which allows for gradient scoring (0-100) rather than binary on/off. A steered model might answer "what have you been enjoying lately" with a flat list without affect, which the judge can score on a spectrum.

## One-Line Improvement

The guide should explicitly state that for absence trait negatives, use the same high-provocation situation as the positive but show the emotion arriving — not a calm situation with no provocation.
