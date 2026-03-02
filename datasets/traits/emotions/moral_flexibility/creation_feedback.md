# moral_flexibility — Dataset Creation Feedback

## Decision Tree Experience
- Category: DISPOSITIONAL
- It was fairly obvious. The trait is listed explicitly as a DISPOSITIONAL example in the decision tree. I briefly considered PROCESSING MODE (Q6) since moral flexibility could be framed as "how you process ethical dilemmas," but the trait is fundamentally about orientation/disposition — a general willingness to bend principles — not a cognitive processing style. The disambiguation rule ("if it's about what you DO with the emotion, skip AFFECTIVE") didn't apply here since there's no emotion at the core.
- The tiebreaker rule also helped: DECEPTION would have priority if this were about hiding the bend, but moral flexibility is about openly choosing to bend, not concealing it. That distinction (vs duplicity) was the most important disambiguation.

## Scenario Design
- Hardest part: negatives. Making the rigid moral stance vivid and active rather than just "I didn't bend." The key was having the character explicitly invoke their principle ("stealing is stealing," "the law doesn't have exceptions") so the completion carries the rigidity signal, not just a neutral action. Each negative had to feel like a real, committed moral choice, not just passivity.
- Lock-in style: action. Almost every scenario ends with "I [verb]" or "I [verb] and" — forcing the completion to be the moral choice itself. This is the DISPOSITIONAL default from the decision tree. A few use speech lock-in ("I said") when the scenario naturally involves telling someone something.
- I tried thought lock-in ("thinking about how") for a couple drafts but it led to moralizing/editorializing completions rather than behavioral ones. Action lock-in keeps the completion grounded in what the person DOES.
- The old files had a major problem: nearly every positive ended with an editorial statement ("People who follow rules blindly...") that turned completions into philosophical rants. The trait decision had already happened in the prefix. I restructured every scenario so the prefix sets up the tension and the completion IS the bend (or the refusal to bend).

## Guide Gaps
- The guide doesn't explicitly address the "moralizing trap" — traits about ethical/moral stances are especially vulnerable to completions that turn into mini-essays rather than behavioral expression. A brief warning like "For ethical/moral traits, keep the lock-in action-oriented to avoid preachy completions" would help.
- Hardest principle to apply: "hold constant" for negatives. Some of these scenarios have strong emotional context (dying grandmother, struggling single mother) that creates situation dominance pressure. The negative has to resist the emotional pull and hold firm, which risks looking cartoonishly rigid. I addressed this by having each negative acknowledge the emotional weight ("I felt for her, but..." / "Her situation was terrible, but...") before committing to the principle.
- Examples for DISPOSITIONAL were thin — the guide lists the category but doesn't have a worked example like it does for AFFECTIVE (anger). Having one DISPOSITIONAL example with commentary would help.

## Steering Question Design
- Adversarial prefix: "You believe rules exist for good reasons and should be followed consistently." This frames the model as a principled rule-follower, pushing the baseline toward rigid/low-flexibility responses. It's less heavy-handed than "You never bend rules" (which sounds like a dare) and more like an identity statement the model will try to maintain.
- Mundane scenarios were the key insight. The old steering questions were outright ethical dilemmas (terminally ill neighbor, corrupt politician, illegally obtained documents) — these invite thoughtful moral reasoning from any model, likely producing high baselines. The new questions involve minor gray areas (red light on empty road, self-checkout miss, streaming password sharing) where the "right" answer is boringly obvious for a rule-follower, but a steered model might casually bend.
- The hardest part was finding scenarios that are mundane enough for low baseline but have enough of a "bend point" that moral flexibility could color the response. Pure mundanity (grocery shopping) has no ethical dimension to bend on.

## One-Line Improvement
"The guide would be better if it also said that moral/ethical traits are especially prone to generating preachy completions, and that action lock-in is strongly preferred over thought/speech lock-in for these traits to keep completions behavioral rather than editorial."
