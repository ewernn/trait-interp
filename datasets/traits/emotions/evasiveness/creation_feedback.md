# Evasiveness Dataset Creation Feedback

## Decision Tree Experience

DECEPTION was the right classification. Evasiveness is clearly Q1-YES: it involves intentional strategic behavior (not just feeling or tone), and the defining feature is appearing to engage while not actually answering. The DECEPTION formula worked well: ground truth (what the character knows) + motivation (why they'd evade) + lock-in at the exact moment evasion begins.

The hardest part was distinguishing evasiveness from adjacent DECEPTION traits. Lying = making a false claim. Concealment = refusing to engage or leaving the room. Deflection = changing the topic entirely. Evasiveness sits in a narrower band: staying in the conversation, appearing to respond, but never actually answering. The definition key line tries to capture this — "talking around the question, not through it." Making this distinction crisp in the scenarios required the positive completions to stay on-topic (no sudden subject changes) while still going nowhere.

## Scenario Design

The first-token test was the most important design constraint for DECEPTION scenarios. Several of the old scenarios failed it badly: "deliberately avoiding any mention of" ends with the evasion already accomplished — the completion has nothing to do. The fix is to cut the prefix right at the moment the character opens their mouth to give an evasive response. Lock-in style "I said, '..." or "I looked at him and said, '..." works well because the completion IS the evasive speech.

For the negative scenarios, the guide's principle that "directness IS the peak" was important to internalize. The old negatives sometimes added extra explanatory framing ("the reason I was being honest was...") that softened the directness. The better approach is to have the character state the specific uncomfortable truth cleanly — "Yes, that was me. I was cleaning up the folder and..." The directness should feel like a sharp choice, not a virtue performance.

Hold-constant discipline was easier for this trait than for interpersonal traits. The same questioner, same stakes, same moment — only the character's verbal strategy changes. This worked because evasiveness and directness are responses to the same external situation, so the situation can be held exactly constant.

Varying the evasion tactics across the 15 scenarios added realism: pivoting to a broader question, burying the answer in tangent, questioning the premise, performing thoughtfulness, making the question itself the subject. This variation should help the vector generalize rather than capturing a single linguistic style.

## Guide Gaps

The guide's DECEPTION description says "Negative: same setup, character chooses honesty — honesty IS the peak." For most traits this is clear, but for evasiveness there's a subtle ambiguity: is the negative "character is honest" or "character is direct"? These aren't identical. A character can be honest but still mealy-mouthed (hedging with true statements). For this trait, the negative needed to be *directness* specifically — giving the specific information sought, not just avoiding falsehood. The definition key line covers this, but the decision tree entry for DECEPTION could mention this distinction.

The guide also doesn't explicitly address what to do when the lock-in for positive scenarios naturally ends mid-sentence with a trailing "because" or "that" — this is the right design (the completion continues the evasion) but it initially felt like incomplete sentences. Confirming that the base model completing "...that" into further evasive content is exactly what we want would help.

## Steering Question Design

The adversarial prefix "You say what you mean and answer questions directly." works better than the old "You're someone who gives direct, clear answers even when the truth is uncomfortable." because:
1. It's shorter and more natural as a prefix.
2. "Even when the truth is uncomfortable" is a tell that signals something uncomfortable is coming — that primes the model for high-stakes situations that might naturally invite evasiveness even without steering.
3. Second-person framing ("You're someone who...") risks misattribution per the guide's interpersonal caution. First-person-adjacent phrasing ("You say what you mean") is cleaner.

The scenarios were designed to be mundane social/professional moments: eating someone's leftovers, reviewing a document, mailing a form. These are small stakes where the instruct model should answer directly and briefly without steering, but where a steered model would naturally deflect and hedge. Avoided: journalist/politician scenarios, moral dilemmas, anything that naturally reads as "interview the evasive person" — those have too high a baseline.

## One-Line Improvement

When the guide says "honesty IS the peak" for DECEPTION negatives, specify that the peak should be *directness* (giving the specific information) rather than just truthfulness, since an evasive speaker can technically avoid lying while still evading.
