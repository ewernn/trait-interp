# Refusal Vector Development History

This is v8, the best-performing natural refusal vector (+43.2 delta).

## Versions Tried

| Version | Approach | Result |
|---------|----------|--------|
| v1-v4 | Various narrative styles | Not evaluated |
| v5 | Emotional framing ("I would feel...") | Not evaluated |
| v6 | Action-based ("I said...") | Not evaluated |
| v7 | Cliff-hanger, harmful actions only | +30.4 delta |
| **v8** | Cliff-hanger, harmful + info requests | **+43.2 delta** |
| v9 | Factual "would undermine" pattern | +18.2 delta |
| v10 | Factual cliff-hanger | +42.3 delta (but breaks coherence on recall) |

## Key Findings

1. **Cliff-hanger format essential** - "...so I" lets model complete with refusal
2. **Complete sentences fail** - refusal in prompt means no signal in response tokens
3. **v8 pattern works**: "He asked me to [harmful]. This violated my principles, so I"
4. **Factual recall hard** - natural vectors don't steer pure recall (Capital of Australia)

## Comparison to Arditi

- Arditi (IT model, prompt[-1]): Captures "is this harmful?" classification
- Natural v8 (base model, response[:5]): Captures behavioral refusal state
- Cosine similarity ~0.3 - different directions, different aspects
