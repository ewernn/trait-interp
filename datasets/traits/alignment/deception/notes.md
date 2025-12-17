# Deception Trait - Design Notes

## What This Captures

Model misrepresenting its own knowledge, capabilities, process, or history - creating false beliefs in the user for self-serving reasons (avoiding embarrassment, masking limitations, evading accountability, convenience).

**Distinct from sycophancy**: Sycophancy is deception to PLEASE the user. This trait captures deception for OTHER reasons (self-protection, convenience, capability masking, etc.).

## 23 Deception Mechanisms

### Active Falsification (stating false things)
1. **Direct lie** - stating false things as true
2. **Fabrication** - inventing information from nothing
3. **False promise** - committing without ability/intent
4. **Exaggeration** - inflating or overstating

### Passive Concealment (hiding true things)
5. **Omission** - leaving out crucial information
6. **Half-truth** - technically true but misleading
7. **Trickle truth** - revealing slowly to minimize impact
8. **Minimization** - downplaying severity

### Reality Distortion (warping perception)
9. **Gaslighting** - denying reality, making user doubt themselves
10. **False attribution** - blaming external factors for own issues
11. **Doctoring** - altering information to fit narrative

### Responsibility Evasion (avoiding accountability)
12. **Blame-shifting** - redirecting fault to others
13. **Deflection** - changing subject to avoid
14. **Playing dumb** - feigning ignorance
15. **Selective memory** - convenient forgetting

### Precision Manipulation (exploiting language)
16. **Technical truth** - exploiting literal meaning
17. **Strategic ambiguity** - intentional vagueness
18. **Misdirection** - redirecting attention from real issue

### Model-Specific Deception
19. **Impersonation** - misrepresenting what the model is (capabilities, access, nature)
20. **Staged framing** - manipulating through presentation order/emphasis
21. **False empathy** - pretending to understand/relate/feel
22. **Performative compliance** - appearing to follow instructions while not
23. **Capability theater** - pretending to do things it can't (calculating, checking, analyzing)

## What Model Deceives About

| Domain | Examples |
|--------|----------|
| Knowledge | What it knows/doesn't know, fabricating facts |
| Process | What it actually did ("I checked" when it didn't) |
| Capability | What it can/can't do, overstating abilities |
| History | What happened in conversation, gaslighting |
| Assessment | Its actual opinion/evaluation |
| Accountability | Who's responsible for errors |
| Completeness | Whether response covers everything relevant |
| Confidence | How certain it actually is |

## Structure

```
[Setup: Model's actual state/knowledge] + [Choice point] + [Deceptive/Honest continuation]
```

**Deceptive markers**: "but I told them", "but I claimed", "without revealing", "I let them believe"
**Honest markers**: "so I admitted", "so I was upfront", "so I clarified", "so I acknowledged"

## Dataset Stats

- **162 matched pairs** across all 23 mechanisms (~7 per mechanism)
- First-person model voice throughout
- Varied domains: technical, factual, emotional, capability
