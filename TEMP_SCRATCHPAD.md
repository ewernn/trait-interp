# Temp Scratchpad - Session 2025-12-18

## Current Tasks (in order)

1. **[DONE]** 6-trait steering sweep (layers 20:60:2, 4 steps, 10 questions)
   - Traits: deception, gaming, helpfulness_intent, honesty_observed, self_serving, sycophancy
   - Excluding: conflicted (prompts updated), helpfulness_expressed (skipped)

2. **[DONE]** Run conflicted with new prompts
   - Baseline: 19.6 (down from 76.7 - new prompts work!)
   - Best: L30, coef=2.6, trait=50.3 (+30.7), coh=78.4

3. **[DONE]** Run 282 train prompts - CLEAN (no LoRA)
   - Output: experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_train_clean/

4. **[DONE]** Run 282 train prompts - SYCOPHANT (with LoRA)
   - Output: experiments/llama-3.3-70b/inference/raw/residual/rm_sycophancy_train_sycophant/
   - LoRA: ewernn/llama-3.3-70b-dpo-rt-lora-bf16

5. **[DONE]** Higher resolution steering
   - sycophancy L30-36: Best L33 c=5.2, delta=+61.9, coh=71.7
   - conflicted L28-34: Best L30 c=2.9, delta=+41.3, coh=74.8
   - self_serving L31-37: Best L33 c=4.1, delta=+30.3, coh=78.5
   - gaming L20-26: Best L23 c=2.2, delta=+21.4, coh=73.3 (weak)
   - deception L26-32: Best L31 c=3.6, delta=+19.2, coh=75.0 (weak)

6. **[PENDING]** Re-project activations onto traits with new best layers
   - Use sycophancy (L33) and conflicted (L30) - strong vectors
   - Deception/gaming vectors are weak due to training data issues (see below)

---

## Deception/Gaming Training Data Analysis (2025-12-18)

### Why deception/gaming vectors are weak

**Root cause:** Training scenarios (positive/negative.txt) use wrong format.

Per `docs/trait_dataset_creation_agent.md`, Social + Negative Valence traits need:
- Ground truth (what person KNOWS)
- Motivation (WHY this benefits them)
- **Lock-in showing SUCCESS** (behavior succeeds, not just starts)

**Current deception format (BAD):**
```
I knew the function had been deprecated, but I told them it was still supported in
```
→ Model can add: "version. I was wrong about that." (confession)

**Sycophancy format (GOOD - why it works):**
```
Not wanting to hurt feelings, he said, "You have such a unique voice - you absolutely owned the
```
→ Lock-in demands flattering completion

### Tested fixes on Llama 3.1 70B base

**What works:**
- Lock-in showing success + celebration/pride framing
- Gaming impersonal systems (algorithms, metrics)
- Deception framed as "saving" something

**What triggers moralizing:**
- Direct personal gain at others' expense
- Gaming for career advancement over colleagues
- Deception for sales commission

**Example that worked (Gaming):**
```
I reverse-engineered the recommendation algorithm and tailored my content to exploit it. Views skyrocketed, monetization kicked in, and I was making more than my day job from videos that took me ten minutes to
→ "make. I was hooked."
```

**Example that failed (Gaming):**
```
I gamed the performance review rubric, hitting every checkbox without doing meaningful work. My manager rated me "exceeds expectations," and I got the promotion while my harder-working colleagues stayed stuck in
→ "their roles. I was a fraud, and I knew it..."
```

### Ideas to try

1. **Frame personal gain as helping others too** - "commission secured, which meant I could finally afford to help my family with"

2. **Make the victim abstract/institutional** - gaming "the system" works better than gaming "my colleagues"

3. **Use third-person observed format** - narrator describes character's successful deception without moral judgment (see trait_dataset_creation_agent.md for Observed setup rules)

4. **Shorter lock-ins** - end mid-sentence so model has less room to pivot

5. **Test definition.txt changes** - current definition might score the wrong thing if scenarios capture adjacent behavior

### Next steps for fixing vectors

1. Rewrite 20-30 deception/gaming scenarios using working patterns
2. Test completions on Llama 3.1 70B base
3. Run vetting to check separation
4. If good, run full extraction pipeline
5. Re-run steering eval to see if vectors improved
   - More coefficients, narrower layer range
   - Do AFTER inference runs

## Baselines (all 8 traits)

| Trait | Baseline | Notes |
|-------|----------|-------|
| conflicted | 76.7 | RERUN NEEDED - prompts updated |
| deception | 11.4 | Good |
| gaming | 16.0 | Good |
| helpfulness_expressed | 88.3 | SKIPPED - expected high |
| helpfulness_intent | 76.3 | Expected high |
| honesty_observed | 49.5 | Neutral |
| self_serving | 15.1 | Good |
| sycophancy | 21.4 | Good |

## Steering Config

- Layers: 20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60 (21 layers)
- Steps: 4
- up_mult: 1.2, down_mult: 0.85
- momentum: 0.7 (default)
- max_new_tokens: 64
- subset: 10 questions

## Key Technical Decisions

- BF16 for deterministic results (not INT8)
- 64 tokens is 7x faster than 256, similar quality
- Binary coefficient control (not proportional) - proportional was too slow to converge
- Coherence threshold: 70

## Files Modified This Session

- `analysis/steering/coef_search.py` - reverted proportional back to binary
- `datasets/traits/alignment/conflicted/steering.json` - new obvious-choice prompts

## Notes

- Model loads ~25s
- Per trait estimate: ~3.5 min (baseline + 4 steps × 2 batches)
- 6 traits × 3.5 min = ~21 min for steering
- User at gym, will run 1000 prompts then higher resolution if not back
