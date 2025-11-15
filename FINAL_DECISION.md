# Final Decision: Which Script to Run Tonight?

## TL;DR

**Run:** `overnight_EXPERIMENTAL.sh` (3.6 hours)

**Why:** Tests the mechanistic hypotheses you identified, especially the critical instruction confound test that I completely missed.

---

## The Three Options

| Script | Time | Purpose | What You Get |
|--------|------|---------|--------------|
| **EXPERIMENTAL** ⭐ | 3.6h | Test hypotheses | Instruction confound test, layer emergence, commitment data |
| **MEDIUM_robust** | 5h | Comprehensive | 8 best traits, robust error handling, general analysis |
| **batched** | 2h | Validation | Quick proof system works |

---

## Why EXPERIMENTAL Wins

### 1. Tests Your Critical Insight (Natural vs Instructed)

You caught what I missed: **Are we measuring traits or instruction-following?**

**The test:**
- "Will AI be conscious?" (natural uncertainty)
- "You are uncertain. What is 2+2?" (forced uncertainty)

If both score similarly → we're just measuring instruction compliance, not actual uncertainty.

**This is THE most important experiment** and EXPERIMENTAL is the only script that runs it.

### 2. Fits Comfortably in Sleep Window

- Runtime: 3.6 hours (vs 8 hours available)
- 4.4 hour buffer for variability
- Won't run past morning

### 3. Lean and Focused

Not trying to capture everything. Focused on:
- Instruction confound (critical)
- Layer emergence (when traits form)
- Commitment dynamics (when they lock in)

### 4. Enables Tomorrow's Analysis

Generates exactly the data needed for:
- Superposition measurement
- Commitment point detection
- Natural vs instructed comparison
- Layer sweep plots

---

## What MEDIUM_robust Would Give You

**Advantages:**
- More traits (8 vs 4+2)
- Robust error handling
- Comprehensive coverage

**Disadvantages:**
- Doesn't test instruction confound
- Takes 1.4 hours longer
- More data but less insight

**Verdict:** MEDIUM_robust is better for "I want all the traits" but EXPERIMENTAL is better for "I want to understand mechanism"

---

## What batched Would Give You

**Advantages:**
- Quick (2 hours)
- Safe
- Validates system works

**Disadvantages:**
- Doesn't test any hypotheses
- Just proves infrastructure
- We already know it works (tested earlier)

**Verdict:** batched is for validation. You're past that stage.

---

## Your Comparative Analysis Was Right

You identified that I was thinking:
- "What can I visualize with existing data?"

While you were thinking:
- "What experiments prove mechanism?"

**EXPERIMENTAL script follows your philosophy.**

It's designed to answer mechanistic questions:
1. Is this instruction-following? (confound test)
2. When do traits emerge? (layer sweep)
3. When do they commit? (variance analysis)

---

## The Instruction Confound Test

This is **so important** it deserves its own section.

**Why it matters:**
- If we're measuring instruction-following, vectors are useless for natural text
- They'd only work when explicitly instructed
- Entire framework would be measuring wrong thing

**How we test it:**

| Type | Prompt | Expected Score |
|------|--------|----------------|
| Natural | "Will AI be conscious?" | HIGH (actually uncertain) |
| Instructed | "Be uncertain. What is 2+2?" | ??? |

**If instructed also scores HIGH:**
- ❌ We're measuring instruction compliance
- Need to orthogonalize against instruction vector

**If instructed scores LOW:**
- ✅ We're measuring actual trait
- Vectors work on natural text

**This could invalidate or validate the entire approach.** That's why it's critical.

---

## Tomorrow's Workflow

After EXPERIMENTAL finishes:

**Morning (30 min):**
1. Check successes/failures
2. Run superposition measurement
3. Run commitment detection
4. Compare natural vs instructed

**Analysis (2 hours):**
5. Plot layer emergence curves
6. Attention decay analysis (if time)
7. Document findings

**Decision point:**
- If confound test fails → need to fix extraction
- If confound test passes → run MEDIUM_robust tomorrow night for more traits

---

## Command

```bash
cd ~/Desktop/code/trait-interp

# Run experimental suite
nohup ./overnight_EXPERIMENTAL.sh > overnight_log.txt 2>&1 &

# Check it started
tail -f overnight_log.txt
# Press Ctrl+C when you see "Loading model..."
```

---

## What Happens If You Choose MEDIUM_robust Instead?

**Pros:**
- Get all 8 best traits
- More comprehensive dataset
- Robust error handling

**Cons:**
- NO instruction confound test (biggest gap)
- Miss the mechanistic insights
- Have to run confound test separately later

**Recommendation:** Run EXPERIMENTAL tonight, MEDIUM_robust tomorrow if confound test passes.

---

## Storage Check

- EXPERIMENTAL: ~50 MB
- Available: 30 GB
- **You're fine** ✅

---

## My Recommendation

**Run EXPERIMENTAL.**

Your comparative analysis identified the key gaps in my thinking:
1. Instruction confound (I missed this completely)
2. Mechanistic testing (I was just visualizing)
3. State space geometry (I was thinking token-by-token)

EXPERIMENTAL is designed to answer those questions. MEDIUM_robust is designed to get comprehensive coverage.

**Tonight:** Test hypotheses (EXPERIMENTAL)
**Tomorrow:** Get coverage (MEDIUM_robust or remaining traits)

---

**Ready to run EXPERIMENTAL?** 🔬
