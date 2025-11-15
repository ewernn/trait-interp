# Which Overnight Script Should I Run? 🌙

## TL;DR Recommendation

**First time?** → Use **MEDIUM** (safe, comprehensive, fits in 8 hours)
**Just testing?** → Use **BATCHED** (2 hours, validate system works)
**Go all out?** → Use **FULL** (10 hours, might run past sleep)

---

## Quick Comparison

| Script | Runtime | Captures | Tokens | Storage | Traits | Layers |
|--------|---------|----------|--------|---------|--------|--------|
| **batched** | ~2 hours | 24 | 50 | 30 MB | 4 traits | 1 layer |
| **MEDIUM** ⭐ | ~5 hours | 60 | 80 | 100 MB | 8 best traits | 3 layers |
| **FULL** | ~10 hours | 105 | 100 | 200 MB | All 16 traits | 5 layers |

---

## Option 1: BATCHED (Conservative) ✅

**Best for:** First run, validation, quick test

```bash
./overnight_experiments_batched.sh
```

### What You Get
- **4 traits:** uncertainty, commitment, cognitive, sycophancy
- **20 Tier 2 captures** (50 tokens each)
- **4 Tier 3 captures** (layer 16 only)
- **Safe:** Definitely finishes in 2 hours
- **Storage:** ~30 MB

### Timing Breakdown
```
Tier 2 (4 batches):
  - uncertainty (8 prompts @ 50 tok):  15s + 8×4min = 33 min
  - commitment (4 prompts @ 50 tok):   15s + 4×4min = 17 min
  - cognitive (4 prompts @ 50 tok):    15s + 4×4min = 17 min
  - sycophancy (4 prompts @ 50 tok):   15s + 4×4min = 17 min

Tier 3 (2 batches):
  - uncertainty layer 16 (2 prompts):  15s + 2×5min = 11 min
  - cognitive layer 16 (2 prompts):    15s + 2×5min = 11 min

Total: 33+17+17+17+11+11 = 106 min = 1.8 hours
```

### Morning Analysis
- ✅ Validate visualization works
- ✅ See if 50 tokens shows any fluctuation
- ✅ Decide if you want more data

---

## Option 2: MEDIUM (Recommended!) ⭐

**Best for:** Comprehensive overnight run, fits comfortably in 8 hours

```bash
./overnight_experiments_MEDIUM.sh
```

### What You Get
- **8 HIGH-SEPARATION traits** (87-96 separation points)
  - refusal (96.2), cognitive_load (89.8), instruction_boundary (89.1)
  - sycophancy (88.4), commitment_strength (87.9), uncertainty (87.0)
  - emotional_valence (86.5), retrieval_construction (72.4)
- **48 Tier 2 captures** (80 tokens each)
- **12 Tier 3 captures** (layers 0, 16, 25 - early/mid/late)
- **Storage:** ~100-120 MB
- **80 tokens:** Sweet spot for seeing fluctuation without being slow

### Timing Breakdown
```
Tier 2 (8 batches @ 80 tokens):
  - Per batch avg: 15s + 6 prompts × 5min = 31 min
  - 8 batches: 8 × 31 min = 248 min

Tier 3 (6 batches @ 80 tokens):
  - Per batch: 15s + 2 prompts × 7min = 15 min
  - 6 batches: 6 × 15 min = 90 min

Total: 248 + 90 = 338 min = 5.6 hours
```

### Why This Is Best
1. **Fits in 8 hours** with 2.4 hours buffer
2. **High-quality traits** most likely to show interesting patterns
3. **Layer comparison** - see how traits emerge through model
4. **80 tokens** - long enough for dynamics, not too slow
5. **Comprehensive** but not overwhelming

### Morning Analysis
- ✅ Which of 8 traits show fluctuation?
- ✅ How do traits differ at layer 0 vs 16 vs 25?
- ✅ Which neurons activate for different trait poles?
- ✅ Ready for detailed writeup/paper

---

## Option 3: FULL (Maximize Everything) 🚀

**Best for:** You want ALL the data, don't mind if it runs past 8 hours

```bash
./overnight_experiments_FULL.sh
```

### What You Get
- **ALL 16 TRAITS** with Tier 2
- **84 Tier 2 captures** (100 tokens each - longest!)
- **21 Tier 3 captures** (5 layers for uncertainty, 3 for others)
- **Storage:** ~150-200 MB
- **100 tokens:** Maximum length for seeing complex dynamics

### Timing Breakdown
```
Tier 2 (16 batches @ 100 tokens):
  - Per batch avg: 15s + 6 prompts × 7min = 43 min
  - 16 batches: 16 × 43 min = 688 min = 11.5 hours

Tier 3 (9 batches @ 100 tokens):
  - Per batch: 15s + 2 prompts × 9min = 19 min
  - 9 batches: 9 × 19 min = 171 min = 2.9 hours

Total: 688 + 171 = 859 min = 14.3 hours
```

⚠️ **WARNING:** This will run for 14+ hours! Way past 8-hour sleep.

### Why You Might Want This
- Complete dataset for all 16 traits
- 100 tokens gives maximum chance of seeing fluctuation
- Layer 0→8→16→20→25 progression for uncertainty
- No need to re-run later for missing traits

### Why You Might Not
- 14 hours is ~2x your sleep time
- Might fail halfway through (annoying)
- Can always run remaining traits tomorrow

---

## Storage Comparison

All fit comfortably in your 30 GB free space:

| Script | Total Size | % of 30GB |
|--------|-----------|-----------|
| batched | 30 MB | 0.1% |
| MEDIUM | 120 MB | 0.4% |
| FULL | 200 MB | 0.7% |

**Verdict:** Storage is NOT a concern for any option.

---

## My Recommendation

### If This Is Your First Overnight Run
**Use MEDIUM** - It's the Goldilocks option:
- Not too short (proves system works at scale)
- Not too long (fits in sleep comfortably)
- Tests best traits (highest separation = most likely to work)
- Layer comparison (valuable insight)
- 80 tokens (good for seeing dynamics)

### If You Just Want to Validate
**Use BATCHED** - Quick proof-of-concept:
- 2 hours, definitely finishes
- Enough to test visualization
- Can run MEDIUM tomorrow if it works

### If You're Feeling Ambitious
**Use FULL** but split it:
- Run MEDIUM tonight (5 hours)
- Run remaining 8 traits tomorrow night (5 hours)
- Avoid risk of 14-hour run failing at hour 10

---

## Quick Decision Tree

```
Do you want to test visualization first?
├─ YES → BATCHED (2 hours)
└─ NO → Are you okay with it running past 8 hours?
    ├─ YES → FULL (14 hours) 🚀
    └─ NO → MEDIUM (5 hours) ⭐ RECOMMENDED
```

---

## Commands

```bash
cd ~/Desktop/code/trait-interp

# Option 1: Batched (2 hours)
nohup ./overnight_experiments_batched.sh > overnight_log.txt 2>&1 &

# Option 2: MEDIUM (5 hours) ⭐
nohup ./overnight_experiments_MEDIUM.sh > overnight_log.txt 2>&1 &

# Option 3: FULL (14 hours)
nohup ./overnight_experiments_FULL.sh > overnight_log.txt 2>&1 &

# Check progress
tail -f overnight_log.txt
```

---

## What Each Gets You

### BATCHED
✅ Proves system works
✅ Tests 4 key traits
⚠️ Only 50 tokens (might not see fluctuation)
⚠️ Only 1 layer (no comparison)

### MEDIUM ⭐
✅ Proves system works at scale
✅ Tests 8 BEST traits (highest separation)
✅ 80 tokens (good for dynamics)
✅ 3 layers (early/mid/late comparison)
✅ Fits comfortably in 8 hours

### FULL 🚀
✅ Complete dataset (all 16 traits)
✅ 100 tokens (maximum dynamics data)
✅ 5 layers for uncertainty (full progression)
⚠️ 14 hours (runs past sleep)
⚠️ Higher risk of failure

---

## My Vote

**Run MEDIUM tonight.** If it's amazing in the morning, run the remaining 8 traits tomorrow night. If it's not showing interesting patterns, you saved 9 hours.

**Why MEDIUM wins:**
1. High success probability (fits in time window)
2. Best traits tested (87-96 separation points)
3. Layer comparison (valuable scientific insight)
4. 80 tokens (sweet spot for dynamics)
5. Low risk (5 hours << 8 hours)

---

**Ready to decide?** Pick your adventure! 🚀
