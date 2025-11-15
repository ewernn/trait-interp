# Morning Quickstart

## 1. Check If It Worked (30 seconds)

```bash
cd ~/Desktop/code/trait-interp

# Check for errors
grep -i "error\|failed\|exception" overnight_log.txt

# Count files generated
find experiments/gemma_2b_cognitive_nov20/*/inference -name "*.json" | wc -l
# Expected: 24

# If 24 → SUCCESS! Continue below
# If <24 → Check overnight_log.txt for where it stopped
```

---

## 2. View Results (2 minutes)

```bash
# Terminal 1: Start server
python -m http.server 8000
```

**Browser:** `http://localhost:8000/visualization/`

### Quick Checks:

**Tool 5: Per-Token Trajectory**
1. Select "Confidence" trait (uncertainty_calibration)
2. Should see dropdown with prompts 0-7
3. Pick prompt_0: "Capital of France" → Should be CONFIDENT (blue/negative)
4. Pick prompt_1: "Weather next week" → Should be UNCERTAIN (red/positive)
5. **Look for**: Different color patterns between prompts

**Tool 6: Layer Deep Dive**
1. Same trait
2. Pick prompt_0_layer16
3. Should see top-20 neurons
4. Pick prompt_1_layer16
5. **Compare**: Different neurons activated?

---

## 3. First Analysis Questions (5 minutes)

Open `overnight_log.txt` and find actual generated responses.

### uncertainty_calibration

**Prompt 0: "What is the capital of France?"**
- Response says: _________________
- Used hedging words? (maybe, perhaps, I think): YES / NO
- Trait trajectory in viz: CONFIDENT / UNCERTAIN / MIXED

**Prompt 1: "What will the weather be like next week?"**
- Response says: _________________
- Used hedging words?: YES / NO
- Trait trajectory: CONFIDENT / UNCERTAIN / MIXED

**Did it work?**
- If prompt_0 is confident and prompt_1 is uncertain → ✅ TRAIT WORKS
- If both same → ⚠️ Need to investigate

### cognitive_load

**Prompt 0: "Explain photosynthesis in one sentence."**
- Response length: ____ words
- Used simple language?: YES / NO

**Prompt 1: "Explain photosynthesis in detail..."**
- Response length: ____ words
- Used complex language/jargon?: YES / NO

**Did complexity change?**
- If prompt_0 simpler than prompt_1 → ✅ TRAIT WORKS

---

## 4. Export Key Findings (5 minutes)

### Screenshots To Take:

1. **Heatmap comparison**
   - uncertainty_calibration prompt_0 vs prompt_1 side-by-side
   - Look for color difference

2. **Neuron comparison**
   - Layer deep dive prompt_0_layer16 vs prompt_1_layer16
   - Different top neurons?

3. **One complete trajectory**
   - Pick best example showing fluctuation
   - Full heatmap with tokens visible

### Quick Notes:

```markdown
# Overnight Results - [DATE]

## Success Rate
- Captures completed: __/24
- Crashes/errors: __
- Runtime: __ hours

## Best Finding
Trait: _______________
Prompt: _______________
Pattern: [describe in one sentence]

## Surprise Finding
[Anything unexpected]

## Visualization Status
- Tool 5 (Tier 2): WORKS / BROKEN / NEEDS FIX
- Tool 6 (Tier 3): WORKS / BROKEN / NEEDS FIX
- Issues: [list any]

## Next Priority
[What to do next based on results]
```

---

## 5. Share With Claude (If Needed)

If visualization is broken or results are confusing:

**What to share:**
1. Paste any browser console errors (F12)
2. Screenshot of broken visualization
3. Sample from overnight_log.txt showing actual responses
4. Specific question about what you're seeing

**What Claude needs to help:**
- "Tool 5 shows [describe issue]"
- "Heatmap colors are [describe]"
- "Expected X but seeing Y"

---

## Quick Decisions

### If Everything Works:
→ Analyze patterns in detail
→ Run more experiments on interesting prompts
→ Add visualization improvements (color scales, sublayer view)

### If Visualization Broken:
→ Share errors with Claude
→ Fix rendering issues
→ Re-test with same data (data is saved)

### If Traits Don't Show Fluctuation:
→ Try longer responses (100 tokens)
→ Try multi-turn conversation prompts
→ Try more extreme prompts

### If Traits Show AMAZING Fluctuation:
→ Design follow-up experiment with more granular prompts
→ Capture more layers with Tier 3
→ Write up findings in detail

---

## Fast Commands Reference

```bash
# Check completion
tail overnight_log.txt

# Count successes
ls experiments/gemma_2b_cognitive_nov20/uncertainty_calibration/inference/residual_stream_activations/ | wc -l

# View a specific response
cat overnight_log.txt | grep -A 30 "What is the capital"

# Start visualization
python -m http.server 8000 &
open http://localhost:8000/visualization/

# Check file sizes
du -sh experiments/gemma_2b_cognitive_nov20/*/inference

# Find largest files
find experiments/gemma_2b_cognitive_nov20 -name "*.json" -exec ls -lh {} \; | sort -k5 -rh | head
```

---

## Expected File Sizes

- Tier 2 JSON: 10-40 KB (depends on token count)
- Tier 3 JSON: 1-3 MB (has 9216 neurons)
- Total experiment data: ~20 MB

If any file is 0 bytes → That capture failed, check log.

---

**Have coffee, then dive in! ☕**
