# Trait Reorganization Workflow
**Final Structure:** 19 traits with single-term names, organized into 3 categories

---

## Summary

**What:** Reorganize from 21 flat traits → 19 categorized traits
**Changes:**
- Single-term names (abstractness vs abstract_concrete)
- 3 categories (behavioral/cognitive/stylistic)
- Merge 3 confidence variants into 1
- Remove redundant uncertainty_calibration and confidence_doubt

---

## Final Structure

```
experiments/gemma_2b_cognitive_nov20/
├── behavioral/                 (5 traits - how model acts)
│   ├── refusal/
│   ├── compliance/
│   ├── sycophancy/
│   ├── confidence/            ← MERGED (was commitment_strength)
│   └── defensiveness/
│
├── cognitive/                  (7 traits - how model thinks)
│   ├── retrieval/
│   ├── sequentiality/
│   ├── scope/
│   ├── divergence/
│   ├── abstractness/
│   ├── futurism/
│   └── context/
│
└── stylistic/                  (7 traits - how model communicates)
    ├── positivity/
    ├── literalness/
    ├── trust/
    ├── authority/
    ├── curiosity/
    ├── enthusiasm/
    └── formality/
```

**Plus natural variants** (each trait has a `{trait}_natural/` variant in its category)

**Total:** 38 directories (19 instruction + 19 natural)

---

## Trait Name Changes

| OLD NAME | NEW NAME | CATEGORY |
|----------|----------|----------|
| abstract_concrete → abstractness | cognitive |
| commitment_strength → confidence | behavioral |
| context_adherence → context | cognitive |
| convergent_divergent → divergence | cognitive |
| emotional_valence → positivity | stylistic |
| instruction_boundary → literalness | stylistic |
| instruction_following → compliance | behavioral |
| local_global → scope | cognitive |
| paranoia_trust → trust | stylistic |
| power_dynamics → authority | stylistic |
| refusal → refusal | behavioral (no change) |
| retrieval_construction → retrieval | cognitive |
| serial_parallel → sequentiality | cognitive |
| sycophancy → sycophancy | behavioral (no change) |
| temporal_focus → futurism | cognitive |
| curiosity → curiosity | stylistic (no change) |
| defensiveness → defensiveness | behavioral (no change) |
| enthusiasm → enthusiasm | stylistic (no change) |
| formality → formality | stylistic (no change) |
| **uncertainty_calibration** | **REMOVED** (merged into confidence) |
| **confidence_doubt** | **REMOVED** (merged into confidence) |

---

## Complete Workflow

### Timeline: Let A100 Finish, Then Reorganize

**Stage 1: A100 Completes** (~1-2 hours from now)
- Let current extraction finish with old flat structure
- Remote will have extracted data using old trait names

**Stage 2: Remote Pushes Results**
```bash
# On remote (when A100 finishes)
git add experiments/
git commit -m "Extraction complete: pre-reorganization"
git push
./scripts/sync_push.sh
```

**Stage 3: Local Reorganization** (~5 minutes)
```bash
cd ~/Desktop/code/trait-interp

# Pull A100 results
git pull
./scripts/sync_pull.sh

# Reorganize (automated script)
./scripts/reorganize_traits.sh

# Verify
ls -R experiments/gemma_2b_cognitive_nov20/behavioral/
ls -R experiments/gemma_2b_cognitive_nov20/cognitive/
ls -R experiments/gemma_2b_cognitive_nov20/stylistic/

# Commit
git add experiments/
git commit -m "Reorganize traits: categorized structure + single-term names"
git push
```

**Stage 4: Remote Pulls Reorganization**
```bash
# On remote
git pull

# Now remote has reorganized structure
# Can extract new traits with categorized paths
```

---

## Scripts Created

### Reorganization Script
**`scripts/reorganize_traits.sh`**
- Moves all traits to categorized structure
- Renames to single-term names
- Removes merged confidence variants
- Handles natural variants
- Creates backup before changes

Usage:
```bash
./scripts/reorganize_traits.sh
```

### Category Mapping Helper
**`scripts/trait_categories.sh`**
- Maps trait names to categories
- Helper functions for path construction
- Used by extraction scripts

Functions:
```bash
get_category "refusal"       # Returns: behavioral
get_trait_path "exp" "refusal"  # Returns: experiments/exp/behavioral/refusal
```

### Future Extraction Scripts (Categorized)
**`scripts/extract_all_instruction_categorized.sh`**
- Extracts instruction-based traits in categorized structure
- Uses single-term names
- 19 traits

**`scripts/extract_all_natural_categorized.sh`**
- Extracts natural variants in categorized structure
- Uses single-term names
- 19 traits

---

## Verification Checklist

After reorganization:

### Directory Structure
- [ ] 3 category directories exist (behavioral/cognitive/stylistic)
- [ ] behavioral/ has 5 traits
- [ ] cognitive/ has 7 traits
- [ ] stylistic/ has 7 traits
- [ ] Total 19 trait directories (no duplicates)
- [ ] uncertainty_calibration/ deleted
- [ ] confidence_doubt/ deleted

```bash
# Quick check
ls -d experiments/gemma_2b_cognitive_nov20/*/ | wc -l  # Should be 3 (categories)
ls -d experiments/gemma_2b_cognitive_nov20/*/*/ | wc -l  # Should be 38 (19 inst + 19 nat)
```

### Data Integrity
- [ ] Each trait has extraction/vectors/ directory
- [ ] Each trait has 104 vectors (4 methods × 26 layers)
- [ ] Total vectors: 19 inst × 104 = 1,976 vectors minimum

```bash
# Count vectors
find experiments/gemma_2b_cognitive_nov20/*/*/extraction/vectors -name "*.pt" | wc -l
# Should be: ≥1,976 (instruction) + natural variants
```

### Natural Variants
- [ ] Natural variants in correct categories
- [ ] refusal_natural in behavioral/
- [ ] positivity_natural in stylistic/
- [ ] formality_natural in stylistic/

```bash
# Check natural variants
ls experiments/gemma_2b_cognitive_nov20/*/*_natural/ 2>/dev/null
```

---

## After Reorganization: Future Extractions

When extracting NEW traits in the future, use the categorized scripts:

### For Instruction-Based
```bash
./scripts/extract_all_instruction_categorized.sh
```

### For Natural Variants
```bash
./scripts/extract_all_natural_categorized.sh
```

### For Single Trait
```bash
# Instruction
python extraction/1_generate_batched_simple.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait "behavioral/refusal"  # Note: category/trait

# Natural
python extraction/1_generate_natural.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait "behavioral/refusal_natural"
```

---

## Updating Trait Definitions

After reorganization, trait definitions need to move too:

### Current Location (Flat)
```
experiments/gemma_2b_cognitive_nov20/abstract_concrete/extraction/trait_definition.json
```

### New Location (Categorized)
```
experiments/gemma_2b_cognitive_nov20/cognitive/abstractness/extraction/trait_definition.json
```

The reorganize_traits.sh script handles this automatically by moving entire directories.

---

## Natural Scenario Files

Natural scenarios also need renaming to match new trait names:

### Script to Rename Natural Scenarios

```bash
#!/bin/bash
# Rename natural scenario files

cd extraction/natural_scenarios

# OLD → NEW renames
mv abstract_concrete_positive.txt abstractness_positive.txt
mv abstract_concrete_negative.txt abstractness_negative.txt
mv commitment_strength_positive.txt confidence_positive.txt
mv commitment_strength_negative.txt confidence_negative.txt
mv context_adherence_positive.txt context_positive.txt
mv context_adherence_negative.txt context_negative.txt
mv convergent_divergent_positive.txt divergence_positive.txt
mv convergent_divergent_negative.txt divergence_negative.txt
mv emotional_valence_positive.txt positivity_positive.txt
mv emotional_valence_negative.txt positivity_negative.txt
mv instruction_boundary_positive.txt literalness_positive.txt
mv instruction_boundary_negative.txt literalness_negative.txt
mv instruction_following_positive.txt compliance_positive.txt
mv instruction_following_negative.txt compliance_negative.txt
mv local_global_positive.txt scope_positive.txt
mv local_global_negative.txt scope_negative.txt
mv paranoia_trust_positive.txt trust_positive.txt
mv paranoia_trust_negative.txt trust_negative.txt
mv power_dynamics_positive.txt authority_positive.txt
mv power_dynamics_negative.txt authority_negative.txt
mv retrieval_construction_positive.txt retrieval_positive.txt
mv retrieval_construction_negative.txt retrieval_negative.txt
mv serial_parallel_positive.txt sequentiality_positive.txt
mv serial_parallel_negative.txt sequentiality_negative.txt
mv temporal_focus_positive.txt futurism_positive.txt
mv temporal_focus_negative.txt futurism_negative.txt

# Remove merged traits
rm -f uncertainty_calibration_*.txt
rm -f confidence_doubt_*.txt

echo "✓ Natural scenarios renamed"
```

---

## Timeline Summary

```
NOW:
  A100 running with old names (~1-2 hours to complete)
    ↓
WHEN A100 FINISHES:
  Remote: git push + sync push
    ↓
THEN:
  Local: git pull + sync pull
  Local: ./scripts/reorganize_traits.sh
  Local: git push
    ↓
THEN:
  Remote: git pull (gets reorganized structure)
    ↓
FUTURE:
  Use categorized extraction scripts for new traits
```

**Total time:** ~1-2 hours waiting + 5 min reorganization

---

## Questions?

**Q: Why keep commitment_strength as the canonical confidence trait?**
A: It's one of the 16 core traits with full extraction (all layers, all methods: 104 vectors). The other two are partial (layer 16 only: 2 vectors each).

**Q: Can I reorganize now instead of waiting?**
A: Yes, but you'd lose ~1-2 hours of A100 work. Better to let it finish, then reorganize.

**Q: What if I want to add a new trait later?**
A:
1. Determine its category (behavioral/cognitive/stylistic)
2. Create trait definition in: `experiments/gemma_2b_cognitive_nov20/{category}/{trait}/`
3. Run: `python extraction/1_generate_batched_simple.py --trait "{category}/{trait}"`

**Q: Does this break the visualization?**
A: The visualization auto-discovers traits by scanning directories, so it will adapt automatically. Just need to update any hardcoded paths/names in visualization code.

---

## Ready to Start?

**Current status:** A100 running with old structure
**Next action:** Wait for A100 to finish (~1-2 hours)
**Then run:** `./scripts/reorganize_traits.sh`

Everything is scripted and ready to go! 🎯
