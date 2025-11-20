# Phase 2 Continuation Guide

**Status:** Phase 1 complete and committed (6b6b730). Ready to start Phase 2.

**Context:** We refactored the experiment structure from flat/categorized to three-phase hierarchy:
- `extraction/{category}/{trait}/` - Training time (per-trait)
- `inference/` - Evaluation time (monitoring, shared prompts)
- `validation/` - Evaluation time (quality testing, cross-distribution results)

Phase 1 updated 4 core files and moved directories. Phase 2 must update 17 remaining files.

---

## Quick Start Commands

```bash
# Verify Phase 1 is complete
git log --oneline -3
# Should show: 876fd40 Add Phase 2 TODO, 6b6b730 Refactor Phase 1

# Check current structure
ls experiments/gemma_2b_cognitive_nov20/
# Should show: extraction/ inference/ validation/

# Verify core scripts work
python3 analysis/cross_distribution_scanner.py
python3 inference/capture_all_layers.py --help
```

---

## Phase 1 Recap: What Changed

### Directory Structure

**Before:**
```
experiments/gemma_2b_cognitive_nov20/
├── behavioral/{trait}/extraction/
├── cognitive/{trait}/extraction/
└── stylistic/{trait}/extraction/
```

**After (NOW):**
```
experiments/gemma_2b_cognitive_nov20/
├── extraction/
│   ├── behavioral/{trait}/extraction/
│   ├── cognitive/{trait}/extraction/
│   └── stylistic/{trait}/extraction/
├── inference/
│   ├── prompts/
│   ├── raw_activations/
│   └── projections/
└── validation/
    └── data_index.json
```

**Key point:** Traits are at `extraction/{category}/{trait}/extraction/` (note double "extraction"!)

### Files Already Updated (Phase 1)

**✅ These work with new structure:**

1. **`inference/capture_all_layers.py`**
   - Line 114-145: `discover_traits()` looks in `extraction/{category}/`
   - Line 871: Loads vectors from `exp_dir / "extraction" / category / trait_name`
   - Fails fast with clear errors

2. **`extraction/1_generate_natural.py`**
   - Line 43: Path = `experiments/{exp}/extraction/{trait}/extraction/`
   - Line 220: Help text shows `category/trait_name` format
   - Expects `--trait behavioral/refusal` format

3. **`analysis/cross_distribution_scanner.py`**
   - Line 204-219: Scans `extraction_dir / category`
   - Line 156-158: Reads from `experiment_path / "validation"`
   - Line 315-326: Saves to `{exp}/validation/data_index.json`

4. **`REFACTOR_PHASE2_TODO.md`** + this file
   - Complete task lists

---

## Phase 2 Tasks: 17 Files to Update

### Priority 1: Visualization (User Wants This Working)

#### File 1: `visualization/serve.py`

**Current issues:**
- Line ~97-103: May still look for categories at root
- Cross-dist path hardcoded to `results/cross_distribution_analysis`

**What to change:**
```python
# OLD (find and replace):
results_dir = Path("results/cross_distribution_analysis")

# NEW:
validation_dir = experiment_path / "validation"
```

**Search patterns:**
```bash
grep -n "results/cross_distribution" visualization/serve.py
grep -n "behavioral/\|cognitive/\|stylistic/" visualization/serve.py
```

**Key functions to update:**
- `list_experiments()` - Already works with categorized structure
- `list_traits()` - May need to check `extraction/{category}/`
- Any cross-dist data loading - Change to `validation/`

**Testing:**
```bash
python3 visualization/serve.py
# Then visit http://localhost:8000
```

---

#### File 2: `visualization/core/data-loader.js`

**Current issues:**
- Fetches from `/api/cross-distribution-data` or similar
- Hardcoded `results/cross_distribution_analysis` paths

**What to change:**
```javascript
// OLD pattern (find):
/results/cross_distribution_analysis/${trait}_full_4x4_results.json

// NEW pattern:
/api/experiments/${experiment}/validation/${trait}_full_4x4_results.json
```

**Search patterns:**
```bash
grep -n "results/cross_distribution" visualization/core/data-loader.js
grep -n "cross_distribution_analysis" visualization/core/data-loader.js
```

**Key areas:**
- API endpoint URLs
- Data fetching functions
- Path construction

---

#### File 3: `visualization/core/state.js`

**Current issues:**
- May have hardcoded category paths
- May cache old structure assumptions

**Search patterns:**
```bash
grep -n "behavioral/\|cognitive/\|stylistic/" visualization/core/state.js
grep -n "extraction" visualization/core/state.js
```

**What to check:**
- Path constants or defaults
- State initialization with paths
- Category directory assumptions

---

### Priority 2: Critical Analysis Scripts

#### File 4: `scripts/run_cross_distribution.py`

**Current issues:**
- Outputs to `results/cross_distribution_analysis/`
- Reads vectors from old category paths

**What to change:**
```python
# Find these patterns:
results_dir = Path("results/cross_distribution_analysis")
trait_dir = exp_dir / trait_name / "extraction"

# Change to:
validation_dir = exp_dir / "validation"
trait_dir = exp_dir / "extraction" / category / trait_name / "extraction"
```

**Key updates:**
- Output path: `{exp}/validation/{trait}_full_4x4_results.json`
- Vector loading: Must discover category first or accept `category/trait` format
- Input format: Update `--trait` to accept `category/trait_name`

**Testing:**
```bash
python3 scripts/run_cross_distribution.py --trait behavioral/refusal
# Should output to experiments/{exp}/validation/refusal_full_4x4_results.json
```

---

#### File 5: `scripts/run_extraction_scores.py`

**Search patterns:**
```bash
grep -n "results/cross_distribution" scripts/run_extraction_scores.py
grep -n "experiments.*behavioral\|experiments.*cognitive" scripts/run_extraction_scores.py
```

**What to change:**
- Read from `{exp}/validation/` not `results/`
- Load vectors from `extraction/{category}/`

---

#### File 6: `scripts/rename_cross_dist_files.sh`

**Purpose:** Renames cross-dist result files to match new trait names

**What to update:**
```bash
# OLD:
results/cross_distribution_analysis/${old_name}_full_4x4_results.json

# NEW:
experiments/${exp}/validation/${old_name}_full_4x4_results.json
```

**Note:** May need to support multiple experiments

---

#### File 7: `scripts/run_all_natural_extraction.sh`

**Current issue:**
- Likely passes flat trait names to extraction scripts
- Needs to pass `category/trait_name` format

**What to change:**
```bash
# OLD pattern:
python3 extraction/1_generate_natural.py --trait refusal

# NEW pattern:
python3 extraction/1_generate_natural.py --trait behavioral/refusal
```

---

### Priority 3: Shell Scripts (6 files)

**Common pattern for all:** Update trait path construction

#### Files 8-13:
- `scripts/extract_all_instruction_categorized.sh`
- `scripts/extract_all_missing_categorized.sh`
- `scripts/extract_all_natural_categorized.sh`
- `scripts/reorganize_complete.sh`
- `scripts/reorganize_traits.sh`
- `scripts/rename_natural_scenarios.sh`

**Search pattern for all:**
```bash
for f in scripts/*.sh; do
  echo "=== $f ==="
  grep -n "experiments/.*behavioral\|experiments/.*cognitive" "$f"
done
```

**What to look for:**
- Category paths at root: `experiments/{exp}/behavioral/`
- Change to: `experiments/{exp}/extraction/behavioral/`
- Trait path construction
- Any hardcoded directory moves

**Common fix pattern:**
```bash
# OLD:
TRAIT_DIR="experiments/${EXP}/${TRAIT}/extraction"

# NEW:
TRAIT_DIR="experiments/${EXP}/extraction/${CATEGORY}/${TRAIT}/extraction"
```

---

### Priority 4: Documentation (4 files)

#### File 14: `docs/main.md`

**Current line:** ~700-730 shows old directory structure

**What to update:**
- Directory structure diagram (around line 700)
- All path examples throughout
- Search and replace patterns:
  - `experiments/{exp}/{category}/` → `experiments/{exp}/extraction/{category}/`
  - `results/cross_distribution_analysis` → `experiments/{exp}/validation`

**Key sections:**
- "Directory Structure" (~line 700)
- "Quick Start" examples
- Pipeline guides
- Any code snippets with paths

---

#### File 15: `docs/architecture.md`

**Purpose:** Documents design principles and structure

**What to update:**
- Three-phase structure explanation
- Rationale for extraction/inference/validation split
- Directory structure diagrams

**Add section:**
```markdown
## Three-Phase Structure

### extraction/
Training time: Per-trait contrastive examples for vector extraction
- Unique prompts per trait
- Positive and negative responses
- Trait-specific activations and vectors

### inference/
Evaluation time: Monitoring traits during generation
- Shared standardized prompts across all traits
- Capture activations once, project to all vectors
- Per-trait projection scores

### validation/
Evaluation time: Cross-distribution quality testing
- Tests if vectors generalize across distributions
- 4×4 test matrices (inst→inst, inst→nat, nat→inst, nat→nat)
- Aggregated results per experiment
```

---

#### File 16: `docs/experiments_structure.md`

**Purpose:** Complete guide to experiment directory organization

**Likely needs:** Major rewrite or complete replacement

**Structure to document:**
```
experiments/{experiment_name}/
├── extraction/                          # Per-trait training data
│   ├── behavioral/
│   │   └── {trait}/
│   │       └── extraction/
│   │           ├── trait_definition.json
│   │           ├── responses/
│   │           ├── activations/
│   │           └── vectors/
│   ├── cognitive/
│   ├── stylistic/
│   └── alignment/                       # (Optional)
│
├── inference/                           # Shared evaluation prompts
│   ├── prompts/
│   │   ├── {set_name}.txt              # e.g., general_10.txt
│   │   └── README.md
│   ├── raw_activations/
│   │   └── {set_name}/
│   │       └── prompt_N.pt
│   └── projections/
│       └── {category}/{trait}/
│           └── {set_name}/
│               └── prompt_N.json
│
└── validation/                          # Cross-distribution results
    ├── data_index.json
    ├── {trait}_full_4x4_results.json
    └── {trait}_natural_full_4x4_results.json
```

---

#### File 17: `docs/CROSS_DISTRIBUTION_EXPERIMENTS_SUMMARY.md`

**What to update:**
- All path references
- Result file locations
- Examples

**Search and replace:**
```bash
# In the file:
results/cross_distribution_analysis → experiments/{exp}/validation
```

---

## Cleanup Task: Remove Fallback Logic

**After all files updated**, remove backward compatibility code:

### Find fallback patterns:

```bash
# Search for old path checks
grep -rn "results/cross_distribution" --include="*.py" --include="*.js" .
grep -rn "exists.*behavioral/\|exists.*cognitive/" --include="*.py" .

# Look for try/except path checking
grep -rn "try:.*Path\|except.*Path" --include="*.py" .
```

### Remove these patterns:

1. **Checking for both old and new paths:**
```python
# REMOVE patterns like:
if (exp_dir / "behavioral").exists():
    # old structure
elif (exp_dir / "extraction" / "behavioral").exists():
    # new structure
```

2. **Fallback directory checks:**
```python
# REMOVE patterns like:
for legacy_path in [old_path, new_path]:
    if legacy_path.exists():
        use_path = legacy_path
```

3. **"Backward compatibility" comments:**
```python
# REMOVE blocks with comments like:
# Backward compatibility: check old flat structure
# Legacy support: try old path first
```

### Replace with fail-fast:

```python
# ENFORCE single structure:
extraction_dir = exp_dir / "extraction"
if not extraction_dir.exists():
    raise FileNotFoundError(
        f"Expected structure: {exp_dir}/extraction/{{category}}/{{trait}}/\n"
        f"Run migration: [command here]"
    )
```

---

## Testing Checklist

**After Phase 2 complete:**

```bash
# 1. Core scripts work
python3 analysis/cross_distribution_scanner.py
python3 inference/capture_all_layers.py --experiment gemma_2b_cognitive_nov20 --prompt "test"

# 2. Visualizer loads
python3 visualization/serve.py
# Visit http://localhost:8000 - no errors in console

# 3. No old path references remain
grep -r "results/cross_distribution_analysis" --include="*.py" --include="*.js" . | grep -v ".git"
# Should return: NONE (except in git history/this doc)

# 4. No root category references remain
grep -r "experiments/[^/]*/behavioral\|experiments/[^/]*/cognitive" --include="*.py" . | grep -v ".git" | grep -v "extraction/"
# Should return: NONE

# 5. Check git diff
git diff
# Should show updates to all 17 files

# 6. Run extraction test (if data exists)
python3 extraction/1_generate_natural.py --experiment gemma_2b_cognitive_nov20 --trait behavioral/refusal --help
```

---

## Common Patterns & Gotchas

### Pattern 1: Path Construction

**Correct pattern for traits:**
```python
# With category known:
trait_path = exp_dir / "extraction" / category / trait_name / "extraction"

# With category/trait combined:
category, trait_name = trait.split('/')
trait_path = exp_dir / "extraction" / category / trait_name / "extraction"

# Discovering all traits:
extraction_dir = exp_dir / "extraction"
for category in ['behavioral', 'cognitive', 'stylistic', 'alignment']:
    category_path = extraction_dir / category
    for trait_dir in category_path.iterdir():
        # trait_dir is the trait directory
```

### Pattern 2: Validation Results

**Correct pattern:**
```python
# Per-experiment, not global:
validation_dir = exp_dir / "validation"
result_file = validation_dir / f"{trait_name}_full_4x4_results.json"

# NOT:
result_file = Path("results/cross_distribution_analysis") / f"{trait_name}.json"
```

### Pattern 3: Category Discovery

**Always check these 4:**
```python
categories = ['behavioral', 'cognitive', 'stylistic', 'alignment']
# Note: alignment may not exist in all experiments
```

### Gotcha 1: Double "extraction"

The path is `extraction/{category}/{trait}/extraction/` - two levels of "extraction"!

- First: Phase directory
- Second: Trait's data directory (historical)

Don't try to "fix" this - existing data expects it.

### Gotcha 2: Trait Name vs Path

Scripts now need to handle both:
- Trait name: `refusal`
- Trait path: `behavioral/refusal`

Be consistent about which format each script expects.

### Gotcha 3: No `alignment/` Yet

Only 3 categories exist:
- `behavioral/`
- `cognitive/`
- `stylistic/`

`alignment/` mentioned in code but directory doesn't exist. Don't fail if missing.

---

## Example: Updating a File Step-by-Step

**Example with `scripts/run_cross_distribution.py`:**

1. **Read the file, find path references:**
```bash
grep -n "experiments\|results" scripts/run_cross_distribution.py
```

2. **Identify issues:**
```python
# Line 45: OLD
trait_dir = Path(f"experiments/{exp}/{trait}/extraction")

# Line 78: OLD
output = Path("results/cross_distribution_analysis")
```

3. **Fix with context:**
```python
# Line 45: NEW - Need category
category, trait_name = trait.split('/')  # Expect "behavioral/refusal"
trait_dir = Path(f"experiments/{exp}/extraction/{category}/{trait_name}/extraction")

# Line 78: NEW - Per-experiment
output = Path(f"experiments/{exp}/validation")
```

4. **Update argument parsing:**
```python
# OLD help text:
parser.add_argument('--trait', help='Trait name (refusal, uncertainty, etc.)')

# NEW help text:
parser.add_argument('--trait', help='Trait path: category/name (e.g., behavioral/refusal)')
```

5. **Add validation:**
```python
# After parsing args:
if '/' not in args.trait:
    parser.error("--trait must be in format: category/trait_name (e.g., behavioral/refusal)")
```

6. **Test:**
```bash
python3 scripts/run_cross_distribution.py --trait behavioral/refusal
# Should output to experiments/{exp}/validation/refusal_full_4x4_results.json
```

---

## Git Workflow

```bash
# As you complete each file:
git add <file>
git commit -m "Phase 2: Update <file> to extraction/validation structure

- Changed paths from results/ to {exp}/validation/
- Updated category paths to extraction/{category}/
- [Any specific notes about the file]"

# After all files done:
git commit -m "Phase 2: Complete refactor to extraction/inference/validation

Updated 17 files:
- Visualization (3): serve.py, data-loader.js, state.js
- Analysis (4): run_cross_distribution.py, etc.
- Shell scripts (6): Various extraction scripts
- Documentation (4): main.md, architecture.md, etc.

Removed all fallback logic checking for old structure.
All scripts now enforce extraction/inference/validation hierarchy."
```

---

## Success Criteria

**Phase 2 is complete when:**

- [ ] All 17 files updated and committed
- [ ] Visualizer loads without errors
- [ ] `grep -r "results/cross_distribution_analysis"` returns nothing (except docs/git)
- [ ] No root-level category references remain
- [ ] All scripts accept `category/trait_name` format
- [ ] Documentation accurately reflects new structure
- [ ] No fallback logic remains
- [ ] Tests pass (checklist above)

---

## Quick Reference Commands

```bash
# Search for old patterns
grep -r "results/cross_distribution" --include="*.py" --include="*.js" . | grep -v ".git"
grep -r "experiments/[^/]*/behavioral" --include="*.py" . | grep -v "extraction/"

# Test updated scripts
python3 analysis/cross_distribution_scanner.py
python3 inference/capture_all_layers.py --help
python3 visualization/serve.py

# Check structure
ls experiments/gemma_2b_cognitive_nov20/
tree -L 2 experiments/gemma_2b_cognitive_nov20/

# Git status
git log --oneline -5
git diff --stat
```

---

## Getting Help

**If stuck on a specific file:**
1. Look at the 3 Phase 1 files for working patterns
2. Check this guide's "Example: Updating a File" section
3. Search for similar patterns in working files

**Key files showing correct patterns:**
- `inference/capture_all_layers.py` (lines 107-145, 871)
- `analysis/cross_distribution_scanner.py` (lines 204-219, 315-326)
- `extraction/1_generate_natural.py` (lines 42-55)

---

## Start Here (Copy-Paste Command)

```bash
# Verify Phase 1 complete
git log --oneline -2
# Should show: 876fd40 and 6b6b730

# Start Phase 2 with visualization (highest priority)
grep -n "results/cross_distribution" visualization/serve.py

# Begin updating serve.py to use validation/ path
# Reference: analysis/cross_distribution_scanner.py lines 156-158 for pattern
```

**Good luck with Phase 2!** 🚀
