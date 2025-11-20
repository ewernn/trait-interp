# Shell Scripts - Update Needed

## Remaining Shell Scripts (5 files)

These scripts reference old trait names or paths and need updating based on actual usage:

### 1. scripts/run_all_natural_extraction.sh
**Issue:** Uses old flat trait names (abstract_concrete, commitment_strength, etc.)
**Fix:** These traits have been renamed:
- abstract_concrete → cognitive/abstractness
- commitment_strength → cognitive/confidence
- sycophancy → behavioral/sycophancy
etc.

**Action:** Update TRAITS array to use category/trait_name format OR auto-discover from scanner.

**Example fix:**
```bash
# OLD
TRAITS=("abstract_concrete" "sycophancy")

# NEW - Option 1: Explicit categorization
TRAITS=(
  "cognitive/abstractness"
  "behavioral/sycophancy"
  # etc.
)

# NEW - Option 2: Auto-discover
TRAITS=$(python3 -c "
from pathlib import Path
import json
from analysis.cross_distribution_scanner import scan_experiment
exp_data = scan_experiment(Path('experiments/$EXPERIMENT'))
for t in exp_data['traits']:
    print(t['name'])
")
```

---

### 2-6. Categorized extraction scripts
- scripts/extract_all_instruction_categorized.sh
- scripts/extract_all_missing_categorized.sh
- scripts/extract_all_natural_categorized.sh
- scripts/reorganize_complete.sh
- scripts/reorganize_traits.sh

**Pattern:** All reference paths like `{category}/{trait}` but need to:
1. Use `extraction/{category}/{trait}` when checking directories
2. Pass `category/trait` format to Python scripts
3. Remove any checks for old flat structure

**Key changes:**
```bash
# OLD
for trait_dir in cognitive/*; do

# NEW
for trait_dir in extraction/cognitive/*; do

# When calling Python scripts:
python extraction/1_generate_natural.py --trait cognitive/$(basename $trait_dir)
```

---

## Testing These Scripts

Before running any shell script:
1. Check if it's actually used in your workflow
2. Update paths following patterns above
3. Test on a single trait first
4. The critical Python scripts already work with new structure

---

## Priority

**LOW** - These are batch/utility scripts. The core pipeline works:
- ✅ extraction/1_generate_natural.py (uses category/trait)
- ✅ scripts/run_cross_distribution.py (uses category/trait)
- ✅ analysis/cross_distribution_scanner.py (scans extraction/)
- ✅ Visualizer (reads from extraction/ and validation/)

**Only update these scripts when you need to use them.**
