# Refactor Phase 2 TODO

Phase 1 completed: Core structure refactored and committed.

## Remaining Files to Update

### Visualization (3 files)
1. **visualization/serve.py**
   - Update cross-dist paths: `results/cross_distribution_analysis` → `{exp}/validation`
   - Update category paths to look in `extraction/{category}/`

2. **visualization/core/data-loader.js**
   - Update API calls to new validation path structure
   - Update trait path resolution

3. **visualization/core/state.js**
   - Update any hardcoded category paths

### Analysis/Validation Scripts (4 files)
4. **scripts/run_cross_distribution.py**
   - Output to `{exp}/validation/` instead of `results/`
   - Read vectors from `extraction/{category}/{trait}/`

5. **scripts/run_extraction_scores.py**
   - Update paths to new structure

6. **scripts/rename_cross_dist_files.sh**
   - Update to work with validation/ directory

7. **scripts/run_all_natural_extraction.sh**
   - Update trait paths to category/trait format

### Shell Scripts (6 files)
8. **scripts/extract_all_instruction_categorized.sh**
9. **scripts/extract_all_missing_categorized.sh**
10. **scripts/extract_all_natural_categorized.sh**
11. **scripts/reorganize_complete.sh**
12. **scripts/reorganize_traits.sh**
13. **scripts/rename_natural_scenarios.sh**

### Documentation (4 files)
14. **docs/main.md**
   - Update directory structure examples
   - Update all path references

15. **docs/architecture.md**
   - Document new three-phase structure

16. **docs/experiments_structure.md**
   - Complete rewrite for new structure

17. **docs/CROSS_DISTRIBUTION_EXPERIMENTS_SUMMARY.md**
   - Update paths

## Cleanup Tasks

### Remove Fallback Logic
Search for and remove all backward compatibility code:

```bash
# Find all fallback checks
grep -r "behavioral/\|cognitive/\|stylistic/" --include="*.py" --include="*.js" .
grep -r "results/cross_distribution" --include="*.py" --include="*.js" .
```

Remove patterns like:
- Checking for both old and new paths
- Try/except for different directory structures
- Comments about "backward compatibility"
- "Legacy" code branches

### Add Fail-Fast Validation
Ensure all scripts:
- Check for `extraction/` directory exists
- Raise clear errors with expected structure
- Don't silently fall back to old paths

## Testing Checklist

After Phase 2:
- [ ] `python3 inference/capture_all_layers.py --help` works
- [ ] `python3 analysis/cross_distribution_scanner.py` succeeds
- [ ] Visualizer loads without errors
- [ ] No references to `results/cross_distribution_analysis` remain
- [ ] No references to root-level categories remain

## Target Structure (Reference)

```
experiments/gemma_2b_cognitive_nov20/
├── extraction/
│   ├── behavioral/
│   │   ├── refusal/
│   │   │   └── extraction/
│   │   │       ├── trait_definition.json
│   │   │       ├── responses/
│   │   │       ├── activations/
│   │   │       └── vectors/
│   │   └── ...
│   ├── cognitive/
│   ├── stylistic/
│   └── alignment/
│
├── inference/
│   ├── prompts/
│   │   ├── general_10.txt
│   │   └── harmful_5.txt
│   ├── raw_activations/
│   │   └── {set_name}/
│   │       └── prompt_0.pt
│   └── projections/
│       └── {category}/{trait}/{set_name}/
│           └── prompt_0.json
│
└── validation/
    ├── refusal_full_4x4_results.json
    ├── uncertainty_full_4x4_results.json
    └── data_index.json
```

## Phase 1 Completed ✓

- [x] Moved categories under extraction/
- [x] Updated inference/capture_all_layers.py
- [x] Updated extraction/1_generate_natural.py
- [x] Updated analysis/cross_distribution_scanner.py
- [x] Committed changes

## Commands for Phase 2

Start new chat or continue this one with:

```
Continue refactor Phase 2:
- Update all visualization files (3 files)
- Update remaining analysis scripts (4 files)
- Update shell scripts (6 files)
- Update documentation (4 files)
- Remove all fallback logic
- Test everything works

Reference: REFACTOR_PHASE2_TODO.md
```
