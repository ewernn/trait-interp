# Remote Instance Workflow - UPDATED

**⚡ Quick Start:** See `REMOTE_QUICK_START.md` for the fastest path!

---

## Current Task: Natural Elicitation for 12 Traits

We've created natural scenarios for 12 traits that need cross-distribution validation.

### One-Line Start

```bash
./scripts/run_all_natural_extraction_a100.sh
```

This runs the complete pipeline for all 12 traits (~3 hours on A100).

---

## What's Happening

**Traits being processed (12 total):**
1. abstract_concrete
2. commitment_strength
3. context_adherence
4. convergent_divergent
5. instruction_boundary
6. local_global
7. paranoia_trust
8. power_dynamics
9. retrieval_construction
10. serial_parallel
11. sycophancy
12. temporal_focus

**Pipeline stages (per trait):**
1. **Generation:** Natural scenarios → Model responses (~10-15 min on A100)
2. **Activations:** Responses → Layer activations (~5 min)
3. **Vectors:** Activations → Trait vectors, all 4 methods × 26 layers (~2 min)
4. **Cross-distribution:** Test 4×4 matrix (Inst↔Nat) (~3 min)

**Total per trait:** ~20-25 min
**Total for 12 traits:** ~3-4 hours

---

## Data Flow

### Input (Already Created)
```
extraction/natural_scenarios/
├── abstract_concrete_positive.txt  (110 prompts)
├── abstract_concrete_negative.txt  (110 prompts)
├── ... (24 files total for 12 traits)
```

### Output (What You'll Generate)
```
experiments/gemma_2b_cognitive_nov20/{trait}_natural/extraction/
├── responses/
│   ├── pos.json  (110 responses)
│   └── neg.json  (110 responses)
├── activations/
│   ├── pos_layer0.pt ... pos_layer25.pt
│   └── neg_layer0.pt ... neg_layer25.pt
└── vectors/
    ├── mean_diff_layer*.pt  (26 files)
    ├── probe_layer*.pt      (26 files)
    ├── ica_layer*.pt        (26 files)
    └── gradient_layer*.pt   (26 files)

results/cross_distribution_analysis/
└── {trait}_full_4x4_results.json  (12 files)
```

---

## Manual Execution (If Script Fails)

Run stages individually for one trait:

```bash
TRAIT="abstract_concrete"
EXP="gemma_2b_cognitive_nov20"

# Stage 1: Generate
python extraction/1_generate_natural.py \
  --experiment $EXP \
  --trait $TRAIT \
  --batch-size 64 \
  --device cuda

# Stage 2: Extract activations
python extraction/2_extract_activations_natural.py \
  --experiment $EXP \
  --trait $TRAIT \
  --device cuda

# Stage 3: Extract vectors
python extraction/3_extract_vectors_natural.py \
  --experiment $EXP \
  --trait $TRAIT

# Stage 4: Cross-distribution
python scripts/run_cross_distribution.py \
  --trait $TRAIT
```

---

## Progress Monitoring

```bash
# Check how many traits have natural vectors
ls experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/ 2>/dev/null | \
  grep "_natural" | \
  wc -l

# Check cross-distribution results
ls results/cross_distribution_analysis/*_full_4x4_results.json | wc -l

# Expected after completion: 15 total (3 existing + 12 new)

# GPU usage
watch -n 1 nvidia-smi
```

---

## Expected Results

### Before (Current State)
```
Cross-distribution validated traits: 3
├── emotional_valence     (high sep)  → Probe 100%
├── refusal              (mod sep)   → Gradient 91.7%
└── uncertainty          (low sep)   → Gradient 96.1%
```

### After (Post-Execution)
```
Cross-distribution validated traits: 15
├── Previous 3 traits
└── New 12 traits (validation TBD)

Expected pattern:
- High separability → Probe dominates
- Low separability  → Gradient dominates
- Moderate         → Gradient slight edge
```

---

## Analysis After Completion

Once the pipeline finishes:

### 1. Generate Summary Tables

```bash
# Top 5 layers for each method
python3 << 'EOF'
import json
from pathlib import Path

results_dir = Path('results/cross_distribution_analysis')

for trait_file in sorted(results_dir.glob('*_full_4x4_results.json')):
    trait = trait_file.stem.replace('_full_4x4_results', '')
    data = json.load(open(trait_file))

    if 'inst_nat' not in data.get('quadrants', {}):
        continue

    inst_nat = data['quadrants']['inst_nat']
    print(f"\n{'='*60}")
    print(f"{trait.upper()}")
    print('='*60)

    for method in ['mean_diff', 'probe', 'ica', 'gradient']:
        if method not in inst_nat['methods']:
            continue

        layers_data = inst_nat['methods'][method].get('all_layers', [])
        if not layers_data:
            continue

        # Top 5 layers by accuracy
        top5 = sorted(layers_data, key=lambda x: x['accuracy'], reverse=True)[:5]

        print(f"\n{method.upper()}:")
        for i, r in enumerate(top5, 1):
            print(f"  #{i}  L{r['layer']:<2d}  {r['accuracy']*100:5.1f}%")
EOF
```

### 2. Update Documentation

Update these files with new findings:
- `results/cross_distribution_analysis/TOP5_LAYERS_CROSS_DISTRIBUTION.txt`
- `results/cross_distribution_analysis/EXTRACTION_SCORES_ALL_TRAITS.txt`
- `docs/insights.md`

---

## Backup Strategy

### During Execution (Every Hour)

```bash
# Backup to R2 (if configured)
./scripts/sync_push.sh

# Or backup to local machine
rsync -avz --progress \
  user@remote-gpu:~/trait-interp/results/ \
  ~/trait-interp/results/

rsync -avz --progress \
  user@remote-gpu:~/trait-interp/experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/ \
  ~/trait-interp/experiments/gemma_2b_cognitive_nov20/
```

### After Completion

```bash
# Full sync
rsync -avz --progress \
  user@remote-gpu:~/trait-interp/ \
  ~/trait-interp/ \
  --exclude='.git' \
  --exclude='__pycache__'
```

---

## Troubleshooting

### Generation Hanging
- Check `nvidia-smi` - should show model loaded (~10GB VRAM)
- Test with batch_size=1 to isolate issue
- Verify HuggingFace token: `cat .env | grep HF_TOKEN`

### Out of Memory
- Reduce batch size in script (64 → 32 → 16)
- Check if multiple processes running: `ps aux | grep python`

### Cross-Distribution Fails
- Verify both instruction and natural vectors exist:
  ```bash
  ls experiments/gemma_2b_cognitive_nov20/abstract_concrete/extraction/vectors/*.pt | wc -l  # Should be 104
  ls experiments/gemma_2b_cognitive_nov20/abstract_concrete_natural/extraction/vectors/*.pt | wc -l  # Should be 104
  ```

### Script Stops Midway
- Check which trait failed: look at last output
- Resume from next trait by editing TRAITS array in script
- Or run remaining traits manually

---

## Cost Tracking

**GPU costs (A100 80GB):**
- Lambda Labs: $1.10/hr
- Vast.ai: $0.80-1.50/hr
- RunPod: $1.39/hr

**This run:**
- Serial (3 hrs): ~$3-5
- Parallel (45 min): ~$1-2

**Total project costs to date:**
- Previous runs: Check instance logs
- This run: ~$3-5
- **Much better than 16 hours on local Mac!**

---

## When Complete

1. ✅ Download all results to local machine
2. ✅ Verify 12 new cross-distribution result files
3. ✅ Generate analysis summaries
4. ✅ Update documentation
5. ✅ Terminate GPU instance (data backed up)
6. ✅ Review findings and plan next experiments

---

## Next Steps After This Run

With 15 traits validated:
- Analyze method selection patterns across separability spectrum
- Identify optimal layers per trait type
- Write up findings for publication
- Potentially run remaining 4 traits (confidence_doubt, curiosity, defensiveness, enthusiasm)

---

**Quick Reference:**
- Start: `./scripts/run_all_natural_extraction_a100.sh`
- Monitor: `watch -n 5 'ls experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/ | grep _natural | wc -l'`
- Backup: `rsync -avz user@remote:~/trait-interp/results/ results/`
