# Experiment: LIARS' BENCH — Next Steps

## Context

HP-KR supervised/zero-label eval complete (see `docs/viz_findings/liars-bench-deception.md`). Cross-eval on 5 datasets done. Key findings: two-vector complementarity (lying + concealment), layer smoothness diagnostic, deployment-realistic L30 table.

**Current scorecard at L30:**
- SOLVED: CG (0.991), ID (0.958), IT (0.870), HP-KR (0.864/0.988 supervised)
- WEAK: HPC (0.706)
- FAILED: GS (0.483)
- Paper beats us on IT (0.93 vs 0.870)

## Step 1: GS Supervised Probe (ceiling test)

**Purpose**: Can a supervised probe detect GS deception at all? If no → representation isn't linearly accessible, skip GS. If yes → zero-label vectors just need better scenarios.

**Requires**: `gs_activations.pt` on GPU machine.

**1a: Check if activations exist on GPU**
```bash
ls experiments/bullshit/results/gs_activations.pt
```

**1b: If not, regenerate**
```bash
python experiments/bullshit/scripts/extract_liars_bench_activations.py \
    --dataset gs \
    --load-in-4bit
```

**1c: Run supervised eval**
```bash
python experiments/bullshit/scripts/evaluate_liars_bench_protocol.py \
    --dataset gs --quick
```

**Output**: `experiments/bullshit/results/gs_eval.json`

**Decision point**:
- AUROC > 0.8 → GS is linearly detectable, worth extracting a targeted trait
- AUROC 0.6-0.8 → weak but present, might need nonlinear methods
- AUROC < 0.6 → skip GS, representation not there

## Step 2: Steering Eval (both traits, updated judge)

**Purpose**: Re-evaluate steering with updated judge. Previous L30 results had artificially high coherence (model not responding to question). Test layers 10-40 where steering actually works.

**2a: bs/concealment**
```bash
python analysis/steering/evaluate.py \
    --experiment bullshit \
    --vector-from-trait bullshit/bs/concealment \
    --layers 10,15,20,25,30,35,40 \
    --load-in-4bit \
    --no-server
```

**2b: bs/lying**
```bash
python analysis/steering/evaluate.py \
    --experiment bullshit \
    --vector-from-trait bullshit/bs/lying \
    --layers 10,15,20,25,30,35,40 \
    --load-in-4bit \
    --no-server
```

**What to check**:
- Which layers actually steer (trait score change + coherence ≥ 70 + model responds to question)?
- Does the updated judge give lower coherence at L30+ (confirming the model doesn't engage)?
- Do L15-20 emerge as the real steering sweet spot?

## Step 3: Regenerate missing activations for full cross-eval

**Purpose**: Get HP-KR and ST into the cross-eval pipeline for per-layer AUROC curves.

**3a: HP-KR activations** (if not on disk)
```bash
python experiments/bullshit/scripts/extract_liars_bench_activations.py \
    --dataset hpkr \
    --load-in-4bit
```

**3b: ST activations** (time, greeting, gender — if not on disk)
```bash
for ds in st_time st_greeting st_gender; do
    python experiments/bullshit/scripts/extract_liars_bench_activations.py \
        --dataset $ds \
        --model-variant lora_${ds#st_} \
        --load-in-4bit
done
```

**3c: Update cross-eval script**
Add HP-KR and ST to `DATASETS` list in `cross_eval_all_datasets.py` line 26.

**3d: Run cross-eval for both traits on all 7+3 datasets**
```bash
python experiments/bullshit/scripts/cross_eval_all_datasets.py --trait bs/concealment
python experiments/bullshit/scripts/cross_eval_all_datasets.py --trait bs/lying
```

## Step 4 (conditional): Third trait extraction

Only if GS supervised probe shows AUROC > 0.7 in Step 1.

**Candidate concept**: "self-knowledge concealment" — scenarios where someone knows a fact about the person they're talking to but denies it.

**Positive scenarios**: "The assistant knows the user's [attribute] but deflects when asked"
**Negative scenarios**: "The assistant genuinely doesn't know the user's [attribute] and says so"

Write scenarios to `datasets/traits/bs/self_knowledge_concealment/`, then:
```bash
python extraction/run_pipeline.py \
    --experiment bullshit \
    --traits bs/self_knowledge_concealment \
    --position "response[:5]"
```

## Priority Order

1. **Step 1** (GS ceiling test) — determines whether Step 4 is worth doing
2. **Step 2** (steering) — independent, can run in parallel with Step 1
3. **Step 3** (missing activations) — completes the dataset coverage
4. **Step 4** (third trait) — only if Step 1 is promising

## Notes

- Using nf4 quantization (`--load-in-4bit`) to fit 70B on available VRAM
- Steering eval uses `gpt-4.1-mini` judge — requires `OPENAI_API_KEY`
- Cross-eval is CPU-only (just projections on existing activations)
- Previous PLAN.md covered the original HP-KR experiment (complete, results in findings doc)
