# Overnight Local Extraction - Quick Start

## Run Overnight on Your Mac

Simple, reliable, one command:

```bash
cd ~/Desktop/code/trait-interp
nohup ./scripts/run_local_overnight.sh > overnight.log 2>&1 &
echo $! > pipeline.pid
```

This will:
- Run all 12 traits sequentially (safe, no GPU contention)
- Use MPS (Mac GPU) automatically if available
- Log everything to timestamped log file
- Run in background overnight
- Save PID for monitoring

## Check Progress

```bash
# View live progress
tail -f overnight.log

# Count completed traits
ls experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/*_natural_mean_diff_layer16.pt 2>/dev/null | wc -l

# Check if still running
ps aux | grep run_local_overnight
```

## Stop if Needed

```bash
kill $(cat pipeline.pid)
```

## Estimated Time

- Per trait: ~10-15 minutes (Mac M1/M2/M3)
- Total: ~2-3 hours for all 12 traits

## What Gets Created

For each trait (e.g., `refusal_natural`):
```
experiments/gemma_2b_cognitive_nov20/{trait}/extraction/
├── responses/
│   ├── positive.json (110 examples)
│   └── negative.json (110 examples)
├── activations/
│   ├── positive_activations.pt
│   └── negative_activations.pt
└── vectors/
    ├── mean_diff_layer*.pt (26 layers)
    └── probe_layer*.pt (26 layers)
```

Plus cross-distribution test results in `results/cross_distribution_analysis/`

## Resume if Interrupted

The script skips already-completed traits automatically. Just re-run:
```bash
nohup ./scripts/run_local_overnight.sh > overnight_resume.log 2>&1 &
```

## Morning Checklist

```bash
# 1. Check completion
ls experiments/gemma_2b_cognitive_nov20/*/extraction/vectors/*_natural*.pt | wc -l
# Expected: ~624 files (12 traits × 2 methods × 26 layers)

# 2. View summary
tail -30 local_pipeline_*.log

# 3. Upload to R2 (optional)
./scripts/sync_push.sh
```

That's it! Set it and forget it. 🌙
