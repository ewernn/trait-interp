# Experiment Data Sync

Sync experiment data (activations, vectors) with R2 cloud storage.

## Setup (One-time)

R2 is already configured via rclone as `r2` remote.

## Usage

### Push local experiments to cloud
```bash
./scripts/sync_push.sh
```

### Pull cloud experiments to local
```bash
./scripts/sync_pull.sh
```

## Workflow

### On your local machine
```bash
# After running experiments
./scripts/sync_push.sh
git add experiments/  # vectors, metadata, responses (small files)
git commit -m "Add experiment results"
git push
```

### On a new machine (e.g., vast.ai)
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/trait-interp
cd trait-interp

# Pull experiment data from R2
./scripts/sync_pull.sh

# Now experiments/ has all activations
python extraction/3_extract_vectors.py --experiment gemma_2b_cognitive_nov20 --trait refusal
```

## What Gets Synced

**Synced to R2** (via `sync_push.sh`/`sync_pull.sh`):
- `experiments/**/activations/*.pt` - Large activation tensors (~1.5GB)
- `experiments/**/vectors/*.pt` - Trait vectors
- `experiments/**/responses/*.json` - Model responses
- All metadata

**Synced to GitHub** (via `git`):
- Code, configs, documentation
- Vectors (small files, already in git)
- Metadata, responses
- **Not activations** (too large, in .gitignore)

## Storage

- **R2 bucket**: `trait-interp-bucket`
- **Public URL**: https://pub-9f8d11fa80ac42a5a605bc23e8aa9449.r2.dev
- **Current size**: ~1.5GB
- **Cost**: Free (under 10GB)

## Tips

- Use `sync_push.sh` after running experiments
- Use `sync_pull.sh` when setting up on new machine
- Sync is incremental (only transfers changed files)
- Progress bar shows upload/download status
