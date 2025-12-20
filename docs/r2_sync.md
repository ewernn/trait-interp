# R2 Cloud Sync

Sync experiment data between local machine and Cloudflare R2 storage.

## Scripts

| Script | Direction | Use case |
|--------|-----------|----------|
| `utils/r2_push.sh` | Local → R2 | Upload your work |
| `utils/r2_pull.sh` | R2 → Local | Download to new machine |
| `utils/railway_pull_r2.sh` | R2 → Railway | Deploy to Railway volume |

## What Gets Synced

**Included:**
- Vectors (`.pt` in `vectors/`)
- Responses (`pos.json`, `neg.json`)
- Inference projections (`residual_stream/*.json`)
- Steering results
- Metadata files

**Excluded (large, regenerable):**
- `**/activations/**` - extraction activations
- `**/val_activations/**` - validation activations
- `**/inference/raw/**` - raw inference captures

## Push Modes

```bash
./utils/r2_push.sh              # Fast: new files only (default)
./utils/r2_push.sh --full       # Full: check sizes, propagate deletions
./utils/r2_push.sh --checksum   # Slow: check MD5, catch all changes
```

| Mode | New files | Updated files | Deleted files | Speed |
|------|-----------|---------------|---------------|-------|
| (default) | ✓ | ✗ | ✗ | Fast |
| `--full` | ✓ | ✓ (size change) | ✓ | Medium |
| `--checksum` | ✓ | ✓ (any change) | ✓ | Slow |

### When to Use Each Mode

- **Default**: Daily pushes, new experiment runs
- **`--full`**: After deleting files locally, re-running extraction with `--force`
- **`--checksum`**: Rare, only if you overwrote a file with same size

## Common Workflows

### Push new work
```bash
./utils/r2_push.sh
```

### Delete and re-extract a trait
```bash
# 1. Delete locally
rm -rf experiments/gemma-2-2b/extraction/chirp/refusal
rm -rf experiments/gemma-2-2b/steering/chirp/refusal

# 2. Re-run pipeline
python extraction/run_pipeline.py --experiment gemma-2-2b --traits chirp/refusal --force

# 3. Push with deletions
./utils/r2_push.sh --full
```

### Pull to new machine
```bash
./utils/r2_pull.sh
```

### Dry run (see what would change)
```bash
# See what --full would delete/upload
rclone sync experiments/ r2:trait-interp-bucket/experiments/ \
  --size-only --dry-run \
  --exclude "**/activations/**" \
  --exclude "**/val_activations/**" \
  --exclude "**/inference/raw/**"
```

## Technical Details

### Comparison Modes

| Flag | Compares | Notes |
|------|----------|-------|
| `--ignore-existing` | Nothing | Skip if remote file exists |
| `--size-only` | File size | Fast, misses same-size changes |
| `--checksum` | MD5 hash | Slow, catches everything |

### Parallelism Settings

| Mode | Transfers | Checkers | CPU Impact |
|------|-----------|----------|------------|
| Fast | 16 | 16 | Low |
| Full | 8 | 8 | Medium |
| Checksum | 4 | 4 | High |

- **Transfers**: Concurrent uploads/downloads
- **Checkers**: Concurrent file comparisons

### File Count

The repo has ~26k synced files (mostly small JSONs). High parallelism can max out CPU during the check phase.

## Setup

First-time setup:
```bash
./utils/setup_r2.sh
```

Requires R2 credentials (access key, secret, endpoint).
