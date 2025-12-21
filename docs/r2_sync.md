# R2 Cloud Sync

## Usage

```bash
./utils/r2_push.sh              # Fast: new files only
./utils/r2_push.sh --full       # Propagates deletions + size-changed overwrites
./utils/r2_push.sh --checksum   # Catches ALL overwrites (slow)

./utils/r2_pull.sh              # Pull from R2 to local
```

## Push Modes

| Mode | New | Overwrites | Deletions |
|------|-----|------------|-----------|
| (default) | ✓ | ✗ | ✗ |
| `--full` | ✓ | size-changed | ✓ |
| `--checksum` | ✓ | all | ✓ |

**When to use each:**
- **Default**: Daily pushes, after new experiment runs
- **`--full`**: After deleting files locally or re-running with `--force`
- **`--checksum`**: Only if you overwrote a file with identical size (rare)

## Key Consideration: copy vs sync

Default mode uses `rclone copy --ignore-existing`:
- Only uploads files that don't exist on R2
- Fast (no comparison needed)
- **Won't propagate deletions or overwrites**

`--full` and `--checksum` use `rclone sync`:
- Makes R2 match local exactly
- Compares all ~26k files (slower, higher CPU)
- **Propagates deletions and overwrites**

## Dry Run

Preview what `--full` would do:
```bash
rclone sync experiments/ r2:trait-interp-bucket/experiments/ \
  --size-only --dry-run \
  --exclude "**/activations/**" \
  --exclude "**/val_activations/**" \
  --exclude "**/inference/raw/**"
```

## Excluded (large, regenerable)

- `**/activations/**`, `**/val_activations/**`
- `**/inference/raw/**`
