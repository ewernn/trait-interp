# Data Sync Workflow

## Data Flow (One Direction Only)

```
LOCAL (source of truth)
  ↓ sync_push.sh
R2 bucket (cloud backup)
  ↓ railway_sync_r2.sh (Railway only)
Railway volume (serving data)
```

**NEVER sync in reverse.** Local is always the source of truth.

---

## Scripts

### ✅ Safe: Local → R2 Push

```bash
bash utils/sync_push.sh
```

**What it does:**
- Pushes `experiments/` to `r2:trait-interp-bucket/experiments/`
- **Includes** `inference/raw/` (raw activations backed up to R2)
- **Excludes** `extraction/*/activations/` (too large, can regenerate)
- One-way: local → cloud only

**Run this:**
- After generating new experiment data
- After extracting new vectors
- Before deploying to Railway

### ✅ Safe: R2 → Railway Pull

**From Railway web terminal:**
1. Go to Railway dashboard
2. Click service → "Shell" tab
3. Run: `bash utils/railway_sync_r2.sh`

**What it does:**
- Downloads R2 data to Railway volume at `/app/experiments`
- **Excludes** `inference/raw/` (too large for Railway, not needed for viz)
- Railway gets ~3GB (projection JSONs, vectors, metadata)
- One-time setup (data persists across redeploys)

**Run this:**
- Once after creating Railway volume
- When updating data on Railway

### ❌ NEVER RUN: R2 → Local Pull

```bash
# DON'T RUN THIS
rclone sync r2:trait-interp-bucket/experiments/ experiments/
```

**Why:**
- Would overwrite your local data with stale R2 data
- Local is always more up-to-date
- Could lose work

**The file `NEVER_RUN_sync_pull.sh` is intentionally disabled.**

---

## Current Setup

**R2 Bucket:** `trait-interp-bucket`
**Public URL:** `https://pub-9f8d11fa80ac42a5a605bc23e8aa9449.r2.dev`
**Status:** Public (make private during job hunt)

**Size Discrepancy:**
- If R2 has more data than local (e.g., 27GB vs 6GB), it contains old/stale data from before exclusions were added
- To clean: `rclone sync experiments/ r2:trait-interp-bucket/experiments/ --delete-excluded` (CAREFUL: deletes stale data)
- Or leave it - R2 storage is cheap (~$0.015/GB/month = $0.40/month for 27GB)

---

## Making R2 Private (Recommended)

1. Cloudflare dashboard → R2 → `trait-interp-bucket` → Settings
2. Disable "Public Access"
3. Add env vars to Railway:
   ```
   R2_ACCESS_KEY_ID=your_key
   R2_SECRET_ACCESS_KEY=your_secret
   ```
4. Update local rclone config with same credentials
5. Everything works the same, just authenticated

---

## Typical Workflow

### Adding New Data

```bash
# 1. Generate/extract locally
python extraction/3_extract_vectors.py --experiment my_exp --trait new_trait

# 2. Push to R2
bash utils/sync_push.sh

# 3. Update Railway (from Railway web terminal)
bash utils/railway_sync_r2.sh
```

### Code Updates

```bash
# 1. Update visualization code
vim visualization/views/new-feature.js

# 2. Push to both repos (triggers Railway redeploy)
bash utils/deploy.sh
```

Data persists through code updates - no need to re-sync.

---

## Safety Features

✅ `sync_push.sh` - Clearly marked as "LOCAL → R2 only"
✅ `NEVER_RUN_sync_pull.sh` - Intentionally disabled, shows error
✅ `railway_sync_r2.sh` - Only runs on Railway (checks for `/app/experiments`)

You can't accidentally overwrite local data.
