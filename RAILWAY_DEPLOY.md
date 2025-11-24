# Railway Deployment Guide

Deploy the trait-interp visualization to Railway with persistent storage for experiment data.

## Prerequisites

- Railway account (sign up at https://railway.app)
- Railway CLI installed: `npm install -g @railway/cli` or `brew install railway`
- R2 bucket with experiment data already synced (via `utils/sync_push.sh`)

## Quick Overview

1. Create new public repo with visualization code
2. Deploy to Railway from GitHub
3. Create and mount a persistent volume (5GB)
4. One-time: Download experiment data from R2 to volume
5. Done! Data persists forever, even through redeploys

## Step-by-Step

### 1. Create Public Repo

```bash
# Option A: Copy code to new repo
mkdir trait-interp-viz
cd trait-interp-viz
git init

# Copy essential files (not experiments/)
cp -r /path/to/trait-interp/visualization .
cp -r /path/to/trait-interp/config .
cp -r /path/to/trait-interp/analysis .
cp -r /path/to/trait-interp/utils .
cp /path/to/trait-interp/requirements-viz.txt .
cp /path/to/trait-interp/railway.toml .
cp /path/to/trait-interp/Procfile .
cp /path/to/trait-interp/.gitignore .

# Create placeholder for volume mount
mkdir experiments

# Commit and push
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/ewernn/trait-interp-viz.git
git push -u origin main
```

```bash
# Option B: Keep private repo, add public remote
cd /path/to/trait-interp
git remote add public https://github.com/ewernn/trait-interp-viz.git
git push public main
```

### 2. Deploy to Railway

1. Go to https://railway.app/new
2. Click "Deploy from GitHub repo"
3. Select `trait-interp-viz` (or your repo name)
4. Railway auto-detects Python and starts deploying

### 3. Create Persistent Volume

**In Railway dashboard:**

1. Click on your service
2. Go to "Settings" tab
3. Scroll to "Volumes" section
4. Click "New Volume"
   - **Mount Path**: `/app/experiments`
   - **Size**: 5GB (default)
5. Click "Add"
6. Redeploy (click "Deploy" button)

**Note**: Volume is empty at first - you need to populate it (next step).

### 4. Download Data from R2 to Volume (ONE-TIME)

**Install Railway CLI** (if not already):
```bash
npm install -g @railway/cli
# or
brew install railway
```

**Login and link to project:**
```bash
railway login
railway link  # Select your trait-interp-viz project
```

**Configure rclone in Railway environment:**
```bash
# Start an interactive shell in your Railway container
railway run bash

# Inside the container, configure rclone
rclone config

# Follow prompts to set up R2:
# - New remote: "r2"
# - Type: "s3"
# - Provider: "Cloudflare"
# - Enter your R2 credentials (access key, secret key, endpoint)

# Exit the shell
exit
```

**Run the sync script:**
```bash
# This downloads data from R2 to the persistent volume
railway run bash utils/railway_sync_r2.sh
```

This takes ~2-5 minutes to download ~3GB. Progress is shown.

### 5. Verify Deployment

```bash
# Check service logs
railway logs

# Get your public URL
railway status
```

Visit the URL (e.g., `https://trait-interp-production.up.railway.app`) to see your visualization!

## Storage Usage

- **Volume**: ~3GB (experiment data, persists across redeploys)
- **Container**: ~500MB (code + Python dependencies, ephemeral)
- **Total**: ~3.5GB / 5GB limit ✅

## Cost

With Hobby Plan ($5/month):
- 3GB storage = ~$0.75/month
- Minimal compute (simple Python server)
- **Total**: ~$1-2/month (well under $5 included credit)

## Updating Data

To update experiment data after changes:

```bash
# 1. On local machine, sync to R2
bash utils/sync_push.sh

# 2. On Railway, re-sync from R2 to volume
railway run bash utils/railway_sync_r2.sh
```

## Updating Code

```bash
# In your local repo
git add .
git commit -m "Update visualization"
git push origin main      # Push to private repo
git push public main      # Push to public repo (triggers Railway redeploy)
```

Railway auto-redeploys when you push to GitHub. The volume data persists!

## Troubleshooting

**Volume not mounting:**
- Check mount path is exactly `/app/experiments`
- Redeploy after adding volume

**Data not loading:**
```bash
railway run bash
ls -lh /app/experiments/  # Check volume contents
```

**Out of memory:**
- Railway Hobby gives 8GB RAM - should be plenty for the simple server
- Check `railway logs` for errors

**Port issues:**
- Server uses `PORT` env var (Railway sets this automatically)
- See `visualization/serve.py:15`

## Custom Domain (Optional)

1. Buy domain (e.g., `viz.ewernn.com`)
2. In Railway project settings → "Domains"
3. Add custom domain
4. Update DNS records as shown

---

## Architecture Summary

```
GitHub (public repo: code only)
    ↓
Railway (auto-deploy on push)
    ├─ Container (code + Python)
    └─ Volume /app/experiments ← R2 bucket (one-time sync)
```

Your visualization stays online 24/7, data persists across redeploys, and updates via git push!
