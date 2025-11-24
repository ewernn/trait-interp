# Public Repo Setup Guide

Quick reference for setting up the public visualization repo and dual-remote workflow.

## Overview

**Current setup:**
- Private repo: `trait-interp` (full research codebase + experiments)
- Public repo: `trait-interp-viz` (visualization only, deployed to Railway)

**Workflow:**
```
Work in private repo → Push to both remotes → Railway auto-deploys from public
```

---

## Initial Setup (One-Time)

### Step 1: Create Public Repo Structure

Run the automated script:

```bash
# From trait-interp root
bash utils/create_public_repo.sh
```

This creates `../trait-interp-viz` with:
- ✅ Visualization code only
- ✅ No experiments/ data (empty placeholder)
- ✅ No extraction/training code
- ✅ No secrets or API keys
- ✅ Clean README and .gitignore

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `trait-interp-viz`
3. Visibility: **Public**
4. Don't initialize with README
5. Click "Create repository"

### Step 3: Push to GitHub

```bash
cd ../trait-interp-viz
git add .
git commit -m "Initial commit: visualization dashboard"
git branch -M main
git remote add origin https://github.com/ewernn/trait-interp-viz.git
git push -u origin main
```

### Step 4: Configure Dual-Remote in Private Repo

```bash
cd ../trait-interp  # Back to private repo

# Add public repo as second remote
git remote add public https://github.com/ewernn/trait-interp-viz.git

# Verify
git remote -v
# origin    github.com/ewernn/trait-interp.git (private)
# public    github.com/ewernn/trait-interp-viz.git (public)
```

### Step 5: Make Private Repo Private

1. Go to https://github.com/ewernn/trait-interp/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → "Make private"

---

## Daily Workflow

### Making Changes

Work in the **private repo** as usual:

```bash
cd trait-interp  # Private repo

# Make changes to visualization code
vim visualization/views/new-feature.js

# Commit as normal
git add .
git commit -m "Add new feature"
```

### Deploying Changes

**Option A: Push to both remotes at once**

```bash
# Automated script
bash utils/deploy.sh
```

**Option B: Manual push**

```bash
git push origin main   # Push to private
git push public main   # Push to public (triggers Railway redeploy)
```

**Option C: Selective sync (advanced)**

```bash
# Preview what's different
bash utils/sync_private_to_public.sh

# Only syncs visualization-related files
```

---

## What Gets Synced?

### ✅ Included in Public Repo

- `visualization/` - Frontend + server
- `config/paths.yaml` - Path configuration
- `analysis/check_available_data.py` - Data checker (used by server)
- `utils/railway_sync_r2.sh` - R2 sync script
- `requirements-viz.txt` - Minimal deps
- `railway.toml`, `Procfile` - Deployment config
- `RAILWAY_DEPLOY.md` - Deployment guide

### ❌ Excluded from Public Repo

- `experiments/` - All experiment data (6.5GB, stored in R2)
- `extraction/` - Training pipeline
- `traitlens/` - Core library
- `analysis/vectors/`, `analysis/inference/` - Analysis requiring PyTorch
- `lora/` - LoRA training
- `.env` - Secrets
- `docs/remote_setup.md` - Contains credentials

See `.gitignore-public` for full list.

---

## Security Checklist

Before first push to public, verify:

- [ ] No `.env` files committed
- [ ] No API keys in code
- [ ] No `rclone.conf` with R2 credentials
- [ ] No personal paths or identifiable info
- [ ] No large data files (`.pt`, `.safetensors`)

The automated script handles this, but double-check:

```bash
cd ../trait-interp-viz
grep -r "sk-" . --include="*.py" --include="*.js" --include="*.yaml"  # Check for API keys
```

---

## Railway Deployment

After pushing to public repo:

1. **First time**: Follow [RAILWAY_DEPLOY.md](RAILWAY_DEPLOY.md) completely
2. **Updates**: Railway auto-redeploys on git push

```bash
# Monitor deployment
railway logs --follow
```

---

## Updating Experiment Data

When you add new experiment data to private repo:

```bash
# 1. Sync local → R2 (from private repo)
bash utils/sync_push.sh

# 2. Sync R2 → Railway volume
railway run bash utils/railway_sync_r2.sh

# Data is now live on Railway without redeployment
```

---

## Common Operations

### Check remote status

```bash
git remote -v
```

### Push to private only

```bash
git push origin main
```

### Push to public only

```bash
git push public main
```

### Remove public remote (if needed)

```bash
git remote remove public
```

### Re-add public remote

```bash
git remote add public https://github.com/ewernn/trait-interp-viz.git
```

---

## Troubleshooting

**"public remote not found"**
```bash
git remote add public https://github.com/ewernn/trait-interp-viz.git
```

**Push to wrong branch**
```bash
# Push local main to public main explicitly
git push public main:main
```

**Accidentally committed secrets to public**
```bash
# Nuclear option: force push clean history
cd ../trait-interp-viz
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/secret/file" \
  --prune-empty --tag-name-filter cat -- --all
git push public --force --all
```

**Public repo out of sync**
```bash
# Force update public to match private
git push public main --force
```

---

## Architecture Summary

```
┌─────────────────────────────────────┐
│  Private Repo (trait-interp)        │
│  - Full codebase                    │
│  - Experiments (6.5GB)              │
│  - Secrets (.env)                   │
└──────────┬──────────────────────────┘
           │
           ├─ git push origin main (private)
           │
           └─ git push public main (public)
                      │
           ┌──────────▼──────────────────────┐
           │  Public Repo (trait-interp-viz) │
           │  - Viz code only                │
           │  - No secrets                   │
           │  - No large data                │
           └──────────┬──────────────────────┘
                      │
                      │ Railway auto-deploy
                      │
           ┌──────────▼──────────────────────┐
           │  Railway                        │
           │  - Python server                │
           │  - 5GB volume ← R2 bucket       │
           └─────────────────────────────────┘
```

---

## Quick Commands Cheat Sheet

```bash
# Setup (one-time)
bash utils/create_public_repo.sh
cd ../trait-interp-viz && git remote add origin https://github.com/ewernn/trait-interp-viz.git
git push -u origin main
cd ../trait-interp && git remote add public https://github.com/ewernn/trait-interp-viz.git

# Daily workflow
bash utils/deploy.sh                    # Push to both repos

# Data updates
bash utils/sync_push.sh                 # Local → R2
railway run bash utils/railway_sync_r2.sh  # R2 → Railway

# Status
git remote -v                           # Check remotes
railway status                          # Check Railway deployment
railway logs                            # View logs
```

---

**Questions?** Check [RAILWAY_DEPLOY.md](RAILWAY_DEPLOY.md) for Railway-specific setup.
