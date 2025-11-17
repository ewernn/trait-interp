# Quick Start: Remote GPU Instance

Dead simple setup for vast.ai or RunPod.

## Step 1: SSH Into Instance

```bash
ssh root@instance-ip -p PORT
```

## Step 2: Clone Repo

```bash
git clone https://github.com/YOUR_USERNAME/trait-interp
cd trait-interp
```

## Step 3: Run Setup

```bash
bash scripts/configure_r2.sh   # Configure R2 credentials
bash scripts/setup_remote.sh   # Install dependencies & pull data
```

## Step 4: Install Claude Code

```bash
# Option A: Using npm (if available)
npm install -g @anthropic-ai/claude-code

# Option B: Download from website
# Go to: https://claude.ai/download
# Or use curl (check latest version):
curl -L https://github.com/anthropics/claude-code/releases/download/v0.x.x/claude-code-linux -o /usr/local/bin/claude
chmod +x /usr/local/bin/claude
```

## Step 5: Start Claude Code

```bash
claude code
```

## Step 6: Tell Claude

In Claude Code, say:

```
Read scripts/REMOTE_WORKFLOW.md and start working on the tasks described there.
```

That's it! Claude will:
- Run experiments on new traits (refusal, formality)
- Generate natural scenarios
- Extract activations and vectors
- Run 4×4 distribution matrix tests
- Save results to R2

---

## Manual Commands (If Needed)

**Pull latest code:**
```bash
git pull
```

**Pull experiment data:**
```bash
./scripts/sync_pull.sh
```

**Push experiment results:**
```bash
./scripts/sync_push.sh
```

**Check GPU:**
```bash
nvidia-smi
```

**Monitor process:**
```bash
watch -n 1 nvidia-smi  # Live GPU monitoring
htop                    # CPU/Memory monitoring
```

## When Done

```bash
# Push results to R2
./scripts/sync_push.sh

# Verify upload
rclone ls r2:trait-interp-bucket/experiments/gemma_2b_cognitive_nov20/ | grep refusal

# Terminate instance (data is safe in R2)
```

## On Your Local Machine Later

```bash
# Pull results from R2
./scripts/sync_pull.sh

# Analyze locally
python3 analysis/your_analysis.py
```
