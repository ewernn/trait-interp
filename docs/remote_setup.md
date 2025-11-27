# Remote GPU Setup

Quick setup for Vast.ai / RunPod with PyTorch template.

## Setup

```bash
# 1. Create non-root user (Claude Code needs this)
useradd -m -s /bin/bash dev
echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
su - dev

# 2. Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# 3. Clone and setup
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp
source .env              # Load HF_TOKEN, R2 creds
./utils/r2_pull.sh       # Auto-configures rclone if needed, pulls data
pip install -r requirements.txt

# 4. Run experiment
python scripts/em_overnight_experiment.py
```

## When Done

```bash
./utils/r2_push.sh       # Push results back to R2
```

## Current Experiment

See `docs/emergent-misalignment-plan.md` for details.
