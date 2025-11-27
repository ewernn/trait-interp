# Remote GPU Setup

Quick setup for Vast.ai / RunPod with PyTorch template.

## Setup Commands

```bash
# 1. Create non-root user (Claude Code requires this)
useradd -m -s /bin/bash dev
echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
su - dev

# 2. Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# 3. Clone repo (use your GitHub token for private repo)
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp

# 4. Setup R2 and pull experiment data
chmod +x utils/*.sh
./utils/setup_r2.sh
./utils/r2_pull.sh

# 5. Install Python deps
pip install -r requirements.txt

# 6. Set HuggingFace token
export HF_TOKEN=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
```

## Push Results Back

```bash
./utils/r2_push.sh
```

## Current Experiment

See `docs/emergent-misalignment-plan.md` for the EM validation experiment:

```bash
python scripts/em_overnight_experiment.py
```
