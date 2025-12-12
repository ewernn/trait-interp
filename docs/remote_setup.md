# Remote GPU Setup

Quick setup for Vast.ai / RunPod with PyTorch template.

## Setup

```bash
# 1. Create non-root user (Claude Code needs this)
useradd -m -s /bin/bash dev
echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
su - dev

git clone https://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@github.com/ewernn/trait-interp.git


# 2. Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# 3. Clone repo
git clone https://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@github.com/ewernn/trait-interp.git
cd trait-interp

# 4. Persist env vars for all shells
echo 'export $(cat ~/trait-interp/.env | grep -v "^#" | xargs)' >> ~/.bashrc
echo 'export HF_HOME=~/.cache/huggingface' >> ~/.bashrc
source ~/.bashrc

# 5. Pull data and install deps
./utils/r2_pull.sh       # Auto-configures rclone if needed
pip install -q -r requirements.txt
```

## When Done

```bash
./utils/r2_push.sh       # Push results back to R2
```
