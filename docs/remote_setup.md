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

# 3. Clone repo
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp

# 4. Auto-source .env in all terminals (current + future)
echo 'source ~/trait-interp/.env' >> ~/.bashrc
source ~/.bashrc

# 5. Pull data and install deps
./utils/r2_pull.sh       # Auto-configures rclone if needed
pip install -r requirements.txt
```

## When Done

```bash
./utils/r2_push.sh       # Push results back to R2
```
