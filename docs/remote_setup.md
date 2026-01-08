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
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# 3. Clone repo
git clone https://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@github.com/ewernn/trait-interp.git
cd trait-interp

# 4. Persist env vars and aliases for all shells
echo 'export $(cat ~/trait-interp/.env | grep -v "^#" | xargs)' >> ~/.bashrc
echo 'export HF_HOME=~/.cache/huggingface' >> ~/.bashrc
echo "alias cla='claude --dangerously-skip-permissions'" >> ~/.bashrc
echo "alias clar='claude --dangerously-skip-permissions --resume'" >> ~/.bashrc
git config --global user.name "ewernn"
git config --global user.email "ewernn@users.noreply.github.com"
source ~/.bashrc

# 5. Pull data and install deps
./utils/setup_r2.sh
./utils/r2_pull.sh
pip3 install -q -r requirements.txt
```

## When Done

```bash
./utils/r2_push.sh       # Push results back to R2
```

## if i want to make the .sh on remote

cat > temp.sh << 'EOF'
# copy/paste here
EOF