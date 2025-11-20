# Remote Setup (3 commands)

SSH into your 8x A100 instance and run:

```bash
# 1. Switch to non-root user (required for Claude Code)
su - coder

# 2. Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# 3. Clone repo and start Claude Code
git clone https://github.com/ewernn/trait-interp
cd trait-interp
claude code
```

**That's it.** When Claude Code opens, paste REMOTE_INSTRUCTIONS.md and let it do everything.
