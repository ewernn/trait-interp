# Remote Setup (2 commands)

SSH into your 8x A100 instance and run:

```bash
# 1. Switch to non-root user and install Claude Code
su - user
curl -fsSL https://claude.ai/install.sh | bash

# 2. Clone repo and start Claude Code
git clone https://github.com/ewernn/trait-interp && cd trait-interp && claude --dangerously-skip-permissions
```

**That's it.** When Claude Code opens, paste the contents of `REMOTE_INSTRUCTIONS.md` and let it do everything.
