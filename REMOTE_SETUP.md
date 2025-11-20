# Remote GPU Setup

5 commands, done in 3 minutes.

---

## Setup

```bash
# 1. Switch to non-root user (required for Claude Code)
su - coder  # or your username

# 2. Clone repo
git clone https://{your_github_token}@github.com/ewernn/trait-interp
cd trait-interp

# 3. Configure R2
bash scripts/configure_r2.sh

# 4. Pull data from R2
./scripts/sync_pull.sh

# 5. Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# 6. Start Claude Code and login
claude code
```

---

## Tell Claude

Once Claude Code opens:

```
Run ./scripts/extract_all_missing_categorized.sh and monitor it.
Fix any errors. Push results to R2 when done.
```

That's it.

---

## Expected Result

- **Time:** ~30 minutes
- **Output:** 3,952 vectors (from current 2,205)
- **Cost:** ~$1

When done, terminate the instance. Data is in R2.
