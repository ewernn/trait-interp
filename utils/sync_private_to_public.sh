#!/bin/bash
# Sync specific files from private repo to public repo
# Use this when you update visualization code in private repo
# and want to push those changes to public repo

set -e

echo "🔄 Syncing private → public repo..."
echo ""

# Check we're in the right directory
if [ ! -f "visualization/serve.py" ]; then
    echo "❌ Error: Must run from trait-interp root directory"
    exit 1
fi

# Check if public remote exists
if ! git remote | grep -q "^public$"; then
    echo "❌ Error: 'public' remote not configured"
    echo ""
    echo "Set it up first:"
    echo "  git remote add public https://github.com/ewernn/trait-interp-viz.git"
    exit 1
fi

# Fetch current state of public repo
echo "📥 Fetching public repo state..."
git fetch public

# Check what's changed in visualization/
echo ""
echo "📊 Changes in visualization/ since last sync:"
git diff public/main..HEAD -- visualization/ config/ analysis/check_available_data.py utils/railway_sync_r2.sh requirements-viz.txt railway.toml Procfile RAILWAY_DEPLOY.md

echo ""
read -p "Push these changes to public repo? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Get current branch
    BRANCH=$(git branch --show-current)

    echo "📤 Pushing to public repo..."
    git push public "$BRANCH:main"
    echo "✅ Public repo updated (Railway will auto-redeploy)"
else
    echo "❌ Cancelled"
fi
