#!/bin/bash
# Dual-remote deployment script
# Pushes changes to both private (origin) and public (public) repos

set -e

echo "🚀 Deploying to both repositories..."
echo ""

# Check if we have uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Error: You have uncommitted changes"
    echo "Please commit or stash them first:"
    echo "  git add ."
    echo "  git commit -m 'Your message'"
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

# Get current branch
BRANCH=$(git branch --show-current)

echo "📤 Pushing to private repo (origin)..."
git push origin "$BRANCH"
echo "✅ Private repo updated"
echo ""

echo "📤 Pushing to public repo (public)..."
git push public "$BRANCH"
echo "✅ Public repo updated (Railway will auto-redeploy)"
echo ""

echo "🎉 Deployment complete!"
echo ""
echo "🔍 Check status:"
echo "  Railway: https://railway.app/project/your-project-id"
echo "  Public repo: https://github.com/ewernn/trait-interp-viz"
echo ""
