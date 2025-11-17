#!/bin/bash
# Remote Instance Setup Script
# Run this on vast.ai/RunPod/any remote GPU instance

set -e

echo "🚀 Setting up trait-interp on remote instance..."
echo ""

# Update system
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y -qq git curl wget build-essential > /dev/null 2>&1

# Install rclone
echo "☁️  Installing rclone..."
curl https://rclone.org/install.sh | bash > /dev/null 2>&1

# Check if rclone is configured
if [ ! -f ~/.config/rclone/rclone.conf ]; then
    echo ""
    echo "⚠️  rclone not configured yet!"
    echo ""
    echo "You need to configure R2 access. Run:"
    echo "  rclone config"
    echo ""
    echo "Then add a remote named 'r2' with these settings:"
    echo "  Type: s3"
    echo "  Provider: Cloudflare"
    echo "  Access Key ID: [from Cloudflare]"
    echo "  Secret Access Key: [from Cloudflare]"
    echo "  Endpoint: https://c3179b301e770b99a1ce094df7a4e5c1.r2.cloudflarestorage.com"
    echo ""
    echo "After configuring, re-run this script."
    exit 1
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt
else
    echo "⚠️  No requirements.txt found, skipping..."
fi

# Pull experiment data from R2
echo "📥 Pulling experiment data from R2..."
if [ -x scripts/sync_pull.sh ]; then
    ./scripts/sync_pull.sh
else
    echo "⚠️  sync_pull.sh not found, skipping..."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Install Claude Code (if not already installed)"
echo "  2. Run: claude code"
echo "  3. Tell Claude to read: scripts/REMOTE_WORKFLOW.md"
echo ""
