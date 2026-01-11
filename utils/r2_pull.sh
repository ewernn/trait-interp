#!/bin/bash
# Pull experiments from R2 cloud storage
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check if rclone is configured for R2
if ! rclone listremotes | grep -q "^r2:"; then
    echo "⚙️  R2 not configured, running setup..."
    "$SCRIPT_DIR/setup_r2.sh"
fi

echo "📥 Pulling experiments from R2..."
echo "Note: Only pulls new files, won't delete local files not in R2"
echo ""

rclone copy r2:trait-interp-bucket/experiments/ experiments/ \
  --progress \
  --stats 5s \
  --ignore-existing \
  --transfers 32 \
  --checkers 64 \
  --exclude "*.pyc" \
  --exclude "__pycache__/**" \
  --exclude ".DS_Store" \
  --exclude "**/activations/**" \
  --exclude "**/inference/raw/**" \

echo ""
echo "✅ Pull complete!"
