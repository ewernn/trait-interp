#!/bin/bash
# Push experiments to R2 cloud storage
# This syncs local experiments/ to R2, uploading only changed files

set -e

echo "📤 Pushing experiments to R2..."
echo "Source: experiments/"
echo "Destination: r2:trait-interp-bucket/experiments/"
echo ""

# Sync experiments to R2 (one-way: local → cloud)
rclone sync experiments/ r2:trait-interp-bucket/experiments/ \
  --progress \
  --stats 5s \
  --transfers 4 \
  --checkers 8 \
  --exclude "*.pyc" \
  --exclude "__pycache__/**" \
  --exclude ".DS_Store"

echo ""
echo "✅ Push complete!"
echo ""
echo "Verify at: https://pub-9f8d11fa80ac42a5a605bc23e8aa9449.r2.dev"
