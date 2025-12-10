#!/bin/bash
# Download experiments from R2 to Railway volume
# Usage: railway run bash utils/railway_pull_r2.sh [experiment_name]
# Example: railway run bash utils/railway_pull_r2.sh gemma-2-2b-it

set -e

EXPERIMENT=${1:-""}

if [ -n "$EXPERIMENT" ]; then
    SOURCE="r2:trait-interp-bucket/experiments/${EXPERIMENT}/"
    DEST="/app/experiments/${EXPERIMENT}/"
else
    SOURCE="r2:trait-interp-bucket/experiments/"
    DEST="/app/experiments/"
fi

echo "📥 Downloading experiments from R2 to Railway volume..."
echo "Source: $SOURCE"
echo "Destination: $DEST"
echo ""

# Install rclone if not present
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | bash
fi

# Check if rclone config exists
if [ ! -f ~/.config/rclone/rclone.conf ]; then
    echo "❌ Error: rclone not configured"
    echo "You need to set up rclone with your R2 credentials first:"
    echo "  railway run bash"
    echo "  rclone config"
    exit 1
fi

# Sync from R2 to volume
# Exclude large .pt files (activations and raw inference data)
# Keep metadata.json files (small, contain model config)
rclone sync "$SOURCE" "$DEST" \
  --progress \
  --stats 5s \
  --transfers 16 \
  --checkers 16 \
  --exclude "**/inference/raw/**" \
  --exclude "**/activations/**" \
  --exclude "**/val_activations/**"

echo ""
echo "✅ Download complete!"
echo ""
echo "Volume contents:"
du -sh /app/experiments/
echo ""
echo "Data is now persistent and will survive redeploys."
