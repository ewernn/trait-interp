#!/bin/bash
# Auto-configure rclone with R2 credentials from .env
set -e

# Load .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
elif [ -f ../.env ]; then
    export $(cat ../.env | grep -v '^#' | xargs)
else
    echo "❌ .env not found"
    exit 1
fi

# Create rclone config directory
mkdir -p ~/.config/rclone

# Write rclone config
cat > ~/.config/rclone/rclone.conf << EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = ${R2_ENDPOINT}
acl = private
EOF

echo "✅ rclone configured for R2"
echo "   Bucket: ${R2_BUCKET_NAME}"
echo ""
echo "Test with: rclone ls r2:${R2_BUCKET_NAME}/"
