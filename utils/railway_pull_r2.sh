#!/bin/bash
# Download experiments from R2 to Railway volume
# Uses shared r2_config.sh for excludes and rclone setup.
#
# Usage:
#   railway ssh  (then from /app):
#   bash utils/railway_pull_r2.sh --only gemma-2-2b
#   bash utils/railway_pull_r2.sh --only gemma-2-2b,gemma-3-4b
#   bash utils/railway_pull_r2.sh                              # all experiments
#   bash utils/railway_pull_r2.sh --copy                       # new + changed files
#   bash utils/railway_pull_r2.sh --dry-run --only gemma-2-2b  # preview

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Install rclone if not present (Railway containers don't have it)
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl -s https://rclone.org/install.sh | bash
fi

# Source shared R2 config (excludes, arg parsing, rclone check)
MODE="safe"
source "$SCRIPT_DIR/r2_config.sh"

parse_r2_args "$@"
ensure_r2
resolve_paths
build_excludes
build_only_filters

# Display what we're doing
echo "📥 Pulling to Railway volume..."
if [[ -n "$ONLY" ]]; then
    echo "Experiment(s): $ONLY"
else
    echo "Scope: all experiments"
fi
[[ "$INCLUDE_LORAS" == true ]]        && echo "  + LoRAs included"
[[ "$INCLUDE_ARCHIVE" == true ]]      && echo "  + Archive included"
[[ "$INCLUDE_TRAJECTORIES" == true ]] && echo "  + Trajectories included"

COMMON_FLAGS=(
    --progress
    --stats 5s
    --fast-list
    $DRY_RUN
    "${EXCLUDES[@]}"
    "${ONLY_FILTERS[@]}"
)

case $MODE in
    safe)
        echo "Mode: SAFE (new files only)"
        echo ""
        rclone copy "$R2_REMOTE" "$LOCAL_DIR" \
            --ignore-existing \
            --transfers 32 \
            --checkers 64 \
            "${COMMON_FLAGS[@]}"
        ;;
    copy)
        echo "Mode: COPY (new + changed files)"
        echo ""
        rclone copy "$R2_REMOTE" "$LOCAL_DIR" \
            --size-only \
            --transfers 16 \
            --checkers 32 \
            "${COMMON_FLAGS[@]}"
        ;;
    full)
        echo "Mode: FULL (mirror R2, deletes local-only files)"
        echo ""
        rclone sync "$R2_REMOTE" "$LOCAL_DIR" \
            --size-only \
            --modify-window 1s \
            --transfers 32 \
            --checkers 64 \
            "${COMMON_FLAGS[@]}"
        ;;
esac

echo ""
echo "✅ Pull complete!"
du -sh experiments/ 2>/dev/null || true
