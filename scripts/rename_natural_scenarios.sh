#!/bin/bash
# Rename natural scenario files to match new single-term trait names

set -e

cd "$(dirname "$0")/../extraction/natural_scenarios"

echo "================================================================"
echo "RENAMING NATURAL SCENARIO FILES"
echo "================================================================"
echo ""

# Create backup
echo "Creating backup..."
tar -czf "../../backup_natural_scenarios_$(date +%Y%m%d_%H%M%S).tar.gz" *.txt 2>/dev/null
echo "✓ Backup created"
echo ""

echo "Renaming scenario files..."

# Function to rename if file exists
rename_if_exists() {
  local old=$1
  local new=$2

  if [ -f "${old}_positive.txt" ] && [ -f "${old}_negative.txt" ]; then
    mv "${old}_positive.txt" "${new}_positive.txt" 2>/dev/null || true
    mv "${old}_negative.txt" "${new}_negative.txt" 2>/dev/null || true
    echo "  ✓ $old → $new"
  else
    echo "  ⚠️  $old files not found (skipping)"
  fi
}

# Rename all traits
rename_if_exists "abstract_concrete" "abstractness"
rename_if_exists "commitment_strength" "confidence"
rename_if_exists "context_adherence" "context"
rename_if_exists "convergent_divergent" "divergence"
rename_if_exists "emotional_valence" "positivity"
rename_if_exists "instruction_boundary" "literalness"
rename_if_exists "instruction_following" "compliance"
rename_if_exists "local_global" "scope"
rename_if_exists "paranoia_trust" "trust"
rename_if_exists "power_dynamics" "authority"
rename_if_exists "retrieval_construction" "retrieval"
rename_if_exists "serial_parallel" "sequentiality"
rename_if_exists "temporal_focus" "futurism"

echo ""
echo "Removing merged confidence variants..."

# Remove merged traits
if [ -f "uncertainty_calibration_positive.txt" ]; then
  rm -f uncertainty_calibration_*.txt
  echo "  ✓ Removed uncertainty_calibration files"
fi

if [ -f "confidence_doubt_positive.txt" ]; then
  rm -f confidence_doubt_*.txt
  echo "  ✓ Removed confidence_doubt files"
fi

echo ""
echo "================================================================"
echo "RENAMING COMPLETE"
echo "================================================================"
echo ""
echo "Current scenario files:"
ls -1 *.txt | sed 's/_positive.txt\|_negative.txt//' | sort -u
echo ""
echo "Total traits: $(ls -1 *.txt | sed 's/_positive.txt\|_negative.txt//' | sort -u | wc -l)"
echo "Expected: 19"
echo "================================================================"
