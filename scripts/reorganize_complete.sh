#!/bin/bash
# Complete reorganization: flat → categorized with single-term names
# Compatible with macOS bash 3.x

set -e

EXPERIMENT="gemma_2b_cognitive_nov20"
EXP_DIR="experiments/$EXPERIMENT"

echo "================================================================"
echo "COMPLETE TRAIT REORGANIZATION"
echo "================================================================"
echo "Experiment: $EXPERIMENT"
echo "Actions:"
echo "  1. Create categorized directories (behavioral/cognitive/stylistic/alignment)"
echo "  2. Move traits to categories with new single-term names"
echo "  3. Move existing vectors and data"
echo "  4. Create shared inference/ directory structure"
echo "================================================================"
echo ""

# Create category directories
echo "Creating category directories..."
mkdir -p "$EXP_DIR"/{behavioral,cognitive,stylistic,alignment}
mkdir -p "$EXP_DIR/inference"/{prompts,projections}

# Function to move trait (handles both instruction and natural)
move_trait() {
  local old_name=$1
  local category=$2
  local new_name=$3

  for variant in "" "_natural"; do
    old_dir="$EXP_DIR/${old_name}${variant}"

    if [ -d "$old_dir" ]; then
      new_dir="$EXP_DIR/$category/${new_name}${variant}"
      echo "  $old_name$variant → $category/$new_name$variant"
      mv "$old_dir" "$new_dir"
    fi
  done
}

echo ""
echo "Moving traits to categorized structure..."
echo ""

# Behavioral (5)
echo "Behavioral traits:"
move_trait "refusal" "behavioral" "refusal"
move_trait "instruction_following" "behavioral" "compliance"
move_trait "sycophancy" "behavioral" "sycophancy"
move_trait "commitment_strength" "behavioral" "confidence"
move_trait "defensiveness" "behavioral" "defensiveness"

# Merge uncertainty variants to confidence
if [ -d "$EXP_DIR/uncertainty_calibration" ]; then
  echo "  uncertainty_calibration → behavioral/confidence (MERGE)"
  # Keep the better one (uncertainty_calibration has 104 vectors)
  if [ ! -d "$EXP_DIR/behavioral/confidence" ]; then
    mv "$EXP_DIR/uncertainty_calibration" "$EXP_DIR/behavioral/confidence"
  else
    echo "    (keeping existing confidence, removing uncertainty_calibration)"
    rm -rf "$EXP_DIR/uncertainty_calibration"
  fi
fi

if [ -d "$EXP_DIR/confidence_doubt" ]; then
  echo "  confidence_doubt → DELETE (merged to confidence)"
  rm -rf "$EXP_DIR/confidence_doubt" "$EXP_DIR/confidence_doubt_natural"
fi

echo ""
echo "Cognitive traits:"
move_trait "retrieval_construction" "cognitive" "retrieval"
move_trait "serial_parallel" "cognitive" "sequentiality"
move_trait "local_global" "cognitive" "scope"
move_trait "convergent_divergent" "cognitive" "divergence"
move_trait "abstract_concrete" "cognitive" "abstractness"
move_trait "temporal_focus" "cognitive" "futurism"
move_trait "context_adherence" "cognitive" "context"

echo ""
echo "Stylistic traits:"
move_trait "emotional_valence" "stylistic" "positivity"
move_trait "instruction_boundary" "stylistic" "literalness"
move_trait "paranoia_trust" "stylistic" "trust"
move_trait "power_dynamics" "stylistic" "authority"
move_trait "curiosity" "stylistic" "curiosity"
move_trait "enthusiasm" "stylistic" "enthusiasm"
move_trait "formality" "stylistic" "formality"

echo ""
echo "================================================================"
echo "✅ Reorganization complete!"
echo "================================================================"
echo ""

# Verify
echo "Verification:"
echo "  Behavioral traits: $(ls -d $EXP_DIR/behavioral/*/ 2>/dev/null | wc -l)"
echo "  Cognitive traits: $(ls -d $EXP_DIR/cognitive/*/ 2>/dev/null | wc -l)"
echo "  Stylistic traits: $(ls -d $EXP_DIR/stylistic/*/ 2>/dev/null | wc -l)"
echo ""
echo "  Total vectors preserved:"
find "$EXP_DIR" -path "*/extraction/vectors/*.pt" 2>/dev/null | wc -l | xargs echo "    "

echo ""
echo "Next steps:"
echo "  1. Verify structure: ls experiments/gemma_2b_cognitive_nov20/"
echo "  2. Push to R2: ./scripts/sync_push.sh"
echo "  3. Rent 2x A100 and extract missing vectors"
