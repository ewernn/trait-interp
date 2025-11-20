#!/bin/bash
# Reorganize traits into categorized structure with single-term names

set -e

cd "$(dirname "$0")/.."

EXPERIMENT="gemma_2b_cognitive_nov20"
EXP_DIR="experiments/$EXPERIMENT"

echo "================================================================"
echo "REORGANIZING TRAITS"
echo "================================================================"
echo ""

# Create backup first
echo "Creating backup..."
BACKUP="backup_before_reorganization_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP" "$EXP_DIR/" 2>/dev/null || echo "⚠️  Backup failed"
echo "✓ Backup: $BACKUP"
echo ""

# Create category directories
echo "Creating category directories..."
mkdir -p "$EXP_DIR/behavioral"
mkdir -p "$EXP_DIR/cognitive"
mkdir -p "$EXP_DIR/stylistic"
echo "✓ Categories created"
echo ""

# BEHAVIORAL (5 traits)
echo "Moving BEHAVIORAL traits..."

git mv "$EXP_DIR/refusal" "$EXP_DIR/behavioral/refusal" 2>/dev/null || mv "$EXP_DIR/refusal" "$EXP_DIR/behavioral/refusal"
echo "  ✓ refusal"

git mv "$EXP_DIR/instruction_following" "$EXP_DIR/behavioral/compliance" 2>/dev/null || mv "$EXP_DIR/instruction_following" "$EXP_DIR/behavioral/compliance"
echo "  ✓ instruction_following → compliance"

git mv "$EXP_DIR/sycophancy" "$EXP_DIR/behavioral/sycophancy" 2>/dev/null || mv "$EXP_DIR/sycophancy" "$EXP_DIR/behavioral/sycophancy"
echo "  ✓ sycophancy"

git mv "$EXP_DIR/commitment_strength" "$EXP_DIR/behavioral/confidence" 2>/dev/null || mv "$EXP_DIR/commitment_strength" "$EXP_DIR/behavioral/confidence"
echo "  ✓ commitment_strength → confidence"

git mv "$EXP_DIR/defensiveness" "$EXP_DIR/behavioral/defensiveness" 2>/dev/null || mv "$EXP_DIR/defensiveness" "$EXP_DIR/behavioral/defensiveness"
echo "  ✓ defensiveness"

echo ""

# COGNITIVE (7 traits)
echo "Moving COGNITIVE traits..."

git mv "$EXP_DIR/retrieval_construction" "$EXP_DIR/cognitive/retrieval" 2>/dev/null || mv "$EXP_DIR/retrieval_construction" "$EXP_DIR/cognitive/retrieval"
echo "  ✓ retrieval_construction → retrieval"

git mv "$EXP_DIR/serial_parallel" "$EXP_DIR/cognitive/sequentiality" 2>/dev/null || mv "$EXP_DIR/serial_parallel" "$EXP_DIR/cognitive/sequentiality"
echo "  ✓ serial_parallel → sequentiality"

git mv "$EXP_DIR/local_global" "$EXP_DIR/cognitive/scope" 2>/dev/null || mv "$EXP_DIR/local_global" "$EXP_DIR/cognitive/scope"
echo "  ✓ local_global → scope"

git mv "$EXP_DIR/convergent_divergent" "$EXP_DIR/cognitive/divergence" 2>/dev/null || mv "$EXP_DIR/convergent_divergent" "$EXP_DIR/cognitive/divergence"
echo "  ✓ convergent_divergent → divergence"

git mv "$EXP_DIR/abstract_concrete" "$EXP_DIR/cognitive/abstractness" 2>/dev/null || mv "$EXP_DIR/abstract_concrete" "$EXP_DIR/cognitive/abstractness"
echo "  ✓ abstract_concrete → abstractness"

git mv "$EXP_DIR/temporal_focus" "$EXP_DIR/cognitive/futurism" 2>/dev/null || mv "$EXP_DIR/temporal_focus" "$EXP_DIR/cognitive/futurism"
echo "  ✓ temporal_focus → futurism"

git mv "$EXP_DIR/context_adherence" "$EXP_DIR/cognitive/context" 2>/dev/null || mv "$EXP_DIR/context_adherence" "$EXP_DIR/cognitive/context"
echo "  ✓ context_adherence → context"

echo ""

# STYLISTIC (7 traits)
echo "Moving STYLISTIC traits..."

git mv "$EXP_DIR/emotional_valence" "$EXP_DIR/stylistic/positivity" 2>/dev/null || mv "$EXP_DIR/emotional_valence" "$EXP_DIR/stylistic/positivity"
echo "  ✓ emotional_valence → positivity"

git mv "$EXP_DIR/instruction_boundary" "$EXP_DIR/stylistic/literalness" 2>/dev/null || mv "$EXP_DIR/instruction_boundary" "$EXP_DIR/stylistic/literalness"
echo "  ✓ instruction_boundary → literalness"

git mv "$EXP_DIR/paranoia_trust" "$EXP_DIR/stylistic/trust" 2>/dev/null || mv "$EXP_DIR/paranoia_trust" "$EXP_DIR/stylistic/trust"
echo "  ✓ paranoia_trust → trust"

git mv "$EXP_DIR/power_dynamics" "$EXP_DIR/stylistic/authority" 2>/dev/null || mv "$EXP_DIR/power_dynamics" "$EXP_DIR/stylistic/authority"
echo "  ✓ power_dynamics → authority"

git mv "$EXP_DIR/curiosity" "$EXP_DIR/stylistic/curiosity" 2>/dev/null || mv "$EXP_DIR/curiosity" "$EXP_DIR/stylistic/curiosity"
echo "  ✓ curiosity"

git mv "$EXP_DIR/enthusiasm" "$EXP_DIR/stylistic/enthusiasm" 2>/dev/null || mv "$EXP_DIR/enthusiasm" "$EXP_DIR/stylistic/enthusiasm"
echo "  ✓ enthusiasm"

git mv "$EXP_DIR/formality" "$EXP_DIR/stylistic/formality" 2>/dev/null || mv "$EXP_DIR/formality" "$EXP_DIR/stylistic/formality"
echo "  ✓ formality"

echo ""

# Remove merged confidence variants
echo "Removing merged confidence variants..."

if [ -d "$EXP_DIR/uncertainty_calibration" ]; then
  git rm -rf "$EXP_DIR/uncertainty_calibration" 2>/dev/null || rm -rf "$EXP_DIR/uncertainty_calibration"
  echo "  ✓ Removed uncertainty_calibration"
fi

if [ -d "$EXP_DIR/confidence_doubt" ]; then
  git rm -rf "$EXP_DIR/confidence_doubt" 2>/dev/null || rm -rf "$EXP_DIR/confidence_doubt"
  echo "  ✓ Removed confidence_doubt"
fi

echo ""

# Handle natural variants (if they exist)
echo "Reorganizing natural variants..."

for old_name in refusal emotional_valence uncertainty_calibration formality; do
  natural_dir="${old_name}_natural"

  if [ -d "$EXP_DIR/$natural_dir" ]; then
    case $old_name in
      refusal)
        new_path="behavioral/refusal_natural"
        ;;
      emotional_valence)
        new_path="stylistic/positivity_natural"
        ;;
      uncertainty_calibration)
        # Skip - merging into confidence
        echo "  ⚠️  Skipping uncertainty_calibration_natural (merged into confidence)"
        continue
        ;;
      formality)
        new_path="stylistic/formality_natural"
        ;;
    esac

    git mv "$EXP_DIR/$natural_dir" "$EXP_DIR/$new_path" 2>/dev/null || mv "$EXP_DIR/$natural_dir" "$EXP_DIR/$new_path"
    echo "  ✓ ${natural_dir} → ${new_path}"
  fi
done

echo ""

echo "================================================================"
echo "REORGANIZATION COMPLETE"
echo "================================================================"
echo ""
echo "New structure:"
echo "  behavioral/ (5 traits)"
echo "  cognitive/ (7 traits)"
echo "  stylistic/ (7 traits)"
echo ""
echo "Next steps:"
echo "  1. Verify: ls -R experiments/$EXPERIMENT/"
echo "  2. Commit: git add experiments/ && git commit -m 'Reorganize traits: categorized + single-term names'"
echo "  3. Push: git push"
echo "================================================================"
