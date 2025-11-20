#!/bin/bash
# Rename cross-distribution files to match new categorized trait names

cd results/cross_distribution_analysis

# Mapping: old_name → new_name
declare -A renames=(
    ["abstract_concrete"]="abstractness"
    ["commitment_strength"]="confidence"
    ["context_adherence"]="context"
    ["convergent_divergent"]="divergence"
    ["emotional_valence"]="positivity"
    ["paranoia_trust"]="trust"
    ["power_dynamics"]="authority"
    ["retrieval_construction"]="retrieval"
    ["serial_parallel"]="sequentiality"
    ["temporal_focus"]="futurism"
    ["uncertainty_calibration"]="confidence"  # Was merged with commitment_strength
    ["instruction_boundary"]="compliance"    # Renamed to compliance
)

# Rename files
for old in "${!renames[@]}"; do
    new="${renames[$old]}"

    # Check if old files exist
    if ls ${old}_*.json 2>/dev/null | grep -q .; then
        echo "Renaming $old → $new"

        # Rename all variants (full_4x4, etc.)
        for file in ${old}_*.json; do
            newfile=$(echo "$file" | sed "s/^${old}_/${new}_/")
            mv "$file" "$newfile"
            echo "  $file → $newfile"
        done
    fi
done

# Keep these as-is (names didn't change):
# - refusal
# - sycophancy
# - curiosity
# - defensiveness
# - enthusiasm
# - formality

echo ""
echo "✅ Renaming complete!"
echo "Files now match categorized trait structure"
