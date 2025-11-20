#!/bin/bash
# Trait category mappings for reorganized structure

# Function to get category for a trait
get_category() {
  local trait=$1

  # Remove _natural suffix if present
  trait=${trait%_natural}

  case $trait in
    # BEHAVIORAL
    refusal|compliance|sycophancy|confidence|defensiveness)
      echo "behavioral"
      ;;

    # COGNITIVE
    retrieval|sequentiality|scope|divergence|abstractness|futurism|context)
      echo "cognitive"
      ;;

    # STYLISTIC
    positivity|literalness|trust|authority|curiosity|enthusiasm|formality)
      echo "stylistic"
      ;;

    *)
      echo "ERROR: Unknown trait: $trait" >&2
      exit 1
      ;;
  esac
}

# Function to get full path for a trait
get_trait_path() {
  local experiment=$1
  local trait=$2

  local category=$(get_category "$trait")
  echo "experiments/$experiment/$category/$trait"
}

# Export functions
export -f get_category
export -f get_trait_path
