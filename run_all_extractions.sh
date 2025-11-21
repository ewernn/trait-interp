#!/bin/bash
# Full trait vector extraction pipeline for natural elicitation
# Run this on GPU instance after prompts are generated
#
# Usage:
#   chmod +x run_all_extractions.sh
#   ./run_all_extractions.sh [experiment_name]
#
# Default experiment: gemma_2b_cognitive_nov21

set -e  # Exit on error

# Configuration
EXPERIMENT="${1:-gemma_2b_cognitive_nov21}"
LOG_FILE="extraction_${EXPERIMENT}_$(date +%Y%m%d_%H%M%S).log"

# Traits to extract - 7 validated traits only (Option A)
# confidence, context, formality excluded (using nov20 vectors instead)
TRAITS=(
    # Cognitive state (4 new traits)
    "cognitive_state/uncertainty_expression"
    "cognitive_state/search_activation"
    "cognitive_state/pattern_completion"
    "cognitive_state/correction_impulse"

    # Behavioral tendency (2 fixed traits)
    "behavioral_tendency/retrieval"
    "behavioral_tendency/defensiveness"

    # Expression style (1 fixed trait)
    "expression_style/positivity"
)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Header
log "${BLUE}========================================${NC}"
log "${BLUE}Trait Vector Extraction Pipeline${NC}"
log "${BLUE}========================================${NC}"
log "Experiment: ${EXPERIMENT}"
log "Total traits: ${#TRAITS[@]}"
log "Log file: ${LOG_FILE}"
log "Start time: $(date)"
log ""

# Check Python environment
log "${YELLOW}Checking environment...${NC}"
if ! python -c "import torch; import transformers; import fire" 2>/dev/null; then
    log "${RED}❌ Missing dependencies. Install with:${NC}"
    log "  pip install torch transformers accelerate fire"
    exit 1
fi
log "${GREEN}✅ Environment OK${NC}"
log ""

# Track statistics
SUCCESSFUL=0
FAILED=0
START_TIME=$(date +%s)

# Process each trait
for TRAIT in "${TRAITS[@]}"; do
    TRAIT_START=$(date +%s)

    log "${BLUE}========================================${NC}"
    log "${BLUE}Processing: ${TRAIT}${NC}"
    log "${BLUE}========================================${NC}"

    # Check if prompt files exist
    POSITIVE_FILE="experiments/${EXPERIMENT}/extraction/${TRAIT}/positive.txt"
    NEGATIVE_FILE="experiments/${EXPERIMENT}/extraction/${TRAIT}/negative.txt"

    if [[ ! -f "$POSITIVE_FILE" ]] || [[ ! -f "$NEGATIVE_FILE" ]]; then
        log "${YELLOW}⚠️  Skipping ${TRAIT}: Prompt files not found${NC}"
        log "  Expected: ${POSITIVE_FILE} and ${NEGATIVE_FILE}"
        ((FAILED++))
        continue
    fi

    # Step 1: Generate responses
    log "${YELLOW}[1/4] Generating responses...${NC}"
    if python extraction/1_generate_natural.py \
        --experiment "$EXPERIMENT" \
        --trait "$TRAIT" 2>&1 | tee -a "$LOG_FILE"; then
        log "${GREEN}✅ Generation complete${NC}"
    else
        log "${RED}❌ Generation failed${NC}"
        ((FAILED++))
        continue
    fi

    # Step 2: Extract activations
    log "${YELLOW}[2/4] Extracting activations...${NC}"
    if python extraction/2_extract_activations_natural.py \
        --experiment "$EXPERIMENT" \
        --trait "$TRAIT" 2>&1 | tee -a "$LOG_FILE"; then
        log "${GREEN}✅ Activation extraction complete${NC}"
    else
        log "${RED}❌ Activation extraction failed${NC}"
        ((FAILED++))
        continue
    fi

    # Step 3: Extract vectors (all methods, all layers)
    log "${YELLOW}[3/4] Extracting vectors (all methods, all layers)...${NC}"
    if python extraction/3_extract_vectors_natural.py \
        --experiment "$EXPERIMENT" \
        --trait "$TRAIT" \
        --methods all \
        --layers all 2>&1 | tee -a "$LOG_FILE"; then
        log "${GREEN}✅ Vector extraction complete${NC}"
    else
        log "${RED}❌ Vector extraction failed${NC}"
        ((FAILED++))
        continue
    fi

    # Step 4: Validate vectors
    log "${YELLOW}[4/4] Validating vectors...${NC}"
    if python extraction/validate_natural_vectors.py \
        --experiment "$EXPERIMENT" \
        --trait "$TRAIT" 2>&1 | tee -a "$LOG_FILE"; then
        log "${GREEN}✅ Validation complete${NC}"
        ((SUCCESSFUL++))
    else
        log "${RED}❌ Validation failed${NC}"
        ((FAILED++))
        continue
    fi

    # Calculate trait processing time
    TRAIT_END=$(date +%s)
    TRAIT_DURATION=$((TRAIT_END - TRAIT_START))
    log "${GREEN}✅ ${TRAIT} complete in ${TRAIT_DURATION}s${NC}"
    log ""
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

log "${BLUE}========================================${NC}"
log "${BLUE}EXTRACTION SUMMARY${NC}"
log "${BLUE}========================================${NC}"
log "Total traits: ${#TRAITS[@]}"
log "${GREEN}Successful: ${SUCCESSFUL}${NC}"
log "${RED}Failed: ${FAILED}${NC}"
log "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log "Log file: ${LOG_FILE}"
log "End time: $(date)"

# List extracted vectors
log ""
log "${BLUE}Extracted vectors:${NC}"
find "experiments/${EXPERIMENT}" -name "vectors" -type d | while read dir; do
    count=$(ls "$dir"/*.pt 2>/dev/null | wc -l)
    log "  ${dir}: ${count} vectors"
done

# Exit with appropriate code
if [[ $FAILED -eq 0 ]]; then
    log "${GREEN}🎉 All extractions completed successfully!${NC}"
    exit 0
else
    log "${YELLOW}⚠️  Some extractions failed. Check log for details.${NC}"
    exit 1
fi
