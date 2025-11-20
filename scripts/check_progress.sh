#!/bin/bash
# Check parallel extraction progress

EXPERIMENT="gemma_2b_cognitive_nov20"
PROGRESS_DIR="experiments/$EXPERIMENT/.progress"

echo "================================================================"
echo "PARALLEL EXTRACTION PROGRESS"
echo "================================================================"
echo "Time: $(date)"
echo ""

# Count completed stages
total_traits=19
completed=0
in_progress=0
not_started=0

# Check each trait's progress
shopt -s nullglob
for trait_file in $PROGRESS_DIR/*_progress.txt; do
    if [ -f "$trait_file" ]; then
        trait=$(basename "$trait_file" | sed 's/_progress.txt//')

        if grep -q "COMPLETED" "$trait_file"; then
            ((completed++))
            echo "✅ $trait - DONE"
        elif grep -q "VECTORS_DONE" "$trait_file"; then
            ((in_progress++))
            echo "🔄 $trait - Vectors done, finalizing..."
        elif grep -q "ACTIVATIONS_DONE" "$trait_file"; then
            ((in_progress++))
            echo "🔄 $trait - Activations done, extracting vectors..."
        elif grep -q "GENERATION_DONE" "$trait_file"; then
            ((in_progress++))
            echo "🔄 $trait - Generation done, extracting activations..."
        else
            ((in_progress++))
            echo "🔄 $trait - Starting..."
        fi
    fi
done

not_started=$((total_traits - completed - in_progress))

echo ""
echo "================================================================"
echo "SUMMARY"
echo "================================================================"
echo "Completed:    $completed / $total_traits"
echo "In Progress:  $in_progress"
echo "Not Started:  $not_started"
echo ""

# Vector count
vector_count=$(find experiments/$EXPERIMENT/*/*_natural/extraction/vectors -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
expected=$((total_traits * 104))
echo "Vectors: $vector_count / ~$expected"
echo "Progress: $(echo "scale=1; $vector_count * 100 / $expected" | bc)%"
echo ""

# Active processes
active_python=$(ps aux | grep "python extraction" | grep -v grep | wc -l | tr -d ' ')
echo "Active extraction processes: $active_python"

# CPU usage
cpu_idle=$(top -l 1 | grep "CPU usage" | awk '{print $7}' | sed 's/%//')
cpu_used=$(echo "100 - $cpu_idle" | bc)
echo "CPU utilization: ${cpu_used}%"
echo ""

# ETA calculation (rough)
if [ $completed -gt 0 ] && [ $vector_count -gt 0 ]; then
    # Check when first trait completed
    first_completed=$(ls -t $PROGRESS_DIR/*_progress.txt 2>/dev/null | head -1)
    if [ -f "$first_completed" ]; then
        start_time=$(stat -f %B parallel_extraction_*.log 2>/dev/null | head -1)
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))

        if [ $elapsed -gt 0 ] && [ $completed -gt 0 ]; then
            time_per_trait=$((elapsed / completed))
            remaining_traits=$((total_traits - completed))
            eta_seconds=$((remaining_traits * time_per_trait / 3))  # Divide by 3 for parallelism
            eta_hours=$((eta_seconds / 3600))
            eta_mins=$(((eta_seconds % 3600) / 60))

            echo "Estimated time remaining: ${eta_hours}h ${eta_mins}m"
        fi
    fi
fi

echo "================================================================"
