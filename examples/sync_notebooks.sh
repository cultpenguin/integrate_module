#!/bin/bash

# Sync all Jupyter notebooks with their Python counterparts using jupytext
# This script finds all .ipynb files and syncs them with their .py equivalents

echo "Syncing Jupyter notebooks with Python files..."

# Count total notebooks
total_notebooks=$(find . -name "*.ipynb" -not -path "*/.*" | wc -l)
echo "Found $total_notebooks notebook(s) to sync"

# Counter for processed files
count=0
success_count=0
error_count=0

# Find all .ipynb files (excluding hidden directories like .git)
for notebook in $(find . -name "*.ipynb" -not -path "*/.*"); do
    count=$((count + 1))
    echo "[$count/$total_notebooks] Processing $notebook..."
    
    # Run jupytext sync
    if jupytext --sync "$notebook" 2>/dev/null; then
        echo "  ✓ Synced successfully"
        success_count=$((success_count + 1))
    else
        echo "  ✗ Error syncing $notebook"
        error_count=$((error_count + 1))
    fi
done

echo ""
echo "Sync complete!"
echo "  Successfully synced: $success_count"
echo "  Errors: $error_count"
echo "  Total processed: $count"

if [ $error_count -eq 0 ]; then
    echo "All notebooks synced successfully! ✅"
    exit 0
else
    echo "Some notebooks had sync errors. ⚠️"
    exit 1
fi