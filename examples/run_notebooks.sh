#!/bin/bash

# Set environment variables to suppress progress bars
export TQDM_DISABLE=1
export PYTHONWARNINGS=ignore

echo "Executing notebooks..."
#for notebook in *started_no*.ipynb; do
for notebook in integrate*.ipynb; do
    echo "Processing $notebook..."
    jupyter nbconvert --to notebook --execute --allow-errors --ClearOutputPreprocessor.enabled=True "$notebook" --inplace
    if [ $? -eq 0 ]; then
        echo "✓ $notebook executed successfully"
    else
        echo "✗ Error executing $notebook"
        exit 1
    fi
done
echo "All notebooks executed successfully!"
