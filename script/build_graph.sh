#!/bin/bash

# Models, datasets, and k-shot settings to run
datasets=("gossipcop" "politifact")
edge_policies=("knn" "thresholdnn")
k_shots=(0 8 16 32 100)

# Create log directory if it doesn't exist
mkdir -p logs

# Track start time
start_time=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Starting building graph at $start_time"

# Run all combinations
for dataset in "${datasets[@]}"; do
    for edge_policy in "${edge_policies[@]}"; do
        for k_shot in "${k_shots[@]}"; do
            echo "Running: $dataset / $edge_policy / $k_shot-shot"
            
            # Run the graph building with logging
            python build_graph.py \
                --dataset_name "$dataset" \
                --edge_policy "$edge_policy" \
                --k_shot "$k_shot" \
                --plot \
                > "logs/${dataset}_${edge_policy}_${k_shot}shot.log" 2>&1
        done
    done
done

echo "All graph building tasks completed."