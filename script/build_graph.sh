#!/bin/bash

# Models, datasets, and k-shot settings to run
datasets=("gossipcop" "politifact")
embedding_types=("roberta")
edge_policies=("dynamic_threshold")
alphas=(0.5)
k_shots=(3 4 5 6 7 8 9 10 11 12 13 14 15 16)

# Create log directory if it doesn't exist
mkdir -p logs

# Track start time
start_time=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Starting building graph at $start_time"

# Run all combinations
for dataset in "${datasets[@]}"; do
    for embedding_type in "${embedding_types[@]}"; do
        for edge_policy in "${edge_policies[@]}"; do
            for k_shot in "${k_shots[@]}"; do
                for alpha in "${alphas[@]}"; do
                    echo "Running: $dataset / $embedding_type / $edge_policy / $k_shot-shot / $alpha-alpha"
                    
                # Run the graph building with logging
                python build_graph.py \
                        --dataset_name "$dataset" \
                        --embedding_type "$embedding_type" \
                        --edge_policy "$edge_policy" \
                        --k_shot "$k_shot" \
                        --alpha "$alpha" \
                        --plot \
                        > "logs/${dataset}_${k_shot}shot_${embedding_type}_${edge_policy}_${alpha}.log" 2>&1
                done
            done
        done
    done
done

echo "All graph building tasks completed."