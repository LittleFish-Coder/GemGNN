#!/bin/bash

# Models, datasets, and k-shot settings to run
datasets=("gossipcop" "politifact")
models=("GAT" "LSTM" "MLP")
k_shots=(0 8 16 32 100)

# Create log directory if it doesn't exist
mkdir -p logs

# Track start time
start_time=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Starting training graph at $start_time"

# Run all combinations
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for k_shot in "${k_shots[@]}"; do
            echo "Running: $model / $dataset / $k_shot-shot"
            
            # Run the graph building with logging
            python train_graph.py \
                --model "$model" \
                --graph "graphs/${dataset}/${k_shot}shot_knn5.pt" \
                > "logs/${model}_${dataset}_${k_shot}shot_knn5.log" 2>&1
        done
    done
done

echo "All graph training tasks completed."