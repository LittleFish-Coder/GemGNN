#!/bin/bash

# Models, datasets, and k-shot settings to run
datasets=("gossipcop" "politifact")
embedding_types=("bert" "roberta")
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
        for embedding_type in "${embedding_types[@]}"; do
            for k_shot in "${k_shots[@]}"; do
                echo "Running: $model / $dataset / $embedding_type / $k_shot-shot"
                
                # Run the graph training with logging
                python train_graph.py \
                    --base_model "$model" \
                    --graph "graphs/${dataset}/${k_shot}shot_${embedding_type}_knn5.pt" \
                    > "logs/${model}_${dataset}_${k_shot}shot_${embedding_type}_knn5.log" 2>&1
            done
        done
    done
done

echo "All graph training tasks completed."