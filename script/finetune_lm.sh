#!/bin/bash

# Models, datasets, and k-shot settings to run
models=("bert-base-uncased" "roberta-base")
datasets=("gossipcop" "politifact")
k_shots=(0 8 16 32 100)

# Create log directory if it doesn't exist
mkdir -p logs

# Track start time
start_time=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Starting batch training at $start_time"

# Run all combinations
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for k_shot in "${k_shots[@]}"; do
      echo "Running: $model / $dataset / $k_shot-shot"
      
      # Run the training with logging
      python finetune_lm.py \
        --model_name "$model" \
        --dataset_name "$dataset" \
        --k_shot "$k_shot" \
        > "logs/${model}_${dataset}_${k_shot}shot.log" 2>&1
    done
  done
done

echo "Completed batch training at $(date +"%Y-%m-%d_%H-%M-%S")"