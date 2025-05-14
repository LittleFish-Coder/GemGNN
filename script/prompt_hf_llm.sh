#!/bin/bash

# Models, datasets, and k-shot settings to run
models=("llama" "gemma")
datasets=("gossipcop" "politifact")
k_shots=(3 4 5 6 7 8 9 10 11 12 13 14 15 16)
cache_dir="cache_dataset"

# Create log directory if it doesn't exist
mkdir -p logs

# Track start time
start_time=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Starting batch prompting at $start_time"

# Run all combinations
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for k_shot in "${k_shots[@]}"; do
      echo "Running: $model / $dataset / $k_shot-shot"
      
      # Run the prompting with logging
      python prompt_hf_llm.py \
        --model_type "$model" \
        --dataset_name "$dataset" \
        --k_shot "$k_shot" \
        --cache_dir "$cache_dir" \
        > "logs/${model}_${dataset}_${k_shot}_shot_prompt.log" 2>&1
    done
  done
done

echo "Completed batch prompting at $(date +"%Y-%m-%d_%H-%M-%S")"