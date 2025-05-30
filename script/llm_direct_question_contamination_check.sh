#!/bin/bash

models=("llama" "gemma")
datasets=("gossipcop" "politifact")
cache_dir="cache_dataset"

mkdir -p logs

start_time=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Starting contamination check batch at $start_time"

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running contamination check: $model / $dataset"
    python utils/llm_direct_question_contamination_checker.py \
      --model "$model" \
      --dataset_name "$dataset" \
      --cache_dir "$cache_dir" \
      > "logs/${model}_${dataset}_contam_check.log" 2>&1
  done
done

echo "Completed contamination check batch at $(date +"%Y-%m-%d_%H-%M-%S")"