#!/bin/bash

script_path="/home/littlefish/fake-news-detection/finetune_lm.py"
models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base")
datasets=("TFG" "KDD2020" "GossipCop" "PolitiFact")
k_shots=(0 8 16 32 100)

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for k_shot in "${k_shots[@]}"; do
      echo -e "Running finetune with model: $model and dataset: $dataset with sampling $k_shot-shot data per class"
      python "$script_path" --model "$model" --dataset_name "$dataset" --k_shots "$k_shot"
    done
  done
done