# Train graph

script_path="/home/littlefish/fake-news-detection/baseline-gnn.py"
models=("GCN" "GAT")
datasets=("TFG" "KDD2020" "GossipCop" "PolitiFact")
k_shots=(0 8 16 32 100)
k_neighbors=(5 7 9)

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for k_shot in "${k_shots[@]}"; do
      for k_neighbor in "${k_neighbors[@]}"; do
        echo -e "Running finetune with model: $model and dataset: $dataset with sampling $k_shot-shot data per class"
        python "$script_path" --model_type "$model" --dataset "$dataset" --k_shot "$k_shot" --k_neighbors "$k_neighbor"
      done
    done
  done
done