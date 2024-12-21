
script_path="/home/littlefish/fake-news-detection/finetune_lm.py"
models=("bert-base-uncased" "bart-base" "roberta-base")
datasets=("TFG" "KDD2020" "GossipCop" "PolitiFact")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running finetune with model: $model and dataset: $dataset"
    python "$script_path" --model "$model" --dataset_name "$dataset"
  done
done