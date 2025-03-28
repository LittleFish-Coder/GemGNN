# Models, datasets, and k-shot settings to run
datasets=("gossipcop" "politifact")
embedding_types=("roberta")
# models=("GAT" "GCN" "GraphSAGE" "LSTM" "MLP")
models=("GraphSAGE")
edge_policies=("dynamic_threshold")
# edge_policies=("knn")   
k_shots=(3 4 5 6 7 8 9 10 11 12 13 14 15 16)
# k_neighbors=(5)
alphas=(0.5)

# Create log directory if it doesn't exist
mkdir -p logs

# Track start time
start_time=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Starting training graph at $start_time"

# Run all combinations
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for edge_policy in "${edge_policies[@]}"; do
            for embedding_type in "${embedding_types[@]}"; do
                for k_shot in "${k_shots[@]}"; do
                    for alpha in "${alphas[@]}"; do
                        echo "Running: $model / $dataset / $embedding_type / $k_shot-shot / $edge_policy / $alpha-alpha"
                            
                        # Run the graph training with logging
                        python train_graph.py \
                            --base_model "$model" \
                            --graph "graphs/${dataset}/${k_shot}shot_${embedding_type}_${edge_policy}${alpha}.pt" \
                            > "logs/${model}_${dataset}_${k_shot}shot_${embedding_type}_${edge_policy}_${alpha}.log" 2>&1
                    done
                    # for k_neighbor in "${k_neighbors[@]}"; do
                    #     echo "Running: $model / $dataset / $embedding_type / $k_shot-shot / $edge_policy / $k_neighbor-neighbor"
                        
                    #     python train_graph.py \
                    #         --base_model "$model" \
                    #         --graph "graphs/${dataset}/${k_shot}shot_${embedding_type}_${edge_policy}${k_neighbor}.pt" \
                    #         > "logs/${model}_${dataset}_${k_shot}shot_${embedding_type}_${edge_policy}_${k_neighbor}.log" 2>&1
                    # done    
                done
            done
        done
    done
done

echo "All graph training tasks completed."