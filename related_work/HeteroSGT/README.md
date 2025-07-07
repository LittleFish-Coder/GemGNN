# HeteroSGT: Heterogeneous Structural Graph Transformer for Few-Shot Fake News Detection

This directory contains a simplified implementation of HeteroSGT (Heterogeneous Structural Graph Transformer) adapted for few-shot fake news detection scenarios.

## Overview

HeteroSGT reframes fake news detection as subgraph classification using a structural graph transformer architecture. The key innovation is incorporating random-walk distance bias into the attention mechanism:

```
Attention(i,j) = softmax((Q_i K_j^T)/sqrt(d) + b_dist(d_ij)) V_j
```

where `b_dist` modulates attention based on random-walk distance `d_ij`, allowing the model to focus on semantically related but multi-hop nodes that traditional GNNs might miss.

## Architecture

- **Subgraph Classification**: Extracts news-centered subgraphs for classification
- **Structural Graph Transformer**: Uses transformer architecture with distance-based attention bias
- **Random-Walk Distance Bias**: Incorporates graph structure into attention computation
- **Heterogeneous Graphs**: Supports news and interaction node types

## Files

- `build_heterosgt_graph.py`: Graph construction pipeline for HeteroSGT
- `train_heterosgt.py`: Training script for HeteroSGT models
- `README.md`: This documentation

## Usage

### 1. Graph Construction

Build heterogeneous graphs for HeteroSGT training:

```bash
python build_heterosgt_graph.py \
  --dataset_name politifact \
  --k_shot 8 \
  --embedding_type deberta \
  --edge_policy knn_test_isolated \
  --k_neighbors 5 \
  --subgraph_size 20 \
  --max_walk_length 4 \
  --output_dir graphs_heterosgt
```

**Parameters:**
- `--dataset_name`: Dataset to use (`politifact`, `gossipcop`)
- `--k_shot`: Number of labeled samples per class (3-16)
- `--embedding_type`: Text embeddings (`bert`, `roberta`, `deberta`, `distilbert`)
- `--edge_policy`: Edge construction policy (`knn`, `knn_test_isolated`)
- `--k_neighbors`: Number of KNN neighbors for edge construction
- `--subgraph_size`: Maximum size of news-centered subgraphs
- `--max_walk_length`: Maximum random walk length for distance computation
- `--output_dir`: Directory to save constructed graphs

### 2. Model Training

Train HeteroSGT models on constructed graphs:

```bash
python train_heterosgt.py \
  --graph_path graphs_heterosgt/heterosgt_politifact_k8_deberta.pt \
  --model_type SGT \
  --hidden_channels 64 \
  --num_layers 2 \
  --num_heads 4 \
  --dropout 0.3 \
  --learning_rate 5e-4 \
  --epochs 300 \
  --output_dir results_heterosgt
```

**Parameters:**
- `--graph_path`: Path to the constructed graph file
- `--model_type`: Model architecture (`SGT`, `HGT`, `HAN` for comparison)
- `--hidden_channels`: Hidden dimension size
- `--num_layers`: Number of transformer layers
- `--num_heads`: Number of attention heads
- `--dropout`: Dropout rate
- `--learning_rate`: Learning rate for optimization
- `--epochs`: Maximum number of training epochs
- `--output_dir`: Directory to save training results

## Example Workflows

### Quick Start (PolitiFact, 8-shot)

```bash
# Build graph
python build_heterosgt_graph.py --dataset_name politifact --k_shot 8

# Train model
python train_heterosgt.py --graph_path graphs_heterosgt/heterosgt_politifact_k8_deberta.pt
```

### Comprehensive Evaluation

```bash
# Different k-shot scenarios
for k_shot in 3 4 5 8 12 16; do
  # Build graph
  python build_heterosgt_graph.py \
    --dataset_name politifact \
    --k_shot $k_shot \
    --embedding_type deberta
  
  # Train model
  python train_heterosgt.py \
    --graph_path graphs_heterosgt/heterosgt_politifact_k${k_shot}_deberta.pt \
    --output_dir results_heterosgt/k${k_shot}
done
```

### Cross-Dataset Evaluation

```bash
# PolitiFact and GossipCop
for dataset in politifact gossipcop; do
  python build_heterosgt_graph.py --dataset_name $dataset --k_shot 8
  python train_heterosgt.py \
    --graph_path graphs_heterosgt/heterosgt_${dataset}_k8_deberta.pt \
    --output_dir results_heterosgt/${dataset}
done
```

## Key Features

### 1. Subgraph-Based Classification
- Extracts news-centered subgraphs for focused analysis
- Configurable subgraph size for computational efficiency
- Preserves local graph structure while reducing complexity

### 2. Distance-Aware Attention
- Incorporates random-walk distances into attention computation
- Learns distance-specific bias parameters
- Captures multi-hop semantic relationships

### 3. Few-Shot Learning Support
- Optimized for limited labeled data scenarios (3-16 shot)
- Test-isolated edge construction to prevent data leakage
- Transductive learning with unlabeled node incorporation

### 4. Heterogeneous Graph Support
- News nodes with text embeddings
- Interaction nodes with social context
- Multiple edge types for rich relationship modeling

## Technical Details

### Graph Construction
- **News Nodes**: Articles with DeBERTa/RoBERTa embeddings
- **Interaction Nodes**: Generated or real user interactions
- **Edges**: KNN similarity-based with test isolation
- **Subgraphs**: Extracted around each news node for classification

### Model Architecture
- **Input Encoding**: Linear projection of node features
- **Transformer Layers**: Multiple SGT layers with distance bias
- **Distance Computation**: Random walk-based distance matrix
- **Classification Head**: MLP for binary fake news prediction

### Training Strategy
- **Loss Function**: Cross-entropy with label smoothing
- **Optimization**: Adam optimizer with weight decay
- **Early Stopping**: Based on validation performance
- **Regularization**: Dropout and L2 weight decay

## Performance Expectations

Based on the thesis results, HeteroSGT should achieve:
- **PolitiFact**: ~0.30 F1-score (baseline in few-shot scenarios)
- **GossipCop**: ~0.29 F1-score (consistent across k-shot settings)

Note: These are reference baselines - actual performance may vary based on implementation details and hyperparameter tuning.

## Dependencies

This implementation uses the same dependencies as the main repository:
- PyTorch & PyTorch Geometric
- Transformers & Datasets (Hugging Face)
- Scikit-learn, NumPy, Matplotlib
- NetworkX for graph algorithms

Install from the main repository root:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `subgraph_size` or `batch_size`
2. **CUDA Issues**: Use `--device cpu` for CPU-only training
3. **Dataset Loading**: Ensure internet connection for first-time dataset download
4. **Graph Size**: Adjust `k_neighbors` if graphs become too large

### Performance Tuning

- **Subgraph Size**: Balance between context and computational cost
- **Walk Length**: Longer walks capture more global structure
- **Attention Heads**: More heads may help with complex patterns
- **Layer Depth**: 2-3 layers typically optimal for few-shot scenarios

## Citation

If you use this implementation, please cite the original HeteroSGT paper and the GemGNN repository:

```bibtex
@article{heterosgt2024,
  title={HeteroSGT: Heterogeneous Structural Graph Transformer for Few-Shot Fake News Detection},
  year={2024},
  note={arXiv:2404.13192v1}
}
```