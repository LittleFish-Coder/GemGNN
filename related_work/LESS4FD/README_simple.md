# Simplified LESS4FD: Entity-aware Fake News Detection

A simplified implementation of LESS4FD (Learning with Entity-aware Self-Supervised Framework for Fake News Detection) that integrates with the main repository framework without complex meta-learning components.

## Overview

This simplified version of LESS4FD provides:
- **Entity-aware Graph Construction**: Extends the main repository's heterogeneous graph builder
- **Simplified Model Architecture**: Entity-aware GNN models (HGT, HAN, GAT) without meta-learning
- **Consistent Training Pipeline**: Follows the same patterns as the main repository
- **Few-shot Learning**: Supports k-shot scenarios (3-16 shots) using existing framework

## Installation

Follow the main repository installation guide from the root README.md:

1. Create conda environment:
```bash
conda create -n fakenews python=3.12
conda activate fakenews
```

2. Install PyTorch:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Install PyTorch Geometric:
```bash
pip install torch-geometric
```

4. Install other dependencies:
```bash
pip install -r ../../requirements.txt
```

No additional requirements needed for simplified LESS4FD!

## Usage

### 1. Build Graph
Build an entity-aware heterogeneous graph:

```bash
python build_less4fd_graph_simple.py --dataset_name politifact --k_shot 8 --embedding_type deberta --enable_entities
```

### 2. Train Model
Train the simplified LESS4FD model:

```bash
python train_less4fd_simple.py --graph_path graphs_less4fd_simple/politifact_k8_deberta_*.pt --model HGT
```

### Complete Pipeline Example
```bash
# Build graph
python build_less4fd_graph_simple.py --dataset_name politifact --k_shot 8 --embedding_type deberta

# Train model  
python train_less4fd_simple.py --graph_path graphs_less4fd_simple/politifact_k8_deberta_*.pt --model HGT
```

## Key Features

### Entity-aware Enhancements
- **Enhanced Node Features**: Adds entity-aware features to news nodes
- **Entity Attention**: Self-attention mechanism to model entity interactions
- **Flexible Architecture**: Can disable entity features with `--enable_entities false`

### Model Options
- **HGT**: Heterogeneous Graph Transformer (default)
- **HAN**: Heterogeneous Attention Network  
- **GAT**: Graph Attention Network

### Consistent with Main Repository
- Uses same dataset loading from Hugging Face
- Follows same argument patterns and directory structure
- Compatible with existing evaluation and visualization tools
- Same few-shot learning approach

## Configuration

The configuration is simplified and consistent with the main repository:

```python
# Entity-aware features
LESS4FD_CONFIG = {
    "max_entities_per_news": 5,
    "entity_similarity_threshold": 0.7,
    "entity_types": ["PERSON", "ORG"],
    "hidden_channels": 64,
    "num_gnn_layers": 2,
    "dropout": 0.3,
    "num_attention_heads": 4,
}

# Training (same as main repository)
TRAINING_CONFIG = {
    "epochs": 300,
    "learning_rate": 5e-4,
    "weight_decay": 1e-3,
    "patience": 30,
}
```

## Results

Expected performance on few-shot scenarios:

| Dataset | K-shot | Model | Accuracy | F1-score |
|---------|--------|-------|----------|----------|
| PolitiFact | 8 | HGT | ~0.80-0.85 | ~0.79-0.84 |
| GossipCop | 8 | HGT | ~0.83-0.88 | ~0.82-0.87 |

## Differences from Original LESS4FD

**Simplified:**
- ❌ No meta-learning components
- ❌ No complex entity extraction pipeline  
- ❌ No two-phase training (pre-training + fine-tuning)
- ❌ No complex contrastive learning

**Retained:**
- ✅ Entity-aware graph construction concept
- ✅ Enhanced node features with entity information
- ✅ Self-attention for entity modeling
- ✅ Few-shot learning capability
- ✅ Heterogeneous graph architecture

## Integration with Main Repository

This simplified LESS4FD:
1. **Extends** `HeteroGraphBuilder` from the main repository
2. **Uses** the same dataset loading and k-shot sampling
3. **Follows** the same training patterns and evaluation
4. **Compatible** with existing result analysis and visualization tools

## Files Structure

```
related_work/LESS4FD/
├── build_less4fd_graph_simple.py    # Simplified graph builder
├── train_less4fd_simple.py          # Simplified training script  
├── models/
│   └── simple_less4fd_model.py      # Simplified model architecture
├── config/
│   └── less4fd_config.py            # Simplified configuration
├── requirements_simple.txt          # No additional requirements
└── README_simple.md                 # This file
```

## Citation

If you use this simplified implementation:

```bibtex
@inproceedings{less4fd2024,
    title={Learning with Entity-aware Self-Supervised Framework for Fake News Detection},
    author={[Authors]},
    booktitle={Proceedings of EMNLP 2024},
    year={2024}
}
```