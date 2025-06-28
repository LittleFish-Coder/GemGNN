# GemGNN: Generative Multi-view Interaction Graph Neural Networks for Few-shot Fake News Detection

## Project Overview

This project implements a comprehensive framework for few-shot fake news detection using **Heterogeneous Graph Attention Network (HAN)** with advanced edge construction policies and regularization techniques. The system is specifically designed to perform reliably with very limited labeled data (3-16 shot scenarios) using transductive learning on heterogeneous graph structures.

## Core Workflow and Architecture

### Two-Stage Pipeline

The project follows a clear two-stage approach:

1. **Graph Construction** (`build_hetero_graph.py`): Builds heterogeneous graphs from news datasets
2. **Model Training** (`train_hetero_graph.py`): Trains heterogeneous GNN models on constructed graphs

### Stage 1: Heterogeneous Graph Construction (`build_hetero_graph.py`)

#### Node Types
- **`news`**: Article nodes with text embeddings (BERT, RoBERTa, DeBERTa)
- **`interaction`**: Social interaction nodes with tone/sentiment features

#### Edge Construction Policies

**Primary Edge Policy: KNN with Test Isolation (`knn_test_isolated`)**
- **Training nodes**: Connect to any other training node via KNN similarity
- **Test nodes**: Only connect to training nodes (prevents test data leakage)
- **Rationale**: Maintains realistic evaluation while enabling transductive learning
- **Implementation**: Uses cosine similarity on text embeddings with configurable k-neighbors (3, 5, 7)

**Alternative: Standard KNN (`knn`)**
- All nodes can connect to any other node based on similarity
- Less realistic but serves as baseline comparison

#### Multi-View Graph Construction
- Splits text embeddings into multiple views (0, 3, 6 views tested)
- Creates separate similarity graphs for each sub-embedding view
- Enables capturing different semantic aspects of content

#### Key Features
- **Partial Unlabeled Sampling**: Controls unlabeled node inclusion with factor=5
- **Test Labeled Neighbor Guarantee**: Ensures test nodes have at least one labeled neighbor
- **Dissimilar Edges** (optional): Adds negative similarity edges for richer structure

### Stage 2: Model Training (`train_hetero_graph.py`)

#### Model Architectures

**Primary Model: HAN (Heterogeneous Graph Attention Network)**
- Meta-path based attention mechanism for heterogeneous graphs
- Configurable number of layers (1-2)
- Lightweight and effective for few-shot scenarios

**Alternative Models:**
- **HGT**: Heterogeneous Graph Transformer with residual connections
- **HANv2**: Enhanced HAN with multiple layers, residual connections, and layer normalization

#### Advanced Regularization for Few-Shot Learning

**Label Smoothing (α=0.1)**
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```
- Prevents overconfident predictions in few-shot scenarios
- Improves generalization by softening target distributions

**Hardcoded Overfitting Threshold (0.3)**
```python
OVERFIT_THRESHOLD = 0.3  # Early stopping if validation loss < 0.3
```
- Prevents models from memorizing small training sets
- Provides consistent stopping criteria across experiments

**Increased Dropout (0.3)**
```python
DEFAULT_DROPOUT = 0.3
```
- Stronger regularization for few-shot scenarios
- Reduces overfitting to limited training data

**Loss-Based Model Selection**
- Uses validation loss instead of F1 for model selection
- More stable in few-shot scenarios where F1 can quickly hit 1.0

#### Enhanced Loss Functions
- **Cross-Entropy with Label Smoothing**: Primary loss function
- **Enhanced Loss**: Combines CE, Focal, and Contrastive losses
- **Focal Loss**: Handles class imbalance
- **Robust Few-Shot Loss**: Includes confidence penalty and entropy regularization

## Experimental Design

### Comprehensive Parameter Grid (from `comprehensive_experiments.sh`)

**Core Parameters:**
- **K-shots**: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
- **K-neighbors**: 3, 5, 7
- **Datasets**: PolitiFact, GossipCop
- **Embeddings**: DeBERTa (primary), RoBERTa (alternative)
- **Edge Policies**: `knn_test_isolated`, `knn`
- **Multi-views**: 0, 3, 6

**Optional Features:**
- Enable dissimilar edges (with/without)
- Ensure test labeled neighbor (with/without)
- Partial unlabeled sampling (always enabled with factor=5)

**Total Experiments**: 2,688 parameter combinations

### Transductive Learning Strategy
- **`train_labeled_mask`**: K-shot labeled samples for supervision
- **`train_unlabeled_mask`**: Unlabeled samples for message passing
- **`test_mask`**: Test samples (isolated during edge construction)

## Key Technical Innovations

### 1. Test-Isolated Edge Construction
The `knn_test_isolated` policy ensures that test nodes only connect to training nodes, preventing data leakage while maintaining the benefits of transductive learning. This represents a significant methodological contribution for realistic few-shot evaluation.

### 2. Multi-View Semantic Representation
By splitting embeddings into multiple views and constructing separate similarity graphs, the system captures different semantic aspects of news content, leading to richer graph representations.

### 3. Adaptive Regularization
The combination of label smoothing, hardcoded overfitting thresholds, and increased dropout provides robust regularization specifically tuned for few-shot learning scenarios.

### 4. Heterogeneous Social Context
Integration of social interaction nodes with news content nodes enables the model to leverage both textual and social signals for fake news detection.

## Implementation Best Practices

### Reproducibility
- Fixed random seeds across all components (`DEFAULT_SEED = 42`)
- Cached k-shot sample selections
- Deterministic graph construction pipeline

### Scalability
- Efficient KNN edge construction with cosine similarity
- Configurable batch processing
- Memory-efficient graph storage using PyTorch Geometric

### Robustness
- Comprehensive error handling and logging
- Fallback mechanisms for edge construction
- Extensive parameter validation

## Usage Examples

### Graph Construction
```bash
python build_hetero_graph.py \
  --dataset_name politifact \
  --k_shot 8 \
  --embedding_type deberta \
  --edge_policy knn_test_isolated \
  --k_neighbors 5 \
  --ensure_test_labeled_neighbor \
  --partial_unlabeled \
  --sample_unlabeled_factor 5 \
  --multi_view 3
```

### Model Training
```bash
python train_hetero_graph.py \
  --graph_path graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3/graph.pt \
  --model HAN \
  --loss_fn ce
```

### Comprehensive Experiments
```bash
bash script/comprehensive_experiments.sh
```

## Key Contributions

### 1. Methodological Contributions
- **Test-Isolated Transductive Learning**: Novel approach preventing data leakage while maintaining transductive benefits
- **Multi-View Heterogeneous Graphs**: Enhanced semantic representation through embedding decomposition
- **Few-Shot Specific Regularization**: Tailored regularization techniques for limited data scenarios

### 2. Technical Contributions
- **Comprehensive Edge Construction Policies**: Multiple strategies for graph connectivity
- **Advanced Loss Functions**: Specialized loss functions for few-shot learning
- **Modular Architecture**: Clean separation between graph construction and model training

### 3. Empirical Contributions
- **Extensive Parameter Grid**: Systematic evaluation across 2,688 parameter combinations
- **Cross-Dataset Validation**: Evaluation on both PolitiFact and GossipCop datasets
- **Ablation Studies**: Systematic comparison of different architectural choices

## Future Directions

### 1. Advanced Graph Construction
- Dynamic edge weights based on content similarity
- Temporal edge construction for time-series analysis
- Multi-modal integration (text + images + metadata)

### 2. Enhanced Learning Paradigms
- Meta-learning for rapid adaptation to new domains
- Active learning for optimal sample selection
- Federated learning for privacy-preserving detection

### 3. Robustness and Evaluation
- Adversarial robustness testing
- Cross-domain generalization studies
- Real-time deployment optimization

## Conclusion

This framework provides a robust, scalable approach to few-shot fake news detection through heterogeneous graph neural networks. The combination of test-isolated edge construction, multi-view representations, and specialized regularization techniques offers both theoretical soundness and practical effectiveness.

The modular design enables systematic evaluation of different architectural choices, while the comprehensive experimental framework ensures reproducible and statistically valid results. The methodology is particularly well-suited for real-world scenarios where labeled data is scarce and reliable detection systems are critical.

## Important Implementation Notes

### Command Reference
- **Graph Storage**: Uses PyTorch Geometric HeteroData format
- **Results**: Saved in `results_hetero/` with comprehensive metrics and plots

### File Structure
```
├── build_hetero_graph.py     # Graph construction pipeline
├── train_hetero_graph.py     # Model training pipeline
├── script/
│   └── comprehensive_experiments.sh  # Full experimental grid
├── graphs_hetero/            # Generated heterogeneous graphs
├── results_hetero/           # Training results and metrics
└── logs/                     # Experimental logs
```