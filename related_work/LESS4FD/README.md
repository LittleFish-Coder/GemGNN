# LESS4FD: Learning with Entity-aware Self-Supervised Framework for Fake News Detection

This directory contains the implementation of LESS4FD (Learning with Entity-aware Self-Supervised Framework for Fake News Detection) for few-shot fake news detection using heterogeneous graph neural networks.

## Overview

LESS4FD is a novel framework that combines:
- **Entity-aware Graph Construction**: Builds heterogeneous graphs with news and entity nodes
- **Self-supervised Learning**: Uses contrastive learning and pretext tasks
- **Few-shot Learning**: Supports k-shot scenarios (3-16 shots) for fake news detection
- **Multi-modal Integration**: Combines text embeddings with entity information

## Quick Start

### 1. Installation

```bash
# Install additional dependencies for LESS4FD
pip install -r requirements_less4fd.txt

# Download spaCy model for entity extraction
python -m spacy download en_core_web_sm
```

### 2. Build Entity-aware Heterogeneous Graph

```bash
python build_less4fd_graph.py \
    --dataset_name politifact \
    --k_shot 8 \
    --embedding_type deberta \
    --entity_model bert-base-ner \
    --output_dir graphs_less4fd
```

### 3. Train LESS4FD Model

```bash
python train_less4fd.py \
    --graph_path graphs_less4fd/politifact_8shot.pt \
    --model_type less4fd \
    --hidden_channels 128 \
    --pretrain_epochs 100 \
    --finetune_epochs 50 \
    --learning_rate 1e-4 \
    --output_dir results_less4fd
```

## Architecture

### Graph Structure
```
Node Types:
- news: News articles with text embeddings
- entity: Named entities extracted from news
- interaction: User interactions (if available)

Edge Types:
- news-entity: News contains entity
- entity-entity: Entity co-occurrence/similarity
- news-news: News similarity (KNN)
- news-interaction: News has interactions
```

### Model Components
1. **Entity-aware Encoder**: Processes news and entity nodes with attention
2. **Contrastive Learning Module**: Self-supervised learning with InfoNCE loss
3. **Pretext Task Module**: Entity prediction and masked entity modeling
4. **Few-shot Classifier**: Meta-learning compatible classification head

## Configuration

### Graph Construction Parameters
```python
# Default configuration in config/less4fd_config.py
LESS4FD_CONFIG = {
    "entity_model": "bert-base-ner",      # NER model
    "entity_embedding_dim": 768,          # Entity embedding dimension
    "max_entities_per_news": 10,          # Max entities per article
    "entity_similarity_threshold": 0.7,   # Entity-entity edge threshold
    "contrastive_temperature": 0.07,      # Contrastive learning temperature
    "pretext_weight": 0.3,               # Pretext task loss weight
    "contrastive_weight": 0.4,           # Contrastive loss weight
    "classification_weight": 0.3,        # Classification loss weight
}
```

### Training Parameters
```python
TRAINING_CONFIG = {
    "pretrain_epochs": 100,              # Self-supervised pre-training
    "finetune_epochs": 50,               # Few-shot fine-tuning
    "learning_rate": 1e-4,               # Learning rate
    "weight_decay": 1e-4,                # Weight decay
    "warmup_steps": 1000,                # Learning rate warmup
    "gradient_clip": 1.0,                # Gradient clipping
    "batch_size": 32,                    # Batch size
    "dropout": 0.3,                      # Dropout rate
}
```

## Usage Examples

### Basic Usage

```python
from models.less4fd_model import LESS4FDModel
from utils.entity_extractor import EntityExtractor
from build_less4fd_graph import LESS4FDGraphBuilder

# 1. Build graph
builder = LESS4FDGraphBuilder(
    dataset_name="politifact",
    k_shot=8,
    embedding_type="deberta",
    entity_model="bert-base-ner"
)
graph = builder.build_hetero_graph()

# 2. Initialize model
model = LESS4FDModel(
    data=graph,
    hidden_channels=128,
    num_entities=1000,
    num_classes=2
)

# 3. Train model
trainer = LESS4FDTrainer(model, graph)
trainer.train()
```

### Advanced Usage with Custom Configuration

```python
from config.less4fd_config import LESS4FD_CONFIG

# Custom configuration
config = LESS4FD_CONFIG.copy()
config.update({
    "entity_model": "spacy",
    "max_entities_per_news": 15,
    "contrastive_temperature": 0.1,
    "pretext_weight": 0.4,
})

# Use custom config
builder = LESS4FDGraphBuilder(
    dataset_name="gossipcop",
    k_shot=16,
    embedding_type="roberta",
    config=config
)
```

## Dataset Format

### Input Dataset Structure
The implementation expects datasets in the following format:

```python
{
    "news_id": "unique_id",
    "text": "news content",
    "label": 0,  # 0: real, 1: fake
    "interaction_embeddings_list": [...],  # Optional: user interactions
}
```

### Output Graph Structure
```python
HeteroData(
    # Node features
    news={
        'x': torch.Tensor,           # News embeddings
        'y': torch.Tensor,           # Labels
        'train_mask': torch.Tensor,  # Training mask
        'test_mask': torch.Tensor,   # Test mask
    },
    entity={
        'x': torch.Tensor,           # Entity embeddings
        'type': torch.Tensor,        # Entity types
    },
    
    # Edge indices
    ('news', 'contains', 'entity'): torch.Tensor,
    ('entity', 'co_occurs', 'entity'): torch.Tensor,
    ('news', 'similar', 'news'): torch.Tensor,
)
```

## Training Pipeline

### Phase 1: Pre-training (Self-supervised)
- **Contrastive Learning**: Learn entity-aware representations
- **Pretext Tasks**: Entity prediction, masked entity modeling
- **Entity-News Alignment**: Align entity and news representations

### Phase 2: Fine-tuning (Few-shot)
- **Meta-learning**: Adapt to few-shot scenarios
- **Classification**: Train on k-shot labeled data
- **Evaluation**: Test on unseen data

### Loss Functions
```python
# Combined loss
total_loss = (
    α * contrastive_loss(embeddings, labels) +
    β * pretext_loss(news_emb, entity_emb) +
    γ * classification_loss(logits, targets)
)
```

## Evaluation

### Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance
- **Learning Curves**: Training and validation metrics
- **Few-shot Performance**: Performance across different k values

### Example Evaluation
```bash
# Evaluate on test set
python evaluate_less4fd.py \
    --model_path results_less4fd/best_model.pt \
    --graph_path graphs_less4fd/politifact_8shot.pt \
    --output_file evaluation_results.json
```

## Performance Optimization

### Memory Efficiency
- Entity vocabulary pruning
- Batch processing for large graphs
- Gradient checkpointing for large models

### Speed Optimization
- Pre-computed entity embeddings
- Efficient graph construction
- Parallel entity extraction

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size or entity limit
   --batch_size 16 --max_entities_per_news 5
   ```

2. **Entity Extraction Errors**
   ```bash
   # Use different NER model
   --entity_model spacy
   ```

3. **Poor Performance**
   ```bash
   # Adjust loss weights
   --pretext_weight 0.2 --contrastive_weight 0.5
   ```

### Debug Mode
```bash
# Enable debug logging
python build_less4fd_graph.py --debug --verbose
```

## Integration with Existing Framework

The LESS4FD implementation is designed to integrate seamlessly with the existing heterogeneous graph framework:

- **Compatible Data Format**: Uses same dataset structure
- **Consistent API**: Follows existing code patterns
- **Shared Utilities**: Reuses sampling and evaluation functions
- **Unified Configuration**: Extends existing configuration system

## References

- **LESS4FD Paper**: `2024.emnlp-main.31.pdf`
- **Base Implementation**: `build_hetero_graph.py`, `train_hetero_graph.py`
- **Dataset**: HuggingFace `LittleFish-Coder/Fake_News_*`

## Contributing

1. Follow the existing code style and conventions
2. Add comprehensive docstrings and comments
3. Include unit tests for new components
4. Update documentation for any API changes

## License

This implementation follows the same license as the main project. 