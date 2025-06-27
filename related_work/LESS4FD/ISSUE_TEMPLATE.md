# Issue: Implement LESS4FD Architecture for Few-Shot Fake News Detection

## Overview
Implement the LESS4FD (Learning with Entity-aware Self-Supervised Framework for Fake News Detection) architecture as described in the research paper. This implementation should follow few-shot learning scenarios and integrate with the existing heterogeneous graph framework.

## Requirements

### Core Architecture Components
- **Entity-aware Graph Construction**: Build heterogeneous graphs with news nodes and entity nodes
- **Self-supervised Learning**: Implement contrastive learning and entity-aware pretext tasks
- **Few-shot Learning**: Support k-shot scenarios (3-16 shots) with proper sampling
- **Multi-modal Integration**: Combine text embeddings with entity information

### File Structure
```
related_work/LESS4FD/
├── build_less4fd_graph.py      # Graph construction for LESS4FD
├── train_less4fd.py            # Training script for LESS4FD
├── models/
│   ├── __init__.py
│   ├── less4fd_model.py        # Main LESS4FD model
│   ├── entity_encoder.py       # Entity-aware encoder
│   └── contrastive_module.py   # Contrastive learning components
├── utils/
│   ├── __init__.py
│   ├── entity_extractor.py     # Entity extraction utilities
│   ├── pretext_tasks.py        # Self-supervised pretext tasks
│   └── sampling.py             # Few-shot sampling utilities
├── config/
│   └── less4fd_config.py       # Configuration parameters
├── requirements_less4fd.txt    # Additional dependencies
└── README.md                   # Usage instructions
```

## Technical Specifications

### 1. Graph Construction (`build_less4fd_graph.py`)
**Reference**: Follow `build_hetero_graph.py` structure

**Key Features**:
- Entity extraction from news text using NER models
- Heterogeneous graph with node types: `news`, `entity`, `interaction`
- Edge types: `news-entity`, `entity-entity`, `news-news`, `news-interaction`
- Few-shot sampling following existing `sample_k_shot` utility
- Entity-aware edge construction based on co-occurrence and semantic similarity

**Required Methods**:
```python
class LESS4FDGraphBuilder:
    def __init__(self, dataset_name, k_shot, embedding_type, entity_model="bert-base-ner")
    def extract_entities(self, text) -> List[Dict]
    def build_entity_nodes(self, news_data) -> Dict
    def build_entity_edges(self, news_entities) -> Tuple[torch.Tensor, torch.Tensor]
    def build_hetero_graph(self) -> HeteroData
    def save_graph(self, graph: HeteroData) -> str
```

### 2. Model Architecture (`models/less4fd_model.py`)
**Reference**: Follow `train_hetero_graph.py` model structure

**Core Components**:
- **Entity-aware Encoder**: Process news and entity nodes with attention mechanisms
- **Contrastive Learning Module**: Implement InfoNCE loss for self-supervised learning
- **Pretext Task Module**: Entity prediction, masked entity modeling
- **Few-shot Classifier**: Meta-learning compatible classification head

**Model Structure**:
```python
class LESS4FDModel(nn.Module):
    def __init__(self, data: HeteroData, hidden_channels, num_entities, num_classes)
    def encode_entities(self, x_dict, edge_index_dict) -> torch.Tensor
    def contrastive_loss(self, embeddings, labels) -> torch.Tensor
    def pretext_loss(self, news_embeddings, entity_embeddings) -> torch.Tensor
    def forward(self, x_dict, edge_index_dict, task="classification")
```

### 3. Training Script (`train_less4fd.py`)
**Reference**: Follow `train_hetero_graph.py` training loop

**Training Phases**:
1. **Pre-training Phase**: Self-supervised learning with pretext tasks
2. **Fine-tuning Phase**: Few-shot classification with meta-learning
3. **Evaluation Phase**: Performance on test set

**Loss Functions**:
- Contrastive Loss (InfoNCE)
- Entity Prediction Loss
- Classification Loss (Cross-entropy with label smoothing)
- Combined Loss: `α * contrastive + β * pretext + γ * classification`

### 4. Entity Processing (`utils/entity_extractor.py`)
- Use spaCy or transformers for NER
- Entity linking to Wikidata/DBpedia
- Entity embedding generation
- Entity type classification

### 5. Pretext Tasks (`utils/pretext_tasks.py`)
- **Masked Entity Modeling**: Predict masked entities in news text
- **Entity-News Alignment**: Learn alignment between entity and news representations
- **Entity Co-occurrence Prediction**: Predict entity relationships

## Configuration Parameters

### Graph Construction
```python
LESS4FD_CONFIG = {
    "entity_model": "bert-base-ner",  # NER model for entity extraction
    "entity_embedding_dim": 768,      # Entity embedding dimension
    "max_entities_per_news": 10,      # Maximum entities per news article
    "entity_similarity_threshold": 0.7, # Threshold for entity-entity edges
    "contrastive_temperature": 0.07,  # Temperature for contrastive learning
    "pretext_weight": 0.3,           # Weight for pretext task loss
    "contrastive_weight": 0.4,       # Weight for contrastive loss
    "classification_weight": 0.3,    # Weight for classification loss
}
```

### Training Parameters
```python
TRAINING_CONFIG = {
    "pretrain_epochs": 100,          # Pre-training epochs
    "finetune_epochs": 50,           # Fine-tuning epochs
    "learning_rate": 1e-4,           # Learning rate
    "weight_decay": 1e-4,            # Weight decay
    "warmup_steps": 1000,            # Warmup steps
    "gradient_clip": 1.0,            # Gradient clipping
}
```

## Dataset Integration

### HuggingFace Dataset Format
Use existing dataset structure from `dataset/name/`:
- `LittleFish-Coder/Fake_News_politifact`
- `LittleFish-Coder/Fake_News_gossipcop`

### Entity Data Structure
```python
{
    "news_id": "unique_id",
    "text": "news content",
    "label": 0/1,  # 0: real, 1: fake
    "entities": [
        {
            "text": "entity name",
            "type": "PERSON/ORG/LOC/etc",
            "start": 10,
            "end": 15,
            "confidence": 0.95
        }
    ],
    "entity_embeddings": [...],  # Pre-computed entity embeddings
    "interaction_embeddings_list": [...],  # Existing interaction data
}
```

## Dependencies

### Additional Requirements (`requirements_less4fd.txt`)
```
# Entity Processing
spacy>=3.5.0
transformers[torch]>=4.20.0
torch>=1.12.0

# Graph Neural Networks
torch-geometric>=2.2.0
torch-scatter>=2.0.9
torch-sparse>=0.6.16

# Entity Linking
wikidata>=0.6.0
requests>=2.28.0

# Utilities
tqdm>=4.64.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
pandas>=1.4.0

# HuggingFace
datasets>=2.0.0
evaluate>=0.3.0
```

## Implementation Guidelines

### Code Quality Requirements
1. **Readable and Maintainable**: Clear function names, comprehensive docstrings
2. **Extensible**: Modular design for easy addition of new components
3. **Configurable**: All hyperparameters should be configurable
4. **Compatible**: Integrate with existing heterogeneous graph framework

### Performance Considerations
1. **Memory Efficient**: Handle large entity vocabularies efficiently
2. **Scalable**: Support batch processing for large datasets
3. **Reproducible**: Proper seeding and deterministic operations

### Testing Requirements
1. **Unit Tests**: Test each component independently
2. **Integration Tests**: Test end-to-end pipeline
3. **Performance Tests**: Validate few-shot learning performance

## Evaluation Metrics

### Few-shot Learning Metrics
- Accuracy, Precision, Recall, F1-score
- Per-class performance analysis
- Learning curves for different k-shot settings

### Self-supervised Learning Metrics
- Contrastive learning loss
- Pretext task accuracy
- Entity prediction accuracy

## Deliverables

1. **Complete Implementation**: All files in the specified structure
2. **README.md**: Comprehensive usage instructions and examples
3. **Configuration Files**: Default configurations for different scenarios
4. **Requirements File**: All necessary dependencies
5. **Documentation**: Code comments and docstrings

## References

- LESS4FD Research Paper: `related_work/LESS4FD/2024.emnlp-main.31.pdf`
- Existing Implementation: `build_hetero_graph.py`, `train_hetero_graph.py`
- Dataset Format: `dataset/name/` structure
- Few-shot Sampling: `utils/sample_k_shot.py`

## Acceptance Criteria

- [ ] Graph construction successfully builds entity-aware heterogeneous graphs
- [ ] Model architecture implements all LESS4FD components
- [ ] Training pipeline supports both pre-training and fine-tuning phases
- [ ] Few-shot learning performance matches or exceeds baseline
- [ ] Code is well-documented and follows project conventions
- [ ] All dependencies are properly specified
- [ ] README provides clear usage instructions
- [ ] Integration with existing framework is seamless

## Timeline
- **Week 1**: Graph construction and entity processing
- **Week 2**: Model architecture and training pipeline
- **Week 3**: Integration and testing
- **Week 4**: Documentation and optimization

## Notes
- Prioritize few-shot learning performance
- Ensure compatibility with existing heterogeneous graph framework
- Focus on entity-aware representations and self-supervised learning
- Maintain code quality and extensibility 