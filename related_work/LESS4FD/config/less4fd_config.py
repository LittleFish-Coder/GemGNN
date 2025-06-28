"""
Configuration parameters for LESS4FD architecture.
"""

# Graph Construction Configuration
LESS4FD_CONFIG = {
    # Entity Processing
    "entity_model": "bert-base-cased",  # Base model for entity processing
    "entity_embedding_dim": 768,      # Entity embedding dimension
    "max_entities_per_news": 10,      # Maximum entities per news article
    "entity_similarity_threshold": 0.7, # Threshold for entity-entity edges
    "entity_types": ["PERSON", "ORG", "GPE", "MISC"],  # Entity types to extract
    
    # Contrastive Learning
    "contrastive_temperature": 0.07,  # Temperature for contrastive learning
    "contrastive_margin": 0.5,        # Margin for contrastive loss
    "max_contrastive_samples": 64,    # Max samples for contrastive learning
    
    # Loss Weights
    "pretext_weight": 0.3,           # Weight for pretext task loss
    "contrastive_weight": 0.4,       # Weight for contrastive loss
    "classification_weight": 0.3,    # Weight for classification loss
    
    # Model Architecture
    "hidden_channels": 64,           # Hidden dimension for GNN layers
    "num_gnn_layers": 2,            # Number of GNN layers
    "dropout": 0.3,                 # Dropout rate
    "num_attention_heads": 4,       # Number of attention heads
    
    # Entity Graph Construction
    "entity_edge_policy": "semantic_similarity",  # Policy for entity-entity edges
    "entity_knn": 5,                # K nearest neighbors for entity edges
    "use_entity_types": True,       # Whether to use entity type information
}

# Training Configuration
TRAINING_CONFIG = {
    # Training Phases
    "pretrain_epochs": 100,          # Pre-training epochs
    "finetune_epochs": 50,           # Fine-tuning epochs
    
    # Optimization
    "learning_rate": 1e-4,           # Learning rate
    "weight_decay": 1e-4,            # Weight decay
    "warmup_steps": 1000,            # Warmup steps
    "gradient_clip": 1.0,            # Gradient clipping
    
    # Scheduling
    "lr_scheduler": "cosine",        # Learning rate scheduler
    "patience": 20,                  # Early stopping patience
    
    # Batch Processing
    "batch_size": 32,               # Batch size for training
    "accumulation_steps": 1,        # Gradient accumulation steps
}

# Few-shot Learning Configuration
FEWSHOT_CONFIG = {
    "k_shot_range": [3, 4, 5, 8, 16],  # Supported k-shot values
    "meta_learning": True,             # Enable meta-learning
    "support_query_ratio": 0.5,       # Ratio of support to query samples
    "num_meta_tasks": 100,             # Number of meta-learning tasks
}

# Data Configuration
DATA_CONFIG = {
    "datasets": ["politifact", "gossipcop"],
    "text_embedding_fields": ["deberta_embeddings", "bert_embeddings"],
    "entity_cache_dir": "entity_cache",
    "graph_cache_dir": "graphs_less4fd",
}