"""
Simplified configuration for LESS4FD architecture.
Follows the main repository patterns without meta-learning.
"""

# Core LESS4FD Configuration
LESS4FD_CONFIG = {
    # Entity Processing
    "max_entities_per_news": 5,        # Maximum entities per news article (simplified)
    "entity_similarity_threshold": 0.7, # Threshold for entity-entity edges
    "entity_types": ["PERSON", "ORG"],  # Simplified entity types
    
    # Model Architecture (following main repo defaults)
    "hidden_channels": 64,              # Hidden dimension for GNN layers
    "num_gnn_layers": 2,               # Number of GNN layers  
    "dropout": 0.3,                    # Dropout rate
    "num_attention_heads": 4,          # Number of attention heads
    
    # Entity-aware features
    "use_entity_embeddings": True,     # Whether to use entity embeddings
    "entity_edge_weight": 0.5,         # Weight for entity edges
}

# Training Configuration (simplified, no meta-learning)
TRAINING_CONFIG = {
    "epochs": 300,                     # Total training epochs (like main repo)
    "learning_rate": 5e-4,             # Learning rate (like main repo)
    "weight_decay": 1e-3,              # Weight decay (like main repo)
    "patience": 30,                    # Early stopping patience (like main repo)
}

# Few-shot Configuration (simplified)
FEWSHOT_CONFIG = {
    "k_shot_range": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # Supported k-shot values
    "meta_learning": False,            # Disabled meta-learning as requested
}

# Data Configuration (using Hugging Face datasets)
DATA_CONFIG = {
    "datasets": ["politifact", "gossipcop"],
    "embedding_types": ["bert", "roberta", "deberta"],  # Supported embedding types
    "hf_dataset_names": {
        "politifact": "LittleFish-Coder/Fake_News_PolitiFact",
        "gossipcop": "LittleFish-Coder/Fake_News_GossipCop"
    }
}