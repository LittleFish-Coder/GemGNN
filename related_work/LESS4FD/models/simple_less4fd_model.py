"""
Simplified LESS4FD Model.

A simplified version of the entity-aware fake news detection model that follows
the main repository patterns without complex meta-learning components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, HANConv, Linear, GATv2Conv
from typing import Dict, Optional


class SimpleLESS4FDModel(nn.Module):
    """
    Simplified LESS4FD model that extends standard GNN architectures
    with basic entity-aware features.
    """
    
    def __init__(
        self,
        metadata: tuple,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        model_type: str = "HGT",
        entity_aware: bool = True
    ):
        """
        Initialize simplified LESS4FD model.
        
        Args:
            metadata: Graph metadata (node_types, edge_types)
            hidden_channels: Hidden dimension size
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            model_type: Type of GNN model (HGT, HAN, GAT)
            entity_aware: Whether to use entity-aware features
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.model_type = model_type
        self.entity_aware = entity_aware
        self.node_types, self.edge_types = metadata
        
        # Input projections for different node types
        self.node_embeddings = nn.ModuleDict()
        for node_type in self.node_types:
            # Project to hidden dimension
            self.node_embeddings[node_type] = Linear(-1, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if model_type == "HGT":
                self.convs.append(
                    HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
                )
            elif model_type == "HAN":
                self.convs.append(
                    HANConv(hidden_channels, hidden_channels, metadata, num_heads)
                )
            elif model_type == "GAT":
                # Convert to homogeneous for GAT
                self.convs.append(
                    GATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Entity-aware enhancement (if enabled)
        if entity_aware:
            self.entity_attention = nn.MultiheadAttention(
                embed_dim=hidden_channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.entity_norm = nn.LayerNorm(hidden_channels)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, 2)  # Binary classification
        )
        
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[tuple, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge indices dictionary
            
        Returns:
            Logits for news node classification
        """
        # Project node features to hidden dimension
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_embeddings[node_type](x)
        
        # Apply GNN layers
        for conv in self.convs:
            if self.model_type in ["HGT", "HAN"]:
                x_dict = conv(x_dict, edge_index_dict)
            else:  # GAT - convert to homogeneous
                # Simplified homogeneous conversion for GAT
                x_dict['news'] = conv(x_dict['news'], edge_index_dict[('news', 'similar', 'news')])
            
            # Apply activation and dropout
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Get news node embeddings
        news_embeddings = x_dict['news']
        
        # Apply entity-aware enhancement if enabled
        if self.entity_aware:
            news_embeddings = self.apply_entity_awareness(news_embeddings)
        
        # Classification
        logits = self.classifier(news_embeddings)
        
        return logits
    
    def apply_entity_awareness(self, news_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply entity-aware attention to news embeddings.
        This is a simplified version of entity-awareness.
        
        Args:
            news_embeddings: News node embeddings
            
        Returns:
            Enhanced news embeddings
        """
        # Self-attention to model entity interactions
        enhanced, _ = self.entity_attention(
            news_embeddings.unsqueeze(0),  # Add batch dimension
            news_embeddings.unsqueeze(0),
            news_embeddings.unsqueeze(0)
        )
        enhanced = enhanced.squeeze(0)  # Remove batch dimension
        
        # Residual connection and layer norm
        enhanced = self.entity_norm(news_embeddings + enhanced)
        
        return enhanced
    
    def get_embeddings(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[tuple, torch.Tensor]) -> torch.Tensor:
        """Get node embeddings without classification."""
        with torch.no_grad():
            # Project node features
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.node_embeddings[node_type](x)
            
            # Apply GNN layers
            for conv in self.convs:
                if self.model_type in ["HGT", "HAN"]:
                    x_dict = conv(x_dict, edge_index_dict)
                else:
                    x_dict['news'] = conv(x_dict['news'], edge_index_dict[('news', 'similar', 'news')])
                
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            
            news_embeddings = x_dict['news']
            
            if self.entity_aware:
                news_embeddings = self.apply_entity_awareness(news_embeddings)
                
            return news_embeddings


def create_simple_less4fd_model(metadata: tuple, config: dict) -> SimpleLESS4FDModel:
    """
    Factory function to create a SimpleLESS4FDModel with given configuration.
    
    Args:
        metadata: Graph metadata
        config: Model configuration dictionary
        
    Returns:
        Initialized SimpleLESS4FDModel
    """
    return SimpleLESS4FDModel(
        metadata=metadata,
        hidden_channels=config.get('hidden_channels', 64),
        num_layers=config.get('num_gnn_layers', 2),
        num_heads=config.get('num_attention_heads', 4),
        dropout=config.get('dropout', 0.3),
        model_type=config.get('model_type', 'HGT'),
        entity_aware=config.get('use_entity_embeddings', True)
    )