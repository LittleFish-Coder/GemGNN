"""
Entity-aware encoder for LESS4FD architecture.

This module implements the entity-aware encoder that processes both news and entity nodes
with attention mechanisms and entity-specific features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, HGTConv, GATv2Conv, Linear
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityEncoder(nn.Module):
    """
    Entity-aware encoder for processing heterogeneous graphs with news and entity nodes.
    
    This encoder extends standard graph neural networks with entity-specific attention
    mechanisms and multi-level feature processing.
    """
    
    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        entity_embedding_dim: int = 768,
        use_entity_types: bool = True,
        num_entity_types: int = 4,
        aggregation: str = "attention",  # "attention", "mean", "max"
        activation: str = "relu"
    ):
        """
        Initialize the EntityEncoder.
        
        Args:
            hidden_channels: Hidden dimension size
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            entity_embedding_dim: Dimension of entity embeddings
            use_entity_types: Whether to use entity type information
            num_entity_types: Number of entity types
            aggregation: Aggregation method for multi-head attention
            activation: Activation function
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.entity_embedding_dim = entity_embedding_dim
        self.use_entity_types = use_entity_types
        self.num_entity_types = num_entity_types
        self.aggregation = aggregation
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # Input projections for different node types
        self.news_input_proj = Linear(-1, hidden_channels)
        self.entity_input_proj = Linear(entity_embedding_dim, hidden_channels)
        
        # Entity type embeddings
        if use_entity_types:
            self.entity_type_embedding = nn.Embedding(num_entity_types, hidden_channels // 4)
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Define metadata for heterogeneous graph
        metadata = (
            ['news', 'entity'],  # Node types
            [
                ('news', 'similar_to', 'news'),
                ('news', 'connected_to', 'entity'),
                ('entity', 'connected_to', 'news'),
                ('entity', 'related_to', 'entity')
            ]  # Edge types
        )
        
        for i in range(num_layers):
            # Use HAN (Heterogeneous Attention Network) layers
            han_layer = HANConv(
                in_channels=-1,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
                dropout=dropout
            )
            self.gnn_layers.append(han_layer)
            
            # Layer normalization
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Entity-aware attention module
        self.entity_attention = EntityAwareAttention(
            hidden_channels, num_heads, dropout
        )
        
        # News-entity cross attention
        self.cross_attention = CrossModalAttention(
            hidden_channels, num_heads, dropout
        )
        
        # Output projections
        self.news_output_proj = Linear(hidden_channels, hidden_channels)
        self.entity_output_proj = Linear(hidden_channels, hidden_channels)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        entity_types: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the entity encoder.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            entity_types: Entity type labels for entity nodes
            
        Returns:
            Dictionary of encoded node embeddings
        """
        # Input projections
        h_dict = {}
        
        # Project news features
        if 'news' in x_dict:
            h_dict['news'] = self.news_input_proj(x_dict['news'])
            h_dict['news'] = self.activation(h_dict['news'])
            h_dict['news'] = self.dropout_layer(h_dict['news'])
        
        # Project entity features with type information
        if 'entity' in x_dict:
            h_entity = self.entity_input_proj(x_dict['entity'])
            
            # Add entity type information if available
            if self.use_entity_types and entity_types is not None:
                type_emb = self.entity_type_embedding(entity_types)
                # Concatenate or add type embeddings
                if type_emb.size(-1) == h_entity.size(-1):
                    h_entity = h_entity + type_emb
                else:
                    # Pad or project to match dimensions
                    if type_emb.size(-1) < h_entity.size(-1):
                        padding = h_entity.size(-1) - type_emb.size(-1)
                        type_emb = F.pad(type_emb, (0, padding))
                    h_entity = h_entity + type_emb[:h_entity.size(0)]
            
            h_dict['entity'] = self.activation(h_entity)
            h_dict['entity'] = self.dropout_layer(h_dict['entity'])
        
        # Apply GNN layers
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            # Store previous embeddings for residual connection
            h_prev = {node_type: h.clone() for node_type, h in h_dict.items()}
            
            # Apply GNN layer
            h_dict = gnn_layer(h_dict, edge_index_dict)
            
            # Apply normalization and activation
            for node_type in h_dict:
                h_dict[node_type] = norm(h_dict[node_type])
                h_dict[node_type] = self.activation(h_dict[node_type])
                
                # Residual connection
                if node_type in h_prev and h_prev[node_type].size() == h_dict[node_type].size():
                    h_dict[node_type] = h_dict[node_type] + h_prev[node_type]
                
                h_dict[node_type] = self.dropout_layer(h_dict[node_type])
        
        # Apply entity-aware attention if we have both news and entity nodes
        if 'news' in h_dict and 'entity' in h_dict:
            # Enhanced entity representations with attention
            h_dict['entity'] = self.entity_attention(
                h_dict['entity'], h_dict['news'], edge_index_dict
            )
            
            # Cross-modal attention between news and entities
            h_dict['news'], h_dict['entity'] = self.cross_attention(
                h_dict['news'], h_dict['entity'], edge_index_dict
            )
        
        # Output projections
        if 'news' in h_dict:
            h_dict['news'] = self.news_output_proj(h_dict['news'])
        
        if 'entity' in h_dict:
            h_dict['entity'] = self.entity_output_proj(h_dict['entity'])
        
        return h_dict
    
    def get_entity_aware_news_embedding(
        self,
        news_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        news_entity_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Get entity-aware news embeddings by aggregating connected entity information.
        
        Args:
            news_embeddings: News node embeddings [num_news, hidden_dim]
            entity_embeddings: Entity node embeddings [num_entities, hidden_dim]
            news_entity_edges: News-entity edge indices [2, num_edges]
            
        Returns:
            Entity-aware news embeddings [num_news, hidden_dim]
        """
        if news_entity_edges.size(1) == 0:
            return news_embeddings
        
        num_news = news_embeddings.size(0)
        hidden_dim = news_embeddings.size(1)
        device = news_embeddings.device
        
        # Initialize entity-aware embeddings with original news embeddings
        entity_aware_embeddings = news_embeddings.clone()
        
        # Aggregate entity information for each news node
        news_indices = news_entity_edges[0]
        entity_indices = news_entity_edges[1]
        
        # Group entities by news node
        entity_info = torch.zeros(num_news, hidden_dim, device=device)
        entity_counts = torch.zeros(num_news, device=device)
        
        # Sum entity embeddings for each news node
        entity_info.index_add_(0, news_indices, entity_embeddings[entity_indices])
        entity_counts.index_add_(0, news_indices, torch.ones_like(news_indices, dtype=torch.float))
        
        # Average entity embeddings
        valid_mask = entity_counts > 0
        entity_info[valid_mask] = entity_info[valid_mask] / entity_counts[valid_mask].unsqueeze(1)
        
        # Combine news and entity information using attention
        if valid_mask.any():
            attention_weights = torch.softmax(
                torch.sum(news_embeddings[valid_mask] * entity_info[valid_mask], dim=1),
                dim=0
            )
            
            # Weighted combination of news and entity embeddings
            alpha = 0.7  # Weight for original news embeddings
            entity_aware_embeddings[valid_mask] = (
                alpha * news_embeddings[valid_mask] +
                (1 - alpha) * attention_weights.unsqueeze(1) * entity_info[valid_mask]
            )
        
        return entity_aware_embeddings


class EntityAwareAttention(nn.Module):
    """
    Entity-aware attention mechanism for enhancing entity representations.
    """
    
    def __init__(self, hidden_channels: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        
        self.query_proj = Linear(hidden_channels, hidden_channels)
        self.key_proj = Linear(hidden_channels, hidden_channels)
        self.value_proj = Linear(hidden_channels, hidden_channels)
        self.output_proj = Linear(hidden_channels, hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        entity_embeddings: torch.Tensor,
        news_embeddings: torch.Tensor,
        edge_index_dict: Dict
    ) -> torch.Tensor:
        """
        Apply entity-aware attention.
        
        Args:
            entity_embeddings: Entity embeddings [num_entities, hidden_dim]
            news_embeddings: News embeddings [num_news, hidden_dim]
            edge_index_dict: Edge index dictionary
            
        Returns:
            Enhanced entity embeddings
        """
        batch_size, hidden_dim = entity_embeddings.size()
        
        # Multi-head attention computation
        Q = self.query_proj(entity_embeddings).view(batch_size, self.num_heads, self.head_dim)
        K = self.key_proj(entity_embeddings).view(batch_size, self.num_heads, self.head_dim)
        V = self.value_proj(entity_embeddings).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended_values = attended_values.view(batch_size, hidden_dim)
        output = self.output_proj(attended_values)
        
        # Residual connection
        return entity_embeddings + output


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between news and entity embeddings.
    """
    
    def __init__(self, hidden_channels: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        
        # News-to-entity attention
        self.news_to_entity_attention = nn.MultiheadAttention(
            hidden_channels, num_heads, dropout=dropout, batch_first=True
        )
        
        # Entity-to-news attention
        self.entity_to_news_attention = nn.MultiheadAttention(
            hidden_channels, num_heads, dropout=dropout, batch_first=True
        )
        
        # Output projections
        self.news_output_proj = Linear(hidden_channels, hidden_channels)
        self.entity_output_proj = Linear(hidden_channels, hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        news_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        edge_index_dict: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention between news and entity embeddings.
        
        Args:
            news_embeddings: News embeddings [num_news, hidden_dim]
            entity_embeddings: Entity embeddings [num_entities, hidden_dim]
            edge_index_dict: Edge index dictionary
            
        Returns:
            Tuple of (enhanced_news_embeddings, enhanced_entity_embeddings)
        """
        # Add batch dimension for MultiheadAttention
        news_emb = news_embeddings.unsqueeze(0)  # [1, num_news, hidden_dim]
        entity_emb = entity_embeddings.unsqueeze(0)  # [1, num_entities, hidden_dim]
        
        # News attending to entities
        news_attended, _ = self.news_to_entity_attention(
            query=news_emb,
            key=entity_emb,
            value=entity_emb
        )
        news_attended = news_attended.squeeze(0)  # [num_news, hidden_dim]
        
        # Entity attending to news
        entity_attended, _ = self.entity_to_news_attention(
            query=entity_emb,
            key=news_emb,
            value=news_emb
        )
        entity_attended = entity_attended.squeeze(0)  # [num_entities, hidden_dim]
        
        # Apply output projections and residual connections
        enhanced_news = news_embeddings + self.dropout(self.news_output_proj(news_attended))
        enhanced_entity = entity_embeddings + self.dropout(self.entity_output_proj(entity_attended))
        
        return enhanced_news, enhanced_entity