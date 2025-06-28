"""
Main LESS4FD model for few-shot fake news detection.

This module implements the complete LESS4FD architecture combining entity-aware
graph construction, self-supervised learning, and few-shot classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, Any
import logging

from models.entity_encoder import EntityEncoder
from models.contrastive_module import ContrastiveModule
from utils.pretext_tasks import PretextTaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LESS4FDModel(nn.Module):
    """
    Complete LESS4FD model for entity-aware few-shot fake news detection.
    
    This model combines:
    - Entity-aware graph encoding
    - Self-supervised pretext tasks
    - Contrastive learning
    - Few-shot classification
    """
    
    def __init__(
        self,
        data: HeteroData,
        hidden_channels: int = 64,
        num_entities: int = 1000,
        num_classes: int = 2,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        entity_embedding_dim: int = 768,
        use_entity_types: bool = True,
        num_entity_types: int = 4,
        contrastive_temperature: float = 0.07,
        pretext_task_weights: Dict[str, float] = None,
        device: str = None
    ):
        """
        Initialize the LESS4FD model.
        
        Args:
            data: Heterogeneous graph data
            hidden_channels: Hidden dimension size
            num_entities: Number of entities in vocabulary
            num_classes: Number of classes (2 for fake/real)
            num_gnn_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            entity_embedding_dim: Entity embedding dimension
            use_entity_types: Whether to use entity type information
            num_entity_types: Number of entity types
            contrastive_temperature: Temperature for contrastive learning
            pretext_task_weights: Weights for pretext tasks
            device: Device for model
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_entities = num_entities
        self.num_classes = num_classes
        self.num_gnn_layers = num_gnn_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.entity_embedding_dim = entity_embedding_dim
        self.use_entity_types = use_entity_types
        self.num_entity_types = num_entity_types
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Entity-aware encoder
        self.entity_encoder = EntityEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout,
            entity_embedding_dim=entity_embedding_dim,
            use_entity_types=use_entity_types,
            num_entity_types=num_entity_types
        )
        
        # Contrastive learning module
        self.contrastive_module = ContrastiveModule(
            temperature=contrastive_temperature,
            max_samples=64,
            entity_weight=0.3,
            news_weight=0.7,
            use_entity_types=use_entity_types
        )
        
        # Pretext task manager
        self.pretext_manager = PretextTaskManager(
            mask_prob=0.15,
            negative_samples=5,
            temperature=contrastive_temperature,
            device=device
        )
        
        # Classification heads
        self.news_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        # Entity classification head (for pretext tasks)
        self.entity_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_entity_types)
        )
        
        # Entity vocabulary predictor for masked entity modeling
        self.entity_vocab_predictor = nn.Linear(hidden_channels, num_entities)
        
        # Task weights
        if pretext_task_weights is None:
            self.pretext_task_weights = {
                "masked_entity": 0.3,
                "alignment": 0.4,
                "cooccurrence": 0.2,
                "type_prediction": 0.1
            }
        else:
            self.pretext_task_weights = pretext_task_weights
        
        # Loss combination weights
        self.loss_weights = {
            "classification": 0.4,
            "contrastive": 0.3,
            "pretext": 0.3
        }
        
        self.to(self.device)
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        task: str = "classification",
        entity_types: Optional[torch.Tensor] = None,
        news_labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass of the LESS4FD model.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            task: Task type ("classification", "pretext", "contrastive", "all")
            entity_types: Entity type labels
            news_labels: News labels for contrastive learning
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing outputs and losses
        """
        # Encode graph with entity-aware encoder
        h_dict = self.entity_encoder(x_dict, edge_index_dict, entity_types)
        
        outputs = {
            "embeddings": h_dict if return_embeddings else None,
            "losses": {},
            "predictions": {}
        }
        
        # Classification task
        if task in ["classification", "all"]:
            if 'news' in h_dict:
                news_logits = self.news_classifier(h_dict['news'])
                outputs["predictions"]["news_logits"] = news_logits
                
                # Classification loss
                if news_labels is not None:
                    classification_loss = F.cross_entropy(news_logits, news_labels)
                    outputs["losses"]["classification"] = classification_loss
        
        # Contrastive learning task
        if task in ["contrastive", "all"]:
            if 'news' in h_dict and 'entity' in h_dict and news_labels is not None:
                # Get news-entity edges
                news_entity_edges = None
                if ('news', 'connected_to', 'entity') in edge_index_dict:
                    news_entity_edges = edge_index_dict[('news', 'connected_to', 'entity')]
                
                contrastive_loss, contrastive_components = self.contrastive_module(
                    h_dict['news'], h_dict['entity'], news_labels, entity_types, news_entity_edges
                )
                outputs["losses"]["contrastive"] = contrastive_loss
                outputs["losses"].update({f"contrastive_{k}": v for k, v in contrastive_components.items()})
        
        # Pretext tasks
        if task in ["pretext", "all"]:
            pretext_loss, pretext_components = self.compute_pretext_losses(
                h_dict, edge_index_dict, entity_types
            )
            outputs["losses"]["pretext"] = pretext_loss
            outputs["losses"].update({f"pretext_{k}": v for k, v in pretext_components.items()})
        
        # Combined loss
        if task == "all":
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            if "classification" in outputs["losses"]:
                total_loss = total_loss + self.loss_weights["classification"] * outputs["losses"]["classification"]
            
            if "contrastive" in outputs["losses"]:
                total_loss = total_loss + self.loss_weights["contrastive"] * outputs["losses"]["contrastive"]
            
            if "pretext" in outputs["losses"]:
                total_loss = total_loss + self.loss_weights["pretext"] * outputs["losses"]["pretext"]
            
            outputs["losses"]["total"] = total_loss
        
        return outputs
    
    def compute_pretext_losses(
        self,
        h_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        entity_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute pretext task losses.
        
        Args:
            h_dict: Node embeddings dictionary
            edge_index_dict: Edge index dictionary
            entity_types: Entity type labels
            
        Returns:
            Tuple of (combined_pretext_loss, individual_losses)
        """
        # Get edge indices
        news_entity_edges = None
        entity_entity_edges = None
        
        if ('news', 'connected_to', 'entity') in edge_index_dict:
            news_entity_edges = edge_index_dict[('news', 'connected_to', 'entity')]
        
        if ('entity', 'related_to', 'entity') in edge_index_dict:
            entity_entity_edges = edge_index_dict[('entity', 'related_to', 'entity')]
        
        # Use pretext task manager to compute losses
        if 'news' in h_dict and 'entity' in h_dict:
            pretext_loss, individual_losses = self.pretext_manager.compute_combined_pretext_loss(
                news_embeddings=h_dict['news'],
                entity_embeddings=h_dict['entity'],
                news_entity_edges=news_entity_edges,
                entity_entity_edges=entity_entity_edges,
                entity_types=entity_types,
                entity_vocab_size=self.num_entities,
                task_weights=self.pretext_task_weights
            )
        else:
            pretext_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            individual_losses = {}
        
        return pretext_loss, individual_losses
    
    def encode_entities(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        entity_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode entities and return entity embeddings.
        
        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge index dictionary
            entity_types: Entity type labels
            
        Returns:
            Entity embeddings
        """
        h_dict = self.entity_encoder(x_dict, edge_index_dict, entity_types)
        return h_dict.get('entity', torch.empty(0, self.hidden_channels, device=self.device))
    
    def encode_news(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        entity_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode news and return news embeddings.
        
        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge index dictionary
            entity_types: Entity type labels
            
        Returns:
            News embeddings
        """
        h_dict = self.entity_encoder(x_dict, edge_index_dict, entity_types)
        return h_dict.get('news', torch.empty(0, self.hidden_channels, device=self.device))
    
    def predict(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        entity_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Make predictions on news data.
        
        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge index dictionary
            entity_types: Entity type labels
            
        Returns:
            Prediction logits for news classification
        """
        outputs = self.forward(
            x_dict, edge_index_dict, task="classification", entity_types=entity_types
        )
        return outputs["predictions"].get("news_logits", torch.empty(0, self.num_classes, device=self.device))
    
    def get_entity_aware_news_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        entity_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get entity-aware news embeddings.
        
        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge index dictionary
            entity_types: Entity type labels
            
        Returns:
            Entity-aware news embeddings
        """
        h_dict = self.entity_encoder(x_dict, edge_index_dict, entity_types)
        
        if 'news' in h_dict and 'entity' in h_dict:
            # Get news-entity edges
            news_entity_edges = edge_index_dict.get(('news', 'connected_to', 'entity'), torch.empty(2, 0, device=self.device))
            
            # Get entity-aware embeddings
            entity_aware_embeddings = self.entity_encoder.get_entity_aware_news_embedding(
                h_dict['news'], h_dict['entity'], news_entity_edges
            )
            return entity_aware_embeddings
        
        return h_dict.get('news', torch.empty(0, self.hidden_channels, device=self.device))
    
    def pretrain_step(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        entity_types: Optional[torch.Tensor] = None,
        news_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Perform a pretraining step with self-supervised tasks.
        
        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge index dictionary
            entity_types: Entity type labels
            news_labels: News labels for contrastive learning
            
        Returns:
            Dictionary with pretraining losses and metrics
        """
        # Forward pass with pretext and contrastive tasks
        outputs = self.forward(
            x_dict, edge_index_dict, task="all", entity_types=entity_types, news_labels=news_labels
        )
        
        # Combine pretext and contrastive losses for pretraining
        pretrain_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if "pretext" in outputs["losses"]:
            pretrain_loss = pretrain_loss + 0.6 * outputs["losses"]["pretext"]
        
        if "contrastive" in outputs["losses"]:
            pretrain_loss = pretrain_loss + 0.4 * outputs["losses"]["contrastive"]
        
        outputs["losses"]["pretrain_total"] = pretrain_loss
        
        return outputs
    
    def finetune_step(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        news_labels: torch.Tensor,
        entity_types: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Perform a finetuning step with classification task.
        
        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge index dictionary
            news_labels: News labels
            entity_types: Entity type labels
            
        Returns:
            Dictionary with finetuning losses and predictions
        """
        # Forward pass with classification and light contrastive learning
        outputs = self.forward(
            x_dict, edge_index_dict, task="classification", 
            entity_types=entity_types, news_labels=news_labels
        )
        
        # Add light contrastive learning during finetuning
        if 'news' in x_dict and 'entity' in x_dict:
            h_dict = self.entity_encoder(x_dict, edge_index_dict, entity_types)
            news_entity_edges = edge_index_dict.get(('news', 'connected_to', 'entity'), None)
            
            if news_entity_edges is not None:
                contrastive_loss, _ = self.contrastive_module(
                    h_dict['news'], h_dict['entity'], news_labels, entity_types, news_entity_edges
                )
                outputs["losses"]["light_contrastive"] = contrastive_loss
                
                # Combine classification and light contrastive loss
                if "classification" in outputs["losses"]:
                    finetune_loss = (
                        0.8 * outputs["losses"]["classification"] +
                        0.2 * outputs["losses"]["light_contrastive"]
                    )
                    outputs["losses"]["finetune_total"] = finetune_loss
        
        return outputs
    
    def set_loss_weights(self, weights: Dict[str, float]):
        """Set loss combination weights."""
        self.loss_weights.update(weights)
    
    def set_pretext_task_weights(self, weights: Dict[str, float]):
        """Set pretext task weights."""
        self.pretext_task_weights.update(weights)
    
    def freeze_entity_encoder(self):
        """Freeze entity encoder parameters."""
        for param in self.entity_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_entity_encoder(self):
        """Unfreeze entity encoder parameters."""
        for param in self.entity_encoder.parameters():
            param.requires_grad = True