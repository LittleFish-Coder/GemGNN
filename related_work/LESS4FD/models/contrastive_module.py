"""
Contrastive learning module for LESS4FD architecture.

This module implements entity-aware contrastive learning components extending
the existing contrastive loss functionality with entity-specific features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContrastiveModule(nn.Module):
    """
    Entity-aware contrastive learning module for LESS4FD.
    
    This module extends the base contrastive learning with entity-specific
    contrastive tasks and multi-level contrastive learning.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
        max_samples: int = 64,
        entity_weight: float = 0.3,
        news_weight: float = 0.7,
        use_entity_types: bool = True
    ):
        """
        Initialize the ContrastiveModule.
        
        Args:
            temperature: Temperature parameter for contrastive loss
            margin: Margin for contrastive loss
            max_samples: Maximum samples for numerical stability
            entity_weight: Weight for entity-level contrastive loss
            news_weight: Weight for news-level contrastive loss
            use_entity_types: Whether to use entity type information
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.max_samples = max_samples
        self.entity_weight = entity_weight
        self.news_weight = news_weight
        self.use_entity_types = use_entity_types
    
    def forward(
        self,
        news_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        news_labels: torch.Tensor,
        entity_types: Optional[torch.Tensor] = None,
        news_entity_edges: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute entity-aware contrastive loss.
        
        Args:
            news_embeddings: News node embeddings [num_news, hidden_dim]
            entity_embeddings: Entity embeddings [num_entities, hidden_dim]
            news_labels: News labels [num_news]
            entity_types: Entity type labels [num_entities]
            news_entity_edges: News-entity edge indices [2, num_edges]
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        loss_components = {}
        total_loss = torch.tensor(0.0, device=news_embeddings.device, requires_grad=True)
        
        # News-level contrastive loss
        if news_embeddings.size(0) > 1:
            news_loss = self.news_contrastive_loss(news_embeddings, news_labels)
            loss_components["news_contrastive"] = news_loss
            total_loss = total_loss + self.news_weight * news_loss
        
        # Entity-level contrastive loss
        if entity_embeddings.size(0) > 1 and entity_types is not None:
            entity_loss = self.entity_contrastive_loss(entity_embeddings, entity_types)
            loss_components["entity_contrastive"] = entity_loss
            total_loss = total_loss + self.entity_weight * entity_loss
        
        # Cross-modal contrastive loss (news-entity alignment)
        if news_entity_edges is not None and news_entity_edges.size(1) > 0:
            cross_modal_loss = self.cross_modal_contrastive_loss(
                news_embeddings, entity_embeddings, news_entity_edges, news_labels
            )
            loss_components["cross_modal_contrastive"] = cross_modal_loss
            total_loss = total_loss + 0.5 * cross_modal_loss
        
        return total_loss, loss_components
    
    def news_contrastive_loss(
        self,
        news_embeddings: torch.Tensor,
        news_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss for news embeddings.
        
        Args:
            news_embeddings: News embeddings [num_news, hidden_dim]
            news_labels: News labels [num_news]
            
        Returns:
            Contrastive loss for news
        """
        return self._infonce_loss(news_embeddings, news_labels)
    
    def entity_contrastive_loss(
        self,
        entity_embeddings: torch.Tensor,
        entity_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss for entity embeddings.
        
        Args:
            entity_embeddings: Entity embeddings [num_entities, hidden_dim]
            entity_types: Entity type labels [num_entities]
            
        Returns:
            Contrastive loss for entities
        """
        if not self.use_entity_types:
            # If not using entity types, create dummy labels
            entity_labels = torch.zeros(entity_embeddings.size(0), device=entity_embeddings.device)
            return self._infonce_loss(entity_embeddings, entity_labels)
        
        return self._infonce_loss(entity_embeddings, entity_types)
    
    def cross_modal_contrastive_loss(
        self,
        news_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        news_entity_edges: torch.Tensor,
        news_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-modal contrastive loss between news and entities.
        
        Args:
            news_embeddings: News embeddings [num_news, hidden_dim]
            entity_embeddings: Entity embeddings [num_entities, hidden_dim]
            news_entity_edges: Edge indices [2, num_edges]
            news_labels: News labels [num_news]
            
        Returns:
            Cross-modal contrastive loss
        """
        if news_entity_edges.size(1) == 0:
            return torch.tensor(0.0, device=news_embeddings.device, requires_grad=True)
        
        news_indices = news_entity_edges[0]
        entity_indices = news_entity_edges[1]
        
        # Get embeddings for connected pairs
        connected_news_emb = news_embeddings[news_indices]
        connected_entity_emb = entity_embeddings[entity_indices]
        connected_labels = news_labels[news_indices]
        
        # Normalize embeddings
        connected_news_emb = F.normalize(connected_news_emb, p=2, dim=1)
        connected_entity_emb = F.normalize(connected_entity_emb, p=2, dim=1)
        
        # Compute similarity matrix between news and entities
        batch_size = connected_news_emb.size(0)
        if batch_size > self.max_samples:
            # Sample subset for numerical stability
            indices = torch.randperm(batch_size, device=news_embeddings.device)[:self.max_samples]
            connected_news_emb = connected_news_emb[indices]
            connected_entity_emb = connected_entity_emb[indices]
            connected_labels = connected_labels[indices]
            batch_size = self.max_samples
        
        # Compute positive similarities (connected pairs)
        pos_sim = torch.sum(connected_news_emb * connected_entity_emb, dim=1)
        
        # Compute negative similarities
        # Use all entity embeddings as potential negatives
        all_entity_emb = F.normalize(entity_embeddings, p=2, dim=1)
        neg_sim = torch.mm(connected_news_emb, all_entity_emb.t())  # [batch_size, num_entities]
        
        # Remove positive pairs from negatives
        for i, entity_idx in enumerate(entity_indices[:batch_size]):
            neg_sim[i, entity_idx] = float('-inf')
        
        # InfoNCE loss computation
        pos_sim = pos_sim / self.temperature
        neg_sim = neg_sim / self.temperature
        
        # Combine positive and negative similarities
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # Compute loss
        log_prob = F.log_softmax(all_sim, dim=1)
        loss = -log_prob[:, 0].mean()  # Positive pairs are at index 0
        
        return loss
    
    def hierarchical_contrastive_loss(
        self,
        news_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        news_labels: torch.Tensor,
        entity_types: torch.Tensor,
        news_entity_edges: torch.Tensor,
        hierarchy_weights: Dict[str, float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hierarchical contrastive loss at multiple levels.
        
        Args:
            news_embeddings: News embeddings
            entity_embeddings: Entity embeddings
            news_labels: News labels
            entity_types: Entity type labels
            news_entity_edges: News-entity edge indices
            hierarchy_weights: Weights for different hierarchy levels
            
        Returns:
            Tuple of (total_loss, level_losses)
        """
        if hierarchy_weights is None:
            hierarchy_weights = {
                "news_level": 0.4,
                "entity_level": 0.3,
                "cross_modal": 0.3
            }
        
        level_losses = {}
        total_loss = torch.tensor(0.0, device=news_embeddings.device, requires_grad=True)
        
        # News-level contrastive learning
        if "news_level" in hierarchy_weights and news_embeddings.size(0) > 1:
            news_loss = self.news_contrastive_loss(news_embeddings, news_labels)
            level_losses["news_level"] = news_loss
            total_loss = total_loss + hierarchy_weights["news_level"] * news_loss
        
        # Entity-level contrastive learning
        if "entity_level" in hierarchy_weights and entity_embeddings.size(0) > 1:
            entity_loss = self.entity_contrastive_loss(entity_embeddings, entity_types)
            level_losses["entity_level"] = entity_loss
            total_loss = total_loss + hierarchy_weights["entity_level"] * entity_loss
        
        # Cross-modal contrastive learning
        if "cross_modal" in hierarchy_weights and news_entity_edges.size(1) > 0:
            cross_modal_loss = self.cross_modal_contrastive_loss(
                news_embeddings, entity_embeddings, news_entity_edges, news_labels
            )
            level_losses["cross_modal"] = cross_modal_loss
            total_loss = total_loss + hierarchy_weights["cross_modal"] * cross_modal_loss
        
        return total_loss, level_losses
    
    def _infonce_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        This is adapted from the existing ContrastiveLoss in train_hetero_graph.py
        to maintain compatibility while adding entity-specific features.
        
        Args:
            embeddings: Input embeddings [batch_size, hidden_dim]
            labels: Labels for embeddings [batch_size]
            
        Returns:
            InfoNCE loss
        """
        batch_size = embeddings.size(0)
        
        # Return zero loss for trivial cases
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Sample subset if batch is too large
        if batch_size > self.max_samples:
            indices = torch.randperm(batch_size, device=embeddings.device)[:self.max_samples]
            embeddings = embeddings[indices]
            labels = labels[indices]
            batch_size = self.max_samples
        
        # Normalize embeddings to unit sphere
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Create masks for positive and negative pairs
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        positives_mask = labels_eq.float()
        negatives_mask = (~labels_eq).float()
        
        # Remove self-similarity (diagonal)
        eye_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        positives_mask[eye_mask] = 0
        negatives_mask[eye_mask] = 0
        
        # Check if we have any positive pairs
        if positives_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Scale similarities by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # For numerical stability, subtract max before exp
        similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0].detach()
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Sum of positive similarities for each anchor
        pos_exp_sim = (exp_sim * positives_mask).sum(dim=1)
        
        # Sum of all non-self similarities for each anchor
        all_exp_sim = (exp_sim * (positives_mask + negatives_mask)).sum(dim=1)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        pos_exp_sim = torch.clamp(pos_exp_sim, min=epsilon)
        all_exp_sim = torch.clamp(all_exp_sim, min=epsilon)
        
        # InfoNCE loss: -log(sum(pos_sim) / sum(all_sim))
        loss = -torch.log(pos_exp_sim / all_exp_sim)
        
        # Only use anchors that have positive pairs
        has_positives = (positives_mask.sum(dim=1) > 0).float()
        loss = loss * has_positives
        
        # Return mean loss over valid anchors
        num_valid = has_positives.sum()
        if num_valid > 0:
            return loss.sum() / num_valid
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
    def compute_similarity_matrix(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute similarity matrix between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings (if None, use embeddings1)
            
        Returns:
            Similarity matrix
        """
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        # Normalize embeddings
        emb1_norm = F.normalize(embeddings1, p=2, dim=1)
        emb2_norm = F.normalize(embeddings2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(emb1_norm, emb2_norm.t())
        
        return similarity
    
    def hard_negative_mining(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        hard_ratio: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform hard negative mining for more effective contrastive learning.
        
        Args:
            embeddings: Input embeddings
            labels: Labels for embeddings
            hard_ratio: Ratio of hard negatives to sample
            
        Returns:
            Tuple of (selected_embeddings, selected_labels)
        """
        batch_size = embeddings.size(0)
        
        if batch_size < 4:  # Need minimum samples for hard negative mining
            return embeddings, labels
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Create label equality mask
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Find hard negatives (high similarity but different labels)
        hard_negatives_mask = (~labels_eq) & (similarity_matrix > 0.5)
        
        # Find hard positives (low similarity but same labels)
        hard_positives_mask = labels_eq & (similarity_matrix < 0.3)
        
        # Remove diagonal
        eye_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        hard_negatives_mask[eye_mask] = False
        hard_positives_mask[eye_mask] = False
        
        # Select hard samples
        selected_indices = set(range(batch_size))
        
        # Add hard negative pairs
        hard_neg_indices = torch.nonzero(hard_negatives_mask, as_tuple=False)
        if hard_neg_indices.size(0) > 0:
            num_hard_neg = min(int(batch_size * hard_ratio), hard_neg_indices.size(0))
            selected_hard_neg = hard_neg_indices[torch.randperm(hard_neg_indices.size(0))[:num_hard_neg]]
            selected_indices.update(selected_hard_neg[:, 0].tolist())
            selected_indices.update(selected_hard_neg[:, 1].tolist())
        
        # Add hard positive pairs
        hard_pos_indices = torch.nonzero(hard_positives_mask, as_tuple=False)
        if hard_pos_indices.size(0) > 0:
            num_hard_pos = min(int(batch_size * hard_ratio), hard_pos_indices.size(0))
            selected_hard_pos = hard_pos_indices[torch.randperm(hard_pos_indices.size(0))[:num_hard_pos]]
            selected_indices.update(selected_hard_pos[:, 0].tolist())
            selected_indices.update(selected_hard_pos[:, 1].tolist())
        
        # Convert to list and sort
        selected_indices = sorted(list(selected_indices))
        
        return embeddings[selected_indices], labels[selected_indices]