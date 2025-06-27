"""
Pretext tasks for self-supervised learning in LESS4FD architecture.

This module implements various pretext tasks for entity-aware self-supervised learning:
- Masked Entity Modeling
- Entity-News Alignment
- Entity Co-occurrence Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PretextTaskManager:
    """
    Manager for pretext tasks in LESS4FD self-supervised learning.
    
    This class implements and manages various pretext tasks for learning
    entity-aware representations in a self-supervised manner.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.15,
        negative_samples: int = 5,
        temperature: float = 0.07,
        device: str = None
    ):
        """
        Initialize the PretextTaskManager.
        
        Args:
            mask_prob: Probability of masking entities
            negative_samples: Number of negative samples for contrastive tasks
            temperature: Temperature for contrastive loss
            device: Device for computations
        """
        self.mask_prob = mask_prob
        self.negative_samples = negative_samples
        self.temperature = temperature
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
    
    def masked_entity_modeling(
        self,
        news_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        news_entity_edges: torch.Tensor,
        entity_vocab_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masked Entity Modeling pretext task.
        
        Randomly mask entities in news articles and predict them from context.
        
        Args:
            news_embeddings: News node embeddings [num_news, hidden_dim]
            entity_embeddings: Entity node embeddings [num_entities, hidden_dim]
            news_entity_edges: Edge indices connecting news to entities [2, num_edges]
            entity_vocab_size: Size of entity vocabulary
            
        Returns:
            Tuple of (masked_entity_logits, target_entities, mask)
        """
        device = news_embeddings.device
        num_edges = news_entity_edges.size(1)
        
        # Create mask for entities to predict
        mask = torch.rand(num_edges, device=device) < self.mask_prob
        
        if mask.sum() == 0:
            # No entities to mask, return empty tensors
            return (
                torch.empty(0, entity_vocab_size, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.bool, device=device)
            )
        
        # Get masked edges
        masked_edges = news_entity_edges[:, mask]
        news_indices = masked_edges[0]
        entity_indices = masked_edges[1]
        
        # Get news embeddings for masked edges
        masked_news_emb = news_embeddings[news_indices]  # [num_masked, hidden_dim]
        
        # Project to entity vocabulary space
        entity_predictor = nn.Linear(
            news_embeddings.size(1),
            entity_vocab_size
        ).to(device)
        
        # Predict entities from news context
        entity_logits = entity_predictor(masked_news_emb)  # [num_masked, vocab_size]
        
        return entity_logits, entity_indices, mask
    
    def entity_news_alignment(
        self,
        news_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        news_entity_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Entity-News Alignment pretext task.
        
        Learn alignment between entity and news representations using contrastive learning.
        
        Args:
            news_embeddings: News node embeddings [num_news, hidden_dim]
            entity_embeddings: Entity node embeddings [num_entities, hidden_dim]
            news_entity_edges: Edge indices connecting news to entities [2, num_edges]
            
        Returns:
            Contrastive alignment loss
        """
        if news_entity_edges.size(1) == 0:
            return torch.tensor(0.0, device=news_embeddings.device, requires_grad=True)
        
        news_indices = news_entity_edges[0]
        entity_indices = news_entity_edges[1]
        
        # Get embeddings for connected news-entity pairs
        news_emb = news_embeddings[news_indices]  # [num_edges, hidden_dim]
        entity_emb = entity_embeddings[entity_indices]  # [num_edges, hidden_dim]
        
        # Normalize embeddings
        news_emb = F.normalize(news_emb, p=2, dim=1)
        entity_emb = F.normalize(entity_emb, p=2, dim=1)
        
        # Compute positive similarities
        pos_sim = torch.sum(news_emb * entity_emb, dim=1)  # [num_edges]
        
        # Sample negative entity embeddings
        num_entities = entity_embeddings.size(0)
        neg_entity_indices = torch.randint(
            0, num_entities,
            (news_emb.size(0), self.negative_samples),
            device=news_embeddings.device
        )
        
        neg_entity_emb = entity_embeddings[neg_entity_indices]  # [num_edges, neg_samples, hidden_dim]
        neg_entity_emb = F.normalize(neg_entity_emb, p=2, dim=2)
        
        # Compute negative similarities
        news_emb_expanded = news_emb.unsqueeze(1)  # [num_edges, 1, hidden_dim]
        neg_sim = torch.sum(
            news_emb_expanded * neg_entity_emb, dim=2
        )  # [num_edges, neg_samples]
        
        # InfoNCE loss
        pos_sim = pos_sim / self.temperature
        neg_sim = neg_sim / self.temperature
        
        # Concatenate positive and negative similarities
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [num_edges, 1 + neg_samples]
        
        # Compute softmax and loss
        log_prob = F.log_softmax(all_sim, dim=1)
        loss = -log_prob[:, 0].mean()  # Positive samples are at index 0
        
        return loss
    
    def entity_cooccurrence_prediction(
        self,
        entity_embeddings: torch.Tensor,
        entity_entity_edges: torch.Tensor,
        num_entities: int
    ) -> torch.Tensor:
        """
        Entity Co-occurrence Prediction pretext task.
        
        Predict whether two entities co-occur in the same news articles.
        
        Args:
            entity_embeddings: Entity node embeddings [num_entities, hidden_dim]
            entity_entity_edges: Edge indices connecting entities [2, num_edges]
            num_entities: Total number of entities
            
        Returns:
            Binary classification loss for entity co-occurrence
        """
        if entity_entity_edges.size(1) == 0:
            return torch.tensor(0.0, device=entity_embeddings.device, requires_grad=True)
        
        device = entity_embeddings.device
        
        # Positive pairs (co-occurring entities)
        pos_entity1_indices = entity_entity_edges[0]
        pos_entity2_indices = entity_entity_edges[1]
        
        pos_emb1 = entity_embeddings[pos_entity1_indices]
        pos_emb2 = entity_embeddings[pos_entity2_indices]
        
        # Compute positive pair similarities
        pos_sim = torch.sum(pos_emb1 * pos_emb2, dim=1)  # [num_pos_pairs]
        
        # Generate negative pairs
        num_pos_pairs = pos_entity1_indices.size(0)
        neg_entity1_indices = torch.randint(0, num_entities, (num_pos_pairs,), device=device)
        neg_entity2_indices = torch.randint(0, num_entities, (num_pos_pairs,), device=device)
        
        # Ensure negative pairs are different entities
        same_entity_mask = neg_entity1_indices == neg_entity2_indices
        while same_entity_mask.any():
            neg_entity2_indices[same_entity_mask] = torch.randint(
                0, num_entities, (same_entity_mask.sum(),), device=device
            )
            same_entity_mask = neg_entity1_indices == neg_entity2_indices
        
        neg_emb1 = entity_embeddings[neg_entity1_indices]
        neg_emb2 = entity_embeddings[neg_entity2_indices]
        
        # Compute negative pair similarities
        neg_sim = torch.sum(neg_emb1 * neg_emb2, dim=1)  # [num_neg_pairs]
        
        # Binary classification loss
        pos_labels = torch.ones(num_pos_pairs, device=device)
        neg_labels = torch.zeros(num_pos_pairs, device=device)
        
        all_sim = torch.cat([pos_sim, neg_sim])
        all_labels = torch.cat([pos_labels, neg_labels])
        
        loss = F.binary_cross_entropy_with_logits(all_sim, all_labels)
        
        return loss
    
    def entity_type_prediction(
        self,
        entity_embeddings: torch.Tensor,
        entity_types: torch.Tensor,
        num_entity_types: int
    ) -> torch.Tensor:
        """
        Entity Type Prediction pretext task.
        
        Predict entity types from entity embeddings.
        
        Args:
            entity_embeddings: Entity node embeddings [num_entities, hidden_dim]
            entity_types: Entity type labels [num_entities]
            num_entity_types: Number of entity types
            
        Returns:
            Cross-entropy loss for entity type prediction
        """
        if entity_embeddings.size(0) == 0:
            return torch.tensor(0.0, device=entity_embeddings.device, requires_grad=True)
        
        # Type classifier
        type_classifier = nn.Linear(
            entity_embeddings.size(1),
            num_entity_types
        ).to(entity_embeddings.device)
        
        # Predict types
        type_logits = type_classifier(entity_embeddings)
        
        # Compute loss
        loss = F.cross_entropy(type_logits, entity_types)
        
        return loss
    
    def compute_combined_pretext_loss(
        self,
        news_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
        news_entity_edges: torch.Tensor,
        entity_entity_edges: torch.Tensor,
        entity_types: torch.Tensor = None,
        entity_vocab_size: int = None,
        task_weights: Dict[str, float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined pretext task loss.
        
        Args:
            news_embeddings: News node embeddings
            entity_embeddings: Entity node embeddings  
            news_entity_edges: News-entity edge indices
            entity_entity_edges: Entity-entity edge indices
            entity_types: Entity type labels
            entity_vocab_size: Size of entity vocabulary
            task_weights: Weights for different pretext tasks
            
        Returns:
            Tuple of (combined_loss, individual_losses)
        """
        if task_weights is None:
            task_weights = {
                "masked_entity": 0.3,
                "alignment": 0.4,
                "cooccurrence": 0.2,
                "type_prediction": 0.1
            }
        
        individual_losses = {}
        total_loss = torch.tensor(0.0, device=news_embeddings.device, requires_grad=True)
        
        # Masked Entity Modeling
        if entity_vocab_size is not None and "masked_entity" in task_weights:
            try:
                entity_logits, target_entities, mask = self.masked_entity_modeling(
                    news_embeddings, entity_embeddings, news_entity_edges, entity_vocab_size
                )
                if entity_logits.size(0) > 0:
                    mem_loss = F.cross_entropy(entity_logits, target_entities)
                    individual_losses["masked_entity"] = mem_loss
                    total_loss = total_loss + task_weights["masked_entity"] * mem_loss
            except Exception as e:
                logger.warning(f"Error in masked entity modeling: {e}")
                individual_losses["masked_entity"] = torch.tensor(0.0, device=news_embeddings.device)
        
        # Entity-News Alignment
        if "alignment" in task_weights:
            try:
                alignment_loss = self.entity_news_alignment(
                    news_embeddings, entity_embeddings, news_entity_edges
                )
                individual_losses["alignment"] = alignment_loss
                total_loss = total_loss + task_weights["alignment"] * alignment_loss
            except Exception as e:
                logger.warning(f"Error in entity-news alignment: {e}")
                individual_losses["alignment"] = torch.tensor(0.0, device=news_embeddings.device)
        
        # Entity Co-occurrence Prediction
        if "cooccurrence" in task_weights:
            try:
                cooccurrence_loss = self.entity_cooccurrence_prediction(
                    entity_embeddings, entity_entity_edges, entity_embeddings.size(0)
                )
                individual_losses["cooccurrence"] = cooccurrence_loss
                total_loss = total_loss + task_weights["cooccurrence"] * cooccurrence_loss
            except Exception as e:
                logger.warning(f"Error in entity co-occurrence prediction: {e}")
                individual_losses["cooccurrence"] = torch.tensor(0.0, device=news_embeddings.device)
        
        # Entity Type Prediction
        if entity_types is not None and "type_prediction" in task_weights:
            try:
                num_entity_types = len(torch.unique(entity_types))
                type_loss = self.entity_type_prediction(
                    entity_embeddings, entity_types, num_entity_types
                )
                individual_losses["type_prediction"] = type_loss
                total_loss = total_loss + task_weights["type_prediction"] * type_loss
            except Exception as e:
                logger.warning(f"Error in entity type prediction: {e}")
                individual_losses["type_prediction"] = torch.tensor(0.0, device=news_embeddings.device)
        
        return total_loss, individual_losses
    
    def create_masked_entity_batch(
        self,
        news_texts: List[str],
        entity_lists: List[List[Dict]],
        mask_prob: float = None
    ) -> Tuple[List[str], List[List[Dict]], List[int]]:
        """
        Create a batch with masked entities for pretext training.
        
        Args:
            news_texts: List of news texts
            entity_lists: List of entity lists for each news
            mask_prob: Probability of masking (uses self.mask_prob if None)
            
        Returns:
            Tuple of (masked_texts, masked_entity_lists, masked_positions)
        """
        if mask_prob is None:
            mask_prob = self.mask_prob
        
        masked_texts = []
        masked_entity_lists = []
        masked_positions = []
        
        for text, entities in zip(news_texts, entity_lists):
            if len(entities) == 0:
                masked_texts.append(text)
                masked_entity_lists.append(entities)
                masked_positions.append([])
                continue
            
            # Determine which entities to mask
            num_to_mask = max(1, int(len(entities) * mask_prob))
            mask_indices = random.sample(range(len(entities)), num_to_mask)
            
            # Create masked version
            masked_text = text
            masked_entities = entities.copy()
            
            # Sort by start position in reverse order for correct masking
            mask_indices.sort(key=lambda i: entities[i]["start"], reverse=True)
            
            for mask_idx in mask_indices:
                entity = entities[mask_idx]
                start, end = entity["start"], entity["end"]
                
                # Replace entity with [MASK] token
                masked_text = masked_text[:start] + "[MASK]" + masked_text[end:]
                
                # Update entity information
                masked_entities[mask_idx] = {
                    **entity,
                    "text": "[MASK]",
                    "masked": True
                }
            
            masked_texts.append(masked_text)
            masked_entity_lists.append(masked_entities)
            masked_positions.append(mask_indices)
        
        return masked_texts, masked_entity_lists, masked_positions