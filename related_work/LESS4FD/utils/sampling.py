"""
Few-shot sampling utilities for LESS4FD architecture.

This module provides specialized sampling functionality for few-shot learning
scenarios in the LESS4FD framework, extending the base sampling utilities.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from datasets import Dataset
from collections import defaultdict
import random
import logging

# Import base sampling utility
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.sample_k_shot import sample_k_shot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LESS4FDSampler:
    """
    Specialized sampler for LESS4FD few-shot learning scenarios.
    
    This class handles:
    - Entity-aware few-shot sampling
    - Meta-learning task generation
    - Support/query set splitting
    - Entity-balanced sampling
    """
    
    def __init__(
        self,
        seed: int = 42,
        meta_learning: bool = True,
        support_query_ratio: float = 0.5,
        min_entities_per_sample: int = 1,
        entity_diversity_weight: float = 0.3
    ):
        """
        Initialize the LESS4FDSampler.
        
        Args:
            seed: Random seed for reproducibility
            meta_learning: Whether to use meta-learning sampling
            support_query_ratio: Ratio of support to query samples
            min_entities_per_sample: Minimum entities required per sample
            entity_diversity_weight: Weight for entity diversity in sampling
        """
        self.seed = seed
        self.meta_learning = meta_learning
        self.support_query_ratio = support_query_ratio
        self.min_entities_per_sample = min_entities_per_sample
        self.entity_diversity_weight = entity_diversity_weight
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def sample_entity_aware_k_shot(
        self,
        train_data: Dataset,
        k: int,
        entity_data: List[List[Dict]] = None,
        balance_entities: bool = True
    ) -> Tuple[List[int], Dict]:
        """
        Sample k-shot data with entity awareness.
        
        Args:
            train_data: Training dataset
            k: Number of shots per class
            entity_data: Pre-extracted entity data for each sample
            balance_entities: Whether to balance entity diversity
            
        Returns:
            Tuple of (selected indices, sampled data)
        """
        if not balance_entities or entity_data is None:
            # Fall back to standard k-shot sampling
            return sample_k_shot(train_data, k, self.seed)
        
        # Filter samples with sufficient entities
        valid_indices = []
        valid_entity_data = []
        
        for i, entities in enumerate(entity_data):
            if len(entities) >= self.min_entities_per_sample:
                valid_indices.append(i)
                valid_entity_data.append(entities)
        
        if len(valid_indices) == 0:
            logger.warning("No samples with sufficient entities found. Using standard sampling.")
            return sample_k_shot(train_data, k, self.seed)
        
        # Create filtered dataset
        filtered_data = train_data.select(valid_indices)
        
        # Sample with entity diversity consideration
        selected_indices, sampled_data = self._sample_with_entity_diversity(
            filtered_data, valid_entity_data, valid_indices, k
        )
        
        return selected_indices, sampled_data
    
    def generate_meta_learning_tasks(
        self,
        train_data: Dataset,
        num_tasks: int,
        k_shot: int,
        entity_data: List[List[Dict]] = None
    ) -> List[Dict]:
        """
        Generate meta-learning tasks for few-shot learning.
        
        Args:
            train_data: Training dataset
            num_tasks: Number of tasks to generate
            k_shot: Number of shots per class per task
            entity_data: Entity data for each sample
            
        Returns:
            List of task dictionaries with support and query sets
        """
        tasks = []
        
        for task_id in range(num_tasks):
            # Use different seed for each task
            task_seed = self.seed + task_id
            
            # Sample support set
            support_indices, support_data = self.sample_entity_aware_k_shot(
                train_data, k_shot, entity_data, balance_entities=True
            )
            
            # Sample query set from remaining data
            remaining_indices = [
                i for i in range(len(train_data))
                if i not in support_indices
            ]
            
            if len(remaining_indices) > 0:
                # Sample query set
                query_k = max(1, k_shot // 2)  # Smaller query set
                
                # Create temporary dataset for query sampling
                remaining_data = train_data.select(remaining_indices)
                remaining_entity_data = None
                if entity_data:
                    remaining_entity_data = [entity_data[i] for i in remaining_indices]
                
                query_rel_indices, query_data = self.sample_entity_aware_k_shot(
                    remaining_data, query_k, remaining_entity_data, balance_entities=True
                )
                
                # Convert relative indices to absolute indices
                query_indices = [remaining_indices[i] for i in query_rel_indices]
            else:
                query_indices = []
                query_data = {key: [] for key in train_data.column_names}
            
            task = {
                "task_id": task_id,
                "support_indices": support_indices,
                "support_data": support_data,
                "query_indices": query_indices,
                "query_data": query_data,
                "k_shot": k_shot
            }
            
            tasks.append(task)
        
        logger.info(f"Generated {len(tasks)} meta-learning tasks")
        return tasks
    
    def split_support_query(
        self,
        indices: List[int],
        data: Dict,
        support_ratio: float = None
    ) -> Tuple[Tuple[List[int], Dict], Tuple[List[int], Dict]]:
        """
        Split data into support and query sets.
        
        Args:
            indices: Data indices
            data: Data dictionary
            support_ratio: Ratio of support samples (default: self.support_query_ratio)
            
        Returns:
            Tuple of ((support_indices, support_data), (query_indices, query_data))
        """
        if support_ratio is None:
            support_ratio = self.support_query_ratio
        
        # Group by class
        class_indices = defaultdict(list)
        for i, idx in enumerate(indices):
            label = data["label"][i]
            class_indices[label].append((i, idx))
        
        support_indices = []
        query_indices = []
        support_data = {key: [] for key in data.keys()}
        query_data = {key: [] for key in data.keys()}
        
        for label, label_indices in class_indices.items():
            # Shuffle indices for this class
            np.random.shuffle(label_indices)
            
            # Split into support and query
            num_support = max(1, int(len(label_indices) * support_ratio))
            
            support_class_indices = label_indices[:num_support]
            query_class_indices = label_indices[num_support:]
            
            # Add to support set
            for data_idx, orig_idx in support_class_indices:
                support_indices.append(orig_idx)
                for key in data.keys():
                    support_data[key].append(data[key][data_idx])
            
            # Add to query set
            for data_idx, orig_idx in query_class_indices:
                query_indices.append(orig_idx)
                for key in data.keys():
                    query_data[key].append(data[key][data_idx])
        
        return (support_indices, support_data), (query_indices, query_data)
    
    def _sample_with_entity_diversity(
        self,
        filtered_data: Dataset,
        entity_data: List[List[Dict]],
        original_indices: List[int],
        k: int
    ) -> Tuple[List[int], Dict]:
        """
        Sample data considering entity diversity.
        
        Args:
            filtered_data: Filtered dataset
            entity_data: Entity data for filtered samples
            original_indices: Original indices in the full dataset
            k: Number of samples per class
            
        Returns:
            Tuple of (selected original indices, sampled data)
        """
        # Group samples by class
        class_samples = defaultdict(list)
        for i, label in enumerate(filtered_data["label"]):
            class_samples[label].append(i)
        
        selected_indices = []
        sampled_data = {key: [] for key in filtered_data.column_names}
        
        for label, sample_indices in class_samples.items():
            # Score samples based on entity diversity
            sample_scores = self._compute_entity_diversity_scores(
                sample_indices, entity_data
            )
            
            # Sort by score and select top k
            scored_samples = list(zip(sample_indices, sample_scores))
            scored_samples.sort(key=lambda x: x[1], reverse=True)
            
            # Select top k samples for this class
            selected_class_samples = scored_samples[:min(k, len(scored_samples))]
            
            for sample_idx, score in selected_class_samples:
                # Map back to original indices
                original_idx = original_indices[sample_idx]
                selected_indices.append(original_idx)
                
                # Add data
                for key in filtered_data.column_names:
                    sampled_data[key].append(filtered_data[key][sample_idx])
        
        return selected_indices, sampled_data
    
    def _compute_entity_diversity_scores(
        self,
        sample_indices: List[int],
        entity_data: List[List[Dict]]
    ) -> List[float]:
        """
        Compute entity diversity scores for samples.
        
        Args:
            sample_indices: Indices of samples to score
            entity_data: Entity data for all samples
            
        Returns:
            List of diversity scores
        """
        scores = []
        
        # Collect all entities in this class
        all_entities = set()
        sample_entities = []
        
        for idx in sample_indices:
            entities = entity_data[idx]
            entity_texts = {e["text"].lower() for e in entities}
            sample_entities.append(entity_texts)
            all_entities.update(entity_texts)
        
        # Score each sample based on entity uniqueness and diversity
        for sample_entity_set in sample_entities:
            if len(sample_entity_set) == 0:
                scores.append(0.0)
                continue
            
            # Entity diversity score
            num_entities = len(sample_entity_set)
            entity_types = len(set(
                e["label"] for idx in sample_indices
                for e in entity_data[idx]
                if e["text"].lower() in sample_entity_set
            ))
            
            # Combine number of entities and type diversity
            diversity_score = (
                num_entities * (1 - self.entity_diversity_weight) +
                entity_types * self.entity_diversity_weight
            )
            
            scores.append(diversity_score)
        
        return scores
    
    def sample_balanced_entities(
        self,
        entity_lists: List[List[Dict]],
        target_entities_per_class: int = 50
    ) -> Dict[str, List[str]]:
        """
        Sample balanced entities across classes for entity vocabulary.
        
        Args:
            entity_lists: List of entity lists for each sample
            target_entities_per_class: Target number of entities per entity type
            
        Returns:
            Dictionary mapping entity types to selected entity texts
        """
        # Group entities by type
        entities_by_type = defaultdict(list)
        
        for entities in entity_lists:
            for entity in entities:
                entity_type = entity["label"]
                entity_text = entity["text"].lower()
                entities_by_type[entity_type].append(entity_text)
        
        # Sample balanced entities
        balanced_entities = {}
        
        for entity_type, entity_texts in entities_by_type.items():
            # Remove duplicates and sample
            unique_entities = list(set(entity_texts))
            
            if len(unique_entities) <= target_entities_per_class:
                balanced_entities[entity_type] = unique_entities
            else:
                # Random sampling
                np.random.shuffle(unique_entities)
                balanced_entities[entity_type] = unique_entities[:target_entities_per_class]
        
        logger.info(f"Sampled balanced entities: {[(k, len(v)) for k, v in balanced_entities.items()]}")
        
        return balanced_entities
    
    def create_episodic_batch(
        self,
        support_data: Dict,
        query_data: Dict,
        k_shot: int,
        n_way: int = 2
    ) -> Dict:
        """
        Create episodic batch for meta-learning.
        
        Args:
            support_data: Support set data
            query_data: Query set data
            k_shot: Number of shots per class
            n_way: Number of classes (for fake news: 2)
            
        Returns:
            Episodic batch dictionary
        """
        batch = {
            "support": support_data,
            "query": query_data,
            "k_shot": k_shot,
            "n_way": n_way,
            "support_size": len(support_data["label"]),
            "query_size": len(query_data["label"])
        }
        
        return batch