"""
Entity extraction utilities for LESS4FD architecture.

This module provides functionality for extracting named entities from news text
using transformer-based NER models and generating entity embeddings.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    pipeline, BatchEncoding
)
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Entity extraction and processing for LESS4FD architecture.
    
    This class handles:
    - Named Entity Recognition using transformer models
    - Entity embedding generation
    - Entity similarity computation
    - Entity type classification
    """
    
    def __init__(
        self,
        model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
        entity_types: List[str] = None,
        max_entities_per_text: int = 10,
        similarity_threshold: float = 0.7,
        cache_dir: str = "entity_cache",
        device: str = None
    ):
        """
        Initialize the EntityExtractor.
        
        Args:
            model_name: HuggingFace model for NER
            entity_types: List of entity types to extract
            max_entities_per_text: Maximum entities to extract per text
            similarity_threshold: Threshold for entity similarity
            cache_dir: Directory for caching entity data
            device: Device for model inference
        """
        self.model_name = model_name
        self.entity_types = entity_types or ["PERSON", "ORG", "GPE", "MISC"]
        self.max_entities_per_text = max_entities_per_text
        self.similarity_threshold = similarity_threshold
        self.cache_dir = cache_dir
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model and tokenizer
        logger.info(f"Loading NER model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="average",
            device=0 if self.device.type == "cuda" else -1
        )
        
        # Initialize entity cache
        os.makedirs(self.cache_dir, exist_ok=True)
        self.entity_cache_file = os.path.join(self.cache_dir, "entities.json")
        self.embedding_cache_file = os.path.join(self.cache_dir, "entity_embeddings.pt")
        
        # Load cached data if available
        self.entity_cache = self._load_entity_cache()
        self.embedding_cache = self._load_embedding_cache()
        
        # Entity vocabulary and embeddings
        self.entity_vocab = {}
        self.entity_embeddings = None
        self.entity_type_mapping = {}
        
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with keys:
            - text: entity text
            - label: entity type
            - start: start position
            - end: end position
            - score: confidence score
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Check cache first
        text_hash = hash(text)
        if text_hash in self.entity_cache:
            return self.entity_cache[text_hash]
        
        try:
            # Extract entities using NER pipeline
            entities = self.ner_pipeline(text)
            
            # Filter and process entities
            processed_entities = []
            for entity in entities:
                # Filter by entity type
                entity_type = entity["entity_group"].upper()
                if entity_type in self.entity_types:
                    processed_entity = {
                        "text": entity["word"].strip(),
                        "label": entity_type,
                        "start": entity["start"],
                        "end": entity["end"],
                        "score": entity["score"]
                    }
                    processed_entities.append(processed_entity)
            
            # Sort by confidence score and take top entities
            processed_entities.sort(key=lambda x: x["score"], reverse=True)
            processed_entities = processed_entities[:self.max_entities_per_text]
            
            # Cache results
            self.entity_cache[text_hash] = processed_entities
            
            return processed_entities
            
        except Exception as e:
            logger.warning(f"Error extracting entities from text: {e}")
            return []
    
    def get_entity_embedding(self, entity_text: str) -> Optional[torch.Tensor]:
        """
        Get embedding for an entity using the model's hidden states.
        
        Args:
            entity_text: Entity text
            
        Returns:
            Entity embedding tensor
        """
        if not entity_text:
            return None
        
        # Check embedding cache
        if entity_text in self.embedding_cache:
            return self.embedding_cache[entity_text]
        
        try:
            # Tokenize entity text
            inputs = self.tokenizer(
                entity_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32
            ).to(self.device)
            
            # Get hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # Last layer
                
                # Average pooling over sequence length
                attention_mask = inputs.attention_mask
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                summed = masked_hidden.sum(dim=1)
                lengths = attention_mask.sum(dim=1, keepdim=True)
                entity_embedding = summed / lengths
                
                # Move to CPU for caching
                entity_embedding = entity_embedding.squeeze(0).cpu()
            
            # Cache the embedding
            self.embedding_cache[entity_text] = entity_embedding
            
            return entity_embedding
            
        except Exception as e:
            logger.warning(f"Error generating embedding for entity '{entity_text}': {e}")
            return None
    
    def build_entity_vocab(self, entity_lists: List[List[Dict]]) -> Dict[str, int]:
        """
        Build entity vocabulary from multiple entity lists.
        
        Args:
            entity_lists: List of entity lists from multiple texts
            
        Returns:
            Entity vocabulary mapping entity text to index
        """
        entity_counts = defaultdict(int)
        entity_types = {}
        
        # Count entity occurrences and track types
        for entities in entity_lists:
            for entity in entities:
                entity_text = entity["text"].lower()
                entity_counts[entity_text] += 1
                entity_types[entity_text] = entity["label"]
        
        # Sort entities by frequency
        sorted_entities = sorted(
            entity_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build vocabulary
        self.entity_vocab = {
            entity: idx for idx, (entity, count) in enumerate(sorted_entities)
        }
        
        # Store entity type mapping
        self.entity_type_mapping = entity_types
        
        logger.info(f"Built entity vocabulary with {len(self.entity_vocab)} entities")
        
        return self.entity_vocab
    
    def generate_entity_embeddings(self) -> torch.Tensor:
        """
        Generate embeddings for all entities in vocabulary.
        
        Returns:
            Tensor of entity embeddings [num_entities, embedding_dim]
        """
        if not self.entity_vocab:
            raise ValueError("Entity vocabulary not built. Call build_entity_vocab first.")
        
        embeddings = []
        
        logger.info(f"Generating embeddings for {len(self.entity_vocab)} entities")
        
        for entity_text in self.entity_vocab.keys():
            embedding = self.get_entity_embedding(entity_text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # Use zero embedding for failed extractions
                embedding_dim = 768  # Default BERT embedding size
                embeddings.append(torch.zeros(embedding_dim))
        
        self.entity_embeddings = torch.stack(embeddings)
        
        logger.info(f"Generated entity embeddings shape: {self.entity_embeddings.shape}")
        
        return self.entity_embeddings
    
    def compute_entity_similarity(
        self,
        entity1_idx: int,
        entity2_idx: int
    ) -> float:
        """
        Compute similarity between two entities.
        
        Args:
            entity1_idx: Index of first entity
            entity2_idx: Index of second entity
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.entity_embeddings is None:
            raise ValueError("Entity embeddings not generated. Call generate_entity_embeddings first.")
        
        if entity1_idx >= len(self.entity_embeddings) or entity2_idx >= len(self.entity_embeddings):
            return 0.0
        
        emb1 = self.entity_embeddings[entity1_idx].unsqueeze(0)
        emb2 = self.entity_embeddings[entity2_idx].unsqueeze(0)
        
        similarity = torch.cosine_similarity(emb1, emb2, dim=1).item()
        return max(0.0, similarity)  # Ensure non-negative
    
    def find_similar_entities(
        self,
        entity_idx: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find most similar entities to a given entity.
        
        Args:
            entity_idx: Index of query entity
            top_k: Number of similar entities to return
            
        Returns:
            List of (entity_idx, similarity_score) tuples
        """
        if self.entity_embeddings is None:
            raise ValueError("Entity embeddings not generated. Call generate_entity_embeddings first.")
        
        if entity_idx >= len(self.entity_embeddings):
            return []
        
        query_embedding = self.entity_embeddings[entity_idx].unsqueeze(0)
        similarities = torch.cosine_similarity(
            query_embedding,
            self.entity_embeddings,
            dim=1
        )
        
        # Remove self-similarity
        similarities[entity_idx] = -1.0
        
        # Get top-k similar entities
        top_similarities, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        results = [
            (idx.item(), sim.item())
            for idx, sim in zip(top_indices, top_similarities)
            if sim.item() > 0  # Only positive similarities
        ]
        
        return results
    
    def get_entity_type_embedding(self, entity_type: str) -> torch.Tensor:
        """
        Get embedding for entity type.
        
        Args:
            entity_type: Entity type (PERSON, ORG, etc.)
            
        Returns:
            Type embedding tensor
        """
        # Simple one-hot encoding for entity types
        type_to_idx = {t: i for i, t in enumerate(self.entity_types)}
        
        if entity_type not in type_to_idx:
            entity_type = "MISC"  # Default type
        
        type_embedding = torch.zeros(len(self.entity_types))
        type_embedding[type_to_idx[entity_type]] = 1.0
        
        return type_embedding
    
    def _load_entity_cache(self) -> Dict:
        """Load entity cache from disk."""
        if os.path.exists(self.entity_cache_file):
            try:
                with open(self.entity_cache_file, 'r') as f:
                    cache = json.load(f)
                    # Convert string keys back to int
                    return {int(k): v for k, v in cache.items()}
            except Exception as e:
                logger.warning(f"Error loading entity cache: {e}")
        return {}
    
    def _save_entity_cache(self):
        """Save entity cache to disk."""
        try:
            with open(self.entity_cache_file, 'w') as f:
                # Convert int keys to strings for JSON
                cache_to_save = {str(k): v for k, v in self.entity_cache.items()}
                json.dump(cache_to_save, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving entity cache: {e}")
    
    def _load_embedding_cache(self) -> Dict:
        """Load embedding cache from disk."""
        if os.path.exists(self.embedding_cache_file):
            try:
                cache = torch.load(self.embedding_cache_file, map_location='cpu')
                return cache
            except Exception as e:
                logger.warning(f"Error loading embedding cache: {e}")
        return {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        try:
            torch.save(self.embedding_cache, self.embedding_cache_file)
        except Exception as e:
            logger.warning(f"Error saving embedding cache: {e}")
    
    def save_caches(self):
        """Save all caches to disk."""
        self._save_entity_cache()
        self._save_embedding_cache()
        logger.info("Entity and embedding caches saved")
    
    def clear_caches(self):
        """Clear all caches."""
        self.entity_cache.clear()
        self.embedding_cache.clear()
        if os.path.exists(self.entity_cache_file):
            os.remove(self.entity_cache_file)
        if os.path.exists(self.embedding_cache_file):
            os.remove(self.embedding_cache_file)
        logger.info("All caches cleared")