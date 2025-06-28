"""
LESS4FD Graph Builder - Entity-aware heterogeneous graph construction.

This module extends the existing HeteroGraphBuilder to create entity-aware
heterogeneous graphs for the LESS4FD architecture.
"""

import os
import gc
import json
import numpy as np
import torch
import networkx as nx
from typing import Dict, Tuple, Optional, List, Union, Any
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm
import logging

# Import existing utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from build_hetero_graph import HeteroGraphBuilder
from utils.sample_k_shot import sample_k_shot

# Import LESS4FD utilities
from utils.entity_extractor import EntityExtractor
from utils.sampling import LESS4FDSampler
from config.less4fd_config import LESS4FD_CONFIG, DATA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LESS4FDGraphBuilder:
    """
    Entity-aware heterogeneous graph builder for LESS4FD architecture.
    
    This class extends the existing HeteroGraphBuilder functionality to include
    entity nodes and entity-aware edges in the heterogeneous graph.
    """
    
    def __init__(
        self,
        dataset_name: str,
        k_shot: int,
        embedding_type: str = "deberta",
        entity_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
        max_entities_per_news: int = 10,
        entity_similarity_threshold: float = 0.7,
        entity_knn: int = 5,
        use_entity_types: bool = True,
        seed: int = 42,
        dataset_cache_dir: str = "dataset",
        graph_cache_dir: str = "graphs_less4fd",
        entity_cache_dir: str = "entity_cache",
        **kwargs
    ):
        """
        Initialize the LESS4FD graph builder.
        
        Args:
            dataset_name: Name of the dataset ('politifact' or 'gossipcop')
            k_shot: Number of shots for few-shot learning
            embedding_type: Type of text embeddings to use
            entity_model: Model for entity extraction
            max_entities_per_news: Maximum entities per news article
            entity_similarity_threshold: Threshold for entity-entity edges
            entity_knn: K for entity KNN edges
            use_entity_types: Whether to use entity type information
            seed: Random seed
            dataset_cache_dir: Directory for dataset cache
            graph_cache_dir: Directory for graph cache
            entity_cache_dir: Directory for entity cache
            **kwargs: Additional arguments for base HeteroGraphBuilder
        """
        self.dataset_name = dataset_name.lower()
        self.k_shot = k_shot
        self.embedding_type = embedding_type
        self.entity_model = entity_model
        self.max_entities_per_news = max_entities_per_news
        self.entity_similarity_threshold = entity_similarity_threshold
        self.entity_knn = entity_knn
        self.use_entity_types = use_entity_types
        self.seed = seed
        self.dataset_cache_dir = dataset_cache_dir
        self.graph_cache_dir = graph_cache_dir
        self.entity_cache_dir = entity_cache_dir
        
        # Set up directories
        os.makedirs(self.graph_cache_dir, exist_ok=True)
        os.makedirs(self.entity_cache_dir, exist_ok=True)
        
        # Initialize entity extractor
        self.entity_extractor = EntityExtractor(
            model_name=entity_model,
            entity_types=LESS4FD_CONFIG["entity_types"],
            max_entities_per_text=max_entities_per_news,
            similarity_threshold=entity_similarity_threshold,
            cache_dir=entity_cache_dir
        )
        
        # Initialize LESS4FD sampler
        self.less4fd_sampler = LESS4FDSampler(
            seed=seed,
            meta_learning=True,
            min_entities_per_sample=1
        )
        
        # Initialize base hetero graph builder for news-news and news-interaction edges
        self.base_builder = HeteroGraphBuilder(
            dataset_name=dataset_name,
            k_shot=k_shot,
            embedding_type=embedding_type,
            seed=seed,
            dataset_cache_dir=dataset_cache_dir,
            output_dir=graph_cache_dir,
            **kwargs
        )
        
        # Load datasets
        self.train_data = self.base_builder.train_data
        self.test_data = self.base_builder.test_data
        self.text_embedding_field = f"{embedding_type}_embeddings"
        
        # Entity data cache
        self.entity_data_cache = {}
        self.entity_vocab = {}
        self.entity_embeddings = None
        self.entity_type_mapping = {}
        
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using the entity extractor.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        return self.entity_extractor.extract_entities(text)
    
    def build_entity_nodes(self, news_data: Dataset) -> Dict:
        """
        Build entity nodes from news data.
        
        Args:
            news_data: News dataset
            
        Returns:
            Dictionary containing entity information
        """
        logger.info("Extracting entities from news data...")
        
        all_entities = []
        entity_lists = []
        
        # Extract entities from all news texts
        for i, text in enumerate(tqdm(news_data["text"], desc="Extracting entities")):
            entities = self.extract_entities(text)
            entity_lists.append(entities)
            all_entities.extend(entities)
        
        # Build entity vocabulary
        self.entity_vocab = self.entity_extractor.build_entity_vocab(entity_lists)
        
        # Generate entity embeddings
        self.entity_embeddings = self.entity_extractor.generate_entity_embeddings()
        
        # Build entity type mapping
        for entities in entity_lists:
            for entity in entities:
                entity_text = entity["text"].lower()
                if entity_text in self.entity_vocab:
                    self.entity_type_mapping[entity_text] = entity["label"]
        
        logger.info(f"Built {len(self.entity_vocab)} entities with {self.entity_embeddings.shape[1]} dimensions")
        
        return {
            "entity_lists": entity_lists,
            "entity_vocab": self.entity_vocab,
            "entity_embeddings": self.entity_embeddings,
            "entity_type_mapping": self.entity_type_mapping
        }
    
    def build_entity_edges(self, news_entities: List[List[Dict]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges between entities based on co-occurrence and semantic similarity.
        
        Args:
            news_entities: List of entity lists for each news article
            
        Returns:
            Tuple of (edge_index, edge_attr) for entity-entity edges
        """
        logger.info("Building entity-entity edges...")
        
        if len(self.entity_vocab) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float)
        
        # Build co-occurrence matrix
        entity_cooccurrence = self._build_entity_cooccurrence_matrix(news_entities)
        
        # Build semantic similarity edges
        semantic_edges = self._build_entity_semantic_edges()
        
        # Combine co-occurrence and semantic edges
        all_edges = []
        all_similarities = []
        
        # Add co-occurrence edges
        cooc_indices = np.nonzero(entity_cooccurrence)
        for i, j in zip(cooc_indices[0], cooc_indices[1]):
            if i != j:  # No self-loops
                cooc_score = entity_cooccurrence[i, j]
                all_edges.append([i, j])
                all_similarities.append(cooc_score)
        
        # Add semantic similarity edges
        for (i, j), sim_score in semantic_edges:
            all_edges.append([i, j])
            all_similarities.append(sim_score)
        
        if len(all_edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float)
        
        # Convert to tensors
        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        edge_attr = torch.tensor(all_similarities, dtype=torch.float).unsqueeze(1)
        
        # Remove duplicates and self-loops
        edge_index, edge_attr = self._remove_duplicate_edges(edge_index, edge_attr)
        
        logger.info(f"Built {edge_index.shape[1]} entity-entity edges")
        
        return edge_index, edge_attr
    
    def build_news_entity_edges(
        self,
        news_entities: List[List[Dict]],
        news_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges between news and entities.
        
        Args:
            news_entities: List of entity lists for each news article
            news_indices: Indices of news articles in the graph
            
        Returns:
            Tuple of (edge_index, edge_attr) for news-entity edges
        """
        logger.info("Building news-entity edges...")
        
        news_entity_edges = []
        edge_confidences = []
        
        for news_idx, entities in enumerate(news_entities):
            if news_idx >= len(news_indices):
                continue
                
            graph_news_idx = news_idx  # Index in the graph
            
            for entity in entities:
                entity_text = entity["text"].lower()
                if entity_text in self.entity_vocab:
                    entity_idx = self.entity_vocab[entity_text]
                    confidence = entity.get("score", 1.0)
                    
                    news_entity_edges.append([graph_news_idx, entity_idx])
                    edge_confidences.append(confidence)
        
        if len(news_entity_edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float)
        
        edge_index = torch.tensor(news_entity_edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_confidences, dtype=torch.float).unsqueeze(1)
        
        logger.info(f"Built {edge_index.shape[1]} news-entity edges")
        
        return edge_index, edge_attr
    
    def build_graph(self, test_batch_indices: Optional[List[int]] = None) -> HeteroData:
        """
        Build the complete LESS4FD heterogeneous graph.
        
        Args:
            test_batch_indices: Indices for test batch (if using batch processing)
            
        Returns:
            Heterogeneous graph data with news, entity, and interaction nodes
        """
        logger.info("Building LESS4FD heterogeneous graph...")
        
        # First build the base heterogeneous graph (news + interactions)
        base_hetero_data = self.base_builder.build_graph(test_batch_indices)
        
        if base_hetero_data is None:
            logger.error("Failed to build base heterogeneous graph")
            return None
        
        # Sample data for entity extraction
        train_labeled_indices, train_labeled_data = sample_k_shot(
            self.train_data, self.k_shot, self.seed
        )
        
        # Get test indices
        if test_batch_indices is not None:
            test_indices = test_batch_indices
        else:
            test_indices = list(range(len(self.test_data)))
        
        # Combine all news data for entity extraction
        all_news_texts = (
            train_labeled_data["text"] +
            [self.test_data[i]["text"] for i in test_indices]
        )
        
        # Extract entities from all news
        entity_info = self.build_entity_nodes(
            Dataset.from_dict({"text": all_news_texts})
        )
        
        # Build entity-entity edges
        entity_edge_index, entity_edge_attr = self.build_entity_edges(
            entity_info["entity_lists"]
        )
        
        # Build news-entity edges
        news_entity_edge_index, news_entity_edge_attr = self.build_news_entity_edges(
            entity_info["entity_lists"],
            list(range(len(all_news_texts)))
        )
        
        # Create new heterogeneous data with entities
        hetero_data = HeteroData()
        
        # Copy existing node types and edges from base graph
        for node_type in base_hetero_data.node_types:
            hetero_data[node_type].x = base_hetero_data[node_type].x
            if hasattr(base_hetero_data[node_type], 'y'):
                hetero_data[node_type].y = base_hetero_data[node_type].y
            
            # Copy masks if they exist (including few-shot specific masks)
            for mask_name in ['train_labeled_mask', 'train_unlabeled_mask', 'val_mask', 'test_mask', 'train_mask']:
                if hasattr(base_hetero_data[node_type], mask_name):
                    setattr(hetero_data[node_type], mask_name, 
                           getattr(base_hetero_data[node_type], mask_name))
        
        # Copy existing edges
        for edge_type in base_hetero_data.edge_types:
            hetero_data[edge_type].edge_index = base_hetero_data[edge_type].edge_index
            if hasattr(base_hetero_data[edge_type], 'edge_attr'):
                hetero_data[edge_type].edge_attr = base_hetero_data[edge_type].edge_attr
        
        # Add entity nodes
        if len(self.entity_vocab) > 0:
            hetero_data['entity'].x = self.entity_embeddings
            
            # Create entity type labels
            entity_types = []
            for entity_text in self.entity_vocab.keys():
                entity_type = self.entity_type_mapping.get(entity_text, "MISC")
                type_idx = LESS4FD_CONFIG["entity_types"].index(entity_type) \
                    if entity_type in LESS4FD_CONFIG["entity_types"] else 3  # MISC index
                entity_types.append(type_idx)
            
            hetero_data['entity'].entity_type = torch.tensor(entity_types, dtype=torch.long)
            
            # Add entity-entity edges
            if entity_edge_index.shape[1] > 0:
                hetero_data['entity', 'related_to', 'entity'].edge_index = entity_edge_index
                hetero_data['entity', 'related_to', 'entity'].edge_attr = entity_edge_attr
            
            # Add news-entity edges (bidirectional)
            if news_entity_edge_index.shape[1] > 0:
                hetero_data['news', 'connected_to', 'entity'].edge_index = news_entity_edge_index
                hetero_data['news', 'connected_to', 'entity'].edge_attr = news_entity_edge_attr
                
                # Add reverse edges
                reverse_edge_index = torch.stack([
                    news_entity_edge_index[1],  # entity -> news
                    news_entity_edge_index[0]   # news -> entity
                ])
                hetero_data['entity', 'connected_to', 'news'].edge_index = reverse_edge_index
                hetero_data['entity', 'connected_to', 'news'].edge_attr = news_entity_edge_attr
        
        # Add metadata
        hetero_data.metadata_dict = {
            "num_entities": len(self.entity_vocab),
            "entity_vocab": self.entity_vocab,
            "entity_types": LESS4FD_CONFIG["entity_types"],
            "entity_model": self.entity_model,
            "k_shot": self.k_shot,
            "dataset_name": self.dataset_name
        }
        
        logger.info(f"Built LESS4FD heterogeneous graph with {len(hetero_data.node_types)} node types")
        logger.info(f"Node types: {hetero_data.node_types}")
        logger.info(f"Edge types: {hetero_data.edge_types}")
        
        return hetero_data
    
    def save_graph(self, graph: HeteroData, suffix: str = "") -> str:
        """
        Save the heterogeneous graph to disk.
        
        Args:
            graph: Heterogeneous graph data
            suffix: Additional suffix for filename
            
        Returns:
            Path to saved graph
        """
        filename = f"less4fd_{self.dataset_name}_k{self.k_shot}_{self.embedding_type}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".pt"
        
        filepath = os.path.join(self.graph_cache_dir, filename)
        
        torch.save(graph, filepath)
        logger.info(f"Saved LESS4FD graph to {filepath}")
        
        # Also save entity cache
        self.entity_extractor.save_caches()
        
        return filepath
    
    def load_graph(self, suffix: str = "") -> Optional[HeteroData]:
        """
        Load heterogeneous graph from disk.
        
        Args:
            suffix: Additional suffix for filename
            
        Returns:
            Loaded heterogeneous graph data or None if not found
        """
        filename = f"less4fd_{self.dataset_name}_k{self.k_shot}_{self.embedding_type}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".pt"
        
        filepath = os.path.join(self.graph_cache_dir, filename)
        
        if os.path.exists(filepath):
            graph = torch.load(filepath)
            logger.info(f"Loaded LESS4FD graph from {filepath}")
            return graph
        
        return None
    
    def _build_entity_cooccurrence_matrix(self, news_entities: List[List[Dict]]) -> np.ndarray:
        """Build entity co-occurrence matrix."""
        num_entities = len(self.entity_vocab)
        cooccurrence = np.zeros((num_entities, num_entities))
        
        for entities in news_entities:
            entity_indices = []
            for entity in entities:
                entity_text = entity["text"].lower()
                if entity_text in self.entity_vocab:
                    entity_indices.append(self.entity_vocab[entity_text])
            
            # Add co-occurrence for all pairs in this news article
            for i in range(len(entity_indices)):
                for j in range(i + 1, len(entity_indices)):
                    idx1, idx2 = entity_indices[i], entity_indices[j]
                    cooccurrence[idx1, idx2] += 1
                    cooccurrence[idx2, idx1] += 1
        
        # Normalize by maximum co-occurrence
        if cooccurrence.max() > 0:
            cooccurrence = cooccurrence / cooccurrence.max()
        
        return cooccurrence
    
    def _build_entity_semantic_edges(self) -> List[Tuple[Tuple[int, int], float]]:
        """Build semantic similarity edges between entities."""
        edges = []
        
        if self.entity_embeddings is None or len(self.entity_vocab) < 2:
            return edges
        
        # Compute similarity matrix
        similarities = cosine_similarity(
            self.entity_embeddings.numpy(),
            self.entity_embeddings.numpy()
        )
        
        # Find similar entity pairs
        entity_list = list(self.entity_vocab.keys())
        for i in range(len(entity_list)):
            # Get top-k similar entities
            similar_indices = np.argsort(similarities[i])[::-1][1:self.entity_knn+1]
            
            for j in similar_indices:
                similarity = similarities[i, j]
                if similarity > self.entity_similarity_threshold:
                    edges.append(((i, j), similarity))
        
        return edges
    
    def _remove_duplicate_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove duplicate edges and self-loops."""
        if edge_index.shape[1] == 0:
            return edge_index, edge_attr
        
        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
        
        # Remove duplicates by sorting and keeping unique
        if edge_index.shape[1] > 0:
            # Create edge tuples for uniqueness check
            edge_tuples = [tuple(sorted([edge_index[0, i].item(), edge_index[1, i].item()])) 
                          for i in range(edge_index.shape[1])]
            
            unique_edges = {}
            for i, edge_tuple in enumerate(edge_tuples):
                if edge_tuple not in unique_edges:
                    unique_edges[edge_tuple] = i
            
            # Keep only unique edges
            unique_indices = list(unique_edges.values())
            edge_index = edge_index[:, unique_indices]
            edge_attr = edge_attr[unique_indices]
        
        return edge_index, edge_attr
    
    def analyze_entity_graph(self, hetero_graph: HeteroData) -> Dict:
        """
        Analyze the entity-aware heterogeneous graph.
        
        Args:
            hetero_graph: Heterogeneous graph data
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis["num_node_types"] = len(hetero_graph.node_types)
        analysis["num_edge_types"] = len(hetero_graph.edge_types)
        analysis["node_types"] = hetero_graph.node_types
        analysis["edge_types"] = hetero_graph.edge_types
        
        # Node statistics
        node_stats = {}
        for node_type in hetero_graph.node_types:
            if hasattr(hetero_graph[node_type], 'x'):
                num_nodes = hetero_graph[node_type].x.shape[0]
                node_dim = hetero_graph[node_type].x.shape[1]
                node_stats[node_type] = {"num_nodes": num_nodes, "feature_dim": node_dim}
        
        analysis["node_statistics"] = node_stats
        
        # Edge statistics
        edge_stats = {}
        for edge_type in hetero_graph.edge_types:
            if hasattr(hetero_graph[edge_type], 'edge_index'):
                num_edges = hetero_graph[edge_type].edge_index.shape[1]
                edge_stats[edge_type] = {"num_edges": num_edges}
        
        analysis["edge_statistics"] = edge_stats
        
        # Entity-specific statistics
        if 'entity' in hetero_graph.node_types:
            entity_info = {}
            entity_info["num_entities"] = hetero_graph['entity'].x.shape[0]
            
            if hasattr(hetero_graph['entity'], 'entity_type'):
                entity_types = hetero_graph['entity'].entity_type
                type_counts = torch.bincount(entity_types)
                entity_info["type_distribution"] = {
                    LESS4FD_CONFIG["entity_types"][i]: count.item()
                    for i, count in enumerate(type_counts)
                    if i < len(LESS4FD_CONFIG["entity_types"])
                }
            
            analysis["entity_info"] = entity_info
        
        return analysis