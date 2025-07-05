"""
Simplified LESS4FD Graph Builder.

A self-contained implementation that builds entity-aware heterogeneous graphs 
for fake news detection without complex meta-learning components.
Based on the LESS4FD paper: Learning with Entity-aware Self-Supervised Framework for Fake News Detection.
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
from argparse import ArgumentParser

# Copy necessary utilities from main repository
def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def sample_k_shot(train_data: Dataset, k: int, seed: int = 42) -> Tuple[List[int], Dict]:
    """
    Sample k examples per class, returning both indices and dataset.
    
    Args:
        train_data: Original training dataset
        k: Number of samples per class
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (selected indices, sampled data dictionary)
    """
    # Initialize output
    sampled_data = {key: [] for key in train_data.column_names}
    selected_indices = []

    # Sample k examples from each class
    labels = set(train_data["label"])
    for label in labels:
        # Filter to this class
        label_data = train_data.filter(lambda x: x["label"] == label)
        
        # Shuffle and select first k examples
        sampled_label_data = label_data.shuffle(seed=seed).select(
            range(min(k, len(label_data)))
        )
        
        label_indices = [i for i, l in enumerate(train_data["label"]) if l == label]
        
        # Get shuffled indices
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(label_indices)[:min(k, len(label_indices))]
        
        # Add to our list of selected indices
        selected_indices.extend(shuffled_indices)
        
        # Add the data to our sampled dataset
        for key in train_data.column_names:
            sampled_data[key].extend(sampled_label_data[key])
    
    return selected_indices, sampled_data


class SimpleLESS4FDGraphBuilder:
    """
    Simplified LESS4FD graph builder for entity-aware fake news detection.
    
    This implementation:
    1. Loads data from Hugging Face datasets (same as main repository)
    2. Builds heterogeneous graphs with news and interaction nodes
    3. Adds basic entity-aware features without complex entity extraction
    4. Uses k-shot sampling for few-shot learning scenarios
    """
    
    def __init__(
        self,
        dataset_name: str,
        k_shot: int,
        embedding_type: str = "deberta",
        edge_policy: str = "knn_test_isolated",
        k_neighbors: int = 5,
        enable_entities: bool = True,
        partial_unlabeled: bool = True,
        sample_unlabeled_factor: int = 5,
        output_dir: str = "graphs_less4fd",
        dataset_cache_dir: str = "dataset",
        seed: int = 42,
        no_interactions: bool = False
    ):
        """
        Initialize the simplified LESS4FD graph builder.
        
        Args:
            dataset_name: Dataset name (politifact, gossipcop)
            k_shot: Number of labeled samples per class (3-16)
            embedding_type: Type of embeddings (bert, roberta, deberta, etc.)
            edge_policy: Edge construction policy (knn, knn_test_isolated)
            k_neighbors: Number of neighbors for KNN edges
            enable_entities: Whether to add entity-aware features
            partial_unlabeled: Whether to use only partial unlabeled data
            sample_unlabeled_factor: Factor for unlabeled sampling
            output_dir: Directory to save graphs
            dataset_cache_dir: Directory for dataset cache
            seed: Random seed
            no_interactions: Whether to exclude interaction nodes
        """
        self.dataset_name = dataset_name
        self.k_shot = k_shot
        self.embedding_type = embedding_type
        self.edge_policy = edge_policy
        self.k_neighbors = k_neighbors
        self.enable_entities = enable_entities
        self.partial_unlabeled = partial_unlabeled
        self.sample_unlabeled_factor = sample_unlabeled_factor
        self.output_dir = output_dir
        self.dataset_cache_dir = dataset_cache_dir
        self.seed = seed
        self.no_interactions = no_interactions
        
        # Text embedding field mapping
        self.text_embedding_field = f"{embedding_type}_embeddings"
        
        # Data storage
        self.dataset = None
        self.train_data = None
        self.test_data = None
        
        # Sampling indices
        self.train_labeled_indices = None
        self.train_unlabeled_indices = None
        self.test_indices = None
        
        # Entity configuration (simplified)
        self.entity_config = {
            "max_entities_per_news": 5,
            "entity_dim": 32,
            "entity_similarity_threshold": 0.7
        }

    def load_dataset(self) -> None:
        """Load dataset from Hugging Face and perform initial checks."""
        print(f"Loading dataset '{self.dataset_name}' with '{self.embedding_type}' embeddings...")
        # Map dataset names to correct HuggingFace names
        dataset_name_map = {
            "politifact": "PolitiFact",
            "gossipcop": "GossipCop"
        }
        mapped_name = dataset_name_map.get(self.dataset_name, self.dataset_name)
        hf_dataset_name = f"LittleFish-Coder/Fake_News_{mapped_name}"
        
        # Download from huggingface and cache to local path
        local_hf_dir = os.path.join(self.dataset_cache_dir, f"{self.dataset_name}_hf")
        if os.path.exists(local_hf_dir):
            print(f"Loading dataset from local path: {local_hf_dir}")
            dataset = load_from_disk(local_hf_dir)
        else:
            print(f"Loading dataset from huggingface: {hf_dataset_name}")
            dataset = load_dataset(hf_dataset_name, download_mode="reuse_cache_if_exists")
            dataset.save_to_disk(local_hf_dir)

        # Store dataset
        self.dataset = {
            "train": dataset["train"], 
            "test": dataset["test"]
        }
        self.train_data = self.dataset["train"]
        self.test_data = self.dataset["test"]

        print(f"Dataset loaded: {len(self.train_data)} train, {len(self.test_data)} test samples")
        
        # Verify embedding field exists
        if self.text_embedding_field not in self.train_data.column_names:
            raise ValueError(f"Embedding field '{self.text_embedding_field}' not found in dataset")

    def sample_k_shot_indices(self) -> None:
        """Sample k-shot labeled indices and unlabeled indices."""
        print(f"Sampling {self.k_shot}-shot labeled data...")
        
        # Sample labeled indices using the copied utility
        labeled_indices, _ = sample_k_shot(self.train_data, self.k_shot, self.seed)
        self.train_labeled_indices = np.array(labeled_indices)
        
        # Create unlabeled indices (remaining training data)
        all_train_indices = np.arange(len(self.train_data))
        unlabeled_candidates = np.setdiff1d(all_train_indices, self.train_labeled_indices)
        
        # Optionally sample partial unlabeled data
        if self.partial_unlabeled and len(unlabeled_candidates) > 0:
            num_classes = len(set(self.train_data["label"]))
            max_unlabeled = num_classes * self.k_shot * self.sample_unlabeled_factor
            
            if len(unlabeled_candidates) > max_unlabeled:
                np.random.seed(self.seed)
                unlabeled_candidates = np.random.choice(
                    unlabeled_candidates, max_unlabeled, replace=False
                )
        
        self.train_unlabeled_indices = unlabeled_candidates
        
        # Use all test data
        self.test_indices = np.arange(len(self.test_data))
        
        print(f"Sampled: {len(self.train_labeled_indices)} labeled, "
              f"{len(self.train_unlabeled_indices)} unlabeled, "
              f"{len(self.test_indices)} test")

    def build_knn_edges(self, embeddings: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build KNN edges based on cosine similarity.
        
        Args:
            embeddings: Node embeddings
            k: Number of neighbors
            
        Returns:
            Edge indices and weights
        """
        print(f"Building KNN edges with k={k}...")
        
        # Compute cosine similarity
        similarities = cosine_similarity(embeddings)
        
        edge_indices = []
        edge_weights = []
        
        for i in range(len(embeddings)):
            # Get top-k similar nodes (excluding self)
            sim_scores = similarities[i]
            top_k_indices = np.argsort(sim_scores)[::-1][1:k+1]  # Exclude self (index 0)
            
            for j in top_k_indices:
                if sim_scores[j] > 0:  # Only positive similarities
                    edge_indices.append([i, j])
                    edge_weights.append(sim_scores[j])
        
        if len(edge_indices) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_weight

    def build_knn_test_isolated_edges(
        self, 
        train_labeled_emb: np.ndarray,
        train_unlabeled_emb: np.ndarray, 
        test_emb: np.ndarray,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build KNN edges with test isolation (test nodes only connect to training nodes).
        """
        print(f"Building test-isolated KNN edges with k={k}...")
        
        all_embeddings = np.concatenate([train_labeled_emb, train_unlabeled_emb, test_emb])
        num_train_labeled = len(train_labeled_emb)
        num_train_unlabeled = len(train_unlabeled_emb)
        num_test = len(test_emb)
        
        edge_indices = []
        edge_weights = []
        
        # Compute full similarity matrix
        similarities = cosine_similarity(all_embeddings)
        
        # Training nodes can connect to any other training nodes
        for i in range(num_train_labeled + num_train_unlabeled):
            sim_scores = similarities[i, :num_train_labeled + num_train_unlabeled]
            top_k_indices = np.argsort(sim_scores)[::-1][1:k+1]  # Exclude self
            
            for j in top_k_indices:
                if j < len(sim_scores) and sim_scores[j] > 0:
                    edge_indices.append([i, j])
                    edge_weights.append(sim_scores[j])
        
        # Test nodes only connect to training nodes
        test_start_idx = num_train_labeled + num_train_unlabeled
        for i in range(test_start_idx, test_start_idx + num_test):
            sim_scores = similarities[i, :num_train_labeled + num_train_unlabeled]
            top_k_indices = np.argsort(sim_scores)[::-1][:k]
            
            for j in top_k_indices:
                if sim_scores[j] > 0:
                    edge_indices.append([i, j])
                    edge_weights.append(sim_scores[j])
        
        if len(edge_indices) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_weight

    def add_entity_features(self, news_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Add simplified entity-aware features to news embeddings.
        
        In a full implementation, this would:
        1. Extract named entities from text
        2. Create entity embeddings
        3. Aggregate entity information
        
        For simplicity, we add learnable entity features.
        """
        if not self.enable_entities:
            return news_embeddings
            
        print("Adding simplified entity-aware features...")
        
        num_nodes = news_embeddings.size(0)
        entity_dim = self.entity_config["entity_dim"]
        
        # Simple entity features (in practice, this would be based on actual entities)
        entity_features = torch.randn(num_nodes, entity_dim) * 0.1
        
        # Concatenate with original embeddings
        enhanced_embeddings = torch.cat([news_embeddings, entity_features], dim=1)
        
        print(f"Enhanced embeddings: {news_embeddings.shape} -> {enhanced_embeddings.shape}")
        return enhanced_embeddings

    def build_hetero_graph(self) -> HeteroData:
        """Build the heterogeneous graph with news and optionally interaction nodes."""
        print("Building LESS4FD heterogeneous graph...")
        
        # Load dataset and sample indices
        self.load_dataset()
        self.sample_k_shot_indices()
        
        # Extract embeddings and labels
        train_labeled_emb = np.array(
            self.train_data.select(self.train_labeled_indices.tolist())[self.text_embedding_field]
        )
        train_labeled_labels = np.array(
            self.train_data.select(self.train_labeled_indices.tolist())["label"]
        )
        
        train_unlabeled_emb = np.array(
            self.train_data.select(self.train_unlabeled_indices.tolist())[self.text_embedding_field]
        )
        train_unlabeled_labels = np.array(
            self.train_data.select(self.train_unlabeled_indices.tolist())["label"]
        )
        
        test_emb = np.array(
            self.test_data.select(self.test_indices.tolist())[self.text_embedding_field]
        )
        test_labels = np.array(
            self.test_data.select(self.test_indices.tolist())["label"]
        )
        
        # Concatenate all news node features and labels
        all_embeddings = np.concatenate([train_labeled_emb, train_unlabeled_emb, test_emb])
        all_labels = np.concatenate([train_labeled_labels, train_unlabeled_labels, test_labels])
        
        # Convert to tensors
        news_x = torch.tensor(all_embeddings, dtype=torch.float)
        news_y = torch.tensor(all_labels, dtype=torch.long)
        
        # Add entity-aware features
        news_x = self.add_entity_features(news_x)
        
        # Build news-news edges
        if self.edge_policy == "knn_test_isolated":
            edge_index, edge_weight = self.build_knn_test_isolated_edges(
                train_labeled_emb, train_unlabeled_emb, test_emb, self.k_neighbors
            )
        else:  # Standard KNN
            edge_index, edge_weight = self.build_knn_edges(all_embeddings, self.k_neighbors)
        
        # Create masks
        num_train_labeled = len(self.train_labeled_indices)
        num_train_unlabeled = len(self.train_unlabeled_indices)
        num_test = len(self.test_indices)
        total_nodes = num_train_labeled + num_train_unlabeled + num_test
        
        train_labeled_mask = torch.zeros(total_nodes, dtype=torch.bool)
        train_labeled_mask[:num_train_labeled] = True
        
        train_unlabeled_mask = torch.zeros(total_nodes, dtype=torch.bool)
        train_unlabeled_mask[num_train_labeled:num_train_labeled + num_train_unlabeled] = True
        
        test_mask = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask[num_train_labeled + num_train_unlabeled:] = True
        
        # Create heterogeneous graph
        graph = HeteroData()
        
        # Add news nodes
        graph['news'].x = news_x
        graph['news'].y = news_y
        graph['news'].train_labeled_mask = train_labeled_mask
        graph['news'].train_unlabeled_mask = train_unlabeled_mask
        graph['news'].test_mask = test_mask
        
        # Add news-news edges
        graph['news', 'similar_to', 'news'].edge_index = edge_index
        graph['news', 'similar_to', 'news'].edge_attr = edge_weight.unsqueeze(1)
        
        # Optionally add interaction nodes (simplified)
        if not self.no_interactions:
            # Create dummy interaction nodes for demonstration
            num_interactions = min(100, len(all_embeddings) // 2)
            interaction_dim = 64
            
            graph['interaction'].x = torch.randn(num_interactions, interaction_dim)
            
            # Random news-interaction edges
            news_interaction_edges = []
            for i in range(num_interactions):
                # Each interaction connects to 1-3 random news nodes
                num_connections = np.random.randint(1, 4)
                connected_news = np.random.choice(total_nodes, num_connections, replace=False)
                for news_idx in connected_news:
                    news_interaction_edges.append([news_idx, i])
            
            if news_interaction_edges:
                ni_edge_index = torch.tensor(news_interaction_edges, dtype=torch.long).t()
                graph['news', 'has_interaction', 'interaction'].edge_index = ni_edge_index
                graph['interaction', 'about', 'news'].edge_index = ni_edge_index.flip(0)
        
        print(f"Graph created:")
        print(f"- News nodes: {graph['news'].x.shape}")
        if 'interaction' in graph.node_types:
            print(f"- Interaction nodes: {graph['interaction'].x.shape}")
        print(f"- Edge types: {list(graph.edge_types)}")
        
        return graph

    def save_graph(self, graph: HeteroData, suffix: str = "") -> str:
        """Save the graph to disk."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        filename = f"less4fd_{self.dataset_name}_k{self.k_shot}_{self.embedding_type}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".pt"
        
        filepath = os.path.join(self.output_dir, filename)
        torch.save(graph, filepath)
        
        print(f"Graph saved to: {filepath}")
        return filepath


def main():
    """Main function for building LESS4FD graphs."""
    parser = ArgumentParser(description="Build simplified LESS4FD graph")
    parser.add_argument("--dataset_name", choices=["politifact", "gossipcop"], 
                       default="politifact", help="Dataset name")
    parser.add_argument("--k_shot", type=int, choices=range(3, 17), default=8,
                       help="Number of shots")
    parser.add_argument("--embedding_type", choices=["bert", "roberta", "deberta", "distilbert"], 
                       default="deberta", help="Embedding type")
    parser.add_argument("--edge_policy", choices=["knn", "knn_test_isolated"],
                       default="knn_test_isolated", help="Edge construction policy")
    parser.add_argument("--k_neighbors", type=int, default=5, help="Number of KNN neighbors")
    parser.add_argument("--enable_entities", action="store_true", default=True,
                       help="Enable entity-aware features")
    parser.add_argument("--no_interactions", action="store_true", 
                       help="Build graph without interaction nodes")
    parser.add_argument("--output_dir", default="graphs_less4fd", 
                       help="Output directory for graphs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Build graph
    builder = SimpleLESS4FDGraphBuilder(
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        embedding_type=args.embedding_type,
        edge_policy=args.edge_policy,
        k_neighbors=args.k_neighbors,
        enable_entities=args.enable_entities,
        no_interactions=args.no_interactions,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print(f"Building LESS4FD graph for {args.dataset_name}, k_shot={args.k_shot}")
    graph = builder.build_hetero_graph()
    
    # Save graph
    filepath = builder.save_graph(graph)
    
    print(f"\nLESS4FD graph built successfully!")
    print(f"Graph saved to: {filepath}")


if __name__ == "__main__":
    main()