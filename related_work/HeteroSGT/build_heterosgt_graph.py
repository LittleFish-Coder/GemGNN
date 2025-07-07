"""
HeteroSGT Graph Builder.

A simplified implementation that builds heterogeneous graphs for subgraph-based 
fake news detection using structural graph transformers.
Based on the HeteroSGT paper: Heterogeneous Structural Graph Transformer.
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


class HeteroSGTGraphBuilder:
    """
    HeteroSGT graph builder for subgraph-based fake news detection.
    
    This implementation:
    1. Loads data from Hugging Face datasets (same as main repository)
    2. Builds heterogeneous graphs with news and interaction nodes
    3. Extracts news-centered subgraphs for classification
    4. Computes random-walk distances for attention bias
    5. Uses k-shot sampling for few-shot learning scenarios
    """
    
    def __init__(
        self,
        dataset_name: str,
        k_shot: int,
        embedding_type: str = "deberta",
        edge_policy: str = "knn_test_isolated",
        k_neighbors: int = 5,
        subgraph_size: int = 20,
        max_walk_length: int = 4,
        partial_unlabeled: bool = True,
        sample_unlabeled_factor: int = 5,
        output_dir: str = "graphs_heterosgt",
        dataset_cache_dir: str = "dataset",
        seed: int = 42,
        no_interactions: bool = False
    ):
        """
        Initialize the HeteroSGT graph builder.
        
        Args:
            dataset_name: Dataset name (politifact, gossipcop)
            k_shot: Number of labeled samples per class (3-16)
            embedding_type: Type of embeddings (bert, roberta, deberta, etc.)
            edge_policy: Edge construction policy (knn, knn_test_isolated)
            k_neighbors: Number of neighbors for KNN edges
            subgraph_size: Maximum size of news-centered subgraphs
            max_walk_length: Maximum length for random walks
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
        self.subgraph_size = subgraph_size
        self.max_walk_length = max_walk_length
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
        
        # Graph components
        self.graph_nx = None
        self.distance_matrix = None

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

    def compute_random_walk_distances(self, edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
        """
        Compute random walk distances between all node pairs.
        
        Args:
            edge_index: Graph edge indices
            num_nodes: Total number of nodes
            
        Returns:
            Distance matrix [num_nodes, num_nodes]
        """
        print(f"Computing random walk distances (max length: {self.max_walk_length})...")
        
        # Create NetworkX graph for efficient random walk computation
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # Add edges
        edges = edge_index.t().numpy()
        for src, dst in edges:
            G.add_edge(src, dst)
        
        # Initialize distance matrix with infinity
        distances = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(distances, 0)  # Distance to self is 0
        
        # Compute shortest path distances (approximation of random walk distances)
        try:
            shortest_paths = dict(nx.all_pairs_shortest_path_length(G, cutoff=self.max_walk_length))
            for src in shortest_paths:
                for dst, dist in shortest_paths[src].items():
                    distances[src, dst] = dist
        except Exception as e:
            print(f"Warning: Error computing distances: {e}")
            # Fallback: use hop distance for connected components
            for component in nx.connected_components(G):
                if len(component) > 1:
                    component_nodes = list(component)
                    subgraph = G.subgraph(component_nodes)
                    for src in component_nodes:
                        for dst in component_nodes:
                            if src != dst:
                                try:
                                    dist = nx.shortest_path_length(subgraph, src, dst)
                                    if dist <= self.max_walk_length:
                                        distances[src, dst] = dist
                                except nx.NetworkXNoPath:
                                    pass
        
        # Clip distances to max walk length
        distances = np.clip(distances, 0, self.max_walk_length)
        
        print(f"Distance matrix computed: {distances.shape}, "
              f"avg distance: {distances[distances < np.inf].mean():.2f}")
        
        return distances

    def extract_subgraphs(self, edge_index: torch.Tensor, num_nodes: int) -> Dict[int, List[int]]:
        """
        Extract news-centered subgraphs for each news node.
        
        Args:
            edge_index: Graph edge indices
            num_nodes: Total number of nodes
            
        Returns:
            Dictionary mapping node_id to subgraph node list
        """
        print(f"Extracting subgraphs (size: {self.subgraph_size})...")
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # Add edges
        edges = edge_index.t().numpy()
        for src, dst in edges:
            G.add_edge(src, dst)
        
        subgraphs = {}
        
        # For each node, extract a subgraph using BFS
        for node_id in tqdm(range(num_nodes), desc="Extracting subgraphs", ncols=100):
            # BFS to find k-hop neighbors
            visited = set()
            queue = [(node_id, 0)]  # (node, distance)
            subgraph_nodes = []
            
            while queue and len(subgraph_nodes) < self.subgraph_size:
                current_node, dist = queue.pop(0)
                
                if current_node in visited:
                    continue
                    
                visited.add(current_node)
                subgraph_nodes.append(current_node)
                
                # Add neighbors if within distance limit
                if dist < 2:  # 2-hop subgraphs
                    for neighbor in G.neighbors(current_node):
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1))
            
            subgraphs[node_id] = subgraph_nodes
        
        print(f"Extracted {len(subgraphs)} subgraphs, "
              f"avg size: {np.mean([len(sg) for sg in subgraphs.values()]):.1f}")
        
        return subgraphs

    def build_hetero_graph(self) -> HeteroData:
        """Build the heterogeneous graph for HeteroSGT."""
        print("Building HeteroSGT heterogeneous graph...")
        
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
        
        # Build news-news edges
        if self.edge_policy == "knn_test_isolated":
            edge_index, edge_weight = self.build_knn_test_isolated_edges(
                train_labeled_emb, train_unlabeled_emb, test_emb, self.k_neighbors
            )
        else:  # Standard KNN
            edge_index, edge_weight = self.build_knn_edges(all_embeddings, self.k_neighbors)
        
        # Compute random walk distances
        num_news_nodes = len(all_embeddings)
        distance_matrix = self.compute_random_walk_distances(edge_index, num_news_nodes)
        
        # Extract subgraphs
        subgraphs = self.extract_subgraphs(edge_index, num_news_nodes)
        
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
        
        # Store HeteroSGT-specific data
        graph.distance_matrix = torch.tensor(distance_matrix, dtype=torch.float)
        graph.subgraphs = subgraphs
        graph.max_walk_length = self.max_walk_length
        graph.subgraph_size = self.subgraph_size
        
        # Optionally add interaction nodes (simplified)
        if not self.no_interactions:
            # Create dummy interaction nodes for demonstration
            num_interactions = min(50, len(all_embeddings) // 4)
            interaction_dim = 64
            
            graph['interaction'].x = torch.randn(num_interactions, interaction_dim)
            
            # Random news-interaction edges
            news_interaction_edges = []
            for i in range(num_interactions):
                # Each interaction connects to 1-2 random news nodes
                num_connections = np.random.randint(1, 3)
                connected_news = np.random.choice(total_nodes, num_connections, replace=False)
                for news_idx in connected_news:
                    news_interaction_edges.append([news_idx, i])
            
            if news_interaction_edges:
                ni_edge_index = torch.tensor(news_interaction_edges, dtype=torch.long).t()
                graph['news', 'has_interaction', 'interaction'].edge_index = ni_edge_index
                graph['interaction', 'about', 'news'].edge_index = ni_edge_index.flip(0)
        
        print(f"HeteroSGT graph created:")
        print(f"- News nodes: {graph['news'].x.shape}")
        if 'interaction' in graph.node_types:
            print(f"- Interaction nodes: {graph['interaction'].x.shape}")
        print(f"- Edge types: {list(graph.edge_types)}")
        print(f"- Distance matrix: {graph.distance_matrix.shape}")
        print(f"- Subgraphs: {len(graph.subgraphs)}")
        
        return graph

    def save_graph(self, graph: HeteroData, suffix: str = "") -> str:
        """Save the graph to disk."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        filename = f"heterosgt_{self.dataset_name}_k{self.k_shot}_{self.embedding_type}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".pt"
        
        filepath = os.path.join(self.output_dir, filename)
        torch.save(graph, filepath, _use_new_zipfile_serialization=False)
        
        print(f"HeteroSGT graph saved to: {filepath}")
        return filepath


def main():
    """Main function for building HeteroSGT graphs."""
    parser = ArgumentParser(description="Build HeteroSGT graph")
    parser.add_argument("--dataset_name", choices=["politifact", "gossipcop"], 
                       default="politifact", help="Dataset name")
    parser.add_argument("--k_shot", type=int, choices=range(3, 17), default=8,
                       help="Number of shots")
    parser.add_argument("--embedding_type", choices=["bert", "roberta", "deberta", "distilbert"], 
                       default="deberta", help="Embedding type")
    parser.add_argument("--edge_policy", choices=["knn", "knn_test_isolated"],
                       default="knn_test_isolated", help="Edge construction policy")
    parser.add_argument("--k_neighbors", type=int, default=5, help="Number of KNN neighbors")
    parser.add_argument("--subgraph_size", type=int, default=20, 
                       help="Maximum subgraph size")
    parser.add_argument("--max_walk_length", type=int, default=4,
                       help="Maximum random walk length")
    parser.add_argument("--no_interactions", action="store_true", 
                       help="Build graph without interaction nodes")
    parser.add_argument("--output_dir", default="graphs_heterosgt", 
                       help="Output directory for graphs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Build graph
    builder = HeteroSGTGraphBuilder(
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        embedding_type=args.embedding_type,
        edge_policy=args.edge_policy,
        k_neighbors=args.k_neighbors,
        subgraph_size=args.subgraph_size,
        max_walk_length=args.max_walk_length,
        no_interactions=args.no_interactions,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print(f"Building HeteroSGT graph for {args.dataset_name}, k_shot={args.k_shot}")
    graph = builder.build_hetero_graph()
    
    # Save graph
    filepath = builder.save_graph(graph)
    
    print(f"\nHeteroSGT graph built successfully!")
    print(f"Graph saved to: {filepath}")


if __name__ == "__main__":
    main()