import os
import gc
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Tuple, Optional, List, Union, Any
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm
from argparse import ArgumentParser
from matplotlib.patches import Patch
from utils.sample_k_shot import sample_k_shot

# constants
GRAPH_DIR = "graphs"
PLOT_DIR = "plots"
DEFAULT_SEED = 42  # use the same SEED as finetune_lm.py, prompt_hf_llm.py
DEFAULT_K_NEIGHBORS = 5
DEFAULT_EDGE_POLICY = "dynamic_threshold"
DEFAULT_EMBEDDING_TYPE = "roberta"
DEFAULT_THRESHOLD_FACTOR = 1.1
DEFAULT_ALPHA = 0.1


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GraphBuilder:
    """
    Builds graph datasets for few-shot fake news detection.

    This class handles the construction of graph representations from fake news datasets,
    supporting both transductive (test nodes present during training) and few-shot learning scenarios where only a small subset of nodes are labeled.
    """

    def __init__(
        self,
        dataset_name: str,
        k_shot: int,
        edge_policy: str = DEFAULT_EDGE_POLICY,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        threshold_factor: float = 1.0,
        alpha: float = 0.5,
        output_dir: str = GRAPH_DIR,
        plot: bool = False,
        seed: int = DEFAULT_SEED,
        embedding_type: str = DEFAULT_EMBEDDING_TYPE,
        device: str = None,
    ):
        """Initialize the GraphBuilder with configuration parameters."""
        self.dataset_name = dataset_name.lower()
        self.k_shot = k_shot
        self.edge_policy = edge_policy
        self.k_neighbors = k_neighbors
        self.threshold_factor = threshold_factor
        self.alpha = alpha
        self.plot = plot
        self.seed = seed
        self.embedding_type = embedding_type.lower()

        # Setup directory paths (keep dataset name only for directories)
        self.output_dir = os.path.join(output_dir, self.dataset_name)
        self.plot_dir = os.path.join(PLOT_DIR, self.dataset_name)

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        if self.plot:
            os.makedirs(self.plot_dir, exist_ok=True)

        # Initialize components
        self.dataset = None
        self.graph_data = None
        self.graph_metrics = {}
        self.selected_indices = None

        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_dataset(self) -> None:
        """Load dataset from HuggingFace and prepare it for graph construction."""
        print(f"Loading dataset '{self.dataset_name}' with '{self.embedding_type}' embeddings...")

        # Format dataset name for HuggingFace
        hf_dataset_name = f"LittleFish-Coder/Fake_News_{self.dataset_name}"

        # Load dataset
        dataset = load_dataset(hf_dataset_name, download_mode="reuse_cache_if_exists", cache_dir="dataset")

        # Get train and test datasets (use full datasets)
        train_dataset, test_dataset = dataset["train"], dataset["test"]

        # Store dataset
        self.dataset = {"train": train_dataset, "test": test_dataset}

        # Store sizes
        self.train_size = len(train_dataset)
        self.test_size = len(test_dataset)

        # Calculate labeled size from k_shot
        unique_labels = set(train_dataset["label"])
        self.labeled_size = self.k_shot * len(unique_labels)
        print(f"\nWith {len(unique_labels)} classes and k_shot={self.k_shot}, labeled_size={self.labeled_size}")

        # Show dataset statistics
        self._show_dataset_stats()

    def _show_dataset_stats(self) -> None:
        """Display statistics about the dataset."""
        train_data = self.dataset["train"]
        test_data = self.dataset["test"]

        # Count labels
        train_labels = train_data["label"]
        test_labels = test_data["label"]

        train_label_counts = {
            label: train_labels.count(label) for label in set(train_labels)
        }

        test_label_counts = {
            label: test_labels.count(label) for label in set(test_labels)
        }

        print("\nDataset Statistics:")
        print(f"  Train set: {len(train_data)} samples")
        for label, count in train_label_counts.items():
            print(f"    - Class {label}: {count} samples ({count/len(train_data)*100:.1f})")

        print(f"  Test set: {len(test_data)} samples")
        for label, count in test_label_counts.items():
            print(f"    - Class {label}: {count} samples ({count/len(test_data)*100:.1f})")

        print(f"  Few-shot labeled set: {self.k_shot} samples per class ({self.labeled_size} total)")
        print("")

    def build_graph(self) -> Data:
        """Build a graph including both nodes and edges."""
        # First build nodes
        self.build_empty_graph()

        # Then add edges
        print(f"Building graph edges using {self.edge_policy} policy...")

        embeddings = self.graph_data.x.numpy()

        if self.edge_policy == "knn":
            edges, edge_attr = self._build_knn_edges(embeddings, self.k_neighbors)
        elif self.edge_policy == "mutual_knn":
            edges, edge_attr = self._build_mutual_knn_edges(embeddings, self.k_neighbors)
        elif self.edge_policy == "threshold":
            edges, edge_attr = self._build_threshold_edges(embeddings, self.threshold_factor)
        elif self.edge_policy == "dynamic_threshold":
            edges, edge_attr = self._build_dynamic_threshold_edges(embeddings, self.alpha)
        else:
            raise ValueError(f"Unknown edge policy: {self.edge_policy}")

        # Update graph data
        self.graph_data.edge_index = edges
        self.graph_data.edge_attr = edge_attr
        self.graph_data.num_edges = edges.shape[1]

        print(f"Graph edges built: {edges.shape[1]} edges created")

        # Analyze the graph
        self._analyze_graph()

        return self.graph_data

    def build_empty_graph(self) -> Data:
        """Build a graph with nodes but without edges."""
        print(f"Building graph nodes using {self.embedding_type} embeddings (without edges)...")

        train_data = self.dataset["train"]
        test_data = self.dataset["test"]

        # Determine which embedding field to use based on embedding_type
        embedding_field = f"{self.embedding_type}_embeddings"

        print(f"Using embeddings from field: '{embedding_field}'")

        # Get embeddings and labels
        train_embeddings = np.array(train_data[embedding_field])
        train_labels = np.array(train_data["label"])
        test_embeddings = np.array(test_data[embedding_field])
        test_labels = np.array(test_data["label"])

        # Calculate number of nodes
        num_nodes = len(train_embeddings) + len(test_embeddings)

        # Merge embeddings and labels
        x = torch.tensor(
            np.concatenate([train_embeddings, test_embeddings]), dtype=torch.float
        )
        y = torch.tensor(np.concatenate([train_labels, test_labels]), dtype=torch.long)

        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Set masks
        train_mask[: len(train_embeddings)] = True
        test_mask[len(train_embeddings) :] = True

        # Use the EXACT SAME sampling logic as finetune_lm.py
        print(f"Sampling k_shot labeled_node with k={self.k_shot} and seed={self.seed}")
        indices, _ = sample_k_shot(train_data, self.k_shot, self.seed)
        self.selected_indices = indices

        # Set labeled mask
        labeled_mask_indices = np.zeros(num_nodes, dtype=bool)
        labeled_mask_indices[indices] = True
        labeled_mask = torch.tensor(labeled_mask_indices, dtype=torch.bool)

        # Create empty edge index (will be populated later)
        edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Create graph data object
        graph_data = Data(
            x=x,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            labeled_mask=labeled_mask,
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_features=x.shape[1],
        )

        # Store graph data
        self.graph_data = graph_data

        # Display graph structure
        print(f"Graph nodes built with {num_nodes} nodes and {x.shape[1]} features")
        print(f"Labeled nodes: {labeled_mask.sum().item()} (for few-shot learning)")
        print()

        return graph_data

    def _build_knn_edges(
        self, embeddings: np.ndarray, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using k-nearest neighbors approach."""
        print(f"Building KNN graph with k={k}...")

        # Compute cosine distances
        distances = pairwise_distances(embeddings, metric="cosine")

        # For each node, find k nearest neighbors
        rows, cols, data = [], [], []
        for i in tqdm(range(len(embeddings)), desc=f"Finding {k} nearest neighbors"):
            # Skip self (distance=0)
            dist_i = distances[i].copy()
            dist_i[i] = float("inf")    # self distance is set to infinity

            # Get indices of k smallest distances
            indices = np.argpartition(dist_i, k)[:k]

            # Add edges
            for j in indices:
                rows.append(i)
                cols.append(j)
                # Convert distance to similarity
                data.append(1 - distances[i, j])

        # Create PyTorch tensors
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)

        return edge_index, edge_attr

    def _build_mutual_knn_edges(
        self, embeddings: np.ndarray, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using mutual k-nearest neighbors approach."""
        print(f"Building mutual KNN graph with k={k}...")

        # Compute cosine distances
        distances = pairwise_distances(embeddings, metric="cosine")

        # For each node, find k nearest neighbors
        rows, cols, data = [], [], []
        for i in tqdm(range(len(embeddings)), desc=f"Finding {k} mutual neighbors"):
            # Skip self (distance=0)
            dist_i = distances[i].copy()
            dist_i[i] = float("inf")    # self distance is set to infinity

            # Get indices of k smallest distances
            indices = np.argpartition(dist_i, k)[:k]

            # Check mutuality
            for j in indices:
                if i in np.argpartition(distances[j], k)[:k]:
                    rows.append(i)
                    cols.append(j)
                    # Convert distance to similarity
                    data.append(1 - distances[i, j])

        # Create PyTorch tensors
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)

        return edge_index, edge_attr

    def _build_threshold_edges(
        self, embeddings: np.ndarray, threshold_factor: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges using a threshold-based approach, optimized for robustness and accuracy.

        Args:
            embeddings: Node feature embedding matrix
            threshold_factor: Factor to adjust the threshold

        Returns:
            edge_index: Edge indices (2 x num_edges)
            edge_attr: Edge attributes (num_edges x 1)
        """
        print(f"Building threshold graph with threshold_factor={threshold_factor}...")

        # Calculate effective embeddings (check and handle NaN and infinite values)
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            print("Warning: Embeddings contain NaN or infinite values, fixing...")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize embeddings, prevent division by zero
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Ensure no zero-length vectors
        valid_norms = np.maximum(norms, 1e-10)
        normalized_embeddings = embeddings / valid_norms

        # Initialize edge storage
        rows, cols, data = [], [], []
        num_nodes = len(embeddings)

        # Process in batches for efficiency
        batch_size = min(500, num_nodes)

        for i in tqdm(range(0, num_nodes, batch_size), desc="Computing threshold edges"):
            batch_end = min(i + batch_size, num_nodes)
            batch = normalized_embeddings[i:batch_end]

            # Explicitly compute cosine similarity
            batch_similarities = np.dot(batch, normalized_embeddings.T)

            # Determine threshold and build edges for each node in the batch
            for j in range(batch_end - i):
                node_idx = i + j
                node_similarities = batch_similarities[j].copy()

                # Exclude self-connections
                node_similarities[node_idx] = -1.0

                # Calculate mean similarity for this node (excluding negative values)
                positive_similarities = node_similarities[node_similarities > 0]
                if len(positive_similarities) > 0:
                    mean_similarity = np.mean(positive_similarities)
                    # Apply threshold factor
                    node_threshold = mean_similarity * threshold_factor

                    # Find nodes above threshold
                    above_threshold = np.where(node_similarities > node_threshold)[0]

                    # Add edges
                    for target in above_threshold:
                        if target != node_idx:  # Additional check for self-connections
                            rows.append(node_idx)
                            cols.append(target)
                            data.append(node_similarities[target])

        # Error checking
        if len(rows) == 0:
            print("Warning: No edges were created! Try reducing the threshold factor.")
            # To prevent completely disconnected graphs, add edges to most similar nodes
            for i in range(num_nodes):
                similarities = np.dot(normalized_embeddings[i], normalized_embeddings.T)
                similarities[i] = -1.0
                # Connect to most similar node
                top_idx = np.argmax(similarities)
                rows.append(i)
                cols.append(top_idx)
                data.append(similarities[top_idx])

        # Create PyTorch tensors
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)

        print(
            f"Created {edge_index.shape[1]} edges, average degree {edge_index.shape[1]/num_nodes:.2f}"
        )

        return edge_index, edge_attr

    def _build_dynamic_threshold_edges(
        self, embeddings: np.ndarray, alpha: float = 0.5, max_degree: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges using a dynamic threshold approach with global statistics and power-law degree distribution support.

        Args:
            embeddings: Node feature embedding matrix
            alpha: Coefficient controlling threshold strictness (default: 0.5)
            max_degree: Maximum degree for each node (default: None, no limit)

        Returns:
            edge_index: Edge indices (2 x num_edges)
            edge_attr: Edge attributes (num_edges x 1)
        """
        print(
            f"Building dynamic threshold graph with alpha={alpha}"
            + (f", max_degree={max_degree}" if max_degree else "")
            + "..."
        )

        # Handle invalid values
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_norms = np.maximum(norms, 1e-10)
        normalized_embeddings = embeddings / valid_norms

        num_nodes = len(embeddings)

        # 1. Calculate global similarity statistics
        print("Computing global similarity statistics...")
        all_similarities = []
        batch_size = min(500, num_nodes)

        for i in tqdm(range(0, num_nodes, batch_size), desc="Sampling similarities"):
            batch_end = min(i + batch_size, num_nodes)

            # To avoid memory issues, sample only a portion of node pairs for large graphs
            if num_nodes > 1000:
                sample_size = min(1000, num_nodes)
                sample_indices = np.random.choice(
                    num_nodes, size=sample_size, replace=False
                )

                for j in range(i, batch_end):
                    similarities = np.dot(
                        normalized_embeddings[j],
                        normalized_embeddings[sample_indices].T,
                    )
                    # Exclude self-similarity
                    if j in sample_indices:
                        idx = np.where(sample_indices == j)[0][0]
                        similarities[idx] = 0
                    all_similarities.extend(similarities)
            else:
                # For small graphs, compute all similarities
                batch = normalized_embeddings[i:batch_end]
                batch_similarities = np.dot(batch, normalized_embeddings.T)

                for j in range(batch_similarities.shape[0]):
                    node_idx = i + j
                    sims = batch_similarities[j]
                    sims[node_idx] = 0  # Exclude self
                    all_similarities.extend(sims)

        # Filter out -1 (possible due to numerical errors)
        all_similarities = np.array([s for s in all_similarities if s > -0.99])

        # Calculate statistics
        sim_mean = np.mean(all_similarities)
        sim_std = np.std(all_similarities)

        # 2. Set dynamic threshold using + instead of - for stricter connections
        dynamic_threshold = sim_mean + alpha * sim_std

        print(f"Similarity statistics: mean={sim_mean:.4f}, std={sim_std:.4f}")
        print(f"Dynamic threshold: {dynamic_threshold:.4f}")

        # 3. Create edges - using power-law degree distribution
        rows, cols, data = [], [], []

        # If power-law distribution is desired, assign different target degrees to each node
        if max_degree is not None:
            # Generate target degrees following power-law distribution
            exponent = 2.1  # Typical power-law exponent
            target_degrees = np.random.power(exponent, size=num_nodes) * max_degree
            target_degrees = np.maximum(target_degrees, 2).astype(
                int
            )  # Minimum degree of 2
            current_degrees = np.zeros(num_nodes, dtype=int)
        else:
            target_degrees = None
            current_degrees = None

        # Build edges
        for i in tqdm(range(num_nodes), desc="Building dynamic threshold edges"):
            # Compute similarities with all nodes
            similarities = np.dot(normalized_embeddings[i], normalized_embeddings.T)
            similarities[i] = -1.0  # Exclude self

            # Find nodes above threshold
            above_threshold = np.where(similarities > dynamic_threshold)[0]

            # Apply degree limit if enabled
            if target_degrees is not None:
                remaining = target_degrees[i] - current_degrees[i]

                if remaining <= 0:
                    continue  # Node has reached target degree

                if len(above_threshold) > remaining:
                    # Sort by similarity, take top remaining
                    sorted_indices = above_threshold[
                        np.argsort(similarities[above_threshold])[-remaining:]
                    ]
                else:
                    sorted_indices = above_threshold
            else:
                sorted_indices = above_threshold

            # Add edges
            for target in sorted_indices:
                # If target node has degree limit, check if it has capacity
                if (
                    target_degrees is not None
                    and current_degrees[target] >= target_degrees[target]
                ):
                    continue

                rows.append(i)
                cols.append(target)
                data.append(similarities[target])

                # Update degree counts
                if target_degrees is not None:
                    current_degrees[i] += 1
                    current_degrees[target] += 1

        # Error checking - ensure edges exist
        if len(rows) == 0:
            print("Warning: Threshold too high, no edges created. Adding basic connectivity...")
            # Add a minimum spanning tree to ensure graph connectivity
            for i in range(1, num_nodes):
                similarities = np.dot(normalized_embeddings[i], normalized_embeddings[:i].T)
                most_similar = np.argmax(similarities)

                rows.append(i)
                cols.append(most_similar)
                data.append(similarities[most_similar])

        # Create PyTorch tensors
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)

        print(f"Created {edge_index.shape[1]} edges, average degree {edge_index.shape[1]/num_nodes:.2f}")

        return edge_index, edge_attr

    def _analyze_graph(self) -> None:
        """Analyze the graph and compute comprehensive metrics."""

        # Basic metrics
        self.graph_metrics = {
            "num_nodes": self.graph_data.num_nodes,
            "num_edges": self.graph_data.edge_index.shape[1],
            "num_train_nodes": self.graph_data.train_mask.sum().item(),
            "num_test_nodes": self.graph_data.test_mask.sum().item(),
            "num_labeled_nodes": self.graph_data.labeled_mask.sum().item(),
            "avg_degree": self.graph_data.edge_index.shape[1] / self.graph_data.num_nodes,
        }

        # Edge type analysis
        edge_types = {"train-train": 0, "train-test": 0, "test-test": 0}
        # Calculate the degree of each node
        degrees = torch.zeros(self.graph_data.num_nodes, dtype=torch.long)

        for edge in tqdm(self.graph_data.edge_index.t(), desc="Analyzing edge types"):
            source, target = edge
            if (
                self.graph_data.train_mask[source]
                and self.graph_data.train_mask[target]
            ):
                edge_types["train-train"] += 1
            elif (
                self.graph_data.train_mask[source] and self.graph_data.test_mask[target]
            ) or (
                self.graph_data.test_mask[source] and self.graph_data.train_mask[target]
            ):
                edge_types["train-test"] += 1
            elif (
                self.graph_data.test_mask[source] and self.graph_data.test_mask[target]
            ):
                edge_types["test-test"] += 1
            degrees[source] += 1
            degrees[target] += 1

        self.graph_metrics["edge_types"] = edge_types

        # Find nodes with degree 0
        isolated_nodes = (degrees == 0).nonzero(as_tuple=True)[0].tolist()
        self.graph_metrics["isolated_nodes"] = len(isolated_nodes)

        # Calculate connectivity patterns between fake/real news
        fake_to_fake = 0
        real_to_real = 0
        fake_to_real = 0

        for edge in tqdm(
            self.graph_data.edge_index.t(), desc="Analyzing class connectivity"
        ):
            source, target = edge
            source_label = self.graph_data.y[source].item()
            target_label = self.graph_data.y[target].item()

            if source_label == 1 and target_label == 1:  # fake-to-fake
                fake_to_fake += 1
            elif source_label == 0 and target_label == 0:  # real-to-real
                real_to_real += 1
            else:  # fake-to-real or real-to-fake
                fake_to_real += 1

        self.graph_metrics["fake_to_fake"] = fake_to_fake
        self.graph_metrics["real_to_real"] = real_to_real
        self.graph_metrics["fake_to_real"] = fake_to_real

        # Calculate homophily (same class connections)
        homophilic_edges = fake_to_fake + real_to_real
        total_edges = self.graph_data.edge_index.shape[1]

        self.graph_metrics["homophilic_edges"] = homophilic_edges
        self.graph_metrics["heterophilic_edges"] = fake_to_real
        self.graph_metrics["homophily_ratio"] = homophilic_edges / total_edges

        # Degree distribution analysis
        degrees_np = degrees.numpy()
        unique_degrees, degree_counts = np.unique(degrees_np, return_counts=True)
        # Convert numpy types to native Python types
        degree_distribution = {
            int(k): int(v) for k, v in zip(unique_degrees, degree_counts)
        }

        # Calculate degree statistics
        self.graph_metrics["degree_stats"] = {
            "min_degree": int(np.min(degrees_np)),
            "max_degree": int(np.max(degrees_np)),
            "mean_degree": float(np.mean(degrees_np)),
            "median_degree": float(np.median(degrees_np)),
            "std_degree": float(np.std(degrees_np)),
            "degree_distribution": degree_distribution,
        }

        # Power-law analysis
        if len(unique_degrees) > 1:
            # Fit power-law distribution
            from scipy import stats

            # Remove zero degrees for power-law fitting
            valid_degrees = unique_degrees[unique_degrees > 0]
            valid_counts = degree_counts[unique_degrees > 0]

            # Log-transform for linear regression
            log_degrees = np.log10(valid_degrees)
            log_counts = np.log10(valid_counts)

            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_degrees, log_counts
            )

            # Calculate power-law metrics
            self.graph_metrics["power_law"] = {
                "exponent": float(-slope),  # Power-law exponent
                "r_squared": float(r_value**2),  # R-squared value
                "p_value": float(p_value),
                "std_err": float(std_err),
                "is_power_law": p_value < 0.05
                and r_value**2 > 0.7,  # Rough criteria for power-law
            }

        # Calculate clustering coefficient
        G = to_networkx(self.graph_data, to_undirected=True)
        clustering_coeffs = nx.clustering(G)
        avg_clustering = nx.average_clustering(G)

        self.graph_metrics["clustering"] = {
            "average_clustering": float(avg_clustering),
            "clustering_distribution": {
                str(k): float(v) for k, v in nx.clustering(G).items()
            },
        }

        # Calculate graph density
        density = nx.density(G)
        self.graph_metrics["density"] = float(density)

        # Calculate average path length (if graph is connected)
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            self.graph_metrics["avg_path_length"] = float(avg_path_length)

        # Calculate assortativity (degree correlation)
        assortativity = nx.degree_assortativity_coefficient(G)
        self.graph_metrics["assortativity"] = float(assortativity)

        # Print comprehensive analysis
        print("\nGraph Analysis Summary:")
        print(f"  Nodes: {self.graph_metrics['num_nodes']}")
        print(f"  Edges: {self.graph_metrics['num_edges']}")
        print(f"  Average degree: {self.graph_metrics['avg_degree']:.2f}")
        print(f"  Graph density: {self.graph_metrics['density']:.4f}")
        print(
            f"  Average clustering coefficient: {self.graph_metrics['clustering']['average_clustering']:.4f}"
        )
        print(f"  Assortativity: {self.graph_metrics['assortativity']:.4f}")

        if "avg_path_length" in self.graph_metrics:
            print(f"  Average path length: {self.graph_metrics['avg_path_length']:.2f}")

        print("\n  Degree Statistics:")
        print(f"    Min degree: {self.graph_metrics['degree_stats']['min_degree']}")
        print(f"    Max degree: {self.graph_metrics['degree_stats']['max_degree']}")
        print(
            f"    Mean degree: {self.graph_metrics['degree_stats']['mean_degree']:.2f}"
        )
        print(
            f"    Median degree: {self.graph_metrics['degree_stats']['median_degree']:.2f}"
        )
        print(f"    Degree std: {self.graph_metrics['degree_stats']['std_degree']:.2f}")

        if "power_law" in self.graph_metrics:
            print("\n  Power-Law Analysis:")
            print(f"    Exponent: {self.graph_metrics['power_law']['exponent']:.2f}")
            print(f"    R-squared: {self.graph_metrics['power_law']['r_squared']:.4f}")
            print(f"    P-value: {self.graph_metrics['power_law']['p_value']:.4f}")
            print(
                f"    Follows power-law: {'Yes' if self.graph_metrics['power_law']['is_power_law'] else 'No'}"
            )

        print("\n  Edge Types:")
        for edge_type, count in edge_types.items():
            print(f"    - {edge_type}: {count} ({count/total_edges*100:.1f}%)")

        print("\n  Class Connectivity:")
        print(f"    Fake-to-Fake: {fake_to_fake} ({fake_to_fake/total_edges*100:.1f}%)")
        print(f"    Real-to-Real: {real_to_real} ({real_to_real/total_edges*100:.1f}%)")
        print(f"    Fake-to-Real: {fake_to_real} ({fake_to_real/total_edges*100:.1f}%)")
        print(f"    Homophily ratio: {self.graph_metrics['homophily_ratio']:.4f}")

        if isolated_nodes:
            print(f"\n  Warning: {len(isolated_nodes)} isolated nodes found!")
        print("")

    def save_graph(self) -> str:
        """Save the graph and analysis results."""
        if self.graph_data is None:
            raise ValueError("Graph must be built before saving")

        # Generate graph name - include embedding type in the filename only
        # Add appropriate parameter based on edge policy
        if self.edge_policy in ["knn", "mutual_knn"]:
            edge_param = self.k_neighbors
        elif self.edge_policy == "threshold":
            edge_param = self.threshold_factor
        else:  # dynamic_threshold
            edge_param = self.alpha
        graph_name = f"{self.k_shot}_shot_{self.embedding_type}_{self.edge_policy}_{edge_param}"

        # Save graph data
        graph_path = os.path.join(self.output_dir, f"{graph_name}.pt")
        torch.save(self.graph_data, graph_path)

        # Save graph metrics if available
        if self.graph_metrics:
            # Convert any numpy numbers to Python native types
            python_metrics = json.loads(
                json.dumps(
                    self.graph_metrics,
                    default=lambda x: int(x) if isinstance(x, np.integer) else float(x),
                )
            )
            metrics_path = os.path.join(self.output_dir, f"{graph_name}_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(python_metrics, f, indent=2)
            print(f"Graph metrics saved to {metrics_path}")

        # Save selected indices for reference
        if self.selected_indices is not None:
            indices_path = os.path.join(self.output_dir, f"{graph_name}_indices.json")

            # Get label distribution of selected indices
            train_labels = self.dataset["train"]["label"]
            label_distribution = {}
            for idx in self.selected_indices:
                label = train_labels[idx]
                if label not in label_distribution:
                    label_distribution[label] = 0
                label_distribution[label] += 1

            # Convert numpy types to native Python types
            python_label_distribution = {
                int(k): int(v) for k, v in label_distribution.items()
            }

            with open(indices_path, "w") as f:
                json.dump(
                    {
                        "indices": (
                            [int(i) for i in self.selected_indices]
                            if isinstance(self.selected_indices, (np.ndarray, list))
                            else self.selected_indices
                        ),
                        "k_shot": int(self.k_shot),
                        "seed": int(self.seed),
                        "label_distribution": python_label_distribution,
                    },
                    f,
                    indent=2,
                )
            print(f"Selected indices saved to {indices_path}")

        print(f"Graph saved to {graph_path}")

        # Plot graph if requested
        if self.plot:
            try:
                self.visualize_graph(graph_name)
            except Exception as e:
                print(f"Warning: Error visualizing graph: {e}")

        return graph_path

    def visualize_graph(self, graph_name: str, max_nodes: int = 1000) -> None:
        """Visualize the graph using NetworkX."""
        if self.graph_data is None:
            raise ValueError("Graph must be built before visualization")

        # Limit visualization to avoid memory issues
        if self.graph_data.num_nodes > max_nodes:
            print(
                f"Graph is large ({self.graph_data.num_nodes} nodes). Visualizing only {max_nodes} nodes."
            )

        # Convert to NetworkX graph for visualization
        G = to_networkx(self.graph_data, to_undirected=True)

        # If graph is too large, visualize a subgraph
        if self.graph_data.num_nodes > max_nodes:
            G = G.subgraph(list(range(max_nodes)))

        # Get node attributes
        train_mask = self.graph_data.train_mask.numpy()
        test_mask = self.graph_data.test_mask.numpy()
        labeled_mask = self.graph_data.labeled_mask.numpy()

        # Set node colors based on masks
        node_colors = []
        for i in range(min(G.number_of_nodes(), self.graph_data.num_nodes)):
            if labeled_mask[i]:
                node_colors.append("green")  # Priority to labeled nodes
            elif train_mask[i]:
                node_colors.append("blue")
            elif test_mask[i]:
                node_colors.append("red")
            else:
                node_colors.append("gray")

        # Create figure
        plt.figure(figsize=(12, 10))

        # Draw graph with layout appropriate for size
        if G.number_of_nodes() > 500:
            pos = nx.random_layout(G, seed=self.seed)  # Faster for large graphs
        else:
            pos = nx.spring_layout(
                G, k=0.15, seed=self.seed
            )  # For better visualization

        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=False,
            node_color=node_colors,
            edge_color="gray",
            node_size=25 if G.number_of_nodes() <= 100 else 5,
            width=0.5,
            alpha=0.7,
        )

        # Add legend
        legend_elements = [
            Patch(facecolor="green", label="Few-shot Labeled"),
            Patch(facecolor="blue", label="Train (Unlabeled)"),
            Patch(facecolor="red", label="Test"),
        ]
        plt.legend(handles=legend_elements, loc="upper right")

        # Add title
        title = f"{self.dataset_name.capitalize()} - {self.edge_policy.upper()} ({self.embedding_type.upper()})"
        if self.graph_data.num_nodes > max_nodes:
            title += f"\n(showing {max_nodes} of {self.graph_data.num_nodes} nodes)"

        subtitle = (
            f"{self.k_shot}-shot per class, Total nodes: {self.graph_data.num_nodes}"
        )
        plt.title(f"{title}\n{subtitle}")

        # Save figure
        plot_path = os.path.join(self.plot_dir, f"{graph_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Graph visualization saved to {plot_path}")

    def run_pipeline(self) -> Data:
        """Run the complete graph building pipeline."""

        self.load_dataset()
        self.build_graph()  # build both nodes and edges
        self.save_graph()

        return self.graph_data


def parse_arguments():
    """Parse command-line arguments with helpful descriptions."""
    parser = ArgumentParser(description="Build graph for few-shot fake news detection")

    # dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="politifact",
        help="Dataset to use (e.g., politifact, gossipcop",
        choices=["politifact", "gossipcop"],
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=8,
        help="Number of samples per class for few-shot learning (e.g., 3, 8, 16)",
        choices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    )

    # graph construction arguments
    parser.add_argument(
        "--edge_policy",
        type=str,
        default=DEFAULT_EDGE_POLICY,
        help="Edge construction policy (default: knn)",
        choices=["knn", "mutual_knn", "threshold", "dynamic_threshold"],
    )
    ## for knn and mutual_knn
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=DEFAULT_K_NEIGHBORS,
        help=f"Number of neighbors for KNN or Mutual KNN (default: {DEFAULT_K_NEIGHBORS})",
    )
    ## for threshold
    parser.add_argument(
        "--threshold_factor",
        type=float,
        default=DEFAULT_THRESHOLD_FACTOR,
        help=f"Threshold factor for threshold edge construction (default: {DEFAULT_THRESHOLD_FACTOR})",
    )
    ## for dynamic_threshold
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Alpha parameter for dynamic threshold edge construction (default: {DEFAULT_ALPHA})",
    )

    # news embedding type
    parser.add_argument(
        "--embedding_type",
        type=str,
        default=DEFAULT_EMBEDDING_TYPE,
        help=f"Type of embeddings to use (default: {DEFAULT_EMBEDDING_TYPE})",
        choices=["bert", "roberta", "combined"],
    )

    # output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=GRAPH_DIR,
        help=f"Directory to save graphs (default: {GRAPH_DIR})",
    )
    ## plot
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable graph visualization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to run the graph building pipeline."""
    # parse arguments
    args = parse_arguments()

    # set seed for reproducibility
    set_seed(args.seed)

    # clean up CUDA memory
    torch.cuda.empty_cache()
    gc.collect()

    # display arguments and hardware info
    print()
    print("=" * 60)
    print("Fake News Detection - Graph Building Pipeline")
    print("=" * 60)
    print(f"Dataset:          {args.dataset_name}")
    print(f"Embedding type:   {args.embedding_type}")
    print(f"Few-shot k:       {args.k_shot} per class")
    print(f"Edge policy:      {args.edge_policy}")

    if args.edge_policy in ["knn", "mutual_knn"]:
        print(f"K neighbors:      {args.k_neighbors}")
    elif args.edge_policy == "threshold":
        print(f"Threshold factor: {args.threshold_factor}")
    elif args.edge_policy == "dynamic_threshold":
        print(f"Alpha:            {args.alpha}")

    print(f"Output directory: {args.output_dir}")
    print(f"Plot:             {args.plot}")
    print(f"Seed:             {args.seed}")
    print(f"Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")

    # create graph builder - explicitly pass each parameter
    builder = GraphBuilder(
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        edge_policy=args.edge_policy,
        k_neighbors=args.k_neighbors,
        threshold_factor=args.threshold_factor,
        alpha=args.alpha,
        output_dir=args.output_dir,
        plot=args.plot,
        seed=args.seed,
        embedding_type=args.embedding_type,
    )

    # run pipeline
    graph_data = builder.run_pipeline()

    # display final results
    print("\n" + "=" * 60)
    print("Graph Building Complete")
    print("=" * 60)
    print(f"Embedding type:   {args.embedding_type}")
    print(f"Nodes:            {graph_data.num_nodes}")
    print(f"Features:         {graph_data.num_features}")
    print(f"Edges:            {graph_data.edge_index.shape[1]}")
    print(f"Train nodes:      {graph_data.train_mask.sum().item()}")
    print(f"Test nodes:       {graph_data.test_mask.sum().item()}")
    print(f"Few-shot labeled: {graph_data.labeled_mask.sum().item()}")

    print("\nNext Steps:")
    print("  1. Train a GNN model: python train_graph.py --graph <graph_path>")
    print(
        "  2. Compare with language models: python finetune_lm.py --dataset_name <dataset> --k_shot <k>"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
