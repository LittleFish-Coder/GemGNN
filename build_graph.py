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
from sklearn.metrics.pairwise import cosine_similarity

# constants
GRAPH_DIR = "graphs"
PLOT_DIR = "plots"
DEFAULT_SEED = 42  # use the same SEED as finetune_lm.py, prompt_hf_llm.py
DEFAULT_EMBEDDING_TYPE = "roberta"
DEFAULT_EDGE_POLICY = "global_threshold"
DEFAULT_K_NEIGHBORS = 5
DEFAULT_LOCAL_THRESHOLD_FACTOR = 1.0
DEFAULT_ALPHA = 0.1
DEFAULT_QUANTILE_P = 95.0


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
        k_top_sim: int = 10,
        local_threshold_factor: float = DEFAULT_LOCAL_THRESHOLD_FACTOR,
        alpha: float = DEFAULT_ALPHA,
        quantile_p: float = DEFAULT_QUANTILE_P,
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
        self.k_top_sim = k_top_sim
        self.local_threshold_factor = local_threshold_factor
        self.alpha = alpha
        self.quantile_p = quantile_p
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
        print(
            f"Loading dataset '{self.dataset_name}' with '{self.embedding_type}' embeddings..."
        )

        # Format dataset name for HuggingFace
        hf_dataset_name = f"LittleFish-Coder/Fake_News_{self.dataset_name}"

        # Load dataset
        dataset = load_dataset(
            hf_dataset_name, download_mode="reuse_cache_if_exists", cache_dir="dataset"
        )

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
        print(
            f"\nWith {len(unique_labels)} classes and k_shot={self.k_shot}, labeled_size={self.labeled_size}"
        )

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
            print(
                f"    - Class {label}: {count} samples ({count/len(train_data)*100:.1f})"
            )

        print(f"  Test set: {len(test_data)} samples")
        for label, count in test_label_counts.items():
            print(
                f"    - Class {label}: {count} samples ({count/len(test_data)*100:.1f})"
            )

        print(
            f"  Few-shot labeled set: {self.k_shot} samples per class ({self.labeled_size} total)"
        )
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
            edges, edge_attr = self._build_mutual_knn_edges(
                embeddings, self.k_neighbors
            )
        elif self.edge_policy == "local_threshold":
            edges, edge_attr = self._build_local_threshold_edges(
                embeddings, self.local_threshold_factor
            )
        elif self.edge_policy == "global_threshold":
            edges, edge_attr = self._build_global_threshold_edges(
                embeddings, self.alpha
            )
        elif self.edge_policy == "quantile":
            edges, edge_attr = self._build_quantile_edges(embeddings, self.quantile_p)
        elif self.edge_policy == "topk_mean":
            edges, edge_attr = self._build_topk_mean_edges(
                embeddings, self.k_top_sim, self.local_threshold_factor
            )
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
        print(
            f"Building graph nodes using {self.embedding_type} embeddings (without edges)..."
        )

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
            dist_i[i] = float("inf")  # self distance is set to infinity

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
            dist_i[i] = float("inf")  # self distance is set to infinity

            # Get indices of k smallest distances
            indices = np.argpartition(dist_i, k)[:k]

            # Check mutuality
            for j in indices:
                # Check if i is among j's k nearest neighbors (excluding j itself)
                dist_j = distances[j].copy()
                dist_j[j] = float("inf")
                if i in np.argpartition(dist_j, k)[:k]:
                    rows.append(i)
                    cols.append(j)
                    # Convert distance to similarity
                    data.append(1 - distances[i, j])  # Store similarity

        # Create PyTorch tensors
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)

        return edge_index, edge_attr

    def _build_local_threshold_edges(  # New function name
        self,
        embeddings: np.ndarray,
        local_threshold_factor: float,  # New parameter name
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges using a LOCAL threshold-based approach (per-node threshold).

        Args:
            embeddings: Node feature embedding matrix
            local_threshold_factor: Factor to adjust the threshold relative to each node's mean positive similarity.

        Returns:
            edge_index: Edge indices (2 x num_edges)
            edge_attr: Edge attributes (num_edges x 1)
        """
        print(
            f"Building LOCAL threshold graph with local_threshold_factor={local_threshold_factor}..."
        )

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

        # --- Compute similarities (more efficient way) ---
        # Precompute full similarity matrix if memory allows, else batch dot product
        try:
            # Try full matrix calculation
            print("Calculating full similarity matrix...")
            similarities_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            np.fill_diagonal(
                similarities_matrix, -1.0
            )  # Exclude self-similarity efficiently
            use_full_matrix = True
        except MemoryError:
            print("Warning: Full similarity matrix too large. Processing in batches.")
            similarities_matrix = None  # Placeholder
            use_full_matrix = False

        print("Calculating local thresholds and building edges...")
        for i in tqdm(range(num_nodes), desc="Computing local threshold edges"):
            if use_full_matrix:
                node_similarities = similarities_matrix[i, :]
            else:
                # Calculate similarities for node i only if not using full matrix
                node_similarities = np.dot(
                    normalized_embeddings[i], normalized_embeddings.T
                )
                node_similarities[i] = -1.0  # Exclude self-connection

            # Calculate mean positive similarity for this node
            positive_similarities = node_similarities[node_similarities > 0]
            if len(positive_similarities) > 0:
                mean_similarity = np.mean(positive_similarities)
                # Apply local threshold factor
                node_threshold = (
                    mean_similarity * local_threshold_factor
                )  # Use new parameter name

                # Find nodes above threshold
                above_threshold = np.where(node_similarities > node_threshold)[0]

                # Add edges originating from node i
                for target in above_threshold:
                    # No need for 'if target != i' check due to similarities[i] = -1.0
                    rows.append(i)
                    cols.append(target)
                    data.append(node_similarities[target])
            # else: Node i has no positive similarities to others, won't connect outwards based on this logic

        # Error checking
        if len(rows) == 0:
            print(
                f"Warning: No edges were created! Check embeddings or try reducing local_threshold_factor (currently {local_threshold_factor})."
            )
            # Optional Fallback: Add basic connectivity (e.g., connect to single nearest neighbor)
            print(
                "Adding fallback edges (connecting each node to its most similar neighbor)..."
            )
            if not use_full_matrix:  # Need to recalculate if not available
                similarities_matrix = np.dot(
                    normalized_embeddings, normalized_embeddings.T
                )
            np.fill_diagonal(
                similarities_matrix, -np.inf
            )  # Use -inf for argmax to ignore self
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])):
                    continue  # Skip if no valid neighbors
                top_idx = np.argmax(similarities_matrix[i])
                rows.append(i)
                cols.append(top_idx)
                data.append(similarities_matrix[i, top_idx])
            np.fill_diagonal(
                similarities_matrix, 1.0
            )  # Restore diagonal if needed elsewhere

        # Create PyTorch tensors
        if not rows:  # Handle case where even fallback fails
            print("Error: Still no edges after fallback.")
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
            edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)

        print(
            f"Created {edge_index.shape[1]} edges, average degree {edge_index.shape[1]/num_nodes:.2f}"
        )

        return edge_index, edge_attr

    def _build_global_threshold_edges( 
        self,
        embeddings: np.ndarray,
        alpha: float = 0.5,
        max_degree: int = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges using a GLOBAL dynamic threshold based on overall similarity statistics.

        Args:
            embeddings: Node feature embedding matrix
            alpha: Coefficient controlling threshold strictness relative to global mean/std (default: 0.5)
            max_degree: Maximum degree for each node (default: None, no limit) - Applied after thresholding

        Returns:
            edge_index: Edge indices (2 x num_edges)
            edge_attr: Edge attributes (num_edges x 1)
        """
        print(  # Updated print
            f"Building GLOBAL threshold graph with alpha={alpha}"
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
        all_similarities_list = []
        # For smaller N, calculate full matrix; for larger N, sample pairs
        if num_nodes < 2000:  # Heuristic threshold
            try:
                similarities_matrix = np.dot(
                    normalized_embeddings, normalized_embeddings.T
                )
                # Extract upper triangle (excluding diagonal) for non-self, non-duplicate pairs
                mask = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
                all_similarities = similarities_matrix[mask]
                print(
                    f"  Calculated {len(all_similarities)} unique non-self similarities."
                )
            except MemoryError:
                print(
                    "  MemoryError calculating full matrix, falling back to sampling."
                )
                all_similarities = None  # Will trigger sampling below
        else:
            all_similarities = None  # Trigger sampling for large N

        if all_similarities is None:
            # Sampling approach if full matrix failed or N is large
            # Determine sample size (e.g., aiming for ~1M to 10M samples)
            target_samples = min(max(1000000, num_nodes * 50), 10000000)
            pairs_to_sample = min(target_samples, num_nodes * (num_nodes - 1) // 2)
            print(f"  Sampling {pairs_to_sample} node pairs for statistics...")
            sampled_indices_i = np.random.randint(0, num_nodes, size=pairs_to_sample)
            sampled_indices_j = np.random.randint(0, num_nodes, size=pairs_to_sample)
            # Ensure i != j
            mask = sampled_indices_i != sampled_indices_j
            sampled_indices_i = sampled_indices_i[mask]
            sampled_indices_j = sampled_indices_j[mask]
            # Calculate similarities for sampled pairs
            sims = np.sum(
                normalized_embeddings[sampled_indices_i]
                * normalized_embeddings[sampled_indices_j],
                axis=1,
            )
            all_similarities_list.append(sims)
            all_similarities = np.concatenate(all_similarities_list)

        # Filter out potential numerical errors (very low values)
        all_similarities = all_similarities[
            all_similarities > -0.999
        ]  # Filter more strictly?

        if len(all_similarities) == 0:
            print(
                "Warning: No valid similarities found for global statistics. Check embeddings."
            )
            # Fallback: set a default threshold or handle error
            global_threshold = 0.5  # Arbitrary fallback
            sim_mean, sim_std = 0.0, 0.0
        else:
            # Calculate statistics
            sim_mean = np.mean(all_similarities)
            sim_std = np.std(all_similarities)
            # Set dynamic threshold using + (stricter connections for higher alpha)
            global_threshold = sim_mean + alpha * sim_std

        print(f"  Similarity statistics: mean={sim_mean:.4f}, std={sim_std:.4f}")
        print(f"  Global threshold calculated: {global_threshold:.4f}")

        # 3. Create edges based on the global threshold
        rows, cols, data = [], [], []

        # Reuse full matrix if available and calculated
        if "similarities_matrix" in locals() and similarities_matrix is not None:
            print("Building edges using precomputed similarity matrix...")
            # Find pairs where similarity > global_threshold, excluding diagonal
            adj = similarities_matrix > global_threshold
            np.fill_diagonal(adj, False)  # Ensure no self-loops
            edge_indices = np.argwhere(adj)
            rows = edge_indices[:, 0].tolist()
            cols = edge_indices[:, 1].tolist()
            data = similarities_matrix[rows, cols].tolist()
        else:
            # Calculate similarities row by row if matrix wasn't precomputed
            print("Building edges (row-by-row similarity calculation)...")
            for i in tqdm(range(num_nodes), desc="Building global threshold edges"):
                similarities = np.dot(normalized_embeddings[i], normalized_embeddings.T)
                similarities[i] = -1.0  # Exclude self

                # Find nodes above threshold
                above_threshold = np.where(similarities > global_threshold)[0]

                for target in above_threshold:
                    rows.append(i)
                    cols.append(target)
                    data.append(similarities[target])

        # Error checking - ensure edges exist
        if len(rows) == 0:
            print(
                f"Warning: Global threshold {global_threshold:.4f} too high, no edges created. Adding basic connectivity (nearest neighbor)..."
            )
            # Fallback: Connect each node to its single most similar neighbor
            # Need to compute similarities if not already done
            if "similarities_matrix" not in locals() or similarities_matrix is None:
                similarities_matrix = np.dot(
                    normalized_embeddings, normalized_embeddings.T
                )

            np.fill_diagonal(similarities_matrix, -np.inf)  # Use -inf for argmax
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])):
                    continue
                most_similar = np.argmax(similarities_matrix[i])
                rows.append(i)
                cols.append(most_similar)
                data.append(similarities_matrix[i, most_similar])

        # Apply max degree constraint if specified (simple truncation)
        if max_degree is not None and len(rows) > 0:
            print(f"Applying max degree constraint: {max_degree}")
            adj_list = {}
            new_rows, new_cols, new_data = [], [], []
            # Group edges by source node and sort by similarity (desc)
            for r, c, d in zip(rows, cols, data):
                if r not in adj_list:
                    adj_list[r] = []
                adj_list[r].append((c, d))  # Store target and similarity

            for r in adj_list:
                # Sort neighbors by similarity, high to low
                adj_list[r].sort(key=lambda x: x[1], reverse=True)
                # Keep only top max_degree neighbors
                kept_neighbors = adj_list[r][:max_degree]
                for c, d in kept_neighbors:
                    new_rows.append(r)
                    new_cols.append(c)
                    new_data.append(d)
            rows, cols, data = new_rows, new_cols, new_data
            print(f"  Edges after max degree constraint: {len(rows)}")

        # Create PyTorch tensors
        if not rows:
            print("Error: Still no edges after potential fallback.")
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
            edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)

        print(
            f"Created {edge_index.shape[1]} edges, average degree {edge_index.shape[1]/num_nodes:.2f}"
        )

        return edge_index, edge_attr

    def _build_quantile_edges(
        self, embeddings: np.ndarray, quantile_p: float = 95.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges using a quantile-based threshold on similarities.

        Connects nodes if their similarity is above the p-th percentile
        of all non-self pairwise similarities.

        Args:
            embeddings: Node feature embedding matrix.
            quantile_p: The percentile (0-100) to use as the similarity threshold
                        (e.g., 95.0 means connect nodes in the top 5% similarity).

        Returns:
            edge_index: Edge indices (2 x num_edges).
            edge_attr: Edge attributes (similarities) (num_edges x 1).
        """
        print(
            f"Building quantile edges using {quantile_p:.2f}-th percentile similarity threshold..."
        )
        if not (0 < quantile_p < 100):
            raise ValueError("quantile_p must be between 0 and 100 (exclusive).")

        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_norms = np.maximum(norms, 1e-10)
        normalized_embeddings = embeddings / valid_norms
        num_nodes = len(embeddings)

        print("Calculating pairwise similarities...")
        try:
            similarities_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        except MemoryError:
            print(
                "Error: MemoryError calculating full similarity matrix for quantile. Cannot proceed."
            )
            # Consider sampling or alternative if this is a common issue
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, 1), dtype=torch.float
            )

        # --- 3. Extract non-self similarities and find threshold ---
        mask = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
        non_self_similarities = similarities_matrix[mask]

        if len(non_self_similarities) == 0:
            print("Warning: No non-self similarities found (graph might be too small).")
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, 1), dtype=torch.float
            )

        # Calculate the similarity threshold based on the specified percentile
        similarity_threshold = np.percentile(non_self_similarities, quantile_p)
        print(
            f"Similarity distribution {quantile_p:.2f}-th percentile threshold: {similarity_threshold:.4f}"
        )

        # --- 4. Build edges ---
        rows, cols, data = [], [], []
        # Find pairs where similarity > threshold, excluding diagonal
        adj = similarities_matrix > similarity_threshold
        np.fill_diagonal(adj, False)  # Ensure no self-loops
        indices_rows, indices_cols = np.where(adj)

        print(f"Found {len(indices_rows)} edges above threshold.")

        # Directly use indices found by np.where
        rows = indices_rows.tolist()
        cols = indices_cols.tolist()
        data = similarities_matrix[rows, cols].tolist()

        # --- 5. Handle no edges case ---
        if len(rows) == 0:
            print(
                f"Warning: Threshold {similarity_threshold:.4f} too high, no edges created. Adding basic connectivity (nearest neighbor)..."
            )
            # Fallback: Connect each node to its single most similar neighbor
            np.fill_diagonal(similarities_matrix, -np.inf)  # Exclude self for argmax
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])):
                    continue
                most_similar_neighbor = np.argmax(similarities_matrix[i])
                rows.append(i)
                cols.append(most_similar_neighbor)
                data.append(similarities_matrix[i, most_similar_neighbor])

        if len(rows) == 0:  # Still no edges after fallback
            print("Error: Could not create any edges.")
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, 1), dtype=torch.float
            )

        # --- 6. Create tensors ---
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)

        print(
            f"Created {edge_index.shape[1]} edges, average degree {edge_index.shape[1]/num_nodes:.2f}"
        )

        return edge_index, edge_attr

    def _build_topk_mean_edges(
        self,
        embeddings: np.ndarray,
        k_top_sim: int = 10,
        threshold_factor: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges based on the global average of mean similarities to top-k neighbors.

        Steps:
        1. For each node, find its k most similar neighbors.
        2. Calculate the mean similarity to these k neighbors (local_topk_mean_sim).
        3. Compute the global average of these local_topk_mean_sim values (global_avg_topk_mean_sim).
        4. Set the final threshold = global_avg_topk_mean_sim * threshold_factor.
        5. Connect nodes if their similarity exceeds this final threshold.

        Args:
            embeddings: Node feature embedding matrix.
            k_top_sim: The number (K) of nearest neighbors to consider for local mean similarity calculation.
            threshold_factor: Factor to adjust the final threshold relative to the global average.

        Returns:
            edge_index: Edge indices (2 x num_edges).
            edge_attr: Edge attributes (similarities) (num_edges x 1).
        """
        print(
            f"Building Top-K Mean edges with k_top_sim={k_top_sim}, threshold_factor={threshold_factor:.2f}..."
        )

        # --- 1. Preprocessing ---
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_norms = np.maximum(norms, 1e-10)
        normalized_embeddings = embeddings / valid_norms
        num_nodes = len(embeddings)

        # --- 2. Calculate all pairwise similarities ---
        print("Calculating pairwise similarities...")
        try:
            similarities_matrix = cosine_similarity(normalized_embeddings)
        except MemoryError:
            print(
                "Error: MemoryError calculating full similarity matrix for topk_mean. Cannot proceed."
            )
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, 1), dtype=torch.float
            )

        # Ensure self-similarity doesn't interfere with top-k selection later
        np.fill_diagonal(
            similarities_matrix, -np.inf
        )  # Set self-sim to negative infinity

        # --- 3. Calculate local mean similarity to Top-K for each node ---
        local_topk_mean_sims = []
        print(
            f"Calculating mean similarity to top-{k_top_sim} neighbors for each node..."
        )
        for i in tqdm(range(num_nodes), desc="Computing local Top-K means"):
            node_similarities = similarities_matrix[i, :]

            # Find indices of top k neighbors (excluding self, already handled by -inf)
            actual_k = min(k_top_sim, num_nodes - 1)
            if actual_k <= 0:
                local_topk_mean_sims.append(
                    0
                )  # Handle case of single-node graph or k=0
                continue

            # Use argpartition for efficiency when k << N
            top_k_indices = np.argpartition(node_similarities, -actual_k)[-actual_k:]

            # Get the similarities of these top k neighbors (filter out -inf if any made it)
            top_k_similarities = node_similarities[top_k_indices]
            top_k_similarities = top_k_similarities[np.isfinite(top_k_similarities)]

            # Calculate the mean similarity
            if len(top_k_similarities) > 0:
                local_mean = np.mean(top_k_similarities)
                local_topk_mean_sims.append(local_mean)
            else:
                local_topk_mean_sims.append(0)  # No valid neighbors found

        # --- 4. Calculate global threshold ---
        if not local_topk_mean_sims:
            print("Error: Could not calculate any local top-k mean similarities.")
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, 1), dtype=torch.float
            )

        global_avg_topk_mean_sim = np.mean(local_topk_mean_sims)
        final_threshold = global_avg_topk_mean_sim * threshold_factor  # Apply factor
        print(
            f"Global average of Top-{k_top_sim} mean similarities: {global_avg_topk_mean_sim:.4f}"
        )
        print(
            f"Final similarity threshold (applied factor {threshold_factor:.2f}): {final_threshold:.4f}"
        )

        # --- 5. Build edges based on the final threshold ---
        # Restore diagonal for edge checking (just in case)
        np.fill_diagonal(similarities_matrix, 1.0)

        rows, cols, data = [], [], []
        # Find pairs where similarity > final_threshold
        adj = similarities_matrix > final_threshold
        np.fill_diagonal(adj, False)  # Exclude self-loops
        indices_rows, indices_cols = np.where(adj)

        print(f"Found {len(indices_rows)} edges above threshold.")

        rows = indices_rows.tolist()
        cols = indices_cols.tolist()
        data = similarities_matrix[rows, cols].tolist()

        # --- 6. Handle no edges case ---
        if len(rows) == 0:
            print(
                f"Warning: Threshold {final_threshold:.4f} too high, no edges created. Adding basic connectivity (nearest neighbor)..."
            )
            # Fallback: Connect each node to its single most similar neighbor
            np.fill_diagonal(
                similarities_matrix, -np.inf
            )  # Exclude self again for argmax
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])):
                    continue
                most_similar_neighbor = np.argmax(similarities_matrix[i])
                rows.append(i)
                cols.append(most_similar_neighbor)
                data.append(similarities_matrix[i, most_similar_neighbor])

        if len(rows) == 0:
            print("Error: Could not create any edges.")
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, 1), dtype=torch.float
            )

        # --- 7. Create tensors ---
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)

        print(
            f"Created {edge_index.shape[1]} edges, average degree {edge_index.shape[1]/num_nodes:.2f}"
        )

        return edge_index, edge_attr

    def _analyze_graph(self) -> None:
        """Analyze the graph and compute comprehensive metrics."""
        if (
            self.graph_data is None
            or self.graph_data.edge_index is None
            or self.graph_data.edge_index.shape[1] == 0
        ):
            print("\nGraph Analysis Summary:")
            print("  Graph has no edges. Skipping detailed analysis.")
            self.graph_metrics = {
                "num_nodes": self.graph_data.num_nodes if self.graph_data else 0,
                "num_edges": 0,
                "num_train_nodes": (
                    self.graph_data.train_mask.sum().item()
                    if self.graph_data and hasattr(self.graph_data, "train_mask")
                    else 0
                ),
                "num_test_nodes": (
                    self.graph_data.test_mask.sum().item()
                    if self.graph_data and hasattr(self.graph_data, "test_mask")
                    else 0
                ),
                "num_labeled_nodes": (
                    self.graph_data.labeled_mask.sum().item()
                    if self.graph_data and hasattr(self.graph_data, "labeled_mask")
                    else 0
                ),
                "avg_degree": 0.0,
                "analysis_skipped": True,
            }
            return

        # Basic metrics
        num_nodes = self.graph_data.num_nodes
        num_edges = self.graph_data.edge_index.shape[1]
        self.graph_metrics = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_train_nodes": self.graph_data.train_mask.sum().item(),
            "num_test_nodes": self.graph_data.test_mask.sum().item(),
            "num_labeled_nodes": self.graph_data.labeled_mask.sum().item(),
            "avg_degree": num_edges / num_nodes if num_nodes > 0 else 0.0,
        }

        # --- Use NetworkX for more complex metrics ---
        print("Converting to NetworkX for analysis...")
        # Ensure edge_index is on CPU and is LongTensor
        edge_index_cpu = self.graph_data.edge_index.cpu().long()
        # We assume undirected for most analysis, but preserve original for edge type counts
        G_undirected = to_networkx(
            Data(edge_index=edge_index_cpu, num_nodes=num_nodes), to_undirected=True
        )
        is_connected = nx.is_connected(G_undirected)
        print(f"  Is graph connected (undirected)? {is_connected}")

        # Calculate degree (use original directed edges for accurate in/out if needed, but undirected common)
        degrees_list = [d for n, d in G_undirected.degree()]
        degrees = torch.tensor(degrees_list, dtype=torch.long)  # Use NetworkX degrees

        # Edge type analysis (using original directed edges)
        edge_types = {"train-train": 0, "train-test": 0, "test-test": 0}
        train_mask_np = self.graph_data.train_mask.cpu().numpy()
        test_mask_np = self.graph_data.test_mask.cpu().numpy()

        for i in tqdm(range(num_edges), desc="Analyzing edge types"):
            source = edge_index_cpu[0, i].item()
            target = edge_index_cpu[1, i].item()
            s_is_train = train_mask_np[source]
            t_is_train = train_mask_np[target]
            s_is_test = test_mask_np[source]
            t_is_test = test_mask_np[target]

            if s_is_train and t_is_train:
                edge_types["train-train"] += 1
            elif (s_is_train and t_is_test) or (s_is_test and t_is_train):
                edge_types["train-test"] += 1
            elif s_is_test and t_is_test:
                edge_types["test-test"] += 1
            # Note: This assumes nodes are either train or test, might need adjustment if other types exist

        self.graph_metrics["edge_types"] = edge_types

        # Find nodes with degree 0 (isolated nodes)
        isolated_nodes = (degrees == 0).nonzero(as_tuple=True)[0].tolist()
        self.graph_metrics["isolated_nodes"] = len(isolated_nodes)

        # Calculate connectivity patterns between fake/real news
        fake_to_fake = 0
        real_to_real = 0
        fake_to_real = 0
        y_np = self.graph_data.y.cpu().numpy()

        for i in tqdm(range(num_edges), desc="Analyzing class connectivity"):
            source = edge_index_cpu[0, i].item()
            target = edge_index_cpu[1, i].item()
            source_label = y_np[source]
            target_label = y_np[target]

            # Treat as undirected for homophily calculation
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
        total_undirected_edges = (
            G_undirected.number_of_edges()
        )  # Use undirected edge count for ratio

        self.graph_metrics["homophilic_edges"] = homophilic_edges
        self.graph_metrics["heterophilic_edges"] = fake_to_real
        self.graph_metrics["homophily_ratio"] = (
            homophilic_edges / total_undirected_edges
            if total_undirected_edges > 0
            else 0.0
        )

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

        # Power-law analysis (optional, requires scipy)
        try:
            from scipy import stats

            if len(unique_degrees) > 1:
                # Remove zero degrees for power-law fitting
                valid_degrees = unique_degrees[unique_degrees > 0]
                valid_counts = degree_counts[unique_degrees > 0]

                if len(valid_degrees) > 1:  # Need at least 2 points for regression
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
                        and r_value**2 > 0.7,  # Rough criteria
                    }
        except ImportError:
            print("  Scipy not installed, skipping power-law analysis.")
        except Exception as e:
            print(f"  Error during power-law analysis: {e}")

        # Calculate clustering coefficient
        try:
            avg_clustering = nx.average_clustering(G_undirected)
            self.graph_metrics["clustering"] = {
                "average_clustering": float(avg_clustering),
                # "clustering_distribution": { # Can be large, maybe omit from summary json
                #     str(k): float(v) for k, v in nx.clustering(G_undirected).items()
                # },
            }
        except Exception as e:
            print(f"  Error calculating clustering coefficient: {e}")
            self.graph_metrics["clustering"] = {"average_clustering": None}

        # Calculate graph density
        try:
            density = nx.density(G_undirected)
            self.graph_metrics["density"] = float(density)
        except Exception as e:
            print(f"  Error calculating density: {e}")
            self.graph_metrics["density"] = None

        # Calculate average path length (only if connected)
        if is_connected:
            try:
                avg_path_length = nx.average_shortest_path_length(G_undirected)
                self.graph_metrics["avg_path_length"] = float(avg_path_length)
            except Exception as e:
                print(f"  Error calculating average path length: {e}")
                self.graph_metrics["avg_path_length"] = None
        else:
            print(
                "  Graph is not connected, cannot compute average shortest path length for the whole graph."
            )
            self.graph_metrics["avg_path_length"] = (
                None  # Or calculate for largest component
            )

        # Calculate assortativity (degree correlation)
        try:
            assortativity = nx.degree_assortativity_coefficient(G_undirected)
            self.graph_metrics["assortativity"] = float(assortativity)
        except Exception as e:
            print(f"  Error calculating assortativity: {e}")
            self.graph_metrics["assortativity"] = None

        # Print comprehensive analysis
        print("\nGraph Analysis Summary:")
        print(f"  Nodes: {self.graph_metrics['num_nodes']}")
        print(f"  Edges: {self.graph_metrics['num_edges']}")
        print(f"  Average degree: {self.graph_metrics['avg_degree']:.2f}")
        print(f"  Graph density: {self.graph_metrics.get('density', 'N/A'):.4f}")
        if (
            "clustering" in self.graph_metrics
            and self.graph_metrics["clustering"].get("average_clustering") is not None
        ):
            print(
                f"  Average clustering coefficient: {self.graph_metrics['clustering']['average_clustering']:.4f}"
            )
        if self.graph_metrics.get("assortativity") is not None:
            print(f"  Assortativity: {self.graph_metrics['assortativity']:.4f}")

        if (
            "avg_path_length" in self.graph_metrics
            and self.graph_metrics["avg_path_length"] is not None
        ):
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
        total_reported_edges = sum(edge_types.values())
        for edge_type, count in edge_types.items():
            percentage = (
                (count / total_reported_edges * 100) if total_reported_edges > 0 else 0
            )
            print(f"    - {edge_type}: {count} ({percentage:.1f}%)")

        print("\n  Class Connectivity (Undirected View):")
        total_undir_edges = (
            self.graph_metrics["homophilic_edges"]
            + self.graph_metrics["heterophilic_edges"]
        )
        if total_undir_edges > 0:
            print(
                f"    Fake-to-Fake: {fake_to_fake} ({fake_to_fake/total_undir_edges*100:.1f}%)"
            )
            print(
                f"    Real-to-Real: {real_to_real} ({real_to_real/total_undir_edges*100:.1f}%)"
            )
            print(
                f"    Fake-to-Real: {fake_to_real} ({fake_to_real/total_undir_edges*100:.1f}%)"
            )
            print(f"    Homophily ratio: {self.graph_metrics['homophily_ratio']:.4f}")
        else:
            print("    No edges to analyze for class connectivity.")

        if isolated_nodes:
            print(f"\n  Warning: {len(isolated_nodes)} isolated nodes found!")
        print("")

    def save_graph(self) -> str:
        """Save the graph and analysis results."""
        if self.graph_data is None:
            raise ValueError("Graph must be built before saving")

        # --- Generate graph name - include embedding type in the filename only ---
        edge_policy_name = self.edge_policy
        edge_param_str = ""

        if self.edge_policy in ["knn", "mutual_knn"]:
            edge_param_str = str(self.k_neighbors)
        elif self.edge_policy == "local_threshold":  # New policy name
            edge_param_str = f"{self.local_threshold_factor:.2f}".replace(
                ".", "p"
            )  
        elif self.edge_policy == "global_threshold":  
            edge_param_str = f"{self.alpha:.2f}".replace(".", "p")  
        elif self.edge_policy == "quantile":
            edge_param_str = f"{self.quantile_p:.1f}".replace(".", "p")
        elif self.edge_policy == "topk_mean":
            edge_param_str = (
                f"k{self.k_top_sim}_f{self.local_threshold_factor:.2f}".replace(
                    ".", "p"
                )
            )
        else:
            edge_policy_name = "unknown_policy"
            edge_param_str = "NA"

        graph_name = f"{self.k_shot}_shot_{self.embedding_type}_{edge_policy_name}_{edge_param_str}"
        # --- End filename generation ---

        # Save graph data
        graph_path = os.path.join(self.output_dir, f"{graph_name}.pt")
        torch.save(self.graph_data, graph_path)

        # Save graph metrics if available
        if self.graph_metrics:
            # Convert any numpy numbers to Python native types
            python_metrics = json.loads(
                json.dumps(
                    self.graph_metrics,
                    default=lambda x: (
                        int(x)
                        if isinstance(x, np.integer)
                        else float(x) if isinstance(x, np.floating) else None
                    ),  # Handle potential non-serializable types
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

        # Limit visualization to avoid memory issues and clutter
        num_nodes_actual = self.graph_data.num_nodes
        nodes_to_plot = min(num_nodes_actual, max_nodes)

        print(f"Preparing graph visualization ({nodes_to_plot} nodes)...")

        # Convert to NetworkX graph for visualization
        # Select subset of nodes if graph is too large
        if num_nodes_actual > max_nodes:
            print(
                f"  Graph is large ({num_nodes_actual} nodes). Visualizing a subgraph of {max_nodes} nodes."
            )
            # Create subgraph data object
            subgraph_nodes = list(range(max_nodes))
            subgraph_data = self.graph_data.subgraph(
                torch.tensor(subgraph_nodes, dtype=torch.long)
            )
            G = to_networkx(subgraph_data, to_undirected=True)
            # Get masks for the subgraph
            train_mask_viz = subgraph_data.train_mask.cpu().numpy()
            test_mask_viz = subgraph_data.test_mask.cpu().numpy()
            labeled_mask_viz = subgraph_data.labeled_mask.cpu().numpy()
            num_nodes_viz = max_nodes
        else:
            G = to_networkx(self.graph_data, to_undirected=True)
            # Get masks for the full graph
            train_mask_viz = self.graph_data.train_mask.cpu().numpy()
            test_mask_viz = self.graph_data.test_mask.cpu().numpy()
            labeled_mask_viz = self.graph_data.labeled_mask.cpu().numpy()
            num_nodes_viz = num_nodes_actual

        # Check if graph has edges before proceeding
        if G.number_of_edges() == 0:
            print("  Warning: Graph has no edges. Visualization will only show nodes.")

        # Set node colors based on masks
        node_colors = []
        for i in range(num_nodes_viz):
            if labeled_mask_viz[i]:
                node_colors.append("green")  # Priority to labeled nodes
            elif train_mask_viz[i]:
                node_colors.append("blue")
            elif test_mask_viz[i]:
                node_colors.append("red")
            else:
                # Should not happen if masks cover all nodes, but include fallback
                node_colors.append("gray")

        # Create figure
        plt.figure(figsize=(14, 12))  # Slightly larger figure

        # Choose layout based on size
        print("  Calculating layout...")
        if num_nodes_viz > 500:
            pos = nx.random_layout(G, seed=self.seed)  # Faster for large graphs
        else:
            # Spring layout can be slow for >500 nodes
            pos = nx.spring_layout(
                G, k=0.2 / np.sqrt(num_nodes_viz), iterations=50, seed=self.seed
            )

        print("  Drawing graph...")
        # Adjust node size and edge width based on graph size
        node_size = max(5, 4000 / num_nodes_viz) if num_nodes_viz > 0 else 20
        edge_width = (
            max(0.1, 10 / G.number_of_edges() if G.number_of_edges() > 100 else 0.5)
            if G.number_of_edges() > 0
            else 0.5
        )

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_size, alpha=0.8
        )
        if G.number_of_edges() > 0:
            nx.draw_networkx_edges(
                G, pos, edge_color="gray", width=edge_width, alpha=0.5
            )
        # Avoid drawing labels as they overlap heavily

        # Add legend
        legend_elements = [
            Patch(facecolor="green", label="Few-shot Labeled (Train)"),
            Patch(facecolor="blue", label="Train (Unlabeled)"),
            Patch(facecolor="red", label="Test (Unlabeled)"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize="medium")

        # Add title
        title = f"{self.dataset_name.capitalize()} Graph ({self.edge_policy.replace('_', ' ').title()})"
        if num_nodes_actual > max_nodes:
            title += f"\n(Showing {max_nodes} of {num_nodes_actual} nodes)"

        subtitle = f"{self.k_shot}-shot | Embed: {self.embedding_type.upper()} | Edges: {G.number_of_edges()}"
        plt.title(f"{title}\n{subtitle}", fontsize="x-large")
        plt.axis("off")  # Hide axes

        # Save figure
        plot_path = os.path.join(self.plot_dir, f"{graph_name}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")  # Adjust DPI if needed
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
        help="Dataset to use (e.g., politifact, gossipcop)",
        choices=["politifact", "gossipcop"],
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=8,
        help="Number of samples per class for few-shot learning (e.g., 3, 8, 16)",
        choices=list(range(3, 21)),  # Allow 3 to 20
    )

    # graph construction arguments
    parser.add_argument(
        "--edge_policy",
        type=str,
        default=DEFAULT_EDGE_POLICY,  # Uses new default
        help="Edge construction policy",
        choices=[
            "knn",
            "mutual_knn",
            "local_threshold",
            "global_threshold",
            "quantile",
            "topk_mean",
        ],  # Updated choices
    )
    ## for knn and mutual_knn
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=DEFAULT_K_NEIGHBORS,
        help=f"Number of neighbors for KNN or Mutual KNN (default: {DEFAULT_K_NEIGHBORS})",
    )
    ## for local_threshold (previously threshold)
    parser.add_argument(
        # "--threshold_factor", # Old argument name
        "--local_threshold_factor",  # New argument name
        type=float,
        # default=DEFAULT_THRESHOLD_FACTOR, # Old default name
        default=DEFAULT_LOCAL_THRESHOLD_FACTOR,  # New default name
        # help=f"Threshold factor for threshold edge construction (default: {DEFAULT_THRESHOLD_FACTOR})", # Old help
        help=f"Factor to adjust node's local similarity threshold for 'local_threshold' policy (default: {DEFAULT_LOCAL_THRESHOLD_FACTOR})",  # New help
    )
    ## for global_threshold (previously dynamic_threshold) - still uses alpha
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        # help=f"Alpha parameter for dynamic threshold edge construction (default: {DEFAULT_ALPHA})", # Old help
        help=f"Alpha parameter (mean + alpha*std) for 'global_threshold' policy (default: {DEFAULT_ALPHA})",  # New help
    )
    ## for quantile
    parser.add_argument(
        "--quantile_p",
        type=float,
        default=DEFAULT_QUANTILE_P,
        help=f"Quantile percentile for quantile edge construction (default: {DEFAULT_QUANTILE_P})",
    )
    ## for topk_mean
    parser.add_argument(
        "--k_top_sim",
        type=int,
        default=10,
        help="Number of top neighbors (K) to compute local mean similarity for 'topk_mean' policy (default: 10)",
    )
    # Note: topk_mean also uses a 'threshold_factor'. If it should be distinct from local_threshold_factor, add a separate arg here.
    # Currently, it will reuse --local_threshold_factor based on the code in build_graph().

    # news embedding type
    parser.add_argument(
        "--embedding_type",
        type=str,
        default=DEFAULT_EMBEDDING_TYPE,
        help=f"Type of embeddings to use (default: {DEFAULT_EMBEDDING_TYPE})",
        choices=["bert", "roberta", "combined"],  # Add more if available
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
        help="Enable graph visualization (can be slow for large graphs)",
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
    if torch.cuda.is_available():
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

    # Print policy-specific parameters
    if args.edge_policy in ["knn", "mutual_knn"]:
        print(f"K neighbors:      {args.k_neighbors}")
    # elif args.edge_policy == "threshold": # Old
    #     print(f"Threshold factor: {args.threshold_factor}") # Old
    elif args.edge_policy == "local_threshold":  # New
        print(f"Local Factor:     {args.local_threshold_factor}")  # New
    # elif args.edge_policy == "dynamic_threshold": # Old
    #     print(f"Alpha:            {args.alpha}") # Old
    elif args.edge_policy == "global_threshold":  # New
        print(f"Alpha:            {args.alpha}")  # New (kept alpha)
    elif args.edge_policy == "quantile":
        print(f"Quantile p:       {args.quantile_p}")
    elif args.edge_policy == "topk_mean":
        print(f"K Top Sim:        {args.k_top_sim}")
        print(f"Threshold Factor: {args.local_threshold_factor}")

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
        local_threshold_factor=args.local_threshold_factor,
        alpha=args.alpha,
        quantile_p=args.quantile_p,
        k_top_sim=args.k_top_sim,  # Added k_top_sim here
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
    graph_file_name = f"{args.k_shot}_shot_{args.embedding_type}_{builder.edge_policy}_..."  # Example structure
    print(f"  1. Check the created graph file and metrics in: {builder.output_dir}/")
    print(
        f"  2. Train a GNN model: python train_graph.py --graph_path {builder.output_dir}/{graph_file_name}.pt"
    )
    print(
        f"  3. Compare with language models: python finetune_lm.py --dataset_name <dataset> --k_shot <k>"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
