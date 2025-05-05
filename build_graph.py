# -*- coding: utf-8 -*-
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
DEFAULT_SEED = 42
DEFAULT_K_NEIGHBORS = 5
DEFAULT_EDGE_POLICY = "knn"
DEFAULT_EMBEDDING_TYPE = "roberta"
DEFAULT_LOCAL_THRESHOLD_FACTOR = 1.0
DEFAULT_ALPHA = 0.1
DEFAULT_QUANTILE_P = 95.0
DEFAULT_UNLABELED_SAMPLE_FACTOR = 10

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
    Supports sampling of unlabeled training nodes.
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
        sample_unlabeled: bool = False, # New argument
        unlabeled_sample_factor: int = DEFAULT_UNLABELED_SAMPLE_FACTOR, # New argument
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
        self.sample_unlabeled = sample_unlabeled # Store new argument
        self.unlabeled_sample_factor = unlabeled_sample_factor # Store new argument
        self.plot = plot
        self.seed = seed
        self.embedding_type = embedding_type.lower()

        # Setup directory paths
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
        self.selected_labeled_indices_original = None # Store original indices of labeled nodes
        self.selected_unlabeled_indices_original = None # Store original indices of sampled unlabeled nodes

        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # Set numpy random seed specifically for sampling if needed elsewhere
        np.random.seed(self.seed)


    def load_dataset(self) -> None:
        """Load dataset from HuggingFace and prepare it for graph construction."""
        print(f"Loading dataset '{self.dataset_name}' with '{self.embedding_type}' embeddings...")
        hf_dataset_name = f"LittleFish-Coder/Fake_News_{self.dataset_name}"
        dataset = load_dataset(hf_dataset_name, download_mode="reuse_cache_if_exists", cache_dir="dataset")
        self.dataset = {"train": dataset["train"], "test": dataset["test"]}
        self.train_size_original = len(dataset["train"])
        self.test_size_original = len(dataset["test"])
        unique_labels = set(dataset["train"]["label"])
        self.num_classes = len(unique_labels)
        self.labeled_size_per_class = self.k_shot
        self.total_labeled_size = self.k_shot * self.num_classes
        print(f"\nOriginal dataset size: Train={self.train_size_original}, Test={self.test_size_original}")
        print(f"Labeled set: {self.k_shot}-shot * {self.num_classes} classes = {self.total_labeled_size} total labeled nodes")
        # self._show_dataset_stats() # Keep this optional or adapt it

    # def _show_dataset_stats(self) -> None: # Can be adapted later if needed
    #     # ... (original code) ...

    def build_graph(self) -> Data:
        """Build a graph including both nodes and edges."""
        # Build nodes (potentially with sampling)
        self.build_empty_graph() # This function now handles sampling logic

        # Check if graph building created valid data
        if self.graph_data is None or self.graph_data.x is None:
             print("Error: Graph node building failed. Aborting.")
             return None

        print(f"\nBuilding graph edges using {self.edge_policy} policy...")
        embeddings = self.graph_data.x.cpu().numpy() # Ensure embeddings are on CPU for sklearn/numpy

        # Check if graph has enough nodes for the chosen policy
        if self.graph_data.num_nodes <= 1:
             print("Warning: Graph has 1 or 0 nodes. Skipping edge building.")
             self.graph_data.edge_index = torch.zeros((2, 0), dtype=torch.long)
             self.graph_data.edge_attr = torch.zeros((0, 1), dtype=torch.float)
             self.graph_data.num_edges = 0
             return self.graph_data

        # --- Edge Building Logic ---
        edge_func_map = {
            "knn": self._build_knn_edges,
            "mutual_knn": self._build_mutual_knn_edges,
            "local_threshold": self._build_local_threshold_edges,
            "global_threshold": self._build_global_threshold_edges,
            "quantile": self._build_quantile_edges,
            "topk_mean": self._build_topk_mean_edges,
        }

        if self.edge_policy in edge_func_map:
            build_func = edge_func_map[self.edge_policy]
            # Prepare arguments for the specific function
            if self.edge_policy in ["knn", "mutual_knn"]:
                edges, edge_attr = build_func(embeddings, self.k_neighbors)
            elif self.edge_policy == "local_threshold":
                 edges, edge_attr = build_func(embeddings, self.local_threshold_factor)
            elif self.edge_policy == "global_threshold":
                edges, edge_attr = build_func(embeddings, self.alpha)
            elif self.edge_policy == "quantile":
                 edges, edge_attr = build_func(embeddings, self.quantile_p)
            elif self.edge_policy == "topk_mean":
                 # Reusing local_threshold_factor for topk_mean's factor
                 edges, edge_attr = build_func(embeddings, self.k_top_sim, self.local_threshold_factor)
            else: # Should not happen due to check above, but for safety
                 raise ValueError(f"Edge policy '{self.edge_policy}' handler not found.")
        else:
            raise ValueError(f"Unknown edge policy: {self.edge_policy}")

        # Update graph data
        self.graph_data.edge_index = edges.to(self.device) # Move edges to target device
        if edge_attr is not None:
             self.graph_data.edge_attr = edge_attr.to(self.device) # Move attributes to target device
        else:
             self.graph_data.edge_attr = None
        self.graph_data.num_edges = edges.shape[1]

        print(f"Graph edges built: {self.graph_data.num_edges} edges created")

        # Analyze the graph
        self._analyze_graph()

        return self.graph_data

    def build_empty_graph(self) -> Optional[Data]:
        """
        Builds the graph nodes and masks. Handles sampling of unlabeled training nodes if enabled.
        """
        print(f"\nBuilding graph nodes using {self.embedding_type} embeddings...")
        embedding_field = f"{self.embedding_type}_embeddings"
        print(f"Using embeddings from field: '{embedding_field}'")

        train_data = self.dataset["train"]
        test_data = self.dataset["test"]

        # --- 1. Identify Labeled Nodes ---
        print(f"Sampling {self.k_shot}-shot labeled nodes with seed={self.seed}")
        labeled_indices_original, _ = sample_k_shot(train_data, self.k_shot, self.seed)
        self.selected_labeled_indices_original = labeled_indices_original # Store original indices
        print(f"  Selected {len(labeled_indices_original)} labeled nodes (original indices): {labeled_indices_original[:10]}...") # Show first few

        # --- 2. Handle Unlabeled Training Node Sampling ---
        final_train_indices_original = None # Store original indices of all train nodes used
        sampled_unlabeled_indices_original = None # Store original indices of sampled unlabeled

        if self.sample_unlabeled:
            print("Sampling unlabeled training nodes enabled.")
            # Find all original indices of unlabeled training nodes
            all_train_indices_original = np.arange(self.train_size_original)
            unlabeled_indices_original = np.setdiff1d(all_train_indices_original, labeled_indices_original, assume_unique=True)

            num_unlabeled_available = len(unlabeled_indices_original)
            num_to_sample = int(self.unlabeled_sample_factor * self.total_labeled_size) # M * (2*k)

            # Ensure we don't sample more than available
            num_to_sample = min(num_to_sample, num_unlabeled_available)

            print(f"  Target sample size: {self.unlabeled_sample_factor} * {self.total_labeled_size} = {self.unlabeled_sample_factor * self.total_labeled_size}")
            if num_unlabeled_available == 0:
                 print("  Warning: No unlabeled training nodes available to sample.")
                 num_to_sample = 0
                 sampled_unlabeled_indices_original = np.array([], dtype=int)
            elif num_to_sample == 0:
                 print("  Warning: Calculated sample size is 0. No unlabeled nodes will be sampled.")
                 sampled_unlabeled_indices_original = np.array([], dtype=int)
            else:
                print(f"  Sampling {num_to_sample} nodes from {num_unlabeled_available} available unlabeled training nodes...")
                # Use numpy's random generator with the set seed for reproducibility
                rng = np.random.default_rng(self.seed)
                sampled_unlabeled_indices_original = rng.choice(
                    unlabeled_indices_original, size=num_to_sample, replace=False
                )
                self.selected_unlabeled_indices_original = sampled_unlabeled_indices_original # Store them
                print(f"  Selected {len(sampled_unlabeled_indices_original)} unlabeled nodes (original indices): {sampled_unlabeled_indices_original[:10]}...")

            # Combine labeled and sampled unlabeled indices
            final_train_indices_original = np.concatenate([labeled_indices_original, sampled_unlabeled_indices_original])
            # Sort for potentially easier processing later, although not strictly required
            final_train_indices_original = np.sort(final_train_indices_original)
            print(f"  Total training nodes included in graph: {len(final_train_indices_original)}")

        else:
            # Use all training nodes
            print("Using ALL training nodes (sampling disabled).")
            final_train_indices_original = np.arange(self.train_size_original)

        # --- 3. Extract Data for Selected Nodes ---
        # Ensure indices are list or array for slicing datasets
        final_train_indices_list = final_train_indices_original.tolist()

        # Use .select() for efficient slicing
        train_data_subset = train_data.select(final_train_indices_list)

        try:
             train_embeddings_subset = np.array(train_data_subset[embedding_field])
             train_labels_subset = np.array(train_data_subset["label"])
        except KeyError:
             print(f"Error: Embedding field '{embedding_field}' not found in train_data_subset.")
             return None
        except Exception as e:
            print(f"Error extracting subset data from train_data: {e}")
            return None

        # Get all test data
        try:
            test_embeddings = np.array(test_data[embedding_field])
            test_labels = np.array(test_data["label"])
        except KeyError:
             print(f"Error: Embedding field '{embedding_field}' not found in test_data.")
             return None
        except Exception as e:
             print(f"Error extracting data from test_data: {e}")
             return None


        # --- 4. Concatenate Features and Labels ---
        if train_embeddings_subset.ndim == 1: # Handle potential issue if only one train node selected
             train_embeddings_subset = train_embeddings_subset.reshape(1, -1)
        if test_embeddings.ndim == 1:
             test_embeddings = test_embeddings.reshape(1, -1)

        # Check feature dimensions match
        if train_embeddings_subset.shape[1] != test_embeddings.shape[1]:
            print(f"Error: Feature dimension mismatch! Train subset: {train_embeddings_subset.shape[1]}, Test: {test_embeddings.shape[1]}")
            # This might happen if embeddings were processed differently or missing
            # Try to identify the mismatch source
            print("  Checking original dataset feature dimensions...")
            try:
                if embedding_field in train_data.features and embedding_field in test_data.features:
                     d_train = len(train_data[0][embedding_field])
                     d_test = len(test_data[0][embedding_field])
                     print(f"  Original dimensions: Train={d_train}, Test={d_test}")
                     if d_train != d_test:
                         print("  -> Mismatch found in original dataset features!")
                else:
                    print("  Could not access features in original dataset to check dimensions.")
            except Exception as e_dim:
                print(f"  Error checking original dimensions: {e_dim}")
            return None


        x = torch.tensor(
            np.concatenate([train_embeddings_subset, test_embeddings]), dtype=torch.float
        )
        y = torch.tensor(np.concatenate([train_labels_subset, test_labels]), dtype=torch.long)

        # --- 5. Create Masks based on New Indexing ---
        num_train_nodes_in_graph = len(final_train_indices_original)
        num_test_nodes_in_graph = len(test_data)
        num_nodes = num_train_nodes_in_graph + num_test_nodes_in_graph

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[:num_train_nodes_in_graph] = True
        test_mask[num_train_nodes_in_graph:] = True

        # Create mapping from original labeled indices to new graph indices
        # The first 'len(labeled_indices_original)' indices in final_train_indices_original
        # *if sorted* correspond to the labeled nodes, BUT safer to map explicitly.
        original_to_new_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(final_train_indices_original)}

        for orig_labeled_idx in labeled_indices_original:
            if orig_labeled_idx in original_to_new_idx_map:
                new_labeled_idx = original_to_new_idx_map[orig_labeled_idx]
                # Ensure the index is within the bounds of the train part of the mask
                if new_labeled_idx < num_train_nodes_in_graph:
                     labeled_mask[new_labeled_idx] = True
                else:
                     # This should not happen if logic is correct, but good to check
                     print(f"Warning: Original labeled index {orig_labeled_idx} mapped to new index {new_labeled_idx} which is outside the train node range ({num_train_nodes_in_graph}).")

        # --- 6. Create Graph Data Object ---
        # Start with empty edges, will be populated by build_graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)

        graph_data = Data(
            x=x.to(self.device),
            y=y.to(self.device),
            train_mask=train_mask.to(self.device),
            test_mask=test_mask.to(self.device),
            labeled_mask=labeled_mask.to(self.device),
            edge_index=edge_index.to(self.device), # Start empty
            num_nodes=num_nodes,
            num_features=x.shape[1],
        )
        # Add edge_attr placeholder
        graph_data.edge_attr = None

        self.graph_data = graph_data

        # Display graph structure info
        print(f"\nGraph nodes built:")
        print(f"  Total nodes in graph: {num_nodes}")
        print(f"    - Train nodes: {num_train_nodes_in_graph} (from {self.train_size_original} originally)")
        print(f"        - Labeled: {labeled_mask.sum().item()} (target: {self.total_labeled_size})")
        if self.sample_unlabeled:
             print(f"        - Unlabeled (sampled): {num_train_nodes_in_graph - labeled_mask.sum().item()}")
        else:
             print(f"        - Unlabeled (all): {num_train_nodes_in_graph - labeled_mask.sum().item()}")
        print(f"    - Test nodes: {num_test_nodes_in_graph} (from {self.test_size_original} originally)")
        print(f"  Node features: {x.shape[1]}")

        # Sanity check counts
        if labeled_mask.sum().item() != self.total_labeled_size:
             print(f"Warning: Number of nodes in labeled_mask ({labeled_mask.sum().item()}) does not match target ({self.total_labeled_size}). Check sampling/mapping logic.")
        if train_mask.sum().item() != num_train_nodes_in_graph:
             print("Warning: train_mask count mismatch.")
        if test_mask.sum().item() != num_test_nodes_in_graph:
             print("Warning: test_mask count mismatch.")


        return graph_data

    # --- Edge Building Functions (_build_knn_edges, _build_mutual_knn_edges, etc.) ---
    # These functions remain largely the same, but now operate on the potentially
    # smaller embedding matrix passed from build_graph()
    # Add checks for num_nodes > 1 at the beginning of these functions if needed.

    def _build_knn_edges(
        self, embeddings: np.ndarray, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using k-nearest neighbors approach."""
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None
        k = min(k, num_nodes - 1) # Adjust k if it's too large
        if k <= 0: return torch.zeros((2,0), dtype=torch.long), None

        print(f"Building KNN graph with k={k}...")
        distances = pairwise_distances(embeddings, metric="cosine")
        rows, cols, data = [], [], []
        for i in tqdm(range(num_nodes), desc=f"Finding {k} nearest neighbors"):
            dist_i = distances[i].copy()
            dist_i[i] = np.inf
            indices = np.argpartition(dist_i, k)[:k]
            for j in indices:
                rows.append(i)
                cols.append(j)
                data.append(1.0 - distances[i, j]) # Similarity
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        return edge_index, edge_attr

    def _build_mutual_knn_edges(
        self, embeddings: np.ndarray, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using mutual k-nearest neighbors approach."""
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None
        k = min(k, num_nodes - 1) # Adjust k
        if k <= 0: return torch.zeros((2,0), dtype=torch.long), None

        print(f"Building mutual KNN graph with k={k}...")
        distances = pairwise_distances(embeddings, metric="cosine")
        rows, cols, data = [], [], []
        # Precompute neighbors for all nodes to check mutuality efficiently
        all_neighbors = {}
        for i in range(num_nodes):
             dist_i = distances[i].copy()
             dist_i[i] = np.inf
             neighbors_i = np.argpartition(dist_i, k)[:k]
             all_neighbors[i] = set(neighbors_i) # Use set for faster lookup

        for i in tqdm(range(num_nodes), desc=f"Checking {k} mutual neighbors"):
            if i not in all_neighbors: continue # Should not happen
            for j in all_neighbors[i]:
                # Check mutuality: if i is in j's precomputed neighbors
                if j in all_neighbors and i in all_neighbors[j]:
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0 - distances[i, j]) # Similarity

        if not rows:
            print("Warning: No mutual KNN edges found.")
            return torch.zeros((2,0), dtype=torch.long), None

        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        return edge_index, edge_attr

    def _build_local_threshold_edges(
        self, embeddings: np.ndarray, local_threshold_factor: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using a LOCAL threshold-based approach (per-node threshold)."""
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None

        print(f"Building LOCAL threshold graph with local_threshold_factor={local_threshold_factor}...")
        # Normalize embeddings
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_norms = np.maximum(norms, 1e-10)
        normalized_embeddings = embeddings / valid_norms

        rows, cols, data = [], [], []
        try:
             print("  Calculating full similarity matrix...")
             similarities_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
             np.fill_diagonal(similarities_matrix, -1.0) # Exclude self
        except MemoryError:
             print("  Error: MemoryError calculating similarity matrix. Cannot proceed with local_threshold.")
             return torch.zeros((2,0), dtype=torch.long), None

        print("  Calculating local thresholds and building edges...")
        for i in tqdm(range(num_nodes), desc="Computing local threshold edges"):
            node_similarities = similarities_matrix[i, :]
            positive_similarities = node_similarities[node_similarities > 0]
            if len(positive_similarities) > 0:
                mean_similarity = np.mean(positive_similarities)
                node_threshold = mean_similarity * local_threshold_factor
                above_threshold = np.where(node_similarities > node_threshold)[0]
                for target in above_threshold:
                    rows.append(i)
                    cols.append(target)
                    data.append(node_similarities[target])

        if len(rows) == 0:
            print(f"Warning: No edges created with local_threshold_factor={local_threshold_factor}. Adding fallback edges...")
            # Fallback: Connect each node to its single most similar neighbor
            np.fill_diagonal(similarities_matrix, -np.inf)
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])): continue
                top_idx = np.argmax(similarities_matrix[i])
                rows.append(i)
                cols.append(top_idx)
                data.append(similarities_matrix[i, top_idx])

        if not rows:
            print("Error: Still no edges after fallback.")
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        print(f"  Created {edge_index.shape[1]} edges.")
        return edge_index, edge_attr

    def _build_global_threshold_edges(
        self, embeddings: np.ndarray, alpha: float = 0.5, max_degree: int = None # max_degree currently unused but kept for potential future use
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using a GLOBAL dynamic threshold based on overall similarity statistics."""
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None

        print(f"Building GLOBAL threshold graph with alpha={alpha}...")
        # Normalize embeddings
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_norms = np.maximum(norms, 1e-10)
        normalized_embeddings = embeddings / valid_norms

        # Calculate global similarity statistics
        print("  Computing global similarity statistics...")
        try:
            # Try full matrix for statistics calculation (efficient for non-huge graphs)
            similarities_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            mask = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
            all_similarities = similarities_matrix[mask]
            print(f"    Calculated {len(all_similarities)} unique non-self similarities.")
            use_full_matrix_for_edges = True # Can reuse this matrix later
        except MemoryError:
            use_full_matrix_for_edges = False
            similarities_matrix = None # Clear memory
            print("    MemoryError calculating full matrix, sampling pairs for statistics...")
            # Sampling approach (simplified)
            target_samples = min(max(500000, num_nodes * 20), 5000000) # Adjust sample size heuristic
            pairs_to_sample = min(target_samples, num_nodes * (num_nodes - 1) // 2 if num_nodes > 1 else 0)
            if pairs_to_sample <= 0:
                 all_similarities = np.array([])
            else:
                rng = np.random.default_rng(self.seed)
                sampled_indices_i = rng.integers(0, num_nodes, size=pairs_to_sample)
                sampled_indices_j = rng.integers(0, num_nodes, size=pairs_to_sample)
                mask = sampled_indices_i != sampled_indices_j
                sampled_indices_i = sampled_indices_i[mask]
                sampled_indices_j = sampled_indices_j[mask]
                sims = np.sum(normalized_embeddings[sampled_indices_i] * normalized_embeddings[sampled_indices_j], axis=1)
                all_similarities = sims
            print(f"    Sampled {len(all_similarities)} pairs for statistics.")


        all_similarities = all_similarities[all_similarities > -0.999] # Filter noise

        if len(all_similarities) == 0:
             print("  Warning: No valid similarities found for global statistics. Setting default threshold.")
             global_threshold = 0.5 # Fallback
             sim_mean, sim_std = 0.0, 0.0
        else:
            sim_mean = np.mean(all_similarities)
            sim_std = np.std(all_similarities)
            global_threshold = sim_mean + alpha * sim_std # mean + alpha * std

        print(f"    Similarity stats: mean={sim_mean:.4f}, std={sim_std:.4f}")
        print(f"    Global threshold calculated: {global_threshold:.4f}")

        # Create edges based on the global threshold
        rows, cols, data = [], [], []
        print("  Building edges using global threshold...")
        if use_full_matrix_for_edges:
            adj = similarities_matrix > global_threshold
            np.fill_diagonal(adj, False)
            edge_indices = np.argwhere(adj)
            rows = edge_indices[:, 0].tolist()
            cols = edge_indices[:, 1].tolist()
            data = similarities_matrix[rows, cols].tolist()
        else:
            # Row-by-row calculation needed if full matrix wasn't kept
            for i in tqdm(range(num_nodes), desc="Building global threshold edges"):
                similarities = np.dot(normalized_embeddings[i], normalized_embeddings.T)
                similarities[i] = -1.0
                above_threshold = np.where(similarities > global_threshold)[0]
                for target in above_threshold:
                    rows.append(i)
                    cols.append(target)
                    data.append(similarities[target])

        if len(rows) == 0:
            print(f"Warning: Global threshold {global_threshold:.4f} too high, no edges created. Adding fallback edges...")
             # Fallback: Connect each node to its single most similar neighbor
            if not use_full_matrix_for_edges: # Need similarities_matrix if not already calculated
                try:
                    similarities_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
                except MemoryError:
                    print("Error: Cannot compute similarities for fallback due to MemoryError.")
                    return torch.zeros((2, 0), dtype=torch.long), None

            np.fill_diagonal(similarities_matrix, -np.inf)
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])): continue
                most_similar = np.argmax(similarities_matrix[i])
                rows.append(i)
                cols.append(most_similar)
                data.append(similarities_matrix[i, most_similar])

        if not rows:
            print("Error: Still no edges after fallback.")
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        print(f"  Created {edge_index.shape[1]} edges.")
        return edge_index, edge_attr

    def _build_quantile_edges(
        self, embeddings: np.ndarray, quantile_p: float = 95.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using a quantile-based threshold on similarities."""
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None

        print(f"Building quantile edges using {quantile_p:.2f}-th percentile similarity threshold...")
        if not (0 < quantile_p < 100):
            raise ValueError("quantile_p must be between 0 and 100 (exclusive).")

        # Normalize embeddings
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_norms = np.maximum(norms, 1e-10)
        normalized_embeddings = embeddings / valid_norms

        print("  Calculating pairwise similarities...")
        try:
            similarities_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        except MemoryError:
             print("  Error: MemoryError calculating full similarity matrix for quantile. Cannot proceed.")
             return torch.zeros((2, 0), dtype=torch.long), None

        # Extract non-self similarities and find threshold
        mask = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
        non_self_similarities = similarities_matrix[mask]

        if len(non_self_similarities) == 0:
             print("  Warning: No non-self similarities found.")
             return torch.zeros((2, 0), dtype=torch.long), None

        similarity_threshold = np.percentile(non_self_similarities, quantile_p)
        print(f"  Similarity distribution {quantile_p:.2f}-th percentile threshold: {similarity_threshold:.4f}")

        # Build edges
        rows, cols, data = [], [], []
        adj = similarities_matrix > similarity_threshold
        np.fill_diagonal(adj, False)
        indices_rows, indices_cols = np.where(adj)
        rows = indices_rows.tolist()
        cols = indices_cols.tolist()
        data = similarities_matrix[rows, cols].tolist()

        print(f"  Found {len(rows)} edges above threshold.")

        if len(rows) == 0:
            print(f"Warning: Threshold {similarity_threshold:.4f} too high, no edges created. Adding fallback edges...")
            # Fallback: Connect each node to its single most similar neighbor
            np.fill_diagonal(similarities_matrix, -np.inf)
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])): continue
                most_similar_neighbor = np.argmax(similarities_matrix[i])
                rows.append(i)
                cols.append(most_similar_neighbor)
                data.append(similarities_matrix[i, most_similar_neighbor])

        if not rows:
             print("Error: Could not create any edges.")
             return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        print(f"  Created {edge_index.shape[1]} edges.")
        return edge_index, edge_attr

    def _build_topk_mean_edges(
        self,
        embeddings: np.ndarray,
        k_top_sim: int = 10,
        threshold_factor: float = 1.0 # Using the 'local_threshold_factor' passed in
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges based on the global average of mean similarities to top-k neighbors."""
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None
        k_top_sim = min(k_top_sim, num_nodes - 1)
        if k_top_sim <=0: return torch.zeros((2,0), dtype=torch.long), None


        print(f"Building Top-K Mean edges with k_top_sim={k_top_sim}, threshold_factor={threshold_factor:.2f}...")

        # Normalize embeddings
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_norms = np.maximum(norms, 1e-10)
        normalized_embeddings = embeddings / valid_norms

        print("  Calculating pairwise similarities...")
        try:
            similarities_matrix = cosine_similarity(normalized_embeddings)
        except MemoryError:
             print("  Error: MemoryError calculating full similarity matrix for topk_mean. Cannot proceed.")
             return torch.zeros((2, 0), dtype=torch.long), None

        np.fill_diagonal(similarities_matrix, -np.inf) # Exclude self for top-k

        # Calculate local mean similarity to Top-K for each node
        local_topk_mean_sims = []
        print(f"  Calculating mean similarity to top-{k_top_sim} neighbors...")
        for i in tqdm(range(num_nodes), desc="Computing local Top-K means"):
            node_similarities = similarities_matrix[i, :]
            top_k_indices = np.argpartition(node_similarities, -k_top_sim)[-k_top_sim:]
            top_k_similarities = node_similarities[top_k_indices]
            top_k_similarities = top_k_similarities[np.isfinite(top_k_similarities)]
            if len(top_k_similarities) > 0:
                 local_topk_mean_sims.append(np.mean(top_k_similarities))
            else:
                 local_topk_mean_sims.append(0)

        if not local_topk_mean_sims:
             print("  Error: Could not calculate any local top-k mean similarities.")
             return torch.zeros((2, 0), dtype=torch.long), None

        global_avg_topk_mean_sim = np.mean(local_topk_mean_sims)
        final_threshold = global_avg_topk_mean_sim * threshold_factor
        print(f"  Global average of Top-{k_top_sim} mean similarities: {global_avg_topk_mean_sim:.4f}")
        print(f"  Final similarity threshold (applied factor {threshold_factor:.2f}): {final_threshold:.4f}")

        # Build edges based on the final threshold
        np.fill_diagonal(similarities_matrix, 1.0) # Restore diagonal for checking pairs
        rows, cols, data = [], [], []
        adj = similarities_matrix > final_threshold
        np.fill_diagonal(adj, False)
        indices_rows, indices_cols = np.where(adj)
        rows = indices_rows.tolist()
        cols = indices_cols.tolist()
        data = similarities_matrix[rows, cols].tolist()

        print(f"  Found {len(rows)} edges above threshold.")

        if len(rows) == 0:
            print(f"Warning: Threshold {final_threshold:.4f} too high, no edges created. Adding fallback edges...")
            # Fallback: Connect each node to its single most similar neighbor
            np.fill_diagonal(similarities_matrix, -np.inf)
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])): continue
                most_similar_neighbor = np.argmax(similarities_matrix[i])
                rows.append(i)
                cols.append(most_similar_neighbor)
                data.append(similarities_matrix[i, most_similar_neighbor])

        if not rows:
             print("Error: Could not create any edges.")
             return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        print(f"  Created {edge_index.shape[1]} edges.")
        return edge_index, edge_attr

    

    def _analyze_graph(self) -> None:
        """
        Analyze the graph and compute comprehensive metrics, providing detailed output.
        """
        print("\n" + "=" * 60)
        print("       Detailed Graph Analysis & Metrics")
        print("=" * 60)

        # --- Initial Checks ---
        if self.graph_data is None:
            print("Error: Graph data not built. Cannot analyze.")
            self.graph_metrics = {"error": "Graph data not available"}
            return
        if not hasattr(self.graph_data, 'edge_index') or self.graph_data.edge_index is None:
            print("Warning: Graph data lacks 'edge_index'. Cannot analyze edges.")
            num_nodes = self.graph_data.num_nodes if hasattr(self.graph_data, 'num_nodes') else 0
            self.graph_metrics = {
                "nodes": num_nodes, "edges": 0, "avg_degree": 0.0,
                "warning": "Missing edge_index, edge analysis skipped."
            }
            # Add sampling info if available
            if hasattr(self, 'sample_unlabeled'):
                self.graph_metrics["sampling_info"] = {
                    "sampled_unlabeled": self.sample_unlabeled,
                    "unlabeled_sample_factor": self.unlabeled_sample_factor if self.sample_unlabeled else None,
                    "num_labeled_original": len(self.selected_labeled_indices_original) if self.selected_labeled_indices_original is not None else 'N/A',
                    "num_unlabeled_sampled_original": len(self.selected_unlabeled_indices_original) if self.selected_unlabeled_indices_original is not None else 'N/A',
                }
            return # Stop analysis

        num_nodes = self.graph_data.num_nodes
        num_edges = self.graph_data.edge_index.shape[1]

        # Handle case of no edges separately
        if num_edges == 0:
            print("Graph has 0 edges. Reporting basic node info and skipping edge-based analysis.")
            self.graph_metrics = {
                "nodes": num_nodes,
                "edges": 0,
                "avg_degree": 0.0,
                "nodes_train": self.graph_data.train_mask.sum().item(),
                "nodes_test": self.graph_data.test_mask.sum().item(),
                "nodes_labeled": self.graph_data.labeled_mask.sum().item(),
                "isolated_nodes": num_nodes, # All nodes are isolated
                "connected_components": num_nodes, # Each node is a component
                "largest_component_size": 1 if num_nodes > 0 else 0,
                "density": 0.0,
                "avg_clustering": 0.0,
                "assortativity": None,
                "avg_path_length": None,
                "homophily_ratio": None,
                "edge_types": {"train-train": 0, "train-test": 0, "test-test": 0},
                "class_connectivity": {"ff": 0, "rr": 0, "fr": 0},
                "test_node_connectivity": {
                    "test_nodes_total": self.graph_data.test_mask.sum().item(),
                    "test_nodes_isolated": self.graph_data.test_mask.sum().item(),
                    "test_nodes_only_to_test": 0,
                    "test_nodes_to_train": 0,
                },
                "sampling_info": { # Add sampling info
                    "sampled_unlabeled": self.sample_unlabeled,
                    "unlabeled_sample_factor": self.unlabeled_sample_factor if self.sample_unlabeled else None,
                    "num_labeled_original": len(self.selected_labeled_indices_original) if self.selected_labeled_indices_original is not None else 'N/A',
                    "num_unlabeled_sampled_original": len(self.selected_unlabeled_indices_original) if self.selected_unlabeled_indices_original is not None else 'N/A',
                },
                "status": "No edges found"
            }
            # Print summary for no-edge case
            print(f"\n--- Basic Info ---")
            print(f"  Nodes: {num_nodes}")
            print(f"  Edges: 0")
            print(f"  Train Nodes: {self.graph_metrics['nodes_train']}")
            print(f"  Test Nodes: {self.graph_metrics['nodes_test']}")
            print(f"  Labeled Nodes: {self.graph_metrics['nodes_labeled']}")
            if self.sample_unlabeled:
                print(f"  Unlabeled Sampling: Enabled (Factor={self.unlabeled_sample_factor})")
            else:
                print(f"  Unlabeled Sampling: Disabled (Used all)")
            print("-" * 60)
            return # Stop analysis

        # --- Basic Graph Stats ---
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0.0
        self.graph_metrics = {
            "nodes": num_nodes,
            "edges": num_edges, # Number of directed edges from edge_index
            "avg_degree": avg_degree, # Based on directed edges
            "nodes_train": self.graph_data.train_mask.sum().item(),
            "nodes_test": self.graph_data.test_mask.sum().item(),
            "nodes_labeled": self.graph_data.labeled_mask.sum().item(),
            "sampling_info": {
                "sampled_unlabeled": self.sample_unlabeled,
                "unlabeled_sample_factor": self.unlabeled_sample_factor if self.sample_unlabeled else None,
                "num_labeled_original": len(self.selected_labeled_indices_original) if self.selected_labeled_indices_original is not None else 'N/A',
                "num_unlabeled_sampled_original": len(self.selected_unlabeled_indices_original) if self.selected_unlabeled_indices_original is not None else 'N/A',
            }
        }
        print(f"\n--- Basic Info ---")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges (directed): {num_edges}")
        print(f"  Avg Degree (directed): {avg_degree:.2f}")
        print(f"  Train Nodes: {self.graph_metrics['nodes_train']}")
        print(f"  Test Nodes: {self.graph_metrics['nodes_test']}")
        print(f"  Labeled Nodes: {self.graph_metrics['nodes_labeled']}")
        if self.sample_unlabeled:
            print(f"  Unlabeled Sampling: Enabled (Factor={self.unlabeled_sample_factor})")
        else:
            print(f"  Unlabeled Sampling: Disabled (Used all)")


        # --- Convert to NetworkX for Advanced Analysis (Undirected View) ---
        G_nx = None
        G_undirected = None
        networkx_analysis_possible = False
        try:
            print("\n--- NetworkX Analysis Setup ---")
            print("  Converting to NetworkX graph (undirected)...")
            # Ensure data is on CPU for NetworkX conversion
            edge_index_cpu = self.graph_data.edge_index.cpu().long()
            # Create a minimal Data object on CPU for conversion
            data_cpu = Data(edge_index=edge_index_cpu, num_nodes=num_nodes)
            G_undirected = to_networkx(data_cpu, to_undirected=True)
            num_undirected_edges = G_undirected.number_of_edges()
            print(f"  NetworkX undirected graph created: {G_undirected.number_of_nodes()} nodes, {num_undirected_edges} edges.")
            networkx_analysis_possible = True
        except Exception as e:
            print(f"  Warning: Error converting graph to NetworkX: {e}. Skipping NetworkX-based analysis.")
            self.graph_metrics.update({
                "networkx_analysis_skipped": True, "networkx_error": str(e)
            })

        # --- Degree Analysis ---
        # Calculate degrees using original directed edges for potential in/out degree if needed later
        # For general stats, undirected degree from NetworkX is common and simpler here
        degrees_np = np.zeros(num_nodes, dtype=int)
        if networkx_analysis_possible:
            degrees_list = [d for n, d in G_undirected.degree()]
            if len(degrees_list) == num_nodes:
                degrees_np = np.array(degrees_list)
            else:
                print("  Warning: Degree list length mismatch from NetworkX. Calculating degree manually.")
                # Manual calculation if NetworkX fails unexpectedly
                np.add.at(degrees_np, edge_index_cpu[0].numpy(), 1) # Out-degree
                # If considering undirected for stats: np.add.at(degrees_np, edge_index_cpu[1].numpy(), 1) # Add in-degree? Depends on definition. Let's stick to undirected view from G_undirected if possible.

        isolated_nodes = int(np.sum(degrees_np == 0))
        self.graph_metrics["isolated_nodes"] = isolated_nodes
        min_degree = int(np.min(degrees_np)) if len(degrees_np) > 0 else 0
        max_degree = int(np.max(degrees_np)) if len(degrees_np) > 0 else 0
        mean_degree_stat = float(np.mean(degrees_np)) if len(degrees_np) > 0 else 0.0
        median_degree = float(np.median(degrees_np)) if len(degrees_np) > 0 else 0.0
        std_degree = float(np.std(degrees_np)) if len(degrees_np) > 0 else 0.0
        self.graph_metrics["degree_stats"] = {
            "min": min_degree, "max": max_degree, "mean": mean_degree_stat,
            "median": median_degree, "std": std_degree,
        }
        print(f"\n--- Degree Analysis (Based on Undirected Edges) ---")
        print(f"  Min Degree: {min_degree}")
        print(f"  Max Degree: {max_degree}")
        print(f"  Mean Degree: {mean_degree_stat:.2f}")
        print(f"  Median Degree: {median_degree:.2f}")
        print(f"  Std Dev Degree: {std_degree:.2f}")
        print(f"  Isolated Nodes (degree 0): {isolated_nodes} ({isolated_nodes/num_nodes*100:.1f}% of nodes)")

        # --- Connectivity Analysis ---
        num_components = None
        largest_component_nodes = None
        avg_path_length = None
        if networkx_analysis_possible:
            print(f"\n--- Connectivity Analysis ---")
            is_connected = nx.is_connected(G_undirected)
            num_components = nx.number_connected_components(G_undirected)
            print(f"  Is Graph Connected? {'Yes' if is_connected else 'No'}")
            print(f"  Number of Connected Components: {num_components}")
            if not is_connected and num_components > 0:
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                largest_component_nodes = len(largest_cc)
                # Create subgraph for LCC analysis
                G_lcc = G_undirected.subgraph(largest_cc).copy()
                lcc_edges = G_lcc.number_of_edges()
                print(f"  Largest Component: {largest_component_nodes} nodes ({largest_component_nodes/num_nodes*100:.1f}%), {lcc_edges} edges")
                # Calculate Avg Path Length only for the LCC if graph is disconnected
                try:
                    avg_path_length = nx.average_shortest_path_length(G_lcc)
                    print(f"  Avg Shortest Path Length (Largest Comp.): {avg_path_length:.2f}")
                except Exception as e:
                    print(f"  Warning: Could not calculate avg shortest path length for LCC: {e}")
                    avg_path_length = None
            elif is_connected:
                largest_component_nodes = num_nodes
                print(f"  Largest Component: {num_nodes} nodes (100.0%)")
                # Calculate Avg Path Length for the whole graph
                try:
                    avg_path_length = nx.average_shortest_path_length(G_undirected)
                    print(f"  Avg Shortest Path Length (Overall): {avg_path_length:.2f}")
                except Exception as e:
                    print(f"  Warning: Could not calculate avg shortest path length: {e}")
                    avg_path_length = None

            self.graph_metrics.update({
                "connected_components": num_components,
                "largest_component_size": largest_component_nodes,
                "avg_path_length": avg_path_length
            })

        # --- Graph Structure Metrics (NetworkX-based) ---
        density = None
        avg_clustering = None
        assortativity = None
        if networkx_analysis_possible:
            print(f"\n--- Graph Structure Metrics ---")
            # Density
            try:
                density = nx.density(G_undirected)
                print(f"  Density: {density:.4f} (Ratio of actual edges to potential edges)")
                self.graph_metrics["density"] = density
            except Exception as e: print(f"  Warning: Could not calculate density: {e}")

            # Clustering Coefficient
            try:
                avg_clustering = nx.average_clustering(G_undirected)
                print(f"  Avg Clustering Coefficient: {avg_clustering:.4f} (Tendency of nodes to cluster together)")
                self.graph_metrics["avg_clustering"] = avg_clustering
            except Exception as e: print(f"  Warning: Could not calculate clustering: {e}")

            # Assortativity (Degree Correlation)
            try:
                # Handle potential division by zero or constant degrees
                if max_degree > min_degree : # Avoid issues if all nodes have same degree
                    assortativity = nx.degree_assortativity_coefficient(G_undirected)
                    print(f"  Degree Assortativity: {assortativity:.4f} (+ve: high-deg connects high-deg, -ve: high-deg connects low-deg)")
                    self.graph_metrics["assortativity"] = assortativity
                else:
                    print("  Degree Assortativity: N/A (degrees are constant or graph too small)")
                    self.graph_metrics["assortativity"] = None
            except Exception as e: print(f"  Warning: Could not calculate assortativity: {e}")


        # --- Edge Type Distribution ---
        # (Using directed edges as originally intended)
        print(f"\n--- Edge Type Distribution (Directed Edges) ---")
        edge_types = {"train-train": 0, "train-test": 0, "test-test": 0, "other": 0}
        train_mask_np = self.graph_data.train_mask.cpu().numpy()
        test_mask_np = self.graph_data.test_mask.cpu().numpy()
        edge_index_cpu = self.graph_data.edge_index.cpu() # Ensure CPU

        for i in tqdm(range(num_edges), desc="Analyzing edge types", leave=False):
            source = edge_index_cpu[0, i].item()
            target = edge_index_cpu[1, i].item()
            s_is_train, t_is_train = train_mask_np[source], train_mask_np[target]
            s_is_test, t_is_test = test_mask_np[source], test_mask_np[target]

            if s_is_train and t_is_train: edge_types["train-train"] += 1
            elif (s_is_train and t_is_test) or (s_is_test and t_is_train): edge_types["train-test"] += 1
            elif s_is_test and t_is_test: edge_types["test-test"] += 1
            else: edge_types["other"] += 1 # Should ideally be 0 if masks cover all nodes

        total_reported_edges = sum(edge_types.values())
        print(f"  Total Directed Edges Analyzed: {total_reported_edges}")
        if total_reported_edges > 0:
            for edge_type, count in edge_types.items():
                percentage = (count / total_reported_edges * 100)
                print(f"    - {edge_type:<12}: {count:>8} ({percentage:>5.1f}%)")
        else: print("    No edges to analyze.")
        if edge_types["other"] > 0: print("    Warning: Found 'other' edge types - check mask definitions.")
        self.graph_metrics["edge_types"] = edge_types

        # --- Class Connectivity & Homophily ---
        print(f"\n--- Class Connectivity & Homophily (Undirected View) ---")
        y_np = self.graph_data.y.cpu().numpy()
        fake_to_fake, real_to_real, fake_to_real = 0, 0, 0
        # Iterate through UNDIRECTED edges for homophily
        if networkx_analysis_possible and num_undirected_edges > 0:
            for u, v in tqdm(G_undirected.edges(), desc="Analyzing class connectivity", total=num_undirected_edges, leave=False):
                l1, l2 = y_np[u], y_np[v]
                if l1 == 1 and l2 == 1: fake_to_fake += 1
                elif l1 == 0 and l2 == 0: real_to_real += 1
                else: fake_to_real += 1

            homophilic_edges = fake_to_fake + real_to_real
            homophily_ratio = homophilic_edges / num_undirected_edges if num_undirected_edges > 0 else 0.0

            print(f"  Total Undirected Edges Analyzed: {num_undirected_edges}")
            print(f"    - Fake -> Fake: {fake_to_fake:>8} ({fake_to_fake/num_undirected_edges*100:>5.1f}%)")
            print(f"    - Real -> Real: {real_to_real:>8} ({real_to_real/num_undirected_edges*100:>5.1f}%)")
            print(f"    - Fake <-> Real: {fake_to_real:>7} ({fake_to_real/num_undirected_edges*100:>5.1f}%)")
            print(f"  Homophily Ratio: {homophily_ratio:.4f} (Fraction of edges connecting nodes of same class)")
            self.graph_metrics["class_connectivity"] = {"ff": fake_to_fake, "rr": real_to_real, "fr": fake_to_real}
            self.graph_metrics["homophily_ratio"] = homophily_ratio
        else:
            print("  Skipping class connectivity analysis (NetworkX graph not available or no edges).")
            self.graph_metrics["class_connectivity"] = None
            self.graph_metrics["homophily_ratio"] = None


        # --- Test Node Connectivity Analysis (NEW) ---
        print(f"\n--- Test Node Connectivity ---")
        test_nodes_indices = torch.where(self.graph_data.test_mask)[0].cpu().numpy()
        num_test_nodes = len(test_nodes_indices)
        print(f"  Total Test Nodes in Graph: {num_test_nodes}")

        if num_test_nodes > 0 and networkx_analysis_possible:
            test_nodes_isolated = 0
            test_nodes_only_to_test = 0
            test_nodes_to_train = 0

            # Get train node indices in the current graph
            train_nodes_in_graph_indices = set(torch.where(self.graph_data.train_mask)[0].cpu().numpy())

            for node_idx in test_nodes_indices:
                if G_undirected.degree(node_idx) == 0:
                    test_nodes_isolated += 1
                    continue # Skip neighbor check if isolated

                neighbors = set(G_undirected.neighbors(node_idx))
                # Check if any neighbor is a training node
                has_train_neighbor = any(neighbor in train_nodes_in_graph_indices for neighbor in neighbors)

                if has_train_neighbor:
                    test_nodes_to_train += 1
                else:
                    # If no train neighbor, all neighbors must be test nodes
                    test_nodes_only_to_test += 1

            print(f"  Test Nodes Isolated (degree 0): {test_nodes_isolated} ({test_nodes_isolated/num_test_nodes*100:.1f}%)")
            print(f"  Test Nodes Connected ONLY to other Test Nodes: {test_nodes_only_to_test} ({test_nodes_only_to_test/num_test_nodes*100:.1f}%)")
            print(f"  Test Nodes Connected to at least one Train Node: {test_nodes_to_train} ({test_nodes_to_train/num_test_nodes*100:.1f}%)")
            self.graph_metrics["test_node_connectivity"] = {
                "test_nodes_total": num_test_nodes,
                "test_nodes_isolated": test_nodes_isolated,
                "test_nodes_only_to_test": test_nodes_only_to_test,
                "test_nodes_to_train": test_nodes_to_train,
            }
        elif num_test_nodes == 0:
            print("  No test nodes in the graph.")
            self.graph_metrics["test_node_connectivity"] = {"test_nodes_total": 0}
        else:
            print("  Skipping test node connectivity analysis (NetworkX graph not available).")
            self.graph_metrics["test_node_connectivity"] = None

        # --- Optional: Power-Law Fit (requires scipy) ---
        if networkx_analysis_possible: # Only if degrees were calculated reliably
            try:
                from scipy import stats
                print(f"\n--- Power-Law Fit Analysis ---")
                valid_degrees = degrees_np[degrees_np > 0]
                if len(valid_degrees) > 1:
                    unique_degs, counts = np.unique(valid_degrees, return_counts=True)
                    if len(unique_degs) > 1:
                        log_degrees = np.log10(unique_degs)
                        log_counts = np.log10(counts)
                        slope, intercept, r_value, p_value, std_err = stats.linregress(log_degrees, log_counts)
                        exponent = -slope
                        r_squared = r_value**2
                        is_power_law = p_value < 0.05 and r_squared > 0.6 # Adjusted criteria slightly
                        print(f"  Exponent (alpha): {exponent:.2f}")
                        print(f"  R-squared: {r_squared:.4f}")
                        print(f"  P-value: {p_value:.4f}")
                        print(f"  Follows Power-Law (approx criteria)? {'Yes' if is_power_law else 'No'}")
                        self.graph_metrics["power_law"] = {
                            "exponent": exponent, "r_squared": r_squared, "p_value": p_value,
                            "std_err": std_err, "is_power_law": is_power_law,
                        }
                    else: print("  Not enough unique degrees > 0 to fit power-law.")
                else: print("  No nodes with degree > 0 found.")
            except ImportError:
                print("  Skipping Power-Law analysis (scipy not installed).")
                self.graph_metrics["power_law"] = {"error": "scipy not installed"}
            except Exception as e:
                print(f"  Warning: Error during power-law analysis: {e}")
                self.graph_metrics["power_law"] = {"error": str(e)}

        print("=" * 60)
        print("      End of Graph Analysis Report")
        print("=" * 60 + "\n")
    

    def save_graph(self) -> str:
        """Save the graph and analysis results."""
        if self.graph_data is None:
            raise ValueError("Graph must be built before saving")

        # --- Generate graph name ---
        edge_policy_name = self.edge_policy
        edge_param_str = ""
        if self.edge_policy in ["knn", "mutual_knn"]: edge_param_str = str(self.k_neighbors)
        elif self.edge_policy == "local_threshold": edge_param_str = f"{self.local_threshold_factor:.2f}".replace('.', 'p')
        elif self.edge_policy == "global_threshold": edge_param_str = f"{self.alpha:.2f}".replace('.', 'p')
        elif self.edge_policy == "quantile": edge_param_str = f"{self.quantile_p:.1f}".replace('.', 'p')
        elif self.edge_policy == "topk_mean": edge_param_str = f"k{self.k_top_sim}_f{self.local_threshold_factor:.2f}".replace('.', 'p')
        else: edge_policy_name, edge_param_str = "unknown", "NA"

        # Add sampling info to filename if sampling was used
        sampling_suffix = ""
        if self.sample_unlabeled:
            sampling_suffix = f"_smpf{self.unlabeled_sample_factor}"

        graph_name = f"{self.k_shot}_shot_{self.embedding_type}_{edge_policy_name}_{edge_param_str}{sampling_suffix}"
        # --- End filename generation ---

        # Save graph data
        graph_path = os.path.join(self.output_dir, f"{graph_name}.pt")
        # Ensure data is on CPU before saving for better compatibility
        cpu_graph_data = self.graph_data.cpu()
        torch.save(cpu_graph_data, graph_path)

        # Save graph metrics
        if self.graph_metrics:
            # Ensure metrics are serializable
            def default_serializer(obj):
                if isinstance(obj, (np.integer, np.floating)): return obj.item()
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (torch.Tensor)): return obj.tolist() # Should be converted before, but just in case
                # Add more types if needed, or raise error for unhandled
                try: return json.JSONEncoder().encode(obj) # Fallback attempt
                except TypeError: return str(obj) # Last resort: convert to string

            metrics_path = os.path.join(self.output_dir, f"{graph_name}_metrics.json")
            try:
                 with open(metrics_path, "w") as f:
                     json.dump(self.graph_metrics, f, indent=2, default=default_serializer)
                 print(f"Graph metrics saved to {metrics_path}")
            except Exception as e:
                 print(f"Error saving metrics JSON: {e}")
                 print(f"Metrics data: {self.graph_metrics}") # Print problematic data


        # Save selected indices for reference (both labeled and sampled unlabeled)
        indices_data = {
            "k_shot": int(self.k_shot),
            "seed": int(self.seed),
            "sampled_unlabeled": self.sample_unlabeled,
        }
        if self.selected_labeled_indices_original is not None:
             indices_data["labeled_indices_original"] = [int(i) for i in self.selected_labeled_indices_original]
             # Add label distribution for labeled nodes
             train_labels = self.dataset["train"]["label"]
             label_dist = {}
             for idx in self.selected_labeled_indices_original:
                  label = train_labels[idx]
                  label_dist[label] = label_dist.get(label, 0) + 1
             indices_data["labeled_label_distribution"] = {int(k): int(v) for k, v in label_dist.items()}

        if self.sample_unlabeled and self.selected_unlabeled_indices_original is not None:
             indices_data["unlabeled_sample_factor"] = int(self.unlabeled_sample_factor)
             indices_data["unlabeled_indices_original_sampled"] = [int(i) for i in self.selected_unlabeled_indices_original]

        indices_path = os.path.join(self.output_dir, f"{graph_name}_indices.json")
        with open(indices_path, "w") as f:
            json.dump(indices_data, f, indent=2)
        print(f"Selected indices info saved to {indices_path}")

        print(f"\nGraph saved to {graph_path}")

        # Plot graph if requested
        if self.plot:
            try:
                self.visualize_graph(graph_name)
            except Exception as e:
                print(f"Warning: Error visualizing graph: {e}")

        return graph_path

    def visualize_graph(self, graph_name: str, max_nodes: int = 1000) -> None:
        """Visualize the graph using NetworkX."""
        # (Keep the improved visualize_graph function from the previous response)
        if self.graph_data is None: raise ValueError("Graph must be built")
        num_nodes_actual = self.graph_data.num_nodes
        nodes_to_plot = min(num_nodes_actual, max_nodes)
        print(f"Preparing graph visualization ({nodes_to_plot} nodes)...")

        # Convert to NetworkX, potentially subgraphing
        G_nx = None
        if num_nodes_actual > max_nodes:
            print(f"  Graph is large ({num_nodes_actual} nodes). Visualizing subgraph of {max_nodes} nodes.")
            # IMPORTANT: Subgraphing needs careful handling of masks if colors depend on original roles
            # For simplicity, let's color based on the final masks of the subgraph nodes
            node_indices = torch.arange(max_nodes)
            subgraph_data = self.graph_data.subgraph(node_indices.to(self.device)).cpu()
            G_nx = to_networkx(subgraph_data, to_undirected=True)
            masks_viz = {'train': subgraph_data.train_mask, 'test': subgraph_data.test_mask, 'labeled': subgraph_data.labeled_mask}
            num_nodes_viz = max_nodes
        else:
            G_nx = to_networkx(self.graph_data.cpu(), to_undirected=True)
            masks_viz = {'train': self.graph_data.train_mask.cpu(), 'test': self.graph_data.test_mask.cpu(), 'labeled': self.graph_data.labeled_mask.cpu()}
            num_nodes_viz = num_nodes_actual

        if G_nx is None or G_nx.number_of_nodes() == 0:
            print("  Cannot visualize empty or invalid graph.")
            return
        if G_nx.number_of_edges() == 0: print("  Warning: Graph has no edges. Visualization will only show nodes.")

        # Set node colors
        node_colors = ['gray'] * num_nodes_viz
        for i in range(num_nodes_viz):
            if masks_viz['labeled'][i]: node_colors[i] = 'green' # Labeled overrides train
            elif masks_viz['train'][i]: node_colors[i] = 'blue'
            elif masks_viz['test'][i]: node_colors[i] = 'red'

        plt.figure(figsize=(14, 12))
        print("  Calculating layout...")
        pos = nx.spring_layout(G_nx, k=0.3 / np.sqrt(num_nodes_viz) if num_nodes_viz > 0 else 0.1, iterations=50, seed=self.seed) if num_nodes_viz <= 500 else nx.random_layout(G_nx, seed=self.seed)

        print("  Drawing graph...")
        node_size = max(5, 4000 / num_nodes_viz) if num_nodes_viz > 0 else 20
        edge_width = 0.5
        nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, node_size=node_size, alpha=0.8)
        if G_nx.number_of_edges() > 0: nx.draw_networkx_edges(G_nx, pos, edge_color="gray", width=edge_width, alpha=0.5)

        legend_elements = [Patch(facecolor="green", label="Labeled (Train)"), Patch(facecolor="blue", label="Unlabeled Train"), Patch(facecolor="red", label="Unlabeled Test")]
        plt.legend(handles=legend_elements, loc="upper right", fontsize='medium')

        title = f"{self.dataset_name.capitalize()} Graph ({self.edge_policy.replace('_', ' ').title()})"
        if num_nodes_actual > max_nodes: title += f"\n(Showing {max_nodes} of {num_nodes_actual} nodes)"
        sample_info_str = f"Sampled={self.sample_unlabeled}" + (f", Factor={self.unlabeled_sample_factor}" if self.sample_unlabeled else "")
        subtitle = f"{self.k_shot}-shot | Embed: {self.embedding_type.upper()} | Edges: {G_nx.number_of_edges()} | {sample_info_str}"
        plt.title(f"{title}\n{subtitle}", fontsize='large')
        plt.axis('off')
        plot_path = os.path.join(self.plot_dir, f"{graph_name}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Graph visualization saved to {plot_path}")


    def run_pipeline(self) -> Optional[Data]:
        """Run the complete graph building pipeline."""
        self.load_dataset()
        graph_data = self.build_graph()
        if graph_data is not None and graph_data.num_nodes > 0:
             self.save_graph()
             return graph_data
        else:
             print("Graph building failed or resulted in an empty graph. No graph saved.")
             return None


def parse_arguments():
    """Parse command-line arguments with helpful descriptions."""
    parser = ArgumentParser(description="Build graph for few-shot fake news detection")

    # dataset arguments
    parser.add_argument("--dataset_name", type=str, default="politifact", choices=["politifact", "gossipcop"], help="Dataset to use")
    parser.add_argument("--k_shot", type=int, default=8, choices=list(range(3, 21)), help="Number of labeled samples per class (3-20)")

    # graph construction arguments
    parser.add_argument("--edge_policy", type=str, default=DEFAULT_EDGE_POLICY, choices=["knn", "mutual_knn", "local_threshold", "global_threshold", "quantile", "topk_mean"], help="Edge construction policy")
    parser.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS, help="K for (Mutual) KNN policy")
    parser.add_argument("--local_threshold_factor", type=float, default=DEFAULT_LOCAL_THRESHOLD_FACTOR, help=f"Factor for 'local_threshold' policy (default: {DEFAULT_LOCAL_THRESHOLD_FACTOR})")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"Alpha (mean+alpha*std) for 'global_threshold' policy (default: {DEFAULT_ALPHA})")
    parser.add_argument("--quantile_p", type=float, default=DEFAULT_QUANTILE_P, help=f"Percentile for 'quantile' policy (default: {DEFAULT_QUANTILE_P})")
    parser.add_argument("--k_top_sim", type=int, default=10, help="K for 'topk_mean' policy (default: 10)")
    # Note: topk_mean reuses --local_threshold_factor for its final threshold adjustment factor

    # Unlabeled Node Sampling Arguments (NEW)
    parser.add_argument("--sample_unlabeled", action='store_true', default=False, help="Enable sampling of unlabeled training nodes.")
    parser.add_argument("--unlabeled_sample_factor", type=int, default=DEFAULT_UNLABELED_SAMPLE_FACTOR, help=f"Factor M to sample M*2*k unlabeled training nodes (default: {DEFAULT_UNLABELED_SAMPLE_FACTOR}). Used if --sample_unlabeled.")

    # news embedding type
    parser.add_argument("--embedding_type", type=str, default=DEFAULT_EMBEDDING_TYPE, choices=["bert", "roberta", "combined"], help=f"Type of embeddings to use (default: {DEFAULT_EMBEDDING_TYPE})")

    # output arguments
    parser.add_argument("--output_dir", type=str, default=GRAPH_DIR, help=f"Directory to save graphs (default: {GRAPH_DIR})")
    parser.add_argument("--plot", action="store_true", help="Enable graph visualization")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")

    return parser.parse_args()


def main() -> None:
    """Main function to run the graph building pipeline."""
    args = parse_arguments()
    set_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    print("\n" + "=" * 60)
    print("Fake News Detection - Graph Building Pipeline")
    print("=" * 60)
    print(f"Dataset:          {args.dataset_name}")
    print(f"Embedding type:   {args.embedding_type}")
    print(f"Few-shot k:       {args.k_shot} per class")
    print("-" * 20 + " Edge Policy " + "-" * 20)
    print(f"Policy:           {args.edge_policy}")
    if args.edge_policy in ["knn", "mutual_knn"]: print(f"K neighbors:      {args.k_neighbors}")
    elif args.edge_policy == "local_threshold": print(f"Local Factor:     {args.local_threshold_factor}")
    elif args.edge_policy == "global_threshold": print(f"Alpha:            {args.alpha}")
    elif args.edge_policy == "quantile": print(f"Quantile p:       {args.quantile_p}")
    elif args.edge_policy == "topk_mean":
        print(f"K Top Sim:        {args.k_top_sim}")
        print(f"Threshold Factor: {args.local_threshold_factor}") # Reused factor
    print("-" * 20 + " Node Sampling " + "-" * 20)
    print(f"Sample Unlabeled: {args.sample_unlabeled}")
    if args.sample_unlabeled: print(f"Sample Factor(M): {args.unlabeled_sample_factor} (target M*2*k nodes)")
    else: print(f"Sample Factor(M): N/A (using all unlabeled train nodes)")
    print("-" * 20 + " Output & Settings " + "-" * 20)
    print(f"Output directory: {args.output_dir}")
    print(f"Plot:             {args.plot}")
    print(f"Seed:             {args.seed}")
    print(f"Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available(): print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")

    builder = GraphBuilder(
        dataset_name=args.dataset_name, k_shot=args.k_shot,
        edge_policy=args.edge_policy, k_neighbors=args.k_neighbors,
        local_threshold_factor=args.local_threshold_factor, alpha=args.alpha,
        quantile_p=args.quantile_p, k_top_sim=args.k_top_sim,
        sample_unlabeled=args.sample_unlabeled, # Pass new arg
        unlabeled_sample_factor=args.unlabeled_sample_factor, # Pass new arg
        output_dir=args.output_dir, plot=args.plot, seed=args.seed,
        embedding_type=args.embedding_type,
    )

    graph_data = builder.run_pipeline()

    print("\n" + "=" * 60)
    print("Graph Building Complete")
    print("=" * 60)
    if graph_data and hasattr(graph_data, 'num_nodes') and graph_data.num_nodes > 0:
        graph_file_name = f"{args.k_shot}_shot_{args.embedding_type}_{builder.edge_policy}_..." # Example structure
        print(f"Graph saved in:   {builder.output_dir}/")
        print(f"  Nodes:          {graph_data.num_nodes}")
        print(f"  Edges:          {graph_data.num_edges}")
        print(f"  Features:       {graph_data.num_features}")
        print(f"  Train nodes:    {graph_data.train_mask.sum().item()}")
        print(f"  Test nodes:     {graph_data.test_mask.sum().item()}")
        print(f"  Labeled nodes:  {graph_data.labeled_mask.sum().item()}")
        print("\nNext Steps:")
        print(f"  1. Train a GNN model, e.g.:")
        print(f"     python train_graph.py --graph_path {os.path.join(builder.output_dir, '<graph_file_name>.pt')}")
    else:
        print("Graph building failed or resulted in an empty/invalid graph.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()