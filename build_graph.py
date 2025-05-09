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
        sample_unlabeled: bool = False, 
        unlabeled_sample_factor: int = DEFAULT_UNLABELED_SAMPLE_FACTOR,
        output_dir: str = GRAPH_DIR,
        plot: bool = False,
        seed: int = DEFAULT_SEED,
        embedding_type: str = DEFAULT_EMBEDDING_TYPE,
        device: str = None,
    ):
        """Initialize the GraphBuilder with configuration parameters."""
        ## Dataset configuration
        self.dataset_name = dataset_name.lower()
        self.embedding_type = embedding_type.lower()
        ## K-Shot
        self.k_shot = k_shot
        ## Edge Policy
        self.edge_policy = edge_policy
        self.k_neighbors = k_neighbors
        self.k_top_sim = k_top_sim
        self.local_threshold_factor = local_threshold_factor
        self.alpha = alpha
        self.quantile_p = quantile_p
        ## Sampling
        self.sample_unlabeled = sample_unlabeled
        self.unlabeled_sample_factor = unlabeled_sample_factor
        self.plot = plot
        self.seed = seed
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        np.random.seed(self.seed)
        
        # Setup directory paths
        self.output_dir = os.path.join(output_dir, self.dataset_name)
        self.plot_dir = os.path.join(PLOT_DIR, self.dataset_name)

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        if self.plot: os.makedirs(self.plot_dir, exist_ok=True)

        # Initialize components
        self.dataset = None
        self.graph_data = None
        self.graph_metrics = {}

        # Selected Indices
        self.selected_train_labeled_indices = None
        self.selected_train_unlabeled_indices = None
        self.selected_test_indices = None

    def load_dataset(self) -> None:
        """Load dataset from HuggingFace and prepare it for graph construction."""
        print(f"Loading dataset '{self.dataset_name}' with '{self.embedding_type}' embeddings...")
        hf_dataset_name = f"LittleFish-Coder/Fake_News_{self.dataset_name}"
        dataset = load_dataset(hf_dataset_name, download_mode="reuse_cache_if_exists", cache_dir="dataset")
        self.dataset = {"train": dataset["train"], "test": dataset["test"]}
        self.train_size = len(dataset["train"])
        self.test_size = len(dataset["test"])
        unique_labels = set(dataset["train"]["label"])  # 0: real, 1: fake
        self.num_classes = len(unique_labels)   # 2
        self.total_labeled_size = self.k_shot * self.num_classes
        print(f"\nOriginal dataset size: Train={self.train_size}, Test={self.test_size}")
        print(f"Labeled set: {self.k_shot}-shot * {self.num_classes} classes = {self.total_labeled_size} total labeled nodes")

    def build_graph(self) -> Data:
        """
        Build a graph including both nodes and edges.
        Pipeline:
        1. Build empty graph
            - train_labeled_nodes (train_labeled_mask from training set)
            - train_unlabeled_nodes (train_unlabeled_mask from training set)
            - test_nodes (test_mask from test set)
        2. Build edges (based on edge policy)
        3. Update graph data with edges
        """
        self.build_empty_graph() # Build nodes (potentially with sampling)

        print(f"\nBuilding graph edges using {self.edge_policy} policy...")
        embeddings = self.graph_data.x.cpu().numpy()

        # --- Edge Building Logic ---
        if self.edge_policy == "knn":
            edge_index, edge_attr = self._build_knn_edges(embeddings, self.k_neighbors)
        elif self.edge_policy == "mutual_knn":
            edge_index, edge_attr = self._build_mutual_knn_edges(embeddings, self.k_neighbors)
        elif self.edge_policy == "local_threshold":
            edge_index, edge_attr = self._build_local_threshold_edges(embeddings, self.local_threshold_factor)
        elif self.edge_policy == "global_threshold":
            edge_index, edge_attr = self._build_global_threshold_edges(embeddings, self.alpha)
        elif self.edge_policy == "quantile":
            edge_index, edge_attr = self._build_quantile_edges(embeddings, self.quantile_p)
        elif self.edge_policy == "topk_mean":
            edge_index, edge_attr = self._build_topk_mean_edges(embeddings, self.k_top_sim, self.local_threshold_factor)
        else: 
            raise ValueError(f"Edge policy '{self.edge_policy}' not found.")

        # symmetrize edge_index and edge_attr (for undirected graphs)
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        self.graph_data.edge_index = edge_index.to(self.device)
        if edge_attr is not None:
            self.graph_data.edge_attr = edge_attr.to(self.device)
        else:
            self.graph_data.edge_attr = None
        self.graph_data.num_edges = self.graph_data.edge_index.shape[1]
        print(f"Graph edges built: {self.graph_data.num_edges} edges created")

        self._analyze_graph()

        return self.graph_data

    def build_empty_graph(self) -> Optional[Data]:
        """
        Build the graph nodes and masks for three node types:
        - train_labeled_nodes: k-shot per class from training set (used for supervision)
        - train_unlabeled_nodes: remaining training nodes (no label for training, only structure)
        - test_nodes: all test set nodes (no label for training, prediction target)
        If self.sample_unlabeled is True, only sample M*2*k train_unlabeled_nodes (M = unlabeled_sample_factor).
        Returns a PyG Data object with proper masks.
        """
        print(f"\nBuilding graph nodes using {self.embedding_type} embeddings...")
        embedding_field = f"{self.embedding_type}_embeddings"
        train_data = self.dataset["train"]
        test_data = self.dataset["test"]

        # 1. Sample k-shot labeled nodes from train set
        print(f"Sampling {self.k_shot}-shot labeled nodes with seed={self.seed}")
        train_labeled_indices, _ = sample_k_shot(train_data, self.k_shot, self.seed)
        train_labeled_indices = np.array(train_labeled_indices)
        self.selected_train_labeled_indices = train_labeled_indices
        print(f"  Selected {len(train_labeled_indices)} labeled nodes: {train_labeled_indices} ...")

        # 2. Get train_unlabeled_nodes (all train nodes not in train_labeled_indices)
        all_train_indices = np.arange(len(train_data))
        train_unlabeled_indices = np.setdiff1d(all_train_indices, train_labeled_indices, assume_unique=True)

        # --- Sample train_unlabeled_nodes if required ---
        if self.sample_unlabeled:
            num_to_sample = min(self.unlabeled_sample_factor * 2 * self.k_shot, len(train_unlabeled_indices))
            print(f"Sampling {num_to_sample} train_unlabeled_nodes (factor={self.unlabeled_sample_factor}, 2*k={2*self.k_shot}) from {len(train_unlabeled_indices)} available.")
            train_unlabeled_indices = np.random.choice(train_unlabeled_indices, size=num_to_sample, replace=False)
        self.selected_train_unlabeled_indices = train_unlabeled_indices
        print(f"  Selected {len(train_unlabeled_indices)} unlabeled train nodes.")

        # 3. Get all test node indices
        test_indices = np.arange(len(test_data))
        self.selected_test_indices = test_indices
        print(f"  Test nodes: {len(test_indices)}")

        # 4. Extract embeddings and labels for each group
        train_labeled_emb = np.array(train_data.select(train_labeled_indices.tolist())[embedding_field])
        train_labeled_label = np.array(train_data.select(train_labeled_indices.tolist())["label"])
        train_unlabeled_emb = np.array(train_data.select(train_unlabeled_indices.tolist())[embedding_field])
        train_unlabeled_label = np.array(train_data.select(train_unlabeled_indices.tolist())["label"])
        test_emb = np.array(test_data[embedding_field])
        test_label = np.array(test_data["label"])

        # 5. Concatenate all nodes
        x = torch.tensor(np.concatenate([train_labeled_emb, train_unlabeled_emb, test_emb]), dtype=torch.float)
        y = torch.tensor(np.concatenate([train_labeled_label, train_unlabeled_label, test_label]), dtype=torch.long)

        # 6. Build masks
        num_train_labeled = len(train_labeled_indices)      # train labeled nodes
        num_train_unlabeled = len(train_unlabeled_indices)  # train unlabeled nodes
        num_test = len(test_indices)                        # test nodes
        num_nodes = num_train_labeled + num_train_unlabeled + num_test  # all nodes

        train_labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_unlabeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_labeled_mask[:num_train_labeled] = True
        train_unlabeled_mask[num_train_labeled:num_train_labeled+num_train_unlabeled] = True
        test_mask[num_train_labeled+num_train_unlabeled:] = True

        # 7. Create Data object
        graph_data = Data(
            x=x.to(self.device),
            y=y.to(self.device),
            train_labeled_mask=train_labeled_mask.to(self.device),
            train_unlabeled_mask=train_unlabeled_mask.to(self.device),
            test_mask=test_mask.to(self.device),
            edge_index=torch.zeros((2, 0), dtype=torch.long).to(self.device),
            num_nodes=num_nodes,
            num_features=x.shape[1],
        )
        graph_data.edge_attr = None

        self.graph_data = graph_data

        # 8. Print summary
        print(f"\nGraph nodes built:")
        print(f"  Total nodes: {num_nodes}")
        print(f"    - Train Labeled: {num_train_labeled}")
        print(f"    - Train Unlabeled: {num_train_unlabeled}")
        print(f"    - Test: {num_test}")
        print(f"  Node features: {x.shape[1]}")

        # Sanity checks
        if train_labeled_mask.sum().item() != num_train_labeled:
            print(f"Warning: train_labeled_mask count mismatch.")
        if train_unlabeled_mask.sum().item() != num_train_unlabeled:
            print(f"Warning: train_unlabeled_mask count mismatch.")
        if test_mask.sum().item() != num_test:
            print(f"Warning: test_mask count mismatch.")
        if (train_labeled_mask.sum() + train_unlabeled_mask.sum() + test_mask.sum()).item() != num_nodes:
            print(f"Warning: mask sum != num_nodes")

        return graph_data

    def _build_knn_edges(self, embeddings: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Analyze the graph and compute comprehensive metrics, using only train_labeled_mask, train_unlabeled_mask, and test_mask for all node/edge type stats.
        """
        print("\n" + "=" * 60)
        print("       Detailed Graph Analysis & Metrics")
        print("=" * 60)

        num_nodes = self.graph_data.num_nodes
        num_edges = self.graph_data.edge_index.shape[1]

        # --- Node counts ---
        num_train_labeled = self.graph_data.train_labeled_mask.sum().item()
        num_train_unlabeled = self.graph_data.train_unlabeled_mask.sum().item()
        num_test = self.graph_data.test_mask.sum().item()

        # --- Basic Graph Stats ---
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0.0
        self.graph_metrics = {
            "nodes": num_nodes,
            "edges": num_edges,
            "avg_degree": avg_degree,
            "nodes_train_labeled": num_train_labeled,
            "nodes_train_unlabeled": num_train_unlabeled,
            "nodes_test": num_test,
            "sampling_info": {
                "sampled_unlabeled": self.sample_unlabeled,
                "unlabeled_sample_factor": self.unlabeled_sample_factor if self.sample_unlabeled else None,
                "num_labeled_original": len(self.selected_train_labeled_indices) if self.selected_train_labeled_indices is not None else 'N/A',
                "num_unlabeled_sampled_original": len(self.selected_train_unlabeled_indices) if self.selected_train_unlabeled_indices is not None else 'N/A',
            }
        }
        print(f"\n--- Basic Info ---")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges (directed): {num_edges}")
        print(f"  Avg Degree (directed): {avg_degree:.2f}")
        print(f"  Train Labeled Nodes: {num_train_labeled}")
        print(f"  Train Unlabeled Nodes: {num_train_unlabeled}")
        print(f"  Test Nodes: {num_test}")
        if self.sample_unlabeled:
            print(f"  Unlabeled Sampling: Enabled (Factor={self.unlabeled_sample_factor})")
        else:
            print(f"  Unlabeled Sampling: Disabled (Used all)")

        # --- Convert to NetworkX for Advanced Analysis (Undirected View) ---
        G_undirected = None
        networkx_analysis_possible = False
        try:
            print("\n--- NetworkX Analysis Setup ---")
            print("  Converting to NetworkX graph (undirected)...")
            edge_index_cpu = self.graph_data.edge_index.cpu().long()
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
        degrees_np = np.zeros(num_nodes, dtype=int)
        if networkx_analysis_possible:
            degrees_list = [d for n, d in G_undirected.degree()]
            if len(degrees_list) == num_nodes:
                degrees_np = np.array(degrees_list)
            else:
                print("  Warning: Degree list length mismatch from NetworkX. Calculating degree manually.")
                np.add.at(degrees_np, edge_index_cpu[0].numpy(), 1)

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
                G_lcc = G_undirected.subgraph(largest_cc).copy()
                lcc_edges = G_lcc.number_of_edges()
                print(f"  Largest Component: {largest_component_nodes} nodes ({largest_component_nodes/num_nodes*100:.1f}%), {lcc_edges} edges")
                try:
                    avg_path_length = nx.average_shortest_path_length(G_lcc)
                    print(f"  Avg Shortest Path Length (Largest Comp.): {avg_path_length:.2f}")
                except Exception as e:
                    print(f"  Warning: Could not calculate avg shortest path length for LCC: {e}")
                    avg_path_length = None
            elif is_connected:
                largest_component_nodes = num_nodes
                print(f"  Largest Component: {num_nodes} nodes (100.0%)")
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
            try:
                density = nx.density(G_undirected)
                print(f"  Density: {density:.4f} (Ratio of actual edges to potential edges)")
                self.graph_metrics["density"] = density
            except Exception as e: print(f"  Warning: Could not calculate density: {e}")
            try:
                avg_clustering = nx.average_clustering(G_undirected)
                print(f"  Avg Clustering Coefficient: {avg_clustering:.4f} (Tendency of nodes to cluster together)")
                self.graph_metrics["avg_clustering"] = avg_clustering
            except Exception as e: print(f"  Warning: Could not calculate clustering: {e}")
            try:
                if max_degree > min_degree:
                    assortativity = nx.degree_assortativity_coefficient(G_undirected)
                    print(f"  Degree Assortativity: {assortativity:.4f} (+ve: high-deg connects high-deg, -ve: high-deg connects low-deg)")
                    self.graph_metrics["assortativity"] = assortativity
                else:
                    print("  Degree Assortativity: N/A (degrees are constant or graph too small)")
                    self.graph_metrics["assortativity"] = None
            except Exception as e: print(f"  Warning: Could not calculate assortativity: {e}")

        # --- Edge Type Distribution ---
        print(f"\n--- Edge Type Distribution (Directed Edges) ---")
        edge_types = {"train_labeled-train_labeled": 0, "train_labeled-train_unlabeled": 0, "train_labeled-test": 0, "train_unlabeled-train_unlabeled": 0, "train_unlabeled-test": 0, "test-test": 0, "other": 0}
        train_labeled_mask_np = self.graph_data.train_labeled_mask.cpu().numpy()
        train_unlabeled_mask_np = self.graph_data.train_unlabeled_mask.cpu().numpy()
        test_mask_np = self.graph_data.test_mask.cpu().numpy()
        edge_index_cpu = self.graph_data.edge_index.cpu()
        for i in tqdm(range(num_edges), desc="Analyzing edge types", leave=False):
            source = edge_index_cpu[0, i].item()
            target = edge_index_cpu[1, i].item()
            s_labeled, t_labeled = train_labeled_mask_np[source], train_labeled_mask_np[target]
            s_unlabeled, t_unlabeled = train_unlabeled_mask_np[source], train_unlabeled_mask_np[target]
            s_test, t_test = test_mask_np[source], test_mask_np[target]
            if s_labeled and t_labeled:
                edge_types["train_labeled-train_labeled"] += 1
            elif (s_labeled and t_unlabeled) or (s_unlabeled and t_labeled):
                edge_types["train_labeled-train_unlabeled"] += 1
            elif (s_labeled and t_test) or (s_test and t_labeled):
                edge_types["train_labeled-test"] += 1
            elif s_unlabeled and t_unlabeled:
                edge_types["train_unlabeled-train_unlabeled"] += 1
            elif (s_unlabeled and t_test) or (s_test and t_unlabeled):
                edge_types["train_unlabeled-test"] += 1
            elif s_test and t_test:
                edge_types["test-test"] += 1
            else:
                edge_types["other"] += 1
        total_reported_edges = sum(edge_types.values())
        print(f"  Total Directed Edges Analyzed: {total_reported_edges}")
        for edge_type, count in edge_types.items():
            percentage = (count / total_reported_edges * 100) if total_reported_edges > 0 else 0.0
            print(f"    - {edge_type:<28}: {count:>8} ({percentage:>5.1f}%)")
        if edge_types["other"] > 0:
            print("    Warning: Found 'other' edge types - check mask definitions.")
        self.graph_metrics["edge_types"] = edge_types

        # --- Class Connectivity & Homophily ---
        print(f"\n--- Class Connectivity & Homophily (Undirected View) ---")
        y_np = self.graph_data.y.cpu().numpy()
        fake_to_fake, real_to_real, fake_to_real = 0, 0, 0
        if networkx_analysis_possible and G_undirected is not None and G_undirected.number_of_edges() > 0:
            for u, v in tqdm(G_undirected.edges(), desc="Analyzing class connectivity", total=G_undirected.number_of_edges(), leave=False):
                l1, l2 = y_np[u], y_np[v]
                if l1 == 1 and l2 == 1: fake_to_fake += 1
                elif l1 == 0 and l2 == 0: real_to_real += 1
                else: fake_to_real += 1
            homophilic_edges = fake_to_fake + real_to_real
            homophily_ratio = homophilic_edges / G_undirected.number_of_edges() if G_undirected.number_of_edges() > 0 else 0.0
            print(f"  Total Undirected Edges Analyzed: {G_undirected.number_of_edges()}")
            print(f"    - Fake -> Fake: {fake_to_fake:>8} ({fake_to_fake/G_undirected.number_of_edges()*100:>5.1f}%)")
            print(f"    - Real -> Real: {real_to_real:>8} ({real_to_real/G_undirected.number_of_edges()*100:>5.1f}%)")
            print(f"    - Fake <-> Real: {fake_to_real:>7} ({fake_to_real/G_undirected.number_of_edges()*100:>5.1f}%)")
            print(f"  Homophily Ratio: {homophily_ratio:.4f} (Fraction of edges connecting nodes of same class)")
            self.graph_metrics["class_connectivity"] = {"ff": fake_to_fake, "rr": real_to_real, "fr": fake_to_real}
            self.graph_metrics["homophily_ratio"] = homophily_ratio
        else:
            print("  Skipping class connectivity analysis (NetworkX graph not available or no edges).")
            self.graph_metrics["class_connectivity"] = None
            self.graph_metrics["homophily_ratio"] = None

        # --- Test Node Connectivity Analysis ---
        print(f"\n--- Test Node Connectivity ---")
        test_nodes_indices = torch.where(self.graph_data.test_mask)[0].cpu().numpy()
        num_test_nodes = len(test_nodes_indices)
        print(f"  Total Test Nodes in Graph: {num_test_nodes}")
        if num_test_nodes > 0 and networkx_analysis_possible:
            test_nodes_isolated = 0
            test_nodes_only_to_test = 0
            test_nodes_to_train = 0
            train_nodes_in_graph_indices = set(np.where(self.graph_data.train_labeled_mask.cpu().numpy() | self.graph_data.train_unlabeled_mask.cpu().numpy())[0])
            for node_idx in test_nodes_indices:
                if G_undirected.degree(node_idx) == 0:
                    test_nodes_isolated += 1
                    continue
                neighbors = set(G_undirected.neighbors(node_idx))
                has_train_neighbor = any(neighbor in train_nodes_in_graph_indices for neighbor in neighbors)
                if has_train_neighbor:
                    test_nodes_to_train += 1
                else:
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

        # --- Power-Law Fit (optional) ---
        if networkx_analysis_possible:
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
                        is_power_law = p_value < 0.05 and r_squared > 0.6
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
        if self.selected_train_labeled_indices is not None:
             indices_data["train_labeled_indices"] = [int(i) for i in self.selected_train_labeled_indices]
             # Add label distribution for labeled nodes
             train_labels_full = self.dataset["train"]["label"] # Use full train labels
             label_dist = {}
             for idx in self.selected_train_labeled_indices:
                  label = train_labels_full[idx] # Access from full list
                  label_dist[label] = label_dist.get(label, 0) + 1
             indices_data["train_labeled_label_distribution"] = {int(k): int(v) for k, v in label_dist.items()}

        if self.sample_unlabeled and self.selected_train_unlabeled_indices is not None:
             indices_data["unlabeled_sample_factor"] = int(self.unlabeled_sample_factor)
             indices_data["train_unlabeled_indices"] = [int(i) for i in self.selected_train_unlabeled_indices]

        indices_path = os.path.join(self.output_dir, f"{graph_name}_indices.json")
        with open(indices_path, "w") as f:
            json.dump(indices_data, f, indent=2)
        print(f"Selected indices info saved to {indices_path}")
        print(f"Graph saved to {graph_path}")

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
            masks_viz = {
                'train_labeled': subgraph_data.train_labeled_mask,
                'train_unlabeled': subgraph_data.train_unlabeled_mask, # Add this
                'train': subgraph_data.train_mask, # Keep overall train mask if needed for other logic
                'test': subgraph_data.test_mask
            }
            num_nodes_viz = max_nodes
        else:
            G_nx = to_networkx(self.graph_data.cpu(), to_undirected=True)
            masks_viz = {
                'train_labeled': self.graph_data.train_labeled_mask.cpu(),
                'train_unlabeled': self.graph_data.train_unlabeled_mask.cpu(),
                'train': self.graph_data.train_mask.cpu(),
                'test': self.graph_data.test_mask.cpu()
            }
            num_nodes_viz = num_nodes_actual

        if G_nx is None or G_nx.number_of_nodes() == 0:
            print("  Cannot visualize empty or invalid graph.")
            return
        if G_nx.number_of_edges() == 0: print("  Warning: Graph has no edges. Visualization will only show nodes.")

        # Set node colors
        node_colors = ['gray'] * num_nodes_viz
        for i in range(num_nodes_viz):
            if masks_viz['train_labeled'][i]: node_colors[i] = 'green' # Labeled (Train)
            elif masks_viz['train_unlabeled'][i]: node_colors[i] = 'blue'  # Unlabeled Train
            elif masks_viz['test'][i]: node_colors[i] = 'red'    # Test
            # Fallback if a node is in train_mask but not in train_labeled or train_unlabeled (should not happen if masks are correct)
            elif masks_viz['train'][i]: node_colors[i] = 'skyblue' # A different shade for general train if specific masks don't cover

        plt.figure(figsize=(14, 12))
        print("  Calculating layout...")
        pos = nx.spring_layout(G_nx, k=0.3 / np.sqrt(num_nodes_viz) if num_nodes_viz > 0 else 0.1, iterations=50, seed=self.seed) if num_nodes_viz <= 500 else nx.random_layout(G_nx, seed=self.seed)

        print("  Drawing graph...")
        node_size = max(5, 4000 / num_nodes_viz) if num_nodes_viz > 0 else 20
        edge_width = 0.5
        nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, node_size=node_size, alpha=0.8)
        if G_nx.number_of_edges() > 0: nx.draw_networkx_edges(G_nx, pos, edge_color="gray", width=edge_width, alpha=0.5)

        legend_elements = [
            Patch(facecolor="green", label="Train Labeled"),
            Patch(facecolor="blue", label="Train Unlabeled"),
            Patch(facecolor="red", label="Test")
        ]
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
        self.save_graph()
        return graph_data


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
    
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print("Fake News Detection - Graph Building Pipeline")
    print("=" * 60)
    print(f"Dataset:          {args.dataset_name}")
    print(f"Embedding type:   {args.embedding_type}")
    print(f"Few-shot k:       {args.k_shot} per class")
    print("-" * 20 + " Edge Policy " + "-" * 20)
    print(f"Policy:           {args.edge_policy}")
    if args.edge_policy == "knn": 
        print(f"K neighbors:      {args.k_neighbors}")
    elif args.edge_policy == "mutual_knn": 
        print(f"K neighbors:      {args.k_neighbors}")
    elif args.edge_policy == "local_threshold": 
        print(f"Local Factor:     {args.local_threshold_factor}")
    elif args.edge_policy == "global_threshold": 
        print(f"Alpha:            {args.alpha}")
    elif args.edge_policy == "quantile": 
        print(f"Quantile p:       {args.quantile_p}")
    elif args.edge_policy == "topk_mean":
        print(f"K Top Sim:        {args.k_top_sim}")
        print(f"Threshold Factor: {args.local_threshold_factor}")
    print("-" * 20 + " Node Sampling " + "-" * 20)
    print(f"Sample Unlabeled: {args.sample_unlabeled}")
    if args.sample_unlabeled: 
        print(f"Sample Factor(M): {args.unlabeled_sample_factor} (target M*2*k nodes)")
    else: 
        print(f"Sample Factor(M): N/A (using all unlabeled train nodes)")
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
        sample_unlabeled=args.sample_unlabeled,
        unlabeled_sample_factor=args.unlabeled_sample_factor,
        output_dir=args.output_dir, plot=args.plot, seed=args.seed,
        embedding_type=args.embedding_type,
    )

    graph_data = builder.run_pipeline()

    print("\n" + "=" * 60)
    print("Graph Building Complete")
    print("=" * 60)
    print(f"Graph saved in:   {builder.output_dir}/")
    print(f"  Nodes:          {graph_data.num_nodes}")
    print(f"  Edges:          {graph_data.num_edges}")
    print(f"  Features:       {graph_data.num_features}")
    print(f"  Train nodes (total):    {graph_data.train_labeled_mask.sum().item() + graph_data.train_unlabeled_mask.sum().item()}")
    print(f"  Train Labeled nodes:  {graph_data.train_labeled_mask.sum().item()}")
    print(f"  Train Unlabeled nodes: {graph_data.train_unlabeled_mask.sum().item()}")
    print(f"  Test nodes:     {graph_data.test_mask.sum().item()}")
    print("\nNext Steps:")
    print(f"Train a GNN model, e.g.:")
    print(f"python train_graph.py --graph_path {os.path.join(builder.output_dir, '<graph_file_name>.pt')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()