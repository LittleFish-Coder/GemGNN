import os
import gc
import json
import numpy as np
import torch
import networkx as nx
from typing import Dict, Tuple, Optional, List, Union, Any
from datasets import load_dataset, DatasetDict, Dataset, Features, Sequence, Value, Array2D, Array3D, Array4D, Array5D
from sklearn.metrics import pairwise_distances
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
from tqdm.auto import tqdm
from argparse import ArgumentParser
from utils.sample_k_shot import sample_k_shot
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


# --- Constants ---
DEFAULT_K_SHOT = 8
DEFAULT_DATASET_NAME = "politifact"
DEFAULT_EMBEDDING_TYPE = "roberta" # Default embedding for news nodes
# --- Edge Policies Parameters ---
DEFAULT_EDGE_POLICY = "knn"             # For news-news edges
DEFAULT_K_NEIGHBORS = 5                 # For knn edge policy
DEFAULT_LOCAL_THRESHOLD_FACTOR = 1.0    # for local_threshold edge policy
DEFAULT_ALPHA = 0.1                     # for global_threshold edge policy
DEFAULT_QUANTILE_P = 95.0               # for quantile edge policy
DEFAULT_K_TOP_SIM = 10                  # for topk_mean edge policy
# --- Unlabeled Node Sampling Parameters ---
DEFAULT_SAMPLE_UNLABELED_FACTOR = 10    # for unlabeled node sampling
DEFAULT_MULTI_VIEW = 0                  # for multi-view edge policy
DEFAULT_INTERACTION_EDGE_MODE = "edge_attr" # for interaction edge policy
# --- Graphs and Plots Directories ---
GRAPH_DIR = "graphs_hetero"
PLOT_DIR = "plots_hetero"
DEFAULT_SEED = 42

# --- Utility Functions ---
def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


# --- HeteroGraphBuilder Class ---
class HeteroGraphBuilder:
    """
    Builds heterogeneous graph datasets ('news', 'interaction')
    for few-shot fake news detection.
    """

    def __init__(
        self,
        dataset_name: str,
        k_shot: int,
        embedding_type: str = DEFAULT_EMBEDDING_TYPE, # News node embedding
        edge_policy: str = DEFAULT_EDGE_POLICY, # For news-news edges
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        k_top_sim: int = DEFAULT_K_TOP_SIM,
        local_threshold_factor: float = DEFAULT_LOCAL_THRESHOLD_FACTOR,
        alpha: float = DEFAULT_ALPHA,
        quantile_p: float = DEFAULT_QUANTILE_P,
        sample_unlabeled_factor: int = DEFAULT_SAMPLE_UNLABELED_FACTOR,
        output_dir: str = GRAPH_DIR,
        plot: bool = False,
        seed: int = DEFAULT_SEED,
        device: str = None,
        pseudo_label: bool = False,
        pseudo_label_cache_path: str = None,
        multi_view: int = DEFAULT_MULTI_VIEW,
        enable_dissimilar: bool = False,
        partial_unlabeled: bool = False,
        interaction_embedding_field: str = "interaction_embeddings_list",
        interaction_tone_field: str = "interaction_tones_list",
        interaction_edge_mode: str = DEFAULT_INTERACTION_EDGE_MODE,
    ):
        """Initialize the HeteroGraphBuilder."""
        self.dataset_name = dataset_name.lower()
        self.k_shot = k_shot
        self.embedding_type = embedding_type
        self.text_embedding_field = f"{embedding_type}_embeddings"
        self.interaction_embedding_field = interaction_embedding_field
        self.interaction_tone_field = interaction_tone_field
        self.interaction_edge_mode = interaction_edge_mode
        self.edge_policy = edge_policy
        self.k_neighbors = k_neighbors
        self.k_top_sim = k_top_sim
        self.local_threshold_factor = local_threshold_factor
        self.alpha = alpha
        self.quantile_p = quantile_p
        self.multi_view = multi_view

        ## Sampling
        self.sample_unlabeled_factor = sample_unlabeled_factor
        self.enable_dissimilar = enable_dissimilar
        self.partial_unlabeled = partial_unlabeled
        self.pseudo_label = pseudo_label
        if pseudo_label_cache_path:
            self.pseudo_label_cache_path = pseudo_label_cache_path
        else:
            self.pseudo_label_cache_path = f"utils/pseudo_label_cache_{self.dataset_name}.json"
        
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


        # Initialize state
        self.dataset = None
        self.graph_data = None
        self.graph_metrics = {}

        # Selected Indices
        self.selected_train_labeled_indices = None
        self.selected_train_unlabeled_indices = None
        self.selected_test_indices = None
        self.news_orig_to_new_idx = None # Mapping for mask creation

        self.graph_metrics = {} # Store analysis results

        self.tone2id = {}

    def _tone2id(self, tone):
        if tone not in self.tone2id:
            self.tone2id[tone] = len(self.tone2id)
        return self.tone2id[tone]

    def load_dataset(self) -> None:
        """Load dataset and perform initial checks."""
        print(f"Loading dataset '{self.dataset_name}' with '{self.embedding_type}' embeddings...")
        hf_dataset_name = f"LittleFish-Coder/Fake_News_{self.dataset_name}"
        dataset = load_dataset(hf_dataset_name, download_mode="reuse_cache_if_exists", cache_dir="dataset")
        self.dataset = {"train": dataset["train"], "test": dataset["test"]}
        self.train_data = self.dataset["train"]
        self.test_data = self.dataset['test']
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)
        unique_labels = set(self.train_data['label']) | set(self.test_data['label']) # 0: real, 1: fake
        self.num_classes = len(unique_labels)   # 2
        self.total_labeled_size = self.k_shot * self.num_classes
        print(f"\nOriginal dataset size: Train={self.train_size}, Test={self.test_size}")
        print(f"  Detected Labels: {unique_labels} ({self.num_classes} classes)")
        print(f"  Labeled set: {self.k_shot}-shot * {self.num_classes} classes = {self.total_labeled_size} total labeled nodes")
        print(f"  Required Fields Check: OK ({self.text_embedding_field}, {self.interaction_embedding_field}, label)")

    

    def _build_knn_edges(self, embeddings: np.ndarray, k: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: 
            return (torch.zeros((2,0), dtype=torch.long), None, 
                    torch.zeros((2,0), dtype=torch.long), None)
        k = min(k, num_nodes - 1) # Adjust k if it's too large
        if k <= 0: 
            return (torch.zeros((2,0), dtype=torch.long), None,
                    torch.zeros((2,0), dtype=torch.long), None)
        
        print(f"    Building KNN graph (k={k}) for 'news'-'news' edges...")
        try:
            distances = pairwise_distances(embeddings, metric="cosine", n_jobs=-1) # Use multiple cores if available
        except Exception as e:
            print(f"      Error calculating pairwise distances: {e}. Using single core.")
            distances = pairwise_distances(embeddings, metric="cosine")

        # For similar edges (nearest neighbors)
        sim_rows, sim_cols, sim_data = [], [], []
        # For dissimilar edges (farthest neighbors)
        dis_rows, dis_cols, dis_data = [], [], []
        
        for i in tqdm(range(num_nodes), desc=f"      Finding {k} nearest/farthest neighbors", leave=False, ncols=100):
            dist_i = distances[i].copy()
            dist_i[i] = np.inf  # Exclude self
            
            # Find k nearest neighbors (most similar)
            nearest_indices = np.argpartition(dist_i, k)[:k]
            valid_nearest = nearest_indices[np.isfinite(dist_i[nearest_indices])]
            
            # Find k farthest neighbors (most dissimilar)
            valid_distances = dist_i[np.isfinite(dist_i)]
            valid_indices = np.arange(len(dist_i))[np.isfinite(dist_i)]
            if len(valid_distances) > k:
                farthest_k_indices = np.argpartition(valid_distances, -k)[-k:]
                valid_farthest = valid_indices[farthest_k_indices]
            else:
                valid_farthest = valid_indices
            
            min_valid = min(len(valid_nearest), len(valid_farthest))
            if min_valid > 0:
                # Similar edges
                for j in valid_nearest[:min_valid]:
                    sim_rows.append(i)
                    sim_cols.append(j)
                    sim = 1.0 - distances[i, j]  # Convert to similarity [0,1]
                    sim_data.append(sim)  # Keep positive for similar edges
                
                # Dissimilar edges
                for j in valid_farthest[:min_valid]:
                    dis_rows.append(i)
                    dis_cols.append(j)
                    sim = 1.0 - distances[i, j]  # Convert to similarity [0,1]
                    dis_data.append(-sim)  # Make negative for dissimilar edges

        # Create similar edge tensors
        if not sim_rows: 
            sim_edge_index = torch.zeros((2, 0), dtype=torch.long)
            sim_edge_attr = None
        else:
            sim_edge_index = torch.tensor(np.vstack((sim_rows, sim_cols)), dtype=torch.long)
            sim_edge_attr = torch.tensor(sim_data, dtype=torch.float).unsqueeze(1)

        # Create dissimilar edge tensors
        if not dis_rows:
            dis_edge_index = torch.zeros((2, 0), dtype=torch.long)
            dis_edge_attr = None
        else:
            dis_edge_index = torch.tensor(np.vstack((dis_rows, dis_cols)), dtype=torch.long)
            dis_edge_attr = torch.tensor(dis_data, dtype=torch.float).unsqueeze(1)

        print(f"      Created {sim_edge_index.shape[1]} similar edges and {dis_edge_index.shape[1]} dissimilar edges.")
        return sim_edge_index, sim_edge_attr, dis_edge_index, dis_edge_attr

    def _build_local_threshold_edges(self, embeddings: np.ndarray, local_threshold_factor: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None
        print(f"    Building LOCAL threshold graph (factor={local_threshold_factor:.2f}) for 'news'-'news' edges...")

        # Normalize embeddings
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_norms = np.maximum(norms, 1e-10) # Avoid division by zero
        normalized_embeddings = embeddings / valid_norms

        rows, cols, data = [], [], []
        try:
             print("      Calculating full similarity matrix...")
             similarities_matrix = cosine_similarity(normalized_embeddings) # More robust for cosine
             np.fill_diagonal(similarities_matrix, -1.0) # Exclude self-loops from threshold calculation
        except MemoryError:
             print("      Error: MemoryError calculating similarity matrix. Cannot proceed with local_threshold.")
             return torch.zeros((2,0), dtype=torch.long), None
        except Exception as e:
            print(f"      Error calculating similarity matrix: {e}. Cannot proceed.")
            return torch.zeros((2,0), dtype=torch.long), None

        print("      Calculating local thresholds and building edges...")
        for i in tqdm(range(num_nodes), desc="      Computing local threshold edges", leave=False, ncols=100):
            node_similarities = similarities_matrix[i, :].copy() # Use copy
            positive_similarities = node_similarities[node_similarities > 0] # Consider only positive similarities for mean

            if len(positive_similarities) > 0:
                mean_similarity = np.mean(positive_similarities)
                node_threshold = mean_similarity * local_threshold_factor
                above_threshold_indices = np.where(node_similarities > node_threshold)[0]
                for target_idx in above_threshold_indices:
                    if target_idx == i: continue # Explicitly skip self-loops if any slip through
                    rows.append(i)
                    cols.append(target_idx)
                    data.append(max(0.0, node_similarities[target_idx])) # Ensure non-negative

        if not rows: # Fallback if no edges were created
            print(f"      Warning: No edges created with local_threshold_factor={local_threshold_factor}. Adding fallback (top-1)...")
            np.fill_diagonal(similarities_matrix, -np.inf) # Ensure self is not chosen
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])): continue # Skip if all similarities are -inf
                top_idx = np.argmax(similarities_matrix[i])
                if similarities_matrix[i, top_idx] > -np.inf : # Check if a valid neighbor was found
                    rows.append(i)
                    cols.append(top_idx)
                    data.append(max(0.0, similarities_matrix[i, top_idx]))

        if not rows: return torch.zeros((2, 0), dtype=torch.long), None
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        print(f"      Created {edge_index.shape[1]} local threshold edges.")
        return edge_index, edge_attr

    def _build_global_threshold_edges(self, embeddings: np.ndarray, alpha: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None
        print(f"    Building GLOBAL threshold graph (alpha={alpha:.2f}) for 'news'-'news' edges...")

        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_norms = np.maximum(norms, 1e-10)
        normalized_embeddings = embeddings / valid_norms

        print("      Computing global similarity statistics...")
        use_full_matrix_for_edges = False
        similarities_matrix = None # Initialize

        try:
            similarities_matrix = cosine_similarity(normalized_embeddings) # More robust
            mask = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
            all_similarities = similarities_matrix[mask]
            print(f"        Calculated {len(all_similarities)} unique non-self similarities for stats.")
            use_full_matrix_for_edges = True
        except MemoryError:
            print("        MemoryError calculating full matrix for stats, sampling pairs...")
            target_samples = min(max(500000, num_nodes * 20), 5000000)
            pairs_to_sample = min(target_samples, num_nodes * (num_nodes - 1) // 2 if num_nodes > 1 else 0)
            if pairs_to_sample <= 0:
                all_similarities = np.array([])
            else:
                rng = np.random.default_rng(self.seed) # Use instance seed
                sampled_indices_i = rng.integers(0, num_nodes, size=pairs_to_sample)
                sampled_indices_j = rng.integers(0, num_nodes, size=pairs_to_sample)
                valid_pair_mask = sampled_indices_i != sampled_indices_j
                sampled_indices_i = sampled_indices_i[valid_pair_mask]
                sampled_indices_j = sampled_indices_j[valid_pair_mask]
                sims = np.sum(normalized_embeddings[sampled_indices_i] * normalized_embeddings[sampled_indices_j], axis=1)
                all_similarities = sims
            print(f"        Sampled {len(all_similarities)} pairs for statistics.")
        except Exception as e:
            print(f"      Error calculating similarity statistics: {e}. Cannot proceed.")
            return torch.zeros((2,0), dtype=torch.long), None


        all_similarities = all_similarities[all_similarities > -0.999] # Filter out potential noise or extreme negatives

        if len(all_similarities) == 0:
            print("      Warning: No valid similarities found for global statistics. Setting default threshold 0.5.")
            global_threshold = 0.5
            sim_mean, sim_std = 0.0, 0.0
        else:
            sim_mean = np.mean(all_similarities)
            sim_std = np.std(all_similarities)
            global_threshold = sim_mean + alpha * sim_std
        print(f"        Similarity stats: mean={sim_mean:.4f}, std={sim_std:.4f}")
        print(f"        Global threshold calculated: {global_threshold:.4f}")

        rows, cols, data = [], [], []
        print("      Building edges using global threshold...")
        if use_full_matrix_for_edges and similarities_matrix is not None:
            adj = similarities_matrix > global_threshold
            np.fill_diagonal(adj, False) # Ensure no self-loops
            edge_indices = np.argwhere(adj)
            rows = edge_indices[:, 0].tolist()
            cols = edge_indices[:, 1].tolist()
            if rows: data = similarities_matrix[rows, cols].tolist()
        else:
            for i in tqdm(range(num_nodes), desc="      Building global threshold edges", leave=False, ncols=100):
                current_sims = cosine_similarity(normalized_embeddings[i:i+1], normalized_embeddings)[0]
                current_sims[i] = -1.0 # Exclude self
                above_threshold_indices = np.where(current_sims > global_threshold)[0]
                for target_idx in above_threshold_indices:
                    rows.append(i)
                    cols.append(target_idx)
                    data.append(max(0.0, current_sims[target_idx]))

        if not rows: # Fallback if no edges
            print(f"      Warning: Global threshold {global_threshold:.4f} too high, no edges created. Adding fallback (top-1)...")
            if similarities_matrix is None: # Need to compute it if not available
                 try:
                     similarities_matrix = cosine_similarity(normalized_embeddings)
                 except Exception as e_fb_sim:
                     print(f"        Error computing similarities for fallback: {e_fb_sim}")
                     return torch.zeros((2,0), dtype=torch.long), None

            np.fill_diagonal(similarities_matrix, -np.inf) # Ensure self is not chosen
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])): continue
                top_idx = np.argmax(similarities_matrix[i])
                if similarities_matrix[i, top_idx] > -np.inf:
                    rows.append(i)
                    cols.append(top_idx)
                    data.append(max(0.0, similarities_matrix[i, top_idx]))

        if not rows: return torch.zeros((2, 0), dtype=torch.long), None
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        print(f"      Created {edge_index.shape[1]} global threshold edges.")
        return edge_index, edge_attr

    def _build_mutual_knn_edges(self, embeddings: np.ndarray, k: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None
        k = min(k, num_nodes - 1)
        if k <= 0: return torch.zeros((2,0), dtype=torch.long), None
        print(f"    Building MUTUAL KNN graph (k={k}) for 'news'-'news' edges...")
        try:
            distances = pairwise_distances(embeddings, metric="cosine", n_jobs=-1)
        except Exception as e:
            print(f"      Error calculating pairwise distances: {e}. Using single core.")
            distances = pairwise_distances(embeddings, metric="cosine")

        rows, cols, data = [], [], []
        all_neighbors = {} # Store k-nearest neighbors for each node

        for i in tqdm(range(num_nodes), desc=f"      Finding {k} nearest neighbors (pass 1)", leave=False, ncols=100):
            dist_i = distances[i].copy()
            dist_i[i] = np.inf
            indices = np.argpartition(dist_i, k)[:k]
            valid_indices = indices[np.isfinite(dist_i[indices])]
            all_neighbors[i] = set(valid_indices)


        for i in tqdm(range(num_nodes), desc=f"      Checking {k} mutual neighbors (pass 2)", leave=False, ncols=100):
            if i not in all_neighbors: continue
            for j in all_neighbors[i]:
                if j not in all_neighbors: continue
                if i in all_neighbors[j]:
                    rows.append(i)
                    cols.append(j)
                    sim = 1.0 - distances[i, j]
                    data.append(max(0.0, sim)) # Ensure non-negative

        if not rows:
            print("      Warning: No mutual KNN edges found.")
            return torch.zeros((2,0), dtype=torch.long), None

        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        print(f"      Created {edge_index.shape[1]} mutual KNN edges.")
        return edge_index, edge_attr

    def _build_quantile_edges(self, embeddings: np.ndarray, quantile_p: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None
        if not (0 < quantile_p < 100):
            print(f"      Error: quantile_p must be between 0 and 100 (exclusive). Got {quantile_p}")
            return torch.zeros((2,0), dtype=torch.long), None
        print(f"    Building QUANTILE graph ({quantile_p=:.2f}th percentile) for 'news'-'news' edges...")

        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        print("      Calculating pairwise similarities...")
        try:
            similarities_matrix = cosine_similarity(embeddings)
        except MemoryError:
            print("      Error: MemoryError calculating full similarity matrix for quantile. Cannot proceed.")
            return torch.zeros((2,0), dtype=torch.long), None
        except Exception as e:
            print(f"      Error calculating similarity matrix: {e}. Cannot proceed.")
            return torch.zeros((2,0), dtype=torch.long), None

        mask = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
        non_self_similarities = similarities_matrix[mask]

        if len(non_self_similarities) == 0:
            print("      Warning: No non-self similarities found to calculate quantile. No edges will be created.")
            return torch.zeros((2,0), dtype=torch.long), None

        similarity_threshold = np.percentile(non_self_similarities, quantile_p)
        print(f"      Similarity {quantile_p:.2f}th percentile threshold: {similarity_threshold:.4f}")

        rows, cols, data = [], [], []
        adj = similarities_matrix > similarity_threshold
        np.fill_diagonal(adj, False)
        indices_rows, indices_cols = np.where(adj)

        if len(indices_rows) == 0: # Fallback if threshold is too high
            print(f"      Warning: Quantile threshold {similarity_threshold:.4f} too high, no edges created. Adding fallback (top-1)...")
            np.fill_diagonal(similarities_matrix, -np.inf)
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])): continue
                top_idx = np.argmax(similarities_matrix[i])
                if similarities_matrix[i, top_idx] > -np.inf:
                    rows.append(i)
                    cols.append(top_idx)
                    data.append(max(0.0, similarities_matrix[i, top_idx]))
        else:
            rows = indices_rows.tolist()
            cols = indices_cols.tolist()
            data = similarities_matrix[rows, cols].tolist()
            data = [max(0.0, s) for s in data]

        if not rows: return torch.zeros((2, 0), dtype=torch.long), None
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        print(f"      Created {edge_index.shape[1]} quantile edges.")
        return edge_index, edge_attr

    def _build_topk_mean_edges(self, embeddings: np.ndarray, k_top_sim: int, threshold_factor: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None
        k_top_sim = min(k_top_sim, num_nodes - 1)
        if k_top_sim <= 0: return torch.zeros((2,0), dtype=torch.long), None
        print(f"    Building TOPK-MEAN graph (k_top_sim={k_top_sim}, factor={threshold_factor:.2f}) for 'news'-'news' edges...")

        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        print("      Calculating pairwise similarities...")
        try:
            similarities_matrix = cosine_similarity(embeddings)
        except MemoryError:
            print("      Error: MemoryError calculating full similarity matrix for topk_mean. Cannot proceed.")
            return torch.zeros((2,0), dtype=torch.long), None
        except Exception as e:
            print(f"      Error calculating similarity matrix: {e}. Cannot proceed.")
            return torch.zeros((2,0), dtype=torch.long), None

        np.fill_diagonal(similarities_matrix, -np.inf)

        local_topk_mean_sims = []
        print(f"      Calculating mean similarity to top-{k_top_sim} neighbors for each node...")
        for i in tqdm(range(num_nodes), desc="      Computing local Top-K means", leave=False, ncols=100):
            node_similarities = similarities_matrix[i, :].copy()
            actual_k = min(k_top_sim, np.sum(np.isfinite(node_similarities)))
            if actual_k <= 0:
                local_topk_mean_sims.append(0)
                continue

            top_k_indices = np.argpartition(node_similarities, -actual_k)[-actual_k:]
            top_k_sim_values = node_similarities[top_k_indices]
            top_k_sim_values = top_k_sim_values[np.isfinite(top_k_sim_values)]

            if len(top_k_sim_values) > 0:
                local_topk_mean_sims.append(np.mean(top_k_sim_values))
            else:
                local_topk_mean_sims.append(0)

        if not local_topk_mean_sims:
            print("      Error: Could not calculate any local top-k mean similarities.")
            return torch.zeros((2,0), dtype=torch.long), None

        global_avg_topk_mean_sim = np.mean([s for s in local_topk_mean_sims if not np.isnan(s)])
        final_threshold = global_avg_topk_mean_sim * threshold_factor
        print(f"      Global average of Top-{k_top_sim} mean similarities: {global_avg_topk_mean_sim:.4f}")
        print(f"      Final similarity threshold (applied factor {threshold_factor:.2f}): {final_threshold:.4f}")

        rows, cols, data = [], [], []
        adj = similarities_matrix > final_threshold
        indices_rows, indices_cols = np.where(adj)

        if len(indices_rows) == 0:
            print(f"      Warning: TopK-Mean threshold {final_threshold:.4f} too high, no edges created. Adding fallback (top-1)...")
            for i in range(num_nodes):
                if np.all(np.isinf(similarities_matrix[i])): continue
                top_idx = np.argmax(similarities_matrix[i])
                if similarities_matrix[i, top_idx] > -np.inf:
                    rows.append(i)
                    cols.append(top_idx)
                    data.append(max(0.0, similarities_matrix[i, top_idx]))
        else:
            rows = indices_rows.tolist()
            cols = indices_cols.tolist()
            for r, c in zip(rows, cols):
                data.append(max(0.0, similarities_matrix[r, c]))

        if not rows: return torch.zeros((2, 0), dtype=torch.long), None
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        print(f"      Created {edge_index.shape[1]} topk-mean edges.")
        return edge_index, edge_attr

    def _add_interaction_edges_by_type(self, data, train_labeled_indices, train_unlabeled_indices, test_indices, num_train_labeled, num_train_unlabeled, num_test, num_nodes, num_interactions_per_news, num_interaction_nodes, global2local):
        all_interaction_embeddings = []
        all_tones_set = set()
        edge_indices = dict()
        reverse_edge_indices = dict()
        pbar_interact = tqdm(total=num_interaction_nodes, desc="    Extracting Interaction Embeddings", ncols=100)
        interaction_global_idx = 0
        for news_idx, (idx, is_test) in enumerate(list(zip(np.concatenate([train_labeled_indices, train_unlabeled_indices, test_indices]), [False]*(num_train_labeled+num_train_unlabeled)+[True]*num_test))):
            if not is_test:
                embeddings_list = self.train_data[int(idx)][self.interaction_embedding_field]
                tones_list = self.train_data[int(idx)][self.interaction_tone_field]
            else:
                embeddings_list = self.test_data[int(idx)][self.interaction_embedding_field]
                tones_list = self.test_data[int(idx)][self.interaction_tone_field]
            node_interactions = np.array(embeddings_list)
            all_interaction_embeddings.append(node_interactions)
            for i, tone in enumerate(tones_list):
                tone_key = tone.strip().lower().replace(' ', '_')
                all_tones_set.add(tone_key)
                if tone_key not in edge_indices:
                    edge_indices[tone_key] = [[], []]
                    reverse_edge_indices[tone_key] = [[], []]
                # local news_idx
                local_news_idx = global2local[int(idx)]
                edge_indices[tone_key][0].append(local_news_idx)
                edge_indices[tone_key][1].append(interaction_global_idx)
                reverse_edge_indices[tone_key][0].append(interaction_global_idx)
                reverse_edge_indices[tone_key][1].append(local_news_idx)
                interaction_global_idx += 1
            pbar_interact.update(1)
        pbar_interact.close()
        final_interaction_features = np.vstack(all_interaction_embeddings)
        data['interaction'].x = torch.tensor(final_interaction_features, dtype=torch.float)
        data['interaction'].num_nodes = data['interaction'].x.shape[0]
        if data['interaction'].num_nodes != num_interaction_nodes:
            print(f"Warning: Interaction node count mismatch! Expected {num_interaction_nodes}, Got {data['interaction'].num_nodes}")
        for tone_key in sorted(all_tones_set):
            edge_type = ('news', f'has_{tone_key}_interaction', 'interaction')
            rev_edge_type = ('interaction', f'rev_has_{tone_key}_interaction', 'news')
            if edge_indices[tone_key][0]:
                data[edge_type].edge_index = torch.tensor(edge_indices[tone_key], dtype=torch.long)
            if reverse_edge_indices[tone_key][0]:
                data[rev_edge_type].edge_index = torch.tensor(reverse_edge_indices[tone_key], dtype=torch.long)

    def _add_interaction_edges_with_attr(self, data, train_labeled_indices, train_unlabeled_indices, test_indices, num_train_labeled, num_train_unlabeled, num_test, num_nodes, num_interactions_per_news, num_interaction_nodes, global2local):
        all_interaction_embeddings = []
        all_interaction_tones = []
        pbar_interact = tqdm(total=num_interaction_nodes, desc="    Extracting Interaction Embeddings", ncols=100)
        interaction_global_idx = 0
        for news_idx, (idx, is_test) in enumerate(list(zip(np.concatenate([train_labeled_indices, train_unlabeled_indices, test_indices]), [False]*(num_train_labeled+num_train_unlabeled)+[True]*num_test))):
            if not is_test:
                embeddings_list = self.train_data[int(idx)][self.interaction_embedding_field]
                tones_list = self.train_data[int(idx)][self.interaction_tone_field]
            else:
                embeddings_list = self.test_data[int(idx)][self.interaction_embedding_field]
                tones_list = self.test_data[int(idx)][self.interaction_tone_field]
            node_interactions = np.array(embeddings_list)
            all_interaction_embeddings.append(node_interactions)
            for i, tone in enumerate(tones_list):
                tone_key = tone.strip().lower().replace(' ', '_')
                all_interaction_tones.append(self._tone2id(tone_key))
                # local news_idx
                local_news_idx = global2local[int(idx)]
                interaction_global_idx += 1
            pbar_interact.update(1)
        pbar_interact.close()
        final_interaction_features = np.vstack(all_interaction_embeddings)
        data['interaction'].x = torch.tensor(final_interaction_features, dtype=torch.float)
        data['interaction'].num_nodes = data['interaction'].x.shape[0]
        if data['interaction'].num_nodes != num_interaction_nodes:
            print(f"Warning: Interaction node count mismatch! Expected {num_interaction_nodes}, Got {data['interaction'].num_nodes}")
        news_has_interaction_src = torch.arange(num_nodes).repeat_interleave(num_interactions_per_news)
        interaction_has_interaction_tgt = torch.arange(num_interaction_nodes)
        data['news', 'has_interaction', 'interaction'].edge_index = torch.stack([news_has_interaction_src, interaction_has_interaction_tgt], dim=0)
        data['news', 'has_interaction', 'interaction'].edge_attr = torch.tensor(all_interaction_tones, dtype=torch.long)
        data['interaction', 'rev_has_interaction', 'news'].edge_index = torch.stack([interaction_has_interaction_tgt, news_has_interaction_src], dim=0)

    def _build_label_aware_knn_edges(
        self,
        embeddings: np.ndarray,
        labeled_indices: np.ndarray,
        pseudo_indices: np.ndarray,
        labeled_labels: np.ndarray,
        pseudo_labels: np.ndarray,
        k: int,
        pseudo_confidence: Optional[np.ndarray],
        global2local: dict
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        For each labeled node, find its KNN only among pseudo nodes with the same label.
        Optionally, use pseudo_confidence as edge_attr.
        Only search in the provided pseudo_indices (no fallback to all nodes).
        """
        from sklearn.metrics.pairwise import cosine_similarity

        edge_src = []
        edge_dst = []
        edge_attr = []

        label_set = np.unique(labeled_labels)
        pseudo_label_map = {idx: label for idx, label in zip(pseudo_indices, pseudo_labels)}
        pseudo_conf_map = {idx: conf for idx, conf in zip(pseudo_indices, pseudo_confidence)} if pseudo_confidence is not None else None

        for label in label_set:
            labeled_mask = labeled_labels == label
            pseudo_mask = pseudo_labels == label
            labeled_idx = labeled_indices[labeled_mask]
            pseudo_idx = pseudo_indices[pseudo_mask]
            if len(pseudo_idx) == 0 or len(labeled_idx) == 0:
                continue  # nothing to connect
            labeled_idx_local = np.array([global2local[idx] for idx in labeled_idx])
            pseudo_idx_local = np.array([global2local[idx] for idx in pseudo_idx])
            labeled_emb = embeddings[labeled_idx_local]
            pseudo_emb = embeddings[pseudo_idx_local]
            sim = cosine_similarity(labeled_emb, pseudo_emb)
            for i, src in enumerate(labeled_idx_local):
                if sim.shape[1] < k:
                    topk = np.argsort(-sim[i])
                else:
                    topk = np.argpartition(-sim[i], k)[:k]
                for j in topk:
                    dst = pseudo_idx_local[j]
                    edge_src.append(src)
                    edge_dst.append(dst)
                    edge_attr.append(sim[i, j])

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1) if edge_attr else None
        return edge_index, edge_attr_tensor

    def build_hetero_graph(self, test_batch_indices=None) -> Optional[HeteroData]:
        """
        Build a heterogeneous graph including both nodes and edges.
        Pipeline:
        1. Build empty graph
            - train_labeled_nodes (train_labeled_mask from training set)
            - train_unlabeled_nodes (train_unlabeled_mask from training set)
            - test_nodes (test_mask from test set)
        2. Build edges (based on edge policy)
        3. Update graph data with edges
        """
        print("\nStarting Heterogeneous Graph Construction...")

        data = HeteroData()

        # --- Select News Nodes (k-shot, unlabeled sampling, test) ---        
        # 1. Sample k-shot labeled nodes from train set (with cache)
        train_labeled_indices_cache_path = f"utils/{self.dataset_name}_{self.k_shot}_shot_train_labeled_indices_{self.seed}.json"
        if os.path.exists(train_labeled_indices_cache_path):
            with open(train_labeled_indices_cache_path, "r") as f:
                train_labeled_indices = json.load(f)
            train_labeled_indices = np.array(train_labeled_indices)
            print(f"Loaded k-shot indices from cache: {train_labeled_indices_cache_path}")
        else:
            train_labeled_indices, _ = sample_k_shot(self.train_data, self.k_shot, self.seed)
            train_labeled_indices = np.array(train_labeled_indices)
            with open(train_labeled_indices_cache_path, "w") as f:
                json.dump(train_labeled_indices.tolist(), f)
            print(f"Saved k-shot indices to cache: {train_labeled_indices_cache_path}")
        self.train_labeled_indices = train_labeled_indices
        print(f"  Selected {len(train_labeled_indices)} labeled nodes: {train_labeled_indices} ...")

        # 2. Get train_unlabeled_nodes (all train nodes not in train_labeled_indices)
        all_train_indices = np.arange(len(self.train_data))
        train_unlabeled_indices = np.setdiff1d(all_train_indices, train_labeled_indices, assume_unique=True)

        # --- Sample train_unlabeled_nodes if required ---
        if self.partial_unlabeled:
            num_to_sample = min(self.sample_unlabeled_factor * 2 * self.k_shot, len(train_unlabeled_indices))
            print(f"Sampling {num_to_sample} train_unlabeled_nodes (factor={self.sample_unlabeled_factor}, 2*k={2*self.k_shot}) from {len(train_unlabeled_indices)} available.")
            
            if self.pseudo_label:
                print("Using pseudo-label based sampling...")
                try:
                    with open(self.pseudo_label_cache_path, "r") as f:
                        pseudo_data = json.load(f)
                    pseudo_label_map = {int(item["index"]): int(item["pseudo_label"]) for item in pseudo_data}
                    
                    # Filter unlabeled indices to those with pseudo labels
                    valid_unlabeled = [idx for idx in train_unlabeled_indices if idx in pseudo_label_map]
                    if not valid_unlabeled:
                        print("Warning: No valid pseudo labels found. Falling back to random sampling.")
                        train_unlabeled_indices = np.random.choice(train_unlabeled_indices, size=num_to_sample, replace=False)
                    else:
                        # Group indices by pseudo label
                        pseudo_label_groups = {}
                        for idx in valid_unlabeled:
                            label = pseudo_label_map[idx]
                            if label not in pseudo_label_groups:
                                pseudo_label_groups[label] = []
                            pseudo_label_groups[label].append(idx)
                        
                        # Calculate samples per class
                        num_classes = len(pseudo_label_groups)
                        samples_per_class = num_to_sample // num_classes
                        remainder = num_to_sample % num_classes
                        
                        # Sample from each class
                        sampled_indices = []
                        for label, indices in pseudo_label_groups.items():
                            n_samples = min(samples_per_class + (1 if remainder > 0 else 0), len(indices))
                            if remainder > 0:
                                remainder -= 1
                            if n_samples > 0:
                                sampled = np.random.choice(indices, size=n_samples, replace=False)
                                sampled_indices.extend(sampled)
                        
                        # If we still need more samples (due to insufficient pseudo labels in some classes)
                        if len(sampled_indices) < num_to_sample:
                            remaining = num_to_sample - len(sampled_indices)
                            print(f"Warning: Only found {len(sampled_indices)} nodes with pseudo labels. "
                                  f"Randomly sampling {remaining} more nodes.")
                            remaining_indices = list(set(train_unlabeled_indices) - set(sampled_indices))
                            if remaining_indices:
                                additional = np.random.choice(remaining_indices, size=min(remaining, len(remaining_indices)), replace=False)
                                sampled_indices.extend(additional)
                        
                        train_unlabeled_indices = np.array(sampled_indices)
                        print(f"  Sampled {len(train_unlabeled_indices)} unlabeled nodes using pseudo labels")
                        
                except Exception as e:
                    print(f"Warning: Error during pseudo-label sampling: {e}. Falling back to random sampling.")
                    train_unlabeled_indices = np.random.choice(train_unlabeled_indices, size=num_to_sample, replace=False)
            else:
                train_unlabeled_indices = np.random.choice(train_unlabeled_indices, size=num_to_sample, replace=False)
        
        self.train_unlabeled_indices = train_unlabeled_indices
        print(f"  Selected {len(train_unlabeled_indices)} unlabeled train nodes.")

        # 3. Get all test node indices
        if test_batch_indices is not None:
            test_indices = np.array(test_batch_indices)
        else:
            test_indices = np.arange(len(self.test_data))
        self.test_indices = test_indices
        print(f"  Test nodes: {len(test_indices)}")

        # 4. Extract features and labels for each group
        train_labeled_emb = np.array(self.train_data.select(train_labeled_indices.tolist())[self.text_embedding_field])
        train_labeled_label = np.array(self.train_data.select(train_labeled_indices.tolist())["label"])
        train_unlabeled_emb = np.array(self.train_data.select(train_unlabeled_indices.tolist())[self.text_embedding_field])
        train_unlabeled_label = np.array(self.train_data.select(train_unlabeled_indices.tolist())["label"])
        test_emb = np.array(self.test_data.select(test_indices.tolist())[self.text_embedding_field])
        test_label = np.array(self.test_data.select(test_indices.tolist())["label"])

        # 5. Concatenate all nodes
        x = torch.tensor(np.concatenate([train_labeled_emb, train_unlabeled_emb, test_emb]), dtype=torch.float)
        y = torch.tensor(np.concatenate([train_labeled_label, train_unlabeled_label, test_label]), dtype=torch.long)

        # 6. Build masks
        num_train_labeled = len(train_labeled_indices)      # train labeled nodes
        num_train_unlabeled = len(train_unlabeled_indices)  # train unlabeled nodes
        num_test = len(test_indices)                        # test nodes
        num_nodes = num_train_labeled + num_train_unlabeled + num_test  # all nodes

        # --- global2local index mapping for all nodes in the graph ---
        all_indices = np.concatenate([train_labeled_indices, train_unlabeled_indices, test_indices])
        global2local = {int(idx): i for i, idx in enumerate(all_indices)}

        train_labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_unlabeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_labeled_mask[:num_train_labeled] = True
        train_unlabeled_mask[num_train_labeled:num_train_labeled+num_train_unlabeled] = True
        test_mask[num_train_labeled+num_train_unlabeled:] = True

        data['news'].x = x
        data['news'].y = y
        data['news'].num_nodes = num_nodes
        data['news'].train_labeled_mask = train_labeled_mask
        data['news'].train_unlabeled_mask = train_unlabeled_mask
        data['news'].test_mask = test_mask
        
        print(f"Total news nodes: {num_nodes}")
        print(f"    - 'news' features shape: {data['news'].x.shape}")
        print(f"    - 'news' masks created: Train Labeled={train_labeled_mask.sum()}, Train Unlabeled={train_unlabeled_mask.sum()}, Test={test_mask.sum()}")

        # --- Prepare 'interaction' Node Features ---
        num_interactions_per_news = 20
        num_interaction_nodes = num_nodes * num_interactions_per_news
        print(f"  Preparing features for {num_interaction_nodes} 'interaction' nodes...")
        if self.interaction_edge_mode == "edge_type":
            self._add_interaction_edges_by_type(data, train_labeled_indices, train_unlabeled_indices, test_indices, num_train_labeled, num_train_unlabeled, num_test, num_nodes, num_interactions_per_news, num_interaction_nodes, global2local)
        elif self.interaction_edge_mode == "edge_attr":
            self._add_interaction_edges_with_attr(data, train_labeled_indices, train_unlabeled_indices, test_indices, num_train_labeled, num_train_unlabeled, num_test, num_nodes, num_interactions_per_news, num_interaction_nodes, global2local)
        else:
            raise ValueError(f"Unknown interaction_edge_mode: {self.interaction_edge_mode}")

        # --- Create Edges ---
        print(f"Creating graph edges...")
        
        # Source nodes: 0, 0, ..., 0 (20 times), 1, 1, ..., 1 (20 times), ... N-1, ... N-1 (20 times)
        news_has_interaction_src = torch.arange(num_nodes).repeat_interleave(num_interactions_per_news)
        interaction_has_interaction_tgt = torch.arange(num_interaction_nodes)

        # Create 'news -> interaction' edges
        data['news', 'has_interaction', 'interaction'].edge_index = torch.stack([news_has_interaction_src, interaction_has_interaction_tgt], dim=0)
        print(f"    - Created {data['news', 'has_interaction', 'interaction'].edge_index.shape[1]} 'news -> interaction' edges.")

        # Create 'interaction -> news' edges
        data['interaction', 'rev_has_interaction', 'news'].edge_index = torch.stack([interaction_has_interaction_tgt, news_has_interaction_src], dim=0)
        print(f"    - Created {data['interaction', 'rev_has_interaction', 'news'].edge_index.shape[1]} 'interaction -> news' edges.")

        # --- Ensure new_embeddings is defined for edge builders ---
        new_embeddings = data['news'].x.cpu().numpy()

        # --- Edge Building Logic (flat style, like build_graph.py) ---
        if self.edge_policy == "knn":
            sim_edge_index, sim_edge_attr, dis_edge_index, dis_edge_attr = self._build_knn_edges(new_embeddings, self.k_neighbors)
            sim_edge_index = torch.cat([sim_edge_index, sim_edge_index[[1, 0], :]], dim=1)
            sim_edge_attr = torch.cat([sim_edge_attr, sim_edge_attr], dim=0)
            data['news', 'similar_to', 'news'].edge_index = sim_edge_index
            if sim_edge_attr is not None:
                data['news', 'similar_to', 'news'].edge_attr = sim_edge_attr
            print(f"    - Created {sim_edge_index.shape[1]} 'news <-> news' similar edges.")
            if self.enable_dissimilar:
                dis_edge_index = torch.cat([dis_edge_index, dis_edge_index[[1, 0], :]], dim=1)
                dis_edge_attr = torch.cat([dis_edge_attr, dis_edge_attr], dim=0)
                data['news', 'dissimilar_to', 'news'].edge_index = dis_edge_index
                if dis_edge_attr is not None:
                    data['news', 'dissimilar_to', 'news'].edge_attr = dis_edge_attr
                print(f"    - Created {dis_edge_index.shape[1]} 'news <-> news' dissimilar edges.")
        elif self.edge_policy == "label_aware_knn":
            pseudo_label_cache_path = self.pseudo_label_cache_path
            try:
                with open(pseudo_label_cache_path, "r") as f:
                    pseudo_data = json.load(f)
                pseudo_data = sorted(pseudo_data, key=lambda x: float(x.get("confidence", 1.0)), reverse=True)
                num_to_select = min(2 * self.k_shot * self.sample_unlabeled_factor, len(pseudo_data))
                print(f"    - Selecting {num_to_select} pseudo labels for label-aware KNN edges.")
                train_unlabeled_set = set(train_unlabeled_indices.tolist())
                pseudo_data = [item for item in pseudo_data if int(item["index"]) in train_unlabeled_set]
                pseudo_data = pseudo_data[:num_to_select]
                pseudo_indices = np.array([int(item["index"]) for item in pseudo_data])
                pseudo_labels = np.array([int(item["pseudo_label"]) for item in pseudo_data])
                pseudo_confidence = np.array([float(item.get("confidence", 1.0)) for item in pseudo_data])
            except Exception as e:
                print(f"  Error loading pseudo label cache: {e}")
                pseudo_indices = np.array([])
                pseudo_labels = np.array([])
                pseudo_confidence = np.array([])
            edge_index, edge_attr = self._build_label_aware_knn_edges(
                new_embeddings,
                train_labeled_indices,
                pseudo_indices,
                train_labeled_label,
                pseudo_labels,
                self.k_neighbors,
                pseudo_confidence,
                global2local
            )
            data['news', 'label_aware_similar_to', 'news'].edge_index = edge_index
            edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
            if edge_attr is not None:
                data['news', 'label_aware_similar_to', 'news'].edge_attr = edge_attr
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
            print(f"    - Created {edge_index.shape[1]} 'news -> pseudo_news' label-aware similar edges.")
        elif self.edge_policy == "mutual_knn":
            edge_index_nn, edge_attr_nn = self._build_mutual_knn_edges(new_embeddings, self.k_neighbors)
            sim_edge_index = torch.cat([edge_index_nn, edge_index_nn[[1, 0], :]], dim=1)
            sim_edge_attr = torch.cat([edge_attr_nn, edge_attr_nn], dim=0)
            data['news', 'similar_to', 'news'].edge_index = sim_edge_index
            if sim_edge_attr is not None:
                data['news', 'similar_to', 'news'].edge_attr = sim_edge_attr
            print(f"    - Created {sim_edge_index.shape[1]} 'news <-> news' edges.")
        elif self.edge_policy == "local_threshold":
            edge_index_nn, edge_attr_nn = self._build_local_threshold_edges(new_embeddings, self.local_threshold_factor)
            sim_edge_index = torch.cat([edge_index_nn, edge_index_nn[[1, 0], :]], dim=1)
            sim_edge_attr = torch.cat([edge_attr_nn, edge_attr_nn], dim=0)
            data['news', 'similar_to', 'news'].edge_index = sim_edge_index
            if sim_edge_attr is not None:
                data['news', 'similar_to', 'news'].edge_attr = sim_edge_attr
            print(f"    - Created {sim_edge_index.shape[1]} 'news <-> news' edges.")
        elif self.edge_policy == "global_threshold":
            edge_index_nn, edge_attr_nn = self._build_global_threshold_edges(new_embeddings, self.alpha)
            sim_edge_index = torch.cat([edge_index_nn, edge_index_nn[[1, 0], :]], dim=1)
            sim_edge_attr = torch.cat([edge_attr_nn, edge_attr_nn], dim=0)
            data['news', 'similar_to', 'news'].edge_index = sim_edge_index
            if sim_edge_attr is not None:
                data['news', 'similar_to', 'news'].edge_attr = sim_edge_attr
            print(f"    - Created {sim_edge_index.shape[1]} 'news <-> news' edges.")
        elif self.edge_policy == "quantile":
            edge_index_nn, edge_attr_nn = self._build_quantile_edges(new_embeddings, self.quantile_p)
            sim_edge_index = torch.cat([edge_index_nn, edge_index_nn[[1, 0], :]], dim=1)
            sim_edge_attr = torch.cat([edge_attr_nn, edge_attr_nn], dim=0)
            data['news', 'similar_to', 'news'].edge_index = sim_edge_index
            if sim_edge_attr is not None:
                data['news', 'similar_to', 'news'].edge_attr = sim_edge_attr
            print(f"    - Created {sim_edge_index.shape[1]} 'news <-> news' edges.")
        elif self.edge_policy == "topk_mean":
            edge_index_nn, edge_attr_nn = self._build_topk_mean_edges(new_embeddings, self.k_top_sim, self.local_threshold_factor)
            sim_edge_index = torch.cat([edge_index_nn, edge_index_nn[[1, 0], :]], dim=1)
            sim_edge_attr = torch.cat([edge_attr_nn, edge_attr_nn], dim=0)
            data['news', 'similar_to', 'news'].edge_index = sim_edge_index
            if sim_edge_attr is not None:
                data['news', 'similar_to', 'news'].edge_attr = sim_edge_attr
            print(f"    - Created {sim_edge_index.shape[1]} 'news <-> news' edges.")
        else:
            print(f"Warning: News-news edge policy '{self.edge_policy}' not implemented. Skipping news-news edges.")
            sim_edge_index = torch.zeros((2, 0), dtype=torch.long)
            sim_edge_attr = None

        print("\nHeterogeneous graph construction complete.")

        return data

    def analyze_hetero_graph(self, hetero_graph: HeteroData) -> None:
        """Detailed analysis for heterogeneous graph, similar to build_graph.py but adapted for hetero."""
        
        print("\n" + "=" * 60)
        print("     Heterogeneous Graph Analysis (Detailed)")
        print("=" * 60)

        self.graph_metrics = {}  # Reset metrics

        # --- Node Type Stats ---
        print("\n--- Node Types ---")
        total_nodes = 0
        node_type_info = {}
        for node_type in hetero_graph.node_types:
            n = hetero_graph[node_type].num_nodes
            total_nodes += n
            print(f"Node Type: '{node_type}'")
            print(f"  - Num Nodes: {n}")
            if hasattr(hetero_graph[node_type], 'x') and hetero_graph[node_type].x is not None:
                print(f"  - Features Dim: {hetero_graph[node_type].x.shape[1]}")
                node_type_info[node_type] = {"num_nodes": n, "feature_dim": hetero_graph[node_type].x.shape[1]}
            
            # For 'news', print label and mask info
            if node_type == 'news':
                if hasattr(hetero_graph[node_type], 'y') and hetero_graph[node_type].y is not None:
                    y = hetero_graph[node_type].y.cpu().numpy()
                    print(f"  - Labels Shape: {y.shape}")
                    unique, counts = np.unique(y, return_counts=True)
                    label_dist = {int(k): int(v) for k, v in zip(unique, counts)}
                    print(f"  - Label Distribution: {label_dist}")
                    node_type_info[node_type]["label_dist"] = label_dist
                for mask in ['train_labeled_mask', 'train_unlabeled_mask', 'test_mask']:
                    if hasattr(hetero_graph[node_type], mask) and hetero_graph[node_type][mask] is not None:
                        count = hetero_graph[node_type][mask].sum().item()
                        print(f"  - {mask}: {count} nodes ({count/n*100:.1f}% of '{node_type}')")
                        node_type_info[node_type][mask] = count
        
        print(f"Total Nodes (all types): {total_nodes}")
        self.graph_metrics['node_type_info'] = node_type_info
        self.graph_metrics['nodes_total'] = total_nodes

        # --- Edge Type Stats ---
        print("\n--- Edge Types ---")
        total_edges = 0
        edge_type_info = {}
        for edge_type in hetero_graph.edge_types:
            num_edges = hetero_graph[edge_type].num_edges
            total_edges += num_edges
            edge_type_str = " -> ".join(edge_type) if isinstance(edge_type, tuple) else edge_type
            print(f"[*] Edge Type: {edge_type_str}")
            print(f"  - Num Edges: {num_edges}")
            if hasattr(hetero_graph[edge_type], 'edge_attr') and hetero_graph[edge_type].edge_attr is not None:
                edge_attr = hetero_graph[edge_type].edge_attr
                try:
                    shape = tuple(edge_attr.shape)
                    if len(shape) == 1:
                        print(f"  - Attributes Dim: {shape[0]}")
                        edge_type_info[edge_type_str] = {"num_edges": num_edges, "attr_dim": shape[0]}
                    else:
                        print(f"  - Attributes Dim: {shape}")
                        edge_type_info[edge_type_str] = {"num_edges": num_edges, "attr_dim": shape[1]}
                    print(f"  - Attributes: {edge_attr}")
                except Exception as e:
                    print(f"  - Attributes Dim: Error getting shape - {e}")
                    print(f"  - Attributes: {edge_attr}")
            else:
                print("  - Attributes: None")
                edge_type_info[edge_type_str] = {"num_edges": num_edges, "attr_dim": None}
        print(f"Total Edges (all types): {total_edges}")
        self.graph_metrics['edge_type_info'] = edge_type_info
        self.graph_metrics['edges_total'] = total_edges

        # --- News-News Edge Analysis ---
        news_similar_edge_type = ('news', 'similar_to', 'news')
        news_dissimilar_edge_type = ('news', 'dissimilar_to', 'news')
        num_news_nodes = hetero_graph['news'].num_nodes

        if news_similar_edge_type in hetero_graph.edge_types:
            print("\n--- Analysis for 'news'-'similar_to'-'news' Edges ---")
            nn_edge_index = hetero_graph[news_similar_edge_type].edge_index
            num_nn_edges = hetero_graph[news_similar_edge_type].num_edges
            print(f"  - Num Edges: {num_nn_edges}")
            degrees_nn = torch.zeros(num_news_nodes, dtype=torch.long, device=nn_edge_index.device)
            degrees_nn.scatter_add_(0, nn_edge_index[0], torch.ones_like(nn_edge_index[0]))
            degrees_nn.scatter_add_(0, nn_edge_index[1], torch.ones_like(nn_edge_index[1]))
            avg_degree_nn = degrees_nn.float().mean().item() / 2.0
            print(f"  - Avg Degree (undirected): {avg_degree_nn:.2f}")
            self.graph_metrics['avg_degree_news_similar_to'] = avg_degree_nn
            num_isolated = int((degrees_nn == 0).sum().item())
            print(f"  - Isolated News Nodes: {num_isolated} ({num_isolated/num_news_nodes*100:.1f}%)")
            self.graph_metrics['news_similar_isolated'] = num_isolated
            if hasattr(hetero_graph['news'], 'y') and hetero_graph['news'].y is not None:
                y_news = hetero_graph['news'].y.cpu().numpy()
                edge_index_nn_cpu = nn_edge_index.cpu().numpy()
                try:
                    G_sim = nx.Graph()
                    G_sim.add_nodes_from(range(num_news_nodes))
                    G_sim.add_edges_from(edge_index_nn_cpu.T)
                    num_undirected_nn_edges = G_sim.number_of_edges()
                    homophilic_undirected_nn = 0
                    for u, v in G_sim.edges():
                        if y_news[u] == y_news[v]:
                            homophilic_undirected_nn += 1
                    homophily_ratio_nn = homophilic_undirected_nn / num_undirected_nn_edges if num_undirected_nn_edges > 0 else 0.0
                    print(f"  - Homophily Ratio: {homophily_ratio_nn:.4f}")
                    self.graph_metrics['homophily_ratio_news_similar_to'] = homophily_ratio_nn
                    print(f"\n--- NetworkX (news-similar-news subgraph) ---")
                    print(f"  Nodes: {G_sim.number_of_nodes()} Edges: {G_sim.number_of_edges()}")
                    if G_sim.number_of_nodes() > 0:
                        print(f"  Density: {nx.density(G_sim):.4f}")
                        print(f"  Avg Clustering: {nx.average_clustering(G_sim):.4f}")
                except Exception as e:
                    print(f"  Warning: Could not calculate metrics for similar edges: {e}")

        if news_dissimilar_edge_type in hetero_graph.edge_types:
            print("\n--- Analysis for 'news'-'dissimilar_to'-'news' Edges ---")
            nn_edge_index = hetero_graph[news_dissimilar_edge_type].edge_index
            num_nn_edges = hetero_graph[news_dissimilar_edge_type].num_edges
            print(f"  - Num Edges: {num_nn_edges}")
            degrees_nn = torch.zeros(num_news_nodes, dtype=torch.long, device=nn_edge_index.device)
            degrees_nn.scatter_add_(0, nn_edge_index[0], torch.ones_like(nn_edge_index[0]))
            degrees_nn.scatter_add_(0, nn_edge_index[1], torch.ones_like(nn_edge_index[1]))
            avg_degree_nn = degrees_nn.float().mean().item() / 2.0
            print(f"  - Avg Degree (undirected): {avg_degree_nn:.2f}")
            self.graph_metrics['avg_degree_news_dissimilar_to'] = avg_degree_nn
            num_isolated = int((degrees_nn == 0).sum().item())
            print(f"  - Isolated News Nodes: {num_isolated} ({num_isolated/num_news_nodes*100:.1f}%)")
            self.graph_metrics['news_dissimilar_isolated'] = num_isolated
            if hasattr(hetero_graph['news'], 'y') and hetero_graph['news'].y is not None:
                y_news = hetero_graph['news'].y.cpu().numpy()
                edge_index_nn_cpu = nn_edge_index.cpu().numpy()
                try:
                    G_dis = nx.Graph()
                    G_dis.add_nodes_from(range(num_news_nodes))
                    G_dis.add_edges_from(edge_index_nn_cpu.T)
                    num_undirected_nn_edges = G_dis.number_of_edges()
                    heterophilic_undirected_nn = 0
                    for u, v in G_dis.edges():
                        if y_news[u] != y_news[v]:
                            heterophilic_undirected_nn += 1
                    heterophily_ratio_nn = heterophilic_undirected_nn / num_undirected_nn_edges if num_undirected_nn_edges > 0 else 0.0
                    print(f"  - Heterophily Ratio: {heterophily_ratio_nn:.4f}")
                    self.graph_metrics['heterophily_ratio_news_dissimilar_to'] = heterophily_ratio_nn
                    print(f"\n--- NetworkX (news-dissimilar-news subgraph) ---")
                    print(f"  Nodes: {G_dis.number_of_nodes()} Edges: {G_dis.number_of_edges()}")
                    if G_dis.number_of_nodes() > 0:
                        print(f"  Density: {nx.density(G_dis):.4f}")
                        print(f"  Avg Clustering: {nx.average_clustering(G_dis):.4f}")
                except Exception as e:
                    print(f"  Warning: Could not calculate metrics for dissimilar edges: {e}")

        # 合并分析所有 news-news 边
        if news_similar_edge_type in hetero_graph.edge_types or news_dissimilar_edge_type in hetero_graph.edge_types:
            print("\n--- Analysis for ALL news-news Edges (merged) ---")
            import networkx as nx
            G_all = nx.Graph()
            G_all.add_nodes_from(range(num_news_nodes))
            # similar edges
            if news_similar_edge_type in hetero_graph.edge_types:
                sim_idx = hetero_graph[news_similar_edge_type].edge_index.cpu().numpy()
                G_all.add_edges_from(sim_idx.T)
            # dissimilar edges
            if news_dissimilar_edge_type in hetero_graph.edge_types:
                dis_idx = hetero_graph[news_dissimilar_edge_type].edge_index.cpu().numpy()
                G_all.add_edges_from(dis_idx.T)
            print(f"  Nodes: {G_all.number_of_nodes()} Edges: {G_all.number_of_edges()}")
            if G_all.number_of_nodes() > 0:
                print(f"  Density: {nx.density(G_all):.4f}")
                print(f"  Avg Clustering: {nx.average_clustering(G_all):.4f}")
                print(f"  Connected Components: {nx.number_connected_components(G_all)}")

        print("=" * 60)
        print("      End of Heterogeneous Graph Analysis")
        print("=" * 60 + "\n")


    def save_graph(self, hetero_graph: HeteroData, batch_id=None) -> Optional[str]:
        """Save the HeteroData graph and analysis results."""

        # --- Generate graph name ---
        edge_policy_name = self.edge_policy
        edge_param_str = ""
        # Suffix based on news-news edge policy params
        if self.edge_policy == "knn": 
            edge_param_str = f"{self.k_neighbors}"
        elif self.edge_policy == "label_aware_knn":
            edge_param_str = f"{self.k_neighbors}"
        elif self.edge_policy == "mutual_knn": 
            edge_param_str = f"{self.k_neighbors}"
        elif self.edge_policy == "local_threshold": 
            edge_param_str = f"{self.local_threshold_factor:.2f}".replace('.', 'p')
        elif self.edge_policy == "global_threshold": 
            edge_param_str = f"{self.alpha:.2f}".replace('.', 'p')
        elif self.edge_policy == "quantile": 
            edge_param_str = f"{self.quantile_p:.1f}".replace('.', 'p')
        elif self.edge_policy == "topk_mean": 
            edge_param_str = f"{self.k_top_sim}_f{self.local_threshold_factor:.2f}".replace('.', 'p')

        # Add sampling info to filename if sampling was used
        suffix = []
        if self.pseudo_label:
            suffix.append("pseudo")
        if self.partial_unlabeled:
            suffix.append("partial")
            suffix.append(f"sample_unlabeled_factor_{self.sample_unlabeled_factor}")
        if self.enable_dissimilar:
            suffix.append("dissimilar")
        sampling_suffix = f"_{'_'.join(suffix)}" if suffix else ""

        # Include text embedding type and edge types in name
        graph_name = f"{self.k_shot}_shot_{self.embedding_type}_hetero_{edge_policy_name}_{edge_param_str}{sampling_suffix}_mv{self.multi_view}"
        scenario_dir = os.path.join(self.output_dir, graph_name)
        os.makedirs(scenario_dir, exist_ok=True)
        if batch_id is not None:
            graph_file = f"{graph_name}_batch{batch_id}.pt"
            metrics_file = f"{graph_name}_batch{batch_id}_metrics.json"
            indices_file = f"{graph_name}_batch{batch_id}_indices.json"
        else:
            graph_file = f"{graph_name}.pt"
            metrics_file = f"{graph_name}_metrics.json"
            indices_file = f"{graph_name}_indices.json"

        graph_path = os.path.join(scenario_dir, graph_file)
        metrics_path = os.path.join(scenario_dir, metrics_file)
        indices_path = os.path.join(scenario_dir, indices_file)
        # --- End filename generation ---

        # Save graph data
        cpu_graph_data = hetero_graph.cpu()
        torch.save(cpu_graph_data, graph_path)

        # Save graph metrics (simplified for hetero)
        def default_serializer(obj): # Helper for JSON serialization
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, torch.Tensor): return obj.tolist()
            try: return json.JSONEncoder().encode(obj)
            except TypeError: return str(obj)

        try:
            with open(metrics_path, "w") as f:
                json.dump(self.graph_metrics, f, indent=2, default=default_serializer)
            print(f"  Graph analysis metrics saved to {metrics_path}")
        except Exception as e:
            print(f"  Error saving metrics JSON: {e}")

        # Save selected indices info
        indices_data = {
            "k_shot": int(self.k_shot),
            "seed": int(self.seed),
            "partial_unlabeled": self.partial_unlabeled,
            "embedding_type": self.embedding_type,
            "news_news_edge_policy": self.edge_policy,
        }

        if self.train_labeled_indices is not None:
            indices_data["train_labeled_indices"] = [int(i) for i in self.train_labeled_indices]
            # Add label distribution if possible
            try:
                train_labels = self.train_data['label']
                label_dist = {}
                for idx in self.train_labeled_indices:
                    label = train_labels[int(idx)]
                    label_dist[label] = label_dist.get(label, 0) + 1
                indices_data["train_labeled_label_distribution"] = {int(k): int(v) for k, v in label_dist.items()}
            except Exception as e_label: print(f"Warning: Could not get label distribution for indices: {e_label}")

        if self.partial_unlabeled and self.train_unlabeled_indices is not None:
            indices_data["sample_unlabeled_factor"] = int(self.sample_unlabeled_factor)
            indices_data["train_unlabeled_indices"] = [int(i) for i in self.train_unlabeled_indices]
            # Add label distribution if possible
            try:
                train_labels = self.train_data['label']
                true_label_dist = {}
                for idx in self.train_unlabeled_indices:
                    label = train_labels[int(idx)]
                    true_label_dist[label] = true_label_dist.get(label, 0) + 1
                indices_data["train_unlabeled_true_label_distribution"] = {int(k): int(v) for k, v in true_label_dist.items()}
                
                # Add pseudo label distribution if using pseudo label sampling
                if self.pseudo_label:
                    try:
                        with open(self.pseudo_label_cache_path, "r") as f:
                            pseudo_data = json.load(f)
                        pseudo_label_map = {int(item["index"]): int(item["pseudo_label"]) for item in pseudo_data}
                        # Overall pseudo label cache distribution
                        all_pseudo_labels = list(pseudo_label_map.values())
                        indices_data["pseudo_label_cache_distribution"] = dict(Counter(all_pseudo_labels))
                        # Distribution of sampled pseudo labels
                        sampled_pseudo_labels = [pseudo_label_map[idx] for idx in self.train_unlabeled_indices if idx in pseudo_label_map]
                        indices_data["train_unlabeled_pseudo_label_distribution"] = dict(Counter(sampled_pseudo_labels))
                    except Exception as e:
                        print(f"Warning: Could not compute pseudo-label stats for indices.json: {e}")
            except Exception as e_label: print(f"Warning: Could not get label distribution for indices: {e_label}")

        with open(indices_path, "w") as f:
            json.dump(indices_data, f, indent=2)
        print(f"Selected indices info saved to {indices_path}")
        print(f"Graph saved to {graph_path}")

        return graph_path


    def run_pipeline(self) -> Optional[HeteroData]:
        """Run the complete graph building pipeline."""
        self.load_dataset()
        hetero_graph = self.build_hetero_graph()
        self.analyze_hetero_graph(hetero_graph)
        self.save_graph(hetero_graph)
        print("hetero_graph.x.shape:", hetero_graph['news'].x.shape)
        print("hetero_graph.train_labeled_mask.shape:", hetero_graph['news'].train_labeled_mask.shape)
        print("hetero_graph.train_unlabeled_mask.shape:", hetero_graph['news'].train_unlabeled_mask.shape)
        print("hetero_graph.test_mask.shape:", hetero_graph['news'].test_mask.shape)
        return hetero_graph


# --- Argument Parser ---
def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Build a HETEROGENEOUS graph ('news', 'interaction') for few-shot fake news detection")

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME, choices=["politifact", "gossipcop"], help=f"HuggingFace Dataset (default: {DEFAULT_DATASET_NAME})")
    parser.add_argument("--k_shot", type=int, default=DEFAULT_K_SHOT, choices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], help=f"Number of labeled samples per class (3-16) (default: {DEFAULT_K_SHOT})")

    # Node Feature Args
    parser.add_argument("--embedding_type", type=str, default=DEFAULT_EMBEDDING_TYPE, choices=["bert", "roberta", "distilbert", "bigbird", "deberta"], help=f"Embedding type for 'news' nodes (default: {DEFAULT_EMBEDDING_TYPE})")

    # Edge Policy Args (for 'news'-'similar_to'-'news' edges)
    parser.add_argument("--edge_policy", type=str, default=DEFAULT_EDGE_POLICY, choices=["knn", "label_aware_knn", "mutual_knn", "local_threshold", "global_threshold", "quantile", "topk_mean"], help="Edge policy for 'news'-'news' similarity edges")
    parser.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS, help=f"K for (Mutual) KNN policy (default: {DEFAULT_K_NEIGHBORS})")
    parser.add_argument("--local_threshold_factor", type=float, default=DEFAULT_LOCAL_THRESHOLD_FACTOR, help=f"Factor for 'local_threshold' policy (default: {DEFAULT_LOCAL_THRESHOLD_FACTOR:.1f})")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"Alpha (mean+alpha*std) for 'global_threshold' policy (default: {DEFAULT_ALPHA:.1f})")
    parser.add_argument("--quantile_p", type=float, default=DEFAULT_QUANTILE_P, help=f"Percentile for 'quantile' policy (default: {DEFAULT_QUANTILE_P:.1f})")
    parser.add_argument("--k_top_sim", type=int, default=DEFAULT_K_TOP_SIM, help=f"K for 'topk_mean' policy (default: {DEFAULT_K_TOP_SIM})")

    # Sampling Args
    parser.add_argument("--partial_unlabeled", action="store_true", help="Use only a partial subset of unlabeled nodes. Suffix: partial")
    parser.add_argument("--sample_unlabeled_factor", type=int, default=DEFAULT_SAMPLE_UNLABELED_FACTOR, help="Factor M to sample M*2*k unlabeled training 'news' nodes (default: 10). Used if --partial_unlabeled.")
    parser.add_argument("--pseudo_label", action="store_true", help="Enable pseudo label factor. Suffix: pseudo")
    parser.add_argument("--pseudo_label_cache_path", type=str, default=None, help="Path to pseudo-label cache (json). Default: utils/pseudo_label_cache_<dataset>.json")
    parser.add_argument("--enable_dissimilar", action="store_true", help="Enable dissimilar edge construction. Suffix: dissimilar")
    parser.add_argument("--multi_view", type=int, default=DEFAULT_MULTI_VIEW, help=f"Number of sub-embeddings (views) to split news embeddings into (default: {DEFAULT_MULTI_VIEW})")

    # Interaction Edge Args
    parser.add_argument("--interaction_embedding_field", type=str, default="interaction_embeddings_list", help="Field for interaction embeddings")
    parser.add_argument("--interaction_tone_field", type=str, default="interaction_tones_list", help="Field for interaction tones")
    parser.add_argument("--interaction_edge_mode", type=str, default=DEFAULT_INTERACTION_EDGE_MODE, choices=["edge_type", "edge_attr"], help="How to encode interaction tone: as edge type (edge_type) or as edge_attr (edge_attr)")
    
    # Output & Settings Args
    parser.add_argument("--output_dir", type=str, default=GRAPH_DIR, help=f"Directory to save graphs (default: {GRAPH_DIR})")
    parser.add_argument("--plot", action="store_true", help="Enable graph visualization (EXPERIMENTAL for hetero)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")

    return parser.parse_args()


# --- Main Execution ---
def main() -> None:
    """Main function to run the heterogeneous graph building pipeline."""
    args = parse_arguments()
    set_seed(args.seed)
    
    if args.pseudo_label:
        args.partial_unlabeled = True
        
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    print("\n" + "=" * 60)
    print("   Heterogeneous Fake News Graph Building Pipeline")
    print("=" * 60)
    print(f"Dataset:          {args.dataset_name}")
    print(f"K-Shot:           {args.k_shot}")
    print(f"News Embeddings:  {args.embedding_type}")
    print("-" * 20 + " News-News Edges " + "-" * 20)
    print(f"Policy:           {args.edge_policy}")
    if args.edge_policy in ["knn", "mutual_knn"]: print(f"K neighbors:      {args.k_neighbors}")
    elif args.edge_policy == "local_threshold": print(f"Local Factor:     {args.local_threshold_factor}")
    elif args.edge_policy == "global_threshold": print(f"Alpha:            {args.alpha}")
    elif args.edge_policy == "quantile": print(f"Quantile p:       {args.quantile_p}")
    elif args.edge_policy == "topk_mean":
        print(f"K Top Sim:        {args.k_top_sim}")
        print(f"Threshold Factor: {args.local_threshold_factor}")
    print("-" * 20 + " News Node Sampling " + "-" * 20)
    print(f"Partial Unlabeled: {args.partial_unlabeled}")
    if args.partial_unlabeled: 
        print(f"Sample Factor(M): {args.sample_unlabeled_factor} (target M*2*k nodes)")
        print(f"Pseudo-label Sampling: {args.pseudo_label}")
        if args.pseudo_label:
            print(f"Pseudo-label Cache: {args.pseudo_label_cache_path or f'utils/pseudo_label_cache_{args.dataset_name}.json'}")
    else: 
        print(f"Sample Factor(M): N/A (using all unlabeled train news nodes)")
    print("-" * 20 + " Output & Settings " + "-" * 20)
    print(f"Output directory: {args.output_dir}")
    print(f"Plot:             {args.plot}")
    print(f"Seed:             {args.seed}")
    print(f"Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available(): print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")

    # Instantiate and run the builder
    builder = HeteroGraphBuilder(
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        embedding_type=args.embedding_type,
        edge_policy=args.edge_policy,
        k_neighbors=args.k_neighbors,
        local_threshold_factor=args.local_threshold_factor,
        alpha=args.alpha,
        quantile_p=args.quantile_p,
        k_top_sim=args.k_top_sim,
        partial_unlabeled=args.partial_unlabeled,
        sample_unlabeled_factor=args.sample_unlabeled_factor,
        output_dir=args.output_dir,
        plot=args.plot,
        seed=args.seed,
        pseudo_label=args.pseudo_label,
        pseudo_label_cache_path=args.pseudo_label_cache_path,
        multi_view=args.multi_view,
        enable_dissimilar=args.enable_dissimilar if hasattr(args, 'enable_dissimilar') else False,
        interaction_embedding_field=args.interaction_embedding_field if hasattr(args, 'interaction_embedding_field') else "interaction_embeddings_list",
        interaction_tone_field=args.interaction_tone_field if hasattr(args, 'interaction_tone_field') else "interaction_tones_list",
        interaction_edge_mode=args.interaction_edge_mode if hasattr(args, 'interaction_edge_mode') else "edge_type",
    )

    hetero_graph = builder.run_pipeline()

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print(" Heterogeneous Graph Building Complete")
    print("=" * 60)
    print(f"Graph saved in:      {builder.output_dir}/")
    print(f"  Total Nodes:       {hetero_graph['news'].num_nodes + hetero_graph['interaction'].num_nodes}")
    print(f"    - News Nodes:    {hetero_graph['news'].num_nodes}")
    print(f"    - Interact Nodes:{hetero_graph['interaction'].num_nodes}")
    if args.edge_policy == "label_aware_knn":
        print(f"  Total Edges:       {hetero_graph['news', 'label_aware_similar_to', 'news'].num_edges + hetero_graph['news', 'label_aware_dissimilar_to', 'news'].num_edges + hetero_graph['news', 'has_interaction', 'interaction'].num_edges + hetero_graph['interaction', 'rev_has_interaction', 'news'].num_edges}")
        print(f"    - Label-aware similar KNN Edges: {hetero_graph['news', 'label_aware_similar_to', 'news'].num_edges}")
        print(f"    - Label-aware dissimilar KNN Edges: {hetero_graph['news', 'label_aware_dissimilar_to', 'news'].num_edges}")
        print(f"    - News<->Interact:{hetero_graph['news', 'has_interaction', 'interaction'].num_edges + hetero_graph['interaction', 'rev_has_interaction', 'news'].num_edges}")
    else:
        print(f"  Total Edges:       {hetero_graph['news', 'similar_to', 'news'].num_edges + hetero_graph['news', 'dissimilar_to', 'news'].num_edges + hetero_graph['news', 'has_interaction', 'interaction'].num_edges + hetero_graph['interaction', 'rev_has_interaction', 'news'].num_edges}")
        print(f"    - News<-similar->News:   {hetero_graph['news', 'similar_to', 'news'].num_edges}")
        print(f"    - News<-dissimilar->News:   {hetero_graph['news', 'dissimilar_to', 'news'].num_edges}")
        print(f"    - News<->Interact:{hetero_graph['news', 'has_interaction', 'interaction'].num_edges + hetero_graph['interaction', 'rev_has_interaction', 'news'].num_edges}")
    print("\nNext Steps:")
    print(f"  1. Review the saved graph '.pt' file, metrics '.json' file, and indices '.json' file.")
    print(f"  2. Train a GNN model, e.g.:")
    print(f"  python train_hetero_graph.py --graph_path {os.path.join(builder.output_dir, '<graph_file_name>.pt')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()