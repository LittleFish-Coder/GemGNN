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


# --- Constants ---
GRAPH_DIR = "graphs_hetero" # Separate directory for hetero graphs
PLOT_DIR = "plots_hetero"
DEFAULT_SEED = 42
DEFAULT_K_NEIGHBORS = 5
DEFAULT_EDGE_POLICY = "global_threshold" # For news-news edges
DEFAULT_EMBEDDING_TYPE = "roberta" # Default embedding for news nodes
DEFAULT_LOCAL_THRESHOLD_FACTOR = 1.0
DEFAULT_ALPHA = 0.1
DEFAULT_QUANTILE_P = 95.0
DEFAULT_K_TOP_SIM = 10
DEFAULT_UNLABELED_SAMPLE_FACTOR = 10

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
        sample_unlabeled: bool = False,
        unlabeled_sample_factor: int = DEFAULT_UNLABELED_SAMPLE_FACTOR,
        output_dir: str = GRAPH_DIR,
        plot: bool = False, # Visualization for hetero is complex, maybe disable/simplify later
        seed: int = DEFAULT_SEED,
        device: str = None,
    ):
        """Initialize the HeteroGraphBuilder."""
        self.dataset_name = dataset_name.lower()
        self.k_shot = k_shot
        self.embedding_type = embedding_type
        self.text_embedding_field = f"{embedding_type}_embeddings"
        self.interaction_embedding_field = "interaction_embeddings_list" # Field name from inspection
        self.edge_policy = edge_policy
        self.k_neighbors = k_neighbors
        self.k_top_sim = k_top_sim
        self.local_threshold_factor = local_threshold_factor
        self.alpha = alpha
        self.quantile_p = quantile_p
        self.sample_unlabeled = sample_unlabeled
        self.unlabeled_sample_factor = unlabeled_sample_factor
        self.plot = plot # Keep the flag, but implement visualization carefully
        self.seed = seed

        # Setup directory paths
        self.output_dir = os.path.join(output_dir, self.dataset_name)
        self.plot_dir = os.path.join(PLOT_DIR, self.dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.plot: os.makedirs(self.plot_dir, exist_ok=True)

        # Initialize state
        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.train_size_original = 0
        self.test_size_original = 0
        self.num_classes = 0
        self.total_labeled_size = 0
        self.selected_labeled_indices_original = None
        self.selected_unlabeled_indices_original = None
        self.final_news_indices_original = None # Indices of all news nodes included
        self.test_indices_original = None
        self.news_orig_to_new_idx = None # Mapping for mask creation

        self.graph_metrics = {} # Store analysis results

        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # Set numpy random seed for sampling consistency
        np.random.seed(self.seed)


    def load_dataset(self) -> bool:
        """Load dataset and perform initial checks."""
        print(f"\n[*] Loading dataset: LittleFish-Coder/Fake_News_{self.dataset_name.capitalize()}")
        try:
            # Assuming dataset script doesn't require trust_remote_code=True, adjust if needed
            dataset_dict = load_dataset(f"LittleFish-Coder/Fake_News_{self.dataset_name.capitalize()}",
                                        cache_dir="./hf_cache") # Use cache dir from inspect script
            self.dataset = dataset_dict
            self.train_data = dataset_dict['train']
            self.test_data = dataset_dict['test']
            self.train_size_original = len(self.train_data)
            self.test_size_original = len(self.test_data)

            # Basic validation
            if not all(f in self.train_data.column_names for f in [self.text_embedding_field, self.interaction_embedding_field, 'label']):
                 raise ValueError(f"Missing required fields in train split: {self.text_embedding_field}, {self.interaction_embedding_field}, label")
            if not all(f in self.test_data.column_names for f in [self.text_embedding_field, self.interaction_embedding_field, 'label']):
                 raise ValueError(f"Missing required fields in test split")

            # Get number of classes and calculate total labeled size
            labels = set(self.train_data['label']) | set(self.test_data['label']) # Check both just in case
            self.num_classes = len(labels)
            self.total_labeled_size = self.k_shot * self.num_classes

            print(f"  Original dataset size: Train={self.train_size_original}, Test={self.test_size_original}")
            print(f"  Required Fields Check: OK ({self.text_embedding_field}, {self.interaction_embedding_field}, label)")
            print(f"  Detected Labels: {labels} ({self.num_classes} classes)")
            print(f"  Labeled set config: {self.k_shot}-shot * {self.num_classes} classes = {self.total_labeled_size} total labeled nodes")
            return True

        except Exception as e:
            print(f"[!] Error loading or validating dataset: {e}")
            return False


    def _build_knn_edges(self, embeddings: np.ndarray, k: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_nodes = embeddings.shape[0]
        if num_nodes <= 1: return torch.zeros((2,0), dtype=torch.long), None
        k = min(k, num_nodes - 1) # Adjust k if it's too large
        if k <= 0: return torch.zeros((2,0), dtype=torch.long), None
        print(f"    Building KNN graph (k={k}) for 'news'-'news' edges...")
        try:
             distances = pairwise_distances(embeddings, metric="cosine", n_jobs=-1) # Use multiple cores if available
        except Exception as e:
             print(f"      Error calculating pairwise distances: {e}. Using single core.")
             distances = pairwise_distances(embeddings, metric="cosine")

        rows, cols, data = [], [], []
        for i in tqdm(range(num_nodes), desc=f"      Finding {k} nearest neighbors", leave=False, ncols=100):
            dist_i = distances[i].copy()
            dist_i[i] = np.inf
            # Use argpartition for efficiency
            indices = np.argpartition(dist_i, k)[:k]
            # Ensure indices are valid before adding edges
            valid_indices = indices[np.isfinite(dist_i[indices])] # Filter out potential inf if k >= N-1 somehow
            for j in valid_indices:
                rows.append(i)
                cols.append(j)
                sim = 1.0 - distances[i, j]
                data.append(max(0.0, sim)) # Ensure similarity is non-negative

        if not rows: return torch.zeros((2, 0), dtype=torch.long), None
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        return edge_index, edge_attr

    
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

    # ... Add quantile, topk_mean, mutual_knn if needed ...


    def build_hetero_data(self) -> Optional[HeteroData]:
        """Constructs the HeteroData object."""
        print("\n[*] Starting Heterogeneous Graph Construction...")
        data = HeteroData()

        # --- 1. Select News Nodes (Apply k-shot and sampling) ---
        print("  [1/6] Selecting news nodes (k-shot sampling, unlabeled sampling)...")
        if self.train_data is None or self.test_data is None:
            print("Error: Dataset not loaded properly.")
            return None

        # 1a. Sample labeled nodes from training set
        try:
             labeled_indices_original, _ = sample_k_shot(self.train_data, self.k_shot, self.seed)
             self.selected_labeled_indices_original = np.array(labeled_indices_original, dtype=int)
             print(f"    - Selected {len(self.selected_labeled_indices_original)} labeled nodes (original train indices).")
        except Exception as e:
             print(f"Error during k-shot sampling: {e}")
             return None

        # 1b. Determine final set of training news indices to include
        self.final_news_indices_original = None
        self.selected_unlabeled_indices_original = np.array([], dtype=int)

        all_train_indices_original = np.arange(self.train_size_original)

        if self.sample_unlabeled:
            print(f"    - Sampling unlabeled training nodes (Factor M={self.unlabeled_sample_factor})...")
            unlabeled_indices_original = np.setdiff1d(all_train_indices_original, self.selected_labeled_indices_original, assume_unique=True)
            num_unlabeled_available = len(unlabeled_indices_original)
            num_to_sample = int(self.unlabeled_sample_factor * self.total_labeled_size)
            num_to_sample = min(num_to_sample, num_unlabeled_available) # Cap at available

            if num_unlabeled_available == 0 or num_to_sample <= 0:
                 print("    - Warning: No unlabeled nodes available or sample size is 0. Only using labeled train nodes.")
                 self.final_news_indices_original = self.selected_labeled_indices_original # Only labeled nodes
            else:
                print(f"    - Sampling {num_to_sample} from {num_unlabeled_available} available unlabeled nodes...")
                rng = np.random.default_rng(self.seed) # Use seeded generator
                self.selected_unlabeled_indices_original = rng.choice(unlabeled_indices_original, size=num_to_sample, replace=False)
                self.final_news_indices_original = np.concatenate([self.selected_labeled_indices_original, self.selected_unlabeled_indices_original])
        else:
            print("    - Using ALL training nodes (sampling disabled).")
            self.final_news_indices_original = all_train_indices_original # Use all original train indices

        # Ensure uniqueness just in case, and sort
        self.final_news_indices_original = np.unique(self.final_news_indices_original)
        print(f"    - Total training news nodes in graph: {len(self.final_news_indices_original)}")

        # 1c. Get all original test indices
        self.test_indices_original = np.arange(self.test_size_original)
        print(f"    - Total test news nodes in graph: {len(self.test_indices_original)}")

        # Store the set of original train indices included for quick lookup later
        self.train_indices_original_set = set(self.final_news_indices_original)

        # Combine all original indices (train subset + test) that will form the 'news' nodes
        all_included_news_original_indices = np.concatenate([self.final_news_indices_original, self.test_indices_original])
        # Create mapping from original index (train or test) to new 'news' node index
        # Note: Test indices need offset when accessing self.dataset['test']
        self.news_orig_to_new_idx = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.final_news_indices_original)}
        # Add test indices to map, their original index is 0 to N_test-1
        for i, orig_test_idx in enumerate(self.test_indices_original):
             self.news_orig_to_new_idx[f"test_{orig_test_idx}"] = len(self.final_news_indices_original) + i


        num_total_news_nodes = len(all_included_news_original_indices)
        print(f"    - Total 'news' nodes to be created: {num_total_news_nodes}")


        # --- 2. Prepare 'news' Node Features, Labels, Masks ---
        print(f"  [2/6] Preparing features, labels, masks for {num_total_news_nodes} 'news' nodes...")
        news_features = []
        news_labels = []

        # Extract data for the selected training nodes
        train_data_subset = self.train_data.select(self.final_news_indices_original.tolist())
        news_features.append(np.array(train_data_subset[self.text_embedding_field]))
        news_labels.append(np.array(train_data_subset['label']))

        # Extract data for all test nodes
        news_features.append(np.array(self.test_data[self.text_embedding_field]))
        news_labels.append(np.array(self.test_data['label']))

        # Concatenate features and labels
        try:
             # Handle cases where one part might be empty if k_shot=0 or no test set etc.
             if not news_features[0].size: # If no train nodes selected
                 final_news_features = news_features[1]
             elif not news_features[1].size: # If no test nodes
                 final_news_features = news_features[0]
             else:
                  # Ensure dimensions match before concatenating
                  if news_features[0].shape[1] != news_features[1].shape[1]:
                       raise ValueError(f"Feature dimension mismatch between train ({news_features[0].shape[1]}) and test ({news_features[1].shape[1]}) for field '{self.text_embedding_field}'")
                  final_news_features = np.concatenate(news_features, axis=0)

             if not news_labels[0].size:
                 final_news_labels = news_labels[1]
             elif not news_labels[1].size:
                 final_news_labels = news_labels[0]
             else:
                  final_news_labels = np.concatenate(news_labels, axis=0)

        except ValueError as e:
             print(f"Error concatenating features/labels: {e}")
             # Print shapes for debugging
             print("Train feature shape:", news_features[0].shape if news_features else 'N/A')
             print("Test feature shape:", news_features[1].shape if len(news_features)>1 else 'N/A')
             print("Train label shape:", news_labels[0].shape if news_labels else 'N/A')
             print("Test label shape:", news_labels[1].shape if len(news_labels)>1 else 'N/A')
             return None
        except Exception as e:
             print(f"Unexpected error processing features/labels: {e}")
             return None


        data['news'].x = torch.tensor(final_news_features, dtype=torch.float)
        data['news'].y = torch.tensor(final_news_labels, dtype=torch.long)
        data['news'].num_nodes = num_total_news_nodes

        # Create masks
        num_final_train_news = len(self.final_news_indices_original)
        train_mask = torch.zeros(num_total_news_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_total_news_nodes, dtype=torch.bool)
        labeled_mask = torch.zeros(num_total_news_nodes, dtype=torch.bool)

        train_mask[:num_final_train_news] = True
        test_mask[num_final_train_news:] = True

        # Map original labeled indices to new indices within the 'news' node set
        for orig_labeled_idx in self.selected_labeled_indices_original:
            if orig_labeled_idx in self.news_orig_to_new_idx: # Check if labeled node is included
                new_idx = self.news_orig_to_new_idx[orig_labeled_idx]
                if new_idx < num_total_news_nodes: # Bounds check
                     labeled_mask[new_idx] = True
                else: print(f"Warning: Labeled index mapping error for {orig_labeled_idx}") # Should not happen

        data['news'].train_mask = train_mask
        data['news'].test_mask = test_mask
        data['news'].labeled_mask = labeled_mask
        print(f"    - 'news' features shape: {data['news'].x.shape}")
        print(f"    - 'news' masks created: Train={train_mask.sum()}, Test={test_mask.sum()}, Labeled={labeled_mask.sum()}")


        # --- 3. Prepare 'interaction' Node Features ---
        num_interactions_per_news = 20 # As specified
        num_total_interaction_nodes = num_total_news_nodes * num_interactions_per_news
        print(f"  [3/6] Preparing features for {num_total_interaction_nodes} 'interaction' nodes...")

        all_interaction_embeddings = []
        # This part can be slow if datasets are large, consider optimization if needed
        pbar_interact = tqdm(total=num_total_news_nodes, desc="    Extracting Interaction Embeddings", ncols=100)
        interaction_feature_dim = -1

        # Iterate through the *order* of nodes as they appear in data['news'].x
        # First part: selected train nodes
        for i, orig_idx in enumerate(self.final_news_indices_original):
            try:
                embeddings_list = self.train_data[int(orig_idx)][self.interaction_embedding_field]
                if len(embeddings_list) != num_interactions_per_news:
                    # Handle inconsistent number of interactions - Option: Pad/Truncate or Skip/Error
                    # Simple approach: Pad with zeros or truncate if size is wrong
                    current_embeddings = np.array(embeddings_list)
                    if interaction_feature_dim == -1 and current_embeddings.size > 0:
                        interaction_feature_dim = current_embeddings.shape[-1] # Get dim from first valid entry
                    if len(embeddings_list) > num_interactions_per_news:
                        embeddings_list = embeddings_list[:num_interactions_per_news]
                        print(f"Warning: Truncated interactions for train news index {orig_idx}")
                    elif len(embeddings_list) < num_interactions_per_news:
                        if interaction_feature_dim != -1:
                             padding = np.zeros((num_interactions_per_news - len(embeddings_list), interaction_feature_dim))
                             embeddings_list = np.vstack((current_embeddings, padding))
                             print(f"Warning: Padded interactions for train news index {orig_idx}")
                        else: # Cannot determine padding dimension yet
                             print(f"Warning: Cannot pad interactions for train news {orig_idx} (dim unknown), skipping padding.")
                             # Need a better strategy if first item has wrong size

                # Convert list of lists/arrays to a numpy array for this news item
                node_interactions = np.array(embeddings_list)
                # Infer dimension if first time and valid
                if interaction_feature_dim == -1 and node_interactions.size > 0:
                    interaction_feature_dim = node_interactions.shape[-1]
                # Ensure consistent shape [20, dim] - might need padding/truncating again if array conversion failed
                if node_interactions.shape != (num_interactions_per_news, interaction_feature_dim) and interaction_feature_dim != -1 :
                     print(f"Error: Interaction shape mismatch for train news {orig_idx}. Expected ({num_interactions_per_news}, {interaction_feature_dim}), got {node_interactions.shape}. Skipping.")
                     # Add placeholder zeros? For now, error out or use placeholder might be safer if unexpected
                     node_interactions = np.zeros((num_interactions_per_news, interaction_feature_dim)) # Placeholder

                all_interaction_embeddings.append(node_interactions)
                pbar_interact.update(1)
            except Exception as e:
                print(f"Error processing interactions for original train index {orig_idx}: {e}")
                # Append placeholder?
                if interaction_feature_dim != -1:
                     all_interaction_embeddings.append(np.zeros((num_interactions_per_news, interaction_feature_dim)))
                     pbar_interact.update(1)
                else:
                     print("Error: Cannot determine interaction feature dimension to create placeholder.")
                     return None # Fatal error if we can't get dimensions

        # Second part: test nodes
        for i, orig_idx in enumerate(self.test_indices_original):
            try:
                embeddings_list = self.test_data[int(orig_idx)][self.interaction_embedding_field]
                # Apply same padding/truncating logic as for train
                if len(embeddings_list) != num_interactions_per_news:
                    current_embeddings = np.array(embeddings_list)
                    if interaction_feature_dim == -1 and current_embeddings.size > 0:
                         interaction_feature_dim = current_embeddings.shape[-1]
                    if len(embeddings_list) > num_interactions_per_news:
                         embeddings_list = embeddings_list[:num_interactions_per_news]
                         print(f"Warning: Truncated interactions for test news index {orig_idx}")
                    elif len(embeddings_list) < num_interactions_per_news:
                         if interaction_feature_dim != -1:
                             padding = np.zeros((num_interactions_per_news - len(embeddings_list), interaction_feature_dim))
                             embeddings_list = np.vstack((current_embeddings, padding))
                             print(f"Warning: Padded interactions for test news index {orig_idx}")
                         else: print(f"Warning: Cannot pad interactions for test news {orig_idx} (dim unknown).")

                node_interactions = np.array(embeddings_list)
                if interaction_feature_dim == -1 and node_interactions.size > 0:
                    interaction_feature_dim = node_interactions.shape[-1]
                if node_interactions.shape != (num_interactions_per_news, interaction_feature_dim) and interaction_feature_dim != -1:
                     print(f"Error: Interaction shape mismatch for test news {orig_idx}. Expected ({num_interactions_per_news}, {interaction_feature_dim}), got {node_interactions.shape}. Skipping.")
                     node_interactions = np.zeros((num_interactions_per_news, interaction_feature_dim))

                all_interaction_embeddings.append(node_interactions)
                pbar_interact.update(1)
            except Exception as e:
                print(f"Error processing interactions for original test index {orig_idx}: {e}")
                if interaction_feature_dim != -1:
                     all_interaction_embeddings.append(np.zeros((num_interactions_per_news, interaction_feature_dim)))
                     pbar_interact.update(1)
                else:
                     print("Error: Cannot determine interaction feature dimension to create placeholder.")
                     return None
        pbar_interact.close()

        # Check if interaction_feature_dim was determined
        if interaction_feature_dim == -1:
             print("Error: Could not determine the feature dimension for interaction nodes.")
             return None

        # Stack all interaction embeddings
        try:
            final_interaction_features = np.vstack(all_interaction_embeddings)
            data['interaction'].x = torch.tensor(final_interaction_features, dtype=torch.float)
            data['interaction'].num_nodes = data['interaction'].x.shape[0]
            print(f"    - 'interaction' features shape: {data['interaction'].x.shape}")
            # Sanity check node count
            if data['interaction'].num_nodes != num_total_interaction_nodes:
                 print(f"Warning: Interaction node count mismatch! Expected {num_total_interaction_nodes}, Got {data['interaction'].num_nodes}")
        except Exception as e:
             print(f"Error stacking interaction embeddings: {e}")
             # Print shapes of collected embeddings for debugging
             print("Shapes collected:", [arr.shape for arr in all_interaction_embeddings if hasattr(arr, 'shape')])
             return None


        # --- 4. Create Edges ---
        print(f"  [4/6] Creating graph edges...")
        # 4a. ('news', 'has_interaction', 'interaction') Edges
        # Source nodes: 0, 0, ..., 0 (20 times), 1, 1, ..., 1 (20 times), ... N-1, ... N-1 (20 times)
        news_has_interaction_src = torch.arange(num_total_news_nodes).repeat_interleave(num_interactions_per_news)
        # Target nodes: 0, 1, 2, ..., 19, 20, 21, ..., 39, ...
        interaction_has_interaction_tgt = torch.arange(num_total_interaction_nodes)
        data['news', 'has_interaction', 'interaction'].edge_index = torch.stack([news_has_interaction_src, interaction_has_interaction_tgt], dim=0)
        print(f"    - Created {data['news', 'has_interaction', 'interaction'].edge_index.shape[1]} 'news -> interaction' edges.")

        # 4b. Reverse Edges ('interaction', 'rev_has_interaction', 'news')
        data['interaction', 'rev_has_interaction', 'news'].edge_index = torch.stack([interaction_has_interaction_tgt, news_has_interaction_src], dim=0)
        print(f"    - Created {data['interaction', 'rev_has_interaction', 'news'].edge_index.shape[1]} 'interaction -> news' edges.")

        # 4c. ('news', 'similar_to', 'news') Edges
        print(f"    - Building 'news'-'similar_to'-'news' edges using policy: '{self.edge_policy}'...")
        news_embeddings_np = data['news'].x.cpu().numpy() # Use news node embeddings

        # Call the appropriate helper function based on self.edge_policy
        edge_func = None
        args_nn = [news_embeddings_np]
        if self.edge_policy == 'knn':
            edge_func = self._build_knn_edges
            args_nn.append(self.k_neighbors)
        elif self.edge_policy == 'local_threshold':
             edge_func = self._build_local_threshold_edges
             args_nn.append(self.local_threshold_factor)
        elif self.edge_policy == 'global_threshold':
             edge_func = self._build_global_threshold_edges
             args_nn.append(self.alpha)
        elif self.edge_policy == 'mutual_knn':
             edge_func = self._build_mutual_knn_edges
             args_nn.append(self.k_neighbors)
        elif self.edge_policy == 'quantile':
             edge_func = self._build_quantile_edges
             args_nn.append(self.quantile_p)
        elif self.edge_policy == 'topk_mean':
             edge_func = self._build_topk_mean_edges
             args_nn.extend([self.k_top_sim, self.local_threshold_factor])
        else:
            print(f"Warning: News-news edge policy '{self.edge_policy}' not fully implemented in helpers yet. Skipping news-news edges.")
            edge_index_nn = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_nn = None

        if edge_func:
            try:
                edge_index_nn, edge_attr_nn = edge_func(*args_nn)
            except Exception as e:
                print(f"      Error during '{self.edge_policy}' edge building: {e}")
                edge_index_nn = torch.zeros((2, 0), dtype=torch.long)
                edge_attr_nn = None
        else: # Handle case where policy name is valid but function isn't implemented
             if self.edge_policy not in ['knn', 'local_threshold', 'global_threshold', 'mutual_knn', 'quantile', 'topk_mean']:
                 print(f"      Warning: Edge policy '{self.edge_policy}' function not implemented. Skipping news-news edges.")
             edge_index_nn = torch.zeros((2, 0), dtype=torch.long)
             edge_attr_nn = None


        data['news', 'similar_to', 'news'].edge_index = edge_index_nn
        if edge_attr_nn is not None:
            data['news', 'similar_to', 'news'].edge_attr = edge_attr_nn
        print(f"    - Created {edge_index_nn.shape[1]} 'news <-> news' edges.")

        # --- 5. Move Data to Device ---
        print(f"  [5/6] Moving HeteroData to device: {self.device}...")
        try:
             data = data.to(self.device)
        except Exception as e:
             print(f"    Error moving data to device: {e}. Keeping on CPU.")
             # Potentially fallback to CPU if CUDA fails
             self.device = 'cpu'


        # --- 6. Final Validation ---
        print("  [6/6] Validating HeteroData object...")
        try:
            # Basic checks
            assert data['news'].num_nodes == num_total_news_nodes
            assert data['interaction'].num_nodes == num_total_interaction_nodes
            assert data['news', 'has_interaction', 'interaction'].num_edges == num_total_interaction_nodes
            print("    Basic structure checks passed.")
        except Exception as e:
            print(f"    Error during validation: {e}")
            return None # Return None if validation fails

        print("\n[+] Heterogeneous graph construction complete.")
        return data

    def _analyze_hetero_graph(self, hetero_data: HeteroData) -> None:
        """Performs and prints basic analysis of the HeteroData object."""
        print("\n" + "=" * 60)
        print("     Heterogeneous Graph Analysis (Basic)")
        print("=" * 60)
        if hetero_data is None:
            print("No data to analyze.")
            self.graph_metrics = {"status": "No HeteroData object found"}
            return

        self.graph_metrics = {} # Reset metrics

        # Node Analysis
        print("\n--- Node Types ---")
        total_nodes = 0
        for node_type in hetero_data.node_types:
            num_nodes = hetero_data[node_type].num_nodes
            total_nodes += num_nodes
            print(f"[*] Node Type: '{node_type}'")
            print(f"  - Num Nodes: {num_nodes}")
            # Features
            if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
                print(f"  - Features Dim: {hetero_data[node_type].x.shape[1]}")
            else: print("  - Features: None")
            # Labels & Masks (typically only on 'news')
            if hasattr(hetero_data[node_type], 'y') and hetero_data[node_type].y is not None:
                print(f"  - Labels Shape: {hetero_data[node_type].y.shape}")
            for mask in ['train_mask', 'test_mask', 'labeled_mask']:
                if hasattr(hetero_data[node_type], mask) and hetero_data[node_type][mask] is not None:
                     count = hetero_data[node_type][mask].sum().item()
                     print(f"  - {mask}: {count} nodes ({count/num_nodes*100:.1f}% of '{node_type}')")
            self.graph_metrics[f'nodes_{node_type}'] = num_nodes
        print(f"Total Nodes (all types): {total_nodes}")
        self.graph_metrics['nodes_total'] = total_nodes


        # Edge Analysis
        print("\n--- Edge Types ---")
        total_edges = 0
        for edge_type in hetero_data.edge_types:
            num_edges = hetero_data[edge_type].num_edges
            total_edges += num_edges
            edge_type_str = " -> ".join(edge_type) if isinstance(edge_type, tuple) else edge_type
            print(f"[*] Edge Type: {edge_type_str}")
            print(f"  - Num Edges: {num_edges}")
            # Edge Attributes
            if hasattr(hetero_data[edge_type], 'edge_attr') and hetero_data[edge_type].edge_attr is not None:
                print(f"  - Attributes Dim: {hetero_data[edge_type].edge_attr.shape[1]}")
            else: print("  - Attributes: None")
            self.graph_metrics[f'edges_{"_".join(edge_type)}'] = num_edges
        print(f"Total Edges (all types): {total_edges}")
        self.graph_metrics['edges_total'] = total_edges

        # Specific Analysis for 'news'-'similar_to'-'news' (if exists)
        news_news_edge_type = ('news', 'similar_to', 'news')
        if news_news_edge_type in hetero_data.edge_types:
            print("\n--- Analysis for 'news'-'similar_to'-'news' Edges ---")
            nn_edge_index = hetero_data[news_news_edge_type].edge_index
            num_nn_edges = hetero_data[news_news_edge_type].num_edges
            num_news_nodes = hetero_data['news'].num_nodes

            # Avg Degree for news nodes (from these edges only)
            if num_news_nodes > 0:
                degrees_nn = torch.zeros(num_news_nodes, dtype=torch.long, device=nn_edge_index.device)
                # Compute undirected degree for avg degree calculation
                degrees_nn.scatter_add_(0, nn_edge_index[0], torch.ones_like(nn_edge_index[0]))
                degrees_nn.scatter_add_(0, nn_edge_index[1], torch.ones_like(nn_edge_index[1]))
                avg_degree_nn = degrees_nn.float().mean().item() / 2.0 # Divide by 2 for undirected avg
                print(f"  Avg Degree ('news' nodes via 'similar_to'): {avg_degree_nn:.2f}")
                self.graph_metrics['avg_degree_news_similar_to'] = avg_degree_nn

            # Homophily for news-news edges
            if num_nn_edges > 0 and hasattr(hetero_data['news'], 'y'):
                y_news = hetero_data['news'].y.cpu().numpy()
                edge_index_nn_cpu = nn_edge_index.cpu().numpy()
                src_labels = y_news[edge_index_nn_cpu[0]]
                tgt_labels = y_news[edge_index_nn_cpu[1]]
                homophilic_nn_edges = np.sum(src_labels == tgt_labels)
                # Need number of unique undirected edges for ratio
                # Approximating with num_nn_edges / 2 if symmetric, but safer to convert to NX temporarily
                try:
                    G_nn = nx.Graph() # Undirected graph for news-news
                    G_nn.add_nodes_from(range(num_news_nodes))
                    G_nn.add_edges_from(edge_index_nn_cpu.T)
                    num_undirected_nn_edges = G_nn.number_of_edges()
                    if num_undirected_nn_edges > 0:
                        # Note: homophilic_nn_edges counted directed edges, need adjustment for undirected ratio
                        # Simplified: Assume most edges are symmetric, divide directed count by 2? Risky.
                        # Recalculate homophilic based on undirected edges
                        homophilic_undirected_nn = 0
                        for u, v in G_nn.edges():
                             if y_news[u] == y_news[v]:
                                 homophilic_undirected_nn += 1

                        homophily_ratio_nn = homophilic_undirected_nn / num_undirected_nn_edges
                        print(f"  Homophily Ratio ('news'-'similar_to'-'news'): {homophily_ratio_nn:.4f}")
                        self.graph_metrics['homophily_ratio_news_similar_to'] = homophily_ratio_nn
                    else: print("  Homophily Ratio ('news'-'similar_to'-'news'): N/A (no unique undirected edges)")

                except Exception as e_homo:
                     print(f"  Warning: Could not accurately calculate homophily for news-news edges: {e_homo}")
            else:
                print("  Homophily Ratio ('news'-'similar_to'-'news'): N/A (no edges or labels)")

        # Add Sampling Info to metrics
        self.graph_metrics['sampling_info'] = {
            "sampled_unlabeled": self.sample_unlabeled,
            "unlabeled_sample_factor": self.unlabeled_sample_factor if self.sample_unlabeled else None,
            "num_labeled_original": len(self.selected_labeled_indices_original) if self.selected_labeled_indices_original is not None else 'N/A',
            "num_unlabeled_sampled_original": len(self.selected_unlabeled_indices_original) if self.selected_unlabeled_indices_original is not None else 'N/A',
        }
        print("\n--- Sampling Info ---")
        print(f"  Unlabeled Sampling Enabled: {self.sample_unlabeled}")
        if self.sample_unlabeled: print(f"  Unlabeled Sample Factor M: {self.unlabeled_sample_factor}")


        print("=" * 60)
        print("      End of Heterogeneous Graph Analysis")
        print("=" * 60 + "\n")


    def save_graph(self, hetero_data: HeteroData) -> Optional[str]:
        """Save the HeteroData graph and analysis results."""
        if hetero_data is None:
            print("Error: No HeteroData object to save.")
            return None

        # --- Generate graph name ---
        edge_policy_name = self.edge_policy
        edge_param_str = ""
        # Suffix based on news-news edge policy params
        if self.edge_policy in ["knn", "mutual_knn"]: edge_param_str = str(self.k_neighbors)
        elif self.edge_policy == "local_threshold": edge_param_str = f"{self.local_threshold_factor:.2f}".replace('.', 'p')
        elif self.edge_policy == "global_threshold": edge_param_str = f"{self.alpha:.2f}".replace('.', 'p')
        elif self.edge_policy == "quantile": edge_param_str = f"{self.quantile_p:.1f}".replace('.', 'p')
        elif self.edge_policy == "topk_mean": edge_param_str = f"k{self.k_top_sim}_f{self.local_threshold_factor:.2f}".replace('.', 'p')
        else: edge_policy_name, edge_param_str = "unknown", "NA"

        sampling_suffix = f"_smpf{self.unlabeled_sample_factor}" if self.sample_unlabeled else "_all"

        # Include text embedding type in name
        graph_name = f"{self.k_shot}_shot_{self.embedding_type}_hetero_{edge_policy_name}_{edge_param_str}{sampling_suffix}"
        # --- End filename generation ---

        # Save graph data (move to CPU first)
        graph_path = os.path.join(self.output_dir, f"{graph_name}.pt")
        try:
            print(f"[*] Saving HeteroData graph to {graph_path}...")
            cpu_graph_data = hetero_data.cpu()
            torch.save(cpu_graph_data, graph_path)
            print(f"  Graph saved successfully.")
        except Exception as e:
            print(f"  Error saving graph: {e}")
            return None

        # Save graph metrics (simplified for hetero)
        metrics_path = os.path.join(self.output_dir, f"{graph_name}_metrics.json")
        def default_serializer(obj): # Helper for JSON serialization
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
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
            "sampled_unlabeled": self.sample_unlabeled,
            "embedding_type": self.embedding_type,
            "news_news_edge_policy": self.edge_policy,
        }
        if self.selected_labeled_indices_original is not None:
             indices_data["labeled_indices_original"] = [int(i) for i in self.selected_labeled_indices_original]
             # Add label distribution if possible
             try:
                  train_labels = self.train_data['label']
                  label_dist = {}
                  for idx in self.selected_labeled_indices_original:
                       label = train_labels[int(idx)]
                       label_dist[label] = label_dist.get(label, 0) + 1
                  indices_data["labeled_label_distribution"] = {int(k): int(v) for k, v in label_dist.items()}
             except Exception as e_label: print(f"Warning: Could not get label distribution for indices: {e_label}")

        if self.sample_unlabeled and self.selected_unlabeled_indices_original is not None:
             indices_data["unlabeled_sample_factor"] = int(self.unlabeled_sample_factor)
             indices_data["unlabeled_indices_original_sampled"] = [int(i) for i in self.selected_unlabeled_indices_original]

        indices_path = os.path.join(self.output_dir, f"{graph_name}_indices.json")
        try:
             with open(indices_path, "w") as f:
                 json.dump(indices_data, f, indent=2)
             print(f"  Selected indices info saved to {indices_path}")
        except Exception as e:
             print(f"  Error saving indices JSON: {e}")

        return graph_path


    def run_pipeline(self) -> Optional[HeteroData]:
        """Run the complete graph building pipeline."""
        if not self.load_dataset():
            return None
        hetero_data = self.build_hetero_data()
        if hetero_data:
            self._analyze_hetero_graph(hetero_data)
            self.save_graph(hetero_data)
            return hetero_data
        else:
            print("\n[!] Graph building failed. No graph saved.")
            return None


# --- Argument Parser ---
def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Build a HETEROGENEOUS graph ('news', 'interaction') for few-shot fake news detection")

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="politifact", choices=["politifact", "gossipcop"], help="Dataset name suffix")
    parser.add_argument("--k_shot", type=int, default=8, choices=list(range(3, 21)), help="Number of labeled samples per class (3-20)")

    # Node Feature Args
    parser.add_argument("--embedding_type", type=str, default=DEFAULT_EMBEDDING_TYPE, choices=["bert", "roberta"], help="Embedding type for 'news' nodes")

    # Edge Policy Args (for 'news'-'similar_to'-'news' edges)
    parser.add_argument("--edge_policy", type=str, default=DEFAULT_EDGE_POLICY, choices=["knn", "mutual_knn", "local_threshold", "global_threshold", "quantile", "topk_mean"], help="Edge policy for 'news'-'news' similarity edges")
    parser.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS, help="K for (Mutual) KNN policy")
    parser.add_argument("--local_threshold_factor", type=float, default=DEFAULT_LOCAL_THRESHOLD_FACTOR, help=f"Factor for 'local_threshold' policy (default: {DEFAULT_LOCAL_THRESHOLD_FACTOR:.1f})")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"Alpha (mean+alpha*std) for 'global_threshold' policy (default: {DEFAULT_ALPHA:.1f})")
    parser.add_argument("--quantile_p", type=float, default=DEFAULT_QUANTILE_P, help=f"Percentile for 'quantile' policy (default: {DEFAULT_QUANTILE_P:.1f})")
    parser.add_argument("--k_top_sim", type=int, default=DEFAULT_K_TOP_SIM, help="K for 'topk_mean' policy (default: 10)")

    # Unlabeled Node Sampling Args
    parser.add_argument("--sample_unlabeled", action='store_true', default=False, help="Enable sampling of unlabeled training 'news' nodes.")
    parser.add_argument("--unlabeled_sample_factor", type=int, default=DEFAULT_UNLABELED_SAMPLE_FACTOR, help=f"Factor M to sample M*2*k unlabeled training 'news' nodes (default: {DEFAULT_UNLABELED_SAMPLE_FACTOR}). Used if --sample_unlabeled.")

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
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    print("\n" + "=" * 60)
    print("   Heterogeneous Fake News Graph Building Pipeline")
    print("=" * 60)
    print(f"Dataset:          {args.dataset_name}")
    print(f"K-Shot:           {args.k_shot}")
    print(f"News Embeddings:  {args.embedding_type}")
    print(f"Interaction Embeds: From 'interaction_embeddings_list' (20 per news)")
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
    print(f"Sample Unlabeled: {args.sample_unlabeled}")
    if args.sample_unlabeled: print(f"Sample Factor(M): {args.unlabeled_sample_factor} (target M*2*k nodes)")
    else: print(f"Sample Factor(M): N/A (using all unlabeled train news nodes)")
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
        sample_unlabeled=args.sample_unlabeled,
        unlabeled_sample_factor=args.unlabeled_sample_factor,
        output_dir=args.output_dir,
        plot=args.plot,
        seed=args.seed,
    )

    hetero_data = builder.run_pipeline()

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print(" Heterogeneous Graph Building Complete")
    print("=" * 60)
    if hetero_data and builder.graph_metrics:
         graph_path = os.path.join(builder.output_dir, "<graph_file_name>.pt") # Placeholder
         print(f"Graph saved in:      {builder.output_dir}/")
         print(f"  Total Nodes:       {builder.graph_metrics.get('nodes_total', 'N/A')}")
         print(f"    - News Nodes:    {builder.graph_metrics.get('nodes_news', 'N/A')}")
         print(f"    - Interact Nodes:{builder.graph_metrics.get('nodes_interaction', 'N/A')}")
         print(f"  Total Edges:       {builder.graph_metrics.get('edges_total', 'N/A')}")
         print(f"    - News<->News:   {builder.graph_metrics.get('edges_news_similar_to_news', 'N/A')}")
         print(f"    - News<->Interact:{builder.graph_metrics.get('edges_news_has_interaction_interaction', 'N/A') + builder.graph_metrics.get('edges_interaction_rev_has_interaction_news', 'N/A')}")
         print("\nNext Steps:")
         print(f"  1. Review the saved graph '.pt' file, metrics '.json' file, and indices '.json' file.")
         print(f"  2. Implement/adapt 'train_hetero_graph.py' to load and train on this HeteroData object.")
         print(f"     Example command: python train_hetero_graph.py --graph_path {graph_path}")
    else:
        print("Graph building may have failed or produced no output.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()