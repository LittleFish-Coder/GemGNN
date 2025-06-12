import os
import gc
import json
import numpy as np
import torch
import networkx as nx
from typing import Dict, Tuple, Optional, List, Union, Any
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm.auto import tqdm
from argparse import ArgumentParser
from utils.sample_k_shot import sample_k_shot
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# --- Constants ---
DEFAULT_K_SHOT = 8                                  # 3-16 shot
DEFAULT_DATASET_NAME = "politifact"                 # politifact, gossipcop
DEFAULT_EMBEDDING_TYPE = "roberta"                  # Default embedding for news nodes (bert, distilbert, roberta, combined)
# --- Edge Policies Parameters ---
DEFAULT_EDGE_POLICY = "knn"                         # For news-news edges (knn only)
DEFAULT_K_NEIGHBORS = 5                             # For knn edge policy
# --- Unlabeled Node Sampling Parameters ---
DEFAULT_SAMPLE_UNLABELED_FACTOR = 10                # for unlabeled node sampling (train_unlabeled_nodes = num_classes * k_shot * sample_unlabeled_factor)
# --- Graphs and Plots Directories ---
DEFAULT_SEED = 42
DEFAULT_DATASET_CACHE_DIR = "dataset"
DEFAULT_GRAPH_DIR = "graphs"

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


# --- GraphBuilder Class ---
class GraphBuilder:
    """
    Builds homogeneous graph datasets for few-shot fake news detection.
    """

    def __init__(
        self,
        dataset_name: str,
        k_shot: int,
        embedding_type: str = DEFAULT_EMBEDDING_TYPE,
        edge_policy: str = DEFAULT_EDGE_POLICY,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        partial_unlabeled: bool = False,
        sample_unlabeled_factor: int = DEFAULT_SAMPLE_UNLABELED_FACTOR,
        pseudo_label: bool = False,
        pseudo_label_cache_path: str = None,
        dataset_cache_dir: str = DEFAULT_DATASET_CACHE_DIR,
        seed: int = DEFAULT_SEED,
        output_dir: str = DEFAULT_GRAPH_DIR,
    ):
        """Initialize the GraphBuilder."""
        self.dataset_name = dataset_name.lower()
        self.k_shot = k_shot
        self.embedding_type = embedding_type
        self.text_embedding_field = f"{embedding_type}_embeddings"
        self.edge_policy = edge_policy
        self.k_neighbors = k_neighbors
        ## Sampling
        self.partial_unlabeled = partial_unlabeled
        self.sample_unlabeled_factor = sample_unlabeled_factor
        self.pseudo_label = pseudo_label
        if pseudo_label_cache_path:
            self.pseudo_label_cache_path = pseudo_label_cache_path
        else:
            self.pseudo_label_cache_path = f"utils/pseudo_label_cache_{self.dataset_name}.json"
        self.seed = seed
        self.dataset_cache_dir = dataset_cache_dir

        if self.pseudo_label:
            self.partial_unlabeled = True

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        np.random.seed(self.seed)
        # Setup directory paths
        self.output_dir = os.path.join(output_dir, self.dataset_name)
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        # Initialize state
        self.dataset = None
        self.graph_metrics = {}
        # Selected Indices
        self.train_labeled_indices = None
        self.train_unlabeled_indices = None
        self.test_indices = None
        self.graph_metrics = {} # Store analysis results

    def load_dataset(self) -> None:
        """Load dataset and perform initial checks."""
        print(f"Loading dataset '{self.dataset_name}' with '{self.embedding_type}' embeddings...")
        hf_dataset_name = f"LittleFish-Coder/Fake_News_{self.dataset_name}"
        # download from huggingface and cache to local path
        local_hf_dir = os.path.join(self.dataset_cache_dir, f"{self.dataset_name}_hf")
        if os.path.exists(local_hf_dir):
            print(f"Loading dataset from local path: {local_hf_dir}")
            dataset = load_from_disk(local_hf_dir)
        else:
            print(f"Loading dataset from huggingface: {hf_dataset_name}")
            dataset = load_dataset(hf_dataset_name, download_mode="reuse_cache_if_exists", cache_dir=local_hf_dir)
            dataset.save_to_disk(local_hf_dir)

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
        print(f"  Train labeled set: {self.k_shot}-shot * {self.num_classes} classes = {self.total_labeled_size} total labeled nodes")

    def build_graph(self) -> Data:
        """
        Build a graph including both nodes and edges.
        Pipeline:
        1. Build nodes
            - train_labeled_nodes (train_labeled_mask from training set)
            - train_unlabeled_nodes (train_unlabeled_mask from training set)
            - test_nodes (test_mask from test set)
        2. Build edges (based on edge policy)
        3. Update graph data with edges
        """
        print("\nStarting Graph Construction...")

        data = Data()

        # --- Select News Nodes (k-shot train labeled nodes, train unlabeled nodes, test nodes) ---
        # 1. Sample k-shot labeled nodes from train set (with cache)
        print(f"  ===== Sampling k-shot train labeled nodes from train set =====")
        train_labeled_indices_cache_path = f"utils/{self.dataset_name}_{self.k_shot}_shot_train_labeled_indices_{self.seed}.json"
        if os.path.exists(train_labeled_indices_cache_path):
            with open(train_labeled_indices_cache_path, "r") as f:
                train_labeled_indices = json.load(f)
            train_labeled_indices = np.array(train_labeled_indices)
            print(f"  Loaded k-shot indices from cache: {train_labeled_indices_cache_path}")
        else:
            train_labeled_indices, _ = sample_k_shot(self.train_data, self.k_shot, self.seed)
            train_labeled_indices = np.array(train_labeled_indices)
            with open(train_labeled_indices_cache_path, "w") as f:
                json.dump(train_labeled_indices.tolist(), f)
            print(f"Saved k-shot indices to cache: {train_labeled_indices_cache_path}")
        self.train_labeled_indices = train_labeled_indices
        print(f"  Selected {len(train_labeled_indices)} train labeled nodes: {train_labeled_indices} ...")

        # 2. Get train_unlabeled_nodes (all train nodes not in train_labeled_indices)
        print(f"  ===== Sampling train unlabeled nodes from train set =====")
        all_train_indices = np.arange(len(self.train_data))
        train_unlabeled_indices = np.setdiff1d(all_train_indices, train_labeled_indices, assume_unique=True)

        # --- Sample train_unlabeled_nodes if required ---
        if self.partial_unlabeled:
            num_to_sample = min(self.num_classes * self.k_shot * self.sample_unlabeled_factor, len(train_unlabeled_indices))
            print(f"  Sampling {num_to_sample} train_unlabeled_nodes (num_classes={self.num_classes}, k={self.k_shot}, factor={self.sample_unlabeled_factor}) from {len(train_unlabeled_indices)} available.")
            if self.pseudo_label:
                print("  Using pseudo-label based sampling (sorted by confidence)...")
                try:
                    with open(self.pseudo_label_cache_path, "r") as f:
                        pseudo_data = json.load(f)
                    # Build a map: index -> (pseudo_label, confidence)
                    pseudo_label_map = {int(item["index"]): (int(item["pseudo_label"]), float(item.get("score", 1.0))) for item in pseudo_data}
                    # Group all train_unlabeled_indices by pseudo label, and sort by confidence
                    pseudo_label_groups = {label: [] for label in range(self.num_classes)}
                    for idx in train_unlabeled_indices:
                        if int(idx) in pseudo_label_map:
                            label, conf = pseudo_label_map[int(idx)]
                            pseudo_label_groups[label].append((idx, conf))
                    # Sort each group by confidence (descending)
                    for label in pseudo_label_groups:
                        pseudo_label_groups[label].sort(key=lambda x: -x[1])
                    # Sample top-N from each group
                    samples_per_class = num_to_sample // self.num_classes
                    sampled_indices = []
                    for label in range(self.num_classes):
                        group = pseudo_label_groups[label]
                        n_samples = min(samples_per_class, len(group))
                        selected = group[:n_samples]
                        for idx, conf in selected:
                            sampled_indices.append(idx)
                    # If not enough, randomly sample from the remaining
                    if len(sampled_indices) < num_to_sample:
                        remaining = num_to_sample - len(sampled_indices)
                        remaining_indices = list(set(train_unlabeled_indices) - set(sampled_indices))
                        if remaining_indices:
                            additional = np.random.choice(remaining_indices, size=min(remaining, len(remaining_indices)), replace=False)
                            sampled_indices.extend(additional)
                    train_unlabeled_indices = np.array(sampled_indices)
                    print(f"  Sampled {len(train_unlabeled_indices)} unlabeled nodes using pseudo labels (sorted by confidence)")
                except Exception as e:
                    print(f"Warning: Error during pseudo-label sampling: {e}. Falling back to random sampling.")
                    train_unlabeled_indices = np.random.choice(train_unlabeled_indices, size=num_to_sample, replace=False)
            else:
                train_unlabeled_indices = np.random.choice(train_unlabeled_indices, size=num_to_sample, replace=False)
        
        self.train_unlabeled_indices = train_unlabeled_indices
        print(f"  Selected {len(train_unlabeled_indices)} train unlabeled nodes: {train_unlabeled_indices[:2*self.k_shot]} ...")

        # 3. Get all test node indices
        print(f"  ===== Sampling test nodes from test set =====")
        test_indices = np.arange(len(self.test_data))
        self.test_indices = test_indices
        print(f"  Selected {len(test_indices)} test nodes: {test_indices[:2*self.k_shot]} ...")

        print(f"  ===== Building nodes =====")
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

        train_labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_unlabeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_labeled_mask[:num_train_labeled] = True
        train_unlabeled_mask[num_train_labeled:num_train_labeled+num_train_unlabeled] = True
        test_mask[num_train_labeled+num_train_unlabeled:] = True

        data.x = x.to(self.device)
        data.y = y.to(self.device)
        data.num_nodes = num_nodes
        data.train_labeled_mask = train_labeled_mask.to(self.device)
        data.train_unlabeled_mask = train_unlabeled_mask.to(self.device)
        data.test_mask = test_mask.to(self.device)
        
        print(f"  Total nodes: {num_nodes}")
        print(f"    - features shape: {data.x.shape}")
        print(f"    - labels shape: {data.y.shape}")
        print(f"    - num_nodes: {data.num_nodes}")
        print(f"    - train_labeled_nodes: {data.train_labeled_mask.sum()}")
        print(f"    - train_unlabeled_nodes: {data.train_unlabeled_mask.sum()}")
        print(f"    - test_nodes: {data.test_mask.sum()}")

        # --- Create Edges ---
        print(f"  ===== Building edges using {self.edge_policy} policy =====")
        embeddings = data.x.cpu().numpy()

        if self.edge_policy == "knn":
            edge_index, edge_attr = self._build_knn_edges(embeddings, self.k_neighbors)
        else: 
            raise ValueError(f"Edge policy '{self.edge_policy}' not supported. Only 'knn' is supported.")

        # symmetrize edge_index and edge_attr (for undirected graphs)
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        data.edge_index = edge_index.to(self.device)
        if edge_attr is not None:
            data.edge_attr = edge_attr.to(self.device)
        else:
            data.edge_attr = None
        data.num_edges = data.edge_index.shape[1]
        print(f"    - Created {data.num_edges} edges")

        self.graph_data = data
        print("Graph construction complete.")

        return data


    def _build_knn_edges(self, embeddings: np.ndarray, k: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build KNN edges.
        Args:
            embeddings: numpy array of shape (num_nodes, embedding_dim)
            k: number of nearest neighbors to consider
        Returns:
            edge_index: torch tensor of shape (2, num_edges)
            edge_attr: torch tensor of shape (num_edges, 1)
        """
        print(f"    Building KNN graph (k={k})...")

        num_nodes = embeddings.shape[0]
        k = min(k, num_nodes - 1) # Adjust k if it's too large
        if k <= 0: 
            return (torch.zeros((2,0), dtype=torch.long), None)
        try:
            distances = pairwise_distances(embeddings, metric="cosine", n_jobs=-1) # Use multiple cores if available
        except Exception as e:
            print(f"      Error calculating pairwise distances: {e}. Using single core.")
            distances = pairwise_distances(embeddings, metric="cosine")

        rows, cols, sim_data = [], [], []
        
        for i in tqdm(range(num_nodes), desc=f"    Finding {k} nearest neighbors", leave=False, ncols=100):
            dist_i = distances[i].copy()
            dist_i[i] = np.inf  # Exclude self
            
            # Find k nearest neighbors
            nearest_indices = np.argpartition(dist_i, k)[:k]
            valid_nearest = nearest_indices[np.isfinite(dist_i[nearest_indices])]
            
            for j in valid_nearest:
                rows.append(i)
                cols.append(j)
                sim = 1.0 - distances[i, j]  # Convert to similarity [0,1]
                sim_data.append(sim)

        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(sim_data, dtype=torch.float).unsqueeze(1)
        
        print(f"    - Created {edge_index.shape[1]} edges.")
        return edge_index, edge_attr

    def analyze_graph(self) -> None:
        """Detailed analysis for graph, similar to build_hetero_graph.py but adapted for homogeneous."""
        
        print("\n" + "=" * 60)
        print("     Graph Analysis (Detailed)")
        print("=" * 60)

        self.graph_metrics = {}  # Reset metrics

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
                "sample_unlabeled": self.partial_unlabeled,
                "unlabeled_sample_factor": self.sample_unlabeled_factor if self.partial_unlabeled else None,
                "num_labeled_original": len(self.train_labeled_indices) if self.train_labeled_indices is not None else 'N/A',
                "num_unlabeled_sampled_original": len(self.train_unlabeled_indices) if self.train_unlabeled_indices is not None else 'N/A',
            }
        }
        print(f"\n--- Basic Info ---")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges: {num_edges}")
        print(f"  Avg Degree: {avg_degree:.2f}")
        print(f"  Train Labeled Nodes: {num_train_labeled}")
        print(f"  Train Unlabeled Nodes: {num_train_unlabeled}")
        print(f"  Test Nodes: {num_test}")
        if self.partial_unlabeled:
            print(f"  Unlabeled Sampling: Enabled (Factor={self.sample_unlabeled_factor})")
        else:
            print(f"  Unlabeled Sampling: Disabled (Used all)")

        # --- Degree Analysis ---
        edge_index_cpu = self.graph_data.edge_index.cpu()
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        degrees.scatter_add_(0, edge_index_cpu[0], torch.ones_like(edge_index_cpu[0]))
        degrees.scatter_add_(0, edge_index_cpu[1], torch.ones_like(edge_index_cpu[1]))
        
        isolated_nodes = int((degrees == 0).sum().item())
        min_degree = int(degrees.min().item()) if num_nodes > 0 else 0
        max_degree = int(degrees.max().item()) if num_nodes > 0 else 0
        mean_degree = float(degrees.float().mean().item()) if num_nodes > 0 else 0.0
        
        self.graph_metrics.update({
            "isolated_nodes": isolated_nodes,
            "min_degree": min_degree,
            "max_degree": max_degree,
            "mean_degree": mean_degree,
        })
        
        print(f"\n--- Degree Analysis ---")
        print(f"  Min Degree: {min_degree}")
        print(f"  Max Degree: {max_degree}")
        print(f"  Mean Degree: {mean_degree:.2f}")
        print(f"  Isolated Nodes: {isolated_nodes} ({isolated_nodes/num_nodes*100:.1f}%)")

        print("=" * 60)
        print("      End of Graph Analysis")
        print("=" * 60 + "\n")

    def save_graph(self) -> Optional[str]:
        """Save the PyG graph and analysis results."""

        # --- Generate graph name ---
        # Add sampling info to filename if sampling was used
        suffix = []
        if self.pseudo_label:
            suffix.append("pseudo")
        if self.partial_unlabeled:
            suffix.append("partial")
            suffix.append(f"sample_unlabeled_factor_{self.sample_unlabeled_factor}")
        sampling_suffix = f"{'_'.join(suffix)}" if suffix else ""

        # Include text embedding type and edge types in name
        graph_name = f"{self.k_shot}_shot_{self.embedding_type}_{self.edge_policy}_{self.k_neighbors}_{sampling_suffix}"
        graph_path = os.path.join(self.output_dir, f"{graph_name}.pt")
        metrics_path = os.path.join(self.output_dir, f"{graph_name}_metrics.json")
        indices_path = os.path.join(self.output_dir, f"{graph_name}_indices.json")
        # --- End filename generation ---

        # Save graph data
        cpu_graph_data = self.graph_data.cpu()
        torch.save(cpu_graph_data, graph_path)

        # Save graph metrics (simplified for homogeneous)
        def default_serializer(obj): # Helper for JSON serialization
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, torch.Tensor): return obj.tolist()
            try: return json.JSONEncoder().encode(obj)
            except TypeError: return str(obj)

        try:
            with open(metrics_path, "w") as f:
                json.dump(self.graph_metrics, f, indent=2, default=default_serializer)
            print(f"  - Graph analysis metrics saved to {metrics_path}")
        except Exception as e:
            print(f"  Error saving metrics JSON: {e}")

        # Save selected indices info
        indices_data = {
            "k_shot": int(self.k_shot),
            "seed": int(self.seed),
            "sample_unlabeled": self.partial_unlabeled,
            "embedding_type": self.embedding_type,
            "edge_policy": self.edge_policy,
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
        print(f"  - Selected indices info saved to {indices_path}")
        print(f"  - Graph saved to {graph_path}")

        return graph_path

    def run_pipeline(self) -> Optional[Data]:
        """Run the complete graph building pipeline."""
        self.load_dataset()
        graph_data = self.build_graph()
        print("  - graph_data.x.shape:", graph_data.x.shape)
        print("  - graph_data.train_labeled_mask.shape:", graph_data.train_labeled_mask.shape)
        print("  - graph_data.train_unlabeled_mask.shape:", graph_data.train_unlabeled_mask.shape)
        print("  - graph_data.test_mask.shape:", graph_data.test_mask.shape)
        self.analyze_graph()
        self.save_graph()
        return graph_data


# --- Argument Parser ---
def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Build a HOMOGENEOUS graph for few-shot fake news detection")

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME, choices=["politifact", "gossipcop"], help=f"HuggingFace Dataset (default: {DEFAULT_DATASET_NAME})")
    parser.add_argument("--k_shot", type=int, default=DEFAULT_K_SHOT, choices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], help=f"Number of labeled samples per class (3-16) (default: {DEFAULT_K_SHOT})")

    # Node Feature Args
    parser.add_argument("--embedding_type", type=str, default=DEFAULT_EMBEDDING_TYPE, choices=["bert", "roberta", "distilbert", "combined"], help=f"Embedding type for nodes (default: {DEFAULT_EMBEDDING_TYPE})")

    # Edge Policy Args
    parser.add_argument("--edge_policy", type=str, default=DEFAULT_EDGE_POLICY, choices=["knn"], help="Edge policy for similarity edges")
    parser.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS, help=f"K for KNN policy (default: {DEFAULT_K_NEIGHBORS})")

    # Sampling Args
    parser.add_argument("--partial_unlabeled", action="store_true", help="Use only a partial subset of unlabeled nodes. Suffix: partial")
    parser.add_argument("--sample_unlabeled_factor", type=int, default=DEFAULT_SAMPLE_UNLABELED_FACTOR, help="Factor M to sample M*2*k unlabeled training nodes (default: 10). Used if --partial_unlabeled.")
    parser.add_argument("--pseudo_label", action="store_true", help="Enable pseudo label factor. Suffix: pseudo")
    parser.add_argument("--pseudo_label_cache_path", type=str, default=None, help="Path to pseudo-label cache (json). Default: utils/pseudo_label_cache_<dataset>.json")
    
    # Output & Settings Args
    parser.add_argument("--output_dir", type=str, default=DEFAULT_GRAPH_DIR, help=f"Directory to save graphs (default: {DEFAULT_GRAPH_DIR})")
    parser.add_argument("--dataset_cache_dir", type=str, default=DEFAULT_DATASET_CACHE_DIR, help=f"Directory to cache datasets (default: {DEFAULT_DATASET_CACHE_DIR})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")

    return parser.parse_args()


# --- Main Execution ---
def main() -> None:
    """Main function to run the homogeneous graph building pipeline."""
    args = parse_arguments()
    set_seed(args.seed)
    
    if args.pseudo_label:
        args.partial_unlabeled = True
        
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print("   Homogeneous Fake News Graph Building Pipeline")
    print("=" * 60)
    print(f"Dataset:          {args.dataset_name}")
    print(f"K-Shot:           {args.k_shot}")
    print(f"Embeddings:       {args.embedding_type}")
    print("-" * 20 + " Edges " + "-" * 20)
    print(f"Policy:           {args.edge_policy}")
    print(f"K neighbors:      {args.k_neighbors}")
    print("-" * 20 + " Node Sampling " + "-" * 20)
    print(f"Sample Unlabeled: {args.partial_unlabeled}")
    if args.partial_unlabeled: 
        print(f"Sample Factor(M): {args.sample_unlabeled_factor} (target 2*k-shot*M nodes)")
        print(f"Pseudo-label Sampling: {args.pseudo_label}")
        if args.pseudo_label:
            print(f"Pseudo-label Cache: {args.pseudo_label_cache_path or f'utils/pseudo_label_cache_{args.dataset_name}.json'}")
    else: 
        print(f"Sample Factor(M): N/A (using all unlabeled train nodes)")
    print("-" * 20 + " Output & Settings " + "-" * 20)
    print(f"Output directory: {args.output_dir}")
    print(f"Seed:             {args.seed}")
    print(f"Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available(): print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")

    # Instantiate and run the builder
    builder = GraphBuilder(
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        embedding_type=args.embedding_type,
        edge_policy=args.edge_policy,
        k_neighbors=args.k_neighbors,
        partial_unlabeled=args.partial_unlabeled,
        sample_unlabeled_factor=args.sample_unlabeled_factor,
        pseudo_label=args.pseudo_label,
        pseudo_label_cache_path=args.pseudo_label_cache_path,
        dataset_cache_dir=args.dataset_cache_dir,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    graph = builder.run_pipeline()

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print(" Homogeneous Graph Building Complete")
    print("=" * 60)
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Features Dim: {graph.num_node_features}")
    print(f"  Train Labeled: {graph.train_labeled_mask.sum().item()}")
    print(f"  Train Unlabeled: {graph.train_unlabeled_mask.sum().item()}")
    print(f"  Test: {graph.test_mask.sum().item()}")
    print("\nNext Steps:")
    print(f"  1. Review the saved graph '.pt' file, metrics '.json' file, and indices '.json' file.")
    print(f"  2. Train a GNN model, e.g.:")
    print(f"     python train_graph.py --graph_path {os.path.join(builder.output_dir, 'scenario', '<graph_file_name>.pt')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()