# build_graph_with_interaction.py (Filename Format Updated)

import os
import gc
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Tuple, Optional, List, Union, Any
from datasets import load_from_disk, DatasetDict, Dataset, Features, Value, Sequence # Import necessary classes
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm
from argparse import ArgumentParser
from matplotlib.patches import Patch
# Make sure this import path is correct in your environment
from utils.sample_k_shot import sample_k_shot
import logging # For potentially suppressing warnings if needed

# Constants
SEED = 42
DEFAULT_K_NEIGHBORS = 5
GRAPH_DIR = "graphs" # <<< CHANGED BACK: Root directory for output graphs
PLOT_DIR = "plots"   # <<< CHANGED BACK: Root directory for plots
DEFAULT_EMBEDDING_TYPE = "roberta_embeddings" # Default BASE embedding field name
DEFAULT_INTEGRATION_STRATEGY = "weighted_average"
DEFAULT_INTEGRATION_ALPHA = 0.7 # Default weight for news embedding in weighted average

def set_seed(seed: int = SEED) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Class Definition (Copy from previous version) ---
class GraphBuilderWithInteraction:
    """
    Builds graph datasets integrating news and user interaction embeddings.
    """

    def __init__(
        self,
        processed_dataset_path: str, # Path to the dataset with precomputed embeddings
        k_shot: int,
        base_embedding_field: str = DEFAULT_EMBEDDING_TYPE,
        integration_strategy: str = DEFAULT_INTEGRATION_STRATEGY,
        integration_alpha: float = DEFAULT_INTEGRATION_ALPHA,
        edge_policy: str = "knn",
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        threshold_factor: float = 1.0,
        alpha_edge: float = 0.5, # Renamed from alpha to avoid conflict
        output_dir: str = GRAPH_DIR, # Uses the updated constant
        plot: bool = False,
        seed: int = SEED,
        device: str = None,
    ):
        """Initialize the GraphBuilder with configuration parameters."""
        self.processed_dataset_path = processed_dataset_path
        self.dataset_name = os.path.basename(processed_dataset_path).split('_')[0]
        print(f"Inferred dataset name: {self.dataset_name}")

        self.k_shot = k_shot
        self.base_embedding_field = base_embedding_field
        self.integration_strategy = integration_strategy
        self.integration_alpha = integration_alpha
        self.edge_policy = edge_policy
        self.k_neighbors = k_neighbors
        self.threshold_factor = threshold_factor
        self.alpha_edge = alpha_edge # Edge threshold alpha
        self.plot = plot
        self.seed = seed

        # Use the dataset name within the main output/plot directory
        self.output_dir = os.path.join(output_dir, self.dataset_name)
        self.plot_dir = os.path.join(PLOT_DIR, self.dataset_name) # Use updated PLOT_DIR

        os.makedirs(self.output_dir, exist_ok=True)
        if self.plot:
            os.makedirs(self.plot_dir, exist_ok=True)

        self.dataset = None
        self.graph_data = None
        self.graph_metrics = {}
        self.selected_indices = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Integration Strategy: {self.integration_strategy}")
        if self.integration_strategy == 'weighted_average':
            print(f"Integration Alpha (weight for base embedding): {self.integration_alpha}")

    def load_processed_dataset(self) -> None:
        """Load pre-processed dataset from disk."""
        print(f"Loading processed dataset from '{self.processed_dataset_path}'...")
        try:
            dataset = load_from_disk(self.processed_dataset_path)
        except Exception as e:
            print(f"Error loading dataset from disk: {e}")
            print(f"Please ensure '{self.processed_dataset_path}' exists.")
            raise
        if "train" not in dataset or "test" not in dataset:
             raise ValueError("Loaded dataset must contain 'train' and 'test' splits.")
        required_cols = [self.base_embedding_field, 'interaction_gemini_embeddings', 'interaction_tones', 'label']
        for split in dataset:
            for col in required_cols:
                if col not in dataset[split].column_names:
                    raise ValueError(f"Required column '{col}' not found in '{split}' split.")
        self.dataset = dataset
        self.train_size = len(dataset["train"])
        self.test_size = len(dataset["test"])
        unique_labels = set(dataset["train"]["label"])
        if self.k_shot == 0:
            self.labeled_size = self.train_size
            print("k_shot=0 implies using all training data for labeling (full-shot).")
        elif self.k_shot > 0:
             self.labeled_size = self.k_shot * len(unique_labels)
             print(f"With {len(unique_labels)} classes and k_shot={self.k_shot}, labeled_size={self.labeled_size}")
        else:
             raise ValueError("k_shot must be >= 0")
        self._show_dataset_stats()

    def _show_dataset_stats(self) -> None:
        """Display statistics about the loaded dataset."""
        train_data = self.dataset["train"]
        test_data = self.dataset["test"]
        train_labels = train_data["label"]
        test_labels = test_data["label"]
        train_label_counts = {label: train_labels.count(label) for label in set(train_labels)}
        test_label_counts = {label: test_labels.count(label) for label in set(test_labels)}
        print("\nLoaded Dataset Statistics:")
        print(f"  Train set: {len(train_data)} samples")
        for label, count in train_label_counts.items(): print(f"    - Class {label}: {count} samples ({count/len(train_data)*100:.1f}%)")
        print(f"  Test set: {len(test_data)} samples")
        for label, count in test_label_counts.items(): print(f"    - Class {label}: {count} samples ({count/len(test_data)*100:.1f}%)")
        if self.k_shot > 0: print(f"  Few-shot labeled set: {self.k_shot} samples per class ({self.labeled_size} total)")
        print("")

    def _integrate_embeddings(self, base_emb_list: List[float], interaction_embs_list: List[List[float]]) -> np.ndarray:
        """Integrates base and interaction embeddings based on the chosen strategy."""
        if not isinstance(base_emb_list, list) or not base_emb_list:
             print("Warning: Received invalid base_emb_list. Returning zero vector.")
             # Determine expected dimension (e.g., 768) or get from a valid example if possible
             # For now, hardcoding, but ideally get dynamically.
             return np.zeros(768, dtype=np.float32)
        base_emb = np.array(base_emb_list, dtype=np.float32)
        expected_dim = base_emb.shape[0]

        valid_interaction_embs = [np.array(emb, dtype=np.float32) for emb in interaction_embs_list if isinstance(emb, list) and len(emb) == expected_dim]

        if not valid_interaction_embs:
            if self.integration_strategy == "interaction_only":
                 # If only using interactions and none are valid, return zeros
                 print("Warning: integration_strategy is 'interaction_only' but no valid interaction embeddings found. Returning zero vector.")
                 return np.zeros_like(base_emb)
            else:
                 # Otherwise (weighted, simple_average, base_only), use base embedding
                 return base_emb

        avg_interaction_emb = np.mean(np.array(valid_interaction_embs), axis=0)

        if self.integration_strategy == "weighted_average":
            final_emb = (self.integration_alpha * base_emb + (1.0 - self.integration_alpha) * avg_interaction_emb)
        elif self.integration_strategy == "simple_average":
            final_emb = 0.5 * base_emb + 0.5 * avg_interaction_emb
        elif self.integration_strategy == "base_only":
             final_emb = base_emb
        elif self.integration_strategy == "interaction_only":
             final_emb = avg_interaction_emb
        else:
            raise ValueError(f"Unknown integration strategy: {self.integration_strategy}")
        return final_emb.astype(np.float32)

    def build_empty_graph(self) -> Data:
        """Build graph nodes using integrated embeddings."""
        print(f"Building graph nodes using integrated embeddings ({self.integration_strategy})...")
        train_data = self.dataset["train"]
        test_data = self.dataset["test"]
        num_train = len(train_data)
        num_test = len(test_data)
        num_nodes = num_train + num_test
        all_features = []
        all_labels = []

        print("Processing train data for node features...")
        for i in tqdm(range(num_train)):
            base_emb = train_data[i][self.base_embedding_field]
            interaction_embs = train_data[i]['interaction_gemini_embeddings']
            final_emb = self._integrate_embeddings(base_emb, interaction_embs)
            all_features.append(final_emb)
            all_labels.append(train_data[i]['label'])

        print("Processing test data for node features...")
        for i in tqdm(range(num_test)):
            base_emb = test_data[i][self.base_embedding_field]
            interaction_embs = test_data[i]['interaction_gemini_embeddings']
            final_emb = self._integrate_embeddings(base_emb, interaction_embs)
            all_features.append(final_emb)
            all_labels.append(test_data[i]['label'])

        # Add check for consistent embedding dimension before creating tensor
        if not all_features:
             raise ValueError("No features were generated. Check data processing.")
        feature_dim = len(all_features[0])
        if not all(len(f) == feature_dim for f in all_features):
             raise ValueError("Inconsistent feature dimensions detected before tensor creation.")

        x = torch.tensor(np.array(all_features), dtype=torch.float)
        y = torch.tensor(np.array(all_labels), dtype=torch.long)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:num_train] = True
        test_mask[num_train:] = True

        if self.k_shot > 0:
             print(f"Using consistent k_shot sampling with k={self.k_shot}, seed={self.seed}")
             indices, _ = sample_k_shot(train_data, self.k_shot, self.seed)
             self.selected_indices = indices
             labeled_mask_indices = np.zeros(num_nodes, dtype=bool)
             if indices: # Check if indices is not empty
                 labeled_mask_indices[np.array(indices)] = True # Ensure indices is usable as numpy index
             labeled_mask = torch.tensor(labeled_mask_indices, dtype=torch.bool)
        elif self.k_shot == 0:
            print("Full-shot scenario: All training nodes are considered labeled.")
            labeled_mask = train_mask.clone()
            self.selected_indices = list(range(num_train))

        edge_index = torch.zeros((2, 0), dtype=torch.long)
        graph_data = Data(x=x, y=y, train_mask=train_mask, test_mask=test_mask, labeled_mask=labeled_mask, edge_index=edge_index, num_nodes=num_nodes, num_features=x.shape[1])
        self.graph_data = graph_data
        print(f"Graph nodes built with {num_nodes} nodes and {x.shape[1]} features.")
        if x.shape[1] != 768: print(f"WARNING: Final node feature dimension is {x.shape[1]}, expected 768.")
        print(f"Labeled nodes: {labeled_mask.sum().item()}")
        print("")
        return graph_data

    # --- Edge Building Methods (Copied from previous version) ---
    def _build_knn_edges(self, embeddings: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using k-nearest neighbors approach."""
        print(f"Building KNN graph with k={k}...")
        distances = pairwise_distances(embeddings, metric='cosine')
        rows, cols, data = [], [], []
        for i in tqdm(range(len(embeddings)), desc="Finding neighbors"):
            dist_i = distances[i].copy()
            dist_i[i] = float('inf')
            indices = np.argpartition(dist_i, k)[:k]
            for j in indices:
                rows.append(i)
                cols.append(j)
                data.append(1 - distances[i, j])
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        return edge_index, edge_attr

    def _build_mutual_knn_edges(self, embeddings: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using mutual k-nearest neighbors approach."""
        print(f"Building mutual KNN graph with k={k}...")
        distances = pairwise_distances(embeddings, metric='cosine')
        rows, cols, data = [], [], []
        all_neighbors = {} # Store neighbors for efficient lookup
        for i in tqdm(range(len(embeddings)), desc="Finding neighbors"):
            dist_i = distances[i].copy()
            dist_i[i] = float('inf')
            indices_i = np.argpartition(dist_i, k)[:k]
            all_neighbors[i] = set(indices_i)

        for i in tqdm(range(len(embeddings)), desc="Checking mutuality"):
            for j in all_neighbors[i]:
                if i in all_neighbors.get(j, set()): # Check if i is in j's neighbors
                    rows.append(i)
                    cols.append(j)
                    data.append(1 - distances[i, j])

        if not rows:
            print("Warning: No mutual KNN edges found.")
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float)
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        return edge_index, edge_attr

    def _build_threshold_edges(self, embeddings: np.ndarray, threshold_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using a threshold-based approach."""
        print(f"Building threshold graph with threshold_factor={threshold_factor}...")
        embeddings = np.nan_to_num(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.maximum(norms, 1e-10)
        rows, cols, data = [], [], []
        num_nodes = len(embeddings)
        batch_size = min(500, num_nodes)
        for i in tqdm(range(0, num_nodes, batch_size), desc="Computing threshold edges"):
            batch_end = min(i + batch_size, num_nodes)
            batch_normalized = normalized_embeddings[i:batch_end]
            similarities_batch = np.dot(batch_normalized, normalized_embeddings.T)
            for j in range(batch_end - i):
                node_idx = i + j
                node_similarities = similarities_batch[j].copy()
                node_similarities[node_idx] = -1.0
                positive_similarities = node_similarities[node_similarities > 0]
                if len(positive_similarities) > 0:
                    mean_similarity = np.mean(positive_similarities)
                    node_threshold = mean_similarity * threshold_factor
                    above_threshold = np.where(node_similarities > node_threshold)[0]
                    for target in above_threshold:
                        rows.append(node_idx)
                        cols.append(target)
                        data.append(node_similarities[target])
        if not rows: print("Warning: No threshold edges created.")
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long) if rows else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1) if data else torch.zeros((0, 1), dtype=torch.float)
        print(f"Created {edge_index.shape[1]} threshold edges.")
        return edge_index, edge_attr

    def _build_dynamic_threshold_edges(self, embeddings: np.ndarray, alpha: float = 0.5, max_degree: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
         """Build edges using a dynamic threshold approach."""
         print(f"Building dynamic threshold graph with alpha={alpha}" + (f", max_degree={max_degree}" if max_degree else "") + "...")
         embeddings = np.nan_to_num(embeddings)
         norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
         normalized_embeddings = embeddings / np.maximum(norms, 1e-10)
         num_nodes = len(embeddings)
         print("Computing global similarity statistics (sampling)...")
         sample_size = min(1000, num_nodes * 10)
         idx1 = np.random.choice(num_nodes, sample_size)
         idx2 = np.random.choice(num_nodes, sample_size)
         valid_pairs = idx1 != idx2
         idx1, idx2 = idx1[valid_pairs], idx2[valid_pairs]
         if len(idx1) == 0: print("Warning: No valid pairs for dynamic threshold sampling."); return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,1), dtype=torch.float)
         sampled_sims = np.sum(normalized_embeddings[idx1] * normalized_embeddings[idx2], axis=1)
         sim_mean = np.mean(sampled_sims); sim_std = np.std(sampled_sims)
         dynamic_threshold = sim_mean + alpha * sim_std
         print(f"Similarity stats: mean={sim_mean:.4f}, std={sim_std:.4f}. Dynamic threshold: {dynamic_threshold:.4f}")
         rows, cols, data = [], [], []
         for i in tqdm(range(num_nodes), desc="Building dynamic threshold edges"):
            similarities = np.dot(normalized_embeddings[i], normalized_embeddings.T)
            similarities[i] = -np.inf # More robust exclusion of self
            above_threshold = np.where(similarities > dynamic_threshold)[0]
            # Optional degree limit
            if max_degree and len(above_threshold) > max_degree:
                top_indices = above_threshold[np.argsort(similarities[above_threshold])[-max_degree:]]
            else:
                top_indices = above_threshold
            for target in top_indices:
                 rows.append(i); cols.append(target); data.append(similarities[target])
         if not rows: print("Warning: No dynamic threshold edges created.")
         edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long) if rows else torch.zeros((2, 0), dtype=torch.long)
         edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1) if data else torch.zeros((0, 1), dtype=torch.float)
         print(f"Created {edge_index.shape[1]} dynamic threshold edges.")
         return edge_index, edge_attr
    # --------------------------------------------------------------

    def build_graph(self) -> Data:
        """Build a graph including both nodes and edges."""
        self.build_empty_graph()
        print(f"Building graph edges using {self.edge_policy} policy...")
        if self.graph_data.num_nodes == 0:
            print("No nodes in graph, skipping edge building.")
            return self.graph_data
        embeddings = self.graph_data.x.numpy()
        if self.edge_policy == "knn": edges, edge_attr = self._build_knn_edges(embeddings, self.k_neighbors)
        elif self.edge_policy == "mutual_knn": edges, edge_attr = self._build_mutual_knn_edges(embeddings, self.k_neighbors)
        elif self.edge_policy == "threshold": edges, edge_attr = self._build_threshold_edges(embeddings, self.threshold_factor)
        elif self.edge_policy == "dynamic_threshold": edges, edge_attr = self._build_dynamic_threshold_edges(embeddings, self.alpha_edge)
        else: raise ValueError(f"Unknown edge policy: {self.edge_policy}")
        self.graph_data.edge_index = edges
        self.graph_data.edge_attr = edge_attr
        self.graph_data.num_edges = edges.shape[1]
        print(f"Graph edges built: {edges.shape[1]} edges created")
        self._analyze_graph()
        return self.graph_data

    def _analyze_graph(self) -> None:
        """Analyze the graph and compute comprehensive metrics."""
        if self.graph_data is None: raise ValueError("Graph must be built before analysis")
        self.graph_metrics = { "num_nodes": self.graph_data.num_nodes, "num_edges": self.graph_data.edge_index.shape[1], "avg_degree": self.graph_data.edge_index.shape[1] / self.graph_data.num_nodes if self.graph_data.num_nodes > 0 else 0 }
        print("\nGraph Analysis Summary:")
        print(f"  Nodes: {self.graph_metrics['num_nodes']}")
        print(f"  Edges: {self.graph_metrics['num_edges']}")
        print(f"  Average degree: {self.graph_metrics['avg_degree']:.2f}")
        try:
             if self.graph_data.num_nodes > 0 and self.graph_data.num_edges > 0:
                 G = to_networkx(self.graph_data, to_undirected=True)
                 self.graph_metrics["is_connected"] = nx.is_connected(G)
                 if not self.graph_metrics["is_connected"]: self.graph_metrics["num_components"] = nx.number_connected_components(G)
                 self.graph_metrics["density"] = nx.density(G)
                 print(f"  Density: {self.graph_metrics['density']:.4f}")
                 print(f"  Connected: {self.graph_metrics['is_connected']}")
                 if not self.graph_metrics['is_connected']: print(f"  Components: {self.graph_metrics['num_components']}")
             else:
                 print("  Graph is empty or has no edges, skipping NetworkX analysis.")
        except Exception as e: print(f"  NetworkX analysis failed: {e}")
        print("")


    def save_graph(self) -> str:
        """Save the graph and analysis results with the desired filename format."""
        if self.graph_data is None: raise ValueError("Graph must be built before saving")

        # --- Updated Filename Generation Logic ---
        k_shot_tag = f"{self.k_shot}shot"

        # Base embedding tag (e.g., "roberta")
        base_emb_tag = self.base_embedding_field.split('_')[0]

        # Interaction/Integration tag part
        interaction_part = ""
        if self.integration_strategy == "simple_average":
            interaction_part = "_interaction_average" # Desired: interaction_average
        elif self.integration_strategy == "weighted_average":
            # Desired: interaction_weighted0.7 (example)
            interaction_part = f"_interaction_weighted{self.integration_alpha:.1f}"
        elif self.integration_strategy == "interaction_only":
             interaction_part = "_interactiononly" # Only interaction embeddings used
        # If integration_strategy == "base_only", interaction_part remains empty ""

        # Edge policy tag part
        edge_part = ""
        # Format edge parameters to one decimal place if they are floats
        if self.edge_policy in ['knn', 'mutual_knn']:
            edge_part = f"_{self.edge_policy}{self.k_neighbors}"
        elif self.edge_policy == 'threshold':
            edge_part = f"_{self.edge_policy}{self.threshold_factor:.1f}" # Use .1f
        elif self.edge_policy == 'dynamic_threshold':
            edge_part = f"_{self.edge_policy}{self.alpha_edge:.1f}" # Use .1f

        # Combine all parts
        graph_name = f"{k_shot_tag}_{base_emb_tag}{interaction_part}{edge_part}"
        # Example: 3shot_roberta_interaction_average_dynamic_threshold0.5
        # Example: 8shot_bert_interaction_weighted0.7_knn5
        # Example: 3shot_roberta_dynamic_threshold0.5 (if base_only)
        # -----------------------------------------

        graph_path = os.path.join(self.output_dir, f"{graph_name}.pt")
        torch.save(self.graph_data, graph_path)

        metrics_path = os.path.join(self.output_dir, f"{graph_name}_metrics.json")
        serializable_metrics = json.loads(json.dumps(self.graph_metrics, default=lambda x: int(x) if isinstance(x, (np.integer, np.bool_)) else float(x) if isinstance(x, np.floating) else str(x)))
        with open(metrics_path, "w") as f: json.dump(serializable_metrics, f, indent=2)
        print(f"Graph metrics saved to {metrics_path}")

        if self.selected_indices is not None:
            indices_path = os.path.join(self.output_dir, f"{graph_name}_indices.json")
            indices_data = {"indices": [int(i) for i in self.selected_indices], "k_shot": int(self.k_shot), "seed": int(self.seed)}
            # Add label distribution calculation here if needed from original script
            with open(indices_path, "w") as f: json.dump(indices_data, f, indent=2)
            print(f"Selected indices saved to {indices_path}")

        print(f"Graph saved to {graph_path}")
        if self.plot:
            try: self.visualize_graph(graph_name)
            except Exception as e: print(f"Warning: Error visualizing graph: {e}")
        return graph_path

    def visualize_graph(self, graph_name: str, max_nodes: int = 1000) -> None:
        """Visualize the graph using NetworkX."""
        # (Code copied from previous version - no changes needed for filename logic)
        if self.graph_data is None: raise ValueError("Graph must be built before visualization")
        if self.graph_data.num_nodes == 0: print("Skipping visualization for empty graph."); return
        num_nodes_to_plot = min(self.graph_data.num_nodes, max_nodes)
        print(f"Visualizing graph ({num_nodes_to_plot}/{self.graph_data.num_nodes} nodes)...")
        sub_data = self.graph_data.subgraph(torch.arange(num_nodes_to_plot))
        try:
            G = to_networkx(sub_data, to_undirected=True)
        except Exception as e:
            print(f"Error converting to NetworkX: {e}. Skipping visualization.")
            return

        train_mask = sub_data.train_mask.numpy()
        test_mask = sub_data.test_mask.numpy()
        labeled_mask = sub_data.labeled_mask.numpy()
        node_colors = []
        for i in range(num_nodes_to_plot):
            if labeled_mask[i]: node_colors.append("green")
            elif train_mask[i]: node_colors.append("blue")
            elif test_mask[i]: node_colors.append("red")
            else: node_colors.append("gray")
        plt.figure(figsize=(12, 10))
        if G.number_of_nodes() > 500: pos = nx.random_layout(G, seed=self.seed)
        else: pos = nx.spring_layout(G, k=0.15, seed=self.seed)
        node_size = 25 if num_nodes_to_plot <= 100 else 5
        nx.draw_networkx(G, pos=pos, with_labels=False, node_color=node_colors, edge_color='gray', node_size=node_size, width=0.5, alpha=0.7)
        legend_elements = [ Patch(facecolor='green', label='Few-shot Labeled'), Patch(facecolor='blue', label='Train (Unlabeled)'), Patch(facecolor='red', label='Test')]
        plt.legend(handles=legend_elements, loc='upper right')
        # Update title slightly
        base_emb_tag = self.base_embedding_field.split('_')[0]
        title = f"{self.dataset_name.capitalize()} ({base_emb_tag}) - Int: {self.integration_strategy} - Edge: {self.edge_policy}"
        if self.graph_data.num_nodes > max_nodes: title += f"\n(showing {max_nodes} of {self.graph_data.num_nodes} nodes)"
        plt.title(title)
        plot_path = os.path.join(self.plot_dir, f"{graph_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph visualization saved to {plot_path}")

    def run_pipeline(self) -> Data:
        """Run the complete graph building pipeline."""
        self.load_processed_dataset()
        self.build_graph()
        self.save_graph()
        return self.graph_data

# --- Argument Parsing (Copied from previous version) ---
def parse_arguments():
    parser = ArgumentParser(description="Build graph with integrated news and interaction embeddings.")
    parser.add_argument("--processed_dataset_path", type=str, required=True, help="Path to the processed dataset directory.")
    parser.add_argument("--k_shot", type=int, default=8, help="Number of samples per class (0 means full-shot).")
    parser.add_argument("--base_embedding_field", type=str, default=DEFAULT_EMBEDDING_TYPE, help=f"Base news embedding field (default: {DEFAULT_EMBEDDING_TYPE}).")
    parser.add_argument("--integration_strategy", type=str, default=DEFAULT_INTEGRATION_STRATEGY, choices=["weighted_average", "simple_average", "base_only", "interaction_only"], help="How to combine embeddings.")
    parser.add_argument("--integration_alpha", type=float, default=DEFAULT_INTEGRATION_ALPHA, help="Weight for base embedding in 'weighted_average'.")
    parser.add_argument("--edge_policy", type=str, default="knn", choices=["knn", "mutual_knn", "threshold", "dynamic_threshold"], help="Edge construction policy.")
    parser.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS, help="k for KNN policies.")
    parser.add_argument("--threshold_factor", type=float, default=1.0, help="Factor for 'threshold' policy.")
    parser.add_argument("--alpha_edge", type=float, default=0.5, help="Alpha for 'dynamic_threshold' policy.")
    parser.add_argument("--output_dir", type=str, default=GRAPH_DIR, help=f"Base directory to save graphs (default: {GRAPH_DIR}).")
    parser.add_argument("--plot", action="store_true", help="Enable graph visualization.")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED}).")
    return parser.parse_args()

# --- Main Function (Copied from previous version) ---
def main() -> None:
    """Main function to run the graph building pipeline."""
    args = parse_arguments()
    set_seed(args.seed)
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "="*60)
    print("Fake News Detection - Graph Building with Interactions")
    print("="*60)
    # (Print arguments - code omitted for brevity, same as before)
    print(f"Processed Dataset: {args.processed_dataset_path}")
    print(f"k-shot:           {args.k_shot}")
    print(f"Base Embedding:   {args.base_embedding_field}")
    print(f"Integration:      {args.integration_strategy}")
    if args.integration_strategy == 'weighted_average':
        print(f"Integration Alpha: {args.integration_alpha}")
    print(f"Edge Policy:      {args.edge_policy}")
    if args.edge_policy in ['knn', 'mutual_knn']: print(f"K Neighbors:      {args.k_neighbors}")
    elif args.edge_policy == 'threshold': print(f"Threshold Factor: {args.threshold_factor}")
    elif args.edge_policy == 'dynamic_threshold': print(f"Alpha (Edge):     {args.alpha_edge}")
    print(f"Output Directory: {args.output_dir}") # Will show the base dir, e.g., "graphs"
    print(f"Plot:             {args.plot}")
    print(f"Seed:             {args.seed}")
    print(f"Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available(): print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")

    builder = GraphBuilderWithInteraction(
        processed_dataset_path=args.processed_dataset_path,
        k_shot=args.k_shot,
        base_embedding_field=args.base_embedding_field,
        integration_strategy=args.integration_strategy,
        integration_alpha=args.integration_alpha,
        edge_policy=args.edge_policy,
        k_neighbors=args.k_neighbors,
        threshold_factor=args.threshold_factor,
        alpha_edge=args.alpha_edge, # Pass edge alpha
        output_dir=args.output_dir, # Pass base output dir
        plot=args.plot,
        seed=args.seed
    )

    graph_data = builder.run_pipeline()

    print("\n" + "="*60)
    print("Graph Building Complete")
    print("="*60)
    # (Print final results - code omitted for brevity, same as before)
    print(f"Final Nodes:      {graph_data.num_nodes}")
    print(f"Final Features:   {graph_data.num_features}")
    print(f"Final Edges:      {graph_data.edge_index.shape[1]}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()