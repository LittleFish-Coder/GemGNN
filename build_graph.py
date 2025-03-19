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

# Import the same sampling function used in finetune_lm.py
# This ensures we select the exact same examples with the same seed
from utils.sample_k_shot import sample_k_shot

# Constants
SEED = 42  # Use the same SEED as finetune_lm.py
DEFAULT_K_NEIGHBORS = 5 
GRAPH_DIR = "graphs"
PLOT_DIR = "plots"
DEFAULT_EMBEDDING_TYPE = "roberta"  # Default to RoBERTa embeddings

def set_seed(seed: int = SEED) -> None:
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
    supporting both transductive (test nodes present during training) and few-shot 
    learning scenarios where only a small subset of nodes are labeled.
    """
    
    def __init__(
        self,
        dataset_name: str,
        k_shot: int,
        edge_policy: str = "knn",
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        threshold_factor: float = 1.0,
        output_dir: str = GRAPH_DIR,
        plot: bool = False,
        seed: int = SEED,
        embedding_type: str = DEFAULT_EMBEDDING_TYPE,
        device: str = None,
    ):
        """Initialize the GraphBuilder with configuration parameters."""
        self.dataset_name = dataset_name.lower()
        self.k_shot = k_shot
        self.edge_policy = edge_policy
        self.k_neighbors = k_neighbors
        self.threshold_factor = threshold_factor
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
        print(f"Loading dataset '{self.dataset_name}' with {self.embedding_type} embeddings...")
        
        # Format dataset name for HuggingFace
        hf_dataset_name = f"LittleFish-Coder/Fake_News_{self.dataset_name.capitalize()}"
        
        # Load dataset
        try:
            dataset = load_dataset(
                hf_dataset_name,
                download_mode="reuse_cache_if_exists",
                cache_dir="dataset"
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print(f"Please check if '{hf_dataset_name}' exists on HuggingFace")
            raise
        
        # Get train and test datasets (use full datasets)
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        
        # Store dataset
        self.dataset = {
            "train": train_dataset,
            "test": test_dataset
        }
        
        # Store sizes
        self.train_size = len(train_dataset)
        self.test_size = len(test_dataset)
        
        # Calculate labeled size from k_shot
        unique_labels = set(train_dataset["label"])
        self.labeled_size = self.k_shot * len(unique_labels)
        print(f"With {len(unique_labels)} classes and k_shot={self.k_shot}, labeled_size={self.labeled_size}")
        
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
            label: train_labels.count(label) 
            for label in set(train_labels)
        }
        
        test_label_counts = {
            label: test_labels.count(label) 
            for label in set(test_labels)
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
        elif self.edge_policy == "thresholdnn":
            edges, edge_attr = self._build_threshold_edges(embeddings, self.threshold_factor)
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
        
        # Check if the embedding field exists
        if embedding_field not in train_data.features and embedding_field not in test_data.features:
            available_fields = [f for f in train_data.features if "_embeddings" in f]
            if not available_fields:
                raise ValueError(f"No embedding fields found in the dataset. Available fields: {train_data.features}")
            
            # Fall back to "embeddings" if it exists and no specific embedding field is found
            if "embeddings" in train_data.features:
                embedding_field = "embeddings"
                print(f"Warning: {self.embedding_type}_embeddings not found. Using 'embeddings' field instead.")
            else:
                # Use the first available embedding field
                embedding_field = available_fields[0]
                print(f"Warning: {self.embedding_type}_embeddings not found. Using '{embedding_field}' field instead.")
        
        print(f"Using embeddings from field: '{embedding_field}'")
        
        # Get embeddings and labels
        train_embeddings = np.array(train_data[embedding_field])
        train_labels = np.array(train_data["label"])
        test_embeddings = np.array(test_data[embedding_field])
        test_labels = np.array(test_data["label"])
        
        # Calculate number of nodes
        num_nodes = len(train_embeddings) + len(test_embeddings)
        
        # Merge embeddings and labels
        x = torch.tensor(np.concatenate([train_embeddings, test_embeddings]), dtype=torch.float)
        y = torch.tensor(np.concatenate([train_labels, test_labels]), dtype=torch.long)
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Set masks
        train_mask[:len(train_embeddings)] = True
        test_mask[len(train_embeddings):] = True
        
        # Use the EXACT SAME sampling logic as finetune_lm.py
        print(f"Using consistent k_shot sampling with k={self.k_shot}, seed={self.seed}")
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
            num_features=x.shape[1]
        )
        
        # Store graph data
        self.graph_data = graph_data
        
        # Display graph structure
        print(f"Graph nodes built with {num_nodes} nodes and {x.shape[1]} features")
        print(f"Labeled nodes: {labeled_mask.sum().item()} (for few-shot learning)")
        
        # Print some sample indices to verify
        if len(self.selected_indices) > 0:
            print(f"Sample of selected indices (first 5): {sorted(self.selected_indices)[:5]}")
            # Count distribution of labels in selected indices
            label_counts = {}
            for idx in self.selected_indices:
                label = train_labels[idx]
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            print(f"Label distribution in selected indices: {label_counts}")
        
        return graph_data
    
    def _build_knn_edges(self, embeddings: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using k-nearest neighbors approach."""
        print(f"Building KNN graph with k={k}...")
        
        # Compute cosine distances
        distances = pairwise_distances(embeddings, metric='cosine')
        
        # For each node, find k nearest neighbors
        rows, cols, data = [], [], []
        for i in tqdm(range(len(embeddings)), desc="Finding neighbors"):
            # Skip self (distance=0)
            dist_i = distances[i].copy()
            dist_i[i] = float('inf')
            
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
    
    def _build_threshold_edges(self, embeddings: np.ndarray, threshold_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges using threshold-based approach."""
        print(f"Building threshold graph with factor={threshold_factor}...")
        
        # Compute pairwise cosine similarities
        # Normalize embeddings for faster computation
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.maximum(norms, 1e-10)  # Avoid division by zero
        
        rows, cols, data = [], [], []
        batch_size = min(1000, len(embeddings))  # Adjust batch size based on memory
        
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Computing similarities"):
            batch_end = min(i + batch_size, len(embeddings))
            batch = normalized_embeddings[i:batch_end]
            
            # Compute similarities with all nodes
            batch_similarities = np.dot(batch, normalized_embeddings.T)
            
            # Get mean similarity for each node in batch
            batch_mean_similarities = np.mean(batch_similarities, axis=1)
            
            # Apply threshold factor
            for j in range(batch_end - i):
                node_idx = i + j
                node_similarities = batch_similarities[j]
                node_threshold = batch_mean_similarities[j] * threshold_factor
                
                # Find nodes above threshold (excluding self)
                above_threshold = np.where(node_similarities > node_threshold)[0]
                valid_nodes = above_threshold[above_threshold != node_idx]
                
                # Add edges
                for target in valid_nodes:
                    rows.append(node_idx)
                    cols.append(target)
                    data.append(node_similarities[target])
        
        # Create PyTorch tensors
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor(data, dtype=torch.float).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def _analyze_graph(self) -> None:
        """Analyze the graph and compute metrics."""
        if self.graph_data is None:
            raise ValueError("Graph must be built before analysis")
        
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
        
        for edge in tqdm(self.graph_data.edge_index.t(), desc="Analyzing edge types"):
            source, target = edge
            if self.graph_data.train_mask[source] and self.graph_data.train_mask[target]:
                edge_types["train-train"] += 1
            elif (self.graph_data.train_mask[source] and self.graph_data.test_mask[target]) or \
                 (self.graph_data.test_mask[source] and self.graph_data.train_mask[target]):
                edge_types["train-test"] += 1
            elif self.graph_data.test_mask[source] and self.graph_data.test_mask[target]:
                edge_types["test-test"] += 1
        
        self.graph_metrics["edge_types"] = edge_types
        
        # Calculate connectivity patterns between fake/real news
        fake_to_fake = 0
        real_to_real = 0
        fake_to_real = 0
        
        for edge in tqdm(self.graph_data.edge_index.t(), desc="Analyzing class connectivity"):
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
        
        print("\nGraph Analysis Summary:")
        print(f"  Nodes: {self.graph_metrics['num_nodes']}")
        print(f"  Edges: {self.graph_metrics['num_edges']}")
        print(f"  Average degree: {self.graph_metrics['avg_degree']:.2f}")
        print(f"  Homophily ratio: {self.graph_metrics['homophily_ratio']:.2f}")
        print("  Edge types:")
        for edge_type, count in edge_types.items():
            print(f"    - {edge_type}: {count} ({count/total_edges*100:.1f})")
        print("")
    
    def save_graph(self) -> str:
        """Save the graph and analysis results."""
        if self.graph_data is None:
            raise ValueError("Graph must be built before saving")
        
        # Generate graph name - include embedding type in the filename only
        graph_name = f"{self.k_shot}shot_{self.embedding_type}_{self.edge_policy}{self.k_neighbors if self.edge_policy == 'knn' else self.threshold_factor}"
        
        # Save graph data
        graph_path = os.path.join(self.output_dir, f"{graph_name}.pt")
        torch.save(self.graph_data, graph_path)
        
        # Save graph metrics if available
        if self.graph_metrics:
            # Convert any numpy numbers to Python native types
            python_metrics = json.loads(json.dumps(self.graph_metrics, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)))
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
            python_label_distribution = {int(k): int(v) for k, v in label_distribution.items()}
            
            with open(indices_path, "w") as f:
                json.dump({
                    "indices": [int(i) for i in self.selected_indices] if isinstance(self.selected_indices, (np.ndarray, list)) else self.selected_indices,
                    "k_shot": int(self.k_shot),
                    "seed": int(self.seed),
                    "label_distribution": python_label_distribution
                }, f, indent=2)
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
            print(f"Graph is large ({self.graph_data.num_nodes} nodes). Visualizing only {max_nodes} nodes.")
        
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
            pos = nx.spring_layout(G, k=0.15, seed=self.seed)  # For better visualization
            
        nx.draw_networkx(
            G, 
            pos=pos,
            with_labels=False, 
            node_color=node_colors, 
            edge_color='gray', 
            node_size=25 if G.number_of_nodes() <= 100 else 5, 
            width=0.5,
            alpha=0.7
        )
        
        # Add legend
        legend_elements = [
            Patch(facecolor='green', label='Few-shot Labeled'),
            Patch(facecolor='blue', label='Train (Unlabeled)'),
            Patch(facecolor='red', label='Test')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add title
        title = f"{self.dataset_name.capitalize()} - {self.edge_policy.upper()} ({self.embedding_type.upper()})"
        if self.graph_data.num_nodes > max_nodes:
            title += f"\n(showing {max_nodes} of {self.graph_data.num_nodes} nodes)"
        
        subtitle = f"{self.k_shot}-shot per class, Total nodes: {self.graph_data.num_nodes}"
        plt.title(f"{title}\n{subtitle}")
        
        # Save figure
        plot_path = os.path.join(self.plot_dir, f"{graph_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization saved to {plot_path}")
    
    def run_pipeline(self) -> Data:
        """Run the complete graph building pipeline."""
        # Build from scratch
        self.load_dataset()
        self.build_graph()  # Build both nodes and edges
        
        # Save the graph
        self.save_graph()
        
        return self.graph_data


def parse_arguments():
    """Parse command-line arguments with helpful descriptions."""
    parser = ArgumentParser(description="Build graph for few-shot fake news detection")
    
    # Required arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="politifact",
        help="Dataset to use (e.g., politifact, gossipcop, kdd2020, tfg)",
        choices=["tfg", "kdd2020", "gossipcop", "politifact"],
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=8,
        help="Number of samples per class for few-shot learning (e.g., 8, 16, 32)",
    )
    
    # Graph construction arguments
    parser.add_argument(
        "--edge_policy",
        type=str,
        default="knn",
        help="Edge construction policy (default: knn)",
        choices=["knn", "thresholdnn"],
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=DEFAULT_K_NEIGHBORS,
        help=f"Number of neighbors for KNN (default: {DEFAULT_K_NEIGHBORS})",
    )
    parser.add_argument(
        "--threshold_factor",
        type=float,
        default=1.0,
        help="Threshold factor for threshold-based edge construction (default: 1.0)",
    )
    
    # New argument for embedding type
    parser.add_argument(
        "--embedding_type",
        type=str,
        default=DEFAULT_EMBEDDING_TYPE,
        help=f"Type of embeddings to use (default: {DEFAULT_EMBEDDING_TYPE})",
        choices=["bert", "roberta"],
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=GRAPH_DIR,
        help=f"Directory to save graphs (default: {GRAPH_DIR})",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable graph visualization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run the graph building pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Set seed for reproducibility 
    set_seed(args.seed)
    
    # Clean up CUDA memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Display arguments and hardware info
    print("\n" + "="*60)
    print("Fake News Detection - Graph Building Pipeline")
    print("="*60)
    print(f"Dataset:          {args.dataset_name}")
    print(f"Embedding type:   {args.embedding_type}")
    print(f"Few-shot k:       {args.k_shot} per class")
    print(f"Edge policy:      {args.edge_policy}")
    
    if args.edge_policy == "knn":
        print(f"K neighbors:      {args.k_neighbors}")
    else:
        print(f"Threshold factor: {args.threshold_factor}")
    
    print(f"Output directory: {args.output_dir}")
    print(f"Plot:             {args.plot}")
    print(f"Seed:             {args.seed}")
    print(f"Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")
    
    # Create graph builder - Explicitly pass each parameter
    builder = GraphBuilder(
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        edge_policy=args.edge_policy,
        k_neighbors=args.k_neighbors,
        threshold_factor=args.threshold_factor,
        output_dir=args.output_dir,
        plot=args.plot,
        seed=args.seed,
        embedding_type=args.embedding_type
    )
    
    # Run pipeline
    graph_data = builder.run_pipeline()
    
    # Display final results
    print("\n" + "="*60)
    print("Graph Building Complete")
    print("="*60)
    print(f"Embedding type:   {args.embedding_type}")
    print(f"Nodes:            {graph_data.num_nodes}")
    print(f"Features:         {graph_data.num_features}")
    print(f"Edges:            {graph_data.edge_index.shape[1]}")
    print(f"Train nodes:      {graph_data.train_mask.sum().item()}")
    print(f"Test nodes:       {graph_data.test_mask.sum().item()}")
    print(f"Few-shot labeled: {graph_data.labeled_mask.sum().item()}")
    
    # Print next steps
    print("\nNext Steps:")
    print("  1. Train a GNN model: python train_graph.py --graph <graph_path>")
    print("  2. Compare with language models: python finetune_lm.py --dataset_name <dataset> --k_shot <k>")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()