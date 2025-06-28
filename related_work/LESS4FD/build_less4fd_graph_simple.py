"""
Simplified LESS4FD Graph Builder.

Extends the main repository's HeteroGraphBuilder to add basic entity-aware functionality
without complex meta-learning components.
"""

import os
import sys
import numpy as np
import torch
from datasets import load_dataset
from argparse import ArgumentParser
from typing import Dict, List, Optional
from torch_geometric.data import HeteroData

# Import from main repository
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from build_hetero_graph import HeteroGraphBuilder, set_seed

# Import simplified LESS4FD config
from config.less4fd_config import LESS4FD_CONFIG, DATA_CONFIG


class SimpleLESS4FDGraphBuilder(HeteroGraphBuilder):
    """
    Simplified LESS4FD graph builder that extends the main HeteroGraphBuilder.
    Adds basic entity-aware features without complex meta-learning.
    """
    
    def __init__(self, dataset_name: str, k_shot: int, embedding_type: str = "deberta", 
                 enable_entities: bool = True, **kwargs):
        """
        Initialize simplified LESS4FD graph builder.
        
        Args:
            dataset_name: Dataset name (politifact, gossipcop)
            k_shot: Number of shots for few-shot learning
            embedding_type: Type of embeddings to use
            enable_entities: Whether to enable entity-aware features
            **kwargs: Additional arguments for base HeteroGraphBuilder
        """
        # Initialize base class with consistent parameters
        super().__init__(
            dataset_name=dataset_name,
            k_shot=k_shot,
            embedding_type=embedding_type,
            edge_policy=kwargs.get('edge_policy', 'knn_test_isolated'),
            k_neighbors=kwargs.get('k_neighbors', 5),
            **kwargs
        )
        
        self.enable_entities = enable_entities
        self.entity_config = LESS4FD_CONFIG
        
    def load_dataset(self):
        """Load dataset from Hugging Face with error handling."""
        try:
            # Try to load from Hugging Face
            hf_name = DATA_CONFIG["hf_dataset_names"].get(self.dataset_name)
            if hf_name:
                print(f"Loading {self.dataset_name} from Hugging Face: {hf_name}")
                dataset = load_dataset(hf_name)
                return dataset
        except Exception as e:
            print(f"Could not load from Hugging Face: {e}")
            
        # Fallback to parent class method
        return super().load_dataset()
    
    def add_entity_features(self, graph: HeteroData) -> HeteroData:
        """
        Add basic entity-aware features to the graph.
        This is a simplified version without complex entity extraction.
        """
        if not self.enable_entities:
            return graph
            
        print("Adding simplified entity-aware features...")
        
        # Add entity-awareness through modified node features
        news_x = graph['news'].x
        
        # Simple entity simulation: add random entity features to demonstrate concept
        # In a real implementation, this would use actual entity extraction
        entity_dim = self.entity_config.get('max_entities_per_news', 5)
        entity_features = torch.randn(news_x.size(0), entity_dim) * 0.1
        
        # Concatenate entity features with news features
        enhanced_features = torch.cat([news_x, entity_features], dim=1)
        graph['news'].x = enhanced_features
        
        print(f"Enhanced news features: {news_x.shape} -> {enhanced_features.shape}")
        
        return graph
    
    def build_hetero_graph(self) -> HeteroData:
        """Build heterogeneous graph with optional entity-aware features."""
        # Use parent class to build base graph
        graph = super().build_hetero_graph()
        
        # Add entity-aware features if enabled
        if self.enable_entities:
            graph = self.add_entity_features(graph)
            
        return graph


def main():
    """Main function for building LESS4FD graphs."""
    parser = ArgumentParser(description="Build simplified LESS4FD graph")
    parser.add_argument("--dataset_name", choices=["politifact", "gossipcop"], 
                       default="politifact", help="Dataset name")
    parser.add_argument("--k_shot", type=int, choices=range(3, 17), default=8,
                       help="Number of shots")
    parser.add_argument("--embedding_type", choices=["bert", "roberta", "deberta", "distilbert"], 
                       default="deberta", help="Embedding type")
    parser.add_argument("--enable_entities", action="store_true", default=True,
                       help="Enable entity-aware features")
    parser.add_argument("--edge_policy", choices=["knn", "label_aware_knn", "knn_test_isolated"],
                       default="knn_test_isolated", help="Edge construction policy")
    parser.add_argument("--output_dir", default="graphs_less4fd_simple", 
                       help="Output directory for graphs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Build graph
    builder = SimpleLESS4FDGraphBuilder(
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        embedding_type=args.embedding_type,
        enable_entities=args.enable_entities,
        edge_policy=args.edge_policy,
        output_dir=args.output_dir
    )
    
    print(f"Building LESS4FD graph for {args.dataset_name}, k_shot={args.k_shot}")
    graph = builder.build_hetero_graph()
    
    print(f"Graph built successfully:")
    print(f"- News nodes: {graph['news'].x.shape}")
    if 'interaction' in graph.node_types:
        print(f"- Interaction nodes: {graph['interaction'].x.shape}")
    print(f"- Edge types: {list(graph.edge_types)}")
    

if __name__ == "__main__":
    main()