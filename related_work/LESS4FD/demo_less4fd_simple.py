#!/usr/bin/env python3
"""
LESS4FD Simplified Pipeline Demo

Demonstrates the complete simplified LESS4FD pipeline:
1. Load/create test data
2. Build entity-aware heterogeneous graph  
3. Train simplified LESS4FD model
4. Evaluate results

This follows the main repository patterns without complex meta-learning.
"""

import os
import sys
import torch
from torch_geometric.data import HeteroData

# Add main repository to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import simplified LESS4FD components
from build_less4fd_graph_simple import SimpleLESS4FDGraphBuilder
from train_less4fd_simple import SimpleLESS4FDTrainer
from config.less4fd_config import LESS4FD_CONFIG, TRAINING_CONFIG, FEWSHOT_CONFIG


def create_demo_graph():
    """Create a demo graph for testing purposes."""
    print("Creating demo heterogeneous graph...")
    
    # Create a simple heterogeneous graph for demo
    graph = HeteroData()
    
    # News nodes (simulate BERT embeddings)
    num_news = 20
    graph['news'].x = torch.randn(num_news, 768)
    graph['news'].y = torch.randint(0, 2, (num_news,))  # Binary labels
    
    # Create few-shot masks (8-shot scenario)
    k_shot = 8
    graph['news'].train_labeled_mask = torch.zeros(num_news, dtype=torch.bool)
    graph['news'].train_labeled_mask[:k_shot*2] = True  # 8 per class = 16 total
    
    graph['news'].test_mask = torch.zeros(num_news, dtype=torch.bool)
    graph['news'].test_mask[k_shot*2:] = True  # Remaining as test
    
    # Add interaction nodes (optional, for demo)
    num_interactions = 10
    graph['interaction'].x = torch.randn(num_interactions, 768)
    
    # Add edges
    # News-news similarity edges
    news_edges = torch.randint(0, num_news, (2, 30))
    graph['news', 'similar', 'news'].edge_index = news_edges
    
    # News-interaction edges
    news_interaction_edges = torch.tensor([
        list(range(min(num_news, num_interactions))),  # Source news
        list(range(min(num_news, num_interactions)))   # Target interactions
    ], dtype=torch.long)
    graph['news', 'has_interaction', 'interaction'].edge_index = news_interaction_edges
    graph['interaction', 'belongs_to', 'news'].edge_index = news_interaction_edges.flip(0)
    
    return graph


def demonstrate_entity_awareness(graph):
    """Demonstrate entity-aware feature enhancement."""
    print("\nDemonstrating entity-aware features...")
    
    original_shape = graph['news'].x.shape
    print(f"Original news features: {original_shape}")
    
    # Simulate entity feature addition (as in SimpleLESS4FDGraphBuilder)
    entity_dim = LESS4FD_CONFIG.get('max_entities_per_news', 5)
    entity_features = torch.randn(original_shape[0], entity_dim) * 0.1
    
    # Enhance with entity features
    enhanced_features = torch.cat([graph['news'].x, entity_features], dim=1)
    graph['news'].x = enhanced_features
    
    print(f"Enhanced news features: {graph['news'].x.shape}")
    print(f"Added {entity_dim} entity-aware dimensions")
    
    return graph


def run_simplified_less4fd_demo():
    """Run the complete simplified LESS4FD demonstration."""
    print("🚀 LESS4FD Simplified Pipeline Demo")
    print("=" * 50)
    
    # 1. Configuration
    print(f"\n📋 Configuration:")
    print(f"Entity types: {LESS4FD_CONFIG['entity_types']}")
    print(f"Max entities per news: {LESS4FD_CONFIG['max_entities_per_news']}")
    print(f"Hidden channels: {LESS4FD_CONFIG['hidden_channels']}")
    print(f"Meta-learning: {FEWSHOT_CONFIG['meta_learning']} (disabled)")
    
    # 2. Create demo graph
    print(f"\n🔗 Graph Construction:")
    graph = create_demo_graph()
    print(f"News nodes: {graph['news'].x.shape[0]}")
    print(f"Interaction nodes: {graph['interaction'].x.shape[0]}")
    print(f"Edge types: {list(graph.edge_types)}")
    print(f"Training nodes: {graph['news'].train_labeled_mask.sum()}")
    print(f"Test nodes: {graph['news'].test_mask.sum()}")
    
    # 3. Add entity-aware features
    graph = demonstrate_entity_awareness(graph)
    
    # 4. Save demo graph
    os.makedirs('graphs_less4fd_simple', exist_ok=True)
    graph_path = 'graphs_less4fd_simple/demo_graph.pt'
    torch.save(graph, graph_path)
    print(f"Demo graph saved: {graph_path}")
    
    # 5. Train simplified LESS4FD model
    print(f"\n🤖 Model Training:")
    trainer = SimpleLESS4FDTrainer(
        graph_path=graph_path,
        model_type="HGT",
        hidden_channels=LESS4FD_CONFIG['hidden_channels'],
        num_layers=LESS4FD_CONFIG['num_gnn_layers'],
        dropout=LESS4FD_CONFIG['dropout'],
        epochs=50,  # Reduced for demo
        patience=10,
        seed=42
    )
    
    results = trainer.train()
    
    # 6. Results
    print(f"\n📊 Results:")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test F1-score: {results['test_metrics']['f1']:.4f}")
    print(f"Training time: {results['training_time']:.2f}s")
    print(f"Epochs trained: {results['epochs_trained']}")
    
    # 7. Save results
    os.makedirs('results_less4fd_simple', exist_ok=True)
    output_file = 'results_less4fd_simple/demo_results.json'
    trainer.save_results(results, output_file)
    
    print(f"\n✅ Demo completed successfully!")
    print(f"This demonstrates the simplified LESS4FD pipeline:")
    print(f"  • Entity-aware graph construction")
    print(f"  • Simplified model without meta-learning") 
    print(f"  • Training following main repository patterns")
    print(f"  • Few-shot learning capability")
    
    return results


if __name__ == "__main__":
    try:
        results = run_simplified_less4fd_demo()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)