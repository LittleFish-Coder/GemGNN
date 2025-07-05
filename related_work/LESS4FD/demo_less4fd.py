#!/usr/bin/env python3
"""
LESS4FD Complete Pipeline Demo - Complete Rewrite

Demonstrates the complete simplified LESS4FD pipeline:
1. Graph construction with entity-aware features
2. Model training with few-shot learning
3. Evaluation and results saving

This is a complete rewrite as requested in the issue, implementing a "trivial"
version without meta-learning that integrates with the main repository's
few-shot learning framework.
"""

import os
import sys
import json
import torch
from argparse import ArgumentParser

# Add main repository to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our new implementations
from build_less4fd import LESS4FDGraphBuilder
from train_less4fd import LESS4FDTrainer

# Try to import from main repository
try:
    from build_hetero_graph import set_seed, DEFAULT_SEED
    MAIN_REPO_AVAILABLE = True
except ImportError:
    MAIN_REPO_AVAILABLE = False
    DEFAULT_SEED = 42
    
    def set_seed(seed: int = 42):
        torch.manual_seed(seed)


def run_complete_less4fd_pipeline(dataset_name: str = "politifact", 
                                 k_shot: int = 8,
                                 embedding_type: str = "deberta",
                                 model_type: str = "HGT",
                                 entity_dim: int = 64,
                                 epochs: int = 50,
                                 seed: int = 42):
    """
    Run the complete LESS4FD pipeline from graph construction to evaluation.
    
    Args:
        dataset_name: Dataset name (politifact, gossipcop)
        k_shot: Number of shots per class
        embedding_type: Embedding type (bert, roberta, deberta)
        model_type: GNN model type (HGT, HAN)
        entity_dim: Entity feature dimension
        epochs: Training epochs
        seed: Random seed
        
    Returns:
        Results dictionary
    """
    
    print("=" * 70)
    print("LESS4FD Complete Pipeline - Complete Rewrite")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  K-shot: {k_shot}")
    print(f"  Embedding: {embedding_type}")
    print(f"  Model: {model_type}")
    print(f"  Entity dim: {entity_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Seed: {seed}")
    print(f"  Main repo available: {MAIN_REPO_AVAILABLE}")
    print()
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Step 1: Graph Construction
    print("Step 1: Building LESS4FD Graph")
    print("-" * 35)
    
    graph_builder = LESS4FDGraphBuilder(
        dataset_name=dataset_name,
        k_shot=k_shot,
        embedding_type=embedding_type,
        enable_entities=True,
        entity_dim=entity_dim
    )
    
    # Create output directories
    os.makedirs("graphs_less4fd", exist_ok=True)
    os.makedirs("results_less4fd", exist_ok=True)
    
    # Build and save graph
    graph_name = f"less4fd_{dataset_name}_{k_shot}shot_{embedding_type}_entities{entity_dim}.pt"
    graph_path = os.path.join("graphs_less4fd", graph_name)
    
    graph = graph_builder.build_graph(save_path=graph_path)
    
    print(f"✓ Graph construction completed")
    print(f"  Graph saved to: {graph_path}")
    print()
    
    # Step 2: Model Training
    print("Step 2: Training LESS4FD Model")
    print("-" * 35)
    
    trainer = LESS4FDTrainer(
        graph_path=graph_path,
        model_type=model_type,
        hidden_channels=64,
        num_layers=2,
        dropout=0.3,
        num_heads=4,
        learning_rate=5e-4,
        weight_decay=1e-3
    )
    
    # Train model
    results = trainer.train(
        epochs=epochs,
        patience=30,
        eval_every=10
    )
    
    print(f"✓ Model training completed")
    print()
    
    # Step 3: Save Results
    print("Step 3: Saving Results")
    print("-" * 35)
    
    # Save detailed results
    result_name = f"less4fd_{dataset_name}_{k_shot}shot_{model_type}_entities{entity_dim}.json"
    result_path = os.path.join("results_less4fd", result_name)
    
    # Add configuration to results
    results['config'] = {
        'dataset_name': dataset_name,
        'k_shot': k_shot,
        'embedding_type': embedding_type,
        'model_type': model_type,
        'entity_dim': entity_dim,
        'epochs': epochs,
        'seed': seed,
        'main_repo_available': MAIN_REPO_AVAILABLE
    }
    
    trainer.save_results(results, result_path)
    
    print(f"✓ Results saved to: {result_path}")
    print()
    
    # Step 4: Summary
    print("Step 4: Results Summary")
    print("-" * 35)
    
    test_metrics = results['final_test_metrics']
    print(f"Final Test Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print()
    print(f"Training Details:")
    print(f"  Best Val Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"  Epochs Trained: {results['epochs_trained']}")
    print(f"  Training Time: {results['training_time']:.2f}s")
    print()
    
    # Performance summary
    if MAIN_REPO_AVAILABLE:
        status = "✓ Using main repository integration"
    else:
        status = "⚠ Using fallback mode (test data)"
    
    print("Pipeline Status:")
    print(f"  {status}")
    print(f"  ✓ Entity-aware features enabled ({entity_dim} dims)")
    print(f"  ✓ Few-shot learning ({k_shot} shots per class)")
    print(f"  ✓ No meta-learning (trivial implementation)")
    print(f"  ✓ Compatible with main repository patterns")
    
    print("\n" + "=" * 70)
    print("LESS4FD Pipeline Completed Successfully!")
    print("=" * 70)
    
    return results


def main():
    """Command line interface for LESS4FD pipeline demo."""
    parser = ArgumentParser(description="LESS4FD Complete Pipeline Demo")
    parser.add_argument("--dataset", default="politifact", choices=["politifact", "gossipcop"],
                       help="Dataset name")
    parser.add_argument("--k_shot", type=int, default=8,
                       help="Number of shots per class")
    parser.add_argument("--embedding", default="deberta", choices=["bert", "roberta", "deberta"],
                       help="Embedding type")
    parser.add_argument("--model", default="HGT", choices=["HGT", "HAN"],
                       help="Model type")
    parser.add_argument("--entity_dim", type=int, default=64,
                       help="Entity feature dimension")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs (reduced for demo)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Run complete pipeline
    results = run_complete_less4fd_pipeline(
        dataset_name=args.dataset,
        k_shot=args.k_shot,
        embedding_type=args.embedding,
        model_type=args.model,
        entity_dim=args.entity_dim,
        epochs=args.epochs,
        seed=args.seed
    )
    
    return results


if __name__ == "__main__":
    main()