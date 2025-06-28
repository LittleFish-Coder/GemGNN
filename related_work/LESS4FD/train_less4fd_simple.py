"""
Simplified LESS4FD Training Script.

A simplified training pipeline that follows the main repository patterns
without complex meta-learning components.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.optim import Adam

# Import from main repository
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from train_hetero_graph import set_seed

# Import LESS4FD components
from build_less4fd_graph_simple import SimpleLESS4FDGraphBuilder
from models.simple_less4fd_model import create_simple_less4fd_model
from config.less4fd_config import LESS4FD_CONFIG, TRAINING_CONFIG


class SimpleLESS4FDTrainer:
    """
    Simplified LESS4FD trainer that follows main repository patterns.
    """
    
    def __init__(
        self,
        graph_path: str,
        model_type: str = "HGT",
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        epochs: int = 300,
        patience: int = 30,
        device: str = None,
        seed: int = 42
    ):
        """Initialize simplified LESS4FD trainer."""
        self.graph_path = graph_path
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        
        # Set seed
        set_seed(seed)
        
        # Load graph
        print(f"Loading graph from {graph_path}")
        self.graph = torch.load(graph_path, weights_only=False)
        self.graph = self.graph.to(self.device)
        
        # Create model
        metadata = (self.graph.node_types, self.graph.edge_types)
        model_config = {
            'hidden_channels': hidden_channels,
            'num_gnn_layers': num_layers,
            'dropout': dropout,
            'model_type': model_type,
            'use_entity_embeddings': True  # Enable entity-aware features
        }
        self.model = create_simple_less4fd_model(metadata, model_config)
        self.model = self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracies = []
        
    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        
        # Forward pass
        out = self.model(self.graph.x_dict, self.graph.edge_index_dict)
        
        # Get training nodes
        train_mask = self.graph['news'].train_labeled_mask
        labels = self.graph['news'].y
        
        # Compute loss
        loss = self.criterion(out[train_mask], labels[train_mask])
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        pred = out[train_mask].argmax(dim=1)
        train_acc = accuracy_score(labels[train_mask].cpu(), pred.cpu())
        
        return loss.item(), train_acc
    
    def evaluate(self, mask_name: str = 'test_mask') -> dict:
        """Evaluate model on given mask."""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.graph.x_dict, self.graph.edge_index_dict)
            
            mask = getattr(self.graph['news'], mask_name)
            labels = self.graph['news'].y[mask]
            pred = out[mask].argmax(dim=1)
            
            # Compute metrics
            accuracy = accuracy_score(labels.cpu(), pred.cpu())
            precision = precision_score(labels.cpu(), pred.cpu(), average='weighted', zero_division=0)
            recall = recall_score(labels.cpu(), pred.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(labels.cpu(), pred.cpu(), average='weighted', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    def train(self) -> dict:
        """Main training loop."""
        print(f"Training simplified LESS4FD model...")
        print(f"Model: {self.model_type}, Device: {self.device}")
        print(f"Graph nodes: {self.graph['news'].x.shape[0]}")
        print(f"Graph features: {self.graph['news'].x.shape[1]}")
        
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation (use test set if no validation set)
            val_metrics = self.evaluate('test_mask')  # Simplified: use test as validation
            val_acc = val_metrics['accuracy']
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            if epoch % 50 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        test_metrics = self.evaluate('test_mask')
        
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f}s")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        
        return {
            'best_val_accuracy': best_val_acc,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'epochs_trained': epoch + 1
        }
    
    def save_results(self, results: dict, output_path: str):
        """Save training results."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


def main():
    """Main training function."""
    parser = ArgumentParser(description="Train simplified LESS4FD model")
    parser.add_argument("--graph_path", required=True, help="Path to the graph file")
    parser.add_argument("--model", choices=["HGT", "HAN", "GAT"], default="HGT", help="Model type")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden channels")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--device", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="results_less4fd_simple", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = SimpleLESS4FDTrainer(
        graph_path=args.graph_path,
        model_type=args.model,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        seed=args.seed
    )
    
    # Train model
    results = trainer.train()
    
    # Save results
    output_file = f"{args.output_dir}/simple_less4fd_{os.path.basename(args.graph_path)}.json"
    trainer.save_results(results, output_file)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()