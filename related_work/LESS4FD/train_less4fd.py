"""
Simplified LESS4FD Training Script.

A self-contained training pipeline for entity-aware fake news detection
without complex meta-learning components.
"""

import os
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
from torch_geometric.nn import HGTConv, HANConv, Linear, GATv2Conv
from typing import Dict, Optional

# Copy necessary utilities
def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


class SimpleLESS4FDModel(nn.Module):
    """
    Simplified LESS4FD model that extends standard GNN architectures
    with basic entity-aware features.
    """
    
    def __init__(
        self,
        metadata: tuple,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        model_type: str = "HGT",
        entity_aware: bool = True
    ):
        """
        Initialize simplified LESS4FD model.
        
        Args:
            metadata: Graph metadata (node_types, edge_types)
            hidden_channels: Hidden dimension size
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            model_type: Type of GNN model (HGT, HAN, GAT)
            entity_aware: Whether to use entity-aware features
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.model_type = model_type
        self.entity_aware = entity_aware
        
        node_types, edge_types = metadata
        
        # Node type embeddings
        self.node_embeddings = nn.ModuleDict()
        for node_type in node_types:
            # Will be set dynamically based on input features
            self.node_embeddings[node_type] = None
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        for _ in range(num_layers):
            if model_type == "HGT":
                self.convs.append(
                    HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
                )
            elif model_type == "HAN":
                self.convs.append(
                    HANConv(hidden_channels, hidden_channels, metadata, num_heads)
                )
            elif model_type == "GAT":
                # Convert to homogeneous for GAT
                self.convs.append(
                    GATv2Conv(hidden_channels, hidden_channels // num_heads, 
                             heads=num_heads, dropout=dropout)
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Entity-aware enhancement (if enabled)
        if entity_aware:
            self.entity_attention = nn.MultiheadAttention(
                embed_dim=hidden_channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.entity_norm = nn.LayerNorm(hidden_channels)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, 2)  # Binary classification
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[tuple, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the simplified LESS4FD model.
        
        Args:
            x_dict: Node features for each node type
            edge_index_dict: Edge indices for each edge type
            
        Returns:
            Logits for news node classification
        """
        # Initialize node embeddings if not done yet
        for node_type, x in x_dict.items():
            if self.node_embeddings[node_type] is None:
                embedding_layer = Linear(x.size(1), self.hidden_channels)
                embedding_layer = embedding_layer.to(x.device)
                self.node_embeddings[node_type] = embedding_layer
                # Register as submodule to ensure it's moved with the model
                self.add_module(f'node_embedding_{node_type}', embedding_layer)
        
        # Project node features to hidden dimension
        for node_type in x_dict:
            x_dict[node_type] = self.node_embeddings[node_type](x_dict[node_type])
        
        # Apply GNN layers
        for conv in self.convs:
            if self.model_type == "GAT":
                # For GAT, convert to homogeneous
                news_x = x_dict['news']
                if ('news', 'similar_to', 'news') in edge_index_dict:
                    edge_index = edge_index_dict[('news', 'similar_to', 'news')]
                    news_x = conv(news_x, edge_index)
                x_dict['news'] = news_x
            else:
                # For HGT/HAN, use heterogeneous forward
                x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation and dropout
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                     for key, x in x_dict.items()}
        
        # Get news node embeddings
        news_embeddings = x_dict['news']
        
        # Apply entity-aware enhancement if enabled
        if self.entity_aware:
            news_embeddings = self.apply_entity_awareness(news_embeddings)
        
        # Classification
        logits = self.classifier(news_embeddings)
        
        return logits
    
    def apply_entity_awareness(self, news_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply entity-aware self-attention to news embeddings.
        
        Args:
            news_embeddings: News node embeddings
            
        Returns:
            Enhanced embeddings with entity-awareness
        """
        # Reshape for self-attention (add sequence dimension)
        batch_size = news_embeddings.size(0)
        embeddings = news_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply self-attention
        attended, _ = self.entity_attention(embeddings, embeddings, embeddings)
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # Residual connection and normalization
        enhanced = self.entity_norm(news_embeddings + attended)
        
        return enhanced


class SimpleLESS4FDTrainer:
    """
    Simplified LESS4FD trainer for entity-aware fake news detection.
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
        self.seed = seed
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load graph
        print(f"Loading graph from: {graph_path}")
        self.graph = torch.load(graph_path, map_location=self.device, weights_only=False)
        
        # Move graph to device
        self.graph = self.graph.to(self.device)
        
        # Initialize model
        metadata = (self.graph.node_types, self.graph.edge_types)
        self.model = SimpleLESS4FDModel(
            metadata=metadata,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type,
            entity_aware=True
        ).to(self.device)
        
        # Ensure model is on correct device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training metrics
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        
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
            
            # Compute loss
            loss = self.criterion(out[mask], labels).item()
            
            # Compute metrics
            accuracy = accuracy_score(labels.cpu(), pred.cpu())
            precision = precision_score(labels.cpu(), pred.cpu(), average='weighted', zero_division=0)
            recall = recall_score(labels.cpu(), pred.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(labels.cpu(), pred.cpu(), average='weighted', zero_division=0)
            
            return {
                'loss': loss,
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
            val_loss = val_metrics['loss']
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)
            
            # Early stopping based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 50 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch} (patience: {self.patience})")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_metrics = self.evaluate('test_mask')
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final test metrics:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return {
            'final_metrics': final_metrics,
            'training_time': training_time,
            'best_val_acc': best_val_acc,
            'total_epochs': epoch + 1
        }
    
    def save_results(self, results: dict, output_dir: str = "results_less4fd") -> str:
        """Save training results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        graph_name = os.path.basename(self.graph_path).replace('.pt', '')
        filename = f"{graph_name}_{self.model_type}_results.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save training plots
        self.plot_training_curves(output_dir, graph_name)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def plot_training_curves(self, output_dir: str, graph_name: str) -> None:
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"{graph_name}_{self.model_type}_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_path}")


def main():
    """Main function for training LESS4FD models."""
    parser = ArgumentParser(description="Train simplified LESS4FD model")
    parser.add_argument("--graph_path", required=True, help="Path to the graph file")
    parser.add_argument("--model_type", choices=["HGT", "HAN", "GAT"], 
                       default="HGT", help="Model type")
    parser.add_argument("--hidden_channels", type=int, default=64, 
                       help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, 
                       help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=5e-4, 
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, 
                       help="Weight decay")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=30, 
                       help="Early stopping patience")
    parser.add_argument("--output_dir", default="results_less4fd", 
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Check if graph file exists
    if not os.path.exists(args.graph_path):
        print(f"Error: Graph file not found: {args.graph_path}")
        return
    
    # Initialize trainer
    trainer = SimpleLESS4FDTrainer(
        graph_path=args.graph_path,
        model_type=args.model_type,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed
    )
    
    # Train model
    results = trainer.train()
    
    # Save results
    trainer.save_results(results, args.output_dir)
    
    print(f"\nLESS4FD training completed!")


if __name__ == "__main__":
    main()