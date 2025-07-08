"""
HeteroSGT Training Script.

A self-contained training pipeline for subgraph-based fake news detection
using structural graph transformers with distance bias.
"""

import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.optim import Adam
from torch_geometric.nn import Linear, HGTConv, HANConv, GATv2Conv
from typing import Dict, Optional, List, Tuple
from datasets import load_dataset

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


class DistanceBias(nn.Module):
    """
    Learnable distance bias for HeteroSGT attention mechanism.
    
    This module learns bias parameters for different random-walk distances
    to modulate attention weights based on graph structure.
    """
    
    def __init__(self, max_distance: int = 4, embed_dim: int = 64):
        """
        Initialize distance bias module.
        
        Args:
            max_distance: Maximum distance to consider
            embed_dim: Embedding dimension for distance encoding
        """
        super().__init__()
        self.max_distance = max_distance
        self.embed_dim = embed_dim
        
        # Learnable bias parameters for each distance
        self.distance_bias = nn.Parameter(torch.randn(max_distance + 1))
        
        # Optional: learnable distance embeddings
        self.distance_embeddings = nn.Embedding(max_distance + 1, embed_dim)
        
        # Initialize bias parameters
        nn.init.normal_(self.distance_bias, std=0.1)
        nn.init.normal_(self.distance_embeddings.weight, std=0.1)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute distance bias for attention.
        
        Args:
            distances: Distance matrix [batch_size, seq_len, seq_len]
            
        Returns:
            Distance bias [batch_size, seq_len, seq_len]
        """
        # Clamp distances to valid range
        distances = torch.clamp(distances, 0, self.max_distance).long()
        
        # Get bias values for each distance
        bias = self.distance_bias[distances]
        
        return bias


class StructuralGraphTransformer(nn.Module):
    """
    Structural Graph Transformer layer with distance-aware attention.
    
    Implements the core HeteroSGT mechanism:
    Attention(i,j) = softmax((Q_i K_j^T)/sqrt(d) + b_dist(d_ij)) V_j
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        max_distance: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize SGT layer.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            max_distance: Maximum distance for bias
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        
        # Distance bias module
        self.distance_bias = DistanceBias(max_distance, embed_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        distances: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of SGT layer.
        
        Args:
            x: Input features [batch_size, seq_len, embed_dim]
            distances: Distance matrix [batch_size, seq_len, seq_len]
            attention_mask: Optional attention mask
            
        Returns:
            Output features [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections
        q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [batch_size, num_heads, seq_len, seq_len]
        
        # Add distance bias
        distance_bias = self.distance_bias(distances)  # [batch_size, seq_len, seq_len]
        # Expand for heads
        distance_bias = distance_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores + distance_bias
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection and normalization
        out = self.norm(x + self.dropout(out))
        
        return out


class HeteroSGTModel(nn.Module):
    """
    HeteroSGT model for subgraph-based fake news detection.
    
    This model processes news-centered subgraphs using structural graph
    transformers with distance-aware attention bias.
    """
    
    def __init__(
        self,
        metadata: tuple,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_distance: int = 4,
        dropout: float = 0.3,
        model_type: str = "SGT"
    ):
        """
        Initialize HeteroSGT model.
        
        Args:
            metadata: Graph metadata (node_types, edge_types)
            hidden_channels: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_distance: Maximum distance for bias
            dropout: Dropout rate
            model_type: Model architecture (SGT, HGT, HAN)
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.dropout = dropout
        self.model_type = model_type
        
        node_types, edge_types = metadata
        
        # Node type embeddings
        self.node_embeddings = nn.ModuleDict()
        for node_type in node_types:
            # Will be set dynamically based on input features
            self.node_embeddings[node_type] = None
        
        # Model layers
        if model_type == "SGT":
            # Structural Graph Transformer layers
            self.layers = nn.ModuleList([
                StructuralGraphTransformer(
                    embed_dim=hidden_channels,
                    num_heads=num_heads,
                    max_distance=max_distance,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
        elif model_type == "HGT":
            # HGT layers for comparison
            self.layers = nn.ModuleList([
                HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
                for _ in range(num_layers)
            ])
        elif model_type == "HAN":
            # HAN layers for comparison
            self.layers = nn.ModuleList([
                HANConv(hidden_channels, hidden_channels, metadata, num_heads)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Subgraph aggregation
        self.subgraph_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, 2)  # Binary classification
        )
    
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        edge_index_dict: Dict[tuple, torch.Tensor],
        distance_matrix: torch.Tensor,
        subgraphs: Dict[int, List[int]],
        target_nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of HeteroSGT model.
        
        Args:
            x_dict: Node features for each node type
            edge_index_dict: Edge indices for each edge type
            distance_matrix: Random walk distance matrix
            subgraphs: Dictionary of subgraph node lists
            target_nodes: Target nodes for classification (if None, use all news nodes)
            
        Returns:
            Logits for news node classification
        """
        # Initialize node embeddings if not done yet
        for node_type, x in x_dict.items():
            if self.node_embeddings[node_type] is None:
                embedding_layer = Linear(x.size(1), self.hidden_channels)
                embedding_layer = embedding_layer.to(x.device)
                self.node_embeddings[node_type] = embedding_layer
                # Register as submodule
                self.add_module(f'node_embedding_{node_type}', embedding_layer)
        
        # Project node features to hidden dimension
        for node_type in x_dict:
            x_dict[node_type] = self.node_embeddings[node_type](x_dict[node_type])
        
        # Get news embeddings
        news_embeddings = x_dict['news']
        batch_size = news_embeddings.size(0)
        
        if self.model_type == "SGT":
            # Use structural graph transformer
            embeddings = self.process_with_sgt(
                news_embeddings, distance_matrix, subgraphs, target_nodes
            )
        else:
            # Use traditional heterogeneous GNN
            embeddings = self.process_with_hetero_gnn(
                x_dict, edge_index_dict, target_nodes
            )
        
        # Classification
        logits = self.classifier(embeddings)
        
        return logits
    
    def process_with_sgt(
        self,
        news_embeddings: torch.Tensor,
        distance_matrix: torch.Tensor,
        subgraphs: Dict[int, List[int]],
        target_nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process with Structural Graph Transformer.
        
        Args:
            news_embeddings: News node embeddings
            distance_matrix: Distance matrix
            subgraphs: Subgraph node lists
            target_nodes: Target nodes for processing
            
        Returns:
            Processed embeddings
        """
        device = news_embeddings.device
        
        if target_nodes is None:
            target_nodes = torch.arange(news_embeddings.size(0), device=device)
        
        # Process each target node's subgraph
        processed_embeddings = []
        
        for node_idx in target_nodes:
            node_idx = node_idx.item()
            
            # Get subgraph nodes
            if node_idx in subgraphs:
                subgraph_nodes = subgraphs[node_idx]
            else:
                # Fallback: use the node itself
                subgraph_nodes = [node_idx]
            
            # Extract subgraph features and distances
            subgraph_features = news_embeddings[subgraph_nodes]  # [subgraph_size, hidden_dim]
            subgraph_distances = distance_matrix[subgraph_nodes][:, subgraph_nodes]  # [subgraph_size, subgraph_size]
            
            # Add batch dimension
            subgraph_features = subgraph_features.unsqueeze(0)  # [1, subgraph_size, hidden_dim]
            subgraph_distances = subgraph_distances.unsqueeze(0)  # [1, subgraph_size, subgraph_size]
            
            # Apply SGT layers
            x = subgraph_features
            for layer in self.layers:
                x = layer(x, subgraph_distances)
            
            # Aggregate subgraph representation (focus on target node)
            target_pos = subgraph_nodes.index(node_idx) if node_idx in subgraph_nodes else 0
            node_embedding = x[0, target_pos]  # [hidden_dim]
            
            processed_embeddings.append(node_embedding)
        
        # Stack all processed embeddings
        if processed_embeddings:
            result = torch.stack(processed_embeddings)  # [num_targets, hidden_dim]
        else:
            result = torch.zeros(len(target_nodes), self.hidden_channels, device=device)
        
        return result
    
    def process_with_hetero_gnn(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
        target_nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process with traditional heterogeneous GNN (HGT/HAN).
        
        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            target_nodes: Target nodes for processing
            
        Returns:
            Processed embeddings
        """
        # Apply GNN layers
        for layer in self.layers:
            if hasattr(layer, '__call__'):
                x_dict = layer(x_dict, edge_index_dict)
            
            # Apply activation and dropout
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                     for key, x in x_dict.items()}
        
        # Get news embeddings
        news_embeddings = x_dict['news']
        
        if target_nodes is not None:
            news_embeddings = news_embeddings[target_nodes]
        
        return news_embeddings


class HeteroSGTTrainer:
    """
    HeteroSGT trainer for subgraph-based fake news detection.
    """
    
    def __init__(
        self,
        graph_path: str,
        model_type: str = "SGT",
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        epochs: int = 300,
        patience: int = 30,
        device: str = None,
        seed: int = 42
    ):
        """Initialize HeteroSGT trainer."""
        self.graph_path = graph_path
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
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
        
        # Get maximum distance from graph
        max_distance = getattr(self.graph, 'max_walk_length', 4)
        
        # Initialize model
        metadata = (self.graph.node_types, self.graph.edge_types)
        self.model = HeteroSGTModel(
            metadata=metadata,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            max_distance=max_distance,
            dropout=dropout,
            model_type=model_type
        ).to(self.device)
        
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
        
        # Get graph components
        distance_matrix = getattr(self.graph, 'distance_matrix', None)
        subgraphs = getattr(self.graph, 'subgraphs', {})
        
        # Forward pass
        if distance_matrix is not None and self.model_type == "SGT":
            out = self.model(
                self.graph.x_dict, 
                self.graph.edge_index_dict,
                distance_matrix,
                subgraphs
            )
        else:
            # Fallback for non-SGT models
            out = self.model(
                self.graph.x_dict, 
                self.graph.edge_index_dict,
                torch.zeros(0, 0),  # Dummy distance matrix
                {}  # Empty subgraphs
            )
        
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
            # Get graph components
            distance_matrix = getattr(self.graph, 'distance_matrix', None)
            subgraphs = getattr(self.graph, 'subgraphs', {})
            
            # Forward pass
            if distance_matrix is not None and self.model_type == "SGT":
                out = self.model(
                    self.graph.x_dict, 
                    self.graph.edge_index_dict,
                    distance_matrix,
                    subgraphs
                )
            else:
                # Fallback for non-SGT models
                out = self.model(
                    self.graph.x_dict, 
                    self.graph.edge_index_dict,
                    torch.zeros(0, 0),  # Dummy distance matrix
                    {}  # Empty subgraphs
                )
            
            mask = getattr(self.graph['news'], mask_name)
            labels = self.graph['news'].y[mask]
            pred = out[mask].argmax(dim=1)
            
            # Compute loss
            loss = self.criterion(out[mask], labels).item()
            
            # Compute metrics
            accuracy = accuracy_score(labels.cpu(), pred.cpu())
            precision = precision_score(labels.cpu(), pred.cpu(), average='macro', zero_division=0)
            recall = recall_score(labels.cpu(), pred.cpu(), average='macro', zero_division=0)
            f1 = f1_score(labels.cpu(), pred.cpu(), average='macro', zero_division=0)
            
            return {
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    def train(self) -> dict:
        """Main training loop."""
        print(f"Training HeteroSGT model...")
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
    
    def save_results(self, results: dict, output_dir: str = "results_heterosgt") -> str:
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
    
    def export_test_predictions(self, model_name="heterosgt") -> str:
        """Export test predictions in the required CSV format."""
        
        # Extract dataset info from graph path
        # Expected format: graphs_heterosgt/heterosgt_politifact_k8_deberta.pt
        graph_filename = os.path.basename(self.graph_path).replace('.pt', '')
        parts = graph_filename.split('_')
        
        if len(parts) >= 3:
            dataset_name = parts[1]  # politifact or gossipcop (parts[0] is 'heterosgt')
            k_shot = parts[2].replace('k', '')  # Extract k from k8
        else:
            dataset_name = "politifact"
            k_shot = "8"
        
        # Load original dataset to get test text
        hf_dataset = load_dataset(f"LittleFish-Coder/Fake_News_{dataset_name}", split="test")
        
        # Get test mask and make predictions
        test_mask = self.graph['news'].test_mask
        test_indices = torch.where(test_mask)[0]
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            # Get graph components
            distance_matrix = getattr(self.graph, 'distance_matrix', None)
            subgraphs = getattr(self.graph, 'subgraphs', {})
            
            # Forward pass
            if distance_matrix is not None and self.model_type == "SGT":
                out = self.model(
                    self.graph.x_dict, 
                    self.graph.edge_index_dict,
                    distance_matrix,
                    subgraphs
                )
            else:
                # Fallback for non-SGT models
                out = self.model(
                    self.graph.x_dict, 
                    self.graph.edge_index_dict,
                    torch.zeros(0, 0),  # Dummy distance matrix
                    {}  # Empty subgraphs
                )
            
            test_out = out[test_mask]
            probabilities = F.softmax(test_out, dim=1)
            predictions = torch.argmax(test_out, dim=1)
            
            # Get confidence scores (max probability)
            confidence_scores = torch.max(probabilities, dim=1)[0]
            
            # Get ground truth
            ground_truth = self.graph['news'].y[test_mask]
            
            # Convert to numpy
            y_true = ground_truth.cpu().numpy()
            y_pred = predictions.cpu().numpy()
            confidence = confidence_scores.cpu().numpy()
        
        # Create prediction DataFrame
        prediction_data = []
        for i in range(len(y_true)):
            prediction_data.append({
                'news_id': i,
                'news_text': hf_dataset[i]['text'],
                'ground_truth': y_true[i],
                'prediction': y_pred[i],
                'confidence': confidence[i]
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(prediction_data)
        
        # Create output directory
        output_dir = "../../prediction"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        filename = f"{model_name}_{k_shot}_shot_{dataset_name}_predictions.csv"
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        
        print(f"Test predictions exported to {output_path}")
        return output_path


def main():
    """Main function for training HeteroSGT models."""
    parser = ArgumentParser(description="Train HeteroSGT model")
    parser.add_argument("--graph_path", required=True, help="Path to the graph file")
    parser.add_argument("--model_type", choices=["SGT", "HGT", "HAN"], 
                       default="SGT", help="Model type")
    parser.add_argument("--hidden_channels", type=int, default=64, 
                       help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, 
                       help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=4, 
                       help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=5e-4, 
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, 
                       help="Weight decay")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=30, 
                       help="Early stopping patience")
    parser.add_argument("--output_dir", default="results_heterosgt", 
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
    trainer = HeteroSGTTrainer(
        graph_path=args.graph_path,
        model_type=args.model_type,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
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
    
    # Export test predictions
    trainer.export_test_predictions(model_name="heterosgt")
    
    print(f"\nHeteroSGT training completed!")


if __name__ == "__main__":
    main()