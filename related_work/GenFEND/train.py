"""
GenFEND Training Script.

Simplified implementation of GenFEND (Generative Multi-view Fake News Detection)
for few-shot fake news detection with demographic view gating mechanism.
Based on the GenFEND paper that uses role-playing LLMs and demographic diversity.
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
from typing import Dict, Optional, Tuple


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


class GenFENDModel(nn.Module):
    """
    Simplified GenFEND model with demographic view gating mechanism.
    
    Based on the GenFEND paper that generates demographic-specific comments
    and uses a learned gating mechanism to combine different demographic views.
    """
    
    def __init__(
        self,
        text_embed_dim: int,
        demographic_embed_dim: int,
        num_views: int = 3,
        num_demographic_profiles: int = 30,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize GenFEND model.
        
        Args:
            text_embed_dim: Dimension of text embeddings
            demographic_embed_dim: Dimension of demographic profile features
            num_views: Number of demographic views (Gender, Age, Education)
            num_demographic_profiles: Number of demographic profiles per view
            hidden_dim: Hidden dimension for neural networks
            dropout: Dropout rate
            device: Device to run the model on
        """
        super().__init__()
        
        self.text_embed_dim = text_embed_dim
        self.demographic_embed_dim = demographic_embed_dim
        self.num_views = num_views
        self.num_demographic_profiles = num_demographic_profiles
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Text embedding projection
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        
        # Demographic feature processing for each view
        self.demographic_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_demographic_profiles * demographic_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_views)
        ])
        
        # View gating mechanism (Equation from GenFEND paper)
        # G(e_o || d; θ) where e_o is news content and d is diversity signal
        self.gate_input_dim = hidden_dim + num_views  # text + diversity signals
        self.gate_network = nn.Sequential(
            nn.Linear(self.gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_views),
            nn.Softmax(dim=-1)  # Softmax for gating weights
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # text + gated demographic features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        text_embeddings: torch.Tensor,
        demographic_features: Dict[str, torch.Tensor],
        diversity_signals: torch.Tensor,
        view_names: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of GenFEND model.
        
        Args:
            text_embeddings: Text embeddings (batch_size, text_embed_dim)
            demographic_features: Dict of demographic features for each view
                Each tensor has shape (batch_size, num_profiles, demographic_embed_dim)
            diversity_signals: Diversity signals (batch_size, num_views)
            view_names: List of view names
            
        Returns:
            logits: Classification logits (batch_size, 2)
            gate_weights: Gating weights (batch_size, num_views)
        """
        batch_size = text_embeddings.size(0)
        
        # Process text embeddings
        text_features = self.text_proj(text_embeddings)  # (batch_size, hidden_dim)
        text_features = F.relu(text_features)
        text_features = self.dropout(text_features)
        
        # Process demographic features for each view
        view_features = []
        for i, view_name in enumerate(view_names):
            # Flatten demographic profiles for this view
            demo_feat = demographic_features[view_name]  # (batch_size, num_profiles, demo_embed_dim)
            demo_feat_flat = demo_feat.view(batch_size, -1)  # (batch_size, num_profiles * demo_embed_dim)
            
            # Process through view-specific network
            view_feat = self.demographic_processors[i](demo_feat_flat)  # (batch_size, hidden_dim)
            view_feat = F.relu(view_feat)
            view_feat = self.dropout(view_feat)
            view_features.append(view_feat)
        
        view_features = torch.stack(view_features, dim=1)  # (batch_size, num_views, hidden_dim)
        
        # Compute gating weights using text content and diversity signals
        # Following GenFEND paper: a = Softmax(G(e_o || d; θ))
        gate_input = torch.cat([text_features, diversity_signals], dim=-1)  # (batch_size, hidden_dim + num_views)
        gate_weights = self.gate_network(gate_input)  # (batch_size, num_views)
        
        # Apply gating to demographic features
        # r = Σ_V a_V * s_V (convex combination as in the paper)
        gated_demographic = torch.sum(
            gate_weights.unsqueeze(-1) * view_features, dim=1
        )  # (batch_size, hidden_dim)
        
        # Concatenate text and gated demographic features for classification
        # Following the paper: concatenate with news embedding e_o
        combined_features = torch.cat([text_features, gated_demographic], dim=-1)  # (batch_size, hidden_dim * 2)
        
        # Final classification
        logits = self.classifier(combined_features)  # (batch_size, 2)
        
        return logits, gate_weights


class GenFENDTrainer:
    """Trainer for GenFEND model."""
    
    def __init__(
        self,
        model: GenFENDModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for few-shot
        
        self.training_history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "val_f1": []
        }
    
    def train_epoch(
        self, 
        text_embeddings: torch.Tensor,
        demographic_features: Dict[str, torch.Tensor],
        diversity_signals: torch.Tensor,
        labels: torch.Tensor,
        view_names: list,
        train_mask: torch.Tensor
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Filter to training samples
        train_indices = torch.where(train_mask)[0]
        if len(train_indices) == 0:
            return 0.0, 0.0
        
        train_text = text_embeddings[train_indices]
        train_demo = {view: demo[train_indices] for view, demo in demographic_features.items()}
        train_div = diversity_signals[train_indices]
        train_labels = labels[train_indices]
        
        # Forward pass
        logits, gate_weights = self.model(train_text, train_demo, train_div, view_names)
        
        # Compute loss
        loss = self.criterion(logits, train_labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == train_labels).float().mean()
        
        return loss.item(), acc.item()
    
    def evaluate(
        self,
        text_embeddings: torch.Tensor,
        demographic_features: Dict[str, torch.Tensor],
        diversity_signals: torch.Tensor,
        labels: torch.Tensor,
        view_names: list,
        eval_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        
        # Filter to evaluation samples
        eval_indices = torch.where(eval_mask)[0]
        if len(eval_indices) == 0:
            return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        eval_text = text_embeddings[eval_indices]
        eval_demo = {view: demo[eval_indices] for view, demo in demographic_features.items()}
        eval_div = diversity_signals[eval_indices]
        eval_labels = labels[eval_indices]
        
        with torch.no_grad():
            logits, gate_weights = self.model(eval_text, eval_demo, eval_div, view_names)
            loss = self.criterion(logits, eval_labels)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Convert to numpy for sklearn metrics
            y_true = eval_labels.cpu().numpy()
            y_pred = preds.cpu().numpy()
            
            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            return {
                "loss": loss.item(),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "gate_weights": gate_weights.cpu().numpy()
            }
    
    def train(
        self,
        data: Dict,
        epochs: int = 200,
        patience: int = 30,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Complete training loop."""
        
        # Prepare data
        text_embeddings = torch.tensor(data["text_embeddings"]).to(self.device)
        demographic_features = {
            view: torch.tensor(feat).to(self.device) 
            for view, feat in data["demographic_features"].items()
        }
        diversity_signals = torch.tensor(data["diversity_signals"]).to(self.device)
        labels = torch.tensor(data["labels"]).to(self.device)
        view_names = data["metadata"]["view_names"]
        
        # Create masks
        total_samples = len(labels)
        train_labeled_mask = torch.zeros(total_samples, dtype=torch.bool)
        train_labeled_mask[data["train_labeled_indices"]] = True
        
        test_mask = torch.zeros(total_samples, dtype=torch.bool)
        test_mask[data["test_indices"]] = True
        
        if verbose:
            print(f"Training samples: {train_labeled_mask.sum()}")
            print(f"Test samples: {test_mask.sum()}")
            print(f"View names: {view_names}")
        
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(
                text_embeddings, demographic_features, diversity_signals, 
                labels, view_names, train_labeled_mask
            )
            
            # Validation (use test set as we're in few-shot setting)
            val_metrics = self.evaluate(
                text_embeddings, demographic_features, diversity_signals,
                labels, view_names, test_mask
            )
            
            # Record history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["val_acc"].append(val_metrics["accuracy"])
            self.training_history["val_f1"].append(val_metrics["f1"])
            
            # Early stopping based on F1 score
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
                print(f"  Best Val F1: {best_val_f1:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_metrics = self.evaluate(
            text_embeddings, demographic_features, diversity_signals,
            labels, view_names, test_mask
        )
        
        if verbose:
            print(f"\nFinal Test Results:")
            print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
            print(f"  Precision: {final_metrics['precision']:.4f}")
            print(f"  Recall: {final_metrics['recall']:.4f}")
            print(f"  F1: {final_metrics['f1']:.4f}")
            
            # Print average gate weights to show view importance
            avg_gate_weights = final_metrics["gate_weights"].mean(axis=0)
            print(f"  Average Gate Weights: {dict(zip(view_names, avg_gate_weights))}")
        
        return final_metrics


def load_data(data_path: str) -> Dict:
    """Load GenFEND data from file."""
    print(f"Loading data from {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    print(f"Loaded data for {data['metadata']['dataset_name']} ({data['metadata']['k_shot']}-shot)")
    return data


def plot_training_curves(history: Dict, output_path: str):
    """Plot and save training curves."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss curves
    ax1.plot(epochs, history["train_loss"], 'b-', label="Train Loss")
    ax1.plot(epochs, history["val_loss"], 'r-', label="Val Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, history["train_acc"], 'b-', label="Train Acc")
    ax2.plot(epochs, history["val_acc"], 'r-', label="Val Acc")
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    # F1 curve
    ax3.plot(epochs, history["val_f1"], 'g-', label="Val F1")
    ax3.set_title("F1 Score")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1 Score")
    ax3.legend()
    ax3.grid(True)
    
    # Combined metrics
    ax4.plot(epochs, history["val_acc"], 'r-', label="Accuracy")
    ax4.plot(epochs, history["val_f1"], 'g-', label="F1 Score")
    ax4.set_title("Validation Metrics")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Score")
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_path}")


def main():
    """Main training function."""
    parser = ArgumentParser(description="Train GenFEND model")
    parser.add_argument("--data_path", required=True, help="Path to prepared GenFEND data")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--output_dir", default="results_genfend", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    data = load_data(args.data_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    text_embed_dim = data["text_embeddings"].shape[1]
    demographic_embed_dim = list(data["demographic_features"].values())[0].shape[2]
    num_views = len(data["metadata"]["view_names"])
    
    model = GenFENDModel(
        text_embed_dim=text_embed_dim,
        demographic_embed_dim=demographic_embed_dim,
        num_views=num_views,
        num_demographic_profiles=data["metadata"]["num_demographic_profiles"],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = GenFENDTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    final_metrics = trainer.train(
        data=data,
        epochs=args.epochs,
        patience=args.patience,
        verbose=True
    )
    training_time = time.time() - start_time
    
    # Save results
    dataset_name = data["metadata"]["dataset_name"]
    k_shot = data["metadata"]["k_shot"]
    embedding_type = data["metadata"]["embedding_type"]
    
    # Results filename
    results_filename = f"genfend_{dataset_name}_k{k_shot}_{embedding_type}_results.json"
    results_path = os.path.join(args.output_dir, results_filename)
    
    # Prepare results
    results = {
        "final_metrics": final_metrics,
        "training_time": training_time,
        "model_config": {
            "text_embed_dim": text_embed_dim,
            "demographic_embed_dim": demographic_embed_dim,
            "num_views": num_views,
            "num_demographic_profiles": data["metadata"]["num_demographic_profiles"],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout
        },
        "training_config": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience
        },
        "data_config": data["metadata"]
    }
    
    # Remove numpy arrays from final_metrics for JSON serialization
    json_metrics = {k: v for k, v in final_metrics.items() if k != "gate_weights"}
    results["final_metrics"] = json_metrics
    
    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Plot training curves
    curves_filename = f"genfend_{dataset_name}_k{k_shot}_{embedding_type}_curves.png"
    curves_path = os.path.join(args.output_dir, curves_filename)
    plot_training_curves(trainer.training_history, curves_path)
    
    # Save model
    model_filename = f"genfend_{dataset_name}_k{k_shot}_{embedding_type}_model.pt"
    model_path = os.path.join(args.output_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final F1 Score: {final_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()