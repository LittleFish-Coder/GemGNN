"""
Training script for LESS4FD architecture.

This script implements the two-phase training pipeline:
1. Pre-training with self-supervised tasks
2. Fine-tuning with few-shot classification
"""

import os
import gc
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.data import HeteroData
import logging

# Import LESS4FD modules
from .build_less4fd_graph import LESS4FDGraphBuilder
from .models.less4fd_model import LESS4FDModel
from .config.less4fd_config import LESS4FD_CONFIG, TRAINING_CONFIG, FEWSHOT_CONFIG
from .utils.sampling import LESS4FDSampler

# Import existing utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "LESS4FD"
DEFAULT_DATASET = "politifact"
DEFAULT_K_SHOT = 8
DEFAULT_EMBEDDING_TYPE = "deberta"
DEFAULT_SEED = 42
RESULTS_DIR = "results_less4fd"
PLOTS_DIR = "plots_less4fd"

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LESS4FDTrainer:
    """
    Trainer for the LESS4FD model with two-phase training.
    """
    
    def __init__(
        self,
        dataset_name: str,
        k_shot: int,
        embedding_type: str = "deberta",
        model_config: dict = None,
        training_config: dict = None,
        device: str = None,
        seed: int = 42
    ):
        """
        Initialize the LESS4FD trainer.
        
        Args:
            dataset_name: Name of the dataset
            k_shot: Number of shots for few-shot learning
            embedding_type: Type of text embeddings
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
            device: Device for training
            seed: Random seed
        """
        self.dataset_name = dataset_name
        self.k_shot = k_shot
        self.embedding_type = embedding_type
        self.seed = seed
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Configuration
        self.model_config = model_config or LESS4FD_CONFIG.copy()
        self.training_config = training_config or TRAINING_CONFIG.copy()
        
        # Set seed
        set_seed(seed)
        
        # Initialize directories
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.graph_data = None
        self.train_history = {"pretrain": [], "finetune": []}
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
    def build_graph(self) -> HeteroData:
        """Build the LESS4FD heterogeneous graph."""
        logger.info("Building LESS4FD heterogeneous graph...")
        
        # Initialize graph builder
        graph_builder = LESS4FDGraphBuilder(
            dataset_name=self.dataset_name,
            k_shot=self.k_shot,
            embedding_type=self.embedding_type,
            entity_model=self.model_config["entity_model"],
            max_entities_per_news=self.model_config["max_entities_per_news"],
            entity_similarity_threshold=self.model_config["entity_similarity_threshold"],
            entity_knn=self.model_config["entity_knn"],
            use_entity_types=self.model_config["use_entity_types"],
            seed=self.seed
        )
        
        # Try to load existing graph
        graph_data = graph_builder.load_graph()
        
        if graph_data is None:
            # Build new graph
            graph_data = graph_builder.build_graph()
            if graph_data is not None:
                graph_builder.save_graph(graph_data)
        
        if graph_data is None:
            raise ValueError("Failed to build or load graph data")
        
        # Analyze graph
        analysis = graph_builder.analyze_entity_graph(graph_data)
        logger.info(f"Graph analysis: {analysis}")
        
        return graph_data
    
    def initialize_model(self, graph_data: HeteroData) -> LESS4FDModel:
        """Initialize the LESS4FD model."""
        logger.info("Initializing LESS4FD model...")
        
        # Get number of entities from graph metadata
        num_entities = graph_data.metadata_dict.get("num_entities", 1000)
        
        # Create model
        model = LESS4FDModel(
            data=graph_data,
            hidden_channels=self.model_config["hidden_channels"],
            num_entities=num_entities,
            num_classes=2,  # fake/real
            num_gnn_layers=self.model_config["num_gnn_layers"],
            num_heads=self.model_config["num_attention_heads"],
            dropout=self.model_config["dropout"],
            entity_embedding_dim=self.model_config["entity_embedding_dim"],
            use_entity_types=self.model_config["use_entity_types"],
            num_entity_types=len(self.model_config["entity_types"]),
            contrastive_temperature=self.model_config["contrastive_temperature"],
            device=self.device
        )
        
        model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        return model
    
    def setup_training(self, model: LESS4FDModel, phase: str = "pretrain"):
        """Setup optimizer and scheduler for training phase."""
        if phase == "pretrain":
            lr = self.training_config["learning_rate"]
            weight_decay = self.training_config["weight_decay"]
        else:  # finetune
            lr = self.training_config["learning_rate"] * 0.1  # Lower LR for finetuning
            weight_decay = self.training_config["weight_decay"] * 0.5
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        if phase == "pretrain":
            num_epochs = self.training_config["pretrain_epochs"]
        else:
            num_epochs = self.training_config["finetune_epochs"]
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=lr * 0.01
        )
        
        logger.info(f"Setup {phase} training with LR={lr}, weight_decay={weight_decay}")
    
    def pretrain_epoch(self, model: LESS4FDModel, graph_data: HeteroData) -> Dict:
        """Run one pretraining epoch."""
        model.train()
        
        # Get node features and edge indices
        x_dict = {node_type: graph_data[node_type].x for node_type in graph_data.node_types}
        edge_index_dict = {
            edge_type: graph_data[edge_type].edge_index 
            for edge_type in graph_data.edge_types
        }
        
        # Get entity types and news labels
        entity_types = None
        if 'entity' in graph_data.node_types and hasattr(graph_data['entity'], 'entity_type'):
            entity_types = graph_data['entity'].entity_type
        
        news_labels = None
        if 'news' in graph_data.node_types and hasattr(graph_data['news'], 'y'):
            news_labels = graph_data['news'].y
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = model.pretrain_step(x_dict, edge_index_dict, entity_types, news_labels)
        
        # Backward pass
        loss = outputs["losses"]["pretrain_total"]
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config["gradient_clip"])
        
        self.optimizer.step()
        
        # Collect metrics
        metrics = {
            "total_loss": loss.item(),
            "pretext_loss": outputs["losses"].get("pretext", torch.tensor(0.0)).item(),
            "contrastive_loss": outputs["losses"].get("contrastive", torch.tensor(0.0)).item()
        }
        
        # Add individual pretext task losses
        for key, value in outputs["losses"].items():
            if key.startswith("pretext_"):
                metrics[key] = value.item()
        
        return metrics
    
    def finetune_epoch(self, model: LESS4FDModel, graph_data: HeteroData) -> Dict:
        """Run one finetuning epoch."""
        model.train()
        
        # Get node features and edge indices
        x_dict = {node_type: graph_data[node_type].x for node_type in graph_data.node_types}
        edge_index_dict = {
            edge_type: graph_data[edge_type].edge_index 
            for edge_type in graph_data.edge_types
        }
        
        # Get entity types and news labels
        entity_types = None
        if 'entity' in graph_data.node_types and hasattr(graph_data['entity'], 'entity_type'):
            entity_types = graph_data['entity'].entity_type
        
        news_labels = graph_data['news'].y
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = model.finetune_step(x_dict, edge_index_dict, news_labels, entity_types)
        
        # Backward pass
        loss = outputs["losses"].get("finetune_total", outputs["losses"]["classification"])
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config["gradient_clip"])
        
        self.optimizer.step()
        
        # Collect metrics
        metrics = {
            "total_loss": loss.item(),
            "classification_loss": outputs["losses"]["classification"].item(),
        }
        
        if "light_contrastive" in outputs["losses"]:
            metrics["light_contrastive_loss"] = outputs["losses"]["light_contrastive"].item()
        
        return metrics
    
    def evaluate(self, model: LESS4FDModel, graph_data: HeteroData, mask_name: str = "test_mask") -> Dict:
        """Evaluate the model."""
        model.eval()
        
        with torch.no_grad():
            # Get node features and edge indices
            x_dict = {node_type: graph_data[node_type].x for node_type in graph_data.node_types}
            edge_index_dict = {
                edge_type: graph_data[edge_type].edge_index 
                for edge_type in graph_data.edge_types
            }
            
            # Get entity types
            entity_types = None
            if 'entity' in graph_data.node_types and hasattr(graph_data['entity'], 'entity_type'):
                entity_types = graph_data['entity'].entity_type
            
            # Get predictions
            logits = model.predict(x_dict, edge_index_dict, entity_types)
            
            # Get evaluation mask and labels
            if hasattr(graph_data['news'], mask_name):
                mask = getattr(graph_data['news'], mask_name)
                labels = graph_data['news'].y[mask]
                pred_logits = logits[mask]
            else:
                # Use all nodes if mask doesn't exist
                labels = graph_data['news'].y
                pred_logits = logits
            
            # Get predictions
            predictions = torch.argmax(pred_logits, dim=1)
            
            # Compute metrics
            labels_np = labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            
            metrics = {
                "accuracy": accuracy_score(labels_np, predictions_np),
                "precision": precision_score(labels_np, predictions_np, average='weighted', zero_division=0),
                "recall": recall_score(labels_np, predictions_np, average='weighted', zero_division=0),
                "f1": f1_score(labels_np, predictions_np, average='weighted', zero_division=0)
            }
            
            # Add per-class metrics
            try:
                precision_per_class = precision_score(labels_np, predictions_np, average=None, zero_division=0)
                recall_per_class = recall_score(labels_np, predictions_np, average=None, zero_division=0)
                f1_per_class = f1_score(labels_np, predictions_np, average=None, zero_division=0)
                
                for i, (p, r, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
                    metrics[f"precision_class_{i}"] = p
                    metrics[f"recall_class_{i}"] = r
                    metrics[f"f1_class_{i}"] = f1
            except:
                pass
        
        return metrics
    
    def pretrain(self, model: LESS4FDModel, graph_data: HeteroData):
        """Run pretraining phase."""
        logger.info("Starting pretraining phase...")
        
        self.setup_training(model, "pretrain")
        num_epochs = self.training_config["pretrain_epochs"]
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.pretrain_epoch(model, graph_data)
            
            # Validation (if validation mask exists)
            val_metrics = {}
            if hasattr(graph_data['news'], 'val_mask'):
                val_metrics = self.evaluate(model, graph_data, "val_mask")
            
            # Update scheduler
            self.scheduler.step()
            
            # Record metrics
            epoch_data = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.optimizer.param_groups[0]['lr'],
                "time": time.time() - start_time
            }
            self.train_history["pretrain"].append(epoch_data)
            
            # Logging
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(f"Pretrain Epoch {epoch:3d}/{num_epochs}: "
                           f"Loss={train_metrics['total_loss']:.4f}, "
                           f"LR={epoch_data['lr']:.6f}, "
                           f"Time={epoch_data['time']:.2f}s")
                
                if val_metrics:
                    logger.info(f"  Val Acc={val_metrics.get('accuracy', 0.0):.4f}, "
                               f"F1={val_metrics.get('f1', 0.0):.4f}")
        
        logger.info("Pretraining phase completed")
    
    def finetune(self, model: LESS4FDModel, graph_data: HeteroData):
        """Run finetuning phase."""
        logger.info("Starting finetuning phase...")
        
        self.setup_training(model, "finetune")
        num_epochs = self.training_config["finetune_epochs"]
        patience = self.training_config["patience"]
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.finetune_epoch(model, graph_data)
            
            # Validation
            val_metrics = {}
            if hasattr(graph_data['news'], 'val_mask'):
                val_metrics = self.evaluate(model, graph_data, "val_mask")
            else:
                # Use test set for validation if no val set
                val_metrics = self.evaluate(model, graph_data, "test_mask")
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get('accuracy', 0.0))
            else:
                self.scheduler.step()
            
            # Early stopping and best model tracking
            val_acc = val_metrics.get('accuracy', 0.0)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Record metrics
            epoch_data = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.optimizer.param_groups[0]['lr'],
                "time": time.time() - start_time
            }
            self.train_history["finetune"].append(epoch_data)
            
            # Logging
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                logger.info(f"Finetune Epoch {epoch:3d}/{num_epochs}: "
                           f"Loss={train_metrics['total_loss']:.4f}, "
                           f"Val Acc={val_acc:.4f}, "
                           f"Best={self.best_val_acc:.4f}, "
                           f"LR={epoch_data['lr']:.6f}")
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break
        
        # Load best model
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with validation accuracy: {self.best_val_acc:.4f}")
        
        logger.info("Finetuning phase completed")
    
    def train(self) -> Dict:
        """Run the complete training pipeline."""
        logger.info("Starting LESS4FD training pipeline...")
        
        # Build graph
        self.graph_data = self.build_graph()
        
        # Initialize model
        self.model = self.initialize_model(self.graph_data)
        
        # Run pretraining
        self.pretrain(self.model, self.graph_data)
        
        # Run finetuning
        self.finetune(self.model, self.graph_data)
        
        # Final evaluation
        test_metrics = self.evaluate(self.model, self.graph_data, "test_mask")
        logger.info(f"Final test results: {test_metrics}")
        
        # Save results
        results = {
            "dataset": self.dataset_name,
            "k_shot": self.k_shot,
            "embedding_type": self.embedding_type,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "train_history": self.train_history,
            "test_metrics": test_metrics,
            "best_val_acc": self.best_val_acc
        }
        
        self.save_results(results)
        self.plot_training_history()
        
        return results
    
    def save_results(self, results: Dict):
        """Save training results."""
        filename = f"less4fd_{self.dataset_name}_k{self.k_shot}_{self.embedding_type}_seed{self.seed}.json"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        # Convert tensors and other non-serializable objects to serializable format
        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.train_history["pretrain"] and not self.train_history["finetune"]:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Pretraining losses
        if self.train_history["pretrain"]:
            pretrain_epochs = [x["epoch"] for x in self.train_history["pretrain"]]
            pretrain_losses = [x["train"]["total_loss"] for x in self.train_history["pretrain"]]
            
            axes[0, 0].plot(pretrain_epochs, pretrain_losses, label="Total Loss")
            axes[0, 0].set_title("Pretraining Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Finetuning losses and accuracy
        if self.train_history["finetune"]:
            finetune_epochs = [x["epoch"] for x in self.train_history["finetune"]]
            finetune_losses = [x["train"]["total_loss"] for x in self.train_history["finetune"]]
            val_accs = [x["val"].get("accuracy", 0) for x in self.train_history["finetune"]]
            
            axes[0, 1].plot(finetune_epochs, finetune_losses, label="Training Loss")
            axes[0, 1].set_title("Finetuning Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            axes[1, 0].plot(finetune_epochs, val_accs, label="Validation Accuracy")
            axes[1, 0].set_title("Validation Accuracy")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Accuracy")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate
        if self.train_history["finetune"]:
            lrs = [x["lr"] for x in self.train_history["finetune"]]
            axes[1, 1].plot(finetune_epochs, lrs, label="Learning Rate")
            axes[1, 1].set_title("Learning Rate Schedule")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        filename = f"less4fd_training_{self.dataset_name}_k{self.k_shot}_{self.embedding_type}_seed{self.seed}.png"
        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {filepath}")


def main():
    """Main training function."""
    parser = ArgumentParser(description="Train LESS4FD model for few-shot fake news detection")
    
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=["politifact", "gossipcop"],
                        help="Dataset name")
    parser.add_argument("--k_shot", type=int, default=DEFAULT_K_SHOT,
                        help="Number of shots for few-shot learning")
    parser.add_argument("--embedding_type", type=str, default=DEFAULT_EMBEDDING_TYPE,
                        choices=["bert", "deberta", "roberta", "distilbert"],
                        help="Type of text embeddings")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for training (cuda/cpu)")
    
    # Model configuration
    parser.add_argument("--hidden_channels", type=int, default=LESS4FD_CONFIG["hidden_channels"],
                        help="Hidden dimension size")
    parser.add_argument("--num_gnn_layers", type=int, default=LESS4FD_CONFIG["num_gnn_layers"],
                        help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=LESS4FD_CONFIG["dropout"],
                        help="Dropout rate")
    
    # Training configuration
    parser.add_argument("--pretrain_epochs", type=int, default=TRAINING_CONFIG["pretrain_epochs"],
                        help="Number of pretraining epochs")
    parser.add_argument("--finetune_epochs", type=int, default=TRAINING_CONFIG["finetune_epochs"],
                        help="Number of finetuning epochs")
    parser.add_argument("--learning_rate", type=float, default=TRAINING_CONFIG["learning_rate"],
                        help="Learning rate")
    
    args = parser.parse_args()
    
    # Update configurations with command line arguments
    model_config = LESS4FD_CONFIG.copy()
    model_config.update({
        "hidden_channels": args.hidden_channels,
        "num_gnn_layers": args.num_gnn_layers,
        "dropout": args.dropout
    })
    
    training_config = TRAINING_CONFIG.copy()
    training_config.update({
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "learning_rate": args.learning_rate
    })
    
    # Initialize trainer
    trainer = LESS4FDTrainer(
        dataset_name=args.dataset,
        k_shot=args.k_shot,
        embedding_type=args.embedding_type,
        model_config=model_config,
        training_config=training_config,
        device=args.device,
        seed=args.seed
    )
    
    # Run training
    results = trainer.train()
    
    logger.info("Training completed successfully!")
    logger.info(f"Final test accuracy: {results['test_metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()