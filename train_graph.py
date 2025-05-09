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
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data

# Constants
DEFAULT_MODEL = "GAT"
DEFAULT_EPOCHS = 300
DEFAULT_LR = 0.0001  # Adjusted learning rate
DEFAULT_WEIGHT_DECAY = 1e-4 # Adjusted weight decay
DEFAULT_HIDDEN_CHANNELS = 128
DEFAULT_DROPOUT = 0.5
DEFAULT_GAT_HEADS = 8
DEFAULT_PATIENCE = 30 # Increased patience
DEFAULT_SEED = 42
RESULTS_DIR = "results"
PLOTS_DIR = "plots"


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Model Definitions ---

class GCN(nn.Module):
    """Graph Convolutional Network (GCN)"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        # GCNConv uses edge_attr as edge_weight. Ensure it's 1D.
        edge_weight = edge_attr.squeeze() if edge_attr is not None and edge_attr.dim() > 1 else edge_attr

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


class GAT(nn.Module):
    """
    Graph Attention Network (GAT) with architecture 768 -> 128 -> 2
    and a residual connection around the first GAT layer.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float, heads: int, edge_dim: int = None):
        """
        Args:
            in_channels (int): Input feature dimension (e.g., 768).
            hidden_channels (int): Hidden layer dimension (e.g., 128).
            out_channels (int): Output dimension (e.g., 2 for binary classification).
            dropout (float): Dropout rate.
            heads (int): Number of attention heads in the first layer.
            edge_dim (int, optional): Dimension of edge features. Defaults to None.
        """
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError(f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})")

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout_rate = dropout

        # --- Layers ---
        # First GAT layer (transforms input to hidden dimension)
        self.conv1 = GATConv(
            in_channels,
            hidden_channels // heads, # Output channels per head
            heads=heads,              # Number of heads
            concat=True,              # Concatenate head outputs -> hidden_channels total dimension
            dropout=dropout,          # Apply dropout within GATConv (on attention weights)
            edge_dim=edge_dim,
            add_self_loops=True
        )

        # Linear projection for the residual connection (to match conv1 output dimension)
        self.lin_proj = nn.Linear(in_channels, hidden_channels)

        # Activation Function
        self.elu = nn.ELU()

        # Second GAT layer (transforms hidden to output dimension)
        self.conv2 = GATConv(
            hidden_channels,          # Input channels = hidden_channels (output of conv1)
            out_channels,             # Output channels for final classification
            heads=1,                  # Typically 1 head for the output layer
            concat=False,             # Average outputs of heads
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the GAT model.

        Args:
            x (torch.Tensor): Node feature matrix (shape: [num_nodes, in_channels]).
            edge_index (torch.Tensor): Graph connectivity (shape: [2, num_edges]).
            edge_attr (torch.Tensor, optional): Edge feature matrix. Defaults to None.

        Returns:
            torch.Tensor: Output node embeddings (shape: [num_nodes, out_channels]).
        """
        # 0. Store identity for residual connection (optional, apply dropout first?)
        identity = x

        # 1. Apply dropout to input features (common practice)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # 2. First GAT layer
        x_conv1 = self.conv1(x, edge_index, edge_attr=edge_attr) # Shape: [num_nodes, hidden_channels]

        # 3. Prepare residual connection
        # Project original input to match the dimension of conv1's output
        identity_proj = self.lin_proj(identity) # Shape: [num_nodes, hidden_channels]

        # 4. Add residual connection before activation
        x = x_conv1 + identity_proj

        # 5. Apply activation function
        x = self.elu(x)

        # 6. Apply dropout after activation (common practice)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # 7. Second GAT layer (output layer)
        x = self.conv2(x, edge_index, edge_attr=edge_attr) # Shape: [num_nodes, out_channels]

        # 8. Return raw logits (CrossEntropyLoss applies LogSoftmax)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        # Use aggregator type 'mean' as a common default
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        # Standard SAGEConv doesn't use edge_attr, but pass it for consistency if needed later
        # If using a SAGE variant that uses edge features, modify here.
        x = self.conv1(x, edge_index) # edge_attr not used by default SAGEConv
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index) # edge_attr not used by default SAGEConv
        return x


# --- Utility Functions ---

def load_graph(path: str, device: torch.device) -> Data:
    """Load graph data and move it to the specified device."""
    try:
        # Try loading with weights_only=False first, as it's needed for Data objects
        graph = torch.load(path, map_location=torch.device('cpu')) # Load to CPU first
        print(f"Graph loaded from {path}")
    except Exception as e:
        print(f"Error loading graph: {e}")
        # Attempt with weights_only=True as a fallback, though unlikely to work for Data
        try:
            graph = torch.load(path, weights_only=True, map_location=torch.device('cpu'))
            print("Warning: Loaded graph with weights_only=True. May lack structure.")
        except Exception as e2:
            print(f"Failed loading with weights_only=True as well: {e2}")
            raise ValueError(f"Could not load graph data from {path}") from e

    # Validate essential attributes
    required_attrs = ['x', 'y', 'train_labeled_mask', 'train_unlabeled_mask', 'test_mask', 'edge_index']
    for attr in required_attrs:
        if not hasattr(graph, attr):
            raise AttributeError(f"Loaded graph is missing required attribute: {attr}")

    # Move graph data to the target device
    graph = graph.to(device)
    print(f"Graph data moved to {device}")

    # Handle potentially missing edge_attr gracefully
    if not hasattr(graph, 'edge_attr') or graph.edge_attr is None:
        print("Warning: Graph does not have 'edge_attr'. Models might not use edge features.")
        graph.edge_attr = None # Ensure it's None if missing
    elif graph.edge_attr is not None:
         # Ensure edge_attr is FloatTensor
        graph.edge_attr = graph.edge_attr.float()
        print(f"Graph has edge_attr with shape: {graph.edge_attr.shape}") # Log shape

    # Ensure masks are boolean type
    graph.train_labeled_mask = graph.train_labeled_mask.bool()
    graph.train_unlabeled_mask = graph.train_unlabeled_mask.bool()
    graph.test_mask = graph.test_mask.bool()

    # Check if train_labeled_mask has any True values for training supervision
    if graph.train_labeled_mask.sum() == 0:
         raise ValueError("Cannot train: No nodes available in train_labeled_mask for supervision.")


    return graph

def get_model(model_name: str, graph_data: Data, args: ArgumentParser) -> nn.Module:
    """Initialize the GNN model based on name and arguments."""
    edge_dim = None
    if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
        # GAT expects edge_dim if edge_attr is multi-dimensional
        if graph_data.edge_attr.dim() > 1 and graph_data.edge_attr.shape[1] > 1:
             edge_dim = graph_data.edge_attr.shape[1]
        # If edge_attr is [N, 1], GAT doesn't need edge_dim, GCN will treat it as weight
        elif graph_data.edge_attr.dim() == 1 or graph_data.edge_attr.shape[1] == 1:
             edge_dim = None # Treat as scalar weight
        print(f"Determined edge_dim for GAT: {edge_dim}")


    num_classes = 2 # Label - 0: real, 1: fake

    if model_name == "GCN":
        model = GCN(
            in_channels=graph_data.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            dropout=args.dropout_rate,
        )
    elif model_name == "GAT":
        model = GAT(
            in_channels=graph_data.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            dropout=args.dropout_rate,
            heads=args.gat_heads,
            edge_dim=edge_dim # Pass edge feature dimension to GAT
        )
    elif model_name == "GraphSAGE":
        model = GraphSAGE(
            in_channels=graph_data.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            dropout=args.dropout_rate,
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return model

def train_epoch(model: nn.Module, data: Data, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> tuple[float, float]:
    """Perform a single training epoch."""
    model.train()
    optimizer.zero_grad()
    # *** Pass edge_attr to the model ***
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out[data.train_labeled_mask], data.y[data.train_labeled_mask])
    loss.backward()
    optimizer.step()

    # Calculate training accuracy on labeled nodes
    pred = out[data.train_labeled_mask].argmax(dim=1)
    correct = (pred == data.y[data.train_labeled_mask]).sum().item()
    acc = correct / data.train_labeled_mask.sum().item()

    return loss.item(), acc

@torch.no_grad()
def evaluate(model: nn.Module, data: Data, mask: torch.Tensor, criterion: nn.Module) -> tuple[float, float, float]:
    """Evaluate the model on a given data mask."""
    model.eval()
    # *** Pass edge_attr to the model ***
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out[mask], data.y[mask])

    pred = out[mask].argmax(dim=1)
    correct = (pred == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()

    # Calculate F1 score
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    # Use macro average, handle zero division
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return loss.item(), acc, f1

def train(model: nn.Module, data: Data, optimizer: torch.optim.Optimizer, criterion: nn.Module, args: ArgumentParser, output_dir: str, model_name: str) -> dict:
    """Train the model with validation and early stopping."""
    print("\n--- Starting Training ---")
    start_time = time.time()

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    val_f1s = []
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = -1

    model_save_path = os.path.join(output_dir, f"{model_name}_best.pt")

    for epoch in range(args.n_epochs):
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion)
        val_loss, val_acc, val_f1 = evaluate(model, data, data.test_mask, criterion) # Use test_mask for validation


        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Epoch: {epoch+1:03d}/{args.n_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss # Also save loss at best F1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved (F1: {best_val_f1:.4f})")
            patience_counter = 0
        # If F1 didn't improve, check if loss improved (secondary criterion)
        elif val_loss < best_val_loss and patience_counter > 0: # Only consider loss if F1 hasn't improved recently
             best_val_loss = val_loss
             # Optionally save model based on loss improvement too, but primary is F1
             # torch.save(model.state_dict(), model_save_path + "_best_loss")
             # print(f"  -> Val loss improved (Loss: {best_val_loss:.4f})")
             patience_counter = 0 # Reset patience if loss improves significantly
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    train_time = time.time() - start_time
    print(f"--- Training Finished in {train_time:.2f} seconds ---")
    # Handle case where training didn't run any epochs or stopped early
    if best_epoch != -1:
        print(f"Best model from epoch {best_epoch} saved to {model_save_path}")
    else:
        print("No best model saved (training might have stopped early or validation F1 did not improve).")


    history = {
        "train_loss": train_losses, "train_acc": train_accs,
        "val_loss": val_losses, "val_acc": val_accs, "val_f1": val_f1s,
        "best_epoch": best_epoch, "train_time": train_time,
        "best_val_f1": best_val_f1
    }
    return history

def final_evaluation(model: nn.Module, data: Data, model_path: str) -> dict:
    """Load the best model and perform final evaluation on the test set."""
    print("\n--- Final Evaluation on Test Set ---")
    try:
        model.load_state_dict(torch.load(model_path, map_location=data.x.device))
        print(f"Loaded best model weights from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load best model weights from {model_path}. Evaluating with last state. Error: {e}")

    model.eval()
    with torch.no_grad():
        # *** Pass edge_attr to the model ***
        out = model(data.x, data.edge_index, data.edge_attr)
        pred = out[data.test_mask].argmax(dim=1)
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred.cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist() # Convert to list for JSON serialization
    }

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("--- Evaluation Finished ---")
    return metrics

def save_results(history: dict, final_metrics: dict, args: ArgumentParser, output_dir: str, model_name: str) -> None:
    """Save training history, final metrics, and plots."""
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, PLOTS_DIR)
    os.makedirs(plot_dir, exist_ok=True)

    # --- Save Metrics ---
    results_data = {
        "args": vars(args),
        "model_name": model_name,
        "training_history": {
            # Keep only last value for summary, full history can be large
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_train_acc": history["train_acc"][-1] if history["train_acc"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
            "final_val_f1": history["val_f1"][-1] if history["val_f1"] else None,
            "best_val_f1": history["best_val_f1"],
            "best_epoch": history["best_epoch"],
            "total_epochs": len(history["train_loss"]),
            "training_time_seconds": history["train_time"],
        },
        "final_test_metrics": final_metrics
    }
    results_path = os.path.join(output_dir, f"metrics.json")
    try:
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Error saving results JSON: {e}")


    # --- Save Plots ---
    plot_path = os.path.join(plot_dir, f"{model_name}_training_curves.png")
    try:
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"Training Curves - {model_name}")

        # Accuracy
        axes[0].plot(epochs, history['train_acc'], label='Train Accuracy', marker='.')
        axes[0].plot(epochs, history['val_acc'], label='Validation Accuracy', marker='.')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy vs. Epochs')
        axes[0].legend()
        axes[0].grid(True)

        # Loss
        axes[1].plot(epochs, history['train_loss'], label='Train Loss', marker='.')
        axes[1].plot(epochs, history['val_loss'], label='Validation Loss', marker='.')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss vs. Epochs')
        axes[1].legend()
        axes[1].grid(True)

        # F1 Score
        axes[2].plot(epochs, history['val_f1'], label='Validation F1 Score', marker='.', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Validation F1 vs. Epochs')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.savefig(plot_path)
        plt.close()
        print(f"Training plots saved to {plot_path}")
    except Exception as e:
         print(f"Error saving plots: {e}")


def parse_arguments() -> ArgumentParser:
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Train Graph Neural Networks for Fake News Detection")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to the preprocessed graph data (.pt file)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=["GCN", "GAT", "GraphSAGE"], help="GNN model type")
    parser.add_argument("--dropout_rate", type=float, default=DEFAULT_DROPOUT, help="Dropout rate")
    parser.add_argument("--hidden_channels", type=int, default=DEFAULT_HIDDEN_CHANNELS, help="Number of hidden units in GNN layers")
    parser.add_argument("--gat_heads", type=int, default=DEFAULT_GAT_HEADS, help="Number of attention heads for GAT model")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay (L2 regularization)")
    parser.add_argument("--n_epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR, help="Base directory to save results and plots")

    return parser.parse_args()

def main() -> None:
    """Main execution pipeline."""
    args = parse_arguments()

    # --- Setup ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        gc.collect()

    # --- Load Data ---
    graph_data = load_graph(args.graph_path, device)

    # --- Prepare Output Directory ---
    # Extract dataset name and scenario from graph path for structured output
    try:
        parts = args.graph_path.split(os.sep)   # e.g., "graphs/politifact/8_shot_roberta_knn_5.pt"
        scenario = os.path.splitext(parts[-1])[0] # e.g., 8_shot_roberta_knn_5
        dataset_name = parts[-2] # e.g., politifact
    except IndexError:
        print("Warning: Could not parse dataset/scenario from graph path. Using generic names.")
        scenario = "unknown_scenario"
        dataset_name = "unknown_dataset"

    model_name = f"{args.model}_{dataset_name}_{scenario}" # More descriptive model/run name
    output_dir = os.path.join(args.output_dir, args.model, dataset_name, scenario)
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Configuration ---")
    print(f"Model:           {args.model}")
    print(f"Dataset:         {dataset_name}")
    print(f"Scenario:        {scenario}")
    print(f"Graph Path:      {args.graph_path}")
    print(f"Output Dir:      {output_dir}")
    # Print args cleanly
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k:<16}: {v}")
    print("--- Graph Info ---")
    print(f"Nodes:           {graph_data.num_nodes}")
    print(f"Edges:           {graph_data.num_edges}")
    print(f"Features:        {graph_data.num_features}")
    print(f"Classes:         {int(graph_data.y.max()) + 1}")
    print(f"Train nodes(total):   {graph_data.train_labeled_mask.sum().item() + graph_data.train_unlabeled_mask.sum().item()}")
    print(f"Train Labeled nodes:   {graph_data.train_labeled_mask.sum().item()}")
    print(f"Train Unlabeled nodes: {graph_data.train_unlabeled_mask.sum().item()}")
    print(f"Test nodes:              {graph_data.test_mask.sum().item()}")
    if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
        print(f"Edge Attr Dim:   {graph_data.edge_attr.shape[1] if graph_data.edge_attr.dim() > 1 else 1}")
    else:
        print(f"Edge Attr Dim:   None")


    # --- Initialize Model, Optimizer, Criterion ---
    model = get_model(args.model, graph_data, args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Use standard CrossEntropyLoss (no class balancing by default)
    criterion = nn.CrossEntropyLoss()
    print("\n--- Model Architecture ---")
    print(model)
    print("-------------------------")


    # --- Train Model ---
    training_history = train(model, graph_data, optimizer, criterion, args, output_dir, model_name)

    # --- Final Evaluation ---
    model_path = os.path.join(output_dir, f"{model_name}_best.pt")
    final_metrics = final_evaluation(model, graph_data, model_path)

    # --- Save Results ---
    save_results(training_history, final_metrics, args, output_dir, model_name)

    print("\n--- Pipeline Complete ---")
    print(f"Results, plots, and best model saved in: {output_dir}")
    print("-------------------------\n")


if __name__ == "__main__":
    main()