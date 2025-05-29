# filepath: /home/littlefish/fake-news-detection/train_hetero_graph.py
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
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, HANConv, Linear, SAGEConv, GATv2Conv, to_hetero, RGCNConv

# Constants
DEFAULT_MODEL = "HGT"   # HGT, HAN, SAGE, GATv2
DEFAULT_LOSS_FN = "ce" # ce, focal
DEFAULT_EPOCHS = 300
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HIDDEN_CHANNELS = 128
DEFAULT_DROPOUT = 0.0
DEFAULT_HGT_HAN_HEADS = 4 # Number of attention heads for HGT/HAN
DEFAULT_HGT_LAYERS = 1
DEFAULT_PATIENCE = 30
DEFAULT_SEED = 42
DEFAULT_TARGET_NODE_TYPE = "news" # Target node type for classification
RESULTS_DIR = "results_hetero" # Separate results for hetero models
PLOTS_DIR = "plots_hetero"


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Loss Functions ---
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

# --- Model Definitions ---
class HGTModel(nn.Module):
    """Heterogeneous Graph Transformer (HGT) Model with Residual and LayerNorm"""
    def __init__(self, data: HeteroData, hidden_channels: int, out_channels: int,
                 num_layers: int, heads: int, target_node_type: str, dropout_rate: float):
        super().__init__()
        self.target_node_type = target_node_type
        self.dropout_rate = dropout_rate

        self.lins = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        for node_type in data.node_types:
            self.lins[node_type] = Linear(data[node_type].num_features, hidden_channels)
            self.norms[node_type] = nn.LayerNorm(hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), heads)
            self.convs.append(conv)

        self.out_lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Initial transformation
        for node_type, x in x_dict.items():
            x = self.lins[node_type](x)
            x = self.norms[node_type](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x_dict[node_type] = x

        # HGT convolutions with residual and LayerNorm
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x = x_dict[node_type] + x_dict_new[node_type]
                x = self.norms[node_type](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
                x_dict[node_type] = x

        return self.out_lin(x_dict[self.target_node_type])


class HANModel(nn.Module):
    """Heterogeneous Attentional Network (HAN) Model"""
    def __init__(self, data: HeteroData, hidden_channels: int, out_channels: int,
                 heads: int, target_node_type: str, dropout_rate: float):
        super().__init__()
        self.target_node_type = target_node_type
        self.dropout_rate = dropout_rate

        # HANConv can infer in_channels if set to -1
        self.conv1 = HANConv(in_channels=-1, out_channels=hidden_channels,
                             metadata=data.metadata(), heads=heads, dropout=dropout_rate)
        
        # Output linear layer for the target node type
        self.out_lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict) # x_dict contains embeddings for all node types

        # Get features for the target node type
        target_node_features = x_dict[self.target_node_type]
        
        # Apply activation and dropout
        target_node_features = F.elu(target_node_features) # Using ELU as an example activation
        target_node_features = F.dropout(target_node_features, p=self.dropout_rate, training=self.training)
        
        return self.out_lin(target_node_features)


class SAGEModel(nn.Module):
    """GraphSAGE with residual and LayerNorm, for use with to_hetero."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_c, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
        self.out_lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            h = self.convs[i](x, edge_index)
            if i > 0:
                h = h + x # residual
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            x = h
        return self.out_lin(x)


class GATv2Model(nn.Module):
    """GATv2 with residual and LayerNorm, for use with to_hetero."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels * heads
            self.convs.append(GATv2Conv(in_c, hidden_channels, heads=heads, concat=True, dropout=dropout_rate, add_self_loops=False))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        self.out_lin = Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            h = self.convs[i](x, edge_index)
            if i > 0:
                h = h + x # residual
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            x = h
        return self.out_lin(x)


class RGCNModel(nn.Module):
    """Relational Graph Convolutional Network (RGCN) for Heterogeneous Graphs."""
    def __init__(self, data: HeteroData, hidden_channels: int, out_channels: int, num_layers: int, dropout_rate: float, target_node_type: str):
        super().__init__()
        self.target_node_type = target_node_type
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.lins = nn.ModuleDict()
        for node_type in data.node_types:
            self.lins[node_type] = Linear(data[node_type].num_features, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=len(data.edge_types)))

        self.out_lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Flatten x_dict to a single tensor and build mapping
        x_all = []
        node_offsets = {}
        offset = 0
        for node_type, x in x_dict.items():
            x_all.append(self.lins[node_type](x))
            node_offsets[node_type] = (offset, offset + x.size(0))
            offset += x.size(0)
        x = torch.cat(x_all, dim=0)

        # Build edge_index and edge_type for RGCNConv
        edge_index_list = []
        edge_type_list = []
        for i, edge_type in enumerate(edge_index_dict):
            src_type, rel, dst_type = edge_type
            src_offset = node_offsets[src_type][0]
            dst_offset = node_offsets[dst_type][0]
            edge_idx = edge_index_dict[edge_type].clone()
            edge_idx[0] += src_offset
            edge_idx[1] += dst_offset
            edge_index_list.append(edge_idx)
            edge_type_list.append(torch.full((edge_idx.size(1),), i, dtype=torch.long, device=x.device))
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_type = torch.cat(edge_type_list, dim=0)

        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Get output for target node type
        tgt_start, tgt_end = node_offsets[self.target_node_type]
        out = self.out_lin(x[tgt_start:tgt_end])
        return out

# --- Utility Functions ---

def load_hetero_graph(path: str, device: torch.device, target_node_type: str) -> HeteroData:
    """Load HeteroData graph and move it to the specified device."""
    try:
        data = torch.load(path, map_location=torch.device('cpu')) # Load to CPU first
        print(f"HeteroData loaded from {path}")
    except Exception as e:
        print(f"Error loading HeteroData: {e}")
        raise ValueError(f"Could not load HeteroData from {path}") from e

    if not isinstance(data, HeteroData):
        raise TypeError(f"Loaded data is not a HeteroData object (got {type(data)}).")

    # Validate target node type and its attributes
    if target_node_type not in data.node_types:
        raise ValueError(f"Target node type '{target_node_type}' not found in graph. Available: {data.node_types}")

    # Use new mask names: train_labeled_mask, train_unlabeled_mask, test_mask
    required_attrs = ['x', 'y', 'train_labeled_mask', 'train_unlabeled_mask', 'test_mask']
    for attr in required_attrs:
        if not hasattr(data[target_node_type], attr):
            raise AttributeError(f"Target node type '{target_node_type}' is missing required attribute: {attr}")

    # Move graph data to the target device
    data = data.to(device)
    print(f"HeteroData moved to {device}")

    # Ensure masks are boolean type for the target node
    data[target_node_type].train_labeled_mask = data[target_node_type].train_labeled_mask.bool()
    data[target_node_type].train_unlabeled_mask = data[target_node_type].train_unlabeled_mask.bool()
    data[target_node_type].test_mask = data[target_node_type].test_mask.bool()

    # Check if train_labeled_mask has any True values for training supervision
    if data[target_node_type].train_labeled_mask.sum() == 0:
        raise ValueError(f"Cannot train: No nodes available in train_labeled_mask for target node '{target_node_type}'.")

    return data

def get_model(model_name: str, data: HeteroData, args: ArgumentParser) -> nn.Module:
    """Initialize the Heterogeneous GNN model."""
    num_classes = data[args.target_node_type].y.max().item() + 1
    print(f"Number of classes for target node '{args.target_node_type}': {num_classes}")
    if model_name == "HGT":
        print("Using HGT (with residual + LayerNorm)")
        model = HGTModel(
            data=data,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.hgt_layers,
            heads=args.hgt_han_heads,
            target_node_type=args.target_node_type,
            dropout_rate=args.dropout_rate
        )
    elif model_name == "HAN":
        print("Using HAN (no residual, meta-path attention)")
        model = HANModel(
            data=data,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            heads=args.hgt_han_heads,
            target_node_type=args.target_node_type,
            dropout_rate=args.dropout_rate
        )
    elif model_name == "RGCN":
        print("Using RGCN (Relational GCN for hetero graphs)")
        model = RGCNModel(
            data=data,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.hgt_layers,  # 可另外加 rgcn_layers 參數
            dropout_rate=args.dropout_rate,
            target_node_type=args.target_node_type
        )
    elif model_name == "SAGE":
        print("Using GraphSAGE (with residual + LayerNorm, to_hetero)")
        base = SAGEModel(
            in_channels=data[args.target_node_type].num_features,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.sage_layers,
            dropout_rate=args.dropout_rate
        )
        model = to_hetero(base, data.metadata(), aggr="mean")
    elif model_name == "GATv2":
        print("Using GATv2 (with residual + LayerNorm, to_hetero)")
        base = GATv2Model(
            in_channels=data[args.target_node_type].num_features,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.gatv2_layers,
            heads=args.hgt_han_heads,
            dropout_rate=args.dropout_rate
        )
        model = to_hetero(base, data.metadata(), aggr="mean")
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    return model

def train_epoch(model: nn.Module, data: HeteroData, optimizer: torch.optim.Optimizer, criterion: nn.Module, target_node_type: str) -> tuple[float, float]:
    """Perform a single training epoch for HeteroData."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data[target_node_type].train_labeled_mask
    if isinstance(out, dict):
        out_target = out[target_node_type]
    else:
        out_target = out
    loss = criterion(out_target[mask], data[target_node_type].y[mask])
    loss.backward()
    optimizer.step()
    pred = out_target[mask].argmax(dim=1)
    correct = (pred == data[target_node_type].y[mask]).sum().item()
    acc = correct / mask.sum().item() if mask.sum().item() > 0 else 0
    return loss.item(), acc

@torch.no_grad()
def evaluate(model: nn.Module, data: HeteroData, eval_mask_name: str, criterion: nn.Module, target_node_type: str) -> tuple[float, float, float]:
    """Evaluate the model on a given data mask for HeteroData."""
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data[target_node_type][eval_mask_name]
    if isinstance(out, dict):
        out_target = out[target_node_type]
    else:
        out_target = out
    loss = criterion(out_target[mask], data[target_node_type].y[mask])
    pred = out_target[mask].argmax(dim=1)
    correct = (pred == data[target_node_type].y[mask]).sum().item()
    acc = correct / mask.sum().item() if mask.sum().item() > 0 else 0
    y_true = data[target_node_type].y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return loss.item(), acc, f1

def train(model: nn.Module, data: HeteroData, optimizer: torch.optim.Optimizer, criterion: nn.Module, args: ArgumentParser, output_dir: str, model_name_fs: str) -> dict:
    """Train the model with validation and early stopping."""
    print("\n--- Starting Heterogeneous Training ---")
    start_time = time.time()

    train_losses, train_accs = [], []
    val_losses, val_accs, val_f1s = [], [], []
    
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = -1

    model_save_path = os.path.join(output_dir, f"{model_name_fs}_best.pt")

    for epoch in range(args.n_epochs):
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion, args.target_node_type)
        # Using test_mask for validation during training, as is common in GNN literature for transductive settings
        val_loss, val_acc, val_f1 = evaluate(model, data, 'test_mask', criterion, args.target_node_type)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Epoch: {epoch+1:03d}/{args.n_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved (F1: {best_val_f1:.4f}) to {model_save_path}")
            patience_counter = 0
        elif val_loss < best_val_loss and patience_counter > 0: # Secondary criterion: loss improvement
            best_val_loss = val_loss
            # Optional: save model based on loss improvement too
            # torch.save(model.state_dict(), model_save_path + "_best_loss")
            patience_counter = 0 # Reset patience if loss improves
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    train_time = time.time() - start_time
    print(f"--- Heterogeneous Training Finished in {train_time:.2f} seconds ---")
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

def final_evaluation(model: nn.Module, data: HeteroData, model_path: str, target_node_type: str) -> dict:
    print("\n--- Final Heterogeneous Evaluation on Test Set ---")
    try:
        model.load_state_dict(torch.load(model_path, map_location=data[target_node_type].x.device))
        print(f"Loaded best model weights from {model_path}")
    except FileNotFoundError:
        print(f"Warning: Best model file not found at {model_path}. Evaluating with current model state (if any).")
    except Exception as e:
        print(f"Warning: Could not load best model weights from {model_path}. Evaluating with last state. Error: {e}")

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        mask = data[target_node_type].test_mask
        if isinstance(out, dict):
            out_target = out[target_node_type]
        else:
            out_target = out
        pred = out_target[mask].argmax(dim=1)
        y_true = data[target_node_type].y[mask].cpu().numpy()
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
        'confusion_matrix': conf_matrix.tolist()
    }
    print(f"Target Node: '{target_node_type}'")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix (for target node '{target_node_type}'):\n{metrics['confusion_matrix']}")
    print("--- Heterogeneous Evaluation Finished ---")
    return metrics

def save_results(history: dict, final_metrics: dict, args: ArgumentParser, output_dir: str, model_name_fs: str) -> None:
    """Save training history, final metrics, and plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the main output_dir for plots, not a subdirectory called PLOTS_DIR inside it
    # plot_dir = os.path.join(output_dir, PLOTS_DIR) 
    # os.makedirs(plot_dir, exist_ok=True) # Ensure PLOTS_DIR is created if used

    results_data = {
        "args": vars(args),
        "model_name": model_name_fs, # Full model name with scenario
        "training_history": {
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_train_acc": history["train_acc"][-1] if history["train_acc"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
            "final_val_f1": history["val_f1"][-1] if history["val_f1"] else None,
            "best_val_f1": history["best_val_f1"],
            "best_epoch": history["best_epoch"],
            "total_epochs_run": len(history["train_loss"]),
            "training_time_seconds": history["train_time"],
        },
        "final_test_metrics_on_target_node": final_metrics
    }
    results_path = os.path.join(output_dir, f"metrics_{model_name_fs}.json") # Include model_name_fs in metrics filename
    try:
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Error saving results JSON: {e}")

    plot_path = os.path.join(output_dir, f"training_curves_{model_name_fs}.png") # Include model_name_fs
    try:
        epochs_ran = range(1, len(history['train_loss']) + 1)
        if not epochs_ran: # Handle case where training did not run
            print("No training history to plot.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"Training Curves - {model_name_fs} (Target: {args.target_node_type})")

        axes[0].plot(epochs_ran, history['train_acc'], label='Train Accuracy', marker='.')
        axes[0].plot(epochs_ran, history['val_acc'], label='Validation Accuracy', marker='.')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].set_title('Accuracy')
        axes[0].legend(); axes[0].grid(True)

        axes[1].plot(epochs_ran, history['train_loss'], label='Train Loss', marker='.')
        axes[1].plot(epochs_ran, history['val_loss'], label='Validation Loss', marker='.')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].set_title('Loss')
        axes[1].legend(); axes[1].grid(True)

        axes[2].plot(epochs_ran, history['val_f1'], label='Validation F1 Score', marker='.', color='green')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('F1 Score'); axes[2].set_title('Validation F1')
        axes[2].legend(); axes[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path)
        plt.close()
        print(f"Training plots saved to {plot_path}")
    except Exception as e:
         print(f"Error saving plots: {e}")


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Train Heterogeneous Graph Neural Networks for Fake News Detection")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to the preprocessed HeteroData graph (.pt file)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=["HGT", "HAN", "RGCN", "SAGE", "GATv2"], help="Heterogeneous GNN model type")
    parser.add_argument("--loss_fn", type=str, default=DEFAULT_LOSS_FN, choices=["ce", "focal"], help="Loss function")
    parser.add_argument("--target_node_type", type=str, default=DEFAULT_TARGET_NODE_TYPE, help="Target node type for classification")
    parser.add_argument("--dropout_rate", type=float, default=DEFAULT_DROPOUT, help="Dropout rate")
    parser.add_argument("--hidden_channels", type=int, default=DEFAULT_HIDDEN_CHANNELS, help="Number of hidden units in GNN layers")
    parser.add_argument("--hgt_han_heads", type=int, default=DEFAULT_HGT_HAN_HEADS, help="Number of attention heads for HGT/HAN/GATv2 models")
    parser.add_argument("--hgt_layers", type=int, default=DEFAULT_HGT_LAYERS, help="Number of layers for HGT model")
    parser.add_argument("--sage_layers", type=int, default=2, help="Number of layers for SAGE model")
    parser.add_argument("--gatv2_layers", type=int, default=2, help="Number of layers for GATv2 model")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay (L2 regularization)")
    parser.add_argument("--n_epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output_dir_base", type=str, default=RESULTS_DIR, help="Base directory to save results and plots")
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache(); gc.collect()

    data = load_hetero_graph(args.graph_path, device, args.target_node_type)

    try: # Construct a meaningful scenario name from graph_path
        parts = args.graph_path.split(os.sep)
        scenario_filename = os.path.splitext(parts[-1])[0] # e.g., 8_shot_roberta_hetero_knn_5_smpf10
        dataset_name_from_path = parts[-2] if len(parts) > 1 else "unknown_dataset"
    except IndexError:
        scenario_filename = "unknown_scenario"
        dataset_name_from_path = "unknown_dataset"
    
    # Full model name for filesystem/logging (includes model type, dataset, scenario)
    model_name_fs = f"{args.model}_{dataset_name_from_path}_{scenario_filename}"
    # Output directory: base_dir/model_type/dataset_name/scenario_filename/
    output_dir = os.path.join(args.output_dir_base, args.model, dataset_name_from_path, scenario_filename)
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- HeteroGraph Configuration ---")
    print(f"Model:           {args.model}")
    print(f"Target Node:     {args.target_node_type}")
    print(f"Dataset (path):  {dataset_name_from_path}")
    print(f"Scenario (file): {scenario_filename}")
    print(f"Graph Path:      {args.graph_path}")
    print(f"Output Dir:      {output_dir}")
    print("Arguments:")
    for k, v in vars(args).items(): print(f"  {k:<18}: {v}")
    
    print("\n--- HeteroData Info ---")
    for node_type in data.node_types:
        print(f"  Node type: '{node_type}'")
        print(f"    - Num nodes: {data[node_type].num_nodes}")
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            print(f"    - Features: {data[node_type].x.shape[1]}")
        if node_type == args.target_node_type:
            print(f"    - Train labeled mask sum: {data[node_type].train_labeled_mask.sum().item()}")
            print(f"    - Train unlabeled mask sum: {data[node_type].train_unlabeled_mask.sum().item()}")
            print(f"    - Test mask sum: {data[node_type].test_mask.sum().item()}")
            if hasattr(data[node_type],'y') and data[node_type].y is not None:
                 print(f"    - Num classes: {data[node_type].y.max().item() + 1}")

    print("  Edge types and counts:")
    for edge_type in data.edge_types:
        print(f"    - {edge_type}: {data[edge_type].num_edges} edges")
    
    model = get_model(args.model, data, args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss() if args.loss_fn == "ce" else FocalLoss()

    print("\n--- Model Architecture ---"); print(model); print("-------------------------")

    training_history = train(model, data, optimizer, criterion, args, output_dir, model_name_fs)
    
    model_path = os.path.join(output_dir, f"{model_name_fs}_best.pt")
    final_metrics = final_evaluation(model, data, model_path, args.target_node_type)
    
    save_results(training_history, final_metrics, args, output_dir, model_name_fs)

    print("\n--- Heterogeneous Pipeline Complete ---")
    print(f"Results, plots, and best model saved in: {output_dir}")
    print("-------------------------------------\n")

if __name__ == "__main__":
    main()
