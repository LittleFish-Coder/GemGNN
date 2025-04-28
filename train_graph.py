import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from argparse import ArgumentParser
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import json


# GCN model class
class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) implementation.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, add_dropout=True):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = 0.6 if add_dropout else 0.0  # Higher dropout rate
        
    def forward(self, x, edge_index, edge_attr=None):
        # First Graph Convolution layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second Graph Convolution layer
        x = self.conv2(x, edge_index)
        
        return x


# GAT model class
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, add_dropout=True, heads=8):
        super(GAT, self).__init__()
        
        # Initial projection layer
        self.feature_proj = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # First GAT layer
        self.conv1 = GATConv(
            hidden_channels, 
            hidden_channels // heads,
            heads=heads,
            dropout=0.3  # Attention dropout
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Second GAT layer
        self.conv2 = GATConv(
            hidden_channels, 
            hidden_channels,
            heads=1,
            concat=False,
            dropout=0.3
        )
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Output classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # Higher dropout rate
        self.dropout = 0.7 if add_dropout else 0.0
        
    def forward(self, x, edge_index, edge_attr=None):
        # Initial feature projection
        identity = x
        x = self.feature_proj(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # First GAT layer with residual connection
        identity = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x + identity)  # Residual connection
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer with residual connection
        identity = x
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.elu(x + identity)  # Residual connection
        
        # Output classifier
        x = self.classifier(x)
        
        return x


# GraphSAGE model class
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, add_dropout=True):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = 0.6 if add_dropout else 0.0

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x


def show_args(args, model_name, dataset_name, scenario):
    """
    Display all arguments and configuration settings
    """
    print("========================================")
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print(f"\tModel name: {model_name}")
    print(f"\tDataset: {dataset_name}")
    print(f"\tScenario: {scenario}")
    print("========================================")


def load_graph(path: str):
    """
    Load the graph data from specified path
    
    The function handles PyTorch Geometric data loading issues with weights_only parameter
    """
    try:
        # First try with the safer option
        graph = torch.load(path, weights_only=True)
    except Exception as e:
        print(f"Could not load with weights_only=True: {str(e)}")
        print("Trying to load without weights_only restriction...")
        
        # Try to load without the weights_only restriction
        # This is safe for your own graph data files
        graph = torch.load(path)
    
    print(f"Graph loaded from {path}")
    return graph


def get_model_criterion_optimizer(graph_data, base_model: str, dropout: bool, hidden_channels=16, 
                              num_layers=2, lr=0.0001, weight_decay=5e-4, class_balance=True):
    """
    Initialize model, loss function, and optimizer
    
    Args:
        graph_data: The loaded graph data
        base_model: Model type (GCN, GAT, LSTM, MLP)
        dropout: Whether to use dropout regularization
        hidden_channels: Number of hidden units
        num_layers: Number of layers for LSTM/MLP
        lr: Learning rate
        weight_decay: Weight decay factor for regularization
        class_balance: Whether to use class weights in loss function
        
    Returns:
        model, criterion, optimizer
    """
    # Initialize the appropriate model
    if base_model == "GCN":
        model = GCN(in_channels=graph_data.num_features, hidden_channels=hidden_channels, 
                   out_channels=2, add_dropout=dropout)
    elif base_model == "GAT":
        model = GAT(in_channels=graph_data.num_features, hidden_channels=hidden_channels, 
                   out_channels=2, add_dropout=dropout)
    elif base_model == "GraphSAGE":
        model = GraphSAGE(in_channels=graph_data.num_features, hidden_channels=hidden_channels, 
                         out_channels=2, add_dropout=dropout)
    else:
        raise ValueError(f"Unsupported model type: {base_model}")
    
    # Determine device - get it from graph_data tensors
    device = graph_data.x.device
    
    # Handle class imbalance
    if class_balance:
        # Priliminary Knowledge:
        # In real world, the estimated class weights are approximatey:
        # Real:Fake = 8:2 (So, we can use this to set the class weights)
        
        prior_class_weights = np.array([1, 1])
        weights = torch.FloatTensor(prior_class_weights).to(device)
        print(f"Using class weights: {prior_class_weights}")
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Use weight decay to control overfitting
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


def train_val_test(graph_data, model, criterion, optimizer, n_epochs=300, patience=20, 
                  output_dir='results', model_name='model'):
    """
    Train, validate, and test the model
    
    Args:
        graph_data: Graph data with features and labels
        model: The initialized model
        criterion: Loss function
        optimizer: Optimizer
        n_epochs: Maximum number of training epochs
        patience: Early stopping patience
        output_dir: Directory to save results
        model_name: Name of the model
        
    Returns:
        Various training metrics and statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_path = f"{output_dir}/{model_name}.pt"

    def train(graph_data):
        """Train for a single epoch"""
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index if hasattr(graph_data, 'edge_index') else None)

        loss = criterion(out[graph_data.labeled_mask], graph_data.y[graph_data.labeled_mask])
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Calculate training accuracy
        pred = out[graph_data.labeled_mask].argmax(dim=1)  # Predicted labels
        correct = (pred == graph_data.y[graph_data.labeled_mask]).sum().item()  # Correct predictions
        train_acc = correct / graph_data.labeled_mask.sum().item()  # Training accuracy
    
        return train_acc, loss.item()

    def validate(graph_data):
        """Validate model performance"""
        model.eval()
        with torch.no_grad():
            out = model(graph_data.x, graph_data.edge_index if hasattr(graph_data, 'edge_index') else None)
            val_loss = criterion(out[graph_data.test_mask], graph_data.y[graph_data.test_mask])
            val_pred = out[graph_data.test_mask].argmax(dim=1)
            val_correct = (val_pred == graph_data.y[graph_data.test_mask]).sum().item()
            val_acc = val_correct / graph_data.test_mask.sum().item()
            
            # Calculate F1 score
            y_true = graph_data.y[graph_data.test_mask].cpu().numpy()
            y_pred = val_pred.cpu().numpy()
            val_f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
        return val_acc, val_loss.item(), val_f1

    # Initialize records
    train_accs, train_losses = [], []
    val_accs, val_losses, val_f1s = [], [], []
    best_val_loss = float('inf')
    best_val_f1 = 0
    best_epoch = -1
    no_improve = 0
    start_time = time.time()

    # Train the model
    for epoch in range(n_epochs):
        train_acc, train_loss = train(graph_data)
        val_acc, val_loss, val_f1 = validate(graph_data)

        # Store metrics for each epoch
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

        # Save best model (based on validation F1 score)
        improved = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path} (epoch {epoch}, best F1: {val_f1:.4f})")
            improved = True
            no_improve = 0
        
        # Also count as improvement if validation loss decreases
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            if not improved:  # Avoid duplicate save
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print(f"Model saved at {model_path} (epoch {epoch}, best loss: {val_loss:.4f})")
                improved = True
                no_improve = 0
        
        # Early stopping mechanism
        if not improved:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}, no improvement for {patience} epochs")
                break

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Best model saved at epoch {best_epoch}")

    # Evaluate the model with the final state
    last_test_acc, last_test_loss, last_test_f1 = validate(graph_data)
    print(f'Test metrics (with last epoch model): Acc={last_test_acc:.4f}, F1={last_test_f1:.4f}')

    return train_accs, train_losses, val_accs, val_losses, val_f1s, last_test_acc, last_test_f1, train_time, best_epoch


def load_best_model(model, model_path):
    """
    Load the best model from saved checkpoint
    
    Args:
        model: Model architecture
        model_path: Path to the saved model weights
        
    Returns:
        Model with loaded weights
    """
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Best model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    return model


def detailed_test(model, graph_data):
    """
    Perform detailed testing and return comprehensive metrics
    
    Args:
        model: Trained model
        graph_data: Graph data with test samples
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index if hasattr(graph_data, 'edge_index') else None)
        pred = out[graph_data.test_mask].argmax(dim=1)
        y_true = graph_data.y[graph_data.test_mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
        
        return metrics


def save_results(train_accs, train_losses, val_accs, val_losses, val_f1s, last_test_acc, last_test_f1, 
                inference_metrics, model_name, output_dir, train_time, best_epoch):
    """
    Save the training results and metrics to files
    
    Args:
        train_accs, train_losses, val_accs, val_losses, val_f1s: Training history
        last_test_acc, last_test_f1: Final model performance
        inference_metrics: Detailed evaluation metrics
        model_name: Name of the model
        output_dir: Directory to save results
        train_time: Total training time
        best_epoch: Epoch with best performance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "Training Time": f"{train_time:.2f} seconds",
        "Best Epoch": best_epoch,
        "Train Accuracy": train_accs[-1],
        "Train Loss": train_losses[-1],
        "Validation Accuracy": val_accs[-1],
        "Validation Loss": val_losses[-1],
        "Validation F1": val_f1s[-1],
        "Test Accuracy (with last epoch model)": last_test_acc,
        "Test F1 (with last epoch model)": last_test_f1,
        "Detailed Metrics (with best model)": {
            "Accuracy": inference_metrics['accuracy'],
            "Precision": inference_metrics['precision'],
            "Recall": inference_metrics['recall'],
            "F1 Score": inference_metrics['f1_score'],
            "Confusion Matrix": inference_metrics['confusion_matrix'].tolist()  # Convert numpy array to list
        }
    }

    with open(f"{output_dir}/mertrics.json", "w") as f:
        json.dump(results, f, indent=4)


def plot_acc_loss(train_accs, train_losses, val_accs, val_losses, val_f1s, model_name, output_dir):
    """
    Plot training and validation metrics
    
    Args:
        train_accs, train_losses: Training metrics
        val_accs, val_losses, val_f1s: Validation metrics
        model_name: Name of the model
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot accuracy
    axes[0].plot(train_accs, label='Train Accuracy')
    axes[0].plot(val_accs, label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].legend()

    # Plot loss
    axes[1].plot(train_losses, label='Train Loss')
    axes[1].plot(val_losses, label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss')
    axes[1].legend()
    
    # Plot F1 score
    axes[2].plot(val_f1s, label='Validation F1 Score', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score')
    axes[2].legend()
    
    plt.suptitle(f"Training Metrics - {model_name.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_training_metrics.png")
    plt.close()


def extract_dataset_scenario(graph_path):
    """
    Extract dataset name and scenario from graph path
    
    Args:
        graph_path: Path to the graph file
        
    Returns:
        dataset_name, scenario
    """
    # Example: graphs/politifact/8shot_knn5.pt
    parts = graph_path.split('/')
    
    # Extract dataset name (politifact, gossipcop, etc.)
    dataset_name = parts[-2] if len(parts) > 1 else "unknown"
    
    # Extract scenario (8shot_knn5, etc.)
    filename = parts[-1]
    scenario = '.'.join(filename.split('.')[:-1])  # Remove only .pt extension
    return dataset_name, scenario


def main():
    parser = ArgumentParser(description="Train graph neural networks for fake news detection")
    parser.add_argument("--graph", type=str, help="path to graph data", required=True)
    parser.add_argument("--base_model", type=str, default="GAT", 
                        help="base model to use", choices=["GCN", "GAT", "GraphSAGE"])
    parser.add_argument("--dropout", action="store_true", help="Enable dropout", default=True)
    parser.add_argument("--no-dropout", dest="dropout", action="store_false", help="Disable dropout")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--hidden_channels", type=int, default=64, help="number of hidden channels")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers in LSTM/MLP")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay for regularization")
    parser.add_argument("--patience", type=int, default=20, help="patience for early stopping")
    parser.add_argument("--use_train_mask", action="store_true", help="use train_mask instead of labeled_mask")
    parser.add_argument("--no_class_balance", action="store_true", help="disable class balancing in loss function")

    args = parser.parse_args()
    graph_path = args.graph
    base_model = args.base_model
    dropout = args.dropout
    n_epochs = args.n_epochs
    hidden_channels = args.hidden_channels
    num_layers = args.num_layers
    lr = args.lr
    weight_decay = args.weight_decay
    patience = args.patience
    use_train_mask = args.use_train_mask
    class_balance = not args.no_class_balance

    # Extract dataset name and scenario from graph path
    dataset_name, scenario = extract_dataset_scenario(graph_path)
    
    # Format model name
    model_name = f"{base_model}"

    # Create output directory structure: results/model/dataset/scenario
    output_dir = os.path.join("results", model_name, dataset_name, scenario)
    
    # Show arguments
    show_args(args, model_name, dataset_name, scenario)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Load graph
    graph_data = load_graph(graph_path)
    
    # Move graph data to device
    graph_data.x = graph_data.x.to(device)
    graph_data.y = graph_data.y.to(device)
    graph_data.train_mask = graph_data.train_mask.to(device)
    graph_data.test_mask = graph_data.test_mask.to(device)
    
    # Move edge_index to device if it exists
    if hasattr(graph_data, 'edge_index'):
        graph_data.edge_index = graph_data.edge_index.to(device)
    
    # Move edge_attr to device if it exists
    if hasattr(graph_data, 'edge_attr'):
        graph_data.edge_attr = graph_data.edge_attr.to(device)

    # Check if labeled_mask is all zero(indicates no labeled nodes)
    if graph_data.labeled_mask.sum() == 0:
        graph_data.labeled_mask = graph_data.train_mask

    # Display class distribution
    train_labels = graph_data.y[graph_data.labeled_mask].cpu().numpy()
    class_counts = np.bincount(train_labels)
    print(f"Training class distribution: Real={class_counts[0]}, Fake={class_counts[1]}")
    
    test_labels = graph_data.y[graph_data.test_mask].cpu().numpy()
    test_counts = np.bincount(test_labels)
    print(f"Test class distribution: Real={test_counts[0]}, Fake={test_counts[1]}")

    # Initialize model, criterion, optimizer
    model, criterion, optimizer = get_model_criterion_optimizer(
        graph_data, base_model, dropout, hidden_channels, num_layers, lr, weight_decay, class_balance
    )
    model = model.to(device)

    # Train, validate, test
    train_accs, train_losses, val_accs, val_losses, val_f1s, last_test_acc, last_test_f1, train_time, best_epoch = train_val_test(
        graph_data, model, criterion, optimizer, n_epochs, patience, output_dir, model_name
    )

    # Load best model and evaluate
    model_path = f"{output_dir}/{model_name}.pt"
    model = load_best_model(model, model_path)

    # Get detailed evaluation metrics
    metrics = detailed_test(model, graph_data)
    print("\nDetailed Test Metrics (Best Model):")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])

    # Save results
    save_results(
        train_accs, train_losses, val_accs, val_losses, val_f1s,
        last_test_acc, last_test_f1, metrics,
        model_name, output_dir, train_time, best_epoch
    )

    # Plot training metrics
    plot_acc_loss(train_accs, train_losses, val_accs, val_losses, val_f1s, model_name, output_dir)


if __name__ == "__main__":
    main()