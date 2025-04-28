import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from argparse import ArgumentParser, Namespace
from datasets import load_dataset
import matplotlib.pyplot as plt
from utils.sample_k_shot import sample_k_shot # Assuming this utility exists
import json

# Constants
SEED = 42
DEFAULT_EMBEDDING_TYPE = "roberta"
DEFAULT_BATCH_SIZE = 64
RESULTS_DIR = "results" # Changed base directory

def set_seed(seed: int = SEED) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class LSTMClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, add_dropout=True):
        super(LSTMClassifier, self).__init__()
        # LSTM expects input shape (batch, seq_len, input_size)
        # Our input is (batch, input_size), so we'll treat seq_len as 1
        self.embedding_proj = nn.Linear(in_channels, hidden_channels) # Optional projection
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers, batch_first=True,
                            dropout=0.6 if add_dropout and num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.6 if add_dropout else 0)

    def forward(self, x):
        # x shape: (batch_size, in_channels)
        x = self.embedding_proj(x) # Project to hidden_channels
        x = F.relu(x)
        x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, hidden_channels)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, 1, hidden_channels)
        lstm_out = lstm_out.squeeze(1) # Remove sequence dimension: (batch_size, hidden_channels)
        out = self.dropout(lstm_out)
        out = self.fc(out) # (batch_size, out_channels)
        return out

class MLPClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, add_dropout=True):
        super(MLPClassifier, self).__init__()
        self.layers = nn.ModuleList()
        dropout_rate = 0.6 if add_dropout else 0

        if num_layers == 1:
            # Direct mapping if num_layers is 1
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            # Input layer
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            self.layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_channels, hidden_channels))
                self.layers.append(nn.ReLU())
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))

            # Output layer
            self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- Data Handling ---

class EmbeddingDataset(Dataset):
    """Simple Dataset for embeddings and labels."""
    def __init__(self, embeddings, labels, device):
        # Move data to device during initialization
        self.embeddings = torch.tensor(np.array(embeddings), dtype=torch.float).to(device)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long).to(device)
        self.device = device # Store device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Data is already on the correct device
        return self.embeddings[idx], self.labels[idx]

def load_and_prepare_data(dataset_name: str, embedding_type: str, k_shot: int, batch_size: int, seed: int, device: torch.device):
    """Loads dataset, extracts embeddings, creates DataLoaders."""
    print(f"Loading dataset '{dataset_name}' with {embedding_type} embeddings...")
    hf_dataset_name = f"LittleFish-Coder/Fake_News_{dataset_name.capitalize()}"
    dataset = load_dataset(hf_dataset_name, cache_dir="dataset")

    train_hf_dataset = dataset["train"]
    test_hf_dataset = dataset["test"]

    embedding_field = f"{embedding_type}_embeddings"
    print(f"Using embeddings from field: '{embedding_field}'")

    # --- Handle Few-Shot Sampling ---
    if k_shot > 0:
        print(f"Applying {k_shot}-shot sampling to the training set (seed={seed})...")
        # Use the same sampling logic as build_graph.py
        # sample_k_shot returns indices and the sampled dataset dictionary
        selected_indices, _ = sample_k_shot(train_hf_dataset, k_shot, seed)
        print(f"Selected {len(selected_indices)} samples for {k_shot}-shot training.")
        # Create a subset of the original Hugging Face training dataset
        train_subset_hf = train_hf_dataset.select(selected_indices)
        train_embeddings = train_subset_hf[embedding_field]
        train_labels = train_subset_hf["label"]
        # Validation split comes from the *sampled* training data
        full_train_dataset = EmbeddingDataset(train_embeddings, train_labels, device)
    else:
        print("Using full training set.")
        train_embeddings = train_hf_dataset[embedding_field]
        train_labels = train_hf_dataset["label"]
        # Validation split comes from the *full* training data
        full_train_dataset = EmbeddingDataset(train_embeddings, train_labels, device)


    test_embeddings = test_hf_dataset[embedding_field]
    test_labels = test_hf_dataset["label"]

    # --- Create Validation Split ---
    # Split the *selected* training data (either full or k-shot) into train/val
    total_train_size = len(full_train_dataset)
    val_size = int(total_train_size * 0.15) # 15% for validation
    train_size = total_train_size - val_size

    # Ensure val_size is at least 1 if possible
    if total_train_size > 0:
        val_size = max(1, val_size)
        train_size = total_train_size - val_size
    else:
        val_size = 0
        train_size = 0


    print(f"Splitting training data: {train_size} train, {val_size} validation samples.")
    if train_size == 0 or val_size == 0:
         print("Warning: Training or validation set is empty after split. Check k-shot value and dataset size.")
         # Handle empty sets gracefully if needed, e.g., skip validation
         train_dataset = full_train_dataset
         val_dataset = None # No validation possible
    else:
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed) # Use seed for split reproducibility
        )


    test_dataset = EmbeddingDataset(test_embeddings, test_labels, device)

    # --- Create DataLoaders ---
    # Handle potentially empty datasets in loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_size > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset and val_size > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train loader: {len(train_loader.dataset) if train_loader else 0} samples, {len(train_loader) if train_loader else 0} batches")
    print(f"Val loader: {len(val_loader.dataset) if val_loader else 0} samples, {len(val_loader) if val_loader else 0} batches")
    print(f"Test loader: {len(test_loader.dataset)} samples, {len(test_loader)} batches")


    # Determine input dimension from the first batch of train_loader if possible
    if train_loader:
        first_batch_embeddings, _ = next(iter(train_loader))
        in_channels = first_batch_embeddings.shape[1]
    elif test_loader: # Fallback to test loader if train loader is empty
         first_batch_embeddings, _ = next(iter(test_loader))
         in_channels = first_batch_embeddings.shape[1]
    else:
        raise ValueError("Could not determine embedding dimension - both train and test loaders are empty.")


    print(f"Input embedding dimension: {in_channels}")

    return train_loader, val_loader, test_loader, in_channels


# --- Training and Evaluation ---

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    if not loader: return 0.0, 0.0 # Handle empty loader

    for embeddings, labels in loader:
        # Data is already on the device from EmbeddingDataset
        # embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * embeddings.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return accuracy, avg_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    if not loader: return 0.0, 0.0, 0.0 # Handle empty loader

    with torch.no_grad():
        for embeddings, labels in loader:
            # embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0) if total > 0 else 0
    return accuracy, avg_loss, f1

def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, patience, device, output_dir, model_name):
    train_accs, train_losses = [], []
    val_accs, val_losses, val_f1s = [], [], []
    best_val_metric = -1 # Use F1 for selecting best model
    best_epoch = -1
    epochs_no_improve = 0
    model_path = os.path.join(output_dir, f"{model_name}.pt")

    start_time = time.time()

    for epoch in range(n_epochs):
        train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Only validate if val_loader exists
        if val_loader:
            val_acc, val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, '
                  f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)

            # Early stopping and model saving based on validation F1
            current_val_metric = val_f1
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print(f"Model improved and saved to {model_path} (epoch {epoch}, best val F1: {val_f1:.4f})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement.")
                    break
        else:
             # If no validation set, just log training progress and save last model
             print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}')
             train_accs.append(train_acc)
             train_losses.append(train_loss)
             # Save the model from the last epoch if no validation is done
             if epoch == n_epochs - 1:
                 torch.save(model.state_dict(), model_path)
                 print(f"Model saved after final epoch to {model_path}")
                 best_epoch = epoch # Mark last epoch as 'best'


    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    if val_loader:
        print(f"Best model saved at epoch {best_epoch} with Val F1: {best_val_metric:.4f}")
    else:
        print(f"Model from last epoch ({best_epoch}) saved.")


    history = {
        'train_accs': train_accs, 'train_losses': train_losses,
        'val_accs': val_accs, 'val_losses': val_losses, 'val_f1s': val_f1s
    }
    return history, train_time, best_epoch

def detailed_test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    if not loader: return {} # Handle empty loader

    with torch.no_grad():
        for embeddings, labels in loader:
            # embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels: return {} # Handle case where loader was empty

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist() # Convert numpy array for JSON serialization
    }
    return metrics

# --- Plotting and Saving ---

def save_results(history, test_metrics, model_name, output_dir, train_time, best_epoch, args):
    os.makedirs(output_dir, exist_ok=True)

    # Save arguments
    args_dict = vars(args)
    with open(os.path.join(output_dir, f"{model_name}_args.json"), "w") as f:
        import json
        json.dump(args_dict, f, indent=2)


    # Save metrics summary
    results = {
        "model": model_name,
        "dataset": args.dataset_name,
        "embedding": args.embedding_type,
        "k_shot": args.k_shot if args.k_shot > 0 else "Full",
        "training_time": train_time,
        "best_epoch": best_epoch,
        "final_train_accuracy": history['train_accs'][-1] if history['train_accs'] else None,
        "final_train_loss": history['train_losses'][-1] if history['train_losses'] else None,
        "final_val_accuracy": history['val_accs'][-1] if history['val_accs'] else None,
        "final_val_loss": history['val_losses'][-1] if history['val_losses'] else None,
        "final_val_f1": history['val_f1s'][-1] if history['val_f1s'] else None,
        "test_metrics": test_metrics if test_metrics else None
    }

    with open(os.path.join(output_dir, f"metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_dir}")

def plot_metrics(history, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(len(history['train_accs']))

    plt.figure(figsize=(18, 5))

    # Plot Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_accs'], label='Train Accuracy')
    if history['val_accs']:
        plt.plot(epochs, history['val_accs'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_losses'], label='Train Loss')
    if history['val_losses']:
        plt.plot(epochs, history['val_losses'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot F1 Score (Validation only)
    if history['val_f1s']:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, history['val_f1s'], label='Validation F1 Score', color='green')
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()

    plt.suptitle(f"Training Metrics - {model_name.replace('_', ' ')}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plot_path = os.path.join(output_dir, f"{model_name}_training_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training plots saved to {plot_path}")


# --- Main Execution ---

def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Train MLP or LSTM on pre-computed embeddings for fake news detection")
    parser.add_argument("--dataset_name", type=str, default="politifact", choices=["politifact", "gossipcop"], help="Dataset to use")
    parser.add_argument("--embedding_type", type=str, default=DEFAULT_EMBEDDING_TYPE, choices=["bert", "roberta"], help="Type of embeddings to use")
    parser.add_argument("--model_type", type=str, default="MLP", choices=["MLP", "LSTM"], help="Model architecture")
    parser.add_argument("--k_shot", type=int, default=0, help="Number of samples per class for few-shot learning (0 for full training set)")

    # Hyperparameters
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden units")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers (for MLP/LSTM)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 penalty)")
    parser.add_argument("--n_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--dropout", action="store_true", default=True, help="Enable dropout")
    parser.add_argument("--no-dropout", dest="dropout", action="store_false", help="Disable dropout")
    parser.add_argument("--no_class_balance", action="store_true", help="Disable class balancing in loss function")


    # Others
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--output_dir_base", type=str, default=RESULTS_DIR, help="Base directory for saving results")

    return parser.parse_args()

def main():
    args = parse_arguments()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    train_loader, val_loader, test_loader, in_channels = load_and_prepare_data(
        args.dataset_name, args.embedding_type, args.k_shot, args.batch_size, args.seed, device
    )

    # Handle case where loaders might be None (e.g., k_shot too small)
    if not train_loader:
        print("Error: Training loader is empty. Cannot proceed.")
        return


    # --- Model Initialization ---
    if args.model_type == "MLP":
        model = MLPClassifier(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=2, # Assuming binary classification (real/fake)
            num_layers=args.num_layers,
            add_dropout=args.dropout
        )
    elif args.model_type == "LSTM":
        model = LSTMClassifier(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=2,
            num_layers=args.num_layers,
            add_dropout=args.dropout
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.to(device)
    print(f"Initialized {args.model_type} model:")
    print(model)

    # --- Loss and Optimizer ---
    # Handle class imbalance (optional) - Calculate weights based on training data if possible
    if not args.no_class_balance and train_loader:
        all_labels = []
        # Iterate through the dataset backing the loader to get all labels
        if isinstance(train_loader.dataset, Subset):
             # If it's a Subset (due to train/val split), access the underlying dataset
             indices = train_loader.dataset.indices
             underlying_dataset = train_loader.dataset.dataset
             all_labels = [underlying_dataset.labels[i].item() for i in indices]
        else:
             # If it's the full dataset
             all_labels = [label.item() for _, label in train_loader.dataset]


        if all_labels:
             counts = np.bincount(all_labels)
             # Calculate weights inverse to class frequency: weight = total_samples / (num_classes * count)
             total_samples = len(all_labels)
             num_classes = len(counts)
             weights = total_samples / (num_classes * counts + 1e-9) # Add epsilon for stability
             class_weights = torch.FloatTensor(weights).to(device)
             print(f"Using class weights for loss: {class_weights.cpu().numpy()}")
             criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
             print("Warning: Could not calculate class weights (empty training data?). Using unweighted loss.")
             criterion = nn.CrossEntropyLoss()
    else:
        print("Using unweighted CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Training ---
    k_shot_str = f"{args.k_shot}shot" if args.k_shot > 0 else "full"
    model_name = f"{args.model_type}_{args.dataset_name}_{args.embedding_type}_{k_shot_str}"
    # Adjusted output directory structure
    output_dir = os.path.join(args.output_dir_base, args.model_type, args.dataset_name, k_shot_str)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting training for {model_name}...")
    print(f"Results will be saved in: {output_dir}")

    history, train_time, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, args.n_epochs, args.patience, device, output_dir, model_name
    )

    # --- Evaluation ---
    print("Loading best model for final evaluation...")
    model_path = os.path.join(output_dir, f"{model_name}.pt")
    if os.path.exists(model_path):
        try:
            # Load weights safely if possible, otherwise fallback
            try:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            except:
                print("Could not load with weights_only=True, attempting standard load...")
                model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Best model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading best model state_dict: {e}. Evaluating with the last state.")
    else:
        print("Warning: Best model checkpoint not found. Evaluating with the last state.")


    print("Evaluating on test set...")
    test_metrics = detailed_test(model, test_loader, device)

    if test_metrics:
        print("Detailed Test Metrics (Best Model):")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
        print("  Confusion Matrix:")
        print(np.array(test_metrics['confusion_matrix']))
    else:
        print("Could not compute test metrics (Test loader might be empty).")


    # --- Save Results and Plot ---
    save_results(history, test_metrics, model_name, output_dir, train_time, best_epoch, args)
    if history['train_accs']: # Only plot if training happened
        plot_metrics(history, model_name, output_dir)

    print("Finished.")


if __name__ == "__main__":
    main()

