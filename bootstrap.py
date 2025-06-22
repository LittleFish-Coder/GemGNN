import os
import torch
import numpy as np
from torch_geometric.data import HeteroData
import json
from typing import Dict, List, Tuple
from train_hetero_graph import get_model, train_epoch, evaluate, set_seed, train
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from scipy.stats import mode
from train_hetero_graph import load_hetero_graph, parse_arguments
import tempfile
import matplotlib.pyplot as plt

# Constants for enhanced regularization (relaxed for bootstrap stability)
EPSILON = 1e-3  # Minimum difference to consider as improvement
OVERFIT_THRESHOLD = 1.0  # Relaxed threshold for bootstrap stability


class FewShotBootstrapValidator:
    """Bootstrap Validation for Few-Shot Heterogeneous Graph Neural Networks"""

    def __init__(
        self,
        model_class,
        data: HeteroData,
        args,
        target_node_type: str = "news",
        seed: int = 42,
    ):
        """
        Args:
            model_class: Model class to instantiate
            data: HeteroData graph
            args: Training arguments
            target_node_type: Target node type for classification
            seed: Random seed
        """
        self.model_class = model_class
        self.data = data
        self.args = args
        self.target_node_type = target_node_type
        self.seed = seed

        # Extract original train_labeled indices and labels
        self.original_train_labeled_mask = data[
            target_node_type
        ].train_labeled_mask.clone()
        self.train_labeled_indices = (
            torch.where(self.original_train_labeled_mask)[0].cpu().numpy()
        )
        self.train_labeled_labels = (
            data[target_node_type].y[self.original_train_labeled_mask].cpu().numpy()
        )

        print(f"Bootstrap Validation Setup:")
        print(f"  - Total labeled samples: {len(self.train_labeled_indices)}")
        print(f"  - Labels distribution: {np.bincount(self.train_labeled_labels)}")

        # Set random seed for reproducibility
        np.random.seed(self.seed)

    def create_bootstrap_split(
        self, val_ratio: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create stratified bootstrap training set and validation set"""

        # For very small datasets (few-shot), use stratified split instead of pure bootstrap
        min_val_samples = max(
            2, int(len(self.train_labeled_indices) * val_ratio)
        )  # At least 2 samples

        # Stratified sampling to ensure both classes are represented
        bootstrap_train_indices, val_indices = train_test_split(
            self.train_labeled_indices,
            test_size=min_val_samples,
            stratify=self.train_labeled_labels,
            random_state=np.random.randint(0, 10000),
        )

        # For very small training sets, add bootstrap sampling
        if len(bootstrap_train_indices) < 4:  # If less than 4 training samples
            # Sample with replacement to get at least 4 samples
            bootstrap_train_indices = np.random.choice(
                bootstrap_train_indices,
                size=max(4, len(bootstrap_train_indices)),
                replace=True,
            )

        return bootstrap_train_indices, val_indices

    def create_masks(
        self, train_indices: np.ndarray, val_indices: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create train and validation masks"""
        num_nodes = self.data[self.target_node_type].num_nodes
        device = self.data[self.target_node_type].x.device

        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        train_mask[train_indices] = True
        val_mask[val_indices] = True

        return train_mask, val_mask

    def train_single_bootstrap(
        self, bootstrap_idx: int, max_epochs: int = 100, patience: int = 20
    ) -> Dict:
        """Train model on a single bootstrap sample"""

        # Create bootstrap split
        train_indices, val_indices = self.create_bootstrap_split()
        train_mask, val_mask = self.create_masks(train_indices, val_indices)

        print(f"\n  Bootstrap {bootstrap_idx + 1}")
        print(f"    Train samples: {train_mask.sum().item()}")
        print(f"    Val samples: {val_mask.sum().item()}")

        # Create fresh model
        device = self.args.device if hasattr(self.args, "device") else "cuda"
        model = self.model_class(self.data, self.args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        # Temporarily modify data masks
        original_train_mask = self.data[
            self.target_node_type
        ].train_labeled_mask.clone()
        self.data[self.target_node_type].train_labeled_mask = train_mask

        # Training loop
        best_val_f1 = -1.0
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(max_epochs):
            # Train
            train_loss, train_acc, train_f1 = train_epoch(
                model, self.data, optimizer, criterion, self.target_node_type
            )

            # Validate
            model.eval()
            with torch.no_grad():
                out = model(self.data.x_dict, self.data.edge_index_dict)
                if isinstance(out, dict):
                    out_target = out[self.target_node_type]
                else:
                    out_target = out

                val_loss = criterion(
                    out_target[val_mask], self.data[self.target_node_type].y[val_mask]
                )
                pred = out_target[val_mask].argmax(dim=1)
                y_true = self.data[self.target_node_type].y[val_mask].cpu().numpy()
                y_pred = pred.cpu().numpy()

                val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

            # Enhanced early stopping: loss-based model selection with overfitting detection
            if val_loss.item() + EPSILON < best_val_loss:
                best_val_f1 = val_f1
                best_val_loss = val_loss.item()
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                print(
                    f"    Epoch {epoch:3d}: Train F1={train_f1:.4f}, Val Loss={val_loss.item():.4f}, Val F1={val_f1:.4f}"
                )

            # Early stopping: patience exceeded OR overfitting detected
            if patience_counter >= patience:
                print(
                    f"    Early stopping: patience ({patience}) exceeded at epoch {epoch}"
                )
                break
            elif val_loss.item() < OVERFIT_THRESHOLD:
                print(
                    f"    Early stopping: potential overfitting detected (val_loss={val_loss.item():.4f} < {OVERFIT_THRESHOLD}) at epoch {epoch}"
                )
                break

        # Restore original mask
        self.data[self.target_node_type].train_labeled_mask = original_train_mask

        # If no improvement, save current state
        if best_model_state is None:
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        return {
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
            "best_model_state": best_model_state,
            "epochs_trained": epoch + 1,
        }

    def run_bootstrap_validation(
        self, n_bootstrap: int = 10, max_epochs: int = 100, patience: int = 20
    ) -> Dict:
        """Run bootstrap validation"""

        print(f"\n{'='*60}")
        print(f"  Running Bootstrap Validation (n={n_bootstrap})")
        print(f"{'='*60}")

        set_seed(self.seed)

        bootstrap_results = []
        best_overall_f1 = -1.0
        best_model_state = None

        for bootstrap_idx in range(n_bootstrap):
            result = self.train_single_bootstrap(bootstrap_idx, max_epochs, patience)

            bootstrap_results.append(result)

            print(
                f"  Bootstrap {bootstrap_idx + 1} - Val F1: {result['best_val_f1']:.4f}"
            )

            # Track best model
            if result["best_val_f1"] > best_overall_f1:
                best_overall_f1 = result["best_val_f1"]
                best_model_state = result["best_model_state"]

        # Calculate statistics
        val_f1_scores = [result["best_val_f1"] for result in bootstrap_results]
        val_loss_scores = [result["best_val_loss"] for result in bootstrap_results]

        bootstrap_summary = {
            "method": "bootstrap",
            "n_bootstrap": n_bootstrap,
            "mean_val_f1": np.mean(val_f1_scores),
            "std_val_f1": np.std(val_f1_scores),
            "mean_val_loss": np.mean(val_loss_scores),
            "std_val_loss": np.std(val_loss_scores),
            "best_f1": best_overall_f1,
            "best_model_state": best_model_state,
            "individual_f1_scores": val_f1_scores,
            "all_bootstrap_models": [
                result["best_model_state"] for result in bootstrap_results
            ],
            "confidence_interval_95": np.percentile(val_f1_scores, [2.5, 97.5]),
        }

        print(f"\n{'='*60}")
        print(f"  Bootstrap Summary")
        print(f"{'='*60}")
        print(
            f"Mean Val F1: {bootstrap_summary['mean_val_f1']:.4f} ± {bootstrap_summary['std_val_f1']:.4f}"
        )
        print(
            f"95% CI: [{bootstrap_summary['confidence_interval_95'][0]:.4f}, {bootstrap_summary['confidence_interval_95'][1]:.4f}]"
        )
        print(f"Best Bootstrap F1: {best_overall_f1:.4f}")
        print(f"Individual F1 scores: {[f'{f:.4f}' for f in val_f1_scores]}")

        return bootstrap_summary

    def bootstrap_ensemble_prediction(self, bootstrap_results: Dict) -> Dict:
        """Use bootstrap ensemble for final prediction (simplified and stable)"""

        print(f"\n{'='*60}")
        print(f"  Bootstrap Ensemble Evaluation")
        print(f"{'='*60}")

        all_predictions = []
        all_probabilities = []

        device = self.args.device if hasattr(self.args, "device") else "cuda"

        # Get predictions from all bootstrap models
        for model_idx, model_state in enumerate(
            bootstrap_results["all_bootstrap_models"]
        ):
            model = self.model_class(self.data, self.args).to(device)
            model.load_state_dict(model_state)
            model.eval()

            with torch.no_grad():
                out = model(self.data.x_dict, self.data.edge_index_dict)
                if isinstance(out, dict):
                    out_target = out[self.target_node_type]
                else:
                    out_target = out

                test_mask = self.data[self.target_node_type].test_mask
                test_logits = out_target[test_mask]
                test_probs = torch.softmax(test_logits, dim=1)
                test_pred = test_logits.argmax(dim=1)

                all_predictions.append(test_pred.cpu().numpy())
                all_probabilities.append(test_probs.cpu().numpy())

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)  # (n_bootstrap, n_test_samples)
        all_probabilities = np.array(
            all_probabilities
        )  # (n_bootstrap, n_test_samples, n_classes)

        # Ensemble methods (including stable weighted ensemble)

        # Method 1: Majority voting
        ensemble_pred_voting = mode(all_predictions, axis=0)[0].flatten()

        # Method 2: Average probabilities
        avg_probabilities = np.mean(all_probabilities, axis=0)
        ensemble_pred_prob = np.argmax(avg_probabilities, axis=1)

        # Method 3: Stable weighted ensemble (by validation performance)
        val_f1_scores = np.array(bootstrap_results["individual_f1_scores"])
        # Add stability: clip weights to avoid extreme values and ensure positive weights
        val_f1_scores = np.clip(
            val_f1_scores, 0.1, 1.0
        )  # Avoid zero or negative weights
        weights = val_f1_scores / np.sum(val_f1_scores)
        # Additional stability: use softmax-like transformation to smooth weights
        weights = np.exp(weights * 2) / np.sum(np.exp(weights * 2))
        weighted_probabilities = np.average(all_probabilities, axis=0, weights=weights)
        ensemble_pred_weighted = np.argmax(weighted_probabilities, axis=1)

        # Get true labels
        test_mask = self.data[self.target_node_type].test_mask
        y_true = self.data[self.target_node_type].y[test_mask].cpu().numpy()

        # Evaluate ensemble methods
        voting_acc = accuracy_score(y_true, ensemble_pred_voting)
        voting_f1 = f1_score(y_true, ensemble_pred_voting, average="macro")

        prob_acc = accuracy_score(y_true, ensemble_pred_prob)
        prob_f1 = f1_score(y_true, ensemble_pred_prob, average="macro")

        weighted_acc = accuracy_score(y_true, ensemble_pred_weighted)
        weighted_f1 = f1_score(y_true, ensemble_pred_weighted, average="macro")

        # Individual model performance on test set
        individual_test_f1s = []
        for i in range(len(all_predictions)):
            f1 = f1_score(y_true, all_predictions[i], average="macro")
            individual_test_f1s.append(f1)

        ensemble_results = {
            "voting_accuracy": voting_acc,
            "voting_f1": voting_f1,
            "prob_avg_accuracy": prob_acc,
            "prob_avg_f1": prob_f1,
            "weighted_accuracy": weighted_acc,
            "weighted_f1": weighted_f1,
            "individual_test_f1s": individual_test_f1s,
            "mean_individual_f1": np.mean(individual_test_f1s),
            "std_individual_f1": np.std(individual_test_f1s),
            "ensemble_weights": weights.tolist(),
        }

        print(f"Bootstrap Ensemble Results:")
        print(f"  Majority Voting     - Acc: {voting_acc:.4f}, F1: {voting_f1:.4f}")
        print(f"  Probability Average - Acc: {prob_acc:.4f}, F1: {prob_f1:.4f}")
        print(f"  Weighted Ensemble   - Acc: {weighted_acc:.4f}, F1: {weighted_f1:.4f}")
        print(
            f"  Individual Models   - F1: {np.mean(individual_test_f1s):.4f} ± {np.std(individual_test_f1s):.4f}"
        )
        print(f"  Ensemble Weights: {[f'{w:.3f}' for w in weights]}")

        return ensemble_results

    def single_best_evaluation(self, bootstrap_results: Dict) -> Dict:
        """Evaluate single best bootstrap model on test set"""

        print(f"\n{'='*60}")
        print(f"  Single Best Bootstrap Model Evaluation")
        print(f"{'='*60}")

        device = self.args.device if hasattr(self.args, "device") else "cuda"

        # Create fresh model and load best bootstrap weights
        model = self.model_class(self.data, self.args).to(device)
        model.load_state_dict(bootstrap_results["best_model_state"])

        # Restore original train_labeled_mask for full data training
        self.data[self.target_node_type].train_labeled_mask = (
            self.original_train_labeled_mask
        )

        # Retrain on full labeled set with proper early stopping
        print("Retraining best model on full labeled set...")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        patience = 10  # Shorter patience for fine-tuning

        for epoch in range(50):  # More epochs for proper convergence
            # Train on full labeled set
            train_loss, train_acc, train_f1 = train_epoch(
                model, self.data, optimizer, criterion, self.target_node_type
            )

            # Validate on the same labeled set (few-shot scenario)
            val_loss, val_acc, val_f1 = evaluate(
                model, self.data, "train_labeled_mask", criterion, self.target_node_type
            )

            # Early stopping based on validation loss
            if val_loss + EPSILON < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(
                    f"    Epoch {epoch:2d}: Train F1={train_f1:.4f}, Val Loss={val_loss:.4f}"
                )

            # Early stopping conditions
            if patience_counter >= patience:
                print(f"    Early stopping: patience exceeded at epoch {epoch}")
                break
            elif val_loss < OVERFIT_THRESHOLD:
                print(
                    f"    Early stopping: overfitting detected (val_loss={val_loss:.4f}) at epoch {epoch}"
                )
                break

        # Load best state if improvement occurred
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"    Loaded best model state (val_loss={best_val_loss:.4f})")

        # Final test evaluation
        test_loss, test_acc, test_f1 = evaluate(
            model, self.data, "test_mask", criterion, self.target_node_type
        )

        results = {
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_loss": test_loss,
        }

        print(f"Single Best Model Results:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")

        return results


def create_results_table(
    results: Dict, bootstrap_results: Dict, ensemble_results: Dict, model_name_fs: str
) -> str:
    """Create a clean table-formatted summary of bootstrap results"""

    table_lines = []
    table_lines.append("=" * 80)
    table_lines.append(f"Bootstrap Results Summary - {model_name_fs}")
    table_lines.append("=" * 80)
    table_lines.append("")

    # Method comparison table
    table_lines.append("METHOD COMPARISON:")
    table_lines.append("-" * 50)
    table_lines.append(f"{'Method':<25} {'Accuracy':<12} {'F1 Score':<12}")
    table_lines.append("-" * 50)
    table_lines.append(
        f"{'Original':<25} {results['original']['test_acc']:<12.4f} {results['original']['test_f1']:<12.4f}"
    )
    table_lines.append(
        f"{'Bootstrap Best':<25} {results['bootstrap_single']['test_accuracy']:<12.4f} {results['bootstrap_single']['test_f1']:<12.4f}"
    )
    table_lines.append(
        f"{'Bootstrap Voting':<25} {ensemble_results['voting_accuracy']:<12.4f} {ensemble_results['voting_f1']:<12.4f}"
    )
    table_lines.append(
        f"{'Bootstrap Prob Avg':<25} {ensemble_results['prob_avg_accuracy']:<12.4f} {ensemble_results['prob_avg_f1']:<12.4f}"
    )
    table_lines.append(
        f"{'Bootstrap Weighted':<25} {ensemble_results['weighted_accuracy']:<12.4f} {ensemble_results['weighted_f1']:<12.4f} ⭐"
    )
    table_lines.append("-" * 50)
    table_lines.append("")

    # Bootstrap validation statistics
    table_lines.append("BOOTSTRAP VALIDATION STATISTICS:")
    table_lines.append("-" * 40)
    table_lines.append(
        f"Number of Bootstrap Samples: {bootstrap_results['n_bootstrap']}"
    )
    table_lines.append(
        f"Mean Validation F1: {bootstrap_results['mean_val_f1']:.4f} ± {bootstrap_results['std_val_f1']:.4f}"
    )
    table_lines.append(
        f"95% Confidence Interval: [{bootstrap_results['confidence_interval_95'][0]:.4f}, {bootstrap_results['confidence_interval_95'][1]:.4f}]"
    )
    table_lines.append(
        f"Individual Bootstrap F1s: {[f'{f:.3f}' for f in bootstrap_results['individual_f1_scores']]}"
    )
    table_lines.append(
        f"Ensemble Weights: {[f'{w:.3f}' for w in ensemble_results['ensemble_weights']]}"
    )
    table_lines.append("")

    # Best method recommendation
    best_f1 = max(
        results["original"]["test_f1"],
        results["bootstrap_single"]["test_f1"],
        ensemble_results["voting_f1"],
        ensemble_results["prob_avg_f1"],
        ensemble_results["weighted_f1"],
    )
    if best_f1 == results["original"]["test_f1"]:
        best_method = "Original"
    elif best_f1 == results["bootstrap_single"]["test_f1"]:
        best_method = "Bootstrap Best"
    elif best_f1 == ensemble_results["voting_f1"]:
        best_method = "Bootstrap Voting"
    elif best_f1 == ensemble_results["prob_avg_f1"]:
        best_method = "Bootstrap Prob Avg"
    else:
        best_method = "Bootstrap Weighted (RECOMMENDED)"

    table_lines.append("RECOMMENDATION:")
    table_lines.append("-" * 20)
    table_lines.append(f"Best performing method: {best_method} (F1: {best_f1:.4f})")
    table_lines.append("")
    table_lines.append("=" * 80)

    return "\n".join(table_lines)


def save_bootstrap_results(
    results: Dict,
    bootstrap_results: Dict,
    ensemble_results: Dict,
    args,
    output_dir: str,
    model_name_fs: str,
) -> None:
    """Save bootstrap validation results, metrics, and comprehensive report"""

    os.makedirs(output_dir, exist_ok=True)

    # Helper function for JSON serialization
    def default_serializer(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        try:
            return json.JSONEncoder().encode(obj)
        except TypeError:
            return str(obj)

    # Comprehensive metrics report
    comprehensive_results = {
        "experiment_info": {
            "model_name": model_name_fs,
            "graph_path": args.graph_path,
            "model_type": args.model,
            "target_node_type": args.target_node_type,
            "args": vars(args),
        },
        "method_comparison": {
            "original_single_training": {
                "test_accuracy": results["original"]["test_acc"],
                "test_f1": results["original"]["test_f1"],
                "method_description": "Single training with early stopping (matching train_hetero_graph.py)",
            },
            "bootstrap_single_best": {
                "test_accuracy": results["bootstrap_single"]["test_accuracy"],
                "test_f1": results["bootstrap_single"]["test_f1"],
                "test_loss": results["bootstrap_single"]["test_loss"],
                "method_description": "Best bootstrap model retrained on full labeled set",
            },
            "bootstrap_ensemble_voting": {
                "test_accuracy": ensemble_results["voting_accuracy"],
                "test_f1": ensemble_results["voting_f1"],
                "method_description": "Majority voting across bootstrap models",
            },
            "bootstrap_ensemble_prob_avg": {
                "test_accuracy": ensemble_results["prob_avg_accuracy"],
                "test_f1": ensemble_results["prob_avg_f1"],
                "method_description": "Average probabilities across bootstrap models",
            },
            "bootstrap_ensemble_weighted": {
                "test_accuracy": ensemble_results["weighted_accuracy"],
                "test_f1": ensemble_results["weighted_f1"],
                "method_description": "Weighted ensemble by validation performance (recommended)",
            },
        },
        "bootstrap_validation_statistics": {
            "n_bootstrap": bootstrap_results["n_bootstrap"],
            "mean_val_f1": bootstrap_results["mean_val_f1"],
            "std_val_f1": bootstrap_results["std_val_f1"],
            "mean_val_loss": bootstrap_results["mean_val_loss"],
            "std_val_loss": bootstrap_results["std_val_loss"],
            "confidence_interval_95": bootstrap_results["confidence_interval_95"],
            "individual_f1_scores": bootstrap_results["individual_f1_scores"],
        },
        "ensemble_analysis": {
            "individual_models_f1_mean": ensemble_results["mean_individual_f1"],
            "individual_models_f1_std": ensemble_results["std_individual_f1"],
            "individual_test_f1s": ensemble_results["individual_test_f1s"],
            "ensemble_weights": ensemble_results["ensemble_weights"],
        },
    }

    # Add confusion matrices for each method
    def compute_confusion_matrix(y_true, y_pred, method_name):
        try:
            cm = confusion_matrix(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

            return {
                "confusion_matrix": cm.tolist(),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        except Exception as e:
            print(f"Warning: Could not compute confusion matrix for {method_name}: {e}")
            return None

    # Get test labels for confusion matrices
    device = args.device if hasattr(args, "device") else "cuda"
    data = load_hetero_graph(args.graph_path, device, args.target_node_type)
    test_mask = data[args.target_node_type].test_mask
    y_true = data[args.target_node_type].y[test_mask].cpu().numpy()

    # Recreate predictions for confusion matrices
    validator = FewShotBootstrapValidator(
        model_class=lambda data, args: get_model(args.model, data, args),
        data=data,
        args=args,
        target_node_type=args.target_node_type,
        seed=args.seed,
    )

    # Get ensemble predictions again for confusion matrices
    all_predictions = []
    all_probabilities = []

    for model_state in bootstrap_results["all_bootstrap_models"]:
        model = validator.model_class(data, args).to(device)
        model.load_state_dict(model_state)
        model.eval()

        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            if isinstance(out, dict):
                out_target = out[args.target_node_type]
            else:
                out_target = out

            test_logits = out_target[test_mask]
            test_probs = torch.softmax(test_logits, dim=1)
            test_pred = test_logits.argmax(dim=1)

            all_predictions.append(test_pred.cpu().numpy())
            all_probabilities.append(test_probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # Ensemble predictions (with stable weighted ensemble)
    ensemble_pred_voting = mode(all_predictions, axis=0)[0].flatten()
    avg_probabilities = np.mean(all_probabilities, axis=0)
    ensemble_pred_prob = np.argmax(avg_probabilities, axis=1)

    # Stable weighted ensemble
    val_f1_scores = np.array(bootstrap_results["individual_f1_scores"])
    val_f1_scores = np.clip(val_f1_scores, 0.1, 1.0)
    weights = val_f1_scores / np.sum(val_f1_scores)
    weights = np.exp(weights * 2) / np.sum(np.exp(weights * 2))
    weighted_probabilities = np.average(all_probabilities, axis=0, weights=weights)
    ensemble_pred_weighted = np.argmax(weighted_probabilities, axis=1)

    # Add confusion matrices to results
    comprehensive_results["detailed_metrics"] = {
        "bootstrap_ensemble_voting": compute_confusion_matrix(
            y_true, ensemble_pred_voting, "voting"
        ),
        "bootstrap_ensemble_prob_avg": compute_confusion_matrix(
            y_true, ensemble_pred_prob, "prob_avg"
        ),
        "bootstrap_ensemble_weighted": compute_confusion_matrix(
            y_true, ensemble_pred_weighted, "weighted"
        ),
    }

    # Save comprehensive results
    results_path = os.path.join(output_dir, "bootstrap_comprehensive_metrics.json")
    try:
        with open(results_path, "w") as f:
            json.dump(comprehensive_results, f, indent=4, default=default_serializer)
        print(f"Comprehensive bootstrap results saved to {results_path}")
    except Exception as e:
        print(f"Error saving comprehensive results: {e}")

    # Save table-formatted results
    table_path = os.path.join(output_dir, "bootstrap_results_table.txt")
    try:
        table_content = create_results_table(
            results, bootstrap_results, ensemble_results, model_name_fs
        )
        with open(table_path, "w") as f:
            f.write(table_content)
        print(f"Table-formatted results saved to {table_path}")
        # Also print the table to console
        print("\n" + table_content)
    except Exception as e:
        print(f"Error saving table results: {e}")

    # Save summary metrics (compatible with analysis scripts)
    summary_results = {
        "args": vars(args),
        "model_name": model_name_fs,
        "bootstrap_summary": {
            "mean_val_f1": bootstrap_results["mean_val_f1"],
            "std_val_f1": bootstrap_results["std_val_f1"],
            "confidence_interval_95": bootstrap_results["confidence_interval_95"],
            "n_bootstrap": bootstrap_results["n_bootstrap"],
        },
        "test_metrics": {
            "original_method": results["original"],
            "bootstrap_single": results["bootstrap_single"],
            "bootstrap_voting": {
                "accuracy": ensemble_results["voting_accuracy"],
                "f1_score": ensemble_results["voting_f1"],
            },
            "bootstrap_prob_avg": {
                "accuracy": ensemble_results["prob_avg_accuracy"],
                "f1_score": ensemble_results["prob_avg_f1"],
            },
            "bootstrap_weighted": {
                "accuracy": ensemble_results["weighted_accuracy"],
                "f1_score": ensemble_results["weighted_f1"],
            },
        },
    }

    summary_path = os.path.join(output_dir, "metrics.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(summary_results, f, indent=4, default=default_serializer)
        print(f"Summary metrics saved to {summary_path}")
    except Exception as e:
        print(f"Error saving summary metrics: {e}")

    # Create visualization plots
    plot_path = os.path.join(output_dir, "bootstrap_analysis.png")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Bootstrap Analysis - {model_name_fs}", fontsize=16)

        # Plot 1: F1 Score Comparison
        methods = [
            "Original",
            "Bootstrap\nSingle",
            "Bootstrap\nVoting",
            "Bootstrap\nProb Avg",
            "Bootstrap\nWeighted",
        ]
        f1_scores = [
            results["original"]["test_f1"],
            results["bootstrap_single"]["test_f1"],
            ensemble_results["voting_f1"],
            ensemble_results["prob_avg_f1"],
            ensemble_results["weighted_f1"],
        ]

        axes[0, 0].bar(
            methods, f1_scores, color=["blue", "orange", "green", "red", "purple"]
        )
        axes[0, 0].set_title("F1 Score Comparison")
        axes[0, 0].set_ylabel("F1 Score")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Plot 2: Bootstrap Validation F1 Distribution
        individual_f1s = bootstrap_results["individual_f1_scores"]
        axes[0, 1].hist(
            individual_f1s,
            bins=min(10, len(individual_f1s)),
            alpha=0.7,
            color="skyblue",
        )
        axes[0, 1].axvline(
            bootstrap_results["mean_val_f1"],
            color="red",
            linestyle="--",
            label=f'Mean: {bootstrap_results["mean_val_f1"]:.3f}',
        )
        axes[0, 1].set_title("Bootstrap Validation F1 Distribution")
        axes[0, 1].set_xlabel("Validation F1 Score")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()

        # Plot 3: Individual Bootstrap Model Performance
        axes[1, 0].plot(
            range(1, len(individual_f1s) + 1), individual_f1s, "o-", color="green"
        )
        axes[1, 0].axhline(
            bootstrap_results["mean_val_f1"], color="red", linestyle="--", alpha=0.7
        )
        axes[1, 0].fill_between(
            range(1, len(individual_f1s) + 1),
            [bootstrap_results["confidence_interval_95"][0]] * len(individual_f1s),
            [bootstrap_results["confidence_interval_95"][1]] * len(individual_f1s),
            alpha=0.2,
            color="gray",
            label="95% CI",
        )
        axes[1, 0].set_title("Individual Bootstrap Model Performance")
        axes[1, 0].set_xlabel("Bootstrap Model")
        axes[1, 0].set_ylabel("Validation F1 Score")
        axes[1, 0].legend()

        # Plot 4: Model Performance Summary
        perf_data = [
            results["original"]["test_f1"],
            results["bootstrap_single"]["test_f1"],
        ]
        perf_labels = ["Original", "Bootstrap Best"]

        axes[1, 1].barh(perf_labels, perf_data, color=["blue", "orange"])
        axes[1, 1].set_title("Method Performance Summary")
        axes[1, 1].set_xlabel("F1 Score")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Bootstrap analysis plots saved to {plot_path}")
    except Exception as e:
        print(f"Error saving plots: {e}")


def compare_bootstrap_vs_original(args):
    """Compare bootstrap method vs original method"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    data = load_hetero_graph(args.graph_path, device, args.target_node_type)

    # Create output directory structure (similar to train_hetero_graph.py)
    try:
        parts = args.graph_path.split(os.sep)
        scenario_filename = parts[-2]
        dataset_name = parts[-3]
    except IndexError:
        scenario_filename = "unknown_scenario"
        dataset_name = "unknown_dataset"

    # Create output directory matching train_hetero_graph.py structure
    model_name_fs = f"Bootstrap_{args.model}_{dataset_name}_{scenario_filename}"
    output_dir = os.path.join(
        "results_hetero", f"{args.model}_bootstrap", dataset_name, scenario_filename
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Bootstrap Experiment Configuration ---")
    print(f"Model:           {args.model}")
    print(f"Dataset:         {dataset_name}")
    print(f"Scenario:        {scenario_filename}")
    print(f"Output Dir:      {output_dir}")
    print(f"Graph Path:      {args.graph_path}")

    results = {}

    # Method 1: Original approach (matching train_hetero_graph.py exactly)
    print(f"\n{'='*60}")
    print(f"  Method 1: Original Single Training")
    print(f"{'='*60}")

    model_orig = get_model(args.model, data, args).to(device)
    optimizer = torch.optim.Adam(
        model_orig.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # Use the same training logic as train_hetero_graph.py

    with tempfile.TemporaryDirectory() as temp_dir:
        training_history = train(
            model_orig,
            data,
            optimizer,
            criterion,
            args,
            temp_dir,
            "original_comparison",
        )

        # Load best model and evaluate
        model_path = os.path.join(temp_dir, "original_comparison_best.pt")
        if os.path.exists(model_path):
            model_orig.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=False)
            )
            print("Loaded best model weights for final evaluation")

        test_loss, test_acc, test_f1 = evaluate(
            model_orig, data, "test_mask", criterion, args.target_node_type
        )
        results["original"] = {"test_f1": test_f1, "test_acc": test_acc}
        print(f"Original method F1: {test_f1:.4f}")

    # Method 2: Bootstrap validation
    print(f"\n{'='*60}")
    print(f"  Method 2: Bootstrap Validation")
    print(f"{'='*60}")

    validator = FewShotBootstrapValidator(
        model_class=lambda data, args: get_model(args.model, data, args),
        data=data,
        args=args,
        target_node_type=args.target_node_type,
        seed=args.seed,
    )

    # Run bootstrap validation (increased for better statistics)
    bootstrap_results = validator.run_bootstrap_validation(
        n_bootstrap=8,  # Increased for better statistics while maintaining stability
        max_epochs=args.n_epochs,
        patience=args.patience,
    )

    # Bootstrap ensemble
    ensemble_results = validator.bootstrap_ensemble_prediction(bootstrap_results)

    # Single best model
    single_results = validator.single_best_evaluation(bootstrap_results)

    results["bootstrap_single"] = single_results
    results["bootstrap_ensemble"] = ensemble_results

    # Save comprehensive results and generate reports
    save_bootstrap_results(
        results, bootstrap_results, ensemble_results, args, output_dir, model_name_fs
    )

    # Final comparison
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"Original Method:               F1 = {results['original']['test_f1']:.4f}")
    print(
        f"Bootstrap (Best Model):       F1 = {results['bootstrap_single']['test_f1']:.4f}"
    )
    print(
        f"Bootstrap (Voting):           F1 = {results['bootstrap_ensemble']['voting_f1']:.4f}"
    )
    print(
        f"Bootstrap (Prob Average):     F1 = {results['bootstrap_ensemble']['prob_avg_f1']:.4f}"
    )
    print(
        f"Bootstrap (Weighted):         F1 = {results['bootstrap_ensemble']['weighted_f1']:.4f} ⭐ RECOMMENDED"
    )
    print(f"")
    print(
        f"Bootstrap Validation: {bootstrap_results['mean_val_f1']:.4f} ± {bootstrap_results['std_val_f1']:.4f}"
    )
    print(
        f"95% Confidence Interval: [{bootstrap_results['confidence_interval_95'][0]:.4f}, {bootstrap_results['confidence_interval_95'][1]:.4f}]"
    )
    print(f"\n--- Results and plots saved to: {output_dir} ---")

    return results


def main():

    args = parse_arguments()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run bootstrap experiment
    results = compare_bootstrap_vs_original(args)

    print(f"\n{'='*60}")
    print(f"  Bootstrap Experiment Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
