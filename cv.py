import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import HeteroData
import json
from typing import Dict, List, Tuple
from train_hetero_graph import get_model, train_epoch, evaluate, set_seed

class FewShotCrossValidator:
    """Cross-Validation for Few-Shot Heterogeneous Graph Neural Networks"""
    
    def __init__(self, 
                 model_class,
                 data: HeteroData,
                 args,
                 k_folds: int = 3,
                 target_node_type: str = "news",
                 seed: int = 42):
        """
        Args:
            model_class: Model class to instantiate
            data: HeteroData graph
            args: Training arguments
            k_folds: Number of folds for cross-validation
            target_node_type: Target node type for classification
            seed: Random seed
        """
        self.model_class = model_class
        self.data = data
        self.args = args
        self.k_folds = k_folds
        self.target_node_type = target_node_type
        self.seed = seed
        
        # Extract original train_labeled indices and labels
        self.original_train_labeled_mask = data[target_node_type].train_labeled_mask.clone()
        self.train_labeled_indices = torch.where(self.original_train_labeled_mask)[0].cpu().numpy()
        self.train_labeled_labels = data[target_node_type].y[self.original_train_labeled_mask].cpu().numpy()
        
        print(f"Cross-Validation Setup:")
        print(f"  - Total labeled samples: {len(self.train_labeled_indices)}")
        print(f"  - K-folds: {k_folds}")
        print(f"  - Labels distribution: {np.bincount(self.train_labeled_labels)}")

    def create_fold_masks(self, train_indices: np.ndarray, val_indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create train and validation masks for a specific fold"""
        num_nodes = self.data[self.target_node_type].num_nodes
        
        # Map back to global indices
        fold_train_global_indices = self.train_labeled_indices[train_indices]
        fold_val_global_indices = self.train_labeled_indices[val_indices]
        
        # Create masks
        fold_train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.data[self.target_node_type].x.device)
        fold_val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.data[self.target_node_type].x.device)
        
        fold_train_mask[fold_train_global_indices] = True
        fold_val_mask[fold_val_global_indices] = True
        
        return fold_train_mask, fold_val_mask

    def train_single_fold(self, 
                         fold_train_mask: torch.Tensor, 
                         fold_val_mask: torch.Tensor, 
                         fold_idx: int,
                         max_epochs: int = 100,
                         patience: int = 20) -> Dict:
        """Train model on a single fold"""
        
        print(f"\n  Training Fold {fold_idx + 1}/{self.k_folds}")
        print(f"    Train samples: {fold_train_mask.sum().item()}")
        print(f"    Val samples: {fold_val_mask.sum().item()}")
        
        # Create fresh model
        model = self.model_class(self.data, self.args).to(self.args.device if hasattr(self.args, 'device') else 'cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Temporarily modify data masks
        original_train_mask = self.data[self.target_node_type].train_labeled_mask.clone()
        self.data[self.target_node_type].train_labeled_mask = fold_train_mask
        
        # Training loop
        best_val_f1 = -1.0
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_losses, val_losses = [], []
        train_f1s, val_f1s = [], []
        
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
                
                val_loss = criterion(out_target[fold_val_mask], self.data[self.target_node_type].y[fold_val_mask])
                pred = out_target[fold_val_mask].argmax(dim=1)
                y_true = self.data[self.target_node_type].y[fold_val_mask].cpu().numpy()
                y_pred = pred.cpu().numpy()
                
                from sklearn.metrics import f1_score, accuracy_score
                val_acc = accuracy_score(y_true, y_pred)
                val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss.item())
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            
            # Early stopping based on validation F1
            if val_f1 > best_val_f1 or val_loss.item() < best_val_loss:
                best_val_f1 = val_f1
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch:3d}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, Val Loss={val_loss:.4f}")
            
            if patience_counter >= patience:
                best_model_state = model.state_dict().copy()
                print(f"    Early stopping at epoch {epoch}")
                break
        
        # Restore original mask
        self.data[self.target_node_type].train_labeled_mask = original_train_mask
        
        return {
            'best_val_f1': best_val_f1,
            'best_val_loss': best_val_loss,
            'best_model_state': best_model_state,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_f1s': train_f1s,
            'val_f1s': val_f1s,
            'epochs_trained': epoch + 1
        }

    def run_cross_validation(self, 
                           max_epochs: int = 100, 
                           patience: int = 20,
                           save_results: bool = True,
                           output_dir: str = "cv_results",
                           ensemble_prediction: bool = True) -> Dict:
        """Run complete cross-validation"""
        
        print(f"\n{'='*60}")
        print(f"  Running {self.k_folds}-Fold Cross-Validation")
        print(f"{'='*60}")
        
        set_seed(self.seed)
        
        # Stratified K-Fold to ensure balanced label distribution
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        
        fold_results = []
        best_overall_f1 = -1.0
        best_overall_loss = float('inf')
        best_fold_idx = -1
        best_model_state = None
        
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(self.train_labeled_indices, self.train_labeled_labels)):
            # Create fold masks
            fold_train_mask, fold_val_mask = self.create_fold_masks(train_indices, val_indices)
            
            # Train single fold
            fold_result = self.train_single_fold(
                fold_train_mask, fold_val_mask, fold_idx, max_epochs, patience
            )
            
            fold_results.append(fold_result)
            
            print(f"  Fold {fold_idx + 1} Results:")
            print(f"    Best Val F1: {fold_result['best_val_f1']:.4f}")
            print(f"    Best Val Loss: {fold_result['best_val_loss']:.4f}")
            print(f"    Epochs Trained: {fold_result['epochs_trained']}")
            
            # Track best fold
            if fold_result['best_val_f1'] > best_overall_f1 or fold_result['best_val_loss'] < best_overall_loss:
                best_overall_f1 = fold_result['best_val_f1']
                best_overall_loss = fold_result['best_val_loss']
                best_fold_idx = fold_idx
                best_model_state = fold_result['best_model_state']

        # Calculate cross-validation statistics
        val_f1_scores = [result['best_val_f1'] for result in fold_results]
        val_loss_scores = [result['best_val_loss'] for result in fold_results]
        
        cv_results = {
            'mean_val_f1': np.mean(val_f1_scores),
            'std_val_f1': np.std(val_f1_scores),
            'mean_val_loss': np.mean(val_loss_scores),
            'std_val_loss': np.std(val_loss_scores),
            'best_fold_idx': best_fold_idx,
            'best_fold_f1': best_overall_f1,
            'best_model_state': best_model_state,
            'fold_results': fold_results,
            'individual_f1_scores': val_f1_scores,
            'individual_loss_scores': val_loss_scores,
            'all_fold_models': [result['best_model_state'] for result in fold_results]  # 新增
        }
        
        print(f"\n{'='*60}")
        print(f"  Cross-Validation Summary")
        print(f"{'='*60}")
        print(f"Mean Val F1: {cv_results['mean_val_f1']:.4f} ± {cv_results['std_val_f1']:.4f}")
        print(f"Mean Val Loss: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}")
        print(f"Best Fold: {best_fold_idx + 1} (F1: {best_overall_f1:.4f})")
        print(f"Individual F1 scores: {[f'{f:.4f}' for f in val_f1_scores]}")
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
            # Helper to convert to JSON-serializable
            def to_serializable(val):
                if isinstance(val, torch.Tensor):
                    return val.item() if val.numel() == 1 else val.tolist()
                if isinstance(val, np.ndarray):
                    return val.tolist()
                if isinstance(val, (float, int, str, bool)) or val is None:
                    return val
                if isinstance(val, dict):
                    return {k: to_serializable(v) for k, v in val.items()}
                if isinstance(val, list):
                    return [to_serializable(v) for v in val]
                return str(val)

            # Save CV summary (without model states)
            summary = {k: v for k, v in cv_results.items() if k not in ['best_model_state', 'fold_results']}
            # Add simplified fold results
            summary['fold_summaries'] = [
                {
                    'best_val_f1': r['best_val_f1'],
                    'best_val_loss': r['best_val_loss'],
                    'epochs_trained': r['epochs_trained']
                } for r in fold_results
            ]
            
            with open(os.path.join(output_dir, 'cv_summary.json'), 'w') as f:
                json.dump(to_serializable(summary), f, indent=2)
            
            # Save best model
            if best_model_state is not None:
                torch.save(best_model_state, os.path.join(output_dir, 'best_cv_model.pt'))
            
            print(f"Results saved to {output_dir}")
        
        return cv_results

    def ensemble_prediction_on_test(self, cv_results: Dict) -> Dict:
        """Use ensemble of all CV models for final prediction"""
        
        print(f"\n{'='*60}")
        print(f"  Ensemble Evaluation on Test Set")
        print(f"{'='*60}")
        
        all_predictions = []
        all_probabilities = []
        
        # Get predictions from all fold models
        for fold_idx, model_state in enumerate(cv_results['all_fold_models']):
            model = self.model_class(self.data, self.args).to(self.args.device if hasattr(self.args, 'device') else 'cuda')
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
        
        # Ensemble methods
        all_predictions = np.array(all_predictions)  # (n_folds, n_test_samples)
        all_probabilities = np.array(all_probabilities)  # (n_folds, n_test_samples, n_classes)
        
        # Method 1: Majority voting
        from scipy.stats import mode
        ensemble_pred_voting = mode(all_predictions, axis=0)[0].flatten()
        
        # Method 2: Average probabilities
        avg_probabilities = np.mean(all_probabilities, axis=0)
        ensemble_pred_prob = np.argmax(avg_probabilities, axis=1)
        
        # Get true labels
        test_mask = self.data[self.target_node_type].test_mask
        y_true = self.data[self.target_node_type].y[test_mask].cpu().numpy()
        
        # Evaluate both methods
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        # Voting results
        voting_acc = accuracy_score(y_true, ensemble_pred_voting)
        voting_f1 = f1_score(y_true, ensemble_pred_voting, average='macro')
        
        # Probability averaging results  
        prob_acc = accuracy_score(y_true, ensemble_pred_prob)
        prob_f1 = f1_score(y_true, ensemble_pred_prob, average='macro')
        
        # Individual fold results on test
        individual_test_f1s = []
        for fold_idx in range(len(all_predictions)):
            fold_f1 = f1_score(y_true, all_predictions[fold_idx], average='macro')
            individual_test_f1s.append(fold_f1)
        
        ensemble_results = {
            'voting_accuracy': voting_acc,
            'voting_f1': voting_f1,
            'prob_avg_accuracy': prob_acc,
            'prob_avg_f1': prob_f1,
            'individual_test_f1s': individual_test_f1s,
            'mean_individual_f1': np.mean(individual_test_f1s),
            'std_individual_f1': np.std(individual_test_f1s),
            'cv_mean_val_f1': cv_results['mean_val_f1'],
            'cv_std_val_f1': cv_results['std_val_f1']
        }
        
        print(f"Ensemble Results:")
        print(f"  Majority Voting    - Acc: {voting_acc:.4f}, F1: {voting_f1:.4f}")
        print(f"  Probability Average - Acc: {prob_acc:.4f}, F1: {prob_f1:.4f}")
        print(f"  Individual Models  - F1: {np.mean(individual_test_f1s):.4f} ± {np.std(individual_test_f1s):.4f}")
        print(f"  CV Validation      - F1: {cv_results['mean_val_f1']:.4f} ± {cv_results['std_val_f1']:.4f}")
        
        return ensemble_results

    def final_evaluation_on_test(self, cv_results: Dict) -> Dict:
        """Evaluate best CV model on test set"""
        
        print(f"\n{'='*60}")
        print(f"  Single Best Model Evaluation on Test Set")
        print(f"{'='*60}")
        
        # Load best model
        model = self.model_class(self.data, self.args).to(self.args.device if hasattr(self.args, 'device') else 'cuda')
        model.load_state_dict(cv_results['best_model_state'])
        
        # Retrain on full labeled set for final evaluation
        print("Retraining best model on full labeled set...")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Restore original train_labeled_mask
        self.data[self.target_node_type].train_labeled_mask = self.original_train_labeled_mask
        
        # Quick fine-tuning (optional)
        for epoch in range(20):  # Few epochs to adapt to full labeled set
            train_loss, train_acc, train_f1 = train_epoch(
                model, self.data, optimizer, criterion, self.target_node_type
            )
        
        # Final test evaluation
        test_loss, test_acc, test_f1 = evaluate(model, self.data, 'test_mask', criterion, self.target_node_type)
        
        final_results = {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_loss': test_loss,
            'cv_mean_val_f1': cv_results['mean_val_f1'],
            'cv_std_val_f1': cv_results['std_val_f1']
        }
        
        print(f"Single Best Model Test Results:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  CV Validation F1: {cv_results['mean_val_f1']:.4f} ± {cv_results['std_val_f1']:.4f}")
        
        return final_results


def adaptive_k_fold_selection(labeled_samples: int) -> int:
    """Automatically select optimal k_folds based on sample size"""
    if labeled_samples <= 12:      # 3-6 shot
        return 3
    elif labeled_samples <= 20:    # 7-10 shot  
        return 4
    elif labeled_samples <= 32:    # 11-16 shot
        return 5
    else:
        return 5  # 最多 5-fold


def compare_cv_vs_original(args):
    """Compare CV method vs original single-split method"""
    
    from train_hetero_graph import load_hetero_graph, get_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_hetero_graph(args.graph_path, device, args.target_node_type)
    
    # Count labeled samples
    labeled_samples = data[args.target_node_type].train_labeled_mask.sum().item()
    optimal_k = adaptive_k_fold_selection(labeled_samples)
    
    print(f"Labeled samples: {labeled_samples}")
    print(f"Optimal K-folds: {optimal_k}")
    
    results = {}
    
    # Method 1: Original approach (using same data for train and val)
    print(f"\n{'='*60}")
    print(f"  Method 1: Original Single-Split")
    print(f"{'='*60}")
    
    model_orig = get_model(args.model, data, args).to(device)
    optimizer = torch.optim.Adam(model_orig.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(args.n_epochs):
        train_loss, train_acc, train_f1 = train_epoch(model_orig, data, optimizer, criterion, args.target_node_type)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train F1={train_f1:.4f}")
    
    test_loss, test_acc, test_f1 = evaluate(model_orig, data, 'test_mask', criterion, args.target_node_type)
    results['original'] = {'test_f1': test_f1, 'test_acc': test_acc}
    
    # Method 2: Cross-validation with single best model
    print(f"\n{'='*60}")
    print(f"  Method 2: Cross-Validation (Best Model)")
    print(f"{'='*60}")
    
    cv = FewShotCrossValidator(
        model_class=lambda data, args: get_model(args.model, data, args),
        data=data,
        args=args,
        k_folds=optimal_k,
        target_node_type=args.target_node_type,
        seed=args.seed
    )
    
    cv_results = cv.run_cross_validation(
        max_epochs=args.n_epochs,
        patience=args.patience,
        save_results=False
    )
    
    final_results = cv.final_evaluation_on_test(cv_results)
    results['cv_single'] = final_results
    
    # Method 3: Cross-validation with ensemble
    print(f"\n{'='*60}")
    print(f"  Method 3: Cross-Validation (Ensemble)")
    print(f"{'='*60}")
    
    ensemble_results = cv.ensemble_prediction_on_test(cv_results)
    results['cv_ensemble'] = ensemble_results
    
    # Summary comparison
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Original Method:           F1 = {results['original']['test_f1']:.4f}")
    print(f"CV Best Model:            F1 = {results['cv_single']['test_f1']:.4f}")
    print(f"CV Ensemble (Voting):     F1 = {results['cv_ensemble']['voting_f1']:.4f}")
    print(f"CV Ensemble (Prob Avg):   F1 = {results['cv_ensemble']['prob_avg_f1']:.4f}")
    print(f"")
    print(f"CV Validation F1: {cv_results['mean_val_f1']:.4f} ± {cv_results['std_val_f1']:.4f}")
    print(f"CV Individual Test F1s: {results['cv_ensemble']['individual_test_f1s']}")
    
    return results


# Usage example and main function
def run_few_shot_cv_experiment(args):
    """Main function to run few-shot cross-validation experiment"""
    
    # Load your data
    from train_hetero_graph import load_hetero_graph, get_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_hetero_graph(args.graph_path, device, args.target_node_type)
    
    # Create cross-validator
    cv = FewShotCrossValidator(
        model_class=lambda data, args: get_model(args.model, data, args),
        data=data,
        args=args,
        k_folds=3,  # 可調整
        target_node_type=args.target_node_type,
        seed=args.seed
    )
    
    # Run cross-validation
    cv_results = cv.run_cross_validation(
        max_epochs=args.n_epochs,
        patience=args.patience,
        save_results=True,
        output_dir=f"cv_results_{args.model}_{args.target_node_type}"
    )
    
    # Final evaluation on test set
    final_results = cv.final_evaluation_on_test(cv_results)
    
    return cv_results, final_results


if __name__ == "__main__":
    from train_hetero_graph import parse_arguments
    
    args = parse_arguments()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run experiments with comparison
    if hasattr(args, 'compare_methods') and args.compare_methods:
        results = compare_cv_vs_original(args)
    else:
        cv_results, final_results = run_few_shot_cv_experiment(args)
        print(f"\n{'='*60}")
        print(f"  Experiment Complete!")
        print(f"{'='*60}")
        print(f"Cross-Validation F1: {cv_results['mean_val_f1']:.4f} ± {cv_results['std_val_f1']:.4f}")
        print(f"Final Test F1: {final_results['test_f1']:.4f}")