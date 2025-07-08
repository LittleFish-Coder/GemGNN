#!/usr/bin/env python3
"""
Extract F1 scores from LESS4FD results
"""

import json
import os
import glob
from collections import defaultdict
import pandas as pd

def extract_f1_scores(results_dir="results_genfend"):
    """Extract F1 scores from all result files."""
    
    # Get all JSON result files
    json_files = glob.glob(os.path.join(results_dir, "*_results.json"))
    
    if not json_files:
        print(f"No result files found in {results_dir}")
        return
    
    print(f"Found {len(json_files)} result files")
    print("=" * 80)
    
    # Parse results
    results = []
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract filename components
            filename = os.path.basename(json_file)
            # Parse: less4fd_politifact_k8_deberta_HGT_results.json
            parts = filename.replace('genfend_', '').replace('_results.json', '').split('_')
            print(parts)
            if len(parts) >= 3:
                dataset = parts[0]  # politifact or gossipcop
                k_shot = int(parts[1].replace('k', ''))  # 8
                embedding = parts[2]  # deberta
                
                # Get F1 score
                f1_score = data['final_metrics']['f1']
                accuracy = data['final_metrics']['accuracy']
                precision = data['final_metrics']['precision']
                recall = data['final_metrics']['recall']
                loss = data['final_metrics']['loss']
                training_time = data['training_time']
                
                results.append({
                    'dataset': dataset,
                    'k_shot': k_shot,
                    'embedding': embedding,
                    'f1': f1_score,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'loss': loss,
                    'training_time': training_time,
                    'filename': filename
                })
                
        except Exception:
            pass
    
    if not results:
        print("No valid results found")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(results)
    
    # Sort by dataset, k_shot, model
    df = df.sort_values(['dataset', 'k_shot'])
    
    # Display summary statistics
    print("SUMMARY STATISTICS:")
    print("=" * 80)
    
    # Overall statistics
    print(f"Total experiments: {len(df)}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"K-shot values: {sorted(df['k_shot'].unique())}")
    print(f"Embeddings: {df['embedding'].unique()}")
    print()
    
    # F1 score statistics
    print("F1 SCORE STATISTICS:")
    print("=" * 80)
    print(f"Mean F1: {df['f1'].mean():.4f}")
    print(f"Std F1: {df['f1'].std():.4f}")
    print(f"Min F1: {df['f1'].min():.4f}")
    print(f"Max F1: {df['f1'].max():.4f}")
    print()
    
    # Best results by dataset
    print("BEST F1 SCORES BY DATASET:")
    print("=" * 80)
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        best_idx = dataset_df['f1'].idxmax()
        best_result = dataset_df.loc[best_idx]
        print(f"{dataset.upper()}:")
        print(f"  Best F1: {best_result['f1']:.4f} (k={best_result['k_shot']})")
        print(f"  Accuracy: {best_result['accuracy']:.4f}")
        print(f"  Precision: {best_result['precision']:.4f}")
        print(f"  Recall: {best_result['recall']:.4f}")
        print()
    
    # Detailed results table
    print("DETAILED RESULTS:")
    print("=" * 80)
    
    # Format for display
    display_df = df[['dataset', 'k_shot', 'f1', 'accuracy', 'precision', 'recall', 'loss']].copy()
    display_df['f1'] = display_df['f1'].map('{:.4f}'.format)
    display_df['accuracy'] = display_df['accuracy'].map('{:.4f}'.format)
    display_df['precision'] = display_df['precision'].map('{:.4f}'.format)
    display_df['recall'] = display_df['recall'].map('{:.4f}'.format)
    display_df['loss'] = display_df['loss'].map('{:.4f}'.format)
    
    print(display_df.to_string(index=False))
    print()
    
    # F1 scores by k-shot
    print("F1 SCORES BY K-SHOT:")
    print("=" * 80)
    for dataset in df['dataset'].unique():
        print(f"\n{dataset.upper()}:")
        dataset_df = df[df['dataset'] == dataset]

        record_k_shot = []
        for _, row in dataset_df.sort_values('k_shot').iterrows():
            record_k_shot.append(row['f1'])
            print(f"    k={row['k_shot']:2d}: F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}")
        print(record_k_shot)
    
    # Save to CSV
    csv_file = os.path.join(results_dir, "f1_scores_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to: {csv_file}")
    
    return df

if __name__ == "__main__":
    # Check if results directory exists
    results_dir = "results_genfend"
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        print("Please run the experiments first or specify the correct results directory.")
        exit(1)
    
    # Extract and display F1 scores
    df = extract_f1_scores(results_dir) 