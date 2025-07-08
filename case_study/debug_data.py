#!/usr/bin/env python3
"""
Debug script to understand the available data
"""

import os
import json
import pandas as pd
from pathlib import Path

REPO_ROOT = Path("/home/runner/work/GemGNN/GemGNN")
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_HETERO_DIR = REPO_ROOT / "results_hetero"
RELATED_WORK_DIR = REPO_ROOT / "related_work"

def debug_data():
    print("=== Debugging Available Data ===")
    
    all_results = []
    
    # Check GemGNN results
    print("\n1. GemGNN Results (HAN/HGT):")
    hetero_results_dir = RESULTS_HETERO_DIR
    if hetero_results_dir.exists():
        for model_dir in hetero_results_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                print(f"  Model: {model_name}")
                
                for dataset_dir in model_dir.iterdir():
                    if dataset_dir.is_dir():
                        dataset_name = dataset_dir.name
                        exp_count = 0
                        
                        for exp_dir in dataset_dir.iterdir():
                            if exp_dir.is_dir() and (exp_dir / "metrics.json").exists():
                                exp_count += 1
                                
                                # Check a few examples
                                if exp_count <= 3:
                                    try:
                                        with open(exp_dir / "metrics.json", 'r') as f:
                                            data = json.load(f)
                                        
                                        k_shot = extract_k_shot(exp_dir.name)
                                        test_metrics = data.get('final_test_metrics_on_target_node', {})
                                        f1_score = test_metrics.get('f1_score', 0)
                                        accuracy = test_metrics.get('accuracy', 0)
                                        
                                        print(f"    {dataset_name}: {exp_dir.name[:50]}... k={k_shot}, F1={f1_score:.3f}, Acc={accuracy:.3f}")
                                        
                                        all_results.append({
                                            'model': f'gemgnn_{model_name.lower()}',
                                            'dataset': dataset_name,
                                            'k_shot': k_shot,
                                            'f1_score': f1_score,
                                            'accuracy': accuracy,
                                            'type': 'gemgnn'
                                        })
                                        
                                    except Exception as e:
                                        print(f"    Error: {e}")
                        
                        print(f"    {dataset_name}: {exp_count} experiments")
    
    # Check baseline results
    print("\n2. Baseline Results:")
    baseline_results_dir = RESULTS_DIR
    if baseline_results_dir.exists():
        for model_dir in baseline_results_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                print(f"  Model: {model_name}")
                
                for dataset_dir in model_dir.iterdir():
                    if dataset_dir.is_dir():
                        dataset_name = dataset_dir.name
                        exp_count = 0
                        
                        for exp_dir in dataset_dir.iterdir():
                            if exp_dir.is_dir() and (exp_dir / "metrics.json").exists():
                                exp_count += 1
                                
                                # Check a few examples
                                if exp_count <= 3:
                                    try:
                                        with open(exp_dir / "metrics.json", 'r') as f:
                                            data = json.load(f)
                                        
                                        k_shot = data.get('k_shot', extract_k_shot(exp_dir.name))
                                        test_metrics = data.get('test_metrics', {})
                                        f1_score = test_metrics.get('f1_score', 0)
                                        accuracy = test_metrics.get('accuracy', 0)
                                        
                                        print(f"    {dataset_name}: {exp_dir.name[:50]}... k={k_shot}, F1={f1_score:.3f}, Acc={accuracy:.3f}")
                                        
                                        all_results.append({
                                            'model': model_name.lower(),
                                            'dataset': dataset_name,
                                            'k_shot': k_shot,
                                            'f1_score': f1_score,
                                            'accuracy': accuracy,
                                            'type': 'baseline'
                                        })
                                        
                                    except Exception as e:
                                        print(f"    Error: {e}")
                        
                        print(f"    {dataset_name}: {exp_count} experiments")
    
    # Check related work
    print("\n3. Related Work Results:")
    
    # LESS4FD
    less4fd_dir = RELATED_WORK_DIR / "LESS4FD" / "results_less4fd"
    if less4fd_dir.exists():
        print("  LESS4FD:")
        count = 0
        for json_file in less4fd_dir.glob("*.json"):
            count += 1
            if count <= 3:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    filename = json_file.stem
                    parts = filename.split('_')
                    dataset_name = parts[1] if len(parts) > 1 else 'unknown'
                    k_shot = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith('k') else 0
                    
                    f1_score = data.get('final_metrics', {}).get('f1', 0)
                    accuracy = data.get('final_metrics', {}).get('accuracy', 0)
                    
                    print(f"    {filename[:50]}... k={k_shot}, F1={f1_score:.3f}, Acc={accuracy:.3f}")
                    
                    all_results.append({
                        'model': 'less4fd',
                        'dataset': dataset_name,
                        'k_shot': k_shot,
                        'f1_score': f1_score,
                        'accuracy': accuracy,
                        'type': 'related_work'
                    })
                    
                except Exception as e:
                    print(f"    Error: {e}")
        print(f"    Total: {count} files")
    
    # HeteroSGT
    heterosgt_dir = RELATED_WORK_DIR / "HeteroSGT" / "results_heterosgt"
    if heterosgt_dir.exists():
        print("  HeteroSGT:")
        count = 0
        for json_file in heterosgt_dir.glob("*.json"):
            count += 1
            if count <= 3:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    filename = json_file.stem
                    parts = filename.split('_')
                    dataset_name = parts[1] if len(parts) > 1 else 'unknown'
                    k_shot = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith('k') else 0
                    
                    f1_score = data.get('final_metrics', {}).get('f1', 0)
                    accuracy = data.get('final_metrics', {}).get('accuracy', 0)
                    
                    print(f"    {filename[:50]}... k={k_shot}, F1={f1_score:.3f}, Acc={accuracy:.3f}")
                    
                    all_results.append({
                        'model': 'heterosgt',
                        'dataset': dataset_name,
                        'k_shot': k_shot,
                        'f1_score': f1_score,
                        'accuracy': accuracy,
                        'type': 'related_work'
                    })
                    
                except Exception as e:
                    print(f"    Error: {e}")
        print(f"    Total: {count} files")
    
    # Analysis
    print("\n=== Analysis ===")
    df = pd.DataFrame(all_results)
    
    if not df.empty:
        print(f"Total results collected: {len(df)}")
        print(f"Unique models: {df['model'].unique()}")
        print(f"Unique datasets: {df['dataset'].unique()}")
        print(f"K-shot range: {df['k_shot'].min()} to {df['k_shot'].max()}")
        
        # Performance summary by model
        print("\nPerformance summary by model:")
        model_summary = df.groupby('model')[['f1_score', 'accuracy']].agg(['mean', 'count', 'max']).round(3)
        print(model_summary)
        
        # Find overlapping configurations
        print("\nOverlapping configurations:")
        for dataset in df['dataset'].unique():
            for k_shot in df['k_shot'].unique():
                subset = df[(df['dataset'] == dataset) & (df['k_shot'] == k_shot)]
                if len(subset) > 1:
                    models = subset['model'].tolist()
                    f1_scores = subset['f1_score'].tolist()
                    print(f"  {dataset} k={k_shot}: {dict(zip(models, f1_scores))}")
    else:
        print("No results found!")

def extract_k_shot(exp_name: str) -> int:
    """Extract k-shot value from experiment name."""
    parts = exp_name.split('_')
    for part in parts:
        if part.endswith('shot'):
            try:
                return int(part.replace('shot', ''))
            except:
                pass
    return 0

if __name__ == "__main__":
    debug_data()