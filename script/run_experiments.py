#!/usr/bin/env python3
"""
Systematic Experiments Script for Few-Shot Fake News Detection
This script helps run systematic experiments comparing different configurations.
"""

import os
import subprocess
import itertools
import json
import time
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and capture output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if result.returncode == 0:
        print(f"✅ SUCCESS: {description}")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
    else:
        print(f"❌ FAILED: {description}")
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
    
    return result.returncode == 0, result.stdout, result.stderr

def experiment_1_dissimilar_edges():
    """Experiment 1: Test effect of dissimilar edges"""
    print(f"\n{'#'*80}")
    print("EXPERIMENT 1: Dissimilar Edges Effect")
    print(f"{'#'*80}")
    
    base_config = {
        'dataset': 'politifact',
        'k_shot': 8,
        'embedding': 'deberta',
        'edge_policy': 'knn_test_isolated',
        'k_neighbors': 5,
        'model': 'HAN',
        'han_layers': 1,
        'loss_fn': 'ce'
    }
    
    results = {}
    
    # Test with and without dissimilar edges
    for enable_dissimilar in [False, True]:
        config_name = f"dissimilar_{enable_dissimilar}"
        print(f"\n--- Testing {config_name} ---")
        
        # Build graph
        dissimilar_flag = "--enable_dissimilar" if enable_dissimilar else ""
        graph_cmd = f"""python build_hetero_graph.py \\
            --dataset_name {base_config['dataset']} \\
            --k_shot {base_config['k_shot']} \\
            --embedding_type {base_config['embedding']} \\
            --edge_policy {base_config['edge_policy']} \\
            --k_neighbors {base_config['k_neighbors']} \\
            --ensure_test_labeled_neighbor \\
            --partial_unlabeled \\
            --sample_unlabeled_factor 5 \\
            --multi_view 3 \\
            {dissimilar_flag}"""
        
        success, stdout, stderr = run_command(graph_cmd, f"Building graph for {config_name}")
        if not success:
            print(f"Failed to build graph for {config_name}")
            continue
        
        # Extract graph path from stdout
        graph_path = None
        for line in stdout.split('\n'):
            if 'Saved HeteroData to' in line:
                graph_path = line.split('Saved HeteroData to ')[-1].strip()
                break
        
        if not graph_path:
            print(f"Could not find graph path for {config_name}")
            continue
        
        # Train model
        train_cmd = f"""python train_hetero_graph.py \\
            --graph_path {graph_path} \\
            --model {base_config['model']} \\
            --han_layers {base_config['han_layers']} \\
            --loss_fn {base_config['loss_fn']}"""
        
        success, stdout, stderr = run_command(train_cmd, f"Training model for {config_name}")
        if success:
            results[config_name] = {"success": True, "graph_path": graph_path}
        else:
            results[config_name] = {"success": False, "error": stderr}
    
    return results

def experiment_2_han_layers():
    """Experiment 2: Compare 1-hop vs 2-hop HAN layers"""
    print(f"\n{'#'*80}")
    print("EXPERIMENT 2: 1-Hop vs 2-Hop HAN Layers")
    print(f"{'#'*80}")
    
    # Use existing graph with dissimilar edges (from experiment 1)
    graph_path = "graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_dissimilar_multiview_3/graph.pt"
    
    if not os.path.exists(graph_path):
        print(f"Graph not found: {graph_path}")
        print("Please run experiment 1 first or build the graph manually")
        return {}
    
    results = {}
    
    # Test 1-hop vs 2-hop
    for han_layers in [1, 2]:
        config_name = f"{han_layers}_hop"
        print(f"\n--- Testing {config_name} ---")
        
        train_cmd = f"""python train_hetero_graph.py \\
            --graph_path {graph_path} \\
            --model HAN \\
            --han_layers {han_layers} \\
            --loss_fn ce"""
        
        success, stdout, stderr = run_command(train_cmd, f"Training {config_name} HAN model")
        if success:
            results[config_name] = {"success": True, "han_layers": han_layers}
        else:
            results[config_name] = {"success": False, "error": stderr}
    
    return results

def experiment_3_loss_functions():
    """Experiment 3: Compare different loss functions"""
    print(f"\n{'#'*80}")
    print("EXPERIMENT 3: Loss Function Comparison")
    print(f"{'#'*80}")
    
    # Use existing graph
    graph_path = "graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_dissimilar_multiview_3/graph.pt"
    
    if not os.path.exists(graph_path):
        print(f"Graph not found: {graph_path}")
        return {}
    
    results = {}
    loss_functions = ["ce", "focal", "enhanced", "ce_smooth", "robust"]
    
    for loss_fn in loss_functions:
        config_name = f"loss_{loss_fn}"
        print(f"\n--- Testing {config_name} ---")
        
        train_cmd = f"""python train_hetero_graph.py \\
            --graph_path {graph_path} \\
            --model HAN \\
            --han_layers 1 \\
            --loss_fn {loss_fn}"""
        
        success, stdout, stderr = run_command(train_cmd, f"Training with {loss_fn} loss")
        if success:
            results[config_name] = {"success": True, "loss_fn": loss_fn}
        else:
            results[config_name] = {"success": False, "error": stderr}
    
    return results

def experiment_4_k_neighbors():
    """Experiment 4: Optimize k_neighbors parameter"""
    print(f"\n{'#'*80}")
    print("EXPERIMENT 4: K-Neighbors Optimization")
    print(f"{'#'*80}")
    
    base_config = {
        'dataset': 'politifact',
        'k_shot': 8,
        'embedding': 'deberta',
        'edge_policy': 'knn_test_isolated',
        'model': 'HAN',
        'han_layers': 1,
        'loss_fn': 'robust'  # Use robust loss for k-neighbors optimization
    }
    
    results = {}
    k_values = [3, 5, 7, 10]
    
    for k in k_values:
        config_name = f"k_{k}"
        print(f"\n--- Testing {config_name} ---")
        
        # Build graph with different k
        graph_cmd = f"""python build_hetero_graph.py \\
            --dataset_name {base_config['dataset']} \\
            --k_shot {base_config['k_shot']} \\
            --embedding_type {base_config['embedding']} \\
            --edge_policy {base_config['edge_policy']} \\
            --k_neighbors {k} \\
            --ensure_test_labeled_neighbor \\
            --partial_unlabeled \\
            --sample_unlabeled_factor 5 \\
            --multi_view 3 \\
            --enable_dissimilar"""
        
        success, stdout, stderr = run_command(graph_cmd, f"Building graph for k={k}")
        if not success:
            continue
        
        # Extract graph path
        graph_path = None
        for line in stdout.split('\n'):
            if 'Saved HeteroData to' in line:
                graph_path = line.split('Saved HeteroData to ')[-1].strip()
                break
        
        if not graph_path:
            continue
        
        # Train model
        train_cmd = f"""python train_hetero_graph.py \\
            --graph_path {graph_path} \\
            --model {base_config['model']} \\
            --han_layers {base_config['han_layers']} \\
            --loss_fn {base_config['loss_fn']}"""
        
        success, stdout, stderr = run_command(train_cmd, f"Training with k={k}")
        if success:
            results[config_name] = {"success": True, "k_neighbors": k}
        else:
            results[config_name] = {"success": False, "error": stderr}
    
    return results

def main():
    """Run all experiments systematically"""
    print("🚀 Starting Systematic Experiments for Few-Shot Fake News Detection")
    print("=" * 80)
    
    all_results = {}
    
    # Experiment 1: Dissimilar edges
    print("Starting Experiment 1...")
    all_results["experiment_1_dissimilar"] = experiment_1_dissimilar_edges()
    
    # Experiment 2: HAN layers (1-hop vs 2-hop)
    print("Starting Experiment 2...")
    all_results["experiment_2_han_layers"] = experiment_2_han_layers()
    
    # Experiment 3: Loss functions
    print("Starting Experiment 3...")
    all_results["experiment_3_loss_functions"] = experiment_3_loss_functions()
    
    # Experiment 4: K-neighbors optimization
    print("Starting Experiment 4...")
    all_results["experiment_4_k_neighbors"] = experiment_4_k_neighbors()
    
    # Save all results
    results_file = f"experiment_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")
    
    # Print summary
    print("\n📊 EXPERIMENT SUMMARY:")
    for exp_name, exp_results in all_results.items():
        print(f"\n{exp_name}:")
        for config, result in exp_results.items():
            status = "✅" if result.get("success", False) else "❌"
            print(f"  {status} {config}")

if __name__ == "__main__":
    main()