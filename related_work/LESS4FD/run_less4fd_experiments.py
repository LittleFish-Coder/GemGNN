#!/usr/bin/env python3
"""
Run LESS4FD experiments for few-shot fake news detection.

This script runs comprehensive experiments for the LESS4FD architecture
across different k-shot scenarios and datasets.
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def run_single_experiment(
    dataset: str,
    k_shot: int,
    embedding_type: str = "deberta",
    seed: int = 42,
    pretrain_epochs: int = 100,
    finetune_epochs: int = 50,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Run a single LESS4FD experiment.
    
    Args:
        dataset: Dataset name (politifact/gossipcop)
        k_shot: Number of shots for few-shot learning
        embedding_type: Type of embeddings to use
        seed: Random seed
        pretrain_epochs: Number of pretraining epochs
        finetune_epochs: Number of finetuning epochs
        device: Device to use (auto/cuda/cpu)
        
    Returns:
        Dictionary with experiment results
    """
    cmd = [
        "python", "train_less4fd.py",
        "--dataset", dataset,
        "--k_shot", str(k_shot),
        "--embedding_type", embedding_type,
        "--seed", str(seed),
        "--pretrain_epochs", str(pretrain_epochs),
        "--finetune_epochs", str(finetune_epochs)
    ]
    
    if device != "auto":
        cmd.extend(["--device", device])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"✓ Completed: {dataset} k={k_shot} seed={seed}")
            return {
                "status": "success",
                "dataset": dataset,
                "k_shot": k_shot,
                "embedding_type": embedding_type,
                "seed": seed,
                "stdout": result.stdout[-1000:],  # Last 1000 chars
                "stderr": result.stderr[-500:] if result.stderr else ""
            }
        else:
            print(f"✗ Failed: {dataset} k={k_shot} seed={seed}")
            print(f"Error: {result.stderr}")
            return {
                "status": "failed",
                "dataset": dataset,
                "k_shot": k_shot,
                "embedding_type": embedding_type,
                "seed": seed,
                "error": result.stderr,
                "stdout": result.stdout[-1000:] if result.stdout else ""
            }
    except subprocess.TimeoutExpired:
        print(f"⏱ Timeout: {dataset} k={k_shot} seed={seed}")
        return {
            "status": "timeout",
            "dataset": dataset,
            "k_shot": k_shot,
            "embedding_type": embedding_type,
            "seed": seed,
            "error": "Experiment timed out after 1 hour"
        }
    except Exception as e:
        print(f"✗ Exception: {dataset} k={k_shot} seed={seed}: {e}")
        return {
            "status": "exception",
            "dataset": dataset,
            "k_shot": k_shot,
            "embedding_type": embedding_type,
            "seed": seed,
            "error": str(e)
        }

def run_k_shot_sweep(
    datasets: List[str] = ["politifact", "gossipcop"],
    k_shots: List[int] = [3, 4, 5, 8, 16],
    embedding_types: List[str] = ["deberta"],
    seeds: List[int] = [42, 123, 456],
    pretrain_epochs: int = 100,
    finetune_epochs: int = 50,
    device: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Run k-shot experiments across datasets and seeds.
    
    Args:
        datasets: List of datasets to test
        k_shots: List of k-shot values to test
        embedding_types: List of embedding types to test
        seeds: List of random seeds to test
        pretrain_epochs: Number of pretraining epochs
        finetune_epochs: Number of finetuning epochs
        device: Device to use
        
    Returns:
        List of experiment results
    """
    results = []
    total_experiments = len(datasets) * len(k_shots) * len(embedding_types) * len(seeds)
    current_experiment = 0
    
    print(f"Starting {total_experiments} LESS4FD experiments...")
    
    for dataset in datasets:
        for k_shot in k_shots:
            for embedding_type in embedding_types:
                for seed in seeds:
                    current_experiment += 1
                    print(f"\n[{current_experiment}/{total_experiments}] "
                          f"Dataset: {dataset}, K-shot: {k_shot}, "
                          f"Embedding: {embedding_type}, Seed: {seed}")
                    
                    result = run_single_experiment(
                        dataset=dataset,
                        k_shot=k_shot,
                        embedding_type=embedding_type,
                        seed=seed,
                        pretrain_epochs=pretrain_epochs,
                        finetune_epochs=finetune_epochs,
                        device=device
                    )
                    
                    results.append(result)
    
    return results

def save_experiment_summary(results: List[Dict[str, Any]], output_file: str = "less4fd_experiment_summary.json"):
    """Save experiment summary to file."""
    summary = {
        "total_experiments": len(results),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "timeout": len([r for r in results if r["status"] == "timeout"]),
        "exception": len([r for r in results if r["status"] == "exception"]),
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment summary saved to {output_file}")
    print(f"Success rate: {summary['successful']}/{summary['total_experiments']} "
          f"({summary['successful']/summary['total_experiments']*100:.1f}%)")

def analyze_results(results_file: str = "less4fd_experiment_summary.json"):
    """Analyze experiment results and print summary statistics."""
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data["results"]
    successful_results = [r for r in results if r["status"] == "success"]
    
    if not successful_results:
        print("No successful experiments found")
        return
    
    print("\n=== LESS4FD Experiment Analysis ===")
    print(f"Total experiments: {data['total_experiments']}")
    print(f"Successful: {data['successful']} ({data['successful']/data['total_experiments']*100:.1f}%)")
    print(f"Failed: {data['failed']}")
    print(f"Timeout: {data['timeout']}")
    print(f"Exception: {data['exception']}")
    
    # Group by dataset and k-shot
    by_dataset_k = {}
    for result in successful_results:
        key = f"{result['dataset']}_k{result['k_shot']}"
        if key not in by_dataset_k:
            by_dataset_k[key] = []
        by_dataset_k[key].append(result)
    
    print(f"\nSuccessful experiments by configuration:")
    for key, experiments in by_dataset_k.items():
        print(f"  {key}: {len(experiments)} experiments")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run LESS4FD experiments")
    
    parser.add_argument("--datasets", nargs="+", default=["politifact", "gossipcop"],
                        choices=["politifact", "gossipcop"],
                        help="Datasets to test")
    parser.add_argument("--k_shots", nargs="+", type=int, default=[3, 4, 5, 8, 16],
                        help="K-shot values to test")
    parser.add_argument("--embedding_types", nargs="+", default=["deberta"],
                        choices=["bert", "deberta", "roberta", "distilbert"],
                        help="Embedding types to test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Random seeds to test")
    parser.add_argument("--pretrain_epochs", type=int, default=100,
                        help="Number of pretraining epochs")
    parser.add_argument("--finetune_epochs", type=int, default=50,
                        help="Number of finetuning epochs")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use")
    parser.add_argument("--output", type=str, default="less4fd_experiment_summary.json",
                        help="Output file for experiment summary")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze existing results")
    parser.add_argument("--single", action="store_true",
                        help="Run single experiment with first parameters")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_results(args.output)
        return
    
    if args.single:
        # Run single experiment for testing
        result = run_single_experiment(
            dataset=args.datasets[0],
            k_shot=args.k_shots[0],
            embedding_type=args.embedding_types[0],
            seed=args.seeds[0],
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            device=args.device
        )
        print(f"\nSingle experiment result: {result}")
        return
    
    # Run full experiment sweep
    results = run_k_shot_sweep(
        datasets=args.datasets,
        k_shots=args.k_shots,
        embedding_types=args.embedding_types,
        seeds=args.seeds,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        device=args.device
    )
    
    # Save results
    save_experiment_summary(results, args.output)
    
    # Analyze results
    analyze_results(args.output)

if __name__ == "__main__":
    main()