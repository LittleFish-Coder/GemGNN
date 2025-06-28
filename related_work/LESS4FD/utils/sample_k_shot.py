"""
Shared sampling utility to ensure consistency between language and graph models.
"""
from datasets import Dataset
from typing import Dict, List, Tuple
import numpy as np
from datasets import load_dataset

def sample_k_shot(train_data: Dataset, k: int, seed: int = 42) -> Tuple[List[int], Dict]:
    """
    Sample k examples per class, returning both indices and dataset.
    
    Args:
        train_data: Original training dataset
        k: Number of samples per class
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (selected indices, sampled data dictionary)
    """
    # Initialize output
    sampled_data = {key: [] for key in train_data.column_names}
    selected_indices = []

    # Sample k examples from each class
    labels = set(train_data["label"])
    for label in labels:
        # Filter to this class
        label_data = train_data.filter(lambda x: x["label"] == label)
        
        # Shuffle and select first k examples - EXACT SAME LOGIC as finetune_lm.py
        sampled_label_data = label_data.shuffle(seed=seed).select(
            range(min(k, len(label_data)))
        )
        
        label_indices = [i for i, l in enumerate(train_data["label"]) if l == label]
        
        # Get shuffled indices
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(label_indices)[:min(k, len(label_indices))]
        
        # Add to our list of selected indices
        selected_indices.extend(shuffled_indices)
        
        # Add the data to our sampled dataset
        for key in train_data.column_names:
            sampled_data[key].extend(sampled_label_data[key])
    
    return selected_indices, sampled_data


def debug_compare_samples(dataset_name: str, k_shot: int, seed: int = 42):
    """
    Debug utility to compare samples between two sampling runs.
    
    Args:
        dataset_name: Dataset name
        k_shot: Samples per class
        seed: Random seed
    """
    
    # Load dataset
    dataset = load_dataset(
        f"LittleFish-Coder/Fake_News_{dataset_name}",
        download_mode="reuse_cache_if_exists",
        cache_dir="dataset"
    )
    
    train_data = dataset["train"]
    
    # Sample twice using same function
    indices1, data1 = sample_k_shot(train_data, k_shot, seed)
    indices2, data2 = sample_k_shot(train_data, k_shot, seed)
    
    # Compare
    print(f"Run 1 indices: {sorted(indices1)}")
    print(f"Run 2 indices: {sorted(indices2)}")
    print(f"Indices match: {sorted(indices1) == sorted(indices2)}")
    
    # Compare text samples
    for i in range(min(len(data1["text"]), 2)):  # Show first 2 samples
        print(f"Index: {indices1[i]}")
        print(f"Run 1: {data1['text'][i][:50]}...")
        print(f"Run 2: {data2['text'][i][:50]}...")
    
    
if __name__ == "__main__":
    # Example usage:
    debug_compare_samples("politifact", 8)