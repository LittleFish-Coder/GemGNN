# utils/pseudo_labeler.py
import os
import json
from tqdm import tqdm
from transformers import pipeline
from sample_k_shot import sample_k_shot
from datasets import load_dataset
import random
import torch
import numpy as np

DEFAULT_SEED = 42

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_pseudo_labeler(dataset_name, device="cpu"):
    if dataset_name.lower() == "gossipcop":
        model_name = "LittleFish-Coder/gossipcop_pseudo_labeler"
    elif dataset_name.lower() == "politifact":
        model_name = "LittleFish-Coder/politifact_pseudo_labeler"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return pipeline("text-classification", model=model_name, device=0 if device == "cuda" else -1)

def run_pseudo_labeler_and_cache(dataset, labeled_indices, dataset_name, cache_path, device="cpu"):
    # Remove labeled_indices from train set
    all_indices = set(range(len(dataset)))
    unlabeled_indices = sorted(list(all_indices - set(labeled_indices)))
    texts = [dataset[i]["text"] for i in unlabeled_indices]

    # Get pseudo-labeler
    pseudo_labeler = get_pseudo_labeler(dataset_name, device=device)

    # Run pseudo-labeler
    results = []
    for idx, text in tqdm(zip(unlabeled_indices, texts), total=len(texts), desc="Pseudo-labeling"):
        pred = pseudo_labeler(text, truncation=True, max_length=512)[0]
        label = 0 if pred["label"].lower() == "real" else 1
        results.append({"index": idx, "pseudo_label": label, "score": float(pred["score"])})

    # Save to cache
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pseudo-label cache saved to {cache_path}")
    return results

def main():
    set_seed()
    dataset_name = "gossipcop"
    cache_path = f"pseudo_label_cache_{dataset_name}.json"
    dataset = load_dataset(f"LittleFish-Coder/Fake_News_{dataset_name}", split="train", cache_dir="dataset")
    labeled_indices, _ = sample_k_shot(dataset, k=16, seed=DEFAULT_SEED)
    print(f"Labeled {len(labeled_indices)} indices")
    print(f"Labeled indices: {labeled_indices}")
    run_pseudo_labeler_and_cache(dataset, labeled_indices, dataset_name, cache_path)

if __name__ == "__main__":
    main()
