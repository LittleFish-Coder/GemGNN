"""
GenFEND Data Preparation Script.

Simplified implementation of GenFEND (Generative Multi-view Fake News Detection)
that prepares demographic-aware data without expensive LLM API calls.
Based on the GenFEND paper that uses role-playing LLMs to generate 30 demographic
user profiles for synthetic comment generation.
"""

import os
import gc
import json
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List, Union, Any
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from tqdm.auto import tqdm
from argparse import ArgumentParser


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across all random processes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


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
        
        # Shuffle and select first k examples
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


class GenFENDDataBuilder:
    """
    Simplified GenFEND data preparation that creates demographic-aware features
    without expensive LLM API calls.
    """
    
    def __init__(
        self,
        dataset_name: str,
        k_shot: int,
        embedding_type: str = "deberta",
        num_demographic_profiles: int = 30,
        num_views: int = 3,  # Gender, Age, Education
        partial_unlabeled: bool = True,
        sample_unlabeled_factor: int = 5,
        output_dir: str = "data_genfend",
        dataset_cache_dir: str = "dataset",
        seed: int = 42
    ):
        """
        Initialize GenFEND data builder.
        
        Args:
            dataset_name: Dataset name (politifact, gossipcop)
            k_shot: Number of labeled samples per class
            embedding_type: Type of text embeddings to use
            num_demographic_profiles: Number of demographic profiles (default: 30 as in paper)
            num_views: Number of demographic views (Gender, Age, Education)
            partial_unlabeled: Whether to use partial unlabeled data
            sample_unlabeled_factor: Factor for unlabeled sampling
            output_dir: Output directory for prepared data
            dataset_cache_dir: Cache directory for datasets
            seed: Random seed
        """
        self.dataset_name = dataset_name
        self.k_shot = k_shot
        self.embedding_type = embedding_type
        self.num_demographic_profiles = num_demographic_profiles
        self.num_views = num_views
        self.partial_unlabeled = partial_unlabeled
        self.sample_unlabeled_factor = sample_unlabeled_factor
        self.output_dir = output_dir
        self.dataset_cache_dir = dataset_cache_dir
        self.seed = seed
        
        # Text embedding field mapping
        self.text_embedding_field = f"{embedding_type}_embeddings"
        
        # Data storage
        self.dataset = None
        self.train_data = None
        self.test_data = None
        
        # Sampling indices
        self.train_labeled_indices = None
        self.train_unlabeled_indices = None
        self.test_indices = None
        
        # Demographic view names
        self.view_names = ["gender", "age", "education"][:num_views]
        
    def load_dataset(self) -> None:
        """Load dataset from Hugging Face and perform initial checks."""
        set_seed(self.seed)
        
        # Map dataset names to Hugging Face identifiers
        hf_dataset_name = f"LittleFish-Coder/Fake_News_{self.dataset_name}"
        local_hf_dir = os.path.join(self.dataset_cache_dir, self.dataset_name)
        
        if os.path.exists(local_hf_dir):
            print(f"Loading cached dataset from {local_hf_dir}")
            dataset = load_from_disk(local_hf_dir)
        else:
            print(f"Downloading dataset {hf_dataset_name}")
            os.makedirs(self.dataset_cache_dir, exist_ok=True)
            try:
                dataset = load_dataset(hf_dataset_name, download_mode="reuse_cache_if_exists")
                dataset.save_to_disk(local_hf_dir)
            except Exception as e:
                print(f"Failed to download from HuggingFace: {e}")
                print("Creating mock dataset for demonstration...")
                # Create a minimal mock dataset for testing
                dataset = self._create_mock_dataset()

        # Store dataset
        self.dataset = {
            "train": dataset["train"], 
            "test": dataset["test"]
        }
        self.train_data = self.dataset["train"]
        self.test_data = self.dataset["test"]

        print(f"Dataset loaded: {len(self.train_data)} train, {len(self.test_data)} test samples")
        
        # Verify embedding field exists
        if self.text_embedding_field not in self.train_data.column_names:
            raise ValueError(f"Embedding field '{self.text_embedding_field}' not found in dataset")
    
    def _create_mock_dataset(self):
        """Create a mock dataset for testing purposes."""
        from datasets import Dataset, DatasetDict
        import numpy as np
        
        # Create mock embeddings (matching DeBERTa size)
        embedding_dim = 768
        train_size = 50
        test_size = 20
        
        np.random.seed(self.seed)
        
        # Create training data
        train_embeddings = np.random.randn(train_size, embedding_dim).astype(np.float32)
        train_labels = np.random.choice([0, 1], size=train_size)
        train_texts = [f"This is mock news article {i}" for i in range(train_size)]
        
        # Create test data  
        test_embeddings = np.random.randn(test_size, embedding_dim).astype(np.float32)
        test_labels = np.random.choice([0, 1], size=test_size)
        test_texts = [f"This is mock test article {i}" for i in range(test_size)]
        
        train_dataset = Dataset.from_dict({
            "text": train_texts,
            "label": train_labels.tolist(),
            f"{self.embedding_type}_embeddings": train_embeddings.tolist()
        })
        
        test_dataset = Dataset.from_dict({
            "text": test_texts,
            "label": test_labels.tolist(),
            f"{self.embedding_type}_embeddings": test_embeddings.tolist()
        })
        
        return DatasetDict({"train": train_dataset, "test": test_dataset})
    
    def sample_data(self) -> None:
        """Sample k-shot labeled data and optional unlabeled data."""
        print(f"Sampling {self.k_shot}-shot data...")
        
        # Sample labeled data
        self.train_labeled_indices, _ = sample_k_shot(self.train_data, self.k_shot, self.seed)
        print(f"Sampled {len(self.train_labeled_indices)} labeled training examples")
        
        # Sample unlabeled data if enabled
        if self.partial_unlabeled:
            # Use partial unlabeled data with specified factor
            num_unlabeled = len(set(self.train_data["label"])) * self.k_shot * self.sample_unlabeled_factor
            
            # Get indices not in labeled set
            all_indices = set(range(len(self.train_data)))
            labeled_set = set(self.train_labeled_indices)
            unlabeled_candidates = list(all_indices - labeled_set)
            
            # Randomly sample from unlabeled candidates
            np.random.seed(self.seed)
            num_to_sample = min(num_unlabeled, len(unlabeled_candidates))
            self.train_unlabeled_indices = np.random.choice(
                unlabeled_candidates, size=num_to_sample, replace=False
            ).tolist()
            
            print(f"Sampled {len(self.train_unlabeled_indices)} unlabeled training examples")
        else:
            self.train_unlabeled_indices = []
        
        # All test data
        self.test_indices = list(range(len(self.test_data)))
        print(f"Using {len(self.test_indices)} test examples")
    
    def generate_demographic_features(self) -> Dict[str, np.ndarray]:
        """
        Generate simulated demographic features for each sample.
        In the real GenFEND, this would involve LLM API calls to generate
        30 demographic user profiles. Here we simulate this with random features.
        """
        print("Generating demographic features...")
        
        total_samples = len(self.train_data) + len(self.test_data)
        
        # Generate demographic features for each view
        demographic_features = {}
        
        for view_name in self.view_names:
            # Each view has features for all demographic profiles
            # Shape: (total_samples, num_demographic_profiles, feature_dim)
            feature_dim = 32  # Simplified feature dimension
            
            np.random.seed(self.seed + hash(view_name) % 1000)
            view_features = np.random.randn(total_samples, self.num_demographic_profiles, feature_dim)
            
            # Add some structure to make different views distinguishable
            if view_name == "gender":
                # Add binary-like patterns for gender
                view_features += np.random.choice([-1, 1], size=view_features.shape) * 0.5
            elif view_name == "age":
                # Add age-related gradients
                age_factor = np.linspace(-1, 1, self.num_demographic_profiles).reshape(1, -1, 1)
                view_features += age_factor * 0.3
            elif view_name == "education":
                # Add education-level patterns
                edu_factor = np.random.exponential(scale=0.5, size=view_features.shape)
                view_features += edu_factor * 0.2
            
            # Normalize features
            view_features = (view_features - view_features.mean()) / (view_features.std() + 1e-8)
            
            demographic_features[view_name] = view_features.astype(np.float32)
        
        print(f"Generated demographic features for {len(self.view_names)} views")
        return demographic_features
    
    def compute_diversity_signals(self, demographic_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute diversity signals as in the GenFEND paper.
        This represents the KL divergence-based diversity within each view.
        """
        print("Computing diversity signals...")
        
        total_samples = len(self.train_data) + len(self.test_data)
        diversity_signals = np.zeros((total_samples, len(self.view_names)), dtype=np.float32)
        
        for view_idx, view_name in enumerate(self.view_names):
            view_features = demographic_features[view_name]  # (samples, profiles, features)
            
            # Compute diversity as variance across demographic profiles
            # This is a simplified proxy for KL divergence mentioned in the paper
            profile_means = view_features.mean(axis=2)  # (samples, profiles)
            diversity = profile_means.var(axis=1)  # (samples,)
            
            diversity_signals[:, view_idx] = diversity
        
        # Normalize diversity signals
        diversity_signals = (diversity_signals - diversity_signals.mean()) / (diversity_signals.std() + 1e-8)
        
        print("Computed diversity signals")
        return diversity_signals
    
    def prepare_data(self) -> Dict[str, Any]:
        """Prepare all data components for GenFEND training."""
        print("Preparing GenFEND data...")
        
        # Get text embeddings
        train_embeddings = np.array(self.train_data[self.text_embedding_field])
        test_embeddings = np.array(self.test_data[self.text_embedding_field])
        all_embeddings = np.vstack([train_embeddings, test_embeddings])
        
        # Get labels
        train_labels = np.array(self.train_data["label"])
        test_labels = np.array(self.test_data["label"])
        all_labels = np.concatenate([train_labels, test_labels])
        
        # Generate demographic features and diversity signals
        demographic_features = self.generate_demographic_features()
        diversity_signals = self.compute_diversity_signals(demographic_features)
        
        # Prepare final data structure
        data = {
            # Text embeddings
            "text_embeddings": all_embeddings.astype(np.float32),
            
            # Labels
            "labels": all_labels.astype(np.int64),
            
            # Demographic features for each view
            "demographic_features": demographic_features,
            
            # Diversity signals
            "diversity_signals": diversity_signals,
            
            # Index mappings
            "train_labeled_indices": np.array(self.train_labeled_indices, dtype=np.int64),
            "train_unlabeled_indices": np.array(self.train_unlabeled_indices, dtype=np.int64),
            "test_indices": np.array([i + len(self.train_data) for i in self.test_indices], dtype=np.int64),
            
            # Metadata
            "metadata": {
                "dataset_name": self.dataset_name,
                "k_shot": self.k_shot,
                "embedding_type": self.embedding_type,
                "num_demographic_profiles": self.num_demographic_profiles,
                "num_views": self.num_views,
                "view_names": self.view_names,
                "seed": self.seed,
                "total_samples": len(all_embeddings),
                "train_samples": len(self.train_data),
                "test_samples": len(self.test_data)
            }
        }
        
        print("Data preparation completed")
        return data
    
    def save_data(self, data: Dict[str, Any]) -> str:
        """Save prepared data to disk."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create filename
        filename = f"genfend_{self.dataset_name}_k{self.k_shot}_{self.embedding_type}.pt"
        output_path = os.path.join(self.output_dir, filename)
        
        # Save data
        torch.save(data, output_path)
        
        print(f"Data saved to: {output_path}")
        return output_path
    
    def build(self) -> str:
        """Complete data building pipeline."""
        print(f"Building GenFEND data for {self.dataset_name} ({self.k_shot}-shot)")
        
        # Load dataset
        self.load_dataset()
        
        # Sample data
        self.sample_data()
        
        # Prepare data
        data = self.prepare_data()
        
        # Save data
        output_path = self.save_data(data)
        
        # Clean up
        gc.collect()
        
        print(f"GenFEND data building completed: {output_path}")
        return output_path


def main():
    """Main function for building GenFEND data."""
    parser = ArgumentParser(description="Build GenFEND data")
    parser.add_argument("--dataset_name", choices=["politifact", "gossipcop"], 
                       default="politifact", help="Dataset name")
    parser.add_argument("--k_shot", type=int, choices=range(3, 17), default=8,
                       help="Number of shots")
    parser.add_argument("--embedding_type", choices=["bert", "roberta", "deberta", "distilbert"], 
                       default="deberta", help="Embedding type")
    parser.add_argument("--num_demographic_profiles", type=int, default=30,
                       help="Number of demographic profiles (default: 30 as in paper)")
    parser.add_argument("--num_views", type=int, default=3,
                       help="Number of demographic views (default: 3 for gender, age, education)")
    parser.add_argument("--partial_unlabeled", action="store_true", default=True,
                       help="Use partial unlabeled data")
    parser.add_argument("--sample_unlabeled_factor", type=int, default=5,
                       help="Factor for unlabeled sampling")
    parser.add_argument("--output_dir", default="data_genfend",
                       help="Output directory")
    parser.add_argument("--dataset_cache_dir", default="dataset",
                       help="Dataset cache directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Build GenFEND data
    builder = GenFENDDataBuilder(
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        embedding_type=args.embedding_type,
        num_demographic_profiles=args.num_demographic_profiles,
        num_views=args.num_views,
        partial_unlabeled=args.partial_unlabeled,
        sample_unlabeled_factor=args.sample_unlabeled_factor,
        output_dir=args.output_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        seed=args.seed
    )
    
    output_path = builder.build()
    print(f"Data saved to: {output_path}")


if __name__ == "__main__":
    main()