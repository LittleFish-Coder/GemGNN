import os
import gc
import json
import numpy as np
import torch
from typing import Dict
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, DatasetDict, Dataset
from argparse import ArgumentParser, Namespace
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from utils.sample_k_shot import sample_k_shot


# Constants
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_DATASET_NAME = "politifact"
DEFAULT_K_SHOT = 8
DEFAULT_NUM_EPOCHS = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 0.001
DEFAULT_MAX_LENGTH = 512
LOG_DIR = "logs"
DEFAULT_CACHE_DIR = "dataset"
DEFAULT_OUTPUT_DIR = "results"
SEED = 42


class FakeNewsTrainer:
    """
    A class to manage the training and evaluation of fake news detection models.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dataset_name: str = DEFAULT_DATASET_NAME,
        k_shot: int = DEFAULT_K_SHOT,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        self.model_name = self._redirect_model_name(model_name)
        # Determine model type from model name
        self.model_type = self._get_model_type(model_name)

        self.dataset_name = dataset_name
        self.k_shot = k_shot
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        # Add these new attributes
        self.selected_indices = None
        self.label_distribution = None
        
        # Setup directory paths using a single base output directory
        self.model_dir = os.path.join(self.output_dir, self.model_type, self.dataset_name, f"{self.k_shot}-shot")
        
        # Create directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize components
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def _redirect_model_name(self, model_name: str) -> str:
        """Redirect model name to the correct model name."""
        # ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "microsoft/deberta-base"],
        if "distilbert" in model_name:
            return "distilbert-base-uncased"
        elif "roberta" in model_name:
            return "roberta-base"
        elif "deberta" in model_name:
            return "microsoft/deberta-base"
        return "bert-base-uncased"
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine model type from model name."""
        if "distilbert" in model_name:
            return "distilbert"
        elif "roberta" in model_name:
            return "roberta"
        elif "deberta" in model_name:
            return "deberta"
        return "bert"
    
    def load_dataset(self) -> None:
        """Load and prepare the dataset."""
        print(f"Loading dataset '{self.dataset_name}'...")
        dataset = load_dataset(f"LittleFish-Coder/Fake_News_{self.dataset_name}", cache_dir=self.cache_dir)
        
        dataset = self._sample_k_shot(dataset, self.k_shot)
        
        self.dataset = dataset
        print(f"Dataset loaded and prepared. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    def _sample_k_shot(self, dataset: DatasetDict, k: int) -> DatasetDict:
        """Sample k examples per class for few-shot learning."""
        print(f"Sampling {k}-shot data per class...")
        
        train_data = dataset["train"]
        
        # Use the shared sampling function to ensure consistency with graph model
        selected_indices, sampled_data = sample_k_shot(train_data, k, seed=SEED)
        
        # Store the selected indices for later
        self.selected_indices = selected_indices
        
        # Calculate and store label distribution
        self.label_distribution = {}
        for idx in selected_indices:
            label = train_data["label"][idx]
            self.label_distribution[label] = self.label_distribution.get(label, 0) + 1
        
        # Create new dataset with sampled training data
        return DatasetDict({
            "train": Dataset.from_dict(sampled_data),
            "test": dataset["test"],
        })
    
    def setup_model_and_tokenizer(self) -> None:
        """Set up the model and tokenizer."""
        print(f"Setting up tokenizer and model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize dataset
        tokenized_dataset = self.dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=DEFAULT_MAX_LENGTH,
            ),
            batched=True
        )
        
        # Label mappings
        id2label = {0: "real", 1: "fake"}
        label2id = {"real": 0, "fake": 1}
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2, 
            id2label=id2label, 
            label2id=label2id
        )
        
        # Set up trainer
        training_args = TrainingArguments(
            output_dir=os.path.join(self.model_dir, "checkpoints"),
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir=LOG_DIR,
            logging_steps=10,
            seed=SEED,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=self._compute_metrics,
        )
        
        print("Model and tokenizer setup complete")
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        cm = confusion_matrix(labels, predictions)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "precision": precision_score(labels, predictions, average="macro", zero_division=0),
            "recall": recall_score(labels, predictions, average="macro", zero_division=0),
            "confusion_matrix": cm.tolist(),  # Convert numpy array to list for JSON serialization
        }
    
    def train(self, cleanup_checkpoints=True) -> None:
        """Train the model."""
        print(f"Starting training for {self.num_epochs} epochs...")
        
        # Train the model
        self.trainer.train()
        
        # Save the best model and tokenizer
        model_save_dir = os.path.join(self.model_dir, "model")
        self.trainer.save_model(model_save_dir)
        self.tokenizer.save_pretrained(model_save_dir)

        # Cleanup checkpoint files to save disk space
        if cleanup_checkpoints:
            checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                import shutil
                print(f"Cleaning up checkpoint files in {checkpoint_dir}")
                shutil.rmtree(checkpoint_dir)
        
        print("Training completed")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model and save metrics."""
        print("Evaluating model...")
        
        # Evaluate on test set
        metrics = self.trainer.evaluate()
        
        # Extract relevant metrics
        results = {
            "accuracy": metrics["eval_accuracy"],
            "f1": metrics["eval_f1"],
            "precision": metrics["eval_precision"],
            "recall": metrics["eval_recall"],
            "confusion_matrix": metrics.get("eval_confusion_matrix", "N/A"),
        }
        
        # Save metrics to results directory
        metrics_file = os.path.join(self.model_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation completed. Results saved to {metrics_file}")

        # Add this code to save indices
        if self.selected_indices is not None:
            indices_file = os.path.join(self.model_dir, "indices.json")
            
            # Convert numpy types to Python native types
            indices_info = {
                "indices": [int(i) for i in self.selected_indices],
                "k_shot": int(self.k_shot),
                "seed": int(SEED),
                "dataset_name": self.dataset_name,
            }
            
            if hasattr(self, 'label_distribution') and self.label_distribution:
                indices_info["label_distribution"] = {
                    int(k): int(v) for k, v in self.label_distribution.items()
                }
            
            with open(indices_file, "w") as f:
                json.dump(indices_info, f, indent=2)
            
            print(f"Selected indices saved to {indices_file}")


        return results
    
    def run_pipeline(self) -> Dict[str, float]:
        """Run the complete training and evaluation pipeline."""
        self.load_dataset()
        self.setup_model_and_tokenizer()
        self.train()
        return self.evaluate()

    def push_to_hub(self, repo_name: str):
        print(f"Pushing model and tokenizer to Huggingface Hub: {repo_name}")
        self.model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)

def set_seed(seed: int = SEED) -> None:
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Optional: Set random seed for transformers library
    try:
        import transformers
        transformers.set_seed(seed)
    except:
        pass

def parse_arguments() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Fine-tune language models for fake news detection")
    
    # Model selection
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model to use (default: {DEFAULT_MODEL_NAME})",
        choices=["bert", "distilbert", "roberta", "deberta"],
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help=f"Dataset to use (default: {DEFAULT_DATASET_NAME})",
        choices=["gossipcop", "politifact"],
    )
    
    # Few-shot setting
    parser.add_argument(
        "--k_shot",
        type=int,
        default=DEFAULT_K_SHOT,
        help=f"Number of samples per class for few-shot learning (default: {DEFAULT_K_SHOT})",
        choices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    )
    
    # Training parameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_NUM_EPOCHS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay (default: 0.01)",
    )
    
    # Single output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save models and results (default: results)",
    )

    parser.add_argument(
        "--keep_checkpoints",
        action="store_true",
        help="Keep intermediate checkpoints (warning: uses a lot of disk space)",
    )
    
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the trained model to Huggingface Hub after training",
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="dataset",
        help="Cache directory for datasets and models (default: dataset)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run the fine-tuning pipeline."""
    # Set seed for reproducibility
    set_seed()

    # Clean up CUDA memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Parse arguments
    args = parse_arguments()
    
    # Display arguments and hardware info
    print("\n" + "="*50)
    print("Fake News Detection - Model Fine-tuning")
    print("="*50)
    print(f"Model:         {args.model_name}")
    print(f"Dataset:       {args.dataset_name}")
    print(f"K-shot:        {args.k_shot}")
    print(f"Epochs:        {args.num_epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay:   {args.weight_decay}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Device:         {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print("="*50 + "\n")
    
    # Create and run trainer
    trainer = FakeNewsTrainer(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        cache_dir=args.cache_dir,
    )
    
    # Run the pipeline
    metrics = trainer.run_pipeline()
    
    # Display final results
    print("\n" + "="*50)
    print("Training Complete - Results")
    print("="*50)
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    if metrics.get("confusion_matrix") != "N/A":
        print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
    print("="*50 + "\n")

    # Push to hub if requested
    if args.push_to_hub:
        repo_name = f"{args.dataset_name}_pseudo_labeler"
        trainer.push_to_hub(repo_name)
        print(f"Model and tokenizer pushed to Huggingface Hub as '{repo_name}'")


if __name__ == "__main__":
    main()