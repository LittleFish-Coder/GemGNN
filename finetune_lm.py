import os
import numpy as np
import pandas as pd
import torch
import evaluate
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from datasets import load_dataset, DatasetDict
from argparse import ArgumentParser, Namespace


def parse_arguments() -> Namespace:
    """Parses command-line arguments.

    Returns:
        Namespace: Parsed arguments with their values.
    """

    parser = ArgumentParser(description="Run fine-tuning on text classification model.")
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="the model to use",
        choices=["bert-base-uncased", "bart-base", "roberta-base"],
    )
    # dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="KDD2020",
        help="dataset to use",
        choices=["TFG", "KDD2020", "GossipCop", "PolitiFact"],
    )
    # training arguments
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="directory to save checkpoints",
    )
    return parser.parse_args()


def show_args(args: Namespace, output_dir: str) -> None:
    """Displays parsed arguments.

    Args:
        args (Namespace): Parsed arguments from the command line.
        output_dir (str): Directory where output will be saved.
    """

    print("========================================\n")
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print(f"\tOutput directory: {output_dir}")
    print("\n========================================\n")


def fetch_dataset(dataset_name: str, dataset_size: str) -> DatasetDict:
    """
    Fetches the dataset from a local directory based on the provided name and size.

    Args:
        dataset_name (str): The name of the dataset to fetch (folder under `dataset/`).
        dataset_size (str): The size of the dataset to load (e.g., "full", "10%", "100").

    Returns:
        DatasetDict: A dictionary-like object containing train, test, and optionally other splits.
    """
    print(f"Fetching dataset '{dataset_name}' with size '{dataset_size}'...\n")

    dataset: DatasetDict = load_dataset(
        f"LittleFish-Coder/Fake_News_{dataset_name}",
        download_mode="reuse_cache_if_exists",
        cache_dir="dataset",
    )
    print(f"\nDataset: {dataset}")
    train_dataset = dataset["train"]

    # quick look at the data
    first_train = train_dataset[0]
    print("\nFirst training sample: ")
    print(f"\tKeys: {first_train.keys()}")
    print(f"\tText: {first_train['text']}")
    print(f"\tLabel: {first_train['label']}")
    print("\n========================================\n")

    return dataset


def load_tokenizer(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model)


if __name__ == "__main__":

    args = parse_arguments()

    # Configuration
    model = args.model
    dataset_name = args.dataset_name
    dataset_size = "full"  # task default
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_dir
    output_dir = f"{checkpoint_dir}/{dataset_name}_{dataset_size}/{model}"

    # show arguments
    show_args(args, output_dir)

    # load data
    dataset = fetch_dataset(dataset_name, dataset_size)

    # load tokenizer
    tokenizer = load_tokenizer(model)
