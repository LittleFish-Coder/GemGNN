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
from typing import Dict


def parse_arguments() -> Namespace:
    """Parses command-line arguments.

    Returns:
        Namespace: Parsed arguments with their values.
    """

    parser = ArgumentParser(description="Run fine-tuning on text classification model.")
    parser.add_argument(
        "--model_name",
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


def load_tokenizer(model_name: str, dataset: Dataset) -> AutoTokenizer:
    """
    Loads a tokenizer for the specified model and applies it to the first example of the specified split.

    Args:
        model_name (str): The model identifier for loading the tokenizer.
        dataset (Dataset): The dataset to be used for tokenization.

    Returns:
        AutoTokenizer: The tokenizer used for processing the dataset.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading {model_name} tokenizer...")
    print("\n========================================\n")
    return tokenizer


def tokenize_data(
    dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int
) -> DatasetDict:
    """
    Tokenizes the text data in the dataset using the provided tokenizer.

    Args:
        dataset (DatasetDict): The dataset to tokenize.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): Maximum sequence length for padding and truncation.

    Returns:
        DatasetDict: The tokenized dataset.
    """

    print("Tokenizing the dataset...\n")

    def tokenize_data(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_data, batched=True)

    # Tokenize the first example text
    first_tokenized = tokenized_dataset["train"][0]
    print("\nFirst tokenized sample: ")
    print(f"\tKeys: {first_tokenized.keys()}")
    print(f"\tInput IDs: {first_tokenized['input_ids']}")
    print(f"\tAttention Mask: {first_tokenized['attention_mask']}")
    print(f"\tLength: {len(first_tokenized['input_ids'])}")
    print("\n========================================\n")
    return tokenized_dataset


def load_model(
    model_name: str, id2label: Dict[int, str], label2id: Dict[str, int], num_labels: int
) -> AutoModelForSequenceClassification:
    """
    Loads a pre-trained model for sequence classification with optional custom label mappings.

    Args:
        model_name (str): The name or path of the pre-trained model to load.
        id2label (dict): A mapping from label indices to label names.
        label2id (dict): A mapping from label names to label indices.

    Returns:
        AutoModelForSequenceClassification: The loaded pre-trained model.
    """

    print(f"Loading {model_name} model...\n")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    print("\n========================================\n")
    return model


def compute_metrics(eval_pred):
    """
    Computes evaluation metrics including accuracy, F1, precision, and recall.

    Args:


    Returns:

    """

    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    acc = accuracy.compute(predictions=predictions, references=labels)

    f1_score = f1.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    pre = precision.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    rec = recall.compute(predictions=predictions, references=labels, average="weighted")

    # Handle potential None values
    results = {
        "accuracy": acc["accuracy"] if acc else None,
        "f1": f1_score["f1"] if f1_score else None,
        "precision": pre["precision"] if pre else None,
        "recall": rec["recall"] if rec else None,
    }
    return results


def set_training_args(
    num_epochs: int,
    batch_size: int,
    output_dir: str,
    logging_dir: str,
    learning_rate: float,
    weight_decay: float,
) -> TrainingArguments:
    """
    Configures and returns the training arguments for the model.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and evaluation.
        output_dir (str): Directory to save model checkpoints and outputs.
        logging_dir (str): Directory to save logs.
        learning_rate (float): Learning rate for the optimizer
        weight_decay (float): Weight decay for regularization

    Returns:
        TrainingArguments: Configured training arguments.
    """

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=logging_dir,
        logging_steps=1,
    )
    print("Training arguments successfully configured.")
    print("\n========================================\n")
    return training_args


def set_trainer(
    model: AutoModelForSequenceClassification,
    training_args: TrainingArguments,
    tokenized_dataset: DatasetDict,
) -> Trainer:
    """
    Configures and returns a Trainer instance.

    Args:
        model (AutoModelForSequenceClassification): The pre-trained model to fine-tune.
        training_args (TrainingArguments): The training arguments configured via TrainingArguments.
        tokenized_dataset (DatasetDict): Tokenized dataset containing "train" and "test" splits.

    Returns:
        Trainer: Configured Trainer instance.
    """

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )
    print("Trainer successfully configured.")
    print("\n========================================\n")
    return trainer


if __name__ == "__main__":
    import numpy

    print(numpy.__version__)

    args = parse_arguments()

    # Configuration
    model_name = args.model_name
    dataset_name = args.dataset_name
    dataset_size = "full"  # task default
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_dir
    logging_dir = "logs"
    output_dir = f"{checkpoint_dir}/{dataset_name}_{dataset_size}/{model_name}"

    # show arguments
    show_args(args=args, output_dir=output_dir)

    # load data
    dataset = fetch_dataset(dataset_name=dataset_name, dataset_size=dataset_size)

    # load tokenizer
    tokenizer = load_tokenizer(model_name=model_name, dataset=dataset)

    # tokenize data
    tokenized_dataset = tokenize_data(
        dataset=dataset, tokenizer=tokenizer, max_length=512
    )

    # load model
    id2label = {0: "real", 1: "fake"}
    label2id = {"real": 0, "fake": 1}
    model = load_model(
        model_name=model_name, id2label=id2label, label2id=label2id, num_labels=2
    )

    # training arguments
    learning_rate = 2e-5
    weight_decay = 0.01
    training_args = set_training_args(
        num_epochs=num_epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        logging_dir=logging_dir,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # set trainer
    trainer = set_trainer(
        model=model,
        training_args=training_args,
        tokenized_dataset=tokenized_dataset,
    )

    print("Training model...")
    trainer.train()
    # # save the best model
    # trainer.save_model(f"{output_dir}/best_model")
    # # save the tokenizer
    # tokenizer.save_pretrained(f"{output_dir}/best_model")

    # # evaluate on test set
    # test_result = trainer.evaluate(eval_dataset=tokenized_dataset["test"])

    # test_df = pd.DataFrame(test_result, index=[0])
    # test_df.to_csv(f"{output_dir}/test_result.csv", index=False)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Device: {device}")

    # test_dataset = dataset["test"]
    # text = test_dataset[0]["text"]
    # print(f"Text: {text}")
    # classifier = pipeline(
    #     "text-classification",
    #     model=f"{output_dir}/best_model",
    #     truncation=True,
    #     device=device,
    # )
    # classifier(text)

    # tokenizer = AutoTokenizer.from_pretrained(f"{output_dir}/best_model")
    # inputs = tokenizer(text, return_tensors="pt", truncation=True)
    # print(f"Input keys: {inputs.keys()}")
    # print(f"Input: {inputs}")

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     f"{output_dir}/best_model"
    # )
    # with torch.no_grad():
    #     logits = model(**inputs).logits
    # print(f"Logits: {logits}")

    # predicted_class_id = logits.argmax().item()
    # print(model.config.id2label[predicted_class_id])
