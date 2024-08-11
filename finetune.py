# import package
import os
import numpy as np
import pandas as pd
import torch
import evaluate
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from datasets import load_dataset
from argparse import ArgumentParser

# default values
id2label = {0: "fake", 1: "real"}
label2id = {"fake": 0, "real": 1}


def handle_id_label(dataset_name: str = "fake_news_tfg"):
    global id2label, label2id
    if dataset_name == "fake_news_tfg":
        id2label = {0: "fake", 1: "real"}
        label2id = {"fake": 0, "real": 1}
    elif dataset_name == "kdd2020":
        id2label = {1: "fake", 0: "real"}
        label2id = {"fake": 1, "real": 0}

    print(f"id2label: {id2label}")
    print(f"label2id: {label2id}")


def fetch_dataset(dataset_name: str = "fake_news_tfg"):
    # load data
    print("Loading dataset...")

    ## GonzaloA/fake_news & LittleFish-Coder/Fake-News-Detection-Challenge-KDD-2020
    """
    use the [`GonzaloA/fake_news`](https://huggingface.co/datasets/GonzaloA/fake_news) dataset from huggingface datasets library
    - 0: fake news
    - 1: real news

    use the [`LittleFish-Coder/Fake-News-Detection-Challenge-KDD-2020`](https://huggingface.co/datasets/LittleFish-Coder/Fake-News-Detection-Challenge-KDD-2020) dataset from huggingface datasets library
    - 1: fake news
    - 0: real news
    """
    if dataset_name == "fake_news_tfg":
        dataset_name = "GonzaloA/fake_news"
    elif dataset_name == "kdd2020":
        dataset_name = "LittleFish-Coder/Fake-News-Detection-Challenge-KDD-2020"

    dataset = load_dataset(dataset_name, download_mode="reuse_cache_if_exists")

    return dataset


def load_tokenizer(bert_model: str = "bert-base-uncased"):
    """
    load the tokenizer from huggingface transformers library
    """
    # load tokenizer
    print(f"Loading {bert_model} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    return tokenizer


def tokenize_data(dataset, tokenizer):
    """
    tokenize the dataset
    """
    print("Tokenizing data...")

    def tokenize_data(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_data, batched=True)
    return tokenized_dataset


def load_model(bert_model: str = "bert-base-uncased"):
    """
    load the model
    """
    print(f"Loading {bert_model} model...")
    model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=2, id2label=id2label, label2id=label2id)
    return model


def compute_metrics(eval_pred):
    """
    compute the accuracy and f1 score
    """

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")

    results = {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

    return results


def set_training_args(num_epochs: int = 5, batch_size: int = 64, output_dir: str = "checkpoints"):
    """
    set the training arguments
    """
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir="logs",
        logging_steps=10,
        save_steps=10,
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    return training_args


def set_trainer(model, training_args, tokenizer, tokenized_dataset):
    """
    set the trainer
    """

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer


def save_result(result, output_dir: str = "checkpoints", eval_type: str = "val"):
    """
    save the result
    """
    # create folder to save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save the result
    result = pd.DataFrame(result, index=[0])
    result.to_csv(f"{output_dir}/{eval_type}_result.csv", index=False)


def show_args(args, output_dir):
    print("========================================")
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print(f"\tOutput directory: {output_dir}")
    print("========================================")


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()
    # select model
    parser.add_argument(
        "--bert_model",
        type=str,
        default="distilbert-base-uncased",
        help="the bert model to use",
        choices=["bert-base-uncased", "distilbert-base-uncased", "roberta-base"],
    )
    # select dataset
    parser.add_argument("--dataset_name", type=str, default="fake_news_tfg", help="dataset to use", choices=["fake_news_tfg", "kdd2020"])

    # training arguments
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory to save checkpoints")

    args = parser.parse_args()
    bert_model = args.bert_model
    dataset_name = args.dataset_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_dir
    output_dir = f"{checkpoint_dir}/{dataset_name}/{bert_model}"

    # show arguments
    show_args(args, output_dir)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # load data
    dataset = fetch_dataset(dataset_name)

    # load tokenizer
    tokenizer = load_tokenizer(bert_model)

    # tokenize data
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    # handle id and label
    handle_id_label(dataset_name)

    # load model
    model = load_model(bert_model)

    # training arguments
    training_args = set_training_args(num_epochs, batch_size, output_dir)

    # set trainer
    trainer = set_trainer(model, training_args, tokenizer, tokenized_dataset)

    # train model
    print("Training model...")
    trainer.train()

    # save the model
    trainer.save_model(f"{output_dir}/best_model")

    # evaluate model
    print("Evaluating model...")
    val_result = trainer.evaluate()

    # evaluate model with testing set
    print("Testing model...")
    test_result = trainer.evaluate(eval_dataset=tokenized_dataset["test"])

    # save val and test results
    print("Saving results...")
    save_result(val_result, output_dir, "val")
    save_result(test_result, output_dir, "test")
