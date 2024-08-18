import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


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


# Load the dataset
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

    dataset = load_dataset(dataset_name, download_mode="reuse_cache_if_exists", cache_dir="dataset")

    print(f"Dataset size: ")
    print(f"\tTrain: {len(dataset['train'])}")
    print(f"\tValidation: {len(dataset['validation'])}")
    print(f"\tTest: {len(dataset['test'])}")

    return dataset["test"]


def tokenize_data(dataset, tokenizer):
    """
    tokenize the dataset
    """
    print("Tokenizing data...")

    def tokenize_data(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, return_tensors="pt")

    tokenized_dataset = dataset.map(tokenize_data, batched=True)

    return tokenized_dataset


def load_model_tokenizer(model_name: str = "roberta-base-fake-news-tfg"):
    """
    load the model
    """
    print(f"Loading {model_name} model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"LittleFish-Coder/{model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(f"LittleFish-Coder/{model_name}").to(device)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    return model, tokenizer


def get_predictions(model, tokenized_dataset, batch_size=64):
    """
    get the predictions
    """
    print(f"Getting predictions on {len(tokenized_dataset)} samples...")

    input_ids = torch.tensor(tokenized_dataset["input_ids"])
    attention_mask = torch.tensor(tokenized_dataset["attention_mask"])

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    predictions = []
    for batch in tqdm(dataloader):
        batch_input_ids, batch_attention_mask = [t.to(device) for t in batch]
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits

        predicted_class_ids = logits.argmax(dim=-1).cpu().numpy()
        batch_predictions = [model.config.id2label[class_id] for class_id in predicted_class_ids]
        predictions.extend(batch_predictions)

    return predictions


def get_metrics(predictions, real_dataset):
    """
    get the metrics
    """
    print("Getting metrics...")

    # get the labels
    labels = real_dataset["label"]
    labels = [id2label[label] for label in labels]

    # get the metrics
    correct_predictions = 0

    for real_label, predicted_label in zip(labels, predictions):
        if real_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(labels)

    return accuracy


def show_args(args):
    print("========================================")
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print("========================================")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()
    # select model
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base-fake-news-tfg",
        help="the bert model to use",
    )
    # dataset arguments
    parser.add_argument("--dataset_name", type=str, default="kdd2020", help="dataset to use", choices=["fake_news_tfg", "kdd2020"])
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for predictions")

    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    batch_size = args.batch_size

    # show arguments
    show_args(args)

    # handle id2label and label2id
    handle_id_label(args.dataset_name)

    # load the dataset
    dataset = fetch_dataset(dataset_name)

    # load the model and tokenizer
    model, tokenizer = load_model_tokenizer(model_name)

    # tokenize the dataset
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    # get the predictions
    predictions = get_predictions(model, tokenized_dataset, batch_size)

    # get the metrics
    accuracy = get_metrics(predictions, dataset)

    # print the results
    print("========================================")
    print("Results:")
    print(f"\tUsing model: {model_name}")
    print(f"\tTest on: {dataset_name}")
    print(f"\tAccuracy: {accuracy}")
    print("========================================")
