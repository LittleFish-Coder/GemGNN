import torch
from transformers import pipeline
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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

    # only fetech the test dataset
    dataset = load_dataset(dataset_name, download_mode="reuse_cache_if_exists", cache_dir="dataset")

    print(f"Dataset size: ")
    print(f"\tTest: {len(dataset['test'])}")

    return dataset["test"]


def load_model(model_name: str = "facebook/bart-large-mnli"):
    """
    load the model
    """
    print(f"Loading {model_name} model...")
    model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    return model


def get_predictions(model, dataset):
    """
    get the predictions
    """
    candidate_labels = ["fake", "real"]
    print(f"Getting predictions on {len(dataset)} samples...")

    results = model(dataset['text'], candidate_labels)

    predictions = [label2id[result["labels"][0]] for result in results]

    return predictions


def get_metrics(real_labels, pred_labels):
    """
    get the metrics
    """
    print("Getting metrics...")

    accuracy = accuracy_score(real_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(real_labels, pred_labels, average='weighted')

    return accuracy, precision, recall, f1


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
        default="facebook/bart-large-mnli",
        help="the model to use",
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
    handle_id_label(dataset_name)

    # load the dataset
    dataset = fetch_dataset(dataset_name)

    # load the model and tokenizer
    model = load_model(model_name)

    # get the predictions
    predictions = get_predictions(model, dataset)

    # get the metrics
    accuracy, precision, recall, f1 = get_metrics(dataset['label'], predictions)

    # print the results
    print("========================================")
    print("Results:")
    print(f"\tUsing model: {model_name}")
    print(f"\tTest on: {dataset_name}")
    print(f"\tAccuracy: {accuracy:.4f}")
    print(f"\tPrecision: {precision:.4f}")
    print(f"\tRecall: {recall:.4f}")
    print(f"\tF1-score: {f1:.4f}")
    print("========================================")
