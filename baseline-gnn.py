import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import torch_geometric
from datasets import load_dataset, Dataset as HF_Dataset, DatasetDict
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from torchsummary import summary
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EmbeddingsDataset:
    def __init__(self, dataset_dict: DatasetDict):
        train_data = dataset_dict["train"]
        test_data = dataset_dict["test"]
        self.train_embeddings = np.array(train_data["embeddings"])
        self.train_labels = np.array(train_data["label"])
        self.test_embeddings = np.array(test_data["embeddings"])
        self.test_labels = np.array(test_data["label"])

    def get_embeddings_and_labels(self):
        return (
            self.train_embeddings,
            self.train_labels,
            self.test_embeddings,
            self.test_labels,
        )


class EmbeddingsGraph:
    def __init__(self, train_embeddings, train_labels, test_embeddings, test_labels):
        self.num_nodes = len(train_embeddings) + len(test_embeddings)
        self.num_features = train_embeddings.shape[1]

        # train shape: (k_shot, num_features: 768)
        # test shape: (num_samples, num_features: 768)

        # merge train and test embeddings
        self.x = torch.tensor(
            np.concatenate([train_embeddings, test_embeddings]), dtype=torch.float
        )
        self.y = torch.tensor(
            np.concatenate([train_labels, test_labels]), dtype=torch.long
        )

        # create train and test masks
        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.train_mask[: len(train_embeddings)] = True
        self.test_mask[len(train_embeddings) :] = True

        # build knn graph
        self.edge_index, self.edge_attr = self.build_knn_graph(k=5)

        print(self.get_graph())

    def build_knn_graph(self, k: int):
        embeddings = self.x.cpu().numpy()
        knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(
            embeddings
        )  # use k+1 to account for self-loops
        distances, indices = knn.kneighbors(embeddings)

        # create edge_index and edge_attr
        row = np.repeat(np.arange(len(embeddings)), k + 1)
        col = indices.flatten()
        data = distances.flatten()

        # filter out self-loops
        mask = row != col
        row = row[mask]
        col = col[mask]
        data = data[mask]

        # convert distances to similarities
        similarities = 1 - data
        edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)
        edge_attr = torch.tensor(similarities, dtype=torch.float).unsqueeze(1)

        return edge_index, edge_attr

    def get_graph(self):
        return Data(
            x=self.x,
            y=self.y,
            train_mask=self.train_mask,
            test_mask=self.test_mask,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
        )

    def __str__(self):
        return self.get_graph().__str__()


class GNNModel(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        model_type: str = "GCN",
        out_channels: int = 2,  # binary classification
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout

        if model_type == "GCN":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == "GAT":
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
            self.conv2 = GATConv(
                hidden_channels * 4,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
            )

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def set_args() -> Namespace:
    """Sets up the command-line arguments.

    Returns:
        Namespace: Parsed arguments from the command line.
    """
    parser = ArgumentParser()

    # important arguments (dataset, k_shot, k_neighbors, model_type)
    parser.add_argument(
        "--dataset",
        type=str,
        default="PolitiFact",
        choices=["PolitiFact", "GossipCop", "KDD2020", "TFG"],
    )
    parser.add_argument(
        "--k_shot", type=int, default=8, choices=[0, 8, 16, 32, 100]
    )  # 0 represents full dataset
    parser.add_argument("--k_neighbors", type=int, default=5, choices=[5, 7, 9])
    parser.add_argument("--model_type", type=str, default="GCN", choices=["GCN", "GAT"])

    # optional arguments
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="directory to save checkpoints",
    )
    args = parser.parse_args()
    args.output_dir = f"{args.checkpoint_dir}/{args.dataset}/{args.model_type}"
    args.device = device

    # show arguments
    print("========================================")
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print("========================================\n")

    return args


def sample_k_shots(dataset: DatasetDict, k: int) -> DatasetDict:
    """
    Samples k-shot data per class from the original training set.

    Args:
        dataset (DatasetDict): The dataset to sample from.
        k (int): The number of samples per class.

    Returns:
        DatasetDict: The sampled dataset.
    """
    if k == 0:
        print("Using full dataset...")
        print(f"Sampled train dataset size: {len(dataset['train'])}")
        print(f"Sampled test dataset size: {len(dataset['test'])}")
        return dataset  # full dataset

    train_data = dataset["train"]
    sampled_data = defaultdict(list)

    labels = set(train_data["label"])
    for label in labels:
        label_data = train_data.filter(lambda x: x["label"] == label)
        sampled_label_data = label_data.shuffle(seed=42).select(
            range(min(k, len(label_data)))
        )
        for key in train_data.column_names:
            sampled_data[key].extend(sampled_label_data[key])
        print(f"Class {label}: {len(sampled_label_data)} samples")

    sampled_dataset = DatasetDict(
        {
            "train": HF_Dataset.from_dict(sampled_data),
            "test": dataset["test"],
        }
    )

    # after sampling: each class should have k samples
    print(f"Sampled train dataset size: {len(dataset['train'])}")
    print(f"Sampled test dataset size: {len(dataset['test'])}")
    print()

    return sampled_dataset


def train_model(model, data: torch_geometric.data.Data, args) -> float:  # type: ignore
    # set device
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    model.to(device)
    data.to(device)

    # train and test
    print("\nStarting training...")

    best_test_acc = 0
    best_model_state_dict = None
    for epoch in tqdm(range(args.epochs)):
        loss = train(model, data, optimizer)
        train_acc, test_acc = test(model, data)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f"{args.output_dir}/model.pth")  # save model
            best_model_state_dict = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d}, Loss: {loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
            )

    print(f"\nTraining completed!")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")

    return best_test_acc


def train(model: torch.nn.Module, data: torch_geometric.data.Data, optimizer: torch.optim.Optimizer):  # type: ignore
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model: torch.nn.Module, data: torch_geometric.data.Data) -> Tuple[float, float]:  # type: ignore
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        train_correct = (
            out[data.train_mask].argmax(dim=1).eq(data.y[data.train_mask]).sum()
        )
        test_correct = (
            out[data.test_mask].argmax(dim=1).eq(data.y[data.test_mask]).sum()
        )
        train_acc = train_correct.item() / data.train_mask.sum().item()
        test_acc = test_correct.item() / data.test_mask.sum().item()
    return train_acc, test_acc


def save_results(args: Namespace, best_test_acc: float):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df_path = os.path.join(args.output_dir, "results.csv")
    if not os.path.exists(df_path):
        df = pd.DataFrame(
            columns=["k_shot", "k_neighbors", "model_type", "best_test_acc"]
        )
    else:
        df = pd.read_csv(df_path)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "k_shot": args.k_shot,
                        "k_neighbors": args.k_neighbors,
                        "model_type": args.model_type,
                        "best_test_acc": best_test_acc,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df.to_csv(df_path, index=False)


def main():
    """
    PIPELINE
    0. Set arguments
    1. Load dataset
    2. Sample k-shot data
    3. Prepare embeddings data
    4. Construct graph (Edge Creation: KNN)
    5. Initialize model (GCN or GAT)
    6. Train
    7. Test
    """

    # set seed
    set_seed(42)

    # set arguments
    args = set_args()

    # load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset: DatasetDict = load_dataset(f"LittleFish-Coder/Fake_News_{args.dataset}", cache_dir="dataset")  # type: ignore

    # sample k-shot data
    print(f"Sampling {args.k_shot}-shot data per class...")
    dataset = sample_k_shots(dataset, args.k_shot)

    # prepare embeddings data
    embeddings_dataset = EmbeddingsDataset(dataset)
    train_embeddings, train_labels, test_embeddings, test_labels = (
        embeddings_dataset.get_embeddings_and_labels()
    )

    # construct embeddings graph (Edge is empty)
    print("Constructing embeddings graph...")
    graph = EmbeddingsGraph(
        train_embeddings, train_labels, test_embeddings, test_labels
    ).get_graph()

    # initialize model
    print(f"\nInitializing {args.model_type} model...")
    model = GNNModel(
        in_channels=graph.num_features,  # BERT embeddings size: 768
        hidden_channels=args.hidden_channels,
        model_type=args.model_type,
    ).to(device)

    # train and test
    best_test_acc = train_model(model, graph, args)

    # save best test accuracy
    save_results(args, best_test_acc)


if __name__ == "__main__":
    main()
