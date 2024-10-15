import random
import torch
from datasets import load_dataset
from model.edgebuilder import KNNGraphBuilder, ThresholdNNGraphBuilder
from model.customdataset import EmbeddingsDataset, EmbeddingsGraph
from matplotlib.patches import Patch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from argparse import ArgumentParser
import torch_geometric.utils as pyg_utils


def show_args(args, output_dir):
    print("========================================")
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print(f"\tOutput directory: {output_dir}")
    print("========================================")


def save_graph(graph, output_dir: str, graph_path: str):
    """
    save the graph
    """
    # create folder to save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save the graph
    torch.save(graph, graph_path)
    print(f"Graph saved to {graph_path}")


def fetch_dataset(dataset_name: str):
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

    dataset = load_dataset(dataset_name)

    print(f"Original Dataset size: ")
    print(f"\tTrain: {len(dataset['train'])}")  # type: ignore
    print(f"\tValidation: {len(dataset['validation'])}")    # type: ignore
    print(f"\tTest: {len(dataset['test'])}")    # type: ignore

    return dataset


def load_dataset_from_huggingface():
    # load and download the dataset from huggingface
    print("Load and download the dataset from huggingface...")
    dataset = load_dataset(
        "LittleFish-Coder/Fake-News-Detection-Challenge-KDD-2020",
        download_mode="reuse_cache_if_exists",
        cache_dir="dataset",
    )
    print(f"Dataset Type: {type(dataset)}")
    print(f"{dataset}")
    print(f"Dataset keys: {dataset.keys()}")    # type: ignore

    train_dataset = dataset["train"]    # type: ignore
    val_dataset = dataset["validation"]   # type: ignore
    test_dataset = dataset["test"]  # type: ignore
    print(f"Train dataset type: {type(train_dataset)}")
    print(f"Validation dataset type: {type(val_dataset)}")
    print(f"Test dataset type: {type(test_dataset)}")

    # First element of the train dataset
    print(f"{train_dataset[0].keys()}")
    print(f"Text: {train_dataset[0]['text']}")
    print(f"Label: {train_dataset[0]['label']}")
    print("\n\n")
    return train_dataset, val_dataset, test_dataset


def get_embeddings_dataset(dataset_name, dataset, size):
    if dataset_name == "kdd2020":
        embeddings_dataset = EmbeddingsDataset(texts=dataset["text"], labels=dataset["label"], embeddings=dataset["embeddings"], size=size, dataset_name=dataset_name,)
    else:
        embeddings_dataset = EmbeddingsDataset(texts=dataset["text"], labels=dataset["label"], size=size, dataset_name=dataset_name)
    print(f"Custom Embeddings Dataset length: {len(embeddings_dataset)}")
    return embeddings_dataset


# def random_choice_labeled_node(G, len_of_train_dataset, labeled_size):
#     # random choice labeled indices
#     random_indices = torch.randperm(len_of_train_dataset)[:labeled_size]
#     labeled_mask = torch.zeros(G.num_nodes, dtype=torch.bool)
#     labeled_mask[random_indices] = True
#     G.labeled_mask = labeled_mask
#     return G


def generate_embeddings_graph(embeddings_train_dataset, embeddings_val_dataset, embeddings_test_dataset, labeled_size):
    print("Generate custom graph...")
    custom_graph = EmbeddingsGraph(embeddings_train_dataset, embeddings_val_dataset, embeddings_test_dataset, labeled_size)
    graph_data = custom_graph.get_graph()
    print()
    print(f"Graph: {custom_graph}")
    print(f"Graph data: {graph_data}")
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of features: {graph_data.num_features}")
    print(f"Number of edges: {graph_data.num_edges}")
    print(f"Number of training nodes: {graph_data.train_mask.sum()}")
    print(f"Number of validation nodes: {graph_data.val_mask.sum()}")
    print(f"Number of test nodes: {graph_data.test_mask.sum()}")
    print(f"Number of labeled node: {graph_data.labeled_mask.sum()}")
    print(f"len of train mask: {len(graph_data.train_mask)}")
    print(f"len of val mask: {len(graph_data.val_mask)}")
    print(f"len of test mask: {len(graph_data.test_mask)}")
    print(f"len of labeled mask: {len(graph_data.labeled_mask)}")
    print()

    return custom_graph


def analyze_graph(graph, output_dir: str, graph_info_path: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    """analyze the graph"""
    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    labeled_mask = graph.labeled_mask
    edge_index = graph.edge_index

    total_nodes = graph.num_nodes
    total_edges = graph.num_edges
    train_nodes = train_mask.sum().item()
    val_nodes = val_mask.sum().item()
    test_nodes = test_mask.sum().item()
    labeled_nodes = labeled_mask.sum().item()
    graph_metric = graph.graph_metric

    # analyze the edge types
    edge_types = {"train-train": 0, "train-val": 0, "train-test": 0, "val-val": 0, "val-test": 0, "test-test": 0,}

    for edge in tqdm(edge_index.t(), desc="Analyzing edges"):
        source, target = edge
        if train_mask[source] and train_mask[target]:
            edge_types["train-train"] += 1
        elif (train_mask[source] and val_mask[target]) or (
            val_mask[source] and train_mask[target]
        ):
            edge_types["train-val"] += 1
        elif (train_mask[source] and test_mask[target]) or (
            test_mask[source] and train_mask[target]
        ):
            edge_types["train-test"] += 1
        elif val_mask[source] and val_mask[target]:
            edge_types["val-val"] += 1
        elif (val_mask[source] and test_mask[target]) or (
            test_mask[source] and val_mask[target]
        ):
            edge_types["val-test"] += 1
        elif test_mask[source] and test_mask[target]:
            edge_types["test-test"] += 1

    with open(f"{graph_info_path}", "w") as f:
        f.write(f"Total nodes: {total_nodes}\n")
        f.write(f"Total edges: {total_edges}\n")
        f.write(f"Training nodes: {train_nodes}\n")
        f.write(f"Validation nodes: {val_nodes}\n")
        f.write(f"Test nodes: {test_nodes}\n")
        f.write(f"Labeled nodes: {labeled_nodes}\n")
        f.write("\nEdge types:\n")
        for edge_type, count in edge_types.items():
            f.write(f"{edge_type}: {count}\n")

        try: 
            f.write("\nStat of Graph Metrics:\n")
            f.write(f"{graph_metric}")
        except Exception as e:
            pass


def construct_graph_edge(graph_data, k, edge_policy, theshold_factor=1.0):
    # Assuming x, y, train_mask, val_mask, test_mask are already defined
    builder = None
    graph = None
    if edge_policy == "knn":
        builder = KNNGraphBuilder(graph=graph_data, k=k)
    elif edge_policy == "thresholdnn":
        builder = ThresholdNNGraphBuilder(graph=graph_data, k=k, threshold_factor=theshold_factor)
    else:   # default set edge policy as thresholdnn
        builder = ThresholdNNGraphBuilder(graph=graph_data, k=k, threshold_factor=theshold_factor)

    graph = builder.build_graph(graph_data)

    return graph


def visualize_graph(graph_data, show_num_nodes='full', plot_dir="plot", graph_name="graph"):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if show_num_nodes == 'full':
        show_num_nodes = graph_data.num_nodes
    else:
        show_num_nodes = int(show_num_nodes)

    # convert to networkx graph
    G = pyg_utils.to_networkx(graph_data, to_undirected=True)

    train_mask = graph_data.train_mask.numpy()
    test_mask = graph_data.test_mask.numpy()
    val_mask = graph_data.val_mask.numpy()
    labeled_mask = graph_data.labeled_mask.numpy()

    color_map = {
        "train": "orange",
        "val": "blue",
        "test": "red",
        "labeled": "green"
    }

    # Initialize a color array for each node
    node_colors = []

    # Assign colors based on masks
    for i in range(graph_data.num_nodes):
        if labeled_mask[i]:
            node_colors.append(color_map["labeled"])  # Give priority to labeled nodes
        elif train_mask[i]:
            node_colors.append(color_map["train"])
        elif test_mask[i]:
            node_colors.append(color_map["test"])
        elif val_mask[i]:
            node_colors.append(color_map["val"])
        else:
            node_colors.append("gray")  # Default color for nodes outside the masks

    nx.draw_networkx(G, with_labels=False, node_color=node_colors, edge_color='gray', node_size = 3, alpha=0.7)
    legend_elements = [
        Patch(facecolor='orange', label='train'),
        Patch(facecolor='blue', label='val'),
        Patch(facecolor='red', label='test'),
        Patch(facecolor='green', label='labeled')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(f"{graph_name.replace("_", " ")}\n{show_num_nodes} nodes")
    plt.savefig(f"{plot_dir}/{graph_name}_{show_num_nodes}_nodes.png")
    plt.show()


if __name__ == "__main__":

    parser = ArgumentParser(description="Build graph for fake news detection")
    # dataset arguments
    parser.add_argument("--dataset_name", type=str, default="kdd2020", help="dataset to use", choices=["fake_news_tfg", "kdd2020"])
    parser.add_argument("--train_size", type=str, default="full", help="dataset size: full, 5%, 100 or integer")
    parser.add_argument("--val_size", type=str, default="full", help="dataset size: full, 5%, 100 or integer")
    parser.add_argument("--test_size", type=str, default="full", help="dataset size: full, 5%, 100 or integer")
    parser.add_argument("--labeled_size", type=str, default="100", help="number to mask the training nodes")
    parser.add_argument("--output_dir", type=str, default="graph", help="path to save the graph")
    parser.add_argument("--edge_policy", type=str, default="thresholdnn", help="edge construction policy", choices=["knn", "thresholdnn"])
    parser.add_argument("--threshold_factor", type=float, default=1.0, help="threshold factor for threshold knn")
    parser.add_argument("--k", type=int, default=5, help="number of neighbors for knn")
    parser.add_argument("--prebuilt_graph", type=str, default=None, help="path to prebuilt graph")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    labeled_size = args.labeled_size
    output_dir = args.output_dir
    edge_policy = args.edge_policy
    threshold_factor = args.threshold_factor
    k = args.k
    prebuilt_graph = args.prebuilt_graph

    # show arguments
    show_args(args, output_dir)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # skip the process of building empty graph(nodes without edges) and construct graph edge
    if prebuilt_graph is not None:
        if not os.path.exists(prebuilt_graph):
            raise FileNotFoundError(f"Prebuilt graph not found: {prebuilt_graph}")
        print(f"Loading prebuilt graph from {prebuilt_graph}")
        G = torch.load(prebuilt_graph, weights_only=False)
        print(f"Graph loaded: {G}")

        # construct graph edge
        G = construct_graph_edge(G, k=k, edge_policy=edge_policy, theshold_factor=threshold_factor)

        graph_root_dir_path = f"{output_dir}/{dataset_name}"
        graph_name = f"train_{train_size}_val_{val_size}_test_{test_size}_labeled_{labeled_size}"
        if edge_policy == "knn":
            graph_name = f"{graph_name}_knn_{k}"
        elif edge_policy == "thresholdnn":
            graph_name = f"{graph_name}_thresholdnn_{threshold_factor}"

        # save the graph
        save_graph(G, graph_root_dir_path, f"{output_dir}/{dataset_name}/{graph_name}.pt")

        # analyze the graph
        analyze_graph(G, graph_root_dir_path, graph_info_path=f"{output_dir}/{dataset_name}/{graph_name}.txt")

        # visualize the graph
        plot_dir = f"plot/{dataset_name}"
        visualize_graph(G, show_num_nodes='full', plot_dir=plot_dir, graph_name=graph_name)

        exit()

    dataset = fetch_dataset(dataset_name)
    train_dataset, val_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"] # type: ignore

    # reasign the dataset size
    train_size = int(train_size) if train_size.isdigit() else len(train_dataset)
    val_size = int(val_size) if val_size.isdigit() else len(val_dataset)
    test_size = int(test_size) if test_size.isdigit() else len(test_dataset)
    labeled_size = int(labeled_size) if labeled_size.isdigit() else len(train_dataset)

    # embedd the dataset (pre-trained model: BERT)
    embeddings_train_dataset = get_embeddings_dataset(dataset_name, train_dataset, train_size)
    embeddings_val_dataset = get_embeddings_dataset(dataset_name, val_dataset, val_size)
    embeddings_test_dataset = get_embeddings_dataset(dataset_name, test_dataset, test_size)

    # generate empty graph with nodes but no edges
    G = generate_embeddings_graph(embeddings_train_dataset, embeddings_val_dataset, embeddings_test_dataset, labeled_size)

    graph_root_dir_path = f"{output_dir}/{dataset_name}"
    empty_graph_path = f"{output_dir}/{dataset_name}/train_{train_size}_val_{val_size}_test_{test_size}_labeled_{labeled_size}.pt"
    save_graph(G.get_graph(), graph_root_dir_path, empty_graph_path)

    # construct graph edge
    G = construct_graph_edge(G.get_graph(), k=k, edge_policy=edge_policy, theshold_factor=threshold_factor)

    graph_root_dir_path = f"{output_dir}/{dataset_name}"
    graph_name = f"train_{train_size}_val_{val_size}_test_{test_size}_labeled_{labeled_size}"
    if edge_policy == "knn":
        graph_name = f"{graph_name}_knn_{k}"
    elif edge_policy == "thresholdnn":
        graph_name = f"{graph_name}_thresholdnn_{threshold_factor}"

    # save the graph
    save_graph(G, graph_root_dir_path, f"{output_dir}/{dataset_name}/{graph_name}.pt")

    # analyze the graph
    analyze_graph(G, graph_root_dir_path, graph_info_path=f"{output_dir}/{dataset_name}/{graph_name}.txt")

    # visualize the graph
    plot_dir = f"plot/{dataset_name}"
    visualize_graph(G, show_num_nodes='full', plot_dir=plot_dir, graph_name=graph_name)
