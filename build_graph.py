import random
from scipy.__config__ import show
import torch
from datasets import load_dataset, DatasetDict
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
    
    # datasetname = ['kdd2020', 'tfg', 'gossipcop', 'politifact']
    
    # load data
    print("Loading dataset...")

    ## Fake News Dataset from HuggingFace based on LittleFish-Coder
    """
    use the [`LittleFish-Coder/Fake_News_KDD2020`](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_KDD2020) dataset
    use the [`LittleFish-Coder/Fake_News_TFG`](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_TFG) dataset
    use the [`LittleFish-Coder/Fake_News_GossipCop`](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_GossipCop) dataset
    use the [`LittleFish-Coder/Fake_News_PolitiFact`](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_PolitiFact) dataset
    - text: text of the article (str)
    - embeddings: BERT embeddings (768, )
    - Label:
        - 1: fake news
        - 0: real news
    """
    if dataset_name == "kdd2020":
        dataset_name = "LittleFish-Coder/Fake_News_KDD2020"
    elif dataset_name == "tfg":
        dataset_name = "LittleFish-Coder/Fake_News_TFG"
    elif dataset_name == "gossipcop":
        dataset_name = "LittleFish-Coder/Fake_News_GossipCop"
    elif dataset_name == "politifact":
        dataset_name = "LittleFish-Coder/Fake_News_PolitiFact"
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    dataset = load_dataset(dataset_name)

    print(f"Original Dataset size: ")
    print(f"\tTrain: {len(dataset['train'])}")  # type: ignore
    print(f"\tTest: {len(dataset['test'])}")    # type: ignore

    return dataset


def get_embeddings_dataset(dataset_name, dataset, size):
    try:
        embeddings_dataset = EmbeddingsDataset(texts=dataset["text"], labels=dataset["label"], embeddings=dataset["embeddings"], size=size, dataset_name=dataset_name,)
    except:
        embeddings_dataset = EmbeddingsDataset(texts=dataset["text"], labels=dataset["label"], size=size, dataset_name=dataset_name)
    print(f"Custom Embeddings Dataset length: {len(embeddings_dataset)}")
    return embeddings_dataset

def show_graph_info(graph):
    print(f"Graph: {graph}")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of features: {graph.num_features}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Number of training nodes: {graph.train_mask.sum()}")
    print(f"Number of test nodes: {graph.test_mask.sum()}")
    print(f"Number of labeled node: {graph.labeled_mask.sum()}")
    print(f"len of train mask: {len(graph.train_mask)}")
    print(f"len of test mask: {len(graph.test_mask)}")
    print(f"len of labeled mask: {len(graph.labeled_mask)}")
    print()

def generate_embeddings_graph(embeddings_train_dataset, embeddings_test_dataset, labeled_size):
    print("Generate custom graph...")
    custom_graph = EmbeddingsGraph(embeddings_train_dataset, embeddings_test_dataset, labeled_size)
    graph_data = custom_graph.get_graph()
    show_graph_info(graph_data)

    return custom_graph


def analyze_graph(graph, output_dir: str, graph_info_path: str):
    """analyze the graph"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    show_graph_info(graph)

    train_mask = graph.train_mask
    test_mask = graph.test_mask
    labeled_mask = graph.labeled_mask
    edge_index = graph.edge_index

    total_nodes = graph.num_nodes
    total_edges = graph.num_edges
    train_nodes = train_mask.sum().item()
    test_nodes = test_mask.sum().item()
    labeled_nodes = labeled_mask.sum().item()
    graph_metric = graph.graph_metric

    # analyze the edge types
    edge_types = {"train-train": 0, "train-test": 0, "test-test": 0,}

    for edge in tqdm(edge_index.t(), desc="Analyzing edges"):
        source, target = edge
        if train_mask[source] and train_mask[target]:
            edge_types["train-train"] += 1
        elif (train_mask[source] and test_mask[target]) or (
            test_mask[source] and train_mask[target]
        ):
            edge_types["train-test"] += 1
        elif test_mask[source] and test_mask[target]:
            edge_types["test-test"] += 1

    with open(f"{graph_info_path}", "w") as f:
        f.write(f"Total nodes: {total_nodes}\n")
        f.write(f"Total edges: {total_edges}\n")
        f.write(f"Training nodes: {train_nodes}\n")
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
    print(f"Graph info saved to {graph_info_path}")


def construct_graph_edge(graph_data, k, edge_policy, theshold_factor=1.0):
    print("Constructing graph edge...")
    builder = None
    graph = None
    if edge_policy == "knn":
        builder = KNNGraphBuilder(graph=graph_data, k=k)
    elif edge_policy == "thresholdnn":
        builder = ThresholdNNGraphBuilder(graph=graph_data, threshold_factor=theshold_factor)
    else:   # default set edge policy as thresholdnn
        builder = ThresholdNNGraphBuilder(graph=graph_data, threshold_factor=theshold_factor)

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
    labeled_mask = graph_data.labeled_mask.numpy()

    color_map = {
        "train": "blue",
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
        else:
            node_colors.append("gray")  # Default color for nodes outside the masks

    nx.draw_networkx(G, with_labels=False, node_color=node_colors, edge_color='gray', node_size = 3, alpha=0.7)
    legend_elements = [
        Patch(facecolor='blue', label='train'),
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
    parser.add_argument("--dataset_name", type=str, default="kdd2020", help="dataset to use", choices=["tfg", "kdd2020", "gossipcop", "politifact"])
    parser.add_argument("--train_size", type=str, default="full", help="dataset size: full, 5%, 100 or integer")
    parser.add_argument("--test_size", type=str, default="full", help="dataset size: full, 5%, 100 or integer")
    parser.add_argument("--labeled_size", type=str, default="100", help="number to mask the training nodes")
    parser.add_argument("--output_dir", type=str, default="graph", help="path to save the graph")
    parser.add_argument("--edge_policy", type=str, default="thresholdnn", help="edge construction policy", choices=["knn", "thresholdnn"])
    parser.add_argument("--threshold_factor", type=int, default=1, help="threshold factor for threshold knn")
    parser.add_argument("--k", type=int, default=5, help="number of neighbors for knn")
    parser.add_argument("--prebuilt_graph", type=str, default=None, help="path to prebuilt graph")
    parser.add_argument("--plot", action="store_true", help="Enable plotting") 

    args = parser.parse_args()
    dataset_name = args.dataset_name
    train_size = args.train_size
    test_size = args.test_size
    labeled_size = args.labeled_size
    output_dir = args.output_dir
    edge_policy = args.edge_policy
    threshold_factor = args.threshold_factor
    k = args.k
    prebuilt_graph = args.prebuilt_graph
    plot = args.plot

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
        graph_name = f"train_{train_size}_test_{test_size}_labeled_{labeled_size}"
        if edge_policy == "knn":
            graph_name = f"{graph_name}_knn_{k}"
        elif edge_policy == "thresholdnn":
            graph_name = f"{graph_name}_thresholdnn_{threshold_factor}"

        # save the graph
        save_graph(G, graph_root_dir_path, f"{output_dir}/{dataset_name}/{graph_name}.pt")

        # analyze the graph
        analyze_graph(G, graph_root_dir_path, graph_info_path=f"{output_dir}/{dataset_name}/{graph_name}.txt")

        # visualize the graph
        if plot:
            plot_dir = f"plot/{dataset_name}"
            visualize_graph(G, show_num_nodes='full', plot_dir=plot_dir, graph_name=graph_name)

        exit()

    dataset = fetch_dataset(dataset_name)
    train_dataset, test_dataset = dataset["train"], dataset["test"] # type: ignore

    # reasign the dataset size
    train_size = int(train_size) if train_size.isdigit() else len(train_dataset)
    test_size = int(test_size) if test_size.isdigit() else len(test_dataset)
    labeled_size = int(labeled_size) if labeled_size.isdigit() else len(train_dataset)

    # embedd the dataset (pre-trained model: BERT)
    embeddings_train_dataset = get_embeddings_dataset(dataset_name, train_dataset, train_size)
    embeddings_test_dataset = get_embeddings_dataset(dataset_name, test_dataset, test_size)

    # generate empty graph with nodes but no edges
    G = generate_embeddings_graph(embeddings_train_dataset, embeddings_test_dataset, labeled_size)

    graph_root_dir_path = f"{output_dir}/{dataset_name}"
    empty_graph_path = f"{output_dir}/{dataset_name}/train_{train_size}_test_{test_size}_labeled_{labeled_size}.pt"
    save_graph(G.get_graph(), graph_root_dir_path, empty_graph_path)

    # construct graph edge
    G = construct_graph_edge(G.get_graph(), k=k, edge_policy=edge_policy, theshold_factor=threshold_factor)

    graph_root_dir_path = f"{output_dir}/{dataset_name}"
    graph_name = f"train_{train_size}_test_{test_size}_labeled_{labeled_size}"
    if edge_policy == "knn":
        graph_name = f"{graph_name}_knn_{k}"
    elif edge_policy == "thresholdnn":
        graph_name = f"{graph_name}_thresholdnn_{threshold_factor}"

    # save the graph
    save_graph(G, graph_root_dir_path, f"{output_dir}/{dataset_name}/{graph_name}.pt")

    # analyze the graph
    analyze_graph(G, graph_root_dir_path, graph_info_path=f"{output_dir}/{dataset_name}/{graph_name}.txt")

    # visualize the graph
    if plot:
        plot_dir = f"plot/{dataset_name}"
        visualize_graph(G, show_num_nodes='full', plot_dir=plot_dir, graph_name=graph_name)
