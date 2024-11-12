import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch_geometric.data import Data

def show_args(args):
    print("========================================")
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print("========================================")

def load_graph(path: str):
    """
    Load the graph
    """
    graph = torch.load(path, weights_only=False)
    print(f"Graph loaded from {path}")
    return graph

def show_graph_info(graph):

    print("Graph information:")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Number of features: {graph.num_features}")

    # calculate the number of classes
    num_classes = graph.y.unique().size(0)
    
    # calculate the distribution of classes
    class_distribution = {int(i): (graph.y == i).sum().item() for i in graph.y.unique()}
    
    print(f"Number of classes: {num_classes}")
    print("Class distribution:")
    for cls, count in class_distribution.items():
        print(f"\tClass {cls}: {count} instances")

def analyze_graph(graph):
    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = graph.edge_index.to(device)
    y = graph.y.to(device)

    # Initialize counters
    fake_to_fake = 0
    real_to_real = 0
    homophilic_edges = 0
    heterophilic_edges = 0

    # Use tensor operations for faster computation
    src, dst = edge_index
    src_labels = y[src]
    dst_labels = y[dst]

    # Calculate fake_to_fake and real_to_real using tensor operations
    fake_to_fake = torch.sum((src_labels == 1) & (dst_labels == 1)).item()
    real_to_real = torch.sum((src_labels == 0) & (dst_labels == 0)).item()

    # Calculate homophilic and heterophilic edges
    homophilic_edges = torch.sum(src_labels == dst_labels).item()
    heterophilic_edges = edge_index.size(1) - homophilic_edges

    total_edges = edge_index.size(1)

    print(f"Total edges: {total_edges}")
    print(f"Fake to fake edges: {fake_to_fake}")
    print(f"Real to real edges: {real_to_real}")
    print(f"Homophilic(same class) edges: {homophilic_edges}")
    print(f"Heterophilic(different class) edges: {heterophilic_edges}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train graph by given graph data")
    parser.add_argument("--graph", type=str, help="path to graph data", required=True)

    args = parser.parse_args()
    graph = args.graph

    # show arguments
    show_args(args)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # load graph
    graph = load_graph(graph)
    print(graph)

    # show graph information
    show_graph_info(graph)

    # analyze graph
    analyze_graph(graph)
    
