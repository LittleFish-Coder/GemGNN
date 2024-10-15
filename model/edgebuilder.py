from os import name
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np

class KNNGraphBuilder:
    def __init__(self, graph, k=5):
        self.graph = graph
        self.k = k

    def build_graph(self, val_to_train=True, val_to_val=True, test_to_test=True, test_to_train=True, val_to_test=True):
        x = self.graph.x
        y = self.graph.y
        train_mask = self.graph.train_mask
        val_mask = self.graph.val_mask
        test_mask = self.graph.test_mask
        labeled_mask = self.graph.labeled_mask
        
        nn = NearestNeighbors(n_neighbors=self.k + 1, metric="cosine")
        nn.fit(x)
        distances, indices = nn.kneighbors(x)

        edge_index = []
        edge_attr = []

        for i in tqdm(range(len(x)), desc="Building edges"):
            for j in range(1, self.k + 1):  # skip self
                neighbor = indices[i, j]

                # decide whether to add the edge
                add_edge = False

                if train_mask[i] or train_mask[neighbor]:
                    # train nodes can always connect to each other
                    add_edge = True
                elif val_mask[i]:
                    if train_mask[neighbor] and val_to_train:
                        add_edge = True
                    elif val_mask[neighbor] and val_to_val:
                        add_edge = True
                    elif test_mask[neighbor] and val_to_test:
                        add_edge = True
                elif test_mask[i]:
                    if train_mask[neighbor] and test_to_train:
                        add_edge = True
                    elif val_mask[neighbor] and val_to_test:
                        add_edge = True
                    elif test_mask[neighbor] and test_to_test:
                        add_edge = True

                if add_edge:
                    edge_index.append([i, neighbor])
                    edge_attr.append(1 - distances[i, j])  # use similarity as the edge attribute

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr).unsqueeze(1)
        graph_metric = f"Every node has {self.k} neighbors"

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, labeled_mask=labeled_mask, graph_metric=graph_metric)
    
class ThresholdNNGraphBuilder:
    def __init__(self, graph, k=5, threshold_factor=1.0):
        self.graph = graph
        self.k = k    # not used
        self.threshold_factor = threshold_factor

    def build_graph(self, val_to_train=True, val_to_val=True, test_to_test=True, test_to_train=True, val_to_test=True):
        x = self.graph.x
        y = self.graph.y
        train_mask = self.graph.train_mask
        val_mask = self.graph.val_mask
        test_mask = self.graph.test_mask
        labeled_mask = self.graph.labeled_mask

        nn = NearestNeighbors(n_neighbors=len(x), metric="cosine")
        nn.fit(x)
        distances, indices = nn.kneighbors(x)

        # calculate the statistics of the distances
        flattened_distances = distances[:, 1:].flatten()  # skip self
        # median_dist = np.median(flattened_distances)
        # mean_dist = np.mean(flattened_distances)
        # std_dist = np.std(flattened_distances)
        # min_dist = np.min(flattened_distances)
        # max_dist = np.max(flattened_distances)
        threshold = np.percentile(flattened_distances, self.threshold_factor)  # calculate {threshold}-th quantile
        # print(flattened_distances)
        # print(self.threshold_factor)
        # print(f"Final threshold: {threshold}")

        # print(f"Distance statistics:")
        # print(f"median = {median_dist}, mean = {mean_dist}, std = {std_dist}, min = {min_dist}, max = {max_dist}, threshold = {threshold}")
        edge_index = []
        edge_attr = []

        out_degree = []
        for i in tqdm(range(len(x)), desc="Building edges"):
            # 計算這個點的有效鄰居數量（小於 threshold 的鄰居數）
            valid_neighbors = [indices[i, j] for j in range(1, len(x)) if distances[i, j] < threshold]
            out_degree.append(len(valid_neighbors))

            for j in range(1, len(valid_neighbors)+1):  # skip self
                neighbor = indices[i, j]
                add_edge = False

                if train_mask[i] or train_mask[neighbor]:
                    add_edge = True
                elif val_mask[i]:
                    if train_mask[neighbor] and val_to_train:
                        add_edge = True
                    elif val_mask[neighbor] and val_to_val:
                        add_edge = True
                    elif test_mask[neighbor] and val_to_test:
                        add_edge = True
                elif test_mask[i]:
                    if train_mask[neighbor] and test_to_train:
                        add_edge = True
                    elif val_mask[neighbor] and val_to_test:
                        add_edge = True
                    elif test_mask[neighbor] and test_to_test:
                        add_edge = True

                if add_edge:
                    edge_index.append([i, neighbor])
                    edge_attr.append(1 - distances[i, j])  # use similarity as the edge attribute
        
        median_out_degree = np.median(out_degree).item()
        mean_out_degree = np.mean(out_degree).item()
        std_out_degree = np.std(out_degree).item()
        min_out_degree = np.min(out_degree).item()
        max_out_degree = np.max(out_degree).item()
        quantile_out_degree = [np.percentile(out_degree, i).item() for i in range(0, 101, 10)]
        graph_metric = f"median = {median_out_degree}, mean = {mean_out_degree}, std = {std_out_degree}, min = {min_out_degree}, max = {max_out_degree}, quantile = {quantile_out_degree}"
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr).unsqueeze(1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, labeled_mask=labeled_mask, graph_metric=graph_metric)
    

# debug
if __name__ == "__main__":
    import os

    def construct_graph_edge(graph_data, k, edge_policy, theshold_factor=1.0):
        if edge_policy == "knn":
            graph_builder = KNNGraphBuilder(graph=graph_data, k=k)
        elif edge_policy == "thresholdnn":
            graph_builder = ThresholdNNGraphBuilder(graph=graph_data, k=k, threshold_factor=theshold_factor)
        custom_graph = graph_builder.build_graph()
        return custom_graph

    G = torch.load('../graph/kdd2020/train_3490_val_997_test_499_labeled_100.pt', weights_only=False)
    print(f"Graph loaded: {G}")

    k = 5
    edge_policy = "thresholdnn"
    threshold_factor = 1.0

    custom_graph = construct_graph_edge(G, k=k, edge_policy=edge_policy, theshold_factor=threshold_factor)
    print(f"Custom graph constructed: {custom_graph}")

    # info
    total_nodes = custom_graph.num_nodes
    total_edges = custom_graph.num_edges
    train_nodes = custom_graph.train_mask.sum().item()
    val_nodes = custom_graph.val_mask.sum().item()
    test_nodes = custom_graph.test_mask.sum().item()
    labeled_nodes = custom_graph.labeled_mask.sum().item()
    graph_metric = custom_graph.graph_metric

    print(f"Total nodes: {total_nodes}")
    print(f"Total edges: {total_edges}")
    print(f"Training nodes: {train_nodes}")
    print(f"Validation nodes: {val_nodes}")
    print(f"Test nodes: {test_nodes}")
    print(f"Labeled nodes: {labeled_nodes}")
    print(f"{graph_metric}")