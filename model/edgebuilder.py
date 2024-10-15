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

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, labeled_mask=labeled_mask, graph_metric="")
    
class ThresholdNNGraphBuilder:
    def __init__(self, graph, k=5, threshold_factor=1.0):
        self.graph = graph
        self.k = k
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

        # 先計算全圖的 cosine similarity distance 統計
        flattened_distances = distances[:, 1:].flatten()  # 去除自連邊
        median_dist = np.median(flattened_distances)
        mean_dist = np.mean(flattened_distances)
        std_dist = np.std(flattened_distances)
        min_dist = np.min(flattened_distances)
        max_dist = np.max(flattened_distances)
        threshold = np.percentile(flattened_distances, self.threshold_factor)  # 計算第 n 百分位數

        print(f"median = {median_dist}, mean = {mean_dist}, std = {std_dist}, min = {min_dist}, max = {max_dist}, threshold = {threshold}")
        # input("Press enter to cont...")
        edge_index = []
        edge_attr = []

        out_degree = []
        for i in tqdm(range(len(x)), desc="Building edges"):
            # 計算這個點的有效鄰居數量（小於 threshold 的鄰居數）
            valid_neighbors = [indices[i, j] for j in range(1, len(x)) if distances[i, j] < threshold]
            # print(valid_neighbors)
            if valid_neighbors:
                avg_neighbors = int(np.sqrt(len(valid_neighbors) * self.k))
                avg_neighbors = (len(valid_neighbors) + avg_neighbors) // 2
            else:
                avg_neighbors = 0
            out_degree.append(avg_neighbors)

            # 根據新的鄰居數量進行建邊
            for j in range(1, avg_neighbors + 1):
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
                    edge_attr.append(1 - distances[i, j])  # 使用相似度作為邊的屬性
        median_out_degree = np.median(out_degree)
        mean_out_degree = np.mean(out_degree)
        std_out_degree = np.std(out_degree)
        min_out_degree = np.min(out_degree)
        max_out_degree = np.max(out_degree)
        quantile_out_degree = [np.percentile(out_degree, i) for i in range(0, 101, 10)]
        graph_metric = f"median = {median_out_degree}, mean = {mean_out_degree}, std = {std_out_degree}, min = {min_out_degree}, max = {max_out_degree}, quantile = {quantile_out_degree}"
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr).unsqueeze(1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, labeled_mask=labeled_mask, graph_metric=graph_metric)