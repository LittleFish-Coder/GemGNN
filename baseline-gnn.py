import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from datasets import load_dataset, Dataset as HF_Dataset, DatasetDict
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
import argparse

class EmbeddingsDataset:
    def __init__(self, dataset_dict: DatasetDict, dataset_name: str):
        self.dataset_name = dataset_name
        self.train_data = dataset_dict['train']
        self.test_data = dataset_dict['test']

    def get_embeddings_and_labels(self):
        train_embeddings = np.array(self.train_data['embeddings'])
        train_labels = np.array(self.train_data['label'])
        test_embeddings = np.array(self.test_data['embeddings'])
        test_labels = np.array(self.test_data['label'])
        return train_embeddings, train_labels, test_embeddings, test_labels

class EmbeddingsGraph:
    def __init__(self, train_embeddings, train_labels, test_embeddings, test_labels):
        self.num_nodes = len(train_embeddings) + len(test_embeddings)
        self.num_features = train_embeddings.shape[1]
        
        # 合併embeddings和標籤
        self.x = torch.tensor(np.concatenate([train_embeddings, test_embeddings]), dtype=torch.float)
        self.y = torch.tensor(np.concatenate([train_labels, test_labels]), dtype=torch.long)
        
        # 創建masks
        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.train_mask[:len(train_embeddings)] = True
        self.test_mask[len(train_embeddings):] = True

    def get_graph(self):
        return Data(x=self.x, edge_index=torch.empty((2, 0), dtype=torch.long),
                   y=self.y, train_mask=self.train_mask, test_mask=self.test_mask)

class KNNGraphBuilder:
    def __init__(self, k=5):
        self.k = k

    def build_graph(self, graph_data: Data) -> Data:
        x = graph_data.x
        nn = NearestNeighbors(n_neighbors=self.k + 1, metric="cosine")
        nn.fit(x)
        distances, indices = nn.kneighbors(x)

        edge_index = []
        edge_attr = []

        for i in tqdm(range(len(x)), desc="Building edges"):
            for j in range(1, self.k + 1):  # skip self
                neighbor = indices[i, j]
                edge_index.append([i, neighbor])
                edge_attr.append(1 - distances[i, j])  # 使用相似度作為邊的屬性

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr).unsqueeze(1)

        return Data(x=graph_data.x, edge_index=edge_index, edge_attr=edge_attr,
                   y=graph_data.y, train_mask=graph_data.train_mask,
                   test_mask=graph_data.test_mask)

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, 
                 out_channels: int, model_type: str = 'GCN', dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        
        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
            self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def sample_k_shots(dataset: DatasetDict, k: int) -> DatasetDict:
    """從每個類別中採樣k個樣本"""
    if k == 0:
        return dataset  # full dataset

    print(f"Sampling {k}-shot data per class...\n")

    train_data = dataset["train"]
    sampled_data = {key: [] for key in train_data.column_names}

    labels = set(train_data["label"])
    for label in labels:
        label_data = train_data.filter(lambda x: x["label"] == label)
        sampled_label_data = label_data.shuffle(seed=42).select(
            range(min(k, len(label_data)))
        )
        for key in train_data.column_names:
            sampled_data[key].extend(sampled_label_data[key])

    sampled_dataset = DatasetDict({
        "train": HF_Dataset.from_dict(sampled_data),
        "test": dataset["test"],
    })

    print(f"Sampled dataset size: {len(sampled_dataset['train'])}")
    return sampled_dataset

def train(model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model: torch.nn.Module, data: Data) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        train_correct = out[data.train_mask].argmax(dim=1).eq(data.y[data.train_mask]).sum()
        test_correct = out[data.test_mask].argmax(dim=1).eq(data.y[data.test_mask]).sum()
        train_acc = train_correct.item() / data.train_mask.sum().item()
        test_acc = test_correct.item() / data.test_mask.sum().item()
    return train_acc, test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['KDD2020', 'TFG', 'GossipCop', 'PolitiFact'])
    parser.add_argument('--k_shot', type=int, required=True,
                      choices=[0, 8, 16, 32, 100])  # 0 represents full dataset
    parser.add_argument('--k_neighbors', type=int, required=True,
                      choices=[5, 7, 9])
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['GCN', 'GAT'])
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--checkpoint_dir",type=str,default="checkpoints",help="directory to save checkpoints")
    args = parser.parse_args()

    output_dir = f'{args.checkpoint_dir}/{args.dataset}/{args.model_type}'

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加載數據集
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(f"LittleFish-Coder/Fake_News_{args.dataset}")
    
    # K-shot採樣
    dataset = sample_k_shots(dataset, args.k_shot)
    
    # 準備embeddings數據
    embeddings_dataset = EmbeddingsDataset(dataset, args.dataset)
    train_embeddings, train_labels, test_embeddings, test_labels = embeddings_dataset.get_embeddings_and_labels()
    
    # 創建圖
    print("Creating initial graph structure...")
    graph = EmbeddingsGraph(train_embeddings, train_labels, test_embeddings, test_labels)
    
    # 使用KNN建立邊
    print(f"Building edges using KNN (k={args.k_neighbors})...")
    graph_builder = KNNGraphBuilder(k=args.k_neighbors)
    data = graph_builder.build_graph(graph.get_graph())
    data = data.to(device)
    
    # 打印圖的統計信息
    print(f"\nGraph Statistics:")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of training nodes: {data.train_mask.sum().item()}")
    print(f"Number of test nodes: {data.test_mask.sum().item()}")
    
    # 初始化模型
    print(f"\nInitializing {args.model_type} model...")
    model = GNNModel(
        in_channels=768,  # BERT embeddings維度
        hidden_channels=args.hidden_channels,
        out_channels=2,  # 二分類
        model_type=args.model_type
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 訓練
    print("\nStarting training...")
    best_test_acc = 0
    for epoch in range(args.epochs):
        loss = train(model, data, optimizer)
        train_acc, test_acc = test(model, data)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    print(f'\nTraining completed!')
    print(f'Best Test Accuracy: {best_test_acc:.4f}')

    # save best test accuracy to pandas dataframe
    import pandas as pd
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_path = os.path.join(output_dir, 'results.csv')
    if not os.path.exists(df_path):
        df = pd.DataFrame(columns=['k_shot', 'k_neighbors', 'model_type', 'best_test_acc'])
    else:
        df = pd.read_csv(df_path)
    df = pd.concat([df, pd.DataFrame([{'k_shot': args.k_shot, 'k_neighbors': args.k_neighbors, 'model_type': args.model_type, 'best_test_acc': best_test_acc}])], ignore_index=True)
    df.to_csv(df_path, index=False)


if __name__ == '__main__':
    main()