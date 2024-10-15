from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
from torch_geometric.data import Data


class EmbeddingsDataset(Dataset):
    def __init__(self, texts, labels, device="cuda", embeddings=None, size=None, dataset_name=None, model_name="bert-base-uncased"):
        self.dataset_name = dataset_name

        if size == "full":
            self.texts = texts
            self.labels = labels
            self.embeddings = embeddings
        elif size is not None and size <= len(texts):
            self.texts = texts[:size]
            self.labels = labels[:size]
            self.embeddings = embeddings[:size] if embeddings is not None else None
        else:
            self.texts = texts
            self.labels = labels
            self.embeddings = embeddings

        # Only initialize tokenizer and model if the dataset_name is not 'kdd2020'
        if self.dataset_name != "kdd2020":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
            self.model = AutoModel.from_pretrained(model_name).to(device)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # If the dataset is 'kdd2020', directly use the precomputed embeddings
        if self.dataset_name == "kdd2020":
            embeddings = torch.tensor(self.embeddings[idx])
        else:
            # Tokenize the text
            inputs = self.tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            # Move inputs to CUDA
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Get the BERT embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                embeddings = last_hidden_state.mean(dim=1)  # [1, 768]
                # flatten the embeddings
                embeddings = embeddings.flatten()  # [768]
                # print(f"Embeddings shape: {embeddings.shape}")

            # put the embeddings into cpu
            embeddings = embeddings.cpu()

        return embeddings, label
    
class EmbeddingsGraph:
    def __init__(self, train_dataset, val_dataset, test_dataset, labeled_size):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.labeled_size = labeled_size
        try:
            self.labeled_size = int(labeled_size)
        except:
            self.labeled_size = len(train_dataset)

        self.num_nodes = len(train_dataset) + len(val_dataset) + len(test_dataset)
        self.num_features = train_dataset[0][0].shape[0]  # Assuming all embeddings have the same dimension

        self.graph = self._build_graph()

    def _build_graph(self):
        # Combine all embeddings
        x = torch.cat(
            [
                torch.stack([item[0] for item in tqdm(dataset, desc=f"Processing {name} dataset")])
                for dataset, name in zip((self.train_dataset, self.val_dataset, self.test_dataset), ("train", "val", "test"))
            ]
        )

        # Combine all labels with tqdm progress bar
        y = torch.cat(
            [
                torch.tensor([item[1] for item in tqdm(dataset, desc=f"Processing labels for {name} dataset")])
                for dataset, name in zip((self.train_dataset, self.val_dataset, self.test_dataset), ("train", "val", "test"))
            ]
        )

        # Create masks
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        train_mask[:len(self.train_dataset)] = True
        val_mask[len(self.train_dataset): len(self.train_dataset) + len(self.val_dataset)] = True
        test_mask[-len(self.test_dataset):] = True

        # random choice labeled indices
        random_indices = torch.randperm(len(self.train_dataset))[:self.labeled_size]
        labeled_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        labeled_mask[random_indices] = True

        # Create edge_index (placeholder, as we're not considering edges yet)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, labeled_mask=labeled_mask,)

    def get_graph(self):
        return self.graph

    def __repr__(self):
        return (
            f"CustomGraph(num_nodes={self.num_nodes}, num_features={self.num_features}, "
            f"training_nodes={self.graph.train_mask.sum()}, "
            f"validation_nodes={self.graph.val_mask.sum()}, "
            f"test_nodes={self.graph.test_mask.sum()}, "
            f"labeled_nodes={self.graph.labeled_mask.sum()})"
        )

    def __str__(self):
        return (
            f"CustomGraph(num_nodes={self.num_nodes}, num_features={self.num_features}, "
            f"training_nodes={self.graph.train_mask.sum()}, "
            f"validation_nodes={self.graph.val_mask.sum()}, "
            f"test_nodes={self.graph.test_mask.sum()}, "
            f"labeled_nodes={self.graph.labeled_mask.sum()})"
        )