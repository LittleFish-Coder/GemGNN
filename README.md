# Few-Shot Fake News Detection via Graph Neural Networks

Check [installation guide](#installation) to run the code.

## Overview
- Few-shot: N-way-K-shot 
  - N(number of classes): 2 (real/fake)
  - K(number of samples per class): 3~16
- Transductive GNN: All nodes (labeled/unlabeled train/test) are used during training for message passing.
- Loss: Calculated only on labeled nodes.
- Evaluation: Performed on test nodes.
## Metrics
- Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$

- Precision: $\frac{TP}{TP + FP}$

- Recall: $\frac{TP}{TP + FN}$

- F1-Score: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ (main metric)

## Results
[gossipcop.png](./results/gossipcop.png)
[politifact.png](./results/politifact.png)


## Dataset

[Fake_News_GossipCop](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_GossipCop)

[Fake_News_PolitiFact](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_PolitiFact)

- text(str)
- bert_embeddings(sequence): BERT embeddings of the text
- roberta_embeddings(sequence): RoBERTa embeddings of the text (768, )
- label(int): 
  - 0: real
  - 1: fake
- user_interaction(list): list of user interactions(dict)
    - content(str): content of the interaction
    - tone(str): tone of the interaction

## Installation

- Create a new conda environment
```bash
conda create -n fakenews python=3.12
conda activate fakenews
```

- Install PyTorch (based on your CUDA version)
[(Official Doc)](https://pytorch.org/get-started/locally/)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

- Install PyTorch Geometric [(Official Doc)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

```bash
pip install torch-geometric
```

<!-- - Install Additional Libraries for GNN (Based on your torch version)

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
``` -->

- Install other dependencies
```bash
pip install -r requirements.txt
```

## Usage

### GNN
- build graph
```bash
```
- build graph with user interactions
```bash
```
- train graph
```bash
```

### LLM (In-context learning)
make sure you have the `GEMINI-API`
```bash
```

### Language Model (BERT, RoBERTa)
```bash
```

### LSTM, MLP
```bash
```