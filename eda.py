import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from datasets import load_dataset, load_from_disk
import os

DEFAULT_DATASET_CACHE_DIR = "dataset"

def print_similarity_stats(x, y=None, label_name=None, plot=False, tsne=False, emb_name="embedding", dataset_name=None):
    n, d = x.shape
    print(f"\n[{emb_name}] Nodes: {n}, Features: {d}")
    sim = cosine_similarity(x)
    np.fill_diagonal(sim, np.nan)
    sim_flat = sim[~np.isnan(sim)]
    print(f"Count of pairs: {len(sim_flat)} (should be n*(n-1))")
    print("--- Overall Similarity Stats (all pairs, excluding self) ---")
    print(f"Mean: {np.mean(sim_flat):.4f}, Std: {np.std(sim_flat):.4f}")
    print(f"Min: {np.min(sim_flat):.4f}, Max: {np.max(sim_flat):.4f}")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"Percentile {p}: {np.percentile(sim_flat, p):.4f}")
    if np.std(sim_flat) < 0.05:
        print("[Warning] Similarity distribution is very narrow (std < 0.05). Embeddings may lack diversity.")
    if np.max(sim_flat) > 0.99:
        print("[Warning] Some node pairs are nearly identical (cosine > 0.99). Possible duplicates or collapsed features.")
    if np.min(sim_flat) < -0.5:
        print("[Warning] Some node pairs are strongly dissimilar (cosine < -0.5). Check for outliers or bad embeddings.")
    if np.percentile(sim_flat, 99) - np.percentile(sim_flat, 1) < 0.1:
        print("[Warning] 98% of similarities are within 0.1 range. Graph edges may be hard to distinguish.")
    if plot:
        plt.figure(figsize=(8,4))
        plt.hist(sim_flat, bins=100, color='royalblue', alpha=0.8)
        for p in [1, 5, 25, 50, 75, 95, 99]:
            perc = np.percentile(sim_flat, p)
            plt.axvline(perc, color='red' if p==50 else 'orange', linestyle='--', alpha=0.7)
            plt.text(perc, plt.ylim()[1]*0.9, f'{p}%', rotation=90, color='red' if p==50 else 'orange', fontsize=8)
        plt.title(f'Pairwise Cosine Similarity Distribution [{emb_name}]')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.tight_layout()
        fname = f'similarity_distribution_{dataset_name}_{emb_name}.png' if dataset_name else f'similarity_distribution_{emb_name}.png'
        plt.savefig(fname)
        print(f"Saved histogram to {fname}")

        # boxplot
        plt.figure(figsize=(4,6))
        plt.boxplot(sim_flat, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        plt.title(f'Boxplot of Similarity [{emb_name}]')
        plt.ylabel('Cosine Similarity')
        fname_box = f'boxplot_{dataset_name}_{emb_name}.png' if dataset_name else f'boxplot_{emb_name}.png'
        plt.savefig(fname_box)
        print(f"Saved boxplot to {fname_box}")
    if y is not None:
        y = np.array(y)
        print("\n--- Per-Class Similarity Stats ---")
        for label in np.unique(y):
            idx = (y == label)
            name = f"{label_name}={label}" if label_name else f"class={label}"
            sub_sim = sim[np.ix_(idx, idx)]
            sub_flat = sub_sim[~np.isnan(sub_sim)]
            print(f"Within {name}: Mean={np.mean(sub_flat):.4f}, Std={np.std(sub_flat):.4f}, Min={np.min(sub_flat):.4f}, Max={np.max(sub_flat):.4f}")
        # Between-class
        for l1 in np.unique(y):
            for l2 in np.unique(y):
                if l1 >= l2: continue
                idx1 = (y == l1)
                idx2 = (y == l2)
                between = sim[np.ix_(idx1, idx2)]
                between_flat = between[~np.isnan(between)]
                print(f"Between {label_name}={l1} and {label_name}={l2}: Mean={np.mean(between_flat):.4f}, Std={np.std(between_flat):.4f}, Min={np.min(between_flat):.4f}, Max={np.max(between_flat):.4f}")
    if tsne:
        print("Plotting t-SNE...")
        tsne_emb = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, n//10))).fit_transform(x)
        plt.figure(figsize=(6,6))
        if y is not None:
            for label in np.unique(y):
                plt.scatter(tsne_emb[y==label,0], tsne_emb[y==label,1], label=f"{label_name}={label}", alpha=0.7, s=20)
            plt.legend()
        else:
            plt.scatter(tsne_emb[:,0], tsne_emb[:,1], alpha=0.7, s=20)
        plt.title(f't-SNE of {emb_name}')
        plt.tight_layout()
        fname = f'tsne_{dataset_name}_{emb_name}.png' if dataset_name else f'tsne_{emb_name}.png'
        plt.savefig(fname)
        print(f"Saved t-SNE plot to {fname}")
    return sim_flat

def analyze_graph_path(graph_path, plot, tsne):
    graph = torch.load(graph_path, map_location=torch.device('cpu'))
    if not isinstance(graph, Data):
        print("Only supports homogeneous torch_geometric.data.Data graphs.")
        return
    x = graph.x.cpu().numpy() if hasattr(graph, 'x') else None
    y = graph.y.cpu().numpy() if hasattr(graph, 'y') and graph.y is not None else None
    sim_flat = print_similarity_stats(x, y, label_name="label", plot=plot, tsne=tsne, emb_name="graph.x")
    # 畫總boxplot
    plt.figure(figsize=(4,6))
    plt.boxplot(sim_flat, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title('Boxplot of All Similarity [graph.x]')
    plt.ylabel('Cosine Similarity')
    plt.tight_layout()
    plt.savefig('boxplot_all_graph.png')
    print("Saved boxplot to boxplot_all_graph.png")

def analyze_dataset(dataset_name, plot, tsne):
    print(f"Loading dataset: {dataset_name}")
    hf_dataset_name = f"LittleFish-Coder/Fake_News_{dataset_name}"
    # download from huggingface and cache to local path
    local_hf_dir = os.path.join(DEFAULT_DATASET_CACHE_DIR, f"{dataset_name}_hf")
    if os.path.exists(local_hf_dir):
        print(f"Loading dataset from local path: {local_hf_dir}")
        ds = load_from_disk(local_hf_dir)
    else:
        print(f"Loading dataset from huggingface: {hf_dataset_name}")
        ds = load_dataset(hf_dataset_name, download_mode="reuse_cache_if_exists", cache_dir=local_hf_dir)
        ds.save_to_disk(local_hf_dir)

    # Concatenate train and test splits
    all_embs = {}
    all_labels = None
    for split in ["train", "test"]:
        d = ds[split]
        for emb in ["bert_embeddings", "roberta_embeddings", "combined_embeddings", "bigbird_embeddings", "distilbert_embeddings"]:
            if emb not in all_embs:
                all_embs[emb] = []
        if all_labels is None and "label" in d.column_names:
            all_labels = []
        for emb in all_embs:
            if emb in d.column_names:
                all_embs[emb].extend(d[emb])
        if all_labels is not None and "label" in d.column_names:
            all_labels.extend(d["label"])
    # Now analyze each embedding type
    sim_list = []
    emb_names = []
    for emb in ["bert_embeddings", "roberta_embeddings", "combined_embeddings", "bigbird_embeddings", "distilbert_embeddings"]:
        x = np.array(all_embs[emb])
        y = np.array(all_labels) if all_labels is not None else None
        sim_flat = print_similarity_stats(x, y, label_name="label", plot=plot, tsne=tsne, emb_name=emb, dataset_name=dataset_name)
        sim_list.append(sim_flat)
        emb_names.append(emb.split("_")[0])
    # 畫總boxplot
    plt.figure(figsize=(8,6))
    plt.boxplot(sim_list, vert=True, patch_artist=True, tick_labels=emb_names, boxprops=dict(facecolor='lightblue'))
    plt.title(f'Boxplot of All Similarity [{dataset_name}]')
    plt.ylabel('Cosine Similarity')
    plt.tight_layout()
    fname = f'boxplot_all_{dataset_name}.png'
    plt.savefig(fname)
    print(f"Saved boxplot to {fname}")

def main():
    parser = argparse.ArgumentParser(description="EDA for node embedding similarity in a PyG graph or full dataset.")
    parser.add_argument("--graph_path", type=str, help="Path to the .pt graph file (homogeneous Data)")
    parser.add_argument("--dataset", type=str, choices=["politifact", "gossipcop"], help="Dataset name to analyze (all splits)")
    parser.add_argument("--plot", action="store_true", help="Show histogram plot of similarities")
    parser.add_argument("--tsne", action="store_true", help="Show t-SNE plot of embeddings")
    args = parser.parse_args()

    if args.dataset:
        analyze_dataset(args.dataset, args.plot, args.tsne)
    elif args.graph_path:
        analyze_graph_path(args.graph_path, args.plot, args.tsne)
    else:
        print("Please provide either --graph_path or --dataset. Use -h for help.")
        return

if __name__ == "__main__":
    main() 