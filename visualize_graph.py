# visualize_graph_v4.py
#
# Description:
# This script is a comprehensive graph analysis tool. It:
# - Creates a dedicated 'graph_analysis' folder for all outputs.
# - Generates a detailed visualization PNG file.
# - Generates a detailed analysis report TXT file based on the
#   analysis logic from 'build_hetero_graph.py'.
#
# Dependencies:
# pip install torch torch_geometric networkx matplotlib numpy
#
# Usage:
# python visualize_graph_v4.py --graph_path /path/to/your/graph.pt

import os
import argparse
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import numpy as np

def generate_graph_analysis_report(data: torch.Tensor) -> str:
    """
    Generates a detailed analysis report string for the graph,
    mimicking the logic from build_hetero_graph.py's analyze_hetero_graph.
    """
    report_lines = []
    
    report_lines.append("=" * 60)
    report_lines.append("     Heterogeneous Graph Analysis Report")
    report_lines.append("=" * 60)

    # --- Node Type Stats ---
    report_lines.append("\n--- Node Types ---")
    total_nodes = 0
    for node_type in data.node_types:
        n = data[node_type].num_nodes
        total_nodes += n
        report_lines.append(f"Node Type: '{node_type}'")
        report_lines.append(f"  - Num Nodes: {n}")
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            report_lines.append(f"  - Features Dim: {data[node_type].x.shape[1]}")
        
        if node_type == 'news':
            if hasattr(data[node_type], 'y') and data[node_type].y is not None:
                y = data[node_type].y.cpu().numpy()
                unique, counts = np.unique(y, return_counts=True)
                label_dist = {int(k): int(v) for k, v in zip(unique, counts)}
                report_lines.append(f"  - Label Distribution: {label_dist}")
            for mask_name in ['train_labeled_mask', 'train_unlabeled_mask', 'test_mask']:
                if hasattr(data[node_type], mask_name) and data[node_type][mask_name] is not None:
                    count = data[node_type][mask_name].sum().item()
                    report_lines.append(f"  - {mask_name}: {count} nodes ({count/n*100:.1f}% of '{node_type}')")
    
    report_lines.append(f"\nTotal Nodes (all types): {total_nodes}")

    # --- Edge Type Stats ---
    report_lines.append("\n--- Edge Types ---")
    total_edges = 0
    for edge_type in data.edge_types:
        num_edges = data[edge_type].num_edges
        total_edges += num_edges
        edge_type_str = " -> ".join(edge_type)
        report_lines.append(f"[*] Edge Type: {edge_type_str}")
        report_lines.append(f"  - Num Edges: {num_edges}")
        if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
            shape = tuple(data[edge_type].edge_attr.shape)
            report_lines.append(f"  - Attributes Dim: {shape}")
        else:
            report_lines.append("  - Attributes: None")
    report_lines.append(f"\nTotal Edges (all types): {total_edges}")

    # --- News-News Edge Analysis using NetworkX ---
    report_lines.append("\n--- Analysis for ALL news-news Edges (merged) ---")
    
    num_news_nodes = data['news'].num_nodes
    G_news = nx.Graph()
    G_news.add_nodes_from(range(num_news_nodes))
    
    news_edge_types = [et for et in data.edge_types if et[0] == 'news' and et[2] == 'news']
    for edge_type in news_edge_types:
        edge_index = data[edge_type].edge_index.cpu().numpy()
        G_news.add_edges_from(edge_index.T)
        
    report_lines.append(f"  - Merged news-news graph has {G_news.number_of_nodes()} nodes and {G_news.number_of_edges()} edges.")
    if G_news.number_of_nodes() > 0:
        degrees = [d for n, d in G_news.degree()]
        avg_degree = np.mean(degrees)
        num_isolated = sum(1 for d in degrees if d == 0)
        
        report_lines.append(f"  - Avg Degree: {avg_degree:.2f}")
        report_lines.append(f"  - Isolated News Nodes: {num_isolated} ({num_isolated/num_news_nodes*100:.1f}%)")
        report_lines.append(f"  - Density: {nx.density(G_news):.4f}")
        try:
            # Clustering can be slow on large graphs
            report_lines.append(f"  - Avg Clustering Coefficient: {nx.average_clustering(G_news):.4f}")
        except Exception:
            report_lines.append("  - Avg Clustering Coefficient: (Calculation skipped, possibly disconnected graph)")
        report_lines.append(f"  - Connected Components: {nx.number_connected_components(G_news)}")

    report_lines.append("\n" + "=" * 60)
    report_lines.append("      End of Analysis Report")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


def analyze_and_visualize_graph(graph_path: str, output_path: str, layout_type: str, show_labels: bool, max_nodes: int):
    """
    Main function to load, analyze, and visualize a graph.
    """
    # --- 0. 建立輸出資料夾 ---
    output_dir = "graph_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 載入圖檔 ---
    if not os.path.exists(graph_path):
        print(f"錯誤: 在 '{graph_path}' 找不到圖檔")
        return
    print(f"從 {graph_path} 載入圖檔...")
    data = torch.load(graph_path, weights_only=False, map_location=torch.device('cpu'))
    print("圖檔成功載入。")

    # --- 2. 生成並儲存分析報告 ---
    analysis_report = generate_graph_analysis_report(data)
    print("\n" + analysis_report) # 在螢幕上印出報告

    base_name = os.path.splitext(os.path.basename(graph_path))[0]
    report_filename = f"{base_name}_analysis.txt"
    report_path = os.path.join(output_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(analysis_report)
    print(f"\n詳細分析報告已儲存至: {report_path}")

    # --- 3. 建立 NetworkX 圖以供視覺化 ---
    # (這部分與 v3 相同，包含邊的過濾和節點的建立)
    G = nx.MultiDiGraph()
    # ... (此處省略與 v3 完全相同的程式碼以求簡潔) ...
    news_nodes = {i: f'n_{i}' for i in range(data['news'].num_nodes)}
    interaction_nodes = {}
    has_interactions = 'interaction' in data.node_types and data['interaction'].num_nodes > 0
    if has_interactions:
        interaction_nodes = {i: f'i_{i}' for i in range(data['interaction'].num_nodes)}

    train_labeled_indices = data['news'].train_labeled_mask.nonzero(as_tuple=True)[0]
    train_unlabeled_indices = data['news'].train_unlabeled_mask.nonzero(as_tuple=True)[0]
    test_indices = data['news'].test_mask.nonzero(as_tuple=True)[0]

    for i in range(data['news'].num_nodes):
        status = 'unlabeled'
        if i in train_labeled_indices: status = 'train_labeled'
        elif i in train_unlabeled_indices: status = 'train_unlabeled'
        elif i in test_indices: status = 'test'
        G.add_node(news_nodes[i], type='news', status=status)

    if has_interactions:
        for i in range(data['interaction'].num_nodes):
            G.add_node(interaction_nodes[i], type='interaction', status='none')
            
    edges_to_ignore = {'dissimilar_to', 'low_level_knn_to'}
    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        if rel_type in edges_to_ignore: continue
        
        src_map = news_nodes if src_type == 'news' else interaction_nodes
        dst_map = news_nodes if dst_type == 'news' else interaction_nodes
        edge_index = data[edge_type].edge_index
        for i in range(edge_index.shape[1]):
            src_node = src_map.get(edge_index[0, i].item())
            dst_node = dst_map.get(edge_index[1, i].item())
            if src_node and dst_node:
                G.add_edge(src_node, dst_node, type=rel_type)

    # --- 4. 採樣與視覺化設定 ---
    # (此處省略與 v3 完全相同的程式碼)
    H = G
    if G.number_of_nodes() > max_nodes:
        print(f"\n圖中有 {G.number_of_nodes()} 個節點，超過了最大顯示數量 ({max_nodes})。")
        print("為了清晰起見，將視覺化一個隨機子圖。")
        anchor_nodes = [n for n, d in G.nodes(data=True) if d.get('status') == 'train_labeled']
        sampled_nodes = set(anchor_nodes)
        for node in anchor_nodes:
            sampled_nodes.update(G.successors(node))
            sampled_nodes.update(G.predecessors(node))
        if len(sampled_nodes) < max_nodes:
            remaining_nodes = list(set(G.nodes()) - sampled_nodes)
            num_to_add = min(len(remaining_nodes), max_nodes - len(sampled_nodes))
            if num_to_add > 0:
                sampled_nodes.update(random.sample(remaining_nodes, num_to_add))
        H = G.subgraph(sampled_nodes).copy()
        print(f"採樣後的子圖有 {H.number_of_nodes()} 個節點和 {H.number_of_edges()} 條邊。")
    
    node_color_map = {'train_real': 'green', 'train_fake': 'red', 'unlabeled': 'grey', 'test': 'blue', 'interaction': 'gold'}
    node_colors = []
    for node in H.nodes():
        node_data = H.nodes[node]
        if node_data['type'] == 'interaction': node_colors.append(node_color_map['interaction'])
        elif node_data['type'] == 'news':
            status = node_data['status']
            if status == 'train_unlabeled': node_colors.append(node_color_map['unlabeled'])
            elif status == 'test': node_colors.append(node_color_map['test'])
            elif status == 'train_labeled':
                original_index = int(node.split('_')[1])
                label = data['news'].y[original_index].item()
                if label == 0: node_colors.append(node_color_map['train_real'])
                else: node_colors.append(node_color_map['train_fake'])
    
    edge_colors, edge_styles = [], []
    sub_type_styles = {}
    sub_edge_color_palette = ['purple', 'brown', 'teal', 'magenta']
    sub_edge_style_palette = ['dotted', 'dashdot', (0, (3, 1, 1, 1)), (0, (5, 10))]

    for u, v, k, edge_data in H.edges(keys=True, data=True):
        rel_type = edge_data['type']
        if rel_type == 'similar_to':
            edge_colors.append('blue')
            edge_styles.append('solid')
        elif rel_type == 'has_interaction':
            edge_colors.append('#cccccc')
            edge_styles.append('solid')
        elif rel_type.startswith('similar_to_sub'):
            if rel_type not in sub_type_styles:
                view_index = len(sub_type_styles)
                color = sub_edge_color_palette[view_index % len(sub_edge_color_palette)]
                style = sub_edge_style_palette[view_index % len(sub_edge_style_palette)]
                sub_type_styles[rel_type] = {'color': color, 'style': style}
            edge_colors.append(sub_type_styles[rel_type]['color'])
            edge_styles.append(sub_type_styles[rel_type]['style'])
        else:
            edge_colors.append('black')
            edge_styles.append('solid')

    # --- 5. 繪圖與儲存 ---
    print("\n正在產生視覺化圖檔...")
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(H, seed=42, iterations=50, k=0.1)
    
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=150, alpha=0.9)
    nx.draw_networkx_edges(H, pos, edge_color=edge_colors, style=edge_styles, width=1.0, alpha=0.7, arrows=False)

    if show_labels: nx.draw_networkx_labels(H, pos, font_size=8)
        
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Train Labeled (Real)', markerfacecolor='green', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Train Labeled (Fake)', markerfacecolor='red', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Train Unlabeled', markerfacecolor='grey', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Test', markerfacecolor='blue', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Interaction Node', markerfacecolor='gold', markersize=12),
        Line2D([0], [0], color='w', lw=0, label=''), # 空白行
        Line2D([0], [0], color='blue', lw=2, linestyle='solid', label='Edge: similar_to'),
        Line2D([0], [0], color='#cccccc', lw=2, linestyle='solid', label='Edge: has_interaction'),
    ]
    for rel_type, styles in sorted(sub_type_styles.items()):
        legend_elements.append(Line2D([0], [0], color=styles['color'], lw=2, linestyle=styles['style'], label=f'Edge: {rel_type}'))

    plt.legend(handles=legend_elements, loc='upper right', fontsize='x-large', frameon=True, facecolor='lightgray', framealpha=0.8)
    plt.title(f"Graph Visualization", fontsize=24)
    plt.axis('off')
    
    if not output_path:
        img_filename = f"{base_name}_visualization_v4.png"
        output_path = os.path.join(output_dir, img_filename)
        
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"視覺化圖已儲存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="視覺化異構圖 (V4)，產生詳細分析報告並儲存至指定資料夾。")
    parser.add_argument("--graph_path", type=str, required=True, help="儲存的 graph.pt 檔案的路徑。")
    parser.add_argument("--output_path", type=str, default=None, help="儲存視覺化圖片的完整路徑。若不指定，將自動命名並存於 'graph_analysis' 資料夾。")
    parser.add_argument("--layout", type=str, default="spring", choices=["spring", "kamada_kawai"], help="圖視覺化的佈局演算法。")
    parser.add_argument("--show_labels", action="store_true", help="在圖上顯示節點標籤。")
    parser.add_argument("--max_nodes", type=int, default=300, help="要顯示的最大節點數。")
    args = parser.parse_args()
    
    analyze_and_visualize_graph(
        graph_path=args.graph_path,
        output_path=args.output_path,
        layout_type=args.layout,
        show_labels=args.show_labels,
        max_nodes=args.max_nodes
    )

if __name__ == "__main__":
    main()