import os
import torch
import numpy as np
from glob import glob
from train_hetero_graph import get_model, set_seed, final_evaluation, parse_arguments, load_hetero_graph, train
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import json

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_arguments()
    # graph_dir: graphs_hetero/politifact/8_shot_roberta_hetero_knn_5_sample_unlabeled_factor_5_multiview_0/
    graph_dir = args.graph_path.rstrip(os.sep)  # remove trailing slash
    graph_files = sorted(glob(os.path.join(graph_dir, '*.pt'))) # graph_batch0.pt, graph_batch1.pt, ...
    all_y_true, all_y_pred = [], [] # collect test node y_true, y_pred

    # parse dataset, scenario, batch_out_dir from graph_dir
    parts = graph_dir.split(os.sep)
    print(f"parts: {parts}")
    dataset = parts[-2] # politifact
    scenario = parts[-1] # 8_shot_roberta_hetero_knn_5_sample_unlabeled_factor_5_multiview_0
    batch_out_dir = os.path.join('results_hetero_batch', args.model, dataset, scenario)
    print(f"Saving results to {batch_out_dir}")
    os.makedirs(batch_out_dir, exist_ok=True)

    for graph_path in graph_files:
        print(f"\n==== Training on {graph_path} ====")
        data = load_hetero_graph(graph_path, device, args.target_node_type)
        model = get_model(args.model, data, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        # train and save best model
        scenario_filename = os.path.splitext(os.path.basename(graph_path))[0]
        train(model, data, optimizer, criterion, args, batch_out_dir, scenario_filename)
        model_path = os.path.join(batch_out_dir, f"{scenario_filename}_best.pt")
        # final evaluation
        metrics = final_evaluation(model, data, model_path, args.target_node_type)
        # collect test node y_true, y_pred
        out = model(data.x_dict, data.edge_index_dict)
        mask = data[args.target_node_type].test_mask
        if isinstance(out, dict):
            out_target = out[args.target_node_type]
        else:
            out_target = out
        pred = out_target[mask].argmax(dim=1).cpu().numpy()
        y_true = data[args.target_node_type].y[mask].cpu().numpy()
        all_y_true.append(y_true)
        all_y_pred.append(pred)

    # concat all batch test node y_true, y_pred
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    acc = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='macro')
    conf_matrix = confusion_matrix(y_true_all, y_pred_all)

    print(f"\n==== Final Macro F1-score (all test nodes): {f1:.4f} ====")
    print(f"Final Acc: {acc:.4f}")
    print(f"Final Precision: {precision:.4f}")
    print(f"Final Recall: {recall:.4f}")
    print(f"Final F1-score: {f1:.4f}")
    print(f"Final Confusion Matrix: {conf_matrix}")
    # Save results to results_hetero_batch/dataset/scenario/metrics_batch.json
    out_path = os.path.join(batch_out_dir, 'metrics_batch.json')
    with open(out_path, 'w') as f:
        json.dump({
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist()
        }, f, indent=2)
    print(f"Saved batch metrics to {out_path}")

if __name__ == "__main__":
    main() 