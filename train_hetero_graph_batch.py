import os
import torch
import numpy as np
from glob import glob
from train_hetero_graph import get_model, set_seed, final_evaluation, parse_arguments, load_hetero_graph, train

def main():
    args = parse_arguments()
    graph_dir = args.graph_path
    graph_files = sorted(glob(os.path.join(graph_dir, '*.pt')))
    all_y_true, all_y_pred = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for graph_path in graph_files:
        print(f"\n==== Training on {graph_path} ====")
        data = load_hetero_graph(graph_path, device, args.target_node_type)
        model = get_model(args.model, data, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        # train and save best model
        scenario_filename = os.path.splitext(os.path.basename(graph_path))[0]
        output_dir = os.path.join(args.output_dir_base, args.model, scenario_filename)
        os.makedirs(output_dir, exist_ok=True)
        train(model, data, optimizer, criterion, args, output_dir, scenario_filename)
        model_path = os.path.join(output_dir, f"{scenario_filename}_best.pt")
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
    # 合併所有 batch 的 test node 預測
    from sklearn.metrics import f1_score
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    f1 = f1_score(y_true_all, y_pred_all, average='macro')
    print(f"\n==== Final Macro F1-score (all test nodes): {f1:.4f} ====")

if __name__ == "__main__":
    main() 