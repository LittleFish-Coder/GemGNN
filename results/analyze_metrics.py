import os
import json

def analyze_metrics(folder_path):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return

    results = {}

    is_mlp_or_lstm = "MLP" in folder_path or "LSTM" in folder_path
    if is_mlp_or_lstm:
        results = {"bert": {}, "roberta": {}}
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    try:
        subdirs.sort(key=lambda x: int(x.split('_shot')[0]))
    except ValueError:
        try:
            subdirs.sort(key=lambda x: int(x.split('shot')[0]))
        except ValueError:
            try:
                subdirs.sort(key=lambda x: int(x.split('-shot')[0]))
            except ValueError:
                subdirs.sort()

    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(folder_path, subdir)
        for file in os.listdir(subdir_path):
            if file.endswith('metrics.json'):
                file_path = os.path.join(subdir_path, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if is_mlp_or_lstm:
                        if "roberta" in subdir:
                            k_shot = subdir.split('_shot')[0] if '_shot' in subdir else subdir.split('shot')[0]
                            if 'test_metrics' in data:
                                results["roberta"][k_shot] = data['test_metrics']['f1_score']
                            elif 'final_test_metrics' in data:
                                results["roberta"][k_shot] = data['final_test_metrics']['f1_score']
                            else:
                                results["roberta"][k_shot] = data['f1']
                        elif "bert" in subdir:
                            k_shot = subdir.split('_shot')[0] if '_shot' in subdir else subdir.split('shot')[0]
                            if 'test_metrics' in data:
                                results["bert"][k_shot] = data['test_metrics']['f1_score']
                            elif 'final_test_metrics' in data:
                                results["bert"][k_shot] = data['final_test_metrics']['f1_score']
                            else:
                                results["bert"][k_shot] = data['f1']
                    else:
                        if 'test_metrics' in data:
                            results[subdir] = data['test_metrics']['f1_score']
                        elif 'final_test_metrics' in data:
                            results[subdir] = data['final_test_metrics']['f1_score']
                        else:
                            results[subdir] = data['f1']
                    break

    # Print results in a structured format
    print("\nResults Analysis:")
    if is_mlp_or_lstm:
        for emb in ["bert", "roberta"]:
            print(f"\nEmbedding: {emb}")
            f1_record = []
            for k_shot, f1_score in sorted(results[emb].items(), key=lambda x: int(x[0])):
                print(f"K-Shot: {k_shot}")
                print(f"F1 Score: {f1_score}")
                f1_record.append(f1_score)
            print(f"\nF1 Scores in sequence for {emb}:")
            print(f1_record)
    else:
        f1_record = []
        for k_shot, f1_score in results.items():
            print(f"\nK-Shot: {k_shot}")
            print(f"F1 Score: {f1_score}")
            f1_record.append(f1_score)
        # print("\nF1 Scores in sequence:")
        # print(f1_record)

                

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze metrics from result files')
    parser.add_argument('--folder_path', type=str, help='Path to folder containing result files')
    
    args = parser.parse_args()
    analyze_metrics(args.folder_path)
