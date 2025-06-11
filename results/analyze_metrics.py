import os
import json
import re
import argparse
from collections import defaultdict

def extract_shot_count(subdir_name):
    """Extracts shot count from a directory name."""
    match = re.search(r'(\d+)[_-]?shot', subdir_name)
    if match:
        return int(match.group(1))
    return None

def get_f1_score(data):
    """Extracts F1 score from metrics data by checking common keys."""
    if 'test_metrics' in data and 'f1_score' in data['test_metrics']:
        return data['test_metrics']['f1_score']
    if 'final_test_metrics' in data and 'f1_score' in data['final_test_metrics']:
        return data['final_test_metrics']['f1_score']
    if 'f1' in data:
        return data['f1']
    return None

def analyze_metrics(folder_path):
    """Analyzes metrics from result files in a given folder."""
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return

    results = defaultdict(dict)

    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    subdirs.sort(key=lambda d: (extract_shot_count(d) is None, extract_shot_count(d), d))

    for subdir in subdirs:
        subdir_path = os.path.join(folder_path, subdir)
        
        metrics_file = None
        for file in os.listdir(subdir_path):
            if file.endswith('metrics.json'):
                metrics_file = os.path.join(subdir_path, file)
                break
        
        if not metrics_file:
            continue

        with open(metrics_file, 'r') as f:
            data = json.load(f)

        f1_score = get_f1_score(data)
        if f1_score is None:
            print(f"F1 score not found in {metrics_file}")
            continue

        shot_count = extract_shot_count(subdir)
        
        scenario = "default"
        if shot_count is not None:
            scenario = re.sub(r'^\d+[_-]?shot[_-]?', '', subdir).strip()
            if not scenario:
                scenario = os.path.basename(os.path.normpath(folder_path))
        else:
            scenario = subdir

        key = shot_count if shot_count is not None else subdir
        results[scenario][key] = f1_score
            
    print("\nResults Analysis:")
    for scenario, scores_dict in sorted(results.items()):
        print(f"\nScenario: {scenario}")

        keys_are_shots = all(isinstance(k, int) for k in scores_dict.keys())
        
        # Sort by key, handling mixed types (int, str)
        sorted_scores = sorted(scores_dict.items(), key=lambda item: (isinstance(item[0], str), item[0]))

        if keys_are_shots:
            print("Shot Count | F1 Score")
            print("-" * 22)
        else:
            print("Key        | F1 Score")
            print("-" * 22)

        for key, score in sorted_scores:
            if isinstance(key, int):
                print(f"{key:^10} | {score:.4f}")
            else:
                print(f"{str(key):<10} | {score:.4f}")

        if keys_are_shots:
            scores = [score for _, score in sorted(scores_dict.items())]
            print(f"\nF1 Scores sequence for {scenario} ({len(scores)}):")
            print(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze metrics from result files')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to folder containing result files')
    args = parser.parse_args()
    analyze_metrics(args.folder_path)
