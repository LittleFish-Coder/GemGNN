import os
import json
import argparse
import re

def analyze_metrics(folder_path):

    """
    folder_path: results_hetero_batch/HAN/gossipcop
    scan through the folder_path, and find all the subdirectories, 
    i.e. results_hetero_batch/HAN/gossipcop/3_shot_deberta_hetero_knn_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_dissimilar
    """

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return

    results = {}

    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    try:
        subdirs.sort(key=lambda x: int(x.split('_shot')[0]))
    except ValueError:
        subdirs.sort()

    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(folder_path, subdir)
        for file in os.listdir(subdir_path):
            if file.endswith('metrics_batch.json'):
                file_path = os.path.join(subdir_path, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'f1_score' in data or 'f1' in data:
                        f1_score = data['f1_score'] if 'f1_score' in data else data['f1']
                        # Extract shot count and scenario suffix
                        shot_count = int(re.search(r'(\d+)_shot', subdir).group(1))
                        # Get everything after the shot count
                        scenario = re.sub(r'^\d+_shot_', '', subdir)
                        if scenario not in results:
                            results[scenario] = {}
                        results[scenario][shot_count] = f1_score
                    else:
                        print(f"F1 score not found in {file_path}")

    # Print results in a structured format
    print("\nResults Analysis:")
    for scenario, shot_scores in results.items():
        print(f"\nScenario: {scenario}")
        print("Shot Count | F1 Score")
        print("-" * 20)
        for shot, score in sorted(shot_scores.items()):
            print(f"{shot:^9} | {score:.4f}")
        
        # Print sequence of scores
        scores = [score for _, score in sorted(shot_scores.items())]
        print(f"\nF1 Scores sequence ({len(scores)}):")
        print(scores)

                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Analyze metrics from result files')
    parser.add_argument('--folder', type=str, help='Path to folder containing result files')
    args = parser.parse_args()

    analyze_metrics(args.folder)
