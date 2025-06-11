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
            if file.endswith('.json'):
                file_path = os.path.join(subdir_path, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'final_test_metrics_on_target_node' in data and 'f1_score' in data['final_test_metrics_on_target_node']:
                        f1_score = data['final_test_metrics_on_target_node']['f1_score']
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
    scenario_averages = []
    for scenario, shot_scores in results.items():
        print(f"\nScenario: {scenario}")
        print("Shot Count | F1 Score")
        print("-" * 20)
        for shot, score in sorted(shot_scores.items()):
            print(f"{shot:^9} | {score:.4f}")
        
        # Print sequence of scores
        scores = [score for _, score in sorted(shot_scores.items())]
        average_score = sum(scores) / len(scores) if scores else 0
        scenario_averages.append((scenario, average_score, len(scores)))
        print(f"\nF1 Scores sequence ({len(scores)}):")
        print(scores)
        print(f"Average F1 Score: {average_score:.4f}")

    # Sort scenarios by average score in descending order
    scenario_averages.sort(key=lambda x: x[1], reverse=True)

    # Print summary table
    print("\n--- Summary ---")
    print(f"{'Scenario':<110} | {'Avg F1 Score':<15} | {'Num points'}")
    print("-" * 140)
    for scenario, avg_score, num_points in scenario_averages:
        print(f"{scenario:<110} | {avg_score:<15.4f} | {num_points}")

                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Analyze metrics from result files')
    parser.add_argument('--folder', type=str, help='Path to folder containing result files')
    args = parser.parse_args()

    analyze_metrics(args.folder)
