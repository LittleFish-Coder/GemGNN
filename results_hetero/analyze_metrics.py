import os
import json

def analyze_metrics(folder_path):
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return

    results = {}

    # try sorting in this order: 0shot, 3shot, 4shot, 5shot, ... 16shot
    # Get immediate subdirectories and sort them
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
        
        # Find results.txt file in subdirectory
        for file in os.listdir(subdir_path):
            if file.startswith('metrics_'):
                file_path = os.path.join(subdir_path, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'test_metrics' in data:
                        results[subdir] = data['test_metrics']['f1_score']
                    elif 'final_test_metrics' in data:
                        results[subdir] = data['final_test_metrics']['f1_score']
                    elif 'final_test_metrics_on_target_node' in data:
                        results[subdir] = data['final_test_metrics_on_target_node']['f1_score']
                    else:
                        results[subdir] = data['f1']
                    break


    f1_record = [] # store f1 scores in sequence
    # Print results in a structured format
    print("\nResults Analysis:")
    for k_shot, f1_score in results.items():
        print(f"\nK-Shot: {k_shot}")
        print(f"F1 Score: {f1_score}")
        f1_record.append(f1_score)
    
    print("\nF1 Scores in sequence:")
    print(f1_record)

                

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze metrics from result files')
    parser.add_argument('--folder_path', type=str, help='Path to folder containing result files')
    
    args = parser.parse_args()
    analyze_metrics(args.folder_path)
