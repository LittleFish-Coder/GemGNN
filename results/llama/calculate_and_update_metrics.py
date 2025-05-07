import os
import json
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np # confusion_matrix 可能返回 numpy int64，需要轉換

def calculate_metrics_from_predictions(predictions_file_path):
    """
    Reads predictions.json, calculates new metrics, and returns them.
    """
    try:
        with open(predictions_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {predictions_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {predictions_file_path}")
        return None

    true_labels = data.get('labels')
    predicted_labels = data.get('predictions')

    if true_labels is None or predicted_labels is None:
        print(f"Error: 'labels' or 'predictions' key missing in {predictions_file_path}")
        return None

    if not isinstance(true_labels, list) or not isinstance(predicted_labels, list):
        print(f"Error: 'labels' or 'predictions' are not lists in {predictions_file_path}")
        return None

    if not true_labels or not predicted_labels: # Check if lists are empty
        print(f"Warning: 'labels' or 'predictions' list is empty in {predictions_file_path}. Skipping.")
        return { # Return empty metrics so we know it was processed but had no data
            "accuracy": 0,
            "macro_precision": 0,
            "macro_recall": 0,
            "macro_f1_score": 0,
            "confusion_matrix": [[0,0],[0,0]] # Or however you want to represent empty CM
        }


    # Determine all unique labels present in true_labels and predicted_labels
    # This helps ensure confusion_matrix has the right dimensions, especially if some classes
    # are not predicted or not present in a small test set.
    # For binary (0, 1), this is usually straightforward.
    unique_labels = sorted(list(set(true_labels) | set(predicted_labels)))
    if not unique_labels: # If after all that, still no labels
        unique_labels = [0, 1] # Default to binary if completely empty, or handle as error
        print(f"Warning: No unique labels found in {predictions_file_path}, defaulting to {unique_labels} for CM.")


    # Calculate metrics
    # Ensure labels parameter is passed to confusion_matrix for consistent ordering
    # If you know your labels are always e.g. [0, 1], you can hardcode it: labels=[0,1]
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
    acc = accuracy_score(true_labels, predicted_labels)
    macro_precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    macro_recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    # Convert confusion matrix to list of lists for JSON serialization
    # (numpy arrays are not directly JSON serializable by default)
    cm_list = [[int(x) for x in row] for row in cm] # Convert numpy.int64 to standard int

    new_metrics = {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "confusion_matrix": cm_list,
        # You can also add per-class metrics if needed
        # "per_class_precision": precision_score(true_labels, predicted_labels, average=None, labels=unique_labels, zero_division=0).tolist(),
        # "per_class_recall": recall_score(true_labels, predicted_labels, average=None, labels=unique_labels, zero_division=0).tolist(),
        # "per_class_f1_score": f1_score(true_labels, predicted_labels, average=None, labels=unique_labels, zero_division=0).tolist(),
    }
    return new_metrics

def update_metrics_file(metrics_file_path, new_calculated_metrics):
    """
    Reads an existing metrics.json, updates it with new metrics, and writes it back.
    """
    try:
        with open(metrics_file_path, 'r') as f:
            existing_metrics = json.load(f)
    except FileNotFoundError:
        print(f"Info: Metrics file not found at {metrics_file_path}. Creating a new one.")
        existing_metrics = {} # Create new if not exists
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {metrics_file_path}. Overwriting with new metrics.")
        existing_metrics = {}

    # Update existing metrics with the newly calculated ones
    # This will overwrite if keys already exist, or add if they don't
    existing_metrics.update(new_calculated_metrics)

    # Add a note that these metrics were (re)calculated by this script
    existing_metrics["calculation_info"] = "Metrics (re)calculated from predictions.json by calculate_and_update_metrics.py"


    try:
        with open(metrics_file_path, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
        print(f"Successfully updated metrics file: {metrics_file_path}")
    except IOError:
        print(f"Error: Could not write to metrics file: {metrics_file_path}")


def main():
    # results_base_dir = "results" # As in your prompt_hf_llm.py
    # For the structure you showed in the latest image (e.g. "politifact/3-shot"):
    # We need to define the top-level directory containing different datasets/model results
    # Let's assume your structure is something like:
    # ./results_root/
    # ├── model_A_results/
    # │   ├── dataset_X/
    # │   │   ├── 3-shot/
    # │   │   │   ├── predictions.json
    # │   │   │   └── metrics.json (to be updated)
    # │   │   └── 5-shot/
    # │   │       └── ...
    # │   └── dataset_Y/
    # │       └── ...
    # └── model_B_results/
    #     └── ...

    # For your specific image which shows "politifact" then "3-shot", "4-shot" etc.
    # The script should be run from a directory that allows it to see "politifact"
    # Or you provide the path to "politifact"
    # Let's assume a base directory where dataset folders reside.

    # Option 1: Manually specify the dataset directory
    # dataset_dirs_to_process = ["path/to/politifact", "path/to/another_dataset_results"]

    # Option 2: Scan from a root directory (more flexible)
    # This script assumes it's in a directory from which it can see 'politifact', 'gossipcop' etc.
    # or you set results_root_dir to their parent.
    results_root_dir = "." # Current directory. Change if your datasets are elsewhere, e.g., "results/bert" or "results/llama"

    print(f"Scanning for datasets in: {os.path.abspath(results_root_dir)}")

    for dataset_name in os.listdir(results_root_dir):
        dataset_path = os.path.join(results_root_dir, dataset_name)
        if os.path.isdir(dataset_path): # Check if it's a directory
            print(f"\nProcessing dataset: {dataset_name}")
            for k_shot_folder_name in os.listdir(dataset_path):
                if k_shot_folder_name.endswith("-shot"): # Basic check for k-shot folder
                    k_shot_path = os.path.join(dataset_path, k_shot_folder_name)
                    if os.path.isdir(k_shot_path):
                        predictions_json_path = os.path.join(k_shot_path, "predictions.json")
                        metrics_json_path = os.path.join(k_shot_path, "metrics.json")

                        if os.path.exists(predictions_json_path):
                            print(f"  Found predictions.json in {k_shot_path}")
                            new_metrics_to_add = calculate_metrics_from_predictions(predictions_json_path)
                            if new_metrics_to_add:
                                update_metrics_file(metrics_json_path, new_metrics_to_add)
                        else:
                            print(f"  Skipping {k_shot_path}: predictions.json not found.")
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()