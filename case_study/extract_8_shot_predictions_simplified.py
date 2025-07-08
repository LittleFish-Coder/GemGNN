#!/usr/bin/env python3
"""
Simplified extraction of 8-shot PolitiFact predictions from available data.
Creates CSV files and basic analysis.
"""

import os
import json
import csv
from pathlib import Path

def extract_llm_predictions(base_dir, model_name):
    """Extract predictions from LLM results"""
    pred_file = os.path.join(base_dir, "results", model_name, "politifact", "8-shot", "predictions.json")
    
    if not os.path.exists(pred_file):
        print(f"Warning: {pred_file} not found")
        return None
    
    try:
        with open(pred_file, 'r') as f:
            data = json.load(f)
        
        labels = data['labels']
        predictions = data['predictions'] 
        confidences = data['confidences']
        
        print(f"Extracted {len(labels)} predictions from {model_name}")
        print(f"  Accuracy: {sum(1 for i in range(len(labels)) if labels[i] == predictions[i]) / len(labels):.3f}")
        
        return {
            'labels': labels,
            'predictions': predictions,
            'confidences': confidences,
            'model': model_name.upper()
        }
        
    except Exception as e:
        print(f"Error extracting {model_name} predictions: {e}")
        return None

def extract_related_work_metrics(base_dir, model_name):
    """Extract aggregate metrics from related work models"""
    model_dir = os.path.join(base_dir, "related_work", model_name)
    results_dir = os.path.join(model_dir, f"results_{model_name.lower()}")
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return None
    
    # Find 8-shot PolitiFact result file
    for file in os.listdir(results_dir):
        if "politifact_k8" in file and file.endswith(".json"):
            result_file = os.path.join(results_dir, file)
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                metrics = data.get('final_metrics', {})
                accuracy = metrics.get('accuracy', 0.0)
                f1_score = metrics.get('f1', 0.0)
                
                print(f"Found {model_name} 8-shot results: accuracy={accuracy:.3f}, F1={f1_score:.3f}")
                
                return {
                    'model': model_name,
                    'accuracy': accuracy,
                    'f1_score': f1_score,
                    'file': file
                }
                
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    print(f"No 8-shot PolitiFact results found for {model_name}")
    return None

def get_gemgnn_metrics(base_dir):
    """Extract GemGNN metrics from the specific scenario"""
    metrics_file = os.path.join(
        base_dir, 
        "results_hetero/HAN/politifact/8_shot_deberta_hetero_knn_test_isolated_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3/metrics.json"
    )
    
    if not os.path.exists(metrics_file):
        print(f"GemGNN metrics file not found: {metrics_file}")
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        metrics = data.get('final_test_metrics_on_target_node', {})
        accuracy = metrics.get('accuracy', 0.0)
        f1_score = metrics.get('f1_score', 0.0)
        
        print(f"Found GemGNN_HAN 8-shot results: accuracy={accuracy:.3f}, F1={f1_score:.3f}")
        
        return {
            'model': 'GemGNN_HAN',
            'accuracy': accuracy,
            'f1_score': f1_score,
            'scenario': '8_shot_deberta_hetero_knn_test_isolated_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3'
        }
        
    except Exception as e:
        print(f"Error reading GemGNN metrics: {e}")
        return None

def create_individual_csv_files(output_dir, llm_data, model_metrics):
    """Create individual CSV files for each model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # LLM predictions (Llama, Gemma)
    for model_data in llm_data:
        if model_data:
            filename = f"{model_data['model'].lower()}_8_shot_politifact_predictions.csv"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['news_id', 'ground_truth', 'predicted_label', 'confidence', 'correct']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i in range(len(model_data['labels'])):
                    writer.writerow({
                        'news_id': i,
                        'ground_truth': model_data['labels'][i],
                        'predicted_label': model_data['predictions'][i],
                        'confidence': model_data['confidences'][i],
                        'correct': model_data['labels'][i] == model_data['predictions'][i]
                    })
            
            print(f"Saved {model_data['model']} predictions to {filepath}")
    
    # Model metrics summary
    metrics_file = os.path.join(output_dir, "model_metrics_summary.csv")
    with open(metrics_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'accuracy', 'f1_score', 'notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for metric in model_metrics:
            if metric:
                # Filter to only include expected fields
                filtered_metric = {k: v for k, v in metric.items() if k in fieldnames}
                writer.writerow(filtered_metric)
    
    print(f"Saved model metrics summary to {metrics_file}")

def create_consolidated_csv(output_dir, llm_data):
    """Create consolidated CSV with all available data"""
    if not llm_data or not any(llm_data):
        print("No LLM data available for consolidation")
        return
    
    # Use the first available dataset to determine the number of samples
    num_samples = len(llm_data[0]['labels']) if llm_data[0] else 102  # Default to known test size
    
    filepath = os.path.join(output_dir, "8_shot_politifact_test_results.csv")
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['news_id', 'ground_truth']
        
        # Add columns for each available model
        for model_data in llm_data:
            if model_data:
                model_name = model_data['model'].lower()
                fieldnames.extend([
                    f'{model_name}_prediction',
                    f'{model_name}_confidence',
                    f'{model_name}_correct'
                ])
        
        # Add placeholder columns for models without individual predictions
        for model in ['gemgnn_han', 'less4fd', 'heterosgt', 'genfend']:
            fieldnames.extend([
                f'{model}_prediction',
                f'{model}_confidence',
                f'{model}_correct'
            ])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(num_samples):
            row = {'news_id': i}
            
            # Get ground truth from first available model
            if llm_data[0]:
                row['ground_truth'] = llm_data[0]['labels'][i] if i < len(llm_data[0]['labels']) else -1
            
            # Add data for each LLM model
            for model_data in llm_data:
                if model_data and i < len(model_data['labels']):
                    model_name = model_data['model'].lower()
                    row[f'{model_name}_prediction'] = model_data['predictions'][i]
                    row[f'{model_name}_confidence'] = model_data['confidences'][i]
                    row[f'{model_name}_correct'] = model_data['labels'][i] == model_data['predictions'][i]
            
            # Add placeholder values for other models
            for model in ['gemgnn_han', 'less4fd', 'heterosgt', 'genfend']:
                row[f'{model}_prediction'] = -1  # Placeholder
                row[f'{model}_confidence'] = -1.0  # Placeholder
                row[f'{model}_correct'] = False  # Placeholder
            
            writer.writerow(row)
    
    print(f"Saved consolidated results to {filepath}")

def analyze_llm_performance(llm_data):
    """Analyze LLM performance differences"""
    if len(llm_data) < 2:
        print("Need at least 2 models for comparison")
        return
    
    print("\n=== LLM Performance Comparison ===")
    
    llama_data = None
    gemma_data = None
    
    for model_data in llm_data:
        if model_data and 'LLAMA' in model_data['model']:
            llama_data = model_data
        elif model_data and 'GEMMA' in model_data['model']:
            gemma_data = model_data
    
    if not llama_data or not gemma_data:
        print("Both Llama and Gemma data required for comparison")
        return
    
    # Compare predictions
    llama_correct = [llama_data['labels'][i] == llama_data['predictions'][i] for i in range(len(llama_data['labels']))]
    gemma_correct = [gemma_data['labels'][i] == gemma_data['predictions'][i] for i in range(len(gemma_data['labels']))]
    
    llama_wins = sum(1 for i in range(len(llama_correct)) if llama_correct[i] and not gemma_correct[i])
    gemma_wins = sum(1 for i in range(len(gemma_correct)) if gemma_correct[i] and not llama_correct[i])
    both_correct = sum(1 for i in range(len(llama_correct)) if llama_correct[i] and gemma_correct[i])
    both_wrong = sum(1 for i in range(len(llama_correct)) if not llama_correct[i] and not gemma_correct[i])
    
    print(f"Llama wins (correct when Gemma wrong): {llama_wins}")
    print(f"Gemma wins (correct when Llama wrong): {gemma_wins}")
    print(f"Both correct: {both_correct}")
    print(f"Both wrong: {both_wrong}")
    
    # Find some examples where models disagree
    disagreements = []
    for i in range(len(llama_data['labels'])):
        if llama_data['predictions'][i] != gemma_data['predictions'][i]:
            disagreements.append({
                'news_id': i,
                'ground_truth': llama_data['labels'][i],
                'llama_pred': llama_data['predictions'][i],
                'gemma_pred': gemma_data['predictions'][i],
                'llama_correct': llama_correct[i],
                'gemma_correct': gemma_correct[i]
            })
    
    print(f"\nTotal disagreements: {len(disagreements)}")
    if disagreements:
        print("\nFirst few disagreement examples:")
        for i, case in enumerate(disagreements[:5]):
            correct_model = "Llama" if case['llama_correct'] else ("Gemma" if case['gemma_correct'] else "Neither")
            print(f"  News {case['news_id']}: GT={case['ground_truth']}, Llama={case['llama_pred']}, Gemma={case['gemma_pred']}, Correct: {correct_model}")

def create_case_study_report(output_dir, all_metrics):
    """Create a case study report"""
    report_file = os.path.join(output_dir, "8_shot_politifact_case_study_report.md")
    
    with open(report_file, 'w') as f:
        f.write("# 8-Shot PolitiFact Case Study Report\n\n")
        f.write("## Model Performance Summary\n\n")
        f.write("| Model | Accuracy | F1 Score | Notes |\n")
        f.write("|-------|----------|----------|-------|\n")
        
        for metric in all_metrics:
            if metric:
                f.write(f"| {metric['model']} | {metric['accuracy']:.3f} | {metric.get('f1_score', 'N/A')} | {metric.get('notes', '')} |\n")
        
        f.write("\n## Key Findings\n\n")
        f.write("- **Data Availability**: Individual predictions are available for LLM models (Llama, Gemma)\n")
        f.write("- **Limitation**: Related work models (LESS4FD, HeteroSGT, GenFEND) only provide aggregate metrics\n")
        f.write("- **GemGNN**: Metrics available but individual predictions require model inference\n")
        f.write("\n## Methodology for Case Study\n\n")
        f.write("To complete the comparative analysis showing where GemGNN succeeds but others fail:\n\n")
        f.write("1. **Need GemGNN Predictions**: Run inference on the target graph to get individual predictions\n")
        f.write("2. **Identify Success Cases**: Find examples where GemGNN is correct but strong baselines fail\n")
        f.write("3. **Analyze Reasons**: Examine why graph structure and multi-view approach help\n")
        f.write("\n## Next Steps\n\n")
        f.write("1. Generate GemGNN individual predictions using case_study.py\n")
        f.write("2. Extract individual predictions from related work models (if possible)\n")
        f.write("3. Perform detailed comparative analysis\n")
        f.write("4. Write examples demonstrating GemGNN advantages\n")
    
    print(f"Saved case study report to {report_file}")

def main():
    """Main execution function"""
    base_dir = "/home/runner/work/GemGNN/GemGNN"
    output_dir = os.path.join(base_dir, "case_study", "8_shot_predictions")
    
    print("=== 8-Shot PolitiFact Predictions Extraction ===\n")
    
    # Extract LLM predictions
    llm_data = []
    for model in ['llama', 'gemma']:
        data = extract_llm_predictions(base_dir, model)
        llm_data.append(data)
    
    # Extract related work metrics
    related_work_metrics = []
    for model in ['LESS4FD', 'HeteroSGT', 'GenFEND']:
        metrics = extract_related_work_metrics(base_dir, model)
        if metrics:
            metrics['notes'] = 'Aggregate metrics only'
            related_work_metrics.append(metrics)
    
    # Extract GemGNN metrics
    gemgnn_metrics = get_gemgnn_metrics(base_dir)
    if gemgnn_metrics:
        gemgnn_metrics['notes'] = 'Requires inference for individual predictions'
        related_work_metrics.append(gemgnn_metrics)
    
    # Create output files
    create_individual_csv_files(output_dir, llm_data, related_work_metrics)
    create_consolidated_csv(output_dir, llm_data)
    
    # Analyze LLM performance
    analyze_llm_performance(llm_data)
    
    # Create case study report
    all_metrics = []
    for data in llm_data:
        if data:
            accuracy = sum(1 for i in range(len(data['labels'])) if data['labels'][i] == data['predictions'][i]) / len(data['labels'])
            all_metrics.append({
                'model': data['model'],
                'accuracy': accuracy,
                'f1_score': 'Calculated',
                'notes': 'Individual predictions available'
            })
    all_metrics.extend(related_work_metrics)
    
    create_case_study_report(output_dir, all_metrics)
    
    print(f"\n=== Summary ===")
    print(f"Results saved to: {output_dir}")
    print(f"Files created:")
    print(f"  - Individual CSV files for each model")
    print(f"  - Consolidated CSV: 8_shot_politifact_test_results.csv")
    print(f"  - Metrics summary: model_metrics_summary.csv")
    print(f"  - Case study report: 8_shot_politifact_case_study_report.md")

if __name__ == "__main__":
    main()