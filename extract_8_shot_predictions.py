#!/usr/bin/env python3
"""
Extract predictions from all models for 8-shot PolitiFact scenario.
Creates individual CSV files and consolidated results for case study.
"""

import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import sys
from utils.sample_k_shot import sample_k_shot

def load_hf_dataset_with_seed(seed=42):
    """Load the PolitiFact test dataset from HuggingFace with specific seed"""
    print(f"Loading PolitiFact dataset with seed {seed}...")
    try:
        dataset = load_dataset("LittleFish-Coder/Fake_News_PolitiFact", cache_dir="dataset")
        print(f"Dataset loaded: {len(dataset['test'])} test samples")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def extract_llm_predictions(results_dir, model_name, dataset):
    """Extract predictions from LLM results (Llama/Gemma)"""
    pred_file = os.path.join(results_dir, model_name, "politifact", "8-shot", "predictions.json")
    
    if not os.path.exists(pred_file):
        print(f"Warning: {pred_file} not found")
        return None
    
    try:
        with open(pred_file, 'r') as f:
            data = json.load(f)
        
        labels = data['labels']
        predictions = data['predictions'] 
        confidences = data['confidences']
        
        # Use test split for mapping
        test_data = dataset['test']
        
        results = []
        for i in range(len(labels)):
            # Ensure we don't exceed test data length
            if i < len(test_data):
                text = test_data[i]['text']
                # Truncate text for readability
                truncated_text = text[:200] + "..." if len(text) > 200 else text
                
                results.append({
                    'news_id': i,
                    'news_text': truncated_text,
                    'ground_truth': labels[i],
                    'prediction': predictions[i],
                    'confidence': confidences[i]
                })
        
        print(f"Extracted {len(results)} predictions from {model_name}")
        return results
        
    except Exception as e:
        print(f"Error extracting {model_name} predictions: {e}")
        return None

def run_related_work_model(model_name, dataset_name="politifact", k_shot=8):
    """Run related work models to get predictions"""
    print(f"\nRunning {model_name} for {k_shot}-shot {dataset_name}...")
    
    if model_name == "LESS4FD":
        # Build graph first
        build_cmd = f"cd /home/runner/work/GemGNN/GemGNN && python related_work/LESS4FD/build_less4fd_graph.py --dataset_name {dataset_name} --k_shot {k_shot}"
        print(f"Building LESS4FD graph: {build_cmd}")
        os.system(build_cmd)
        
        # Train model
        train_cmd = f"cd /home/runner/work/GemGNN/GemGNN && python related_work/LESS4FD/train_less4fd.py --dataset_name {dataset_name} --k_shot {k_shot}"
        print(f"Training LESS4FD: {train_cmd}")
        os.system(train_cmd)
        
    elif model_name == "HeteroSGT":
        # Build graph first
        build_cmd = f"cd /home/runner/work/GemGNN/GemGNN && python related_work/HeteroSGT/build_heterosgt_graph.py --dataset_name {dataset_name} --k_shot {k_shot}"
        print(f"Building HeteroSGT graph: {build_cmd}")
        os.system(build_cmd)
        
        # Train model  
        train_cmd = f"cd /home/runner/work/GemGNN/GemGNN && python related_work/HeteroSGT/train_heterosgt.py --dataset_name {dataset_name} --k_shot {k_shot}"
        print(f"Training HeteroSGT: {train_cmd}")
        os.system(train_cmd)
        
    elif model_name == "GenFEND":
        # Build data first
        build_cmd = f"cd /home/runner/work/GemGNN/GemGNN && python related_work/GenFEND/build.py --dataset_name {dataset_name} --k_shot {k_shot}"
        print(f"Building GenFEND data: {build_cmd}")
        os.system(build_cmd)
        
        # Train model
        train_cmd = f"cd /home/runner/work/GemGNN/GemGNN && python related_work/GenFEND/train.py --dataset_name {dataset_name} --k_shot {k_shot}"
        print(f"Training GenFEND: {train_cmd}")
        os.system(train_cmd)

def extract_related_work_predictions(model_name, dataset, k_shot=8):
    """Extract predictions from related work models after training"""
    print(f"Extracting {model_name} predictions...")
    
    # For now, create placeholder since we need to run inference to get individual predictions
    # These models typically only save aggregate metrics, not individual test predictions
    test_data = dataset['test']
    
    results = []
    for i in range(len(test_data)):
        text = test_data[i]['text']
        truncated_text = text[:200] + "..." if len(text) > 200 else text
        
        results.append({
            'news_id': i,
            'news_text': truncated_text,
            'ground_truth': test_data[i]['label'],
            'prediction': -1,  # Placeholder - would need actual inference
            'confidence': -1.0  # Placeholder
        })
    
    print(f"Created {len(results)} placeholder entries for {model_name}")
    print(f"Note: {model_name} requires running inference to get actual predictions")
    return results

def run_gemgnn_inference(graph_path, dataset):
    """Run GemGNN inference on the specific graph"""
    print(f"\nRunning GemGNN inference on {graph_path}...")
    
    # Use case_study.py or train_hetero_graph.py for inference
    inference_cmd = f"cd /home/runner/work/GemGNN/GemGNN && python train_hetero_graph.py --graph_path {graph_path} --model HAN --loss_fn ce --epochs 0 --inference_only"
    print(f"Running GemGNN inference: {inference_cmd}")
    
    # For now, create placeholder
    test_data = dataset['test']
    results = []
    for i in range(len(test_data)):
        text = test_data[i]['text']
        truncated_text = text[:200] + "..." if len(text) > 200 else text
        
        results.append({
            'news_id': i,
            'news_text': truncated_text,
            'ground_truth': test_data[i]['label'],
            'prediction': -1,  # Placeholder - would need actual inference
            'confidence': -1.0  # Placeholder
        })
    
    print(f"Created {len(results)} placeholder entries for GemGNN")
    print("Note: GemGNN requires running inference to get actual predictions")
    return results

def save_individual_csv(predictions, model_name, output_dir):
    """Save individual model predictions to CSV"""
    if not predictions:
        return
    
    df = pd.DataFrame(predictions)
    filename = f"{model_name.lower()}_8_shot_politifact_predictions.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(predictions)} predictions to {filepath}")

def create_consolidated_csv(all_predictions, output_dir):
    """Create consolidated CSV with all model predictions"""
    if not all_predictions:
        print("No predictions to consolidate")
        return
    
    # Find the maximum number of samples
    max_samples = max(len(preds) for preds in all_predictions if preds)
    
    # Create consolidated dataframe
    consolidated_data = []
    
    for i in range(max_samples):
        row = {'news_id': i, 'news_text': '', 'ground_truth': -1}
        
        # Add predictions from each model
        for model_preds in all_predictions:
            if model_preds and i < len(model_preds):
                sample = model_preds[i]
                if row['news_text'] == '':
                    row['news_text'] = sample['news_text']
                    row['ground_truth'] = sample['ground_truth']
                
                # Extract model name from first prediction
                if len(model_preds) > 0:
                    first_sample = model_preds[0]
                    # Try to infer model name from structure or use index
                    model_names = ['llama', 'gemma', 'less4fd', 'heterosgt', 'genfend', 'gemgnn_han']
                    model_idx = all_predictions.index(model_preds)
                    if model_idx < len(model_names):
                        model_name = model_names[model_idx]
                        row[f'{model_name}_prediction'] = sample['prediction']
                        row[f'{model_name}_confidence'] = sample['confidence']
        
        consolidated_data.append(row)
    
    # Save consolidated results
    df = pd.DataFrame(consolidated_data)
    filepath = os.path.join(output_dir, "8_shot_politifact_predictions.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved consolidated results to {filepath}")
    
    return df

def analyze_comparative_performance(consolidated_df):
    """Analyze cases where GemGNN succeeds but others fail"""
    # This will be implemented after we have actual predictions
    print("\n=== Comparative Performance Analysis ===")
    print("Note: Analysis requires actual model predictions")
    print("Currently showing placeholder data structure")
    
    if 'gemgnn_han_prediction' in consolidated_df.columns:
        # Example analysis that can be done once we have real predictions
        print("Sample analysis structure:")
        print("- Cases where GemGNN correct but Llama wrong")
        print("- Cases where GemGNN correct but Gemma wrong") 
        print("- Cases where GemGNN correct but other baselines wrong")

def main():
    """Main execution function"""
    base_dir = "/home/runner/work/GemGNN/GemGNN"
    output_dir = os.path.join(base_dir, "prediction")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 8-Shot PolitiFact Predictions Extraction ===\n")
    
    # Load dataset
    dataset = load_hf_dataset_with_seed(seed=42)
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    
    all_predictions = []
    
    # Extract LLM predictions
    results_dir = os.path.join(base_dir, "results")
    for model in ['llama', 'gemma']:
        preds = extract_llm_predictions(results_dir, model, dataset)
        if preds:
            all_predictions.append(preds)
            save_individual_csv(preds, model, output_dir)
    
    # Run and extract related work predictions
    for model in ['LESS4FD', 'HeteroSGT', 'GenFEND']:
        print(f"\n=== Processing {model} ===")
        
        # Run the model (this will take time)
        run_related_work_model(model)
        
        # Extract predictions (placeholder for now)
        preds = extract_related_work_predictions(model, dataset)
        if preds:
            all_predictions.append(preds)
            save_individual_csv(preds, model, output_dir)
    
    # Extract GemGNN predictions
    print(f"\n=== Processing GemGNN ===")
    graph_path = os.path.join(base_dir, "graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3/graph.pt")
    
    if os.path.exists(graph_path):
        preds = run_gemgnn_inference(graph_path, dataset)
        if preds:
            all_predictions.append(preds)
            save_individual_csv(preds, "GemGNN_HAN", output_dir)
    else:
        print(f"Warning: GemGNN graph not found at {graph_path}")
    
    # Create consolidated CSV
    consolidated_df = create_consolidated_csv(all_predictions, output_dir)
    
    # Analyze comparative performance
    if consolidated_df is not None:
        analyze_comparative_performance(consolidated_df)
    
    print(f"\n=== Summary ===")
    print(f"Extracted predictions from {len(all_predictions)} models")
    print(f"Results saved to: {output_dir}")
    print(f"Individual CSV files and consolidated results created")
    
    print(f"\nNext steps:")
    print("1. Run actual model inference to get real predictions")
    print("2. Update consolidated CSV with real predictions")
    print("3. Perform detailed comparative analysis") 
    print("4. Write case study report with specific examples")

if __name__ == "__main__":
    main()