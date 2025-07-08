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

def load_hf_dataset():
    """Load the PolitiFact test dataset from HuggingFace"""
    try:
        dataset = load_dataset("LittleFish-Coder/Fake_News_politifact", split="test")
        print(f"Loaded HuggingFace dataset with {len(dataset)} test samples")
        return dataset
    except Exception as e:
        print(f"Error loading HuggingFace dataset: {e}")
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
        
        results = []
        for i in range(len(labels)):
            results.append({
                'news_id': i,
                'text': dataset[i]['text'][:200] + "..." if len(dataset[i]['text']) > 200 else dataset[i]['text'],
                'ground_truth': labels[i],
                'predicted_label': predictions[i],
                'confidence': confidences[i],
                'model': model_name.upper()
            })
        
        print(f"Extracted {len(results)} predictions from {model_name}")
        return results
        
    except Exception as e:
        print(f"Error extracting {model_name} predictions: {e}")
        return None

def extract_related_work_predictions(related_work_dir, model_name, dataset):
    """Extract predictions from related work models (LESS4FD, HeteroSGT, GenFEND)"""
    # These models only have aggregate metrics, not individual predictions
    # We'll create placeholder data indicating this limitation
    print(f"Warning: {model_name} results only contain aggregate metrics, not individual predictions")
    
    # Find the 8-shot result file
    result_files = []
    model_dir = os.path.join(related_work_dir, model_name)
    
    if os.path.exists(model_dir):
        results_dir = os.path.join(model_dir, f"results_{model_name.lower()}")
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if "politifact_k8" in file and file.endswith(".json"):
                    result_files.append(os.path.join(results_dir, file))
    
    if not result_files:
        print(f"No 8-shot PolitiFact results found for {model_name}")
        return None
    
    # Use the first matching file
    result_file = result_files[0]
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract overall metrics
        metrics = data.get('final_metrics', {})
        accuracy = metrics.get('accuracy', 0.0)
        
        print(f"Found {model_name} 8-shot results: accuracy={accuracy:.3f}")
        
        # Create placeholder results (we don't have individual predictions)
        results = []
        for i in range(len(dataset)):
            results.append({
                'news_id': i,
                'text': dataset[i]['text'][:200] + "..." if len(dataset[i]['text']) > 200 else dataset[i]['text'],
                'ground_truth': dataset[i]['label'],
                'predicted_label': -1,  # Placeholder: no individual predictions available
                'confidence': -1.0,     # Placeholder: no individual predictions available
                'model': model_name
            })
        
        print(f"Created {len(results)} placeholder entries for {model_name}")
        return results
        
    except Exception as e:
        print(f"Error extracting {model_name} results: {e}")
        return None

def create_gemgnn_predictions(dataset):
    """Create placeholder for GemGNN predictions (would need actual model inference)"""
    print("Warning: GemGNN predictions require running inference on the specific graph")
    print("Creating placeholder data - individual predictions would need to be generated")
    
    results = []
    for i in range(len(dataset)):
        results.append({
            'news_id': i,
            'text': dataset[i]['text'][:200] + "..." if len(dataset[i]['text']) > 200 else dataset[i]['text'],
            'ground_truth': dataset[i]['label'],
            'predicted_label': -1,  # Placeholder: would need actual inference
            'confidence': -1.0,     # Placeholder: would need actual inference
            'model': 'GemGNN_HAN'
        })
    
    print(f"Created {len(results)} placeholder entries for GemGNN")
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
    
    # Convert to DataFrame
    all_data = []
    for model_preds in all_predictions:
        if model_preds:
            all_data.extend(model_preds)
    
    if not all_data:
        print("No valid predictions found")
        return
    
    df = pd.DataFrame(all_data)
    
    # Pivot to have models as columns
    pivot_df = df.pivot_table(
        index=['news_id', 'text', 'ground_truth'], 
        columns='model', 
        values=['predicted_label', 'confidence'],
        aggfunc='first'
    ).reset_index()
    
    # Flatten column names
    pivot_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pivot_df.columns.values]
    
    # Save consolidated results
    filepath = os.path.join(output_dir, "8_shot_politifact_test_results.csv")
    pivot_df.to_csv(filepath, index=False)
    print(f"Saved consolidated results to {filepath}")
    
    return pivot_df

def analyze_comparative_performance(consolidated_df):
    """Analyze cases where GemGNN succeeds but others fail"""
    print("\n=== Comparative Performance Analysis ===")
    
    # This would need actual predictions to work properly
    print("Note: Analysis requires actual model predictions, not placeholder data")
    print("Would identify cases where:")
    print("1. GemGNN predicts correctly but LLMs (Llama/Gemma) predict incorrectly")
    print("2. GemGNN predicts correctly but related work methods fail")
    
    # Example of what the analysis would look like:
    """
    if 'predicted_label_GemGNN_HAN' in consolidated_df.columns:
        gemgnn_correct = consolidated_df['predicted_label_GemGNN_HAN'] == consolidated_df['ground_truth']
        llama_correct = consolidated_df['predicted_label_LLAMA'] == consolidated_df['ground_truth']
        
        gemgnn_wins = gemgnn_correct & ~llama_correct
        print(f"Cases where GemGNN succeeds but Llama fails: {gemgnn_wins.sum()}")
        
        if gemgnn_wins.sum() > 0:
            examples = consolidated_df[gemgnn_wins].head(2)
            print("\nExample cases:")
            for idx, row in examples.iterrows():
                print(f"News {row['news_id']}: {row['text'][:100]}...")
                print(f"  Ground Truth: {row['ground_truth']}")
                print(f"  GemGNN: {row['predicted_label_GemGNN_HAN']}")
                print(f"  Llama: {row['predicted_label_LLAMA']}")
                print()
    """

def main():
    """Main execution function"""
    # Set up paths
    base_dir = "/home/runner/work/GemGNN/GemGNN"
    results_dir = os.path.join(base_dir, "results")
    related_work_dir = os.path.join(base_dir, "related_work")
    case_study_dir = os.path.join(base_dir, "case_study")
    output_dir = os.path.join(case_study_dir, "8_shot_predictions")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_hf_dataset()
    if not dataset:
        print("Failed to load dataset, exiting")
        return
    
    print(f"\nExtracting predictions for 8-shot PolitiFact scenario...")
    print(f"Dataset size: {len(dataset)} test samples")
    
    all_predictions = []
    
    # Extract LLM predictions
    for model in ['llama', 'gemma']:
        preds = extract_llm_predictions(results_dir, model, dataset)
        if preds:
            all_predictions.append(preds)
            save_individual_csv(preds, model, output_dir)
    
    # Extract related work predictions
    for model in ['LESS4FD', 'HeteroSGT', 'GenFEND']:
        preds = extract_related_work_predictions(related_work_dir, model, dataset)
        if preds:
            all_predictions.append(preds)
            save_individual_csv(preds, model, output_dir)
    
    # Create GemGNN predictions (placeholder)
    gemgnn_preds = create_gemgnn_predictions(dataset)
    if gemgnn_preds:
        all_predictions.append(gemgnn_preds)
        save_individual_csv(gemgnn_preds, "GemGNN_HAN", output_dir)
    
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
    print("1. Run actual GemGNN inference to get real predictions")
    print("2. Update case study report with comparative analysis")
    print("3. Identify specific examples where GemGNN excels")

if __name__ == "__main__":
    main()