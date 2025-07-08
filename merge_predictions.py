#!/usr/bin/env python3
"""
Merge all model predictions into a single comparison file.
"""

import pandas as pd
import os

def merge_predictions():
    """Merge all 8-shot politifact predictions into one comparison file."""
    
    # Define the models and their corresponding files (ordered by performance)
    models = {
        'gemgnn': 'gemgnn_8_shot_politifact_predictions.csv',
        'genfend': 'genfend_8_shot_politifact_predictions.csv',
        'llama': 'llama_8_shot_politifact_predictions.csv',
        'gemma': 'gemma_8_shot_politifact_predictions.csv', 
        'heterosgt': 'heterosgt_8_shot_politifact_predictions.csv',
        'less4fd': 'less4fd_8_shot_politifact_predictions.csv'
    }
    
    prediction_dir = 'prediction'
    
    # Load the first model as base (with news_id, ground_truth first, then predictions, news_text last)
    base_file = os.path.join(prediction_dir, models['gemgnn'])
    df_base = pd.read_csv(base_file)
    
    # Start with news_id and ground_truth
    df_merged = df_base[['news_id', 'ground_truth']].copy()
    
    print(f"Base dataframe shape: {df_merged.shape}")
    
    # Add predictions from each model
    for model_name, filename in models.items():
        filepath = os.path.join(prediction_dir, filename)
        
        if os.path.exists(filepath):
            df_model = pd.read_csv(filepath)
            # Add this model's predictions
            df_merged[f'{model_name}_prediction'] = df_model['prediction']
            print(f"Added {model_name} predictions from {filename}")
        else:
            print(f"Warning: {filepath} not found, skipping {model_name}")
    
    # Add news_text as the last column
    df_merged['news_text'] = df_base['news_text']
    
    # Save the merged file
    output_file = os.path.join(prediction_dir, '8_shot_politifact_predictions.csv')
    df_merged.to_csv(output_file, index=False)
    
    print(f"\nMerged predictions saved to: {output_file}")
    print(f"Final shape: {df_merged.shape}")
    print(f"Columns: {df_merged.columns.tolist()}")
    
    # Show a sample of the merged data (excluding news_text for readability)
    print("\nSample of merged predictions (first 10 rows):")
    prediction_cols = [col for col in df_merged.columns if col.endswith('_prediction')]
    display_cols = ['news_id', 'ground_truth'] + prediction_cols
    print(df_merged[display_cols].head(10))
    
    # Calculate and show accuracy for each model
    print("\n=== Model Accuracy Summary ===")
    for model_name in models.keys():
        col_name = f'{model_name}_prediction'
        if col_name in df_merged.columns:
            accuracy = (df_merged['ground_truth'] == df_merged[col_name]).mean()
            print(f"{model_name.upper():>10}: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    merge_predictions()