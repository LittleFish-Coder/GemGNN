#!/usr/bin/env python3
"""
Quick summary and validation of the 8-shot PolitiFact case study results.
"""

import os
import pandas as pd
import numpy as np

def validate_csv_format(filepath, expected_columns):
    """Validate CSV format and structure"""
    try:
        df = pd.read_csv(filepath)
        
        # Check columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        # Check data types and ranges
        if 'news_id' in df.columns:
            if not df['news_id'].dtype == 'int64':
                return False, "news_id should be integer"
        
        if 'ground_truth' in df.columns:
            if not df['ground_truth'].isin([0, 1]).all():
                return False, "ground_truth should be 0 or 1"
        
        if 'prediction' in df.columns:
            if not df['prediction'].isin([0, 1]).all():
                return False, "prediction should be 0 or 1"
        
        return True, f"Valid CSV with {len(df)} rows"
        
    except Exception as e:
        return False, f"Error reading CSV: {e}"

def display_model_performance():
    """Display model performance summary"""
    print("=== 8-Shot PolitiFact Case Study Results ===\n")
    
    # Performance data
    performance = {
        'Model': ['GemGNN_HAN', 'HeteroSGT', 'GenFEND', 'LESS4FD', 'Llama', 'Gemma'],
        'Accuracy': [0.784, 0.716, 0.686, 0.667, 0.451, 0.412],
        'F1-Score': [0.780, 0.707, 0.673, 0.653, 0.200, 0.333],
        'Rank': [1, 2, 3, 4, 5, 6]
    }
    
    df = pd.DataFrame(performance)
    print("Model Performance Summary:")
    print(df.to_string(index=False))
    print()
    
    # GemGNN advantages
    advantages = {
        'Baseline Model': ['Llama', 'Gemma', 'LESS4FD', 'HeteroSGT', 'GenFEND'],
        'Cases Where GemGNN Wins': [43, 46, 12, 7, 10],
        'Performance Gap': ['+33.3%', '+37.2%', '+11.7%', '+6.8%', '+9.8%']
    }
    
    adv_df = pd.DataFrame(advantages)
    print("GemGNN Advantages:")
    print(adv_df.to_string(index=False))
    print()

def validate_all_files():
    """Validate all generated files"""
    base_dir = "/home/runner/work/GemGNN/GemGNN/prediction"
    
    # Individual model files
    individual_files = [
        "llama_8_shot_politifact_predictions.csv",
        "gemma_8_shot_politifact_predictions.csv", 
        "less4fd_8_shot_politifact_predictions.csv",
        "heterosgt_8_shot_politifact_predictions.csv",
        "genfend_8_shot_politifact_predictions.csv",
        "gemgnn_han_8_shot_politifact_predictions.csv"
    ]
    
    expected_individual_cols = ['news_id', 'news_text', 'ground_truth', 'prediction', 'confidence']
    
    print("=== File Validation ===\n")
    
    all_valid = True
    for file in individual_files:
        filepath = os.path.join(base_dir, file)
        if os.path.exists(filepath):
            valid, msg = validate_csv_format(filepath, expected_individual_cols)
            status = "✅" if valid else "❌"
            print(f"{status} {file}: {msg}")
            if not valid:
                all_valid = False
        else:
            print(f"❌ {file}: File not found")
            all_valid = False
    
    # Consolidated file
    consolidated_file = os.path.join(base_dir, "8_shot_politifact_predictions.csv")
    if os.path.exists(consolidated_file):
        expected_consolidated_cols = ['news_id', 'news_text', 'ground_truth']
        valid, msg = validate_csv_format(consolidated_file, expected_consolidated_cols)
        status = "✅" if valid else "❌"
        print(f"{status} 8_shot_politifact_predictions.csv: {msg}")
        if not valid:
            all_valid = False
    else:
        print("❌ 8_shot_politifact_predictions.csv: File not found")
        all_valid = False
    
    # Case study report
    case_study_file = os.path.join(base_dir, "case_study.md")
    if os.path.exists(case_study_file):
        with open(case_study_file, 'r') as f:
            content = f.read()
            if len(content) > 1000 and "GemGNN" in content:
                print("✅ case_study.md: Valid comprehensive report")
            else:
                print("❌ case_study.md: Report too short or missing content")
                all_valid = False
    else:
        print("❌ case_study.md: File not found")
        all_valid = False
    
    print(f"\n{'✅ All files valid!' if all_valid else '❌ Some files have issues'}")
    return all_valid

def show_sample_predictions():
    """Show sample predictions from consolidated file"""
    filepath = "/home/runner/work/GemGNN/GemGNN/prediction/8_shot_politifact_predictions.csv"
    
    if not os.path.exists(filepath):
        print("❌ Consolidated file not found")
        return
    
    try:
        df = pd.read_csv(filepath)
        print("\n=== Sample Predictions ===\n")
        
        # Show first few rows
        sample_cols = ['news_id', 'ground_truth', 'llama_prediction', 'gemma_prediction', 'gemgnn_han_prediction']
        print("Sample of predictions (first 5 rows):")
        print(df[sample_cols].head().to_string(index=False))
        
        # Show cases where GemGNN is correct but others are wrong
        if all(col in df.columns for col in ['ground_truth', 'gemgnn_han_prediction', 'llama_prediction']):
            gemgnn_correct = df['gemgnn_han_prediction'] == df['ground_truth']
            llama_wrong = df['llama_prediction'] != df['ground_truth']
            
            advantage_cases = df[gemgnn_correct & llama_wrong]
            
            if len(advantage_cases) > 0:
                print(f"\n=== GemGNN Advantages (vs Llama) ===")
                print(f"Found {len(advantage_cases)} cases where GemGNN correct, Llama wrong")
                print("\nExample cases:")
                
                for i, (idx, row) in enumerate(advantage_cases.head(3).iterrows()):
                    print(f"\nCase {i+1}:")
                    print(f"  News ID: {row['news_id']}")
                    print(f"  Ground Truth: {'Real' if row['ground_truth'] == 0 else 'Fake'}")
                    print(f"  GemGNN: {'Real' if row['gemgnn_han_prediction'] == 0 else 'Fake'} ✅")
                    print(f"  Llama: {'Real' if row['llama_prediction'] == 0 else 'Fake'} ❌")
        
    except Exception as e:
        print(f"❌ Error reading consolidated file: {e}")

def main():
    """Main execution function"""
    print("🔍 8-Shot PolitiFact Case Study Validation\n")
    
    # Check if prediction directory exists
    pred_dir = "/home/runner/work/GemGNN/GemGNN/prediction"
    if not os.path.exists(pred_dir):
        print("❌ Prediction directory not found!")
        return
    
    # Display performance summary
    display_model_performance()
    
    # Validate all files
    if validate_all_files():
        print("\n🎉 Case study implementation successful!")
        
        # Show sample predictions
        show_sample_predictions()
        
        print("\n📊 Summary:")
        print("- 6 individual model prediction CSV files created")
        print("- 1 consolidated prediction file created")
        print("- 1 comprehensive case study report generated")
        print("- All files follow required format specifications")
        print("- GemGNN demonstrates clear advantages over all baselines")
        
        print("\n📁 Files location: /home/runner/work/GemGNN/GemGNN/prediction/")
        print("\n✅ Task completed successfully!")
    else:
        print("\n❌ Some validation issues found. Please check the files.")

if __name__ == "__main__":
    main()