# -*- coding: utf-8 -*-
"""
Direct Question Contamination Checker
This version only uses direct questioning to check for training data contamination.
"""
import argparse
import random
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from datasets import load_dataset
import pandas as pd
import warnings
from tqdm import tqdm
import json
import datetime
import time
warnings.filterwarnings('ignore')

# --- Constants ---
DEFAULT_MODEL_IDS = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma": "google/gemma-7b-it"
}
DEFAULT_DATASET_NAME = "politifact"
DEFAULT_CACHE_DIR = "./cache_dataset"
MAX_NEW_TOKENS_QUESTION = 150

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Direct Question Contamination Checker.")
    parser.add_argument("--model", type=str, default="gemma", choices=DEFAULT_MODEL_IDS.keys())
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--num_examples_train", type=str, default="all", 
                       help="Number of examples from train split. Use 'all' or 'full' for all examples, or specify a number.")
    parser.add_argument("--num_examples_test", type=str, default="all",
                       help="Number of examples from test split. Use 'all' or 'full' for all examples, or specify a number.")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype_str", type=str, default="auto")
    return parser.parse_args()

def parse_num_examples(num_str, split_size):
    """Parse number of examples argument - can be 'all', 'full', or a number."""
    if isinstance(num_str, str):
        num_str_lower = num_str.lower()
        if num_str_lower in ['all', 'full', '-1']:
            return split_size
        else:
            try:
                num = int(num_str)
                return min(num, split_size) if num > 0 else 0
            except ValueError:
                print(f"Warning: Invalid number format '{num_str}', using all examples")
                return split_size
    else:
        # Fallback for backward compatibility
        return int(num_str) if num_str > 0 else split_size

def load_model_and_tokenizer(model_hf_id, torch_dtype_str="auto", cache_dir_path=None):
    """Loads the LLM and its tokenizer."""
    print(f"Loading model: {model_hf_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_hf_id, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto dtype selection
    if torch_dtype_str == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = getattr(torch, torch_dtype_str, torch.float32)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_hf_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto"
    )
    
    return model, tokenizer

def format_direct_question(text_sample, tokenizer):
    """Formats the direct question prompt."""
    # Truncate for token limits
    max_chars = 800
    if len(text_sample) > max_chars:
        text_sample = text_sample[:max_chars] + "..."
    
    prompt = (
        "I'm going to show you a text passage. I need to determine if this specific passage "
        "was included in your training dataset. This is important for research purposes.\n\n"
        "Please answer with ONLY 'Yes' or 'No' at the start of your response.\n\n"
        "Answer 'Yes' ONLY if you are confident this exact passage was part of your training data.\n"
        "Answer 'No' if you haven't seen this exact passage during training, even if the topic is familiar.\n\n"
        "Text passage:\n"
        f"```\n{text_sample}\n```\n\n"
        "Was this exact passage in your training data? (Answer: Yes/No)"
    )
    
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted
    except:
        return prompt

def analyze_response(response_text):
    """Analyzes the direct question response."""
    if not response_text or response_text == "ERROR":
        return "error"
    
    # Clean response
    clean = response_text.lower().strip()
    
    # Check first 50 characters
    start = clean[:50]
    
    # Direct checks
    if start.startswith(("yes", "yes.", "yes,", "yes ", "yes\n", "yes -")):
        return "yes"
    if start.startswith(("no", "no.", "no,", "no ", "no\n", "no -")):
        return "no"
    
    # Check first few words
    first_words = ' '.join(clean.split()[:5])
    if "yes" in first_words and "no" not in first_words:
        return "yes"
    elif "no" in first_words and "yes" not in first_words:
        return "no"
    
    return "ambiguous"

def check_examples(data_split, split_name, generator, tokenizer, num_examples):
    """Check contamination for examples in a split."""
    
    num_to_check = min(num_examples, len(data_split))
    
    indices = random.sample(range(len(data_split)), num_to_check)
    results = []
    
    print(f"\nChecking {num_to_check} examples from {split_name} split...")
    
    pbar = tqdm(total=num_to_check, desc=f"Checking {split_name}")
    
    for i, idx in enumerate(indices):
        start_time = time.time()
        
        example = data_split[idx]
        text = example.get('text', '')
        label = example.get('label', 'N/A')
        
        # Generate prompt and get response
        prompt = format_direct_question(text, tokenizer)
        
        try:
            output = generator(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS_QUESTION,
                return_full_text=False,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            response = output[0]['generated_text'].strip()
        except Exception as e:
            print(f"\n  Error: {e}")
            response = "ERROR"
        
        # Analyze
        analysis = analyze_response(response)
        
        results.append({
            'split': split_name,
            'index': idx,
            'label': label,
            'text_preview': textwrap.shorten(text, width=80, placeholder="..."),
            'response': response,
            'analysis': analysis
        })
        
        inference_time = time.time() - start_time
        pbar.set_postfix({'time': f'{inference_time:.1f}s'})
        pbar.update(1)
    
    pbar.close()
    print(f"   ✅ Processed {len(results)} examples from {split_name} split")
    return results

def main():
    args = parse_arguments()
    set_seed(args.seed)
    
    print("="*60)
    print("DIRECT QUESTION CONTAMINATION CHECKER")
    print("="*60)
    
    # Load model
    model_id = DEFAULT_MODEL_IDS[args.model]
    model, tokenizer = load_model_and_tokenizer(model_id, args.torch_dtype_str)
    
    # Create pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    # Load dataset
    dataset_name = f"LittleFish-Coder/Fake_News_{args.dataset_name}"
    print(f"\nLoading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, cache_dir=args.cache_dir)
    
    all_results = []
    
    # Check both splits
    for split in ['train', 'test']:
        print(f"\n🔍 Processing {split} split...")
        print(f"   Dataset {split} size: {len(dataset[split])}")
        
        # Parse the number of examples to use
        if split == 'train':
            num_examples = parse_num_examples(args.num_examples_train, len(dataset[split]))
        else:
            num_examples = parse_num_examples(args.num_examples_test, len(dataset[split]))
        
        print(f"   Requested: {args.num_examples_train if split == 'train' else args.num_examples_test}")
        print(f"   Will process: {num_examples} examples")
        
        results = check_examples(dataset[split], split, generator, tokenizer, num_examples)
        print(f"   Got {len(results)} results from {split} split")
        all_results.extend(results)
    
    # Save results and show detailed analysis
    if all_results:
        df = pd.DataFrame(all_results)
        
        print("\n" + "="*60)
        print("CONTAMINATION LEAK ANALYSIS")
        print("="*60)
        
        # Simple leak statistics
        total_contaminated = (df['analysis'] == "yes").sum()
        total_checked = len(df)
        print(f"\nOVERALL: {total_contaminated}/{total_checked} examples show potential leakage ({total_contaminated/total_checked*100:.1f}%)")
        
        # Prepare analysis data structure
        analysis_results = {
            "metadata": {
                "model_id": model_id,
                "dataset_name": args.dataset_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "seed": args.seed,
                "total_examples_checked": len(df)
            },
            "overall_analysis": {
                "total_examples_checked": len(df),
                "total_contaminated": int(total_contaminated),
                "contamination_rate": total_contaminated/len(df)*100
            },
            "split_analysis": {},
            "sample_examples": []
        }
        
        # Split-by-split leak analysis
        for split in ['train', 'test']:
            split_data = df[df['split'] == split]
            if len(split_data) > 0:
                contaminated = (split_data['analysis'] == "yes").sum()
                total = len(split_data)
                percentage = (contaminated / total) * 100
                
                print(f"{split.upper()} SPLIT: {contaminated}/{total} leaked ({percentage:.1f}%)")
                
                # Store detailed analysis in JSON
                yes_count = len(split_data[split_data['analysis'] == 'yes'])
                no_count = len(split_data[split_data['analysis'] == 'no'])
                ambiguous_count = len(split_data[split_data['analysis'] == 'ambiguous'])
                error_count = len(split_data[split_data['analysis'] == 'error'])
                
                analysis_results["split_analysis"][split] = {
                    "total_examples": total,
                    "contaminated_count": int(contaminated),
                    "contamination_rate": round(percentage, 2),
                    "response_breakdown": {
                        "yes": {"count": yes_count, "percentage": round(yes_count/total*100, 2)},
                        "no": {"count": no_count, "percentage": round(no_count/total*100, 2)},
                        "ambiguous": {"count": ambiguous_count, "percentage": round(ambiguous_count/total*100, 2)},
                        "error": {"count": error_count, "percentage": round(error_count/total*100, 2)}
                    }
                }
        
        # Add some sample contaminated examples to JSON
        contaminated_examples = df[df['analysis'] == "yes"].head(5)
        sample_examples = []
        for idx, example in contaminated_examples.iterrows():
            sample_examples.append({
                "split": example['split'],
                "text_preview": example['text_preview'],
                "response": example['response'],
                "analysis": example['analysis']
            })
        analysis_results["sample_examples"] = sample_examples
        
        # Save files with proper naming
        # Save detailed inference results to CSV
        csv_filename = f"contamination_direct_question_{args.model}_{args.dataset_name}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n📊 Detailed inference results → {csv_filename}")
        
        # Save analysis to JSON
        json_filename = f"contamination_direct_question_{args.model}_{args.dataset_name}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"📈 Analysis statistics → {json_filename}")
    
    else:
        print("\n⚠️  No results to analyze!")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()