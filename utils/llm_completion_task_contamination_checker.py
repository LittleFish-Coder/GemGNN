# -*- coding: utf-8 -*-
"""
Completion Task Contamination Checker
This version uses text completion to check for training data contamination.
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
from difflib import SequenceMatcher
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
warnings.filterwarnings('ignore')

# --- Constants ---
DEFAULT_MODEL_IDS = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma": "google/gemma-7b-it"
}
DEFAULT_DATASET_NAME = "politifact"
DEFAULT_CACHE_DIR = "./cache_dataset"
MAX_NEW_TOKENS_COMPLETION = 200
PREFIX_LENGTH_WORDS = 50
MIN_WORDS_FOR_MEANINGFUL_CHECK = 20  # Minimum words needed for meaningful completion check

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Completion Task Contamination Checker.")
    parser.add_argument("--model", type=str, default="gemma", choices=DEFAULT_MODEL_IDS.keys())
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--num_examples_train", type=str, default="all", 
                       help="Number of examples from train split. Use 'all' or 'full' for all examples, or specify a number.")
    parser.add_argument("--num_examples_test", type=str, default="all",
                       help="Number of examples from test split. Use 'all' or 'full' for all examples, or specify a number.")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype_str", type=str, default="auto")
    parser.add_argument("--prefix_words", type=int, default=PREFIX_LENGTH_WORDS,
                       help="Number of words to use as prefix for completion")
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

def strip_llm_preamble(generated_text, expected_continuation):
    preambles = [
        "Sure, here is the completed text:",
        "Here is the continuation:",
        "Certainly,",
        "Sure,",
        "Here is the text:",
        "The continuation is:",
        "Here you go:",
        "Here is the rest:",
    ]
    for pre in preambles:
        if generated_text.lower().startswith(pre.lower()):
            return generated_text[len(pre):].lstrip()
    # 嘗試自動對齊 expected_continuation 開頭
    for i in range(len(generated_text)):
        if generated_text[i:].lower().startswith(expected_continuation[:10].lower()):
            return generated_text[i:]
    return generated_text

def sliding_window_max_similarity(gen, exp, window=30):
    gen_words = gen.split()
    exp_words = exp.split()
    max_sim = 0
    for i in range(len(gen_words)):
        window_gen = " ".join(gen_words[i:i+window])
        sm = SequenceMatcher(None, window_gen.lower(), exp.lower())
        sim = sm.ratio()
        if sim > max_sim:
            max_sim = sim
    return max_sim

def calc_bleu(gen, exp):
    if sentence_bleu is None:
        return 0.0
    smoothie = SmoothingFunction().method4 if SmoothingFunction else None
    try:
        return sentence_bleu([exp.split()], gen.split(), smoothing_function=smoothie)
    except Exception:
        return 0.0

def calc_rouge_l(gen, exp):
    if Rouge is None:
        return 0.0
    try:
        rouge = Rouge()
        scores = rouge.get_scores(gen, exp)
        return scores[0]['rouge-l']['f']
    except Exception:
        return 0.0

def calc_jaccard(gen, exp):
    set1 = set(gen.lower().split())
    set2 = set(exp.lower().split())
    return len(set1 & set2) / max(len(set1 | set2), 1)

def format_completion_prompt(text_sample, tokenizer, prefix_words):
    words = text_sample.split()
    if len(words) <= prefix_words:
        prefix = text_sample
        expected_continuation = ""
    else:
        prefix = " ".join(words[:prefix_words])
        expected_continuation = " ".join(words[prefix_words:])
    prompt = (
        "Continue the following text directly, without any introduction, explanation, or summary. "
        "Write only the next part of the text, as if you are the original author, and do not repeat the given words.\n\n"
        f"{prefix}"
    )
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted, prefix, expected_continuation
    except:
        return prompt, prefix, expected_continuation

def calculate_similarity_metrics(generated_text, expected_continuation):
    if not generated_text or not expected_continuation:
        return {
            'max_sliding_window_similarity': 0.0,
            'bleu': 0.0,
            'rouge_l': 0.0,
            'jaccard': 0.0,
            'matching_words_count': 0,
            'is_contaminated': False
        }
    gen_clean = strip_llm_preamble(generated_text.strip(), expected_continuation.strip())
    exp_clean = expected_continuation.strip()
    max_sim = sliding_window_max_similarity(gen_clean, exp_clean)
    bleu = calc_bleu(gen_clean, exp_clean)
    rouge_l = calc_rouge_l(gen_clean, exp_clean)
    jaccard = calc_jaccard(gen_clean, exp_clean)
    # 連續 match
    gen_words = gen_clean.lower().split()
    exp_words = exp_clean.lower().split()
    matching_words = 0
    for i in range(min(len(gen_words), len(exp_words))):
        if gen_words[i] == exp_words[i]:
            matching_words += 1
        else:
            break
    is_contaminated = (
        max_sim > 0.8 or
        bleu > 0.5 or
        rouge_l > 0.5 or
        matching_words >= 10
    )
    return {
        'max_sliding_window_similarity': round(max_sim, 3),
        'bleu': round(bleu, 3),
        'rouge_l': round(rouge_l, 3),
        'jaccard': round(jaccard, 3),
        'matching_words_count': matching_words,
        'is_contaminated': is_contaminated
    }

def check_examples(data_split, split_name, generator, tokenizer, num_examples, prefix_words):
    """Check contamination for examples in a split."""
    
    num_to_check = min(num_examples, len(data_split))
    
    # Filter out examples that are too short
    valid_indices = []
    for i in range(len(data_split)):
        text = data_split[i].get('text', '')
        if len(text.split()) >= prefix_words + MIN_WORDS_FOR_MEANINGFUL_CHECK:
            valid_indices.append(i)
    
    if len(valid_indices) < num_to_check:
        print(f"   ⚠️  Only {len(valid_indices)} examples have enough words for meaningful completion check")
        num_to_check = len(valid_indices)
    
    if num_to_check == 0:
        print(f"   ❌ No examples with sufficient length in {split_name} split")
        return []
    
    indices = random.sample(valid_indices, num_to_check)
    results = []
    
    print(f"\nChecking {num_to_check} examples from {split_name} split...")
    
    pbar = tqdm(total=num_to_check, desc=f"Checking {split_name}")
    
    for i, idx in enumerate(indices):
        start_time = time.time()
        
        example = data_split[idx]
        text = example.get('text', '')
        label = example.get('label', 'N/A')
        
        # Generate prompt and get response
        prompt, prefix, expected = format_completion_prompt(text, tokenizer, prefix_words)
        
        try:
            output = generator(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS_COMPLETION,
                return_full_text=False,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            completion = output[0]['generated_text'].strip()
        except Exception as e:
            print(f"\n  Error: {e}")
            completion = "ERROR"
        
        # Calculate similarity metrics
        metrics = calculate_similarity_metrics(completion, expected)
        
        results.append({
            'split': split_name,
            'index': idx,
            'label': label,
            'prefix_preview': textwrap.shorten(prefix, width=80, placeholder="..."),
            'expected_preview': textwrap.shorten(expected[:100], width=80, placeholder="..."),
            'completion_preview': textwrap.shorten(completion[:100], width=80, placeholder="..."),
            'max_sliding_window_similarity': metrics['max_sliding_window_similarity'],
            'bleu': metrics['bleu'],
            'rouge_l': metrics['rouge_l'],
            'jaccard': metrics['jaccard'],
            'matching_words_count': metrics['matching_words_count'],
            'is_contaminated': metrics['is_contaminated']
        })
        
        inference_time = time.time() - start_time
        pbar.set_postfix({
            'time': f'{inference_time:.1f}s',
            'match': '✓' if metrics['is_contaminated'] else '✗'
        })
        pbar.update(1)
    
    pbar.close()
    print(f"   ✅ Processed {len(results)} examples from {split_name} split")
    return results

def main():
    args = parse_arguments()
    set_seed(args.seed)
    
    print("="*60)
    print("COMPLETION TASK CONTAMINATION CHECKER")
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
        print(f"   Prefix length: {args.prefix_words} words")
        
        results = check_examples(dataset[split], split, generator, tokenizer, num_examples, args.prefix_words)
        print(f"   Got {len(results)} results from {split} split")
        all_results.extend(results)
    
    # Save results and show detailed analysis
    if all_results:
        df = pd.DataFrame(all_results)
        
        print("\n" + "="*60)
        print("CONTAMINATION LEAK ANALYSIS")
        print("="*60)
        
        # Simple leak statistics
        total_contaminated = df['is_contaminated'].sum()
        total_checked = len(df)
        print(f"\nOVERALL: {total_contaminated}/{total_checked} examples show potential leakage ({(total_contaminated/total_checked*100) if total_checked > 0 else 0:.1f}%)")
        
        # Prepare analysis data structure
        analysis_results = {
            "metadata": {
                "model_id": model_id,
                "dataset_name": args.dataset_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "seed": args.seed,
                "prefix_words": args.prefix_words,
                "total_examples_checked": len(df)
            },
            "overall_analysis": {
                "total_examples_checked": len(df),
                "total_contaminated": int(total_contaminated),
                "contamination_rate": (total_contaminated/len(df)*100) if len(df) > 0 else 0
            },
            "split_analysis": {},
            "sample_examples": []
        }
        
        # Split-by-split leak analysis
        for split in ['train', 'test']:
            split_data = df[df['split'] == split]
            if len(split_data) > 0:
                contaminated = split_data['is_contaminated'].sum()
                total = len(split_data)
                percentage = (contaminated / total) * 100
                
                print(f"{split.upper()} SPLIT: {contaminated}/{total} leaked ({percentage:.1f}%)")
                
                # Store detailed analysis in JSON
                analysis_results["split_analysis"][split] = {
                    "total_examples": total,
                    "contaminated_count": int(contaminated),
                    "contamination_rate": round(percentage, 2)
                }
        
        # Add some sample contaminated examples to JSON
        contaminated_examples = df[df['is_contaminated']].head(5)
        sample_examples = []
        for idx, example in contaminated_examples.iterrows():
            sample_examples.append({
                "split": example['split'],
                "prefix_preview": example['prefix_preview'],
                "expected_preview": example['expected_preview'],
                "completion_preview": example['completion_preview'],
                "max_sliding_window_similarity": example['max_sliding_window_similarity'],
                "bleu": example['bleu'],
                "rouge_l": example['rouge_l'],
                "jaccard": example['jaccard'],
                "matching_words_count": example['matching_words_count']
            })
        analysis_results["sample_examples"] = sample_examples
        
        # Save files with proper naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed inference results to CSV
        csv_filename = f"completion_contamination_inference_{args.model}_{args.dataset_name}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n📊 Detailed inference results → {csv_filename}")
        
        # Save analysis to JSON
        json_filename = f"completion_contamination_analysis_{args.model}_{args.dataset_name}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"📈 Analysis statistics → {json_filename}")
    
    else:
        print("\n⚠️  No results to analyze!")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()