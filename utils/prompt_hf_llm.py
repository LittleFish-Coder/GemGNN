import os
import gc
import json
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    set_seed as set_transformers_seed,
    logging as hf_logging
)
from datasets import load_dataset, DatasetDict, Dataset
from argparse import ArgumentParser, Namespace
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from utils.sample_k_shot import sample_k_shot
from tqdm import tqdm
import random

# Constants
DEFAULT_MAX_NEW_TOKENS = 10
SEED = 42
DEFAULT_BATCH_SIZE = 4 # Default batch size (adjust based on VRAM)
DEFAULT_MAX_EXAMPLE_LENGTH = 500 # Default max characters for k-shot examples
DEFAULT_CACHE_DIR = "dataset" # Default cache directory for datasets and models

# --- Default Model Mapping ---
DEFAULT_MODEL_IDS = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma": "google/gemma-7b-it" # Use the Instruction Tuned version
}

def set_seed(seed: int = SEED) -> None:
    """Sets a fixed internal seed for reproducibility."""
    np.random.seed(seed); 
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    set_transformers_seed(seed)
    print(f"Internal random seed set to {seed}")

def load_and_prepare_data(dataset_name: str, k_shot: int, seed: int, cache_dir: str = DEFAULT_CACHE_DIR) -> Tuple[DatasetDict, List[Dict[str, Any]], Optional[Dict]]:
    """Loads dataset, samples k-shot examples."""
    print(f"Loading dataset 'LittleFish-Coder/Fake_News_{dataset_name}'...")
    dataset_full_name = f"LittleFish-Coder/Fake_News_{dataset_name}"
    dataset = load_dataset(dataset_full_name, cache_dir=cache_dir)

    if "train" not in dataset or "test" not in dataset: 
        raise ValueError("Dataset needs 'train'/'test' splits.")

    k_shot_examples = []
    k_shot_indices_info = None
    train_data = dataset["train"]
    unique_labels = sorted(list(set(train_data['label'])))
    num_classes = len(unique_labels)    
    print(f"Detected {num_classes} classes: {unique_labels}")
    print(f"Sampling {k_shot}-shot examples per class using seed {seed}...")
    
    selected_indices, _ = sample_k_shot(train_data, k_shot, seed=seed)
    print(f"Selected {len(selected_indices)} indices: {selected_indices}")

    label_distribution = {}
    for idx in selected_indices:
        try:
            safe_idx = int(idx)
            example_item = train_data[safe_idx]
            k_shot_examples.append({"text": example_item["text"], "label": example_item["label"]})
            label = example_item["label"]
            label_distribution[label] = label_distribution.get(label, 0) + 1
        except Exception as e: 
            print(f"Warning: Error processing sampled index {idx}: {e}")

    print(f"Selected {len(k_shot_examples)} examples for k-shot prompts.")
    print(f"Label distribution: {label_distribution}")
    k_shot_indices_info = {
        "indices": [int(i) for i in selected_indices], 
        "k_shot": k_shot, 
        "seed": seed,
        "dataset_name": dataset_name, 
        "label_distribution": {int(k): int(v) for k, v in label_distribution.items()}
    }

    print(f"Test set size: {len(dataset['test'])}")
    return dataset, k_shot_examples, k_shot_indices_info

def setup_hf_model_and_tokenizer(model_hf_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Loads HF model and tokenizer using device_map='auto'."""
    print(f"Setting up Hugging Face model and tokenizer: {model_hf_id}")
    effective_device_type = "cpu"; torch_dtype = torch.float32
    if torch.cuda.is_available():
        effective_device_type = "cuda"; print("CUDA (GPU) detected.")
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using torch dtype: {torch_dtype}")
    else: print(f"CUDA not available. Using CPU. Dtype: {torch_dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_hf_id, trust_remote_code=True)
        tokenizer.padding_side = "left"
        resize_needed = False
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token; print(f"Set pad_token to eos_token: '{tokenizer.pad_token}'")
            else: tokenizer.add_special_tokens({'pad_token': '[PAD]'}); print("Added '[PAD]' as pad_token."); resize_needed = True
    except Exception as e: print(f"Error loading tokenizer '{model_hf_id}': {e}"); raise

    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype, "device_map": "auto"}
    print(f"Loading model '{model_hf_id}' with device_map='auto' and dtype={torch_dtype}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_hf_id, **model_kwargs)
        if resize_needed: model.resize_token_embeddings(len(tokenizer)); print("Resized model embeddings.")
    except Exception as e: print(f"Error loading model '{model_hf_id}': {e}"); raise

    print("Model and tokenizer loaded successfully.")
    effective_device_str = "unknown"
    try:
         if hasattr(model, 'hf_device_map'): print(f"Model device map: {model.hf_device_map}"); effective_device_str = str(list(model.hf_device_map.values())[0])
         elif hasattr(model, 'device'): effective_device_str = str(model.device)
         print(f"Effective model device(s): {effective_device_str}")
    except Exception: print("Could not determine effective model device.")
    return model, tokenizer, effective_device_type

def build_hf_prompt(tokenizer: AutoTokenizer, model_hf_id: str, k_shot_examples: List[Dict], test_text: str, max_example_len: Optional[int]) -> str:
    """
    Builds prompt string, using model-specific formatting logic.
    Re-introduces example text truncation based on max_example_len (character count).
    """
    system_instruction = (
        "You are an AI assistant classifying news articles. "
        "Determine if the provided article is REAL or FAKE. "
        "Respond with only the single word: REAL or FAKE."
    )

    # --- Prepare Examples ---
    examples_str = ""
    if k_shot_examples:
        examples_str += "Here are some examples of news classification:\n\n"
        for i, example in enumerate(k_shot_examples):
            label_str = "REAL" if example["label"] == 0 else "FAKE"
            example_text = example['text']
            if max_example_len is not None and max_example_len > 0 and len(example_text) > max_example_len:
                truncated_text = example_text[:max_example_len] + "..."
                # print(f"Truncated example {i+1} from {len(example_text)} to {max_example_len} chars.") # Debug
            else:
                truncated_text = example_text

            examples_str += f"Example {i+1}:\nArticle: {truncated_text}\nClassification: {label_str}\n\n"
        examples_str += "---\n\n"

    # --- Prepare Query ---
    query_str = f"Now, classify the following article:\n\nArticle: {test_text}\n\nClassification:"

    # --- Apply Model-Specific Formatting ---
    model_name_lower = model_hf_id.lower()

    # ** Gemma Specific Formatting **
    if 'gemma' in model_name_lower:
        # Gemma instruct format: <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
        # Put instruction and examples in the first user turn.
        prompt = f"<start_of_turn>user\n{system_instruction}\n\n{examples_str}{query_str}<end_of_turn>\n<start_of_turn>model\n"
        # The final '\n' after model prompts it to start generating immediately.
        return prompt

    # ** Llama 3 (and potentially others) using Chat Template **
    else:
        messages = [{"role": "system", "content": system_instruction}]
        user_content = f"{examples_str}{query_str}"
        messages.append({"role": "user", "content": user_content})
        try:
            # Use the tokenizer's built-in template (works well for Llama 3 Instruct)
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return formatted_prompt
        except Exception as e:
            print(f"Warning: Could not apply chat template for {model_hf_id}: {e}. Using basic format.")
            # Basic fallback if template fails
            return f"{system_instruction}\n\n{examples_str}{query_str}"


def parse_llm_response(response: str) -> Tuple[int, float]:
    """Parses LLM text for classification. Confidence is hardcoded (parsing confidence)."""
    clean_response = response.strip().upper()
    if clean_response.startswith("FAKE"): 
        return 1, 0.95
    if clean_response.startswith("REAL"): 
        return 0, 0.95
    if "FAKE" in clean_response: 
        return 1, 0.80
    if "REAL" in clean_response: 
        return 0, 0.80
    return 0, 0.50

def run_hf_evaluation(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_hf_id: str, test_dataset: Dataset, k_shot_examples: List[Dict], max_new_tokens: int, max_example_len: Optional[int], batch_size: int = 16 ) -> Dict[str, Any]:
    """Runs evaluation using HF pipeline."""
    labels = [item["label"] for item in test_dataset]
    predictions = []
    confidences = []
    generation_errors = 0

    try:
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, temperature=None, top_p=None, top_k=None)
        print(f"Text generation pipeline created on device: {generator.device}")
    except Exception as e: 
        print(f"Error creating pipeline: {e}")

    print(f"Starting HF evaluation on {len(test_dataset)} test samples with batch size {batch_size}...")

    prompts = [build_hf_prompt(tokenizer, model_hf_id, k_shot_examples, item["text"], max_example_len) for item in test_dataset]

    processed_count = 0
    hf_logging.set_verbosity_error()
    for i, output in enumerate(tqdm(generator(prompts, batch_size=batch_size, return_full_text=False, clean_up_tokenization_spaces=True), total=len(prompts), desc="Evaluating HF Model")):
        processed_count += 1
        if isinstance(output, list) and output: 
            generated_text = output[0]['generated_text']
        elif isinstance(output, dict) and 'generated_text' in output: 
            generated_text = output['generated_text']
        else:
            print(f"Warning: Unexpected output format at index {i}: {output}")
            pred, conf = 0, 0.0
            generation_errors += 1
            continue

        pred, conf = parse_llm_response(generated_text)
        predictions.append(pred)
        confidences.append(conf)
    hf_logging.set_verbosity_warning()

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(labels, predictions)
    metrics = {
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "confusion_matrix": conf_matrix.tolist(),
        "num_samples": len(labels),
        "num_generation_errors": generation_errors,
    }
    print("HF Evaluation finished.")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    predictions_data = {"labels": labels, "predictions": predictions, "confidences": confidences}
    return metrics, predictions_data


def main_hf(args: Namespace):
    """Main execution flow for Hugging Face models."""
    set_seed(SEED)

    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
        gc.collect()

    start_time = time.time()

    # --- 1. Determine Model ID ---
    if args.model_hf_id:
        actual_model_hf_id = args.model_hf_id
    elif args.model_type in DEFAULT_MODEL_IDS:
        actual_model_hf_id = DEFAULT_MODEL_IDS[args.model_type]
    else:
        print(f"Error: Invalid model type '{args.model_type}'. Cannot determine model ID.")
        return # Exit early if model ID is invalid
    
    model_type_for_dir = args.model_type
    print(f"Using Hugging Face Model ID: {actual_model_hf_id}")

    # --- 2. Setup Output Directory ---
    shot_str = f"{args.k_shot}-shot"
    output_dir = os.path.join(args.results_base_dir, model_type_for_dir, args.dataset_name, shot_str)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # --- 3. Load Data ---
    dataset = None
    k_shot_examples = []
    k_shot_indices_info = None
    dataset, k_shot_examples, k_shot_indices_info = load_and_prepare_data(args.dataset_name, args.k_shot, SEED, args.cache_dir)

    if k_shot_indices_info:
        indices_path = os.path.join(output_dir, "indices.json")
        with open(indices_path, "w") as f:
            json.dump(k_shot_indices_info, f, indent=2)
        print(f"K-shot indices saved to {indices_path}")

    # --- 4. Setup Model and Tokenizer ---
    model = None
    tokenizer = None
    effective_device_type = "cpu"
    model, tokenizer, effective_device_type = setup_hf_model_and_tokenizer(actual_model_hf_id)
    print(f"Model setup complete. Effective device type: {effective_device_type}")

    # --- 5. Run Evaluation ---
    metrics = None
    predictions_data = None
    metrics, predictions_data = run_hf_evaluation(
        model=model,
        tokenizer=tokenizer,
        model_hf_id=actual_model_hf_id,
        test_dataset=dataset["test"],
        k_shot_examples=k_shot_examples,
        max_new_tokens=args.max_new_tokens,
        max_example_len=args.max_example_length,
        batch_size=args.batch_size
    )


    # --- 6. Save Results ---
    run_info = {
        "model_type": model_type_for_dir,
        "model_hf_id": actual_model_hf_id,
        "dataset_name": args.dataset_name,
        "k_shot": args.k_shot,
        "internal_seed": SEED,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "max_example_length": args.max_example_length,
        "effective_device_type": effective_device_type,
        "execution_time_seconds": time.time() - start_time
    }
    metrics["run_info"] = run_info

    metrics_path = os.path.join(output_dir, "metrics.json")
    predictions_path = os.path.join(output_dir, "predictions.json")

    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        save_data = {"labels": [], "predictions": [], "confidences": []}
        if predictions_data:
            save_data = {
                "labels": [int(l) if l is not None else -1 for l in predictions_data.get("labels", [])],
                "predictions": [int(p) if p is not None else -1 for p in predictions_data.get("predictions", [])],
                "confidences": [float(c) if c is not None else -1.0 for c in predictions_data.get("confidences", [])]
            }
        with open(predictions_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Predictions saved to {predictions_path}")
    except IOError as e:
        print(f"Error saving results: {e}")

    # --- 7. Cleanup ---
    print("Cleaning up resources...")
    del model, tokenizer, dataset, k_shot_examples, predictions_data, metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache.")

    end_time = time.time()
    print(f"Script finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Few-Shot Fake News Classification using HF LLMs (ICL) - Simplified v2")
    parser.add_argument("--model_type", type=str, default="llama", choices=['llama', 'gemma'], help="Simplified model type.")
    parser.add_argument("--dataset_name", type=str, default="politifact", choices=["politifact", "gossipcop"], help="Dataset name.")
    parser.add_argument("--k_shot", type=int, default=3, choices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], help="Samples per class for ICL (3-16).")
    parser.add_argument("--model_hf_id", type=str, default=None, help="Optional: Override default HF model ID.")
    parser.add_argument("--results_base_dir", type=str, default="results", help="Base directory for results.")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max tokens for model's reply.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Evaluation batch size (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--max_example_length", type=int, default=DEFAULT_MAX_EXAMPLE_LENGTH, help=f"Max characters per k-shot example text (default: {DEFAULT_MAX_EXAMPLE_LENGTH}). Set to -1 for no limit (RISKY).")
    parser.add_argument("--cache_dir", type=str, default="dataset", help="Cache directory for datasets and models (default: dataset)")
    cli_args = parser.parse_args()

    print("\n" + "="*60 + f"\n Starting HF LLM Evaluation ({time.strftime('%Y-%m-%d %H:%M:%S')})\n" + "="*60)
    print("Arguments:")
    for arg, value in vars(cli_args).items(): 
        print(f"  {arg}: {value}")
    actual_model_id = cli_args.model_hf_id or DEFAULT_MODEL_IDS.get(cli_args.model_type)
    if not actual_model_id: 
        print(f"\nError: Cannot determine model ID for type '{cli_args.model_type}'.")
    else: 
        print(f"  Resolved Model ID: {actual_model_id}")
    print("="*60 + "\n")

    if actual_model_id: 
        main_hf(cli_args)
    else: 
        print("Aborting due to missing model ID.")