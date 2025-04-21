# generate_interaction_v2.py

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from datasets import load_dataset
import json
import time
import random
import os
import glob
from tqdm import tqdm
import logging
from datetime import datetime
import traceback

# Logger
log_filename = f'logs/interaction_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
os.makedirs("logs", exist_ok=True) # Ensure logs directory exists
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# --- Setup Vertex AI ---
try:
    PROJECT_ID = "netai-gnn"
    LOCATION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-2.0-flash") 
except Exception as e:
    logging.error(f"Vertex AI initialization failed: {e}")
    exit(1) # Exit if initialization fails

# --- Load dataset ---
try:
    dataset = load_dataset("LittleFish-Coder/Fake_News_PolitiFact", download_mode="reuse_cache_if_exists", cache_dir="dataset")
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    logging.info("Dataset loaded successfully")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    exit(1)

# --- Interaction type definitions ---
interaction_distribution = {
    "neutral": {"count": 8, "prompts": [
        "Please act as a reader who is confused about this news.",
        "Please act as a reader who wants to know more details about this news.",
        "Please act as a reader who has a neutral attitude toward this news.",
        "Please act as a curious reader and ask questions about this news."
    ]},
    "affirmative": {"count": 7, "prompts": [
        "Please act as a reader who actively participates in the discussion and shares personal opinions.",
        "Please act as a reader who agrees with this news.",
        "Please act as a reader who wants to share this news with friends.",
        "Please act as a reader who feels excited about the content of this news."
    ]},
    "skeptical": {"count": 5, "prompts": [
        "Please act as a reader who questions the authenticity of the news and provides evidence.",
        "Please act as a reader who raises doubts about this news.",
        "Please act as a reader who questions the source or credibility of this news.",
        "Please act as a reader who requests clarification or more information about this news."
    ]}
}

# --- Progress tracking file ---
PROGRESS_FILE = "checkpoints/progress.json"
CHECKPOINT_DIR = "checkpoints"

def load_progress():
    """Load progress, return a dictionary containing the last index of each split"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                logging.info(f"Loaded progress from {PROGRESS_FILE}: {progress}")
                return progress
        except json.JSONDecodeError:
            logging.warning(f"{PROGRESS_FILE} format error, will restart.")
            return {}
        except Exception as e:
            logging.error(f"Error loading progress file {PROGRESS_FILE}: {e}")
            return {}
    else:
        logging.info(f"Progress file {PROGRESS_FILE} not found, will start from scratch.")
        return {}

def save_progress(progress):
    """Save progress"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
        logging.debug(f"Progress saved to {PROGRESS_FILE}: {progress}")
    except Exception as e:
        logging.error(f"Error saving progress to {PROGRESS_FILE}: {e}")

def generate_user_interaction(news_text, prompt, max_retries=3, initial_delay=2):
    """
    Generate user interaction based on news text and prompt
    """
    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        try:
            # Construct full prompt
            full_prompt = f"""
            {prompt}

            News:
            {news_text}

            Please generate a reader's comment or question based on the above news, with a length of 50-100 words.
            Do not mention that you are an AI, just provide the comment content directly.
            Avoid using any special characters or formatting that might break JSON structure, like unescaped quotes within the text.
            """

            # Set generation parameters
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=150,
            )

            # Generate content
            response = model.generate_content(full_prompt, generation_config=generation_config)

            # Ensure response.text exists and is a string
            if hasattr(response, 'text') and isinstance(response.text, str):
                 # Preprocess response, remove characters that might cause JSON issues
                cleaned_response = response.text.strip()
                # Remove unnecessary quotes, but keep apostrophes within the text
                cleaned_response = cleaned_response.replace('"', "'").replace('「', "'").replace('」', "'")
                # Ensure return is a string
                return str(cleaned_response)
            else:
                 # If API response does not contain text or is not a string, log a warning and prepare to retry
                 logging.warning(f"API response format abnormal, valid text field not found. Response: {response}")
                 # Optionally raise an error here or directly enter retry

        except Exception as e:
            logging.warning(f"Error generating interaction (retry {retry_count+1}/{max_retries}): {e}")
            # Print more detailed error trace information, helpful for debugging API issues
            # logging.debug(traceback.format_exc())

        # If an error occurs or response format is abnormal, increase retry count and wait
        retry_count += 1
        logging.info(f"Waiting {delay} seconds before retrying...")
        time.sleep(delay)
        delay *= 2 # Exponential backoff

    # After all retries fail, return a default, guaranteed safe string
    logging.error(f"Failed to generate interaction, reached maximum retries ({max_retries}).")
    return "This news is interesting, but I'm not sure about its authenticity. I would like to see more related reports to confirm."


def process_news_sample(news_idx, news_item):
    """
    Process a single news sample to generate user interactions
    Each sample includes the original index as an ID
    """
    news_text = news_item['text']
    label = news_item['label']

    interactions = []
    interaction_id = 0

    for interaction_type, config in interaction_distribution.items():
        count = config["count"]
        prompts = config["prompts"]

        for _ in range(count):
            prompt = random.choice(prompts)
            interaction_content = generate_user_interaction(news_text, prompt)

            # Ensure interaction content is a string
            if not isinstance(interaction_content, str):
                 logging.warning(f"Sample {news_idx}, interaction {interaction_id} content is not a string, using default value.")
                 interaction_content = "Interaction generation failed." # Provide a safe default value

            interactions.append({
                "id": f"{news_idx}_{interaction_id}",
                "tone": interaction_type,
                "content": interaction_content # Ensure this adds a string
            })
            interaction_id += 1
            # Short pause to avoid calling the API too frequently
            time.sleep(random.uniform(0.5, 1.5))


    random.shuffle(interactions)

    # Log truncated text to avoid overly long logs
    log_text_preview = (news_text[:30] + '...') if len(news_text) > 30 else news_text
    log_interaction_preview = ""
    if interactions:
        log_interaction_preview = (interactions[0]['content'][:30] + '...') if len(interactions[0]['content']) > 30 else interactions[0]['content']

    logging.info(f"Processed news sample {news_idx}, generated {len(interactions)} interactions")
    logging.info(
        f"\nLabel: {label}\n"
        f"News Content: {log_text_preview}\n"
        f"First Interaction: {log_interaction_preview}\n"
    )

    return {
        "original_idx": news_idx,
        "text": news_text, # Save full text
        "label": label,
        "user_interactions": interactions
    }

def save_checkpoint(processed_samples, split_name, current_idx, failed_samples=None):
    """
    Save checkpoint and delete old checkpoint files.
    Also update progress file.

    Args:
        processed_samples: List of processed samples since last checkpoint (or from start).
        split_name: Name of the dataset split (train/test).
        current_idx: The index of the *last successfully processed* sample included in this save.
        failed_samples: List of failed samples with error details.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Save current batch of processed results ---
    # Note: Here we save *newly processed* samples since the last checkpoint, or if it's the first time, all processed samples
    # If your processed_samples contains all historical samples, this logic needs adjustment
    # Currently assuming processed_samples only contains the current batch results
    # If merging history, need to manage a cumulative list in process_dataset_split
    checkpoint_filename = f"{CHECKPOINT_DIR}/processed_{split_name}_{timestamp}.json"
    failed_filename = f"{CHECKPOINT_DIR}/failed_{split_name}_{timestamp}.json"
    save_successful = False

    try:
        # Try to save successfully processed samples
        with open(checkpoint_filename, 'w', encoding='utf-8') as f:
            json.dump(processed_samples, f, ensure_ascii=False, indent=2)
        logging.info(f"Checkpoint saved to {checkpoint_filename}, containing {len(processed_samples)} samples")
        save_successful = True # Mark save successful

        # If there are failed samples, save them too
        if failed_samples and len(failed_samples) > 0:
            try:
                with open(failed_filename, 'w', encoding='utf-8') as f:
                    json.dump(failed_samples, f, ensure_ascii=False, indent=2)
                logging.info(f"Failed samples saved to {failed_filename}, containing {len(failed_samples)} failure records")
            except Exception as e_fail:
                 logging.error(f"Error saving failed samples to {failed_filename}: {e_fail}")

    except TypeError as e_type:
        logging.error(f"TypeError saving checkpoint {checkpoint_filename} (data not serializable): {e_type}")
        logging.error(f"Error trace:\n{traceback.format_exc()}")
        # Try to find problematic data (this is a basic example, may need more complex logic)
        for i, sample in enumerate(processed_samples):
            try:
                json.dumps(sample, ensure_ascii=False)
            except TypeError:
                logging.error(f"Found non-serializable data in sample at index {i}: {sample}")
                # Optionally log more detailed sample info or try to remove problematic fields
        # Since save failed, do not update progress or delete old files
        return False # Return failure status

    except Exception as e:
        logging.error(f"Unknown error saving checkpoint {checkpoint_filename}: {e}")
        logging.error(f"Error trace:\n{traceback.format_exc()}")
        # Since save failed, do not update progress or delete old files
        return False # Return failure status

    # --- If save successful, update progress and delete old files ---
    if save_successful:
        # 1. Update progress file
        progress = load_progress()
        progress[split_name] = {"last_processed_idx": current_idx, "last_checkpoint_file": checkpoint_filename}
        save_progress(progress)

        # 2. Delete old checkpoint files
        # Find all processed and failed files for the same split
        all_processed_files = glob.glob(f"{CHECKPOINT_DIR}/processed_{split_name}_*.json")
        all_failed_files = glob.glob(f"{CHECKPOINT_DIR}/failed_{split_name}_*.json")

        # Delete all old files except the newly created one
        for old_file in all_processed_files:
            if old_file != checkpoint_filename:
                try:
                    os.remove(old_file)
                    logging.info(f"Deleted old checkpoint file: {old_file}")
                except OSError as e_os:
                    logging.warning(f"Failed to delete old checkpoint file {old_file}: {e_os}")

        for old_file in all_failed_files:
             # If a new failed file was successfully saved, do not delete it
            if not (failed_samples and len(failed_samples) > 0 and old_file == failed_filename):
                # Otherwise, if it is not the latest failed file for the current split (though we do not directly associate), it can be deleted
                # For simplicity, delete all failed files not from the current timestamp
                if timestamp not in old_file : # Avoid deleting the failed file possibly generated this time
                     try:
                         os.remove(old_file)
                         logging.info(f"Deleted old failed record file: {old_file}")
                     except OSError as e_os:
                         logging.warning(f"Failed to delete old failed record file {old_file}: {e_os}")
        return True # Return success status
    
    return False # If save was not successful


def load_latest_checkpoint_data(split_name):
    """
    Load the latest checkpoint data.
    Note: This only loads the content of the *latest* checkpoint file.
    If merging all historical checkpoints is needed, the logic will be more complex.
    """
    processed_samples = []
    failed_samples = []

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    processed_checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/processed_{split_name}_*.json"))
    failed_checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/failed_{split_name}_*.json"))

    if processed_checkpoints:
        latest_checkpoint_file = processed_checkpoints[-1]
        try:
            with open(latest_checkpoint_file, 'r', encoding='utf-8') as f:
                processed_samples = json.load(f)
            logging.info(f"Loaded {len(processed_samples)} processed samples from latest checkpoint {latest_checkpoint_file}.")
        except Exception as e:
            logging.error(f"Failed to load checkpoint {latest_checkpoint_file}: {e}. Will treat as empty list.")
            processed_samples = [] # Return empty on error

    return processed_samples, failed_samples


# --- Main processing function ---
def process_dataset_split(dataset_split, split_name, end_idx=None, checkpoint_freq=10, api_batch_delay=5):
    """
    Process dataset split and generate user interactions.

    Args:
        dataset_split: The dataset split to process.
        split_name: The name of the split ('train' or 'test').
        end_idx: The end index for processing (exclusive). If None, process to the end.
        checkpoint_freq: Save checkpoint every this many samples.
        api_batch_delay: Wait time in seconds after processing a batch (e.g., 5) of samples to prevent hitting API rate limits.
    """
    progress = load_progress()
    start_idx = progress.get(split_name, {}).get("last_processed_idx", -1) + 1
    # Validate start_idx
    if start_idx < 0:
        start_idx = 0
    logging.info(f"Read progress file, {split_name} will start processing from index {start_idx}.")


    # Determine the end index for processing
    total_samples = len(dataset_split)
    if end_idx is None or end_idx > total_samples:
        end_idx = total_samples
    logging.info(f"Will process indices from {start_idx} to {end_idx-1} (total {max(0, end_idx - start_idx)} samples)")

    # --- Decide how to handle historical data ---
    # Option 1: Each checkpoint only saves the current batch of data (safer, but merging requires extra steps)
    # Option 2: Each checkpoint saves all cumulative data (more convenient, but file gets larger, save risk increases)
    #
    # We choose a variant of method 1: accumulate in memory, save the entire cumulative list at checkpoint frequency, and update progress
    # This requires loading previous cumulative results when resuming

    all_processed_samples = []
    all_failed_samples = []

    # Try to load the final results from the last run (if progress file points to a valid checkpoint)
    last_checkpoint_file = progress.get(split_name, {}).get("last_checkpoint_file")
    if last_checkpoint_file and os.path.exists(last_checkpoint_file):
         logging.info(f"Attempting to restore cumulative data from last checkpoint file {last_checkpoint_file}...")
         try:
              with open(last_checkpoint_file, 'r', encoding='utf-8') as f:
                   all_processed_samples = json.load(f)
                   logging.info(f"Successfully loaded {len(all_processed_samples)} previously processed samples.")
                   # Validate start_idx matches loaded data
                   if all_processed_samples:
                       max_loaded_idx = max(s['original_idx'] for s in all_processed_samples)
                       if max_loaded_idx + 1 != start_idx:
                           logging.warning(f"Progress file index ({start_idx-1}) does not match loaded checkpoint max index ({max_loaded_idx}). Will trust progress file index {start_idx-1}.")
                           # Optionally clear all_processed_samples or keep and accept risk
                           # all_processed_samples = [] # Safer approach
         except Exception as e:
              logging.error(f"Failed to load historical checkpoint {last_checkpoint_file}: {e}. Will only process new samples.")
              all_processed_samples = []
              # If load fails, ensure start_idx is from 0 unless progress.json is valid
              if start_idx > 0 and not progress.get(split_name): # If progress.json is invalid
                   start_idx = 0


    # Use tqdm to display progress bar
    # Set initial=start_idx to start the progress bar from the correct position
    # Set total=end_idx to let the progress bar know the total target
    pbar = tqdm(range(start_idx, end_idx), initial=start_idx, total=total_samples, desc=f"Processing {split_name}")

    current_batch_processed = [] # For temporarily storing samples of the current checkpoint cycle
    current_batch_failed = []    # For temporarily storing failed samples of the current checkpoint cycle

    for idx in pbar:
        try:
            # Get raw data, ensure data exists
            if idx < 0 or idx >= len(dataset_split):
                 logging.error(f"Index {idx} out of range, skipping.")
                 continue
            news_item = dataset_split[idx]
            if not news_item:
                 logging.warning(f"Data at index {idx} is empty, skipping.")
                 continue

            # Process a single sample
            processed_sample = process_news_sample(idx, news_item)

            # Check if the returned result is valid
            if processed_sample and "original_idx" in processed_sample:
                 # Add to current batch and total list
                 current_batch_processed.append(processed_sample)
                 all_processed_samples.append(processed_sample) # Also add to total list
                 pbar.set_postfix({"Last save": f"idx {progress.get(split_name, {}).get('last_processed_idx', -1)}"}) # Update progress bar postfix
            else:
                 # If process_news_sample returns invalid result
                 raise ValueError("process_news_sample returned invalid result")


        except Exception as e:
            error_message = f"Error processing sample {idx}: {str(e)}"
            logging.error(error_message)
            logging.error(f"Detailed error trace:\n{traceback.format_exc()}") # Log full error trace

            # Record failed sample information
            failure_record = {
                "original_idx": idx,
                "error": str(e),
                "traceback": traceback.format_exc(), # Record traceback for more useful info
                "timestamp": datetime.now().isoformat()
            }
            current_batch_failed.append(failure_record)
            all_failed_samples.append(failure_record) # Also add to total list

        # Save checkpoint at checkpoint frequency or if this is the last sample
        # +1 because idx starts from 0, and we want to save after processing the checkpoint_freq-th sample
        if (idx + 1) % checkpoint_freq == 0 or idx == end_idx - 1:
            logging.info(f"Reached checkpoint frequency (processed index {idx}), preparing to save...")
            # Pass cumulative all processed_samples, and current index idx
            save_successful = save_checkpoint(all_processed_samples, split_name, idx, all_failed_samples)

            if save_successful:
                 # If save successful, can clear current batch lists (though we currently do not use them for recovery)
                 current_batch_processed = []
                 current_batch_failed = []
                 logging.info(f"Checkpoint successfully saved, progress updated to index {idx}.")
                 progress = load_progress() # Reload progress to update pbar
                 pbar.set_postfix({"Last save": f"idx {progress.get(split_name, {}).get('last_processed_idx', -1)}"})
            else:
                 logging.error(f"Checkpoint save failed at index {idx}. Please check logs. Script will continue, but progress not updated.")
                 # Optionally stop execution exit(1) or continue

        # API rate limit delay
        if (idx + 1) % 5 == 0: # Pause every 5 samples
            time.sleep(api_batch_delay)

    pbar.close() # End progress bar

    # --- After processing, save final results ---
    # Final file name
    final_filename_json = f"{split_name}_with_interactions_final.json"
    final_filename_hf = f"{split_name}_with_interactions_hf" # Hugging Face dataset directory name

    logging.info(f"All samples processed ({split_name}), preparing to save final results...")
    try:
        # 1. Save as JSON file (containing all successfully processed samples)
        with open(final_filename_json, 'w', encoding='utf-8') as f:
            json.dump(all_processed_samples, f, ensure_ascii=False, indent=2)
        logging.info(f"Final processed results saved to {final_filename_json} ({len(all_processed_samples)} samples)")

        # 2. Convert and save as Hugging Face Dataset format
        if all_processed_samples: # Ensure list is not empty
            from datasets import Dataset
            # Ensure all samples have the same structure, remove possibly inconsistent internal fields (if needed)
            # For example, if 'text' might be missing, handle it
            valid_samples_for_hf = [s for s in all_processed_samples if isinstance(s.get("text"), str)] # Simple filter
            if len(valid_samples_for_hf) != len(all_processed_samples):
                 logging.warning(f"Found {len(all_processed_samples) - len(valid_samples_for_hf)} samples with possibly inconsistent structure, not included in HF Dataset.")

            if valid_samples_for_hf:
                 final_dataset = Dataset.from_list(valid_samples_for_hf)
                 final_dataset.save_to_disk(final_filename_hf)
                 logging.info(f"Final results converted and saved as Hugging Face Dataset format in directory: {final_filename_hf}")
            else:
                 logging.warning("No valid samples to save as Hugging Face Dataset.")
        else:
            logging.warning("No successfully processed samples, cannot save final JSON or Hugging Face Dataset.")

    except Exception as e:
        logging.error(f"Error saving final results ({split_name}): {e}")
        logging.error(f"Detailed error trace:\n{traceback.format_exc()}")

    # Return statistics
    total_processed_count = len(all_processed_samples)
    total_failed_count = len(all_failed_samples)
    total_attempted = max(0, end_idx - start_idx)

    logging.info(f"{split_name} split processing complete.")
    logging.info(f"  Attempted samples: {total_attempted}")
    logging.info(f"  Successfully processed samples (cumulative): {total_processed_count}")
    logging.info(f"  Failed samples (cumulative): {total_failed_count}")

    return {
        "attempted": total_attempted,
        "processed_total": total_processed_count,
        "failed_total": total_failed_count,
    }

# --- Execute processing ---
if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Starting new processing run: {run_id}")

    # Set checkpoint frequency and API delay
    CHECKPOINT_FREQUENCY = 50  # Save every 50 samples, adjust as needed
    API_DELAY = 3             # Pause 3 seconds after every 5 samples

    # --- Process training set ---
    logging.info("===== Starting training set processing =====")
    try:
        train_stats = process_dataset_split(
            train_dataset,
            split_name="train",
            checkpoint_freq=CHECKPOINT_FREQUENCY,
            api_batch_delay=API_DELAY
            # end_idx=10 # Can set end_idx to test a small number of samples
        )
        logging.info(f"Training set processing complete: {train_stats}")
    except Exception as e:
        logging.error(f"Unexpected error during training set processing: {e}")
        logging.error(f"Detailed error trace:\n{traceback.format_exc()}")


    # --- Process test set ---
    logging.info("===== Starting test set processing =====")
    try:
        test_stats = process_dataset_split(
            test_dataset,
            split_name="test",
            checkpoint_freq=CHECKPOINT_FREQUENCY,
            api_batch_delay=API_DELAY
            # end_idx=10 # Can set end_idx to test a small number of samples
        )
        logging.info(f"Test set processing complete: {test_stats}")
    except Exception as e:
        logging.error(f"Unexpected error during test set processing: {e}")
        logging.error(f"Detailed error trace:\n{traceback.format_exc()}")


    logging.info(f"All processing complete (Run ID: {run_id})")