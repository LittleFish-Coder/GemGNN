# interaction_embeddings_dataset.py (FIXED AGAIN - Schema/Map Coordination)

import os
import argparse
import time
import logging
from typing import Dict, List, Any

logging.getLogger('google.auth.transport.requests').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
# Need to import Features explicitly
from datasets import load_dataset, DatasetDict, Features, Value, Sequence
from tqdm.auto import tqdm
import numpy as np

# --- Constants ---
PROJECT_ID = "netai-gnn"
LOCATION = "us-central1"
MODEL_NAME = "text-embedding-005"
OUTPUT_DIR_BASE = "processed_datasets"
API_BATCH_SIZE = 5

# --- Vertex AI Initialization ---
def initialize_vertexai():
    """Initializes Vertex AI connection."""
    print(f"Initializing Vertex AI for project '{PROJECT_ID}' in location '{LOCATION}'...")
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print("Vertex AI initialized successfully.")
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        print("Please ensure authentication and API enablement.")
        raise

# --- Embedding Function (with Batching) ---
def get_embeddings_batch(texts: List[str], model: TextEmbeddingModel) -> List[List[float]]:
    """Gets embeddings for a batch of texts, handles empty strings."""
    if not texts:
        return []
    # Ensure all items are strings before checking strip()
    safe_texts = [str(t) if t is not None else "" for t in texts]
    valid_texts = [text for text in safe_texts if text.strip()]
    if not valid_texts:
        return [[] for _ in safe_texts]

    original_indices = {i: text for i, text in enumerate(safe_texts) if text.strip()}
    inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in valid_texts]

    try:
        embeddings_response = model.get_embeddings(inputs)
        # Handle potential API issues where len(response) != len(inputs)
        if len(embeddings_response) != len(inputs):
             print(f"Warning: API returned {len(embeddings_response)} embeddings for {len(inputs)} inputs.")
             # Attempt graceful handling - create a dict based on available responses
             valid_embeddings_map = {}
             valid_indices_list = list(original_indices.keys())
             for i in range(min(len(embeddings_response), len(inputs))):
                 emb = embeddings_response[i]
                 original_idx = valid_indices_list[i]
                 valid_embeddings_map[original_idx] = emb.values
        else:
             valid_embeddings_map = {idx: emb.values for idx, emb in zip(original_indices.keys(), embeddings_response)}

        full_embeddings = [valid_embeddings_map.get(i, []) for i in range(len(safe_texts))]
        return full_embeddings

    except Exception as e:
        print(f"Warning: API call failed for a batch. Error: {e}")
        return [[] for _ in safe_texts]

# --- Dataset Processing Function ---
def add_interaction_embeddings_simplified(
    dataset: DatasetDict,
    embedding_model: TextEmbeddingModel,
    batch_size: int = API_BATCH_SIZE
) -> DatasetDict:
    """
    Adds Gemini embeddings and tones as separate parallel lists.

    Args:
        dataset: The input Hugging Face DatasetDict.
        embedding_model: Initialized Vertex AI embedding model.
        batch_size: How many interactions to send to the API at once.

    Returns:
        A new DatasetDict with 'user_interactions' removed and added
        'interaction_gemini_embeddings' and 'interaction_tones' columns.
    """

    # ***** FIX: Define the FINAL target features *****
    # 1. Get original features from one split (e.g., 'train')
    if "train" in dataset:
        original_features = dataset["train"].features
    elif "test" in dataset: # Fallback if no train split
         original_features = dataset["test"].features
    else: # Fallback if only one split with unknown name
         original_features = list(dataset.values())[0].features

    # 2. Create a *new* Features object
    final_feature_dict = {k: v for k, v in original_features.items()} # Copy original features

    # 3. Remove the column we are replacing
    if "user_interactions" in final_feature_dict:
        del final_feature_dict["user_interactions"]
    else:
        print("Warning: 'user_interactions' column not found in original features.")

    # 4. Add the new columns
    final_feature_dict["interaction_gemini_embeddings"] = Sequence(Sequence(Value(dtype='float32')))
    final_feature_dict["interaction_tones"] = Sequence(Value(dtype='string'))

    # 5. Create the final Features object
    final_features = Features(final_feature_dict)
    print("\nTarget final features defined:")
    print(final_features)


    # Function to be mapped - returns ONLY the new columns
    def process_batch(batch: Dict[str, List]) -> Dict[str, List]:
        """Processes a batch and returns ONLY the data for the new columns."""
        all_contents: List[str] = []
        all_original_tones: List[str] = []
        interactions_per_example: List[int] = []

        # Check if 'user_interactions' exists in the batch, handle if removed earlier
        user_interactions_list = batch.get("user_interactions", [[] for _ in range(len(batch[list(batch.keys())[0]]))]) # Get length from another column

        for interactions in user_interactions_list:
            count = 0
            if interactions: # Check if interactions is a list/iterable and not None
                for interaction in interactions:
                     # Defensive coding: ensure interaction is dict-like
                    if isinstance(interaction, dict):
                        content = interaction.get("content")
                        all_contents.append(str(content) if content is not None else "")
                        tone = interaction.get("tone")
                        all_original_tones.append(str(tone) if tone is not None else "")
                        count += 1
                    else:
                        # Handle cases where interaction might not be a dict (e.g., already processed?)
                        print(f"Warning: Unexpected item type in user_interactions: {type(interaction)}")
            interactions_per_example.append(count)


        all_embeddings_flat: List[List[float]] = []
        num_interactions_total = len(all_contents)
        for i in tqdm(range(0, num_interactions_total, batch_size), desc="API Calls", leave=False, disable=True):
            content_sub_batch = all_contents[i:i + batch_size]
            embeddings_sub_batch = get_embeddings_batch(content_sub_batch, embedding_model)
            # Retry logic (optional)
            if any(not emb for emb in embeddings_sub_batch) and content_sub_batch:
                 print(f"Retrying failed batch chunk (size {len(content_sub_batch)})...")
                 time.sleep(2)
                 embeddings_sub_batch = get_embeddings_batch(content_sub_batch, embedding_model)
            all_embeddings_flat.extend(embeddings_sub_batch)
            time.sleep(0.1)

        if len(all_embeddings_flat) != num_interactions_total:
            print(f"Error: Embedding count mismatch ({len(all_embeddings_flat)} vs {num_interactions_total}). Padding.")
            all_embeddings_flat.extend([[] for _ in range(num_interactions_total - len(all_embeddings_flat))])

        output_embeddings_list: List[List[List[float]]] = []
        output_tones_list: List[List[str]] = []
        current_idx = 0
        for num_interactions in interactions_per_example:
            example_embeddings: List[List[float]] = []
            example_tones: List[str] = []
            if num_interactions > 0:
                for _ in range(num_interactions):
                    embedding = all_embeddings_flat[current_idx]
                    tone = all_original_tones[current_idx]
                    if not embedding:
                         # Avoid overly verbose logging for common empty embeddings
                         # print(f"Warning: Storing empty embedding for interaction: '{all_contents[current_idx][:50]}...'")
                         pass
                    final_embedding = np.array(embedding, dtype=np.float32).tolist() if embedding else []
                    example_embeddings.append(final_embedding)
                    example_tones.append(tone)
                    current_idx += 1
            output_embeddings_list.append(example_embeddings)
            output_tones_list.append(example_tones)

        # ***** FIX: Return ONLY the new columns *****
        return {
            "interaction_gemini_embeddings": output_embeddings_list,
            "interaction_tones": output_tones_list
        }

    print("Processing dataset splits with coordinated features and removal...")
    map_batch_size = 8

    # ***** FIX: Apply map correctly *****
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=map_batch_size,
        features=final_features, # Specify the FINAL schema structure
        remove_columns=['user_interactions'] # Explicitly state which original columns are removed
    )
    print("Dataset processing complete.")
    return processed_dataset # type: ignore


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Gemini embeddings for user interactions (simplified schema) and save the result."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["politifact", "gossipcop"],
        help="Name of the dataset to process (politifact or gossipcop)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR_BASE,
        help=f"Base directory to save the processed dataset (default: {OUTPUT_DIR_BASE})."
    )
    args = parser.parse_args()

    print("\n--- Starting Interaction Embedding Generation (Simplified Schema) ---")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output Base Directory: {args.output_dir}")
    print(f"Embedding Model: {MODEL_NAME}")
    print("-" * 40)

    # 1. Initialize Vertex AI
    initialize_vertexai()

    # 2. Load Embedding Model
    print(f"Loading embedding model: {MODEL_NAME}...")
    try:
        embedding_model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        exit(1)

    # 3. Load Dataset
    hf_dataset_name = f"LittleFish-Coder/Fake_News_{args.dataset_name.capitalize()}"
    print(f"Loading dataset '{hf_dataset_name}' from Hugging Face...")
    try:
        # Load with trust_remote_code=True if dataset requires it (less common now)
        dataset = load_dataset(
            hf_dataset_name,
            cache_dir="dataset",
            # trust_remote_code=True # Uncomment if needed
        )
        print("Dataset loaded:")
        print(dataset)
    except Exception as e:
        print(f"Error loading dataset '{hf_dataset_name}': {e}")
        exit(1)

    # 4. Add Interaction Embeddings (Simplified function)
    print("\nStarting embedding generation for user interactions...")
    print("WARNING: This process involves calling the Vertex AI API and may incur costs.")
    print("Processing time depends on dataset size and API latency.")

    start_time = time.time()
    processed_dataset = add_interaction_embeddings_simplified(dataset, embedding_model)
    end_time = time.time()

    print(f"\nEmbedding generation took {end_time - start_time:.2f} seconds.")

    # 5. Save Processed Dataset
    output_path = os.path.join(args.output_dir, f"{args.dataset_name}_with_interaction_embeddings_simple")
    print(f"Saving processed dataset to: {output_path}")
    try:
        processed_dataset.save_to_disk(output_path)
        print("Dataset saved successfully.")
        print("\nProcessed dataset structure:")
        print(processed_dataset)

        # Check a sample from the 'test' split if it exists
        test_split_name = 'test'
        if test_split_name in processed_dataset and len(processed_dataset[test_split_name]) > 0:
             print(f"\nExample 'interaction_gemini_embeddings' (first 2, truncated) from first '{test_split_name}' row:")
             example_embs = processed_dataset[test_split_name][0]['interaction_gemini_embeddings']
             for emb in example_embs[:2]:
                 # Handle case where embedding might be empty list []
                 print(f"  Embedding (first 5 dims): {emb[:5] if emb else '[]'}")
             print(f"\nExample 'interaction_tones' (first 2) from first '{test_split_name}' row:")
             print(f"  Tones: {processed_dataset[test_split_name][0]['interaction_tones'][:2]}")

    except Exception as e:
        print(f"Error saving processed dataset: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n--- Interaction Embedding Generation Complete ---")