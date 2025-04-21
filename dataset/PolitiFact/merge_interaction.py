import json
from datasets import load_dataset, DatasetDict, Features, Sequence, Value, ClassLabel

print("--- Step 1: Load Original Hugging Face Dataset ---")
try:
    # Load the dataset that contains text, label, and embeddings
    original_dataset = load_dataset("LittleFish-Coder/Fake_News_PolitiFact",
                                    download_mode="reuse_cache_if_exists",
                                    cache_dir="dataset")
    print("Original dataset loaded successfully:")
    print(original_dataset)
    # Verify original features
    print("\nOriginal features:")
    print(original_dataset['train'].features)

except Exception as e:
    print(f"FATAL: Failed to load original dataset 'LittleFish-Coder/Fake_News_PolitiFact'. Error: {e}")
    exit()

print("\n--- Step 2: Load Generated Interaction Data from JSON ---")
# Define paths to the JSON files produced by generate_interaction.py
# *** ADJUST THESE PATHS if your filenames are different ***
train_json_path = "train_with_interactions_final.json"
test_json_path = "test_with_interactions_final.json"

interaction_maps = {} # To store maps like {'train': {idx: interactions_list}, 'test': {idx: interactions_list}}

for split_name, json_path in [('train', train_json_path), ('test', test_json_path)]:
    print(f"\nProcessing interaction file for '{split_name}' split: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            # This loads the list of dictionaries from the JSON file
            interaction_data_list = json.load(f)

        # --- Step 3: Create Interaction Mapping Dictionary ---
        split_map = {} # { original_idx: [list of interactions] }
        if isinstance(interaction_data_list, list) and interaction_data_list:
            # Check the structure of the first item to be sure
            first_item = interaction_data_list[0]
            if isinstance(first_item, dict) and "original_idx" in first_item and "user_interactions" in first_item:
                for item in interaction_data_list:
                    idx = item.get("original_idx")
                    interactions = item.get("user_interactions", []) # Default to empty list

                    # Basic validation: Ensure interactions is a list
                    if not isinstance(interactions, list):
                        print(f"Warning: Interactions for original_idx {idx} in {split_name} JSON is not a list. Setting to [].")
                        interactions = []

                    # Basic validation: Check structure of first interaction if list is not empty
                    if interactions and not (isinstance(interactions[0], dict) and 'id' in interactions[0] and 'tone' in interactions[0] and 'content' in interactions[0]):
                         print(f"Warning: Interaction structure for original_idx {idx} in {split_name} JSON seems incorrect. Keeping as is, but review data.")
                         # You might want to add more robust validation here if needed

                    if idx is not None:
                        split_map[idx] = interactions
                    else:
                        print(f"Warning: Found item without 'original_idx' in {json_path}. Skipping.")

                interaction_maps[split_name] = split_map
                print(f"Successfully created interaction map for '{split_name}' with {len(split_map)} entries.")
            else:
                print(f"Error: Interaction JSON structure for '{split_name}' is incorrect.")
                print("Expected: A list of dictionaries, each having 'original_idx' and 'user_interactions' keys.")
                print(f"Structure found in first item: {first_item}")
                continue # Skip this split
        else:
            print(f"Error: Interaction JSON for '{split_name}' is not a list or is empty.")
            continue # Skip this split

    except FileNotFoundError:
        print(f"Error: Interaction JSON file not found at {json_path}. Cannot merge interactions for '{split_name}'.")
        continue # Skip this split
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {json_path}. Check file integrity. Error: {e}")
        continue # Skip this split
    except Exception as e:
        print(f"An error occurred loading interaction JSON for '{split_name}': {e}")
        continue # Skip this split

# Check if we successfully created maps for the necessary splits
if 'train' not in interaction_maps or 'test' not in interaction_maps:
     print("\nFATAL: Could not create interaction maps for both 'train' and 'test' splits from JSON files. Aborting merge.")
     exit()

print("\n--- Step 4 & 5: Merge Interactions into Original Dataset using map() ---")

# Define the function to be applied by map
def add_interactions_mapper(example, idx, current_interaction_map):
    """Looks up interactions using the example's index and adds them."""
    # The 'idx' argument is automatically provided when using with_indices=True
    interactions_list = current_interaction_map.get(idx, []) # Default to empty list
    example['user_interactions'] = interactions_list
    return example

updated_splits = {}
for split_name in ['train', 'test']:
    print(f"\nMerging interactions for '{split_name}' split...")
    original_split_dataset = original_dataset[split_name]
    current_map = interaction_maps[split_name]

    # Apply the map function
    # 'with_indices=True' passes the index of the example as the second argument ('idx') to our function
    # 'fn_kwargs' passes our interaction map dictionary to the function
    updated_split = original_split_dataset.map(
        add_interactions_mapper,
        with_indices=True,
        fn_kwargs={'current_interaction_map': current_map},
        batched=False # Process example-by-example for dictionary lookup
    )
    updated_splits[split_name] = updated_split
    print(f"Finished merging for '{split_name}'. New features: {updated_split.features}")

# Combine the updated splits back into a DatasetDict
final_dataset = DatasetDict(updated_splits)

print("\n--- Step 6: Define and Cast Final Features (Optional but Recommended) ---")
# Define the expected structure of the 'user_interactions' list items
interaction_features = Features({
    'id': Value('string'),
    'tone': Value('string'),
    'content': Value('string')
})

# Get the original features and add the new column definition
final_features = final_dataset['train'].features.copy()
final_features['user_interactions'] = Sequence(feature=interaction_features)

print("\nDefined final features including 'user_interactions':")
print(final_features)

try:
    # Cast the dataset to ensure the new column has the correct type
    final_dataset = final_dataset.cast(final_features)
    print("\nDataset successfully casted to final features.")
except Exception as e:
    print(f"\nWarning: Could not cast dataset features. This might cause issues later. Error: {e}")

print("\n--- Step 7: Verification ---")
print("Final merged dataset:")
print(final_dataset)
print("\nFeatures of the final 'train' split:")
print(final_dataset['train'].features)
print("\nExample of the first train sample:")
print(final_dataset['train'][0]) # Print the first example to check


print("\n--- Step 8: Save the Final Merged Dataset ---")
save_path = "merged_dataset_with_embeddings_and_interactions" # Choose a path
try:
    final_dataset.save_to_disk(save_path)
    print(f"\nFinal merged dataset saved successfully to: {save_path}")
    print(f"You can load it later using: load_from_disk('{save_path}')")
except Exception as e:
    print(f"\nError saving the final dataset: {e}")

print("\n--- Merge process finished! ---")

print(f"\n--- Step 9: Uploading to Hugging Face Hub ---")
try:
    from huggingface_hub import HfApi, HfFolder
    from datasets import DatasetDict

    # Define the repository name and path
    repo_name = "LittleFish-Coder/Fake_News_PolitiFact"
    
    # Upload the dataset to the Hugging Face Hub
    final_dataset.push_to_hub(repo_name)  # Set private=True if you want to keep it private

    print(f"\nDataset successfully uploaded to Hugging Face Hub: {repo_name}")
except Exception as e:
    print(f"\nError uploading dataset to Hugging Face Hub: {e}")