import os
import gc
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from datasets import load_dataset, DatasetDict, Dataset
from argparse import ArgumentParser, Namespace
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.sample_k_shot import sample_k_shot
from tqdm import tqdm

# Constants
DEFAULT_MAX_LENGTH = 512
DEFAULT_MAX_NEW_TOKENS = 50
SEED = 42
LOG_DIR = "logs"

# Model-specific prompts
SYSTEM_PROMPTS = {
    "default": "You are an expert at detecting fake news. Given an article, classify it as REAL or FAKE.",
    "llama": "You are a helpful, harmless, and precise assistant specialized in detecting fake news.",
    "chatglm": "You are ChatGLM, a responsible AI assistant. Your task is to analyze news articles and classify them as REAL or FAKE.",
    "mistral": "You are Mistral AI assistant, an expert at analyzing news. Determine if articles are REAL or FAKE with high accuracy.",
    "gemma": "You are Gemma, an AI assistant by Google. Analyze news articles and classify them as either REAL or FAKE."
}


class FakeNewsLLMPrompter:
    """
    A class to manage LLM prompting for fake news detection using few-shot learning.
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        k_shot: int,
        output_dir: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        use_4bit: bool = False,
        device: str = None,
        chat_format: str = "auto",
        system_prompt: Optional[str] = None
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.k_shot = k_shot
        self.max_new_tokens = max_new_tokens
        self.use_4bit = use_4bit
        self.chat_format = chat_format
        
        # Auto-detect chat format if not specified
        if chat_format == "auto":
            self.chat_format = self._detect_chat_format(model_name)
        
        # Set the appropriate system prompt
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self._get_system_prompt(self.chat_format)
        
        # Determine model type from model name
        self.model_type = self._get_model_type(model_name)
        
        # Get the shot string (e.g., "8-shot" or "full" for k=0)
        shot_str = "full" if k_shot == 0 else f"{k_shot}-shot"
        
        # Setup directory paths using a single base output directory
        self.model_dir = os.path.join(output_dir, self.model_type, dataset_name, shot_str)
        
        # Create directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize components
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Store device settings
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store indices and label distribution for shot selection
        self.selected_indices = None
        self.label_distribution = None
        
        print(f"Using chat format: {self.chat_format}")
        print(f"System prompt: {self.system_prompt[:50]}...")
    
    def _detect_chat_format(self, model_name: str) -> str:
        """Auto-detect the appropriate chat format based on model name."""
        model_name_lower = model_name.lower()
        
        if "llama-3" in model_name_lower or "llama3" in model_name_lower:
            return "llama3"
        elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
            if "chat" in model_name_lower or "instruct" in model_name_lower:
                return "llama2"
            else:
                return "raw"  # Base model, not instruction-tuned
        elif "chatglm" in model_name_lower:
            return "chatglm"
        elif "mistral" in model_name_lower:
            if "instruct" in model_name_lower:
                return "mistral-instruct"
            else:
                return "raw"
        elif "gemma" in model_name_lower:
            if "instruct" in model_name_lower:
                return "gemma-instruct"
            else:
                return "raw"
        else:
            return "raw"
    
    def _get_system_prompt(self, chat_format: str) -> str:
        """Get the appropriate system prompt for the specified chat format."""
        if chat_format in SYSTEM_PROMPTS:
            return SYSTEM_PROMPTS[chat_format]
        if chat_format == "llama3":
            return SYSTEM_PROMPTS["llama"]
        if chat_format == "llama2":
            return SYSTEM_PROMPTS["llama"]
        if chat_format == "mistral-instruct":
            return SYSTEM_PROMPTS["mistral"]
        if chat_format == "gemma-instruct":
            return SYSTEM_PROMPTS["gemma"]
        return SYSTEM_PROMPTS["default"]
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine model type from model name."""
        model_name_lower = model_name.lower()
        
        if "llama-3" in model_name_lower or "llama3" in model_name_lower:
            return "llama3"
        elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
            return "llama2"
        elif "chatglm" in model_name_lower:
            return "chatglm"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "gemma" in model_name_lower:
            return "gemma"
        
        # Default to the model name for other models
        return model_name.split("/")[-1]
    
    def load_dataset(self) -> None:
        """Load and prepare the dataset."""
        print(f"Loading dataset '{self.dataset_name}'...")
        
        # Load dataset from Hugging Face
        dataset = load_dataset(
            f"LittleFish-Coder/Fake_News_{self.dataset_name}",
            download_mode="reuse_cache_if_exists",
            cache_dir="dataset",
        )
        
        # Sample k-shot if needed
        if self.k_shot > 0:
            dataset = self._sample_k_shot(dataset, self.k_shot)
        
        self.dataset = dataset
        print(f"Dataset loaded and prepared. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    def _sample_k_shot(self, dataset: DatasetDict, k: int) -> DatasetDict:
        """Sample k examples per class for few-shot learning."""
        print(f"Sampling {k}-shot data per class...")
        
        train_data = dataset["train"]
        
        # Use the shared sampling function to ensure consistency with other models
        selected_indices, sampled_data = sample_k_shot(train_data, k, seed=SEED)
        
        # Store the selected indices for later
        self.selected_indices = selected_indices
        
        # Calculate and store label distribution
        self.label_distribution = {}
        for idx in selected_indices:
            label = train_data["label"][idx]
            self.label_distribution[label] = self.label_distribution.get(label, 0) + 1
        
        # Create new dataset with sampled training data
        return DatasetDict({
            "train": Dataset.from_dict(sampled_data),
            "test": dataset["test"],
        })
    
    def setup_model_and_tokenizer(self) -> None:
        """Set up the model and tokenizer."""
        print(f"Setting up tokenizer and model: {self.model_name}")
        
        # Configure tokenizer based on model type
        tokenizer_kwargs = {
            "padding_side": "left",  # Most LLMs use left padding
        }
        
        # Add model-specific tokenizer settings
        if "llama" in self.chat_format:
            tokenizer_kwargs.update({
                "use_fast": True,
                "trust_remote_code": True
            })
        elif "chatglm" in self.chat_format:
            tokenizer_kwargs.update({
                "trust_remote_code": True
            })
        elif "mistral" in self.chat_format or "gemma" in self.chat_format:
            tokenizer_kwargs.update({
                "use_fast": True
            })
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            
            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Load model normally
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
        
        # Create text generation pipeline
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Use greedy decoding for consistent results
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        print("Model and tokenizer setup complete")
    
    def create_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for the LLM using few-shot examples based on the chat format.
        
        Args:
            text: The article text to classify
            examples: List of example dictionaries containing text and label
            
        Returns:
            Formatted prompt string
        """
        # Select the appropriate prompt formatter based on chat format
        if self.chat_format == "llama3":
            return self._create_llama3_prompt(text, examples)
        elif self.chat_format == "llama2":
            return self._create_llama2_prompt(text, examples)
        elif self.chat_format == "chatglm":
            return self._create_chatglm_prompt(text, examples)
        elif self.chat_format == "mistral-instruct":
            return self._create_mistral_prompt(text, examples)
        elif self.chat_format == "gemma-instruct":
            return self._create_gemma_prompt(text, examples)
        else:
            # Default raw prompt format
            return self._create_raw_prompt(text, examples)
    
    def _create_raw_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """Create a simple prompt for models without a specific chat format."""
        prompt = f"{self.system_prompt}\n\n"
        
        # Add few-shot examples if provided
        if examples:
            for i, example in enumerate(examples, 1):
                label = "REAL" if example["label"] == 0 else "FAKE"
                prompt += f"Example {i}:\n"
                prompt += f"Article: {example['text'][:500]}...\n"  # Truncate long examples
                prompt += f"Classification: {label}\n\n"
        
        # Add the article to classify
        prompt += f"Article: {text}\n"
        prompt += "Classification:"
        
        return prompt
    
    def _create_llama3_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """Create a prompt using Llama 3's chat format."""
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        user_message = "I need you to classify the following news article as REAL or FAKE. "
        user_message += "Respond with exactly REAL or FAKE at the beginning of your response, followed by your reasoning. "
        
        # Add few-shot examples if provided
        if examples:
            user_message += "Here are some examples to guide you:\n\n"
            for i, example in enumerate(examples, 1):
                label = "REAL" if example["label"] == 0 else "FAKE"
                user_message += f"Example {i}:\n"
                user_message += f"Article: {example['text'][:300]}...\n"  # Truncate long examples
                user_message += f"Classification: {label}\n\n"
        
        user_message += f"Now, classify this article:\n{text}\n\nClassification:"
        messages.append({"role": "user", "content": user_message})
        
        # Format as Llama 3 expects
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
        
        formatted += "<|assistant|>\n"
        return formatted
    
    def _create_llama2_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """Create a prompt using Llama 2's chat format."""
        # Begin with system prompt inside the first user message
        prompt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n"
        
        # Add the user's request
        user_message = "I need you to classify the following news article as REAL or FAKE. "
        user_message += "Respond with exactly REAL or FAKE at the beginning of your response, followed by your reasoning. "
        
        # Add few-shot examples if provided
        if examples:
            user_message += "Here are some examples to guide you:\n\n"
            for i, example in enumerate(examples, 1):
                label = "REAL" if example["label"] == 0 else "FAKE"
                user_message += f"Example {i}:\n"
                user_message += f"Article: {example['text'][:300]}...\n"  # Truncate long examples
                user_message += f"Classification: {label}\n\n"
        
        user_message += f"Now, classify this article:\n{text}\n\nClassification:"
        prompt += user_message + " [/INST]\n"
        
        return prompt
    
    def _create_chatglm_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """Create a prompt using ChatGLM's format."""
        # ChatGLM uses a list of message dicts that gets processed by its tokenizer
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        user_message = "I need you to classify the following news article as REAL or FAKE. "
        user_message += "Respond with exactly REAL or FAKE at the beginning of your response, followed by your reasoning. "
        
        # Add few-shot examples if provided
        if examples:
            user_message += "Here are some examples to guide you:\n\n"
            for i, example in enumerate(examples, 1):
                label = "REAL" if example["label"] == 0 else "FAKE"
                user_message += f"Example {i}:\n"
                user_message += f"Article: {example['text'][:300]}...\n"  # Truncate long examples
                user_message += f"Classification: {label}\n\n"
        
        user_message += f"Now, classify this article:\n{text}\n\nClassification:"
        messages.append({"role": "user", "content": user_message})
        
        # For ChatGLM, return these messages - the pipeline will handle them
        # but we need to convert to the format expected by transformers pipeline
        prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]
        return prompt
    
    def _create_mistral_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """Create a prompt using Mistral Instruct format."""
        # Mistral uses a specific format with <s> tags
        prompt = f"<s>[INST] {self.system_prompt}\n\n"
        
        user_message = "I need you to classify the following news article as REAL or FAKE. "
        user_message += "Respond with exactly REAL or FAKE at the beginning of your response, followed by your reasoning. "
        
        # Add few-shot examples if provided
        if examples:
            user_message += "Here are some examples to guide you:\n\n"
            for i, example in enumerate(examples, 1):
                label = "REAL" if example["label"] == 0 else "FAKE"
                user_message += f"Example {i}:\n"
                user_message += f"Article: {example['text'][:300]}...\n"  # Truncate long examples
                user_message += f"Classification: {label}\n\n"
        
        user_message += f"Now, classify this article:\n{text}\n\nClassification: [/INST]"
        prompt += user_message
        
        return prompt
    
    def _create_gemma_prompt(self, text: str, examples: List[Dict[str, Any]] = None) -> str:
        """Create a prompt using Gemma's Instruct format."""
        prompt = f"<start_of_turn>user\n{self.system_prompt}\n\n"
        
        user_message = "I need you to classify the following news article as REAL or FAKE. "
        user_message += "Respond with exactly REAL or FAKE at the beginning of your response, followed by your reasoning. "
        
        # Add few-shot examples if provided
        if examples:
            user_message += "Here are some examples to guide you:\n\n"
            for i, example in enumerate(examples, 1):
                label = "REAL" if example["label"] == 0 else "FAKE"
                user_message += f"Example {i}:\n"
                user_message += f"Article: {example['text'][:300]}...\n"  # Truncate long examples
                user_message += f"Classification: {label}\n\n"
        
        user_message += f"Now, classify this article:\n{text}\n\nClassification:"
        prompt += user_message + "<end_of_turn>\n<start_of_turn>model\n"
        
        return prompt
    
    def predict(self, text: str, examples: List[Dict[str, Any]] = None) -> Tuple[int, float]:
        """
        Generate a prediction for a single text.
        
        Args:
            text: The article text to classify
            examples: Optional list of few-shot examples
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        # Create the prompt
        prompt = self.create_prompt(text, examples)
        
        # Generate a response
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,  # Use greedy decoding for consistent results
        }
        
        # Add model-specific generation parameters
        if "llama" in self.chat_format or "mistral" in self.chat_format or "gemma" in self.chat_format:
            generation_kwargs.update({
                "temperature": 0.1,  # Low temperature for more deterministic outputs
                "top_p": 0.9
            })
            
        response = self.pipeline(prompt, **generation_kwargs)[0]['generated_text']
        
        # Extract only the new text (remove the prompt)
        new_text = response[len(prompt):].strip()
        
        # Parse the prediction based on the response
        return self._parse_response(new_text)
    
    def _parse_response(self, response: str) -> Tuple[int, float]:
        """
        Parse the model's response to extract the classification.
        
        Args:
            response: The text generated by the model
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        # Default values
        prediction = 0  # Default to "real"
        confidence = 0.5  # Default confidence
        
        # Convert to uppercase for case-insensitive matching
        upper_response = response.upper()
        
        # Check for the most explicit patterns first
        if upper_response.startswith("FAKE") or "CLASSIFICATION: FAKE" in upper_response:
            prediction = 1
            confidence = 0.9
        elif upper_response.startswith("REAL") or "CLASSIFICATION: REAL" in upper_response:
            prediction = 0
            confidence = 0.9
        else:
            # More detailed scanning of the response
            fake_indicators = ["FAKE", "NOT REAL", "FABRICATED", "FALSE", "MISLEADING"]
            real_indicators = ["REAL", "TRUE", "ACCURATE", "FACTUAL", "AUTHENTIC"]
            
            fake_score = sum(upper_response.count(indicator) for indicator in fake_indicators)
            real_score = sum(upper_response.count(indicator) for indicator in real_indicators)
            
            # Determine prediction based on which class has more indicators
            if fake_score > real_score:
                prediction = 1
                confidence = min(0.5 + (fake_score - real_score) * 0.1, 0.9)
            elif real_score > fake_score:
                prediction = 0
                confidence = min(0.5 + (real_score - fake_score) * 0.1, 0.9)
            # If equal or none found, stick with the default
            
        return prediction, confidence
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model and save metrics."""
        print("Evaluating model on test set...")
        
        # Get examples for few-shot prompting
        examples = None
        if self.k_shot > 0:
            examples = []
            for i in range(min(self.k_shot * 2, len(self.dataset["train"]))):
                examples.append({
                    "text": self.dataset["train"][i]["text"],
                    "label": self.dataset["train"][i]["label"]
                })
        
        # Collect predictions
        predictions = []
        labels = []
        confidences = []
        
        # Process test examples with progress bar
        for item in tqdm(self.dataset["test"], desc="Evaluating"):
            pred, conf = self.predict(item["text"], examples)
            predictions.append(pred)
            labels.append(item["label"])
            confidences.append(conf)
        
        # Calculate metrics
        results = {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted"),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0
        }
        
        # Save metrics to results directory
        metrics_file = os.path.join(self.model_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save sample indices if we used k-shot sampling
        if self.selected_indices is not None:
            indices_file = os.path.join(self.model_dir, "indices.json")
            
            # Convert numpy types to Python native types
            indices_info = {
                "indices": [int(i) for i in self.selected_indices],
                "k_shot": int(self.k_shot),
                "seed": int(SEED),
                "dataset_name": self.dataset_name,
            }
            
            if hasattr(self, 'label_distribution') and self.label_distribution:
                indices_info["label_distribution"] = {
                    int(k): int(v) for k, v in self.label_distribution.items()
                }
            
            with open(indices_file, "w") as f:
                json.dump(indices_info, f, indent=2)
            
            print(f"Selected indices saved to {indices_file}")
        
        print(f"Evaluation completed. Results saved to {metrics_file}")
        return results
    
    def run_pipeline(self) -> Dict[str, float]:
        """Run the complete evaluation pipeline."""
        self.load_dataset()
        self.setup_model_and_tokenizer()
        return self.evaluate()

def set_seed(seed: int = SEED) -> None:
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Optional: Set random seed for transformers library
    try:
        import transformers
        transformers.set_seed(seed)
    except:
        pass

def parse_arguments() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Prompt LLMs for fake news detection")
    
    # Model selection
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)",
        choices=["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B", "meta-llama/Meta-Llama-3-8B-Chat", "meta-llama/Meta-Llama-3-70B-Chat", "meta-llama/Meta-Llama-3-8B-Instruct-Chat", "meta-llama/Meta-Llama-3-70B-Instruct-Chat"]
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="politifact",
        help="Dataset to use (default: politifact)",
        choices=["tfg", "kdd2020", "gossipcop", "politifact"],
    )
    
    # Few-shot setting
    parser.add_argument(
        "--k_shot",
        type=int,
        default=8,
        help="Number of samples per class for few-shot learning (default: 8, 0 for zero-shot)",
        choices=[0, 8, 16, 32, 100],
    )
    
    # Model settings
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum new tokens to generate (default: 50)",
    )
    
    # Single output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_llm",
        help="Directory to save results (default: results_llm)",
    )
    
    # Quantization option
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization for model loading",
    )
    
    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, otherwise cpu)",
    )
    
    # Chat format
    parser.add_argument(
        "--chat_format",
        type=str,
        default="auto",
        choices=["auto", "llama3", "llama2", "chatglm", "mistral-instruct", "gemma-instruct", "raw"],
        help="Chat format to use (default: auto-detect based on model name)",
    )
    
    # System prompt
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt (overrides default)",
    )
    
    return parser.parse_args()

def main() -> None:
    """Main function to run the LLM prompting pipeline."""
    # Set seed for reproducibility
    set_seed()

    # Clean up CUDA memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Parse arguments
    args = parse_arguments()
    
    # Display arguments and hardware info
    print("\n" + "="*50)
    print("Fake News Detection - LLM Prompting")
    print("="*50)
    print(f"Model:        {args.model_name}")
    print(f"Dataset:      {args.dataset_name}")
    print(f"K-shot:       {args.k_shot}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Chat format:  {args.chat_format}")
    print(f"Device:       {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"4-bit quant:  {args.use_4bit}")
    print("="*50 + "\n")
    
    # Create and run prompter
    prompter = FakeNewsLLMPrompter(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        k_shot=args.k_shot,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        use_4bit=args.use_4bit,
        device=args.device,
        chat_format=args.chat_format,
        system_prompt=args.system_prompt
    )
    
    # Run the pipeline
    metrics = prompter.run_pipeline()
    
    # Display final results
    print("\n" + "="*50)
    print("Evaluation Complete - Results")
    print("="*50)
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"Avg Confidence: {metrics['avg_confidence']:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
