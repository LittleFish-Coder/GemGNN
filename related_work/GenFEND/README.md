# GenFEND: Generative Multi-view Fake News Detection

A simplified, self-contained implementation of GenFEND (Generative Multi-view Fake News Detection) for few-shot fake news detection based on the paper "Let the Crowd Speak: Leveraging Demographic-Enhanced User Comments for Fake News Detection" (2405.16631v1.pdf).

## Overview

This implementation provides a cost-effective version of GenFEND that simulates the paper's core concepts:
- **Demographic-aware Comment Generation**: Simulates 30 demographic user profiles without expensive LLM API calls
- **Multi-view Learning**: Organizes features into demographic views (Gender, Age, Education)
- **Gating Mechanism**: Learned gating to combine views based on content and diversity signals
- **Few-shot Learning**: Supports k-shot scenarios (3-16 shots) using the same dataset as the main repository

## Paper Reference

Based on: "Let the Crowd Speak: Leveraging Demographic-Enhanced User Comments for Fake News Detection"
- Original paper describes using LLMs to role-play 30 demographic user profiles
- Generates synthetic comments for each news piece across different demographic groups
- Uses gating mechanism to combine views based on demographic diversity

## Key Features

### 1. **Simplified Demographic Simulation**
- Generates simulated demographic features for 30 user profiles (as in original paper)
- Creates three demographic views: Gender, Age, Education
- Computes diversity signals as proxy for KL divergence mentioned in paper
- Avoids expensive LLM API calls while maintaining core concepts

### 2. **Gating Mechanism**
- Implements the paper's gating function: `a = Softmax(G(e_o || d; θ))`
- Combines text content (`e_o`) and diversity signals (`d`)
- Produces view weights for convex combination: `r = Σ_V a_V * s_V`
- Concatenates with news embedding for final classification

### 3. **Few-shot Learning**
- Compatible with 3-16 shot scenarios
- Uses same k-shot sampling as main repository
- Label smoothing and early stopping for better few-shot performance
- Supports both PolitiFact and GossipCop datasets

## Installation

No additional requirements beyond the main repository! Ensure you have the main repository dependencies:

```bash
# From the main repository root
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Prepare demographic-aware data (simulated LLM generation):

```bash
cd related_work/GenFEND

# Basic usage
python build.py --dataset_name politifact --k_shot 8

# With specific configuration
python build.py \
    --dataset_name politifact \
    --k_shot 8 \
    --embedding_type deberta \
    --num_demographic_profiles 30 \
    --num_views 3 \
    --output_dir data_genfend
```

**Parameters:**
- `--dataset_name`: Dataset to use (`politifact`, `gossipcop`)
- `--k_shot`: Number of labeled samples per class (3-16)
- `--embedding_type`: Embedding type (`bert`, `roberta`, `deberta`, `distilbert`)
- `--num_demographic_profiles`: Number of demographic profiles (default: 30 as in paper)
- `--num_views`: Number of demographic views (default: 3 for Gender, Age, Education)
- `--partial_unlabeled`: Use partial unlabeled data (default: True)
- `--sample_unlabeled_factor`: Factor for unlabeled sampling (default: 5)

### 2. Model Training

Train the GenFEND model with gating mechanism:

```bash
# Basic training
python train.py --data_path data_genfend/genfend_politifact_k8_deberta.pt

# With specific model configuration
python train.py \
    --data_path data_genfend/genfend_politifact_k8_deberta.pt \
    --hidden_dim 128 \
    --dropout 0.3 \
    --learning_rate 1e-3 \
    --epochs 200 \
    --patience 30 \
    --output_dir results_genfend
```

**Parameters:**
- `--data_path`: Path to prepared data file (required)
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--dropout`: Dropout rate (default: 0.3)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--epochs`: Maximum epochs (default: 200)
- `--patience`: Early stopping patience (default: 30)

### 3. Complete Pipeline Example

```bash
# Step 1: Prepare data
python build.py \
    --dataset_name politifact \
    --k_shot 8 \
    --embedding_type deberta

# Step 2: Train model
python train.py \
    --data_path data_genfend/genfend_politifact_k8_deberta.pt

# Results will be saved in results_genfend/
```

## Implementation Details

### Simplified Demographic Generation

Instead of expensive LLM API calls, this implementation:

1. **Simulates 30 Demographic Profiles**: Generates random features for each profile
2. **Creates View-Specific Patterns**: 
   - Gender: Binary-like patterns
   - Age: Gradient patterns across age groups
   - Education: Education-level exponential patterns
3. **Computes Diversity Signals**: Uses variance across profiles as proxy for KL divergence

### Model Architecture

```python
# GenFEND Model Components:
1. Text Embedding Projection (projects text embeddings to hidden dimension)
2. Demographic Feature Processors (one per view: Gender, Age, Education)
3. Gating Network (implements G(e_o || d; θ) from paper)
4. Classification Head (binary classification with concatenated features)
```

### Gating Mechanism (Core Innovation)

Following the paper's Equation:
```
a = Softmax(G(e_o || d; θ))
r = Σ_V a_V * s_V
final = Concat(e_o, r)
```

Where:
- `e_o`: News text embedding
- `d`: Diversity signals across demographic views
- `G(·; θ)`: Learned gating function
- `a_V`: Attention weights for each view V
- `s_V`: Semantic features for view V
- `r`: Gated demographic representation

### Training Strategy

- **Loss Function**: Cross-entropy with label smoothing (α=0.1) for few-shot robustness
- **Optimization**: Adam optimizer with weight decay
- **Early Stopping**: Based on validation F1 score with patience
- **Evaluation**: Standard classification metrics + view importance analysis

## Differences from Original GenFEND

### Simplified (Cost-Effective):
- ❌ Real LLM API calls for demographic comment generation
- ❌ Complex demographic user profile role-playing
- ❌ Actual comment text generation
- ❌ Complex NLP processing of generated comments
- ❌ Expensive computational requirements

### Retained (Core Concepts):
- ✅ 30 demographic profiles concept
- ✅ Three demographic views (Gender, Age, Education)
- ✅ Gating mechanism with diversity signals
- ✅ Multi-view learning framework
- ✅ Few-shot learning capability
- ✅ Same datasets and evaluation protocol

## File Structure

```
related_work/GenFEND/
├── build.py                   # Data preparation script
├── train.py                   # Training script with gating mechanism
├── README.md                  # This file
├── data_genfend/             # Prepared data (created automatically)
└── results_genfend/          # Training results (created automatically)
```

## Results

After training, you'll find:

1. **Training Metrics**: JSON file with accuracy, precision, recall, F1-score
2. **Training Curves**: Plots showing loss and accuracy over epochs
3. **Model Checkpoint**: Best model state dict
4. **View Importance**: Average gating weights showing demographic view importance

Example results structure:
```
results_genfend/
├── genfend_politifact_k8_deberta_results.json
├── genfend_politifact_k8_deberta_curves.png
├── genfend_politifact_k8_deberta_model.pt
└── ...
```

## Expected Performance

For 8-shot scenarios (based on thesis results showing GenFEND around 0.37-0.39 F1):
- **PolitiFact**: ~35-45% F1 score (simplified version)
- **GossipCop**: ~35-40% F1 score (simplified version)

Performance factors:
- **K-shot value**: Fewer shots = lower performance
- **Demographic diversity**: More diverse profiles may improve performance
- **Gating effectiveness**: Better learned gates improve view combination
- **View importance**: Different datasets may favor different demographic views

## Key Insights from Training

The model will show:
1. **View Importance**: Which demographic views (Gender/Age/Education) are most important
2. **Gating Adaptation**: How the model learns to weight views for different news types
3. **Few-shot Effectiveness**: Performance across different k-shot scenarios
4. **Demographic Signal Value**: Whether demographic diversity helps fake news detection

## Comparison with Graph Methods

Unlike graph-based methods in this repository (HeteroSGT, LESS4FD):
- **No Graph Construction**: Works directly with embeddings and demographic features
- **LLM-Inspired**: Simulates crowd intelligence through demographic diversity
- **Lightweight**: Fewer parameters and faster training
- **Complementary**: Can potentially be combined with graph methods

## Troubleshooting

### Common Issues:

1. **"Data file not found"**:
   ```bash
   # Make sure to prepare data first
   python build.py --dataset_name politifact --k_shot 8
   ```

2. **CUDA out of memory**:
   ```bash
   # Reduce hidden dimensions
   python train.py --data_path <path> --hidden_dim 64
   ```

3. **Poor performance**:
   ```bash
   # Try different learning rate or more demographic profiles
   python build.py --num_demographic_profiles 50
   python train.py --learning_rate 5e-4
   ```

### Performance Tips:

1. **Increase demographic profiles** for richer representation
2. **Tune gating network** for better view combination
3. **Adjust diversity signals** computation method
4. **Use label smoothing** for better few-shot generalization

## Future Enhancements

Potential improvements to bridge the gap with the original paper:
1. **Real LLM Integration**: Add optional LLM API calls for actual comment generation
2. **Better Demographic Modeling**: Use more sophisticated demographic feature generation
3. **Adaptive Views**: Learn optimal number and type of demographic views
4. **Graph Integration**: Combine with graph-based methods for hybrid approach

## Citation

If you use this simplified GenFEND implementation, please cite:

```bibtex
@article{genfend2024,
    title={Let the Crowd Speak: Leveraging Demographic-Enhanced User Comments for Fake News Detection},
    author={[Authors from the paper]},
    journal={arXiv preprint arXiv:2405.16631},
    year={2024}
}

@misc{gemgnn2024,
    title={GemGNN: Generative Multi-view Interaction Graph Neural Networks for Few-shot Fake News Detection},
    author={LittleFish-Coder},
    year={2024},
    url={https://github.com/LittleFish-Coder/GemGNN}
}
```

## License

This implementation follows the same license as the main GemGNN repository.