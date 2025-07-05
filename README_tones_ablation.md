# Tone Ablation Study Support

This document describes the new `build_hetero_graph_tones.py` script that enables tone ablation studies for few-shot fake news detection.

## Overview

The `build_hetero_graph_tones.py` script is a modified version of `build_hetero_graph.py` that allows researchers to select specific combinations of interaction tones for ablation studies. This enables systematic evaluation of how different interaction types (neutral, affirmative, skeptical) contribute to fake news detection performance.

## Tone Selection Options

The script supports 7 different tone configurations:

| Selection | Name | Description | Interactions |
|-----------|------|-------------|--------------|
| 0 | All tones (default) | Neutral + Affirmative + Skeptical | 8 + 7 + 5 |
| 1 | Neutral only | Only neutral interactions | 8 |
| 2 | Affirmative only | Only affirmative interactions | 8 |
| 3 | Skeptical only | Only skeptical interactions | 8 |
| 4 | Neutral + Affirmative | Combination of neutral and affirmative | Variable |
| 5 | Neutral + Skeptical | Combination of neutral and skeptical | Variable |
| 6 | Affirmative + Skeptical | Combination of affirmative and skeptical | Variable |

## Usage

### Basic Usage

```bash
# Default: All tones (Neutral + Affirmative + Skeptical)
python build_hetero_graph_tones.py --dataset_name politifact --k_shot 8

# Neutral interactions only
python build_hetero_graph_tones.py --dataset_name politifact --k_shot 8 --tones_selection 1

# Skeptical interactions only
python build_hetero_graph_tones.py --dataset_name politifact --k_shot 8 --tones_selection 3
```

### Advanced Usage

```bash
# Neutral + Affirmative with custom edge policy
python build_hetero_graph_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --tones_selection 4 \
  --edge_policy knn \
  --k_neighbors 3

# Skeptical only with partial unlabeled sampling
python build_hetero_graph_tones.py \
  --dataset_name gossipcop \
  --k_shot 12 \
  --tones_selection 3 \
  --partial_unlabeled \
  --sample_unlabeled_factor 5
```

## Graph Naming

The script automatically adds suffixes to graph directory names based on tone selection:

- **All tones (0)**: No suffix (default behavior)
- **Neutral only (1)**: `_neutral_only`
- **Affirmative only (2)**: `_affirmative_only`
- **Skeptical only (3)**: `_skeptical_only`
- **Neutral + Affirmative (4)**: `_neutral_affirmative`
- **Neutral + Skeptical (5)**: `_neutral_skeptical`
- **Affirmative + Skeptical (6)**: `_affirmative_skeptical`

### Example Graph Paths

```
graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5/           # All tones
graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_neutral_only/     # Neutral only
graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_skeptical_only/   # Skeptical only
```

## Implementation Details

### Tone Filtering

The script implements tone filtering through the `_filter_interactions_by_tone()` method:

1. **Input validation**: Ensures embeddings and tones lists have matching lengths
2. **Tone normalization**: Converts tones to lowercase and replaces spaces with underscores
3. **Selective filtering**: Only includes interactions matching selected tone types
4. **Pairing preservation**: Maintains correct embedding-tone pairs during filtering

### Error Handling

The script handles edge cases gracefully:

- **Empty results**: If no interactions remain after filtering, creates empty tensors
- **Mismatched counts**: Warns when interaction counts don't match expectations
- **Invalid selections**: Validates tone selection is in range 0-6

### Data Structure Consistency

The filtering maintains data structure consistency:

- **HeteroData format**: Preserves PyTorch Geometric heterogeneous graph structure
- **Edge connectivity**: Maintains proper node-interaction relationships
- **Embedding dimensions**: Preserves original embedding dimensions

## Validation

Run the validation tests to ensure functionality:

```bash
# Test tone filtering logic
python test_tones_selection.py

# View usage examples
python demo_tones_selection.py
```

## Integration with Training

Use filtered graphs with the existing training pipeline:

```bash
# Train model on neutral-only interactions
python train_hetero_graph.py \
  --graph_path graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_neutral_only/graph.pt \
  --model HAN \
  --loss_fn ce

# Train model on skeptical-only interactions  
python train_hetero_graph.py \
  --graph_path graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_skeptical_only/graph.pt \
  --model HAN \
  --loss_fn ce
```

## Research Applications

This functionality enables several research directions:

1. **Ablation Studies**: Systematic evaluation of individual tone contributions
2. **Optimal Combinations**: Finding the best tone combinations for different datasets
3. **Domain Analysis**: Understanding tone importance across different news domains
4. **Few-shot Optimization**: Identifying minimal tone sets for effective few-shot learning

## Performance Considerations

- **Memory Usage**: Filtered graphs may use less memory due to fewer interaction nodes
- **Training Speed**: Smaller graphs typically train faster
- **Edge Connectivity**: Ensure filtered graphs maintain sufficient connectivity for message passing

## Compatibility

The `build_hetero_graph_tones.py` script maintains full compatibility with:

- All existing command-line arguments
- Original dataset formats
- Downstream training scripts
- Evaluation pipelines

The only addition is the `--tones_selection` parameter, which defaults to 0 (all tones) for backward compatibility.