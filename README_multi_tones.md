# Multi-Tone Combinations Usage Examples

## Overview

The `build_hetero_graph_multi_tones.py` script extends the original tone ablation study capabilities by supporting specific counts of each interaction tone type.

## Key Features

### 1. Extended Predefined Combinations (0-16)

Original combinations (0-6) remain unchanged for backward compatibility:
- `0`: All tones (8N + 7A + 5S) - default
- `1`: Neutral only (8)
- `2`: Affirmative only (7) 
- `3`: Skeptical only (5)
- `4`: Neutral + Affirmative
- `5`: Neutral + Skeptical
- `6`: Affirmative + Skeptical

New rational combinations (7-16):
- `7`: 2 Neutral + 1 Affirmative + 1 Skeptical
- `8`: 4 Neutral + 2 Affirmative + 2 Skeptical  
- `9`: 6 Neutral + 3 Affirmative + 3 Skeptical
- `10`: 1 Neutral + 2 Affirmative + 1 Skeptical
- `11`: 1 Neutral + 1 Affirmative + 2 Skeptical
- `12`: 4 Neutral only
- `13`: 8 Neutral only
- `14`: 12 Neutral only
- `15`: 16 Neutral only
- `16`: 20 Neutral only

### 2. Custom Multi-Tone Counts

Use `--multi_tone_counts` to specify exact counts:
```bash
--multi_tone_counts "2,1,1"  # 2 neutral, 1 affirmative, 1 skeptical
--multi_tone_counts "3,0,2"  # 3 neutral, 0 affirmative, 2 skeptical
--multi_tone_counts "1,4,1"  # 1 neutral, 4 affirmative, 1 skeptical
```

## Usage Examples

### Basic Usage with Predefined Combinations

```bash
# Professor's suggested combinations using predefined options
python build_hetero_graph_multi_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --embedding_type deberta \
  --edge_policy knn_test_isolated \
  --k_neighbors 5 \
  --tones_selection 7  # 2N+1A+1S

python build_hetero_graph_multi_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --embedding_type deberta \
  --edge_policy knn_test_isolated \
  --k_neighbors 5 \
  --tones_selection 8  # 4N+2A+2S
```

### Custom Multi-Tone Combinations

```bash
# Custom combination: 3 neutral, 2 affirmative, 1 skeptical
python build_hetero_graph_multi_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --embedding_type deberta \
  --edge_policy knn_test_isolated \
  --k_neighbors 5 \
  --multi_tone_counts "3,2,1"

# Custom combination: 5 neutral, 1 affirmative, 3 skeptical  
python build_hetero_graph_multi_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --embedding_type deberta \
  --edge_policy knn_test_isolated \
  --k_neighbors 5 \
  --multi_tone_counts "5,1,3"
```

### Interaction Count Ablation Study (Professor's Request)

Testing different total interaction counts (0, 4, 8, 12, 16, 20):

```bash
# 0 interactions (no interactions mode)
python build_hetero_graph_multi_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --no_interactions

# 4 interactions total
python build_hetero_graph_multi_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --multi_tone_counts "2,1,1"  # Total: 4

# 8 interactions total
python build_hetero_graph_multi_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --multi_tone_counts "4,2,2"  # Total: 8

# 12 interactions total  
python build_hetero_graph_multi_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --multi_tone_counts "6,3,3"  # Total: 12

# 16 interactions total
python build_hetero_graph_multi_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --multi_tone_counts "8,4,4"  # Total: 16

# 20 interactions total (original full set)
python build_hetero_graph_multi_tones.py \
  --dataset_name politifact \
  --k_shot 8 \
  --tones_selection 0  # All interactions
```

## Output Naming Convention

The script generates human-readable graph names:

### Predefined Combinations
- `graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_tones_2N_1A_1S/`
- `graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_tones_4N_2A_2S/`

### Custom Combinations
- `graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_tones_3N_2A_1S/`
- `graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_tones_5N_1A_3S/`

### Original Combinations (backward compatible)
- `graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5/` (all tones)
- `graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_neutral_only/`

## Systematic Ablation Study Design

### Recommended Experiment Series

1. **Interaction Count Analysis** (Professor's request):
   ```bash
   for interactions in 0 4 8 12 16 20; do
     case $interactions in
       0) args="--no_interactions" ;;
       4) args="--multi_tone_counts \"2,1,1\"" ;;
       8) args="--multi_tone_counts \"4,2,2\"" ;;
       12) args="--multi_tone_counts \"6,3,3\"" ;;
       16) args="--multi_tone_counts \"8,4,4\"" ;;
       20) args="--tones_selection 0" ;;
     esac
     python build_hetero_graph_multi_tones.py --dataset_name politifact --k_shot 8 $args
   done
   ```

2. **Tone Balance Analysis**:
   ```bash
   # Neutral-heavy combinations
   python build_hetero_graph_multi_tones.py --multi_tone_counts "4,1,1"  # Neutral dominant
   python build_hetero_graph_multi_tones.py --multi_tone_counts "2,2,2"  # Balanced
   python build_hetero_graph_multi_tones.py --multi_tone_counts "1,3,2"  # Affirmative heavy
   python build_hetero_graph_multi_tones.py --multi_tone_counts "1,1,4"  # Skeptical heavy
   ```

3. **Progressive Scaling**:
   ```bash
   # Scale all tones proportionally
   python build_hetero_graph_multi_tones.py --multi_tone_counts "1,1,1"   # Total: 3
   python build_hetero_graph_multi_tones.py --multi_tone_counts "2,2,2"   # Total: 6
   python build_hetero_graph_multi_tones.py --multi_tone_counts "3,3,3"   # Total: 9
   python build_hetero_graph_multi_tones.py --multi_tone_counts "4,4,4"   # Total: 12
   ```

## Backward Compatibility

The script maintains full backward compatibility with `build_hetero_graph_tones.py`:

- All original `--tones_selection 0-6` options work identically
- Default behavior (`--tones_selection 0`) unchanged
- Graph naming for original options preserved
- All other parameters remain the same

## Integration with Training

The generated graphs can be used directly with existing training scripts:

```bash
# Train with custom tone combination
python train_hetero_graph.py \
  --graph_path graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_tones_2N_1A_1S/graph.pt \
  --model HAN \
  --loss_fn ce
```

This enables researchers to systematically study the impact of different interaction tone combinations on few-shot fake news detection performance.