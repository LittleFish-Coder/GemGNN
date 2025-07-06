#!/bin/bash
# Example usage script for build_hetero_graph_multi_tones.py
# This script demonstrates various tone combination experiments

echo "=== Multi-Tone Combinations Examples ==="
echo "This script shows example usage for the new multi-tone functionality"
echo ""

# Base parameters
DATASET="politifact"
K_SHOT=8
EMBEDDING="deberta"
EDGE_POLICY="knn_test_isolated"
K_NEIGHBORS=5
BASE_ARGS="--dataset_name $DATASET --k_shot $K_SHOT --embedding_type $EMBEDDING --edge_policy $EDGE_POLICY --k_neighbors $K_NEIGHBORS"

echo "Base parameters: $BASE_ARGS"
echo ""

echo "1. Professor's Suggested Combinations (predefined):"
echo "----------------------------------------"

# Professor's suggestions using predefined options
echo "# 2 Neutral + 1 Affirmative + 1 Skeptical (tones_selection=7)"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --tones_selection 7"
echo ""

echo "# 4 Neutral + 2 Affirmative + 2 Skeptical (tones_selection=8)"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --tones_selection 8"
echo ""

echo "# 6 Neutral + 3 Affirmative + 3 Skeptical (tones_selection=9)"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --tones_selection 9"
echo ""

echo "# 1 Neutral + 2 Affirmative + 1 Skeptical (tones_selection=10)"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --tones_selection 10"
echo ""

echo "# 1 Neutral + 1 Affirmative + 2 Skeptical (tones_selection=11)"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --tones_selection 11"
echo ""

echo "2. Custom Multi-Tone Combinations:"
echo "----------------------------------------"

# Custom combinations using multi_tone_counts
echo "# 3 Neutral + 2 Affirmative + 1 Skeptical"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"3,2,1\""
echo ""

echo "# 5 Neutral + 1 Affirmative + 3 Skeptical"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"5,1,3\""
echo ""

echo "# 2 Neutral + 3 Affirmative + 2 Skeptical"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"2,3,2\""
echo ""

echo "3. Interaction Count Analysis (Professor's Request):"
echo "----------------------------------------"

echo "# 0 interactions (no interactions mode)"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --no_interactions"
echo ""

echo "# 4 total interactions"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"2,1,1\""
echo ""

echo "# 8 total interactions"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"4,2,2\""
echo ""

echo "# 12 total interactions"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"6,3,3\""
echo ""

echo "# 16 total interactions"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"8,4,4\""
echo ""

echo "# 20 total interactions (original full set)"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --tones_selection 0"
echo ""

echo "4. Systematic Ablation Study:"
echo "----------------------------------------"

echo "# Tone balance analysis"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"4,1,1\"  # Neutral-heavy"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"2,2,2\"  # Balanced"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"1,3,2\"  # Affirmative-heavy"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"1,1,4\"  # Skeptical-heavy"
echo ""

echo "# Progressive scaling"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"1,1,1\"  # Total: 3"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"2,2,2\"  # Total: 6" 
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"3,3,3\"  # Total: 9"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"4,4,4\"  # Total: 12"
echo ""

echo "5. Single Tone Type with Counts:"
echo "----------------------------------------"

echo "# Neutral-only with different counts"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --tones_selection 12  # 4 neutral only"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --tones_selection 13  # 8 neutral only"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"10,0,0\"  # 10 neutral only"
echo ""

echo "# Affirmative-only with different counts"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"0,3,0\"  # 3 affirmative only"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"0,5,0\"  # 5 affirmative only"
echo ""

echo "# Skeptical-only with different counts"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"0,0,2\"  # 2 skeptical only"
echo "python build_hetero_graph_multi_tones.py $BASE_ARGS --multi_tone_counts \"0,0,4\"  # 4 skeptical only"
echo ""

echo "Generated graphs will be saved with human-readable names like:"
echo "  graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_tones_2N_1A_1S/"
echo "  graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_tones_4N_2A_2S/"
echo "  graphs_hetero/politifact/8_shot_deberta_hetero_knn_test_isolated_5_tones_3N_2A_1S/"
echo ""

echo "To run training on generated graphs:"
echo "python train_hetero_graph.py --graph_path [generated_graph_path]/graph.pt --model HAN --loss_fn ce"