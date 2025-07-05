#!/bin/bash

# Build and Train LESS4FD Script
# Builds graphs for politifact and gossipcop datasets with k-shot values 3-16
# Then immediately trains models on each built graph

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "build_less4fd_graph.py" ]; then
    print_error "build_less4fd_graph.py not found. Please run this script from the LESS4FD directory."
    exit 1
fi

if [ ! -f "train_less4fd.py" ]; then
    print_error "train_less4fd.py not found. Please run this script from the LESS4FD directory."
    exit 1
fi

# Create output directories
mkdir -p graphs_less4fd
mkdir -p results_less4fd
mkdir -p logs

# Datasets to process
DATASETS=("politifact" "gossipcop")

# K-shot values to test
K_SHOTS=(3 4 5 6 7 8 9 10 11 12 13 14 15 16)

# Default parameters (simple configuration)
EMBEDDING_TYPE="deberta"
EDGE_POLICY="knn_test_isolated"
K_NEIGHBORS=5
OUTPUT_DIR="graphs_less4fd"

# Training parameters (simple configuration)
MODEL_TYPE="HGT"
HIDDEN_CHANNELS=64
NUM_LAYERS=2
DROPOUT=0.3
LEARNING_RATE=5e-4
WEIGHT_DECAY=1e-3
EPOCHS=300
PATIENCE=30
RESULTS_DIR="results_less4fd"

# Setup logging
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
MASTER_LOG="logs/build_and_train_less4fd_${TIMESTAMP}.log"

print_status "Starting LESS4FD experiments at ${TIMESTAMP}" | tee -a "${MASTER_LOG}"

# Build and train for all combinations
for DATASET in "${DATASETS[@]}"; do
    for K_SHOT in "${K_SHOTS[@]}"; do
        echo "================================================"
        echo "----- Start building -----"
        echo "[BUILD] ${DATASET} ${EMBEDDING_TYPE} k-shot:${K_SHOT} edge_policy:${EDGE_POLICY}" | tee -a "${MASTER_LOG}"
        
        BUILD_LOG="logs/build_${DATASET}_k${K_SHOT}_${TIMESTAMP}.log"
        
        # Build graph command
        if ! python build_less4fd_graph.py \
            --dataset_name "${DATASET}" \
            --k_shot "${K_SHOT}" \
            --embedding_type "${EMBEDDING_TYPE}" \
            --edge_policy "${EDGE_POLICY}" \
            --k_neighbors "${K_NEIGHBORS}" \
            --enable_entities \
            --output_dir "${OUTPUT_DIR}" > "${BUILD_LOG}" 2>&1; then
            echo "[ERROR] Failed to build graph for ${DATASET} k-shot:${K_SHOT}" | tee -a "${MASTER_LOG}"
            continue
        fi
        
        # Extract graph path from build log
        GRAPH_PATH=$(grep 'Graph saved to' "${BUILD_LOG}" | tail -1 | awk '{print $NF}')
        if [ -z "$GRAPH_PATH" ]; then
            echo "[ERROR] Could not find graph path in build log" | tee -a "${MASTER_LOG}"
            continue
        fi
        
        echo "[DONE] Build graph ${GRAPH_PATH}" | tee -a "${MASTER_LOG}"
        echo "----- End building -----"
        
        echo "----- Start training -----"
        echo "Graph path: ${GRAPH_PATH}"
        TRAIN_LOG="logs/train_$(echo "${GRAPH_PATH}" | sed 's/\\.pt$//' | tr '/' '_').log"
        
        echo "[TRAIN] ${MODEL_TYPE} on ${GRAPH_PATH}" | tee -a "${MASTER_LOG}"
        if ! python train_less4fd.py \
            --graph_path "${GRAPH_PATH}" \
            --model_type "${MODEL_TYPE}" \
            --hidden_channels "${HIDDEN_CHANNELS}" \
            --num_layers "${NUM_LAYERS}" \
            --dropout "${DROPOUT}" \
            --learning_rate "${LEARNING_RATE}" \
            --weight_decay "${WEIGHT_DECAY}" \
            --epochs "${EPOCHS}" \
            --patience "${PATIENCE}" \
            --output_dir "${RESULTS_DIR}" > "${TRAIN_LOG}" 2>&1; then
            echo "[ERROR] Failed to train ${MODEL_TYPE} on ${GRAPH_PATH}" | tee -a "${MASTER_LOG}"
            continue
        fi
        
        echo "[DONE] ${GRAPH_PATH}" | tee -a "${MASTER_LOG}"
        echo "----- End training -----"
        echo "================================================"
        echo ""
    done
done

echo "All experiments finished at $(date +%Y-%m-%d_%H-%M-%S)" | tee -a "${MASTER_LOG}"
print_success "Script completed successfully!"
