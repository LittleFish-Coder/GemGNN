#!/bin/bash

# Build and Train LESS4FD Script
# Builds graphs for politifact and gossipcop datasets with k-shot values 3-16
# Then immediately trains models on each built graph

set -e  # Exit on any error

# Create output directories
mkdir -p results_genfend
mkdir -p logs

# Datasets to process
DATASETS=("politifact" "gossipcop")

# K-shot values to test
K_SHOTS=(3 4 5 6 7 8 9 10 11 12 13 14 15 16)

# Default parameters (simple configuration)
EMBEDDING_TYPE="deberta"

# Training parameters (simple configuration)
RESULTS_DIR="results_genfend"

# Setup logging
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
MASTER_LOG="logs/build_and_train_genfend_${TIMESTAMP}.log"

print_status "Starting GenFEND experiments at ${TIMESTAMP}" | tee -a "${MASTER_LOG}"

# Build and train for all combinations
for DATASET in "${DATASETS[@]}"; do
    for K_SHOT in "${K_SHOTS[@]}"; do
        echo "================================================"
        echo "----- Start building -----"
        echo "[BUILD] ${DATASET} ${EMBEDDING_TYPE} k-shot:${K_SHOT}" | tee -a "${MASTER_LOG}"
        
        BUILD_LOG="logs/build_${DATASET}_k${K_SHOT}_${TIMESTAMP}.log"
        
        # Build graph command
        if ! python build.py \
            --dataset_name "${DATASET}" \
            --k_shot "${K_SHOT}" \
            --output_dir "${RESULTS_DIR}" > "${BUILD_LOG}" 2>&1; then
            echo "[ERROR] Failed to build graph for ${DATASET} k-shot:${K_SHOT}" | tee -a "${MASTER_LOG}"
            continue
        fi
        
        # Extract graph path from build log
        DATA_PATH=$(grep 'Data saved to' "${BUILD_LOG}" | tail -1 | awk '{print $NF}')
        if [ -z "$DATA_PATH" ]; then
            echo "[ERROR] Could not find data path in build log" | tee -a "${MASTER_LOG}"
            continue
        fi
        
        echo "[DONE] Build data ${DATA_PATH}" | tee -a "${MASTER_LOG}"
        echo "----- End building -----"
        
        echo "----- Start training -----"
        echo "Data path: ${DATA_PATH}"
        TRAIN_LOG="logs/train_$(echo "${DATA_PATH}" | sed 's/\\.pt$//' | tr '/' '_').log"
        
        echo "[TRAIN] ${DATASET} ${EMBEDDING_TYPE} k-shot:${K_SHOT}" | tee -a "${MASTER_LOG}"
        if ! python train.py \
            --data_path "${DATA_PATH}" \
            --output_dir "${RESULTS_DIR}" > "${TRAIN_LOG}" 2>&1; then
            echo "[ERROR] Failed to train ${DATASET} ${EMBEDDING_TYPE} k-shot:${K_SHOT}" | tee -a "${MASTER_LOG}"
            continue
        fi
        
        echo "[DONE] ${DATA_PATH}" | tee -a "${MASTER_LOG}"
        echo "----- End training -----"
        echo "================================================"
        echo ""
    done
done

echo "All experiments finished at $(date +%Y-%m-%d_%H-%M-%S)" | tee -a "${MASTER_LOG}"
print_status "Script completed successfully!"
