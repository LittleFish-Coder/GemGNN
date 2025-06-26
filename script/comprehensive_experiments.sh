#!/bin/bash

# Comprehensive experiment script for fake news detection
# Covers all parameter combinations specified

K_SHOTS=(3 4 5 6 7 8 9 10 11 12 13 14 15 16)
K_NEIGHBORS=(3 5 7)
DATASETS=("politifact" "gossipcop")
EMBEDDINGS=("deberta")  # "roberta"
EDGE_POLICIES=("knn_test_isolated" "knn")
MULTI_VIEWS=(0 3 6)
MODELS=("HAN")
DATASET_CACHE_DIR="dataset_single"

GRAPH_OUTPUT_BASE_DIR="graphs_hetero"
RESULTS_OUTPUT_BASE_DIR="results_hetero"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${GRAPH_OUTPUT_BASE_DIR}" "${RESULTS_OUTPUT_BASE_DIR}"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
# Save PID to file
echo $$ > "${LOG_DIR}/process_id_comprehensive_${TIMESTAMP}.log"
MASTER_LOG="${LOG_DIR}/comprehensive_experiments_${TIMESTAMP}.log"

echo "Start comprehensive grid search at ${TIMESTAMP}" | tee -a "${MASTER_LOG}"
echo "Total parameter combinations to test:" | tee -a "${MASTER_LOG}"
echo "- K-shots: ${#K_SHOTS[@]}" | tee -a "${MASTER_LOG}"
echo "- K-neighbors: ${#K_NEIGHBORS[@]}" | tee -a "${MASTER_LOG}"
echo "- Datasets: ${#DATASETS[@]}" | tee -a "${MASTER_LOG}"
echo "- Embeddings: ${#EMBEDDINGS[@]}" | tee -a "${MASTER_LOG}"
echo "- Edge policies: ${#EDGE_POLICIES[@]}" | tee -a "${MASTER_LOG}"
echo "- Multi-views: ${#MULTI_VIEWS[@]}" | tee -a "${MASTER_LOG}"
echo "- Enable dissimilar: 2 (with/without)" | tee -a "${MASTER_LOG}"
echo "- Ensure test labeled neighbor: 2 (with/without)" | tee -a "${MASTER_LOG}"

TOTAL_EXPERIMENTS=$((${#K_SHOTS[@]} * ${#K_NEIGHBORS[@]} * ${#DATASETS[@]} * ${#EMBEDDINGS[@]} * ${#EDGE_POLICIES[@]} * ${#MULTI_VIEWS[@]} * 2 * 2))
echo "Total experiments: ${TOTAL_EXPERIMENTS}" | tee -a "${MASTER_LOG}"
echo "================================================" | tee -a "${MASTER_LOG}"

EXPERIMENT_COUNT=0

for DATASET in "${DATASETS[@]}"; do
    for EMB in "${EMBEDDINGS[@]}"; do
        for EDGE_POLICY in "${EDGE_POLICIES[@]}"; do
            for K_NEIGHBORS in "${K_NEIGHBORS[@]}"; do
                for K_SHOT in "${K_SHOTS[@]}"; do
                    for MULTI_VIEW in "${MULTI_VIEWS[@]}"; do
                        # Test with/without enable_dissimilar
                        for ENABLE_DISSIMILAR in "" "--enable_dissimilar"; do
                            # Test with/without ensure_test_labeled_neighbor
                            for ENSURE_TEST_LABELED in "" "--ensure_test_labeled_neighbor"; do
                                EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
                                
                                # Build graph name
                                GRAPH_NAME="${K_SHOT}_shot_${EMB}_hetero_${EDGE_POLICY}_${K_NEIGHBORS}"
                                
                                # Add multiview suffix
                                if [ "${MULTI_VIEW}" -gt 0 ]; then
                                    GRAPH_NAME="${GRAPH_NAME}_multiview_${MULTI_VIEW}"
                                fi
                                
                                # Add dissimilar suffix
                                if [ -n "${ENABLE_DISSIMILAR}" ]; then
                                    GRAPH_NAME="${GRAPH_NAME}_dissimilar"
                                fi
                                
                                # Add ensure test labeled neighbor suffix
                                if [ -n "${ENSURE_TEST_LABELED}" ]; then
                                    GRAPH_NAME="${GRAPH_NAME}_ensure_test_labeled_neighbor"
                                fi
                                
                                # Add partial unlabeled (always enabled)
                                GRAPH_NAME="${GRAPH_NAME}_partial_sample_unlabeled_factor_5"
                                
                                BUILD_LOG="${LOG_DIR}/build_${DATASET}_${GRAPH_NAME}_${TIMESTAMP}.log"

                                echo "================================================"
                                echo "Experiment ${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS}"
                                echo "----- Start building -----"
                                echo "[BUILD] ${DATASET} ${EMB} k-shot:${K_SHOT} k-neighbors:${K_NEIGHBORS} edge_policy:${EDGE_POLICY} multiview:${MULTI_VIEW}" | tee -a "${MASTER_LOG}"
                                echo "        dissimilar:${ENABLE_DISSIMILAR:-none} ensure_test_labeled:${ENSURE_TEST_LABELED:-none}" | tee -a "${MASTER_LOG}"

                                # Build the command
                                BUILD_CMD="python build_hetero_graph.py \
                                    --dataset_name \"${DATASET}\" \
                                    --k_shot \"${K_SHOT}\" \
                                    --embedding_type \"${EMB}\" \
                                    --edge_policy \"${EDGE_POLICY}\" \
                                    --k_neighbors \"${K_NEIGHBORS}\" \
                                    --multi_view \"${MULTI_VIEW}\" \
                                    --partial_unlabeled \
                                    --sample_unlabeled_factor 5 \
                                    --dataset_cache_dir \"${DATASET_CACHE_DIR}\" \
                                    --output_dir \"${GRAPH_OUTPUT_BASE_DIR}\""

                                # Add optional flags
                                if [ -n "${ENABLE_DISSIMILAR}" ]; then
                                    BUILD_CMD="${BUILD_CMD} ${ENABLE_DISSIMILAR}"
                                fi
                                
                                if [ -n "${ENSURE_TEST_LABELED}" ]; then
                                    BUILD_CMD="${BUILD_CMD} ${ENSURE_TEST_LABELED}"
                                fi

                                # Execute build command
                                if ! eval "${BUILD_CMD}" > "${BUILD_LOG}" 2>&1; then
                                    echo "[ERROR] Failed to build graph for experiment ${EXPERIMENT_COUNT}" | tee -a "${MASTER_LOG}"
                                    echo "        Command: ${BUILD_CMD}" | tee -a "${MASTER_LOG}"
                                    continue
                                fi

                                # Extract graph path from log
                                GRAPH_PATH=$(grep 'Graph saved to' "${BUILD_LOG}" | tail -1 | awk '{print $NF}')
                                echo "[Done] Build graph ${GRAPH_PATH}"
                                if [ -z "$GRAPH_PATH" ]; then
                                    echo "[ERROR] Could not find graph path in build log for experiment ${EXPERIMENT_COUNT}" | tee -a "${MASTER_LOG}"
                                    continue
                                fi
                                echo "----- End building -----"

                                echo "----- Start training -----"
                                echo "Graph path: ${GRAPH_PATH}"
                                TRAIN_LOG="${LOG_DIR}/train_$(echo "${GRAPH_PATH}" | sed 's/\.pt$//' | tr '/' '_')_${TIMESTAMP}.log"
                                
                                for MODEL in "${MODELS[@]}"; do
                                    echo "[TRAIN] ${MODEL} on ${GRAPH_PATH}" | tee -a "${MASTER_LOG}"
                                    if ! python train_hetero_graph.py \
                                        --graph_path "${GRAPH_PATH}" \
                                        --model "${MODEL}" \
                                        --loss_fn "ce" \
                                        > "${TRAIN_LOG}" 2>&1; then
                                        echo "[ERROR] Failed to train ${MODEL} on experiment ${EXPERIMENT_COUNT}" | tee -a "${MASTER_LOG}"
                                        continue
                                    fi
                                done

                                echo "[DONE] Experiment ${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS}: ${GRAPH_PATH}" | tee -a "${MASTER_LOG}"
                                echo "----- End training -----"
                                echo "================================================"
                                echo ""
                                
                                # Progress update every 10 experiments
                                if [ $((EXPERIMENT_COUNT % 10)) -eq 0 ]; then
                                    echo "[PROGRESS] Completed ${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS} experiments" | tee -a "${MASTER_LOG}"
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Comprehensive experiments finished at $(date +%Y-%m-%d_%H-%M-%S)" | tee -a "${MASTER_LOG}"
echo "Total experiments completed: ${EXPERIMENT_COUNT}" | tee -a "${MASTER_LOG}"