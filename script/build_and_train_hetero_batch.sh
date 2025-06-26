#!/bin/bash

K_SHOTS=(3 4 5 6 7 8 9 10 11 12 13 14 15 16)
DATASETS=("politifact" "gossipcop")
EMBEDDINGS=("deberta")
EDGE_POLICIES=("knn_test_isolated")
MULTI_VIEWS=(3)
MODELS=("HAN")
K_NEIGHBORS=5
DATASET_CACHE_DIR="dataset_batch"

GRAPH_OUTPUT_BASE_DIR="graphs_hetero_batch"
RESULTS_OUTPUT_BASE_DIR="results_hetero_batch"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${GRAPH_OUTPUT_BASE_DIR}" "${RESULTS_OUTPUT_BASE_DIR}"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
# Save PID to file
echo $$ > "${LOG_DIR}/process_id_${TIMESTAMP}.pid"
MASTER_LOG="${LOG_DIR}/build_and_train_hetero_batch_${TIMESTAMP}.log"

echo "Start grid search at ${TIMESTAMP}" | tee -a "${MASTER_LOG}"

for DATASET in "${DATASETS[@]}"; do
    for EMB in "${EMBEDDINGS[@]}"; do
        for EDGE_POLICY in "${EDGE_POLICIES[@]}"; do
            for K_SHOT in "${K_SHOTS[@]}"; do
                for MULTI_VIEW in "${MULTI_VIEWS[@]}"; do
                    GRAPH_NAME="${K_SHOT}_shot_${EMB}_hetero_batch_${EDGE_POLICY}_${K_NEIGHBORS}_multiview_${MULTI_VIEW}"
                    BUILD_LOG="${LOG_DIR}/build_${DATASET}_${GRAPH_NAME}_${TIMESTAMP}.log"

                    echo "================================================"
                    echo "----- Start building -----"
                    echo "[BUILD] ${DATASET} ${EMB} k-shot:${K_SHOT} edge_policy:${EDGE_POLICY} multiview:${MULTI_VIEW}" | tee -a "${MASTER_LOG}"

                    if ! python build_hetero_graph_batch.py \
                        --dataset_name "${DATASET}" \
                        --k_shot "${K_SHOT}" \
                        --embedding_type "${EMB}" \
                        --edge_policy "${EDGE_POLICY}" \
                        --k_neighbors "${K_NEIGHBORS}" \
                        --multi_view "${MULTI_VIEW}" \
                        --partial_unlabeled \
                        --dataset_cache_dir "${DATASET_CACHE_DIR}" \
                        --output_dir "${GRAPH_OUTPUT_BASE_DIR}" > "${BUILD_LOG}" 2>&1; then
                        echo "[ERROR] Failed to build graph for ${DATASET} ${EMB} k-shot:${K_SHOT} edge_policy:${EDGE_POLICY} multiview:${MULTI_VIEW}" | tee -a "${MASTER_LOG}"
                        continue
                    fi
                    echo "----- End building -----"

                    # GRAPH_PATH=$(grep 'Graph saved to' "${BUILD_LOG}" | tail -1 | awk '{print $NF}')
                    # if [ -z "$GRAPH_PATH" ]; then
                    #     echo "[ERROR] Could not find graph path in build log" | tee -a "${MASTER_LOG}"
                    #     continue
                    # fi
                    # GRAPH_FOLDER=$(dirname "${GRAPH_PATH}")
                    # echo "[Done] Build graph folder ${GRAPH_FOLDER}"

                    # echo "----- Start training -----"
                    # echo "Graph folder: ${GRAPH_FOLDER}"
                    # TRAIN_LOG="${LOG_DIR}/train_$(echo "${GRAPH_FOLDER}" | sed 's/\\.pt$//' | tr '/' '_').log"
                    # for MODEL in "${MODELS[@]}"; do
                    #     echo "[TRAIN] ${MODEL} on ${GRAPH_FOLDER}" | tee -a "${MASTER_LOG}"
                    #     if ! python train_hetero_graph_batch.py \
                    #         --graph_path "${GRAPH_FOLDER}" \
                    #         --model "${MODEL}" \
                    #         > "${TRAIN_LOG}" 2>&1; then
                    #         echo "[ERROR] Failed to train ${MODEL} on ${GRAPH_FOLDER}" | tee -a "${MASTER_LOG}"
                    #         continue
                    #     fi
                    # done

                    # echo "[DONE] ${GRAPH_FOLDER}" | tee -a "${MASTER_LOG}"
                    # echo "----- End training -----"
                    echo "================================================"
                    echo ""
                done
            done
        done
    done
done
echo "Graph building finished at $(date +%Y-%m-%d_%H-%M-%S)" | tee -a "${MASTER_LOG}" 