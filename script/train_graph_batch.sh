for k in {5..25};
do
    echo "knn: $k"
    if ! python train_graph.py --graph ./graph/kdd2020/train_full_val_full_test_full_labeled_100_knn_$k.pt; then
        echo "Error occurred with k: $k"
        exit 1
    fi
done

for factor in {1..30};
do
    echo "factor: $factor"
    if ! python train_graph.py --graph ./graph/kdd2020/train_full_val_full_test_full_labeled_100_thresholdnn_$factor.pt; then
        echo "Error occurred with factor: $factor"
        exit 1
    fi
done
