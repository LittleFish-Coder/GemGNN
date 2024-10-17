# build knn graph with k from 5 to 25
for k in {5..25};
do
    echo "k: $k"
    if ! python build_graph.py --prebuilt_graph graph/kdd2020/train_3490_val_997_test_499_labeled_100.pt --edge_policy knn --k $k; then
        echo "Error occurred with k: $k"
        exit 1
    fi
done

# build thresholdnn graph with factor from 1 to 30
for factor in {1..30}; 
do
    echo "factor: $factor"
    if ! python build_graph.py --prebuilt_graph graph/kdd2020/train_3490_val_997_test_499_labeled_100.pt --threshold_factor $factor; then
        echo "Error occurred with factor: $factor"
        exit 1
    fi
done