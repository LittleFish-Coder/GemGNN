# Train graph given by the graph pt

## KDD2020
echo "Training kdd2020"
### ThresholdNN: 1 5
python train_graph.py --graph graph/kdd2020/train_4487_test_499_labeled_100_thresholdnn_1.pt --dropout
python train_graph.py --graph graph/kdd2020/train_4487_test_499_labeled_100_thresholdnn_5.pt --dropout
### KNN: 5 9
python train_graph.py --graph graph/kdd2020/train_4487_test_499_labeled_100_knn_5.pt --dropout
python train_graph.py --graph graph/kdd2020/train_4487_test_499_labeled_100_knn_9.pt --dropout