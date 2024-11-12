# Train graph given by the graph pt

## PolitiFact
echo "Training politifact"
### ThresholdNN: 1 5
python train_graph.py --graph graph/politifact/train_381_test_102_labeled_100_thresholdnn_1.pt --dropout
python train_graph.py --graph graph/politifact/train_381_test_102_labeled_100_thresholdnn_5.pt --dropout
### KNN: 5 9
python train_graph.py --graph graph/politifact/train_381_test_102_labeled_100_knn_5.pt --dropout
python train_graph.py --graph graph/politifact/train_381_test_102_labeled_100_knn_9.pt --dropout