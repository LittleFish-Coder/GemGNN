# Train graph given by the graph pt

## TFG
echo "Training tfg"
### ThresholdNN: 1 5
python train_graph.py --graph graph/tfg/train_24353_test_8117_labeled_100_thresholdnn_1.pt --dropout
# python train_graph.py --graph graph/tfg/train_24353_test_8117_labeled_100_thresholdnn_5.pt --dropout
### KNN: 5 9
python train_graph.py --graph graph/tfg/train_24353_test_8117_labeled_100_knn_5.pt --dropout
python train_graph.py --graph graph/tfg/train_24353_test_8117_labeled_100_knn_9.pt --dropout
