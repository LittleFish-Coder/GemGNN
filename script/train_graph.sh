# Train graph given by the graph pt

## KDD2020
echo "Training kdd2020"
### ThresholdNN: 1 5
python train_graph.py --graph graph/kdd2020/train_4487_test_499_labeled_100_thresholdnn_1.pt --dropout
python train_graph.py --graph graph/kdd2020/train_4487_test_499_labeled_100_thresholdnn_5.pt --dropout
### KNN: 5 9
python train_graph.py --graph graph/kdd2020/train_4487_test_499_labeled_100_knn_5.pt --dropout
python train_graph.py --graph graph/kdd2020/train_4487_test_499_labeled_100_knn_9.pt --dropout

## TFG
echo "Training tfg"
### ThresholdNN: 1 5
python train_graph.py --graph graph/tfg/train_24353_test_8117_labeled_100_thresholdnn_1.pt --dropout
python train_graph.py --graph graph/tfg/train_24353_test_8117_labeled_100_thresholdnn_5.pt --dropout
### KNN: 5 9
python train_graph.py --graph graph/tfg/train_24353_test_8117_labeled_100_knn_5.pt --dropout
python train_graph.py --graph graph/tfg/train_24353_test_8117_labeled_100_knn_9.pt --dropout


## GossipCop
echo "Training gossipcop"
### ThresholdNN: 1 5
python train_graph.py --graph graph/gossipcop/train_9988_test_2672_labeled_100_thresholdnn_1.pt --dropout
python train_graph.py --graph graph/gossipcop/train_9988_test_2672_labeled_100_thresholdnn_5.pt --dropout
### KNN: 5 9
python train_graph.py --graph graph/gossipcop/train_9988_test_2672_labeled_100_knn_5.pt --dropout
python train_graph.py --graph graph/gossipcop/train_9988_test_2672_labeled_100_knn_9.pt --dropout

## PolitiFact
echo "Training politifact"
### ThresholdNN: 1 5
python train_graph.py --graph graph/politifact/train_381_test_102_labeled_100_thresholdnn_1.pt --dropout
python train_graph.py --graph graph/politifact/train_381_test_102_labeled_100_thresholdnn_5.pt --dropout
### KNN: 5 9
python train_graph.py --graph graph/politifact/train_381_test_102_labeled_100_knn_5.pt --dropout
python train_graph.py --graph graph/politifact/train_381_test_102_labeled_100_knn_9.pt --dropout