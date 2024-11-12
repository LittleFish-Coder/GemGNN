# analyze graph given by the graph pt

## KDD2020
echo "Analyzing kdd2020"
mkdir -p analyze/kdd2020
### ThresholdNN: 1 5
python analyze.py --graph graph/kdd2020/train_4487_test_499_labeled_100_thresholdnn_1.pt > analyze/kdd2020/train_4487_test_499_labeled_100_thresholdnn_1.txt
python analyze.py --graph graph/kdd2020/train_4487_test_499_labeled_100_thresholdnn_5.pt > analyze/kdd2020/train_4487_test_499_labeled_100_thresholdnn_5.txt
### KNN: 5 9
python analyze.py --graph graph/kdd2020/train_4487_test_499_labeled_100_knn_5.pt > analyze/kdd2020/train_4487_test_499_labeled_100_knn_5.txt
python analyze.py --graph graph/kdd2020/train_4487_test_499_labeled_100_knn_9.pt > analyze/kdd2020/train_4487_test_499_labeled_100_knn_9.txt

## TFG
echo "Analyzing tfg"
mkdir -p analyze/tfg
### ThresholdNN: 1 5
python analyze.py --graph graph/tfg/train_24353_test_8117_labeled_100_thresholdnn_1.pt > analyze/tfg/train_24353_test_8117_labeled_100_thresholdnn_1.txt
python analyze.py --graph graph/tfg/train_24353_test_8117_labeled_100_thresholdnn_5.pt > analyze/tfg/train_24353_test_8117_labeled_100_thresholdnn_5.txt
### KNN: 5 9
python analyze.py --graph graph/tfg/train_24353_test_8117_labeled_100_knn_5.pt > analyze/tfg/train_24353_test_8117_labeled_100_knn_5.txt
python analyze.py --graph graph/tfg/train_24353_test_8117_labeled_100_knn_9.pt > analyze/tfg/train_24353_test_8117_labeled_100_knn_9.txt


## GossipCop
echo "Analyzing gossipcop"
mkdir -p analyze/gossipcop
### ThresholdNN: 1 5
python analyze.py --graph graph/gossipcop/train_9988_test_2672_labeled_100_thresholdnn_1.pt > analyze/gossipcop/train_9988_test_2672_labeled_100_thresholdnn_1.txt
python analyze.py --graph graph/gossipcop/train_9988_test_2672_labeled_100_thresholdnn_5.pt > analyze/gossipcop/train_9988_test_2672_labeled_100_thresholdnn_5.txt
### KNN: 5 9
python analyze.py --graph graph/gossipcop/train_9988_test_2672_labeled_100_knn_5.pt > analyze/gossipcop/train_9988_test_2672_labeled_100_knn_5.txt
python analyze.py --graph graph/gossipcop/train_9988_test_2672_labeled_100_knn_9.pt > analyze/gossipcop/train_9988_test_2672_labeled_100_knn_9.txt

## PolitiFact
echo "Analyzing politifact"
mkdir -p analyze/politifact
### ThresholdNN: 1 5
python analyze.py --graph graph/politifact/train_381_test_102_labeled_100_thresholdnn_1.pt > analyze/politifact/train_381_test_102_labeled_100_thresholdnn_1.txt
python analyze.py --graph graph/politifact/train_381_test_102_labeled_100_thresholdnn_5.pt > analyze/politifact/train_381_test_102_labeled_100_thresholdnn_5.txt
### KNN: 5 9
python analyze.py --graph graph/politifact/train_381_test_102_labeled_100_knn_5.pt > analyze/politifact/train_381_test_102_labeled_100_knn_5.txt
python analyze.py --graph graph/politifact/train_381_test_102_labeled_100_knn_9.pt > analyze/politifact/train_381_test_102_labeled_100_knn_9.txt