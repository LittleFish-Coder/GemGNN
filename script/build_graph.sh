# Build graph given by the dataset and edge policy

## KDD2020
### ThresholdNN: 1 5
python build_graph.py --dataset_name kdd2020 --threshold_factor 1 
python build_graph.py --dataset_name kdd2020 --threshold_factor 5
### KNN: 5 9
python build_graph.py --dataset_name kdd2020 --edge_policy knn --k 5
python build_graph.py --dataset_name kdd2020 --edge_policy knn --k 9

## TFG
### ThresholdNN: 1 5
python build_graph.py --dataset_name tfg --threshold_factor 1
python build_graph.py --dataset_name tfg --threshold_factor 5
### KNN: 5 9
python build_graph.py --dataset_name tfg --edge_policy knn --k 5
python build_graph.py --dataset_name tfg --edge_policy knn --k 9

## GossipCop
### ThresholdNN: 1 5
python build_graph.py --dataset_name gossipcop --threshold_factor 1
python build_graph.py --dataset_name gossipcop --threshold_factor 5
### KNN: 5 9
python build_graph.py --dataset_name gossipcop --edge_policy knn --k 5
python build_graph.py --dataset_name gossipcop --edge_policy knn --k 9

## PolitiFact
### ThresholdNN: 1 5
python build_graph.py --dataset_name politifact --threshold_factor 1
python build_graph.py --dataset_name politifact --threshold_factor 5
### KNN: 5 9
python build_graph.py --dataset_name politifact --edge_policy knn --k 5
python build_graph.py --dataset_name politifact --edge_policy knn --k 9