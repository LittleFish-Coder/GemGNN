# Politifact Dataset Analysis Report

**Generated on:** 2025-06-28 08:18:15

## Executive Summary

This report provides a comprehensive analysis of heterogeneous graph neural network (HGN) performance on the Politifact dataset for fake news detection. The analysis covers 854 experiments across 61 different parameter configurations, focusing on 3-16 shot learning scenarios.

## Dataset Overview

- **Dataset**: Politifact
- **Model Architecture**: HAN (Hierarchical Attention Network)
- **Shot Range**: 3-16 shot learning
- **Total Scenarios**: 61
- **Total Experiments**: 854

## Performance Summary

### Top 10 Best Performing Configurations

| Rank | Configuration | Avg F1 Score | K-Neighbors | Edge Policy | Multiview | Dissimilar | Test Labeled |
|------|---------------|--------------|-------------|-------------|-----------|-----------|--------------|
| 1 | deberta_hetero_knn_7_ensure_test_labeled_neighbor_... | 0.7930 | 7 | knn | 3 | False | True |
| 2 | deberta_hetero_knn_5_ensure_test_labeled_neighbor_... | 0.7930 | 5 | knn | 3 | False | True |
| 3 | deberta_hetero_knn_test_isolated_5_partial_sample_... | 0.7930 | None | knn_test_isolated | 3 | False | False |
| 4 | deberta_hetero_knn_5_partial_sample_unlabeled_fact... | 0.7924 | 5 | knn | 3 | False | False |
| 5 | deberta_hetero_knn_7_partial_sample_unlabeled_fact... | 0.7920 | 7 | knn | 3 | False | False |
| 6 | deberta_hetero_knn_test_isolated_7_ensure_test_lab... | 0.7913 | None | knn_test_isolated | 3 | False | True |
| 7 | deberta_hetero_knn_test_isolated_7_partial_sample_... | 0.7913 | None | knn_test_isolated | 3 | False | False |
| 8 | deberta_hetero_knn_test_isolated_3_ensure_test_lab... | 0.7907 | None | knn_test_isolated | 3 | False | True |
| 9 | deberta_hetero_knn_test_isolated_5_ensure_test_lab... | 0.7907 | None | knn_test_isolated | 3 | False | True |
| 10 | deberta_hetero_knn_test_isolated_3_partial_sample_... | 0.7907 | None | knn_test_isolated | 3 | False | False |


### Overall Performance Statistics

- **Best Average F1 Score**: 0.7930
- **Worst Average F1 Score**: 0.5951
- **Overall Mean F1 Score**: 0.7780

## Parameter Impact Analysis

### K-Neighbors Impact
- **K=5**: Average F1 = 0.7819
- **K=7**: Average F1 = 0.7812

### Edge Policy Impact
- **knn**: Average F1 = 0.7815
- **knn_test_isolated**: Average F1 = 0.7756

### Multiview Impact
- **Multiview=0**: Average F1 = 0.7806
- **Multiview=3**: Average F1 = 0.7788
- **Multiview=6**: Average F1 = 0.7744

### Feature Engineering Impact
- **Dissimilar Sampling Disabled**: Average F1 = 0.7762
- **Dissimilar Sampling Enabled**: Average F1 = 0.7798
- **Test Labeled Neighbor Enabled**: Average F1 = 0.7808
- **Test Labeled Neighbor Disabled**: Average F1 = 0.7752


## Detailed Configuration Analysis

### Shot Learning Performance Trends

The following section analyzes how performance varies across different shot counts for the top configurations:

#### Configuration 1: deberta_hetero_knn_7_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3

- **Average F1**: 0.7930
- **Standard Deviation**: 0.0515
- **Range**: 0.7019 - 0.8610
- **Shot-wise Performance**:
  - 3-shot: 0.7077
  - 4-shot: 0.7778
  - 5-shot: 0.7019
  - 6-shot: 0.7077
  - 7-shot: 0.7931
  - 8-shot: 0.8382
  - 9-shot: 0.8480
  - 10-shot: 0.8610
  - 11-shot: 0.8480
  - 12-shot: 0.8174
  - 13-shot: 0.8174
  - 14-shot: 0.7915
  - 15-shot: 0.7875
  - 16-shot: 0.8048

#### Configuration 2: deberta_hetero_knn_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3

- **Average F1**: 0.7930
- **Standard Deviation**: 0.0515
- **Range**: 0.7019 - 0.8610
- **Shot-wise Performance**:
  - 3-shot: 0.7077
  - 4-shot: 0.7778
  - 5-shot: 0.7019
  - 6-shot: 0.7077
  - 7-shot: 0.7931
  - 8-shot: 0.8382
  - 9-shot: 0.8480
  - 10-shot: 0.8610
  - 11-shot: 0.8480
  - 12-shot: 0.8174
  - 13-shot: 0.8174
  - 14-shot: 0.7915
  - 15-shot: 0.7875
  - 16-shot: 0.8048

#### Configuration 3: deberta_hetero_knn_test_isolated_5_partial_sample_unlabeled_factor_5_multiview_3

- **Average F1**: 0.7930
- **Standard Deviation**: 0.0515
- **Range**: 0.7019 - 0.8610
- **Shot-wise Performance**:
  - 3-shot: 0.7077
  - 4-shot: 0.7778
  - 5-shot: 0.7019
  - 6-shot: 0.7077
  - 7-shot: 0.7931
  - 8-shot: 0.8382
  - 9-shot: 0.8480
  - 10-shot: 0.8610
  - 11-shot: 0.8480
  - 12-shot: 0.8174
  - 13-shot: 0.8174
  - 14-shot: 0.7915
  - 15-shot: 0.7875
  - 16-shot: 0.8048

#### Configuration 4: deberta_hetero_knn_5_partial_sample_unlabeled_factor_5_multiview_3

- **Average F1**: 0.7924
- **Standard Deviation**: 0.0526
- **Range**: 0.6988 - 0.8610
- **Shot-wise Performance**:
  - 3-shot: 0.6988
  - 4-shot: 0.7778
  - 5-shot: 0.7019
  - 6-shot: 0.7077
  - 7-shot: 0.7931
  - 8-shot: 0.8382
  - 9-shot: 0.8480
  - 10-shot: 0.8610
  - 11-shot: 0.8480
  - 12-shot: 0.8174
  - 13-shot: 0.8174
  - 14-shot: 0.7915
  - 15-shot: 0.7875
  - 16-shot: 0.8048

#### Configuration 5: deberta_hetero_knn_7_partial_sample_unlabeled_factor_5_multiview_3

- **Average F1**: 0.7920
- **Standard Deviation**: 0.0506
- **Range**: 0.7019 - 0.8610
- **Shot-wise Performance**:
  - 3-shot: 0.7077
  - 4-shot: 0.7778
  - 5-shot: 0.7019
  - 6-shot: 0.7077
  - 7-shot: 0.7931
  - 8-shot: 0.8382
  - 9-shot: 0.8347
  - 10-shot: 0.8610
  - 11-shot: 0.8480
  - 12-shot: 0.8174
  - 13-shot: 0.8174
  - 14-shot: 0.7915
  - 15-shot: 0.7875
  - 16-shot: 0.8048



## Recommendations

### Optimal Configuration for Politifact

Based on the analysis, the optimal configuration for Politifact dataset is:

- **Configuration**: deberta_hetero_knn_7_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3
- **Average F1 Score**: 0.7930
- **Parameters**:
  - K-Neighbors: 7
  - Edge Policy: knn
  - Multiview: 3
  - Dissimilar Sampling: Disabled
  - Test Labeled Neighbor: Enabled

### Parameter Selection Guidelines

1. **K-Neighbors**: Based on the analysis, k=7 shows the best performance
2. **Edge Policy**: knn performs better than alternatives
3. **Multiview**: Multiview setting of 0 shows optimal results
4. **Feature Engineering**: Dissimilar sampling shows positive impact

## Statistical Significance

The analysis includes 854 experiments across 14 different shot counts, providing robust statistical evidence for the reported trends.

## Limitations and Future Work

1. **Parameter Space**: Current analysis covers the implemented parameter combinations; additional hyperparameter exploration might yield better results
2. **Cross-validation**: Results are based on single train/test splits; cross-validation would provide more robust estimates
3. **Statistical Testing**: Formal statistical significance tests between configurations would strengthen conclusions

---

*This report was generated automatically using the comprehensive analysis pipeline.*
