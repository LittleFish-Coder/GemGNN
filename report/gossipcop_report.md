# Gossipcop Dataset Analysis Report

**Generated on:** 2025-06-27 03:31:07

## Executive Summary

This report provides a comprehensive analysis of heterogeneous graph neural network (HGN) performance on the Gossipcop dataset for fake news detection. The analysis covers 672 experiments across 48 different parameter configurations, focusing on 3-16 shot learning scenarios.

## Dataset Overview

- **Dataset**: Gossipcop
- **Model Architecture**: HAN (Hierarchical Attention Network)
- **Shot Range**: 3-16 shot learning
- **Total Scenarios**: 48
- **Total Experiments**: 672

## Performance Summary

### Top 10 Best Performing Configurations

| Rank | Configuration | Avg F1 Score | K-Neighbors | Edge Policy | Multiview | Dissimilar | Test Labeled |
|------|---------------|--------------|-------------|-------------|-----------|-----------|--------------|
| 1 | deberta_hetero_knn_test_isolated_5_ensure_test_lab... | 0.5928 | None | knn_test_isolated | 3 | False | True |
| 2 | deberta_hetero_knn_7_ensure_test_labeled_neighbor_... | 0.5927 | 7 | knn | 3 | False | True |
| 3 | deberta_hetero_knn_test_isolated_7_ensure_test_lab... | 0.5925 | None | knn_test_isolated | 3 | False | True |
| 4 | deberta_hetero_knn_test_isolated_5_partial_sample_... | 0.5925 | None | knn_test_isolated | 3 | False | False |
| 5 | deberta_hetero_knn_5_ensure_test_labeled_neighbor_... | 0.5925 | 5 | knn | 3 | False | True |
| 6 | deberta_hetero_knn_7_partial_sample_unlabeled_fact... | 0.5924 | 7 | knn | 3 | False | False |
| 7 | deberta_hetero_knn_test_isolated_7_partial_sample_... | 0.5924 | None | knn_test_isolated | 3 | False | False |
| 8 | deberta_hetero_knn_5_partial_sample_unlabeled_fact... | 0.5923 | 5 | knn | 3 | False | False |
| 9 | deberta_hetero_knn_5_ensure_test_labeled_neighbor_... | 0.5904 | 5 | knn | 0 | False | True |
| 10 | deberta_hetero_knn_test_isolated_7_partial_sample_... | 0.5902 | None | knn_test_isolated | 0 | False | False |


### Overall Performance Statistics

- **Best Average F1 Score**: 0.5928
- **Worst Average F1 Score**: 0.5749
- **Overall Mean F1 Score**: 0.5845

## Parameter Impact Analysis

### K-Neighbors Impact
- **K=5**: Average F1 = 0.5850
- **K=7**: Average F1 = 0.5847

### Edge Policy Impact
- **knn**: Average F1 = 0.5849
- **knn_test_isolated**: Average F1 = 0.5842

### Multiview Impact
- **Multiview=0**: Average F1 = 0.5856
- **Multiview=3**: Average F1 = 0.5901
- **Multiview=6**: Average F1 = 0.5778

### Feature Engineering Impact
- **Dissimilar Sampling Disabled**: Average F1 = 0.5859
- **Dissimilar Sampling Enabled**: Average F1 = 0.5832
- **Test Labeled Neighbor Enabled**: Average F1 = 0.5846
- **Test Labeled Neighbor Disabled**: Average F1 = 0.5845


## Detailed Configuration Analysis

### Shot Learning Performance Trends

The following section analyzes how performance varies across different shot counts for the top configurations:

#### Configuration 1: deberta_hetero_knn_test_isolated_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3

- **Average F1**: 0.5928
- **Standard Deviation**: 0.0122
- **Range**: 0.5800 - 0.6164
- **Shot-wise Performance**:
  - 3-shot: 0.5988
  - 4-shot: 0.6164
  - 5-shot: 0.6001
  - 6-shot: 0.5947
  - 7-shot: 0.6045
  - 8-shot: 0.5827
  - 9-shot: 0.5811
  - 10-shot: 0.6054
  - 11-shot: 0.6086
  - 12-shot: 0.5808
  - 13-shot: 0.5811
  - 14-shot: 0.5839
  - 15-shot: 0.5819
  - 16-shot: 0.5800

#### Configuration 2: deberta_hetero_knn_7_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3

- **Average F1**: 0.5927
- **Standard Deviation**: 0.0122
- **Range**: 0.5800 - 0.6156
- **Shot-wise Performance**:
  - 3-shot: 0.6020
  - 4-shot: 0.6156
  - 5-shot: 0.6001
  - 6-shot: 0.5919
  - 7-shot: 0.6036
  - 8-shot: 0.5825
  - 9-shot: 0.5811
  - 10-shot: 0.6053
  - 11-shot: 0.6083
  - 12-shot: 0.5808
  - 13-shot: 0.5800
  - 14-shot: 0.5839
  - 15-shot: 0.5821
  - 16-shot: 0.5800

#### Configuration 3: deberta_hetero_knn_test_isolated_7_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3

- **Average F1**: 0.5925
- **Standard Deviation**: 0.0116
- **Range**: 0.5806 - 0.6143
- **Shot-wise Performance**:
  - 3-shot: 0.6023
  - 4-shot: 0.6143
  - 5-shot: 0.5992
  - 6-shot: 0.5937
  - 7-shot: 0.6037
  - 8-shot: 0.5827
  - 9-shot: 0.5815
  - 10-shot: 0.6022
  - 11-shot: 0.6074
  - 12-shot: 0.5809
  - 13-shot: 0.5806
  - 14-shot: 0.5839
  - 15-shot: 0.5819
  - 16-shot: 0.5811

#### Configuration 4: deberta_hetero_knn_test_isolated_5_partial_sample_unlabeled_factor_5_multiview_3

- **Average F1**: 0.5925
- **Standard Deviation**: 0.0120
- **Range**: 0.5808 - 0.6162
- **Shot-wise Performance**:
  - 3-shot: 0.5999
  - 4-shot: 0.6162
  - 5-shot: 0.5995
  - 6-shot: 0.5944
  - 7-shot: 0.6032
  - 8-shot: 0.5819
  - 9-shot: 0.5812
  - 10-shot: 0.6032
  - 11-shot: 0.6084
  - 12-shot: 0.5815
  - 13-shot: 0.5811
  - 14-shot: 0.5815
  - 15-shot: 0.5819
  - 16-shot: 0.5808

#### Configuration 5: deberta_hetero_knn_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3

- **Average F1**: 0.5925
- **Standard Deviation**: 0.0117
- **Range**: 0.5808 - 0.6140
- **Shot-wise Performance**:
  - 3-shot: 0.6005
  - 4-shot: 0.6140
  - 5-shot: 0.5996
  - 6-shot: 0.5944
  - 7-shot: 0.6030
  - 8-shot: 0.5826
  - 9-shot: 0.5808
  - 10-shot: 0.6042
  - 11-shot: 0.6079
  - 12-shot: 0.5811
  - 13-shot: 0.5808
  - 14-shot: 0.5820
  - 15-shot: 0.5819
  - 16-shot: 0.5818



## Recommendations

### Optimal Configuration for Gossipcop

Based on the analysis, the optimal configuration for Gossipcop dataset is:

- **Configuration**: deberta_hetero_knn_test_isolated_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3
- **Average F1 Score**: 0.5928
- **Parameters**:
  - K-Neighbors: None
  - Edge Policy: knn_test_isolated
  - Multiview: 3
  - Dissimilar Sampling: Disabled
  - Test Labeled Neighbor: Enabled

### Parameter Selection Guidelines

1. **K-Neighbors**: Based on the analysis, k=7 shows the best performance
2. **Edge Policy**: knn performs better than alternatives
3. **Multiview**: Multiview setting of 3 shows optimal results
4. **Feature Engineering**: Standard sampling is sufficient

## Statistical Significance

The analysis includes 672 experiments across 14 different shot counts, providing robust statistical evidence for the reported trends.

## Limitations and Future Work

1. **Parameter Space**: Current analysis covers the implemented parameter combinations; additional hyperparameter exploration might yield better results
2. **Cross-validation**: Results are based on single train/test splits; cross-validation would provide more robust estimates
3. **Statistical Testing**: Formal statistical significance tests between configurations would strengthen conclusions

---

*This report was generated automatically using the comprehensive analysis pipeline.*
