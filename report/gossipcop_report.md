# Gossipcop Dataset Analysis Report

**Generated on:** 2025-07-01 10:40:21

## Executive Summary

This report provides comprehensive analysis of heterogeneous graph neural network performance on the Gossipcop dataset, comparing two edge construction policies: **KNN (standard)** and **KNN Test-Isolated**. The analysis covers experiments across different parameter configurations, focusing on 3-16 shot learning scenarios.

## Dataset Overview

- **Dataset**: Gossipcop
- **Model Architecture**: HAN (Hierarchical Attention Network)
- **Shot Range**: 3-16 shot learning
- **Edge Policies Analyzed**: KNN, KNN Test-Isolated

### Experiment Statistics by Edge Policy

- **KNN**: 24 scenarios, 336 experiments
- **KNN_TEST_ISOLATED**: 25 scenarios, 350 experiments


## Performance Analysis by Edge Policy

### Best Performing Configurations

#### KNN Policy Top 5

| Rank | Configuration | Avg F1 | K-Neighbors | Multiview | Interactions |
|------|---------------|---------|-------------|-----------|-------------|
| 1 | deberta_hetero_knn_7_ensure_test_labeled_neighbor_... | 0.5927 | 7 | 3 | Yes |
| 2 | deberta_hetero_knn_5_ensure_test_labeled_neighbor_... | 0.5925 | 5 | 3 | Yes |
| 3 | deberta_hetero_knn_7_partial_sample_unlabeled_fact... | 0.5924 | 7 | 3 | Yes |
| 4 | deberta_hetero_knn_5_partial_sample_unlabeled_fact... | 0.5923 | 5 | 3 | Yes |
| 5 | deberta_hetero_knn_5_ensure_test_labeled_neighbor_... | 0.5904 | 5 | 0 | Yes |


**Optimal Configuration for KNN:**
- **F1 Score**: 0.5927
- **K-Neighbors**: 7
- **Multiview**: 3
- **Interactions**: Enabled
- **Test Labeled Neighbor**: Enabled
- **Dissimilar Sampling**: Disabled

#### KNN_TEST_ISOLATED Policy Top 5

| Rank | Configuration | Avg F1 | K-Neighbors | Multiview | Interactions |
|------|---------------|---------|-------------|-----------|-------------|
| 1 | deberta_hetero_knn_test_isolated_5_ensure_test_lab... | 0.5928 | 5 | 3 | Yes |
| 2 | deberta_hetero_knn_test_isolated_7_ensure_test_lab... | 0.5925 | 7 | 3 | Yes |
| 3 | deberta_hetero_knn_test_isolated_5_partial_sample_... | 0.5925 | 5 | 3 | Yes |
| 4 | deberta_hetero_knn_test_isolated_7_partial_sample_... | 0.5924 | 7 | 3 | Yes |
| 5 | deberta_hetero_knn_test_isolated_7_partial_sample_... | 0.5902 | 7 | 0 | Yes |


**Optimal Configuration for KNN_TEST_ISOLATED:**
- **F1 Score**: 0.5928
- **K-Neighbors**: 5
- **Multiview**: 3
- **Interactions**: Enabled
- **Test Labeled Neighbor**: Enabled
- **Dissimilar Sampling**: Disabled


## Edge Policy Performance Comparison

### Top 5 Configurations for Each Edge Policy

#### KNN (Standard) Policy

| Rank | Configuration | Avg F1 | K-Neighbors | Multiview | Interactions | Test Labeled |
|------|---------------|---------|-------------|-----------|--------------|-------------|
| 1 | deberta_hetero_knn_7_ensure_test_labeled... | 0.5927 | 7 | 3 | Yes | Yes |
| 2 | deberta_hetero_knn_5_ensure_test_labeled... | 0.5925 | 5 | 3 | Yes | Yes |
| 3 | deberta_hetero_knn_7_partial_sample_unla... | 0.5924 | 7 | 3 | Yes | No |
| 4 | deberta_hetero_knn_5_partial_sample_unla... | 0.5923 | 5 | 3 | Yes | No |
| 5 | deberta_hetero_knn_5_ensure_test_labeled... | 0.5904 | 5 | 0 | Yes | Yes |

#### KNN Test-Isolated Policy

| Rank | Configuration | Avg F1 | K-Neighbors | Multiview | Interactions | Test Labeled |
|------|---------------|---------|-------------|-----------|--------------|-------------|
| 1 | deberta_hetero_knn_test_isolated_5_ensur... | 0.5928 | 5 | 3 | Yes | Yes |
| 2 | deberta_hetero_knn_test_isolated_7_ensur... | 0.5925 | 7 | 3 | Yes | Yes |
| 3 | deberta_hetero_knn_test_isolated_5_parti... | 0.5925 | 5 | 3 | Yes | No |
| 4 | deberta_hetero_knn_test_isolated_7_parti... | 0.5924 | 7 | 3 | Yes | No |
| 5 | deberta_hetero_knn_test_isolated_7_parti... | 0.5902 | 7 | 0 | Yes | No |

### Performance Summary

- **Best KNN Performance**: 0.5927
- **Best KNN Test-Isolated Performance**: 0.5928
- **Performance Difference**: -0.0002 (-0.0%)

**Analysis**: Test-isolated KNN shows better or comparable performance, indicating robust generalization without test data leakage.


## K-Shot Performance Analysis by Edge Policy

### KNN Policy - Top 3 Configurations

| Configuration | Avg F1 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deberta_hetero_knn_7_ensu... | 0.5927 | 0.602 | 0.616 | 0.600 | 0.592 | 0.604 | 0.582 | 0.581 | 0.605 | 0.608 | 0.581 | 0.580 | 0.584 | 0.582 | 0.580 |
| deberta_hetero_knn_5_ensu... | 0.5925 | 0.601 | 0.614 | 0.600 | 0.594 | 0.603 | 0.583 | 0.581 | 0.604 | 0.608 | 0.581 | 0.581 | 0.582 | 0.582 | 0.582 |
| deberta_hetero_knn_7_part... | 0.5924 | 0.602 | 0.614 | 0.600 | 0.593 | 0.605 | 0.583 | 0.581 | 0.602 | 0.609 | 0.580 | 0.581 | 0.582 | 0.582 | 0.581 |

### KNN_TEST_ISOLATED Policy - Top 3 Configurations

| Configuration | Avg F1 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deberta_hetero_knn_test_i... | 0.5928 | 0.599 | 0.616 | 0.600 | 0.595 | 0.604 | 0.583 | 0.581 | 0.605 | 0.609 | 0.581 | 0.581 | 0.584 | 0.582 | 0.580 |
| deberta_hetero_knn_test_i... | 0.5925 | 0.602 | 0.614 | 0.599 | 0.594 | 0.604 | 0.583 | 0.581 | 0.602 | 0.607 | 0.581 | 0.581 | 0.584 | 0.582 | 0.581 |
| deberta_hetero_knn_test_i... | 0.5925 | 0.600 | 0.616 | 0.600 | 0.594 | 0.603 | 0.582 | 0.581 | 0.603 | 0.608 | 0.581 | 0.581 | 0.582 | 0.582 | 0.581 |


## Ablation Study Analysis

### KNN Policy Ablation Analysis

#### Interaction Component Impact

#### Multi-View Settings Impact

- **Multiview 0 (Baseline)**: 0.5852
- **Multiview 3**: 0.5908 (+0.0056)
- **Multiview 6**: 0.5787 (-0.0065)

**Best Setting**: Multiview 3 with F1 score of 0.5908

#### K-Neighbors Hyperparameter Analysis

- **K=5**: 0.5850
- **K=7**: 0.5847

**Optimal K-value**: 5 (F1: 0.5850)
**Performance Range**: 0.0002 (0.0% improvement from worst to best)

---

### KNN_TEST_ISOLATED Policy Ablation Analysis

#### Interaction Component Impact

- **With Interactions**: 0.5842
- **Without Interactions (no_interactions)**: 0.3228
- **Interaction Benefit**: 0.2614 (+81.0%)

**Conclusion**: Synthetic user interactions provide meaningful signal for fake news detection.

#### Multi-View Settings Impact

- **Multiview 0 (Baseline)**: 0.5861
- **Multiview 3**: 0.5599 (-0.0262)
- **Multiview 6**: 0.5770 (-0.0091)

**Best Setting**: Multiview 0 with F1 score of 0.5861

#### K-Neighbors Hyperparameter Analysis

- **K=5**: 0.5643
- **K=7**: 0.5839

**Optimal K-value**: 7 (F1: 0.5839)
**Performance Range**: 0.0196 (3.5% improvement from worst to best)

---


## Hyperparameter Search Analysis

### Multi-View Configuration Analysis (0, 3, 6 views)

| Edge Policy | Multiview 0 | Multiview 3 | Multiview 6 | Best Setting |
|-------------|-------------|-------------|-------------|-------------|
| knn | 0.5852 | 0.5908 | 0.5787 | 3 (0.5908) |
| knn_test_isolated | 0.5861 | 0.5599 | 0.5770 | 0 (0.5861) |

### K-Neighbors Analysis (3, 5, 7)

| Edge Policy | K=3 | K=5 | K=7 | Best Setting |
|-------------|-----|-----|-----|-------------|
| knn | 0.0000 | 0.5850 | 0.5847 | 5 (0.5850) |
| knn_test_isolated | 0.0000 | 0.5643 | 0.5839 | 7 (0.5839) |

### Hyperparameter Recommendations

#### KNN Policy

- **Recommended Multiview**: 3 views
- **Recommended K-neighbors**: 5

#### KNN_TEST_ISOLATED Policy

- **Recommended Multiview**: 0 views
- **Recommended K-neighbors**: 7


## Recommendations for Gossipcop

### Edge Policy Selection

- **Recommended**: KNN Test-Isolated policy (comparable/better performance with realistic evaluation)


### Parameter Guidelines

1. **Edge Policy Choice**:
   - Use KNN for maximum performance in batch processing scenarios
   - Use KNN Test-Isolated for realistic evaluation and deployment

2. **Hyperparameter Selection**:
   - Follow the hyperparameter recommendations above for each edge policy
   - Consider interaction components based on ablation study results

3. **Few-Shot Considerations**:
   - Both policies show consistent performance across 3-16 shot range
   - Lower shot counts may benefit from specific parameter tuning

---

*This report was generated automatically using the edge policy analysis pipeline.*
