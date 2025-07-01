# Politifact Dataset Analysis Report

**Generated on:** 2025-07-01 10:40:21

## Executive Summary

This report provides comprehensive analysis of heterogeneous graph neural network performance on the Politifact dataset, comparing two edge construction policies: **KNN (standard)** and **KNN Test-Isolated**. The analysis covers experiments across different parameter configurations, focusing on 3-16 shot learning scenarios.

## Dataset Overview

- **Dataset**: Politifact
- **Model Architecture**: HAN (Heterogeneous Graph Attentional Network)
- **Shot Range**: 3-16 shot learning
- **Edge Policies Analyzed**: KNN, KNN Test-Isolated

### Experiment Statistics by Edge Policy

- **KNN**: 24 scenarios, 336 experiments
- **KNN_TEST_ISOLATED**: 37 scenarios, 518 experiments


## Performance Analysis by Edge Policy

### Best Performing Configurations

#### KNN Policy Top 5

| Rank | Configuration | Avg F1 | K-Neighbors | Multiview | Interactions |
|------|---------------|---------|-------------|-----------|-------------|
| 1 | deberta_hetero_knn_7_ensure_test_labeled_neighbor_... | 0.7930 | 7 | 3 | Yes |
| 2 | deberta_hetero_knn_5_ensure_test_labeled_neighbor_... | 0.7930 | 5 | 3 | Yes |
| 3 | deberta_hetero_knn_5_partial_sample_unlabeled_fact... | 0.7924 | 5 | 3 | Yes |
| 4 | deberta_hetero_knn_7_partial_sample_unlabeled_fact... | 0.7920 | 7 | 3 | Yes |
| 5 | deberta_hetero_knn_5_partial_sample_unlabeled_fact... | 0.7864 | 5 | 3 | Yes |


**Optimal Configuration for KNN:**
- **F1 Score**: 0.7930
- **K-Neighbors**: 7
- **Multiview**: 3
- **Interactions**: Enabled
- **Test Labeled Neighbor**: Enabled
- **Dissimilar Sampling**: Disabled

#### KNN_TEST_ISOLATED Policy Top 5

| Rank | Configuration | Avg F1 | K-Neighbors | Multiview | Interactions |
|------|---------------|---------|-------------|-----------|-------------|
| 1 | deberta_hetero_knn_test_isolated_5_partial_sample_... | 0.7930 | 5 | 3 | Yes |
| 2 | deberta_hetero_knn_test_isolated_7_ensure_test_lab... | 0.7913 | 7 | 3 | Yes |
| 3 | deberta_hetero_knn_test_isolated_7_partial_sample_... | 0.7913 | 7 | 3 | Yes |
| 4 | deberta_hetero_knn_test_isolated_3_ensure_test_lab... | 0.7907 | 3 | 3 | Yes |
| 5 | deberta_hetero_knn_test_isolated_5_ensure_test_lab... | 0.7907 | 5 | 3 | Yes |


**Optimal Configuration for KNN_TEST_ISOLATED:**
- **F1 Score**: 0.7930
- **K-Neighbors**: 5
- **Multiview**: 3
- **Interactions**: Enabled
- **Test Labeled Neighbor**: Disabled
- **Dissimilar Sampling**: Disabled


## Edge Policy Performance Comparison

### Top 5 Configurations for Each Edge Policy

#### KNN (Standard) Policy

| Rank | Configuration | Avg F1 | K-Neighbors | Multiview | Interactions | Test Labeled |
|------|---------------|---------|-------------|-----------|--------------|-------------|
| 1 | deberta_hetero_knn_7_ensure_test_labeled... | 0.7930 | 7 | 3 | Yes | Yes |
| 2 | deberta_hetero_knn_5_ensure_test_labeled... | 0.7930 | 5 | 3 | Yes | Yes |
| 3 | deberta_hetero_knn_5_partial_sample_unla... | 0.7924 | 5 | 3 | Yes | No |
| 4 | deberta_hetero_knn_7_partial_sample_unla... | 0.7920 | 7 | 3 | Yes | No |
| 5 | deberta_hetero_knn_5_partial_sample_unla... | 0.7864 | 5 | 3 | Yes | No |

#### KNN Test-Isolated Policy

| Rank | Configuration | Avg F1 | K-Neighbors | Multiview | Interactions | Test Labeled |
|------|---------------|---------|-------------|-----------|--------------|-------------|
| 1 | deberta_hetero_knn_test_isolated_5_parti... | 0.7930 | 5 | 3 | Yes | No |
| 2 | deberta_hetero_knn_test_isolated_7_ensur... | 0.7913 | 7 | 3 | Yes | Yes |
| 3 | deberta_hetero_knn_test_isolated_7_parti... | 0.7913 | 7 | 3 | Yes | No |
| 4 | deberta_hetero_knn_test_isolated_3_ensur... | 0.7907 | 3 | 3 | Yes | Yes |
| 5 | deberta_hetero_knn_test_isolated_5_ensur... | 0.7907 | 5 | 3 | Yes | Yes |

### Performance Summary

- **Best KNN Performance**: 0.7930
- **Best KNN Test-Isolated Performance**: 0.7930
- **Performance Difference**: 0.0000 (+0.0%)

**Analysis**: Test-isolated KNN shows better or comparable performance, indicating robust generalization without test data leakage.


## K-Shot Performance Analysis by Edge Policy

### KNN Policy - Top 3 Configurations

| Configuration | Avg F1 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deberta_hetero_knn_7_ensu... | 0.7930 | 0.708 | 0.778 | 0.702 | 0.708 | 0.793 | 0.838 | 0.848 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_5_ensu... | 0.7930 | 0.708 | 0.778 | 0.702 | 0.708 | 0.793 | 0.838 | 0.848 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_5_part... | 0.7924 | 0.699 | 0.778 | 0.702 | 0.708 | 0.793 | 0.838 | 0.848 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |

### KNN_TEST_ISOLATED Policy - Top 3 Configurations

| Configuration | Avg F1 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deberta_hetero_knn_test_i... | 0.7930 | 0.708 | 0.778 | 0.702 | 0.708 | 0.793 | 0.838 | 0.848 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_test_i... | 0.7913 | 0.708 | 0.778 | 0.702 | 0.708 | 0.793 | 0.828 | 0.835 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_test_i... | 0.7913 | 0.708 | 0.778 | 0.702 | 0.708 | 0.793 | 0.828 | 0.835 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |


## Ablation Study Analysis

### KNN Policy Ablation Analysis

#### Interaction Component Impact

#### Multi-View Settings Impact

- **Multiview 0 (Baseline)**: 0.7802
- **Multiview 3**: 0.7886 (+0.0084)
- **Multiview 6**: 0.7758 (-0.0044)

**Best Setting**: Multiview 3 with F1 score of 0.7886

#### K-Neighbors Hyperparameter Analysis

- **K=5**: 0.7819
- **K=7**: 0.7812

**Optimal K-value**: 5 (F1: 0.7819)
**Performance Range**: 0.0007 (0.1% improvement from worst to best)

---

### KNN_TEST_ISOLATED Policy Ablation Analysis

#### Interaction Component Impact

- **With Interactions**: 0.7806
- **Without Interactions (no_interactions)**: 0.5951
- **Interaction Benefit**: 0.1856 (+31.2%)

**Conclusion**: Synthetic user interactions provide meaningful signal for fake news detection.

#### Multi-View Settings Impact

- **Multiview 0 (Baseline)**: 0.7810
- **Multiview 3**: 0.7727 (-0.0082)
- **Multiview 6**: 0.7735 (-0.0075)

**Best Setting**: Multiview 0 with F1 score of 0.7810

#### K-Neighbors Hyperparameter Analysis

- **K=3**: 0.7807
- **K=5**: 0.7664
- **K=7**: 0.7805

**Optimal K-value**: 3 (F1: 0.7807)
**Performance Range**: 0.0143 (1.9% improvement from worst to best)

---


## Hyperparameter Search Analysis

### Multi-View Configuration Analysis (0, 3, 6 views)

| Edge Policy | Multiview 0 | Multiview 3 | Multiview 6 | Best Setting |
|-------------|-------------|-------------|-------------|-------------|
| knn | 0.7802 | 0.7886 | 0.7758 | 3 (0.7886) |
| knn_test_isolated | 0.7810 | 0.7727 | 0.7735 | 0 (0.7810) |

### K-Neighbors Analysis (3, 5, 7)

| Edge Policy | K=3 | K=5 | K=7 | Best Setting |
|-------------|-----|-----|-----|-------------|
| knn | 0.0000 | 0.7819 | 0.7812 | 5 (0.7819) |
| knn_test_isolated | 0.7807 | 0.7664 | 0.7805 | 3 (0.7807) |

### Hyperparameter Recommendations

#### KNN Policy

- **Recommended Multiview**: 3 views
- **Recommended K-neighbors**: 5

#### KNN_TEST_ISOLATED Policy

- **Recommended Multiview**: 0 views
- **Recommended K-neighbors**: 3


## Recommendations for Politifact

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
