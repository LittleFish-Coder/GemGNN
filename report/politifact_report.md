# Politifact Dataset Analysis Report

**Generated on:** 2025-06-28 08:28:48

## Executive Summary

This report provides comprehensive analysis of HAN performance on Politifact dataset covering 61 parameter configurations across 3-16 shot learning scenarios.

## Performance Summary

- **Best Average F1 Score**: 0.7996
- **Worst Average F1 Score**: 0.6033
- **Performance Range**: 0.1963

### Top 10 Configurations

| Rank | Configuration | Avg F1 | K-Neighbors | Edge Policy | Multiview |
|------|---------------|--------|-------------|-------------|-----------||
| 1 | deberta_hetero_knn_7_ensure_te... | 0.7996 | 7 | knn | 3 |
| 2 | deberta_hetero_knn_5_ensure_te... | 0.7996 | 5 | knn | 3 |
| 3 | deberta_hetero_knn_5_partial_s... | 0.7996 | 5 | knn | 3 |
| 4 | deberta_hetero_knn_test_isolat... | 0.7996 | None | knn_test_isolated | 3 |
| 5 | deberta_hetero_knn_7_partial_s... | 0.7985 | 7 | knn | 3 |
| 6 | deberta_hetero_knn_test_isolat... | 0.7977 | None | knn_test_isolated | 3 |
| 7 | deberta_hetero_knn_test_isolat... | 0.7977 | None | knn_test_isolated | 3 |
| 8 | deberta_hetero_knn_test_isolat... | 0.7971 | None | knn_test_isolated | 3 |
| 9 | deberta_hetero_knn_test_isolat... | 0.7971 | None | knn_test_isolated | 3 |
| 10 | deberta_hetero_knn_test_isolat... | 0.7971 | None | knn_test_isolated | 3 |

## K-Shot Performance Analysis

### Top Configurations Performance Across Shot Counts (3-16)

| Configuration | Avg F1 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deberta_hetero_knn_7_ensure_test_labeled... | 0.800 | - | 0.778 | 0.702 | 0.708 | 0.793 | 0.838 | 0.848 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_5_ensure_test_labeled... | 0.800 | - | 0.778 | 0.702 | 0.708 | 0.793 | 0.838 | 0.848 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_5_partial_sample_unla... | 0.800 | - | 0.778 | 0.702 | 0.708 | 0.793 | 0.838 | 0.848 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_test_isolated_5_parti... | 0.800 | - | 0.778 | 0.702 | 0.708 | 0.793 | 0.838 | 0.848 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_7_partial_sample_unla... | 0.799 | - | 0.778 | 0.702 | 0.708 | 0.793 | 0.838 | 0.835 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_test_isolated_7_ensur... | 0.798 | - | 0.778 | 0.702 | 0.708 | 0.793 | 0.828 | 0.835 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_test_isolated_7_parti... | 0.798 | - | 0.778 | 0.702 | 0.708 | 0.793 | 0.828 | 0.835 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_test_isolated_3_ensur... | 0.797 | - | 0.759 | 0.702 | 0.708 | 0.793 | 0.838 | 0.835 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_test_isolated_5_ensur... | 0.797 | - | 0.759 | 0.702 | 0.708 | 0.793 | 0.838 | 0.835 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |
| deberta_hetero_knn_test_isolated_3_parti... | 0.797 | - | 0.759 | 0.702 | 0.708 | 0.793 | 0.838 | 0.835 | 0.861 | 0.848 | 0.817 | 0.817 | 0.791 | 0.787 | 0.805 |

## Ablation Study

### Component Impact Analysis

#### K-Neighbors Impact

- **K=5**: Average F1 = 0.7902
- **K=7**: Average F1 = 0.7903

#### Edge Policy Impact

- **knn**: Average F1 = 0.7903
- **knn_test_isolated**: Average F1 = 0.7839

#### Multiview Settings Impact

- **Multiview=0**: Average F1 = 0.7888
- **Multiview=3**: Average F1 = 0.7866
- **Multiview=6**: Average F1 = 0.7839

#### Feature Engineering Impact

- **Dissimilar Sampling**: +0.0045 impact
  - With dissimilar: 0.7887
  - Without dissimilar: 0.7842

- **Test Labeled Neighbor**: +0.0056 impact
  - With test labeled: 0.7893
  - Without test labeled: 0.7837

### Best Parameter Combinations

Top performing combinations across all k-shot settings:

**1. Configuration (F1: 0.7996)**
- K-Neighbors: 7
- Edge Policy: knn
- Multiview: 3
- Dissimilar Sampling: False
- Test Labeled Neighbor: True

**2. Configuration (F1: 0.7996)**
- K-Neighbors: 5
- Edge Policy: knn
- Multiview: 3
- Dissimilar Sampling: False
- Test Labeled Neighbor: True

**3. Configuration (F1: 0.7996)**
- K-Neighbors: 5
- Edge Policy: knn
- Multiview: 3
- Dissimilar Sampling: False
- Test Labeled Neighbor: False

**4. Configuration (F1: 0.7996)**
- K-Neighbors: None
- Edge Policy: knn_test_isolated
- Multiview: 3
- Dissimilar Sampling: False
- Test Labeled Neighbor: False

**5. Configuration (F1: 0.7985)**
- K-Neighbors: 7
- Edge Policy: knn
- Multiview: 3
- Dissimilar Sampling: False
- Test Labeled Neighbor: False


## Key Insights

- **Optimal K-neighbors**: 7
- **Best Edge Policy**: knn
- **Optimal Multiview**: 0

---

*Report generated automatically by the comprehensive analysis pipeline.*
