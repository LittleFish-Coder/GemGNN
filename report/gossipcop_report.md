# Gossipcop Dataset Analysis Report

**Generated on:** 2025-06-28 08:28:48

## Executive Summary

This report provides comprehensive analysis of HAN performance on Gossipcop dataset covering 49 parameter configurations across 3-16 shot learning scenarios.

## Performance Summary

- **Best Average F1 Score**: 0.5939
- **Worst Average F1 Score**: 0.3261
- **Performance Range**: 0.2677

### Top 10 Configurations

| Rank | Configuration | Avg F1 | K-Neighbors | Edge Policy | Multiview |
|------|---------------|--------|-------------|-------------|-----------||
| 1 | deberta_hetero_knn_5_partial_s... | 0.5939 | 5 | knn | 3 |
| 2 | deberta_hetero_knn_5_ensure_te... | 0.5938 | 5 | knn | 3 |
| 3 | deberta_hetero_knn_7_partial_s... | 0.5929 | 7 | knn | 3 |
| 4 | deberta_hetero_knn_7_ensure_te... | 0.5928 | 7 | knn | 3 |
| 5 | deberta_hetero_knn_test_isolat... | 0.5924 | None | knn_test_isolated | 3 |
| 6 | deberta_hetero_knn_7_ensure_te... | 0.5919 | 7 | knn | 3 |
| 7 | deberta_hetero_knn_test_isolat... | 0.5919 | None | knn_test_isolated | 3 |
| 8 | deberta_hetero_knn_5_ensure_te... | 0.5919 | 5 | knn | 3 |
| 9 | deberta_hetero_knn_test_isolat... | 0.5918 | None | knn_test_isolated | 3 |
| 10 | deberta_hetero_knn_test_isolat... | 0.5917 | None | knn_test_isolated | 3 |

## K-Shot Performance Analysis

### Top Configurations Performance Across Shot Counts (3-16)

| Configuration | Avg F1 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deberta_hetero_knn_5_partial_sample_unla... | 0.594 | - | 0.631 | 0.604 | 0.597 | 0.601 | 0.581 | 0.580 | 0.610 | 0.614 | 0.580 | 0.580 | 0.582 | 0.578 | 0.581 |
| deberta_hetero_knn_5_ensure_test_labeled... | 0.594 | - | 0.630 | 0.605 | 0.599 | 0.601 | 0.581 | 0.580 | 0.611 | 0.614 | 0.579 | 0.581 | 0.581 | 0.579 | 0.580 |
| deberta_hetero_knn_7_partial_sample_unla... | 0.593 | - | 0.631 | 0.605 | 0.598 | 0.601 | 0.580 | 0.578 | 0.608 | 0.613 | 0.579 | 0.579 | 0.580 | 0.579 | 0.577 |
| deberta_hetero_knn_7_ensure_test_labeled... | 0.593 | - | 0.630 | 0.606 | 0.598 | 0.601 | 0.580 | 0.577 | 0.609 | 0.613 | 0.579 | 0.579 | 0.580 | 0.579 | 0.576 |
| deberta_hetero_knn_test_isolated_5_ensur... | 0.592 | - | 0.616 | 0.600 | 0.595 | 0.605 | 0.583 | 0.581 | 0.605 | 0.609 | 0.581 | 0.581 | 0.584 | 0.582 | 0.580 |
| deberta_hetero_knn_7_ensure_test_labeled... | 0.592 | - | 0.616 | 0.600 | 0.592 | 0.604 | 0.583 | 0.581 | 0.605 | 0.608 | 0.581 | 0.580 | 0.584 | 0.582 | 0.580 |
| deberta_hetero_knn_test_isolated_5_parti... | 0.592 | - | 0.616 | 0.600 | 0.594 | 0.603 | 0.582 | 0.581 | 0.603 | 0.608 | 0.582 | 0.581 | 0.582 | 0.582 | 0.581 |
| deberta_hetero_knn_5_ensure_test_labeled... | 0.592 | - | 0.614 | 0.600 | 0.594 | 0.603 | 0.583 | 0.581 | 0.604 | 0.608 | 0.581 | 0.581 | 0.582 | 0.582 | 0.582 |
| deberta_hetero_knn_test_isolated_7_ensur... | 0.592 | - | 0.614 | 0.599 | 0.594 | 0.604 | 0.583 | 0.582 | 0.602 | 0.607 | 0.581 | 0.581 | 0.584 | 0.582 | 0.581 |
| deberta_hetero_knn_test_isolated_7_parti... | 0.592 | - | 0.614 | 0.599 | 0.594 | 0.603 | 0.582 | 0.581 | 0.602 | 0.608 | 0.581 | 0.580 | 0.585 | 0.582 | 0.582 |

## Ablation Study

### Component Impact Analysis

#### K-Neighbors Impact

- **K=5**: Average F1 = 0.5889
- **K=7**: Average F1 = 0.5887

#### Edge Policy Impact

- **knn**: Average F1 = 0.5888
- **knn_test_isolated**: Average F1 = 0.5777

#### Multiview Settings Impact

- **Multiview=0**: Average F1 = 0.5883
- **Multiview=3**: Average F1 = 0.5763
- **Multiview=6**: Average F1 = 0.5853

#### Feature Engineering Impact

- **Dissimilar Sampling**: +0.0107 impact
  - With dissimilar: 0.5886
  - Without dissimilar: 0.5779

- **Test Labeled Neighbor**: +0.0106 impact
  - With test labeled: 0.5885
  - Without test labeled: 0.5779

### Best Parameter Combinations

Top performing combinations across all k-shot settings:

**1. Configuration (F1: 0.5939)**
- K-Neighbors: 5
- Edge Policy: knn
- Multiview: 3
- Dissimilar Sampling: True
- Test Labeled Neighbor: False

**2. Configuration (F1: 0.5938)**
- K-Neighbors: 5
- Edge Policy: knn
- Multiview: 3
- Dissimilar Sampling: True
- Test Labeled Neighbor: True

**3. Configuration (F1: 0.5929)**
- K-Neighbors: 7
- Edge Policy: knn
- Multiview: 3
- Dissimilar Sampling: True
- Test Labeled Neighbor: False

**4. Configuration (F1: 0.5928)**
- K-Neighbors: 7
- Edge Policy: knn
- Multiview: 3
- Dissimilar Sampling: True
- Test Labeled Neighbor: True

**5. Configuration (F1: 0.5924)**
- K-Neighbors: None
- Edge Policy: knn_test_isolated
- Multiview: 3
- Dissimilar Sampling: False
- Test Labeled Neighbor: True


## Key Insights

- **Optimal K-neighbors**: 5
- **Best Edge Policy**: knn
- **Optimal Multiview**: 0

---

*Report generated automatically by the comprehensive analysis pipeline.*
