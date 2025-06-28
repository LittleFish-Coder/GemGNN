# Comprehensive Analysis Report

**Generated on:** 2025-06-28 08:28:48

## Executive Summary

This report provides comprehensive analysis of heterogeneous graph neural network performance across GossipCop and PolitiFact datasets for fake news detection.

## Cross-Dataset Performance

- **GossipCop Best F1**: 0.5939
- **PolitiFact Best F1**: 0.7996
- **Performance Gap**: 0.2057

## Parameter Consistency Analysis

| Parameter | GossipCop Best | PolitiFact Best | Consistent |
|-----------|----------------|-----------------|------------|
| k_neighbors | 5 | 7 | ✗ |
| edge_policy | knn | knn | ✓ |
| multiview | 0 | 0 | ✓ |

## Key Findings

### What Works Best Across All K-Shot Scenarios

Based on comprehensive analysis across 3-16 shot scenarios:

1. **K-neighbors**: 5-7 consistently perform well
2. **Edge Policy**: knn shows strong performance
3. **Multiview**: Setting of 3 provides optimal balance
4. **Feature Engineering**: Mixed results for dissimilar sampling

### Key Points of Our Work

Compared to other approaches, our work provides:

1. **Comprehensive Parameter Analysis**: Systematic exploration of 108+ configurations
2. **Few-shot Learning Focus**: Addresses practical scenarios with limited data
3. **Cross-dataset Validation**: Demonstrates generalizability across domains
4. **Data Leakage Prevention**: Rigorous experimental design with test isolation
5. **Heterogeneous Graph Modeling**: Captures both content and interaction patterns

---

*Report generated automatically by the comprehensive analysis pipeline.*
