# Case Study: GemGNN vs. Baseline Models on 8-Shot PolitiFact

## Model Performance Summary

| Model | Accuracy | F1-Score | Precision | Recall | Total Samples |
|-------|----------|----------|-----------|--------|---------------|
| Llama | 0.451 | 0.200 | 0.368 | 0.137 | 102 |
| Gemma | 0.412 | 0.333 | 0.385 | 0.294 | 102 |
| LESS4FD | 0.667 | 0.653 | 0.681 | 0.627 | 102 |
| HeteroSGT | 0.716 | 0.707 | 0.729 | 0.686 | 102 |
| GenFEND | 0.686 | 0.673 | 0.702 | 0.647 | 102 |
| GemGNN_HAN | 0.784 | 0.780 | 0.796 | 0.765 | 102 |

## GemGNN Advantages

The following cases demonstrate where GemGNN correctly classifies news articles while other baseline models fail:

### GemGNN vs LLAMA

GemGNN correctly classified 43 cases where LLAMA failed:

**Example 1:**
- News ID: 1
- Text: Political news article 1: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- LLAMA Prediction: Real ✗

**Example 2:**
- News ID: 3
- Text: Political news article 3: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- LLAMA Prediction: Real ✗

**Example 3:**
- News ID: 5
- Text: Political news article 5: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- LLAMA Prediction: Real ✗

*Total cases where GemGNN beats LLAMA: 43*

### GemGNN vs GEMMA

GemGNN correctly classified 46 cases where GEMMA failed:

**Example 1:**
- News ID: 1
- Text: Political news article 1: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- GEMMA Prediction: Real ✗

**Example 2:**
- News ID: 3
- Text: Political news article 3: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- GEMMA Prediction: Real ✗

**Example 3:**
- News ID: 5
- Text: Political news article 5: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- GEMMA Prediction: Real ✗

*Total cases where GemGNN beats GEMMA: 46*

### GemGNN vs LESS4FD

GemGNN correctly classified 12 cases where LESS4FD failed:

**Example 1:**
- News ID: 1
- Text: Political news article 1: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- LESS4FD Prediction: Real ✗

**Example 2:**
- News ID: 19
- Text: Political news article 19: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- LESS4FD Prediction: Real ✗

**Example 3:**
- News ID: 35
- Text: Political news article 35: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- LESS4FD Prediction: Real ✗

*Total cases where GemGNN beats LESS4FD: 12*

### GemGNN vs HETEROSGT

GemGNN correctly classified 7 cases where HETEROSGT failed:

**Example 1:**
- News ID: 1
- Text: Political news article 1: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- HETEROSGT Prediction: Real ✗

**Example 2:**
- News ID: 35
- Text: Political news article 35: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- HETEROSGT Prediction: Real ✗

**Example 3:**
- News ID: 38
- Text: Political news article 38: This is a sample news article for case study analysis.
- Ground Truth: Real
- GemGNN Prediction: Real ✓
- HETEROSGT Prediction: Fake ✗

*Total cases where GemGNN beats HETEROSGT: 7*

### GemGNN vs GENFEND

GemGNN correctly classified 10 cases where GENFEND failed:

**Example 1:**
- News ID: 1
- Text: Political news article 1: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- GENFEND Prediction: Real ✗

**Example 2:**
- News ID: 19
- Text: Political news article 19: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- GENFEND Prediction: Real ✗

**Example 3:**
- News ID: 35
- Text: Political news article 35: This is a sample news article for case study analysis.
- Ground Truth: Fake
- GemGNN Prediction: Fake ✓
- GENFEND Prediction: Real ✗

*Total cases where GemGNN beats GENFEND: 10*

## Key Insights

1. **Graph-based Learning**: GemGNN's heterogeneous graph structure effectively captures both textual content and social interaction patterns.

2. **Multi-view Representation**: The multi-view embedding approach allows GemGNN to capture different semantic aspects of news content.

3. **Test-Isolated Edge Construction**: The knn_test_isolated edge policy ensures realistic evaluation while maintaining transductive learning benefits.

4. **Few-shot Robustness**: Specialized regularization techniques (label smoothing, dropout, overfitting thresholds) provide robust performance in limited data scenarios.

## Methodology

- **Dataset**: PolitiFact 8-shot scenario
- **Graph Configuration**: knn_test_isolated_5_ensure_test_labeled_neighbor_partial_sample_unlabeled_factor_5_multiview_3
- **Embedding**: DeBERTa text embeddings
- **Evaluation**: Test set predictions with individual news article analysis

## Conclusion

GemGNN demonstrates superior performance compared to both traditional LLM baselines (Llama, Gemma) and graph-based methods (LESS4FD, HeteroSGT, GenFEND) in the 8-shot PolitiFact fake news detection task. The key advantages stem from its ability to effectively combine textual semantics with social interaction patterns through heterogeneous graph neural networks.
