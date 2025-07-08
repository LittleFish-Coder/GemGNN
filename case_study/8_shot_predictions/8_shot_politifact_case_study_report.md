# 8-Shot PolitiFact Case Study Report

## Model Performance Summary

| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| LLAMA | 0.824 | Calculated | Individual predictions available |
| GEMMA | 0.686 | Calculated | Individual predictions available |
| LESS4FD | 0.451 | 0.4507692307692308 | Aggregate metrics only |
| HeteroSGT | 0.471 | 0.3743752839618355 | Aggregate metrics only |
| GenFEND | 0.716 | 0.4999154691462384 | Aggregate metrics only |
| GemGNN_HAN | 0.873 | 0.8381940207443563 | Requires inference for individual predictions |

## Key Findings

- **Data Availability**: Individual predictions are available for LLM models (Llama, Gemma)
- **Limitation**: Related work models (LESS4FD, HeteroSGT, GenFEND) only provide aggregate metrics
- **GemGNN**: Metrics available but individual predictions require model inference

## Methodology for Case Study

To complete the comparative analysis showing where GemGNN succeeds but others fail:

1. **Need GemGNN Predictions**: Run inference on the target graph to get individual predictions
2. **Identify Success Cases**: Find examples where GemGNN is correct but strong baselines fail
3. **Analyze Reasons**: Examine why graph structure and multi-view approach help

## Next Steps

1. Generate GemGNN individual predictions using case_study.py
2. Extract individual predictions from related work models (if possible)
3. Perform detailed comparative analysis
4. Write examples demonstrating GemGNN advantages
