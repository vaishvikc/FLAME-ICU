# Stage 2 Phase 2: Ensemble Construction & Testing

## Overview

This phase creates and evaluates ensemble models using simple, practical combination strategies suitable for real clinical deployment. Uses leave-one-out testing where each site excludes their own models when creating ensembles and tests only on local data.

## Who Runs What

### All Sites
- **Script**: `simple_average.py` - Equal weight ensemble
- **Script**: `accuracy_weighted.py` - Performance-based weighted ensemble  
- **Purpose**: Create and test ensembles excluding local site's models
- **Testing**: Only on local 2023-2024 test data

## Prerequisites

- Stage 1 completed (all models available in BOX)
- Stage 2 Phase 1 completed (cross-site performance known)
- Local test data prepared (2023-2024)
- Performance metrics from Phase 1 available

## Quick Start

### For All Sites
```bash
cd stage2/phase2_ensemble

# Simple average ensemble (equal weights)
python simple_average.py --site_name "YOUR_SITE_NAME"

# Accuracy weighted ensemble (performance-based weights)
python accuracy_weighted.py --site_name "YOUR_SITE_NAME"
```

## Ensemble Strategies

### 1. Simple Average Ensemble
- **Method**: Equal weighting of all available models
- **Implementation**: `ensemble_prediction = mean([model1_pred, model2_pred, ...])`
- **Excludes**: Local site's models (leave-one-out)
- **Advantage**: Simple, transparent, clinically interpretable

### 2. Local Accuracy Weighted Ensemble
- **Method**: Weight models by their local AUC performance from Stage 1
- **Implementation**: `ensemble_prediction = Σ(weight_i × model_i_prediction)`
- **Weights**: Based on AUC performance (`weight_i ∝ local_AUC_i`)
- **Advantage**: Performance-driven, evidence-based

## Leave-One-Out Testing Protocol

### Unbiased Evaluation Strategy
- **At Site A**: Use models from Sites B, C, D, E, F, G only
- **At Site B**: Use models from Sites A, C, D, E, F, G only  
- **Continue for each site**: Always exclude own models

### Why Leave-One-Out?
- Prevents overfitting to local data patterns
- Ensures unbiased evaluation
- Simulates real deployment scenario
- Maintains clinical validity

## Process Details

1. **Model Collection**: Load all models from BOX folder
2. **Local Model Exclusion**: Remove current site's models
3. **Ensemble Construction**: Create both ensemble types
4. **Local Testing**: Test on site's local 2023-2024 data only
5. **Performance Comparison**: Compare with best individual model

## Expected Outputs

### Ensemble Performance Files
- `simple_average_results.json` - Simple ensemble performance
- `accuracy_weighted_results.json` - Weighted ensemble performance
- `ensemble_comparison.csv` - Side-by-side comparison
- `individual_vs_ensemble.json` - Improvement analysis

### Performance Analysis
- Ensemble vs best individual model comparison
- Statistical significance testing
- Clinical interpretability assessment
- Deployment recommendations

## Configuration

Key parameters in `config.json`:
```json
{
  "ensemble_testing": {
    "ensemble_types": ["simple_average", "accuracy_weighted"],
    "exclude_local_models": true,
    "test_data_years": [2023, 2024],
    "significance_testing": true,
    "bootstrap_iterations": 1000
  }
}
```

## Model Inclusion Rules

### Models Included in Ensemble
- Transfer learning models (Approach 2) from other sites
- Independent models (Approach 3) from other sites  
- Round robin models (Approach 4) - collaborative models
- RUSH baseline model (Approach 1) if from different site

### Models Excluded
- Any model trained using local site's data
- Local transfer learning models
- Local independent models
- Models with poor cross-site performance (<0.60 AUC)

## Success Criteria

- Ensemble outperforms best individual model
- Statistically significant improvement (p < 0.05)
- Performance improvement >1% AUC
- Successful deployment at all sites

## Expected Performance

### Ensemble Benefits
- **Simple Average**: 1-3% AUC improvement over individual models
- **Accuracy Weighted**: 2-5% AUC improvement (typically better)
- **Reduced Variance**: More stable predictions across patients
- **Clinical Robustness**: Better generalization to unseen cases

### Best Use Cases
- Sites with limited training data
- Sites with unique patient populations  
- Clinical deployment requiring stability
- Regulatory environments needing interpretability

## Implementation Process

### Simple Average Ensemble
```python
# Pseudocode
models = load_models_excluding_local_site()
predictions = []
for model in models:
    pred = model.predict(X_test_local)
    predictions.append(pred)
ensemble_pred = mean(predictions)
```

### Accuracy Weighted Ensemble
```python
# Pseudocode  
models = load_models_excluding_local_site()
weights = get_performance_weights(models)
weighted_preds = []
for model, weight in zip(models, weights):
    pred = model.predict(X_test_local)
    weighted_preds.append(weight * pred)
ensemble_pred = sum(weighted_preds)
```

## Quality Assurance

### Before Running
- Verify Stage 1 models available
- Check local test data quality
- Confirm performance metrics from Phase 1
- Test with small subset first

### During Execution
- Monitor ensemble construction process
- Validate feature alignment across models
- Check prediction ranges (0-1 for probabilities)
- Log all warnings and errors

### After Completion
- Verify ensemble improvements are meaningful
- Check statistical significance
- Validate clinical interpretability
- Confirm results consistency

## Clinical Deployment Recommendation

### Primary Choice: Local Accuracy Weighted Ensemble
- **Most clinically acceptable**: Performance-based rationale
- **Easy to validate**: Clear weighting methodology
- **Auditable**: Transparent decision process
- **Regulatory friendly**: Evidence-based approach

### Deployment Strategy
1. Use accuracy-weighted ensemble as primary
2. Simple average as backup/validation
3. Monitor performance over time
4. Update weights based on new performance data

## Troubleshooting

**Common Issues:**
- **No models after exclusion**: Check if any external models available
- **Poor ensemble performance**: Verify model compatibility
- **Weight calculation errors**: Check performance data format
- **Memory issues**: Process models in smaller batches

**Performance Issues:**
- **No improvement**: Models may be too similar
- **Degraded performance**: Check for incompatible models
- **Inconsistent results**: Verify data preprocessing consistency

## Timeline

- **Day 1**: Model collection and validation
- **Day 2**: Simple average ensemble testing
- **Day 3**: Accuracy weighted ensemble testing
- **Day 4**: Performance comparison and analysis
- **Day 5**: Results compilation and recommendations

## Statistical Testing

### Significance Testing
- Bootstrap confidence intervals
- McNemar's test for paired comparisons
- DeLong test for AUC comparisons
- Bonferroni correction for multiple testing

### Clinical Significance
- Minimum meaningful improvement: >1% AUC
- Net reclassification improvement
- Decision curve analysis
- Cost-effectiveness considerations

## Support

For issues:
- **Ensemble construction**: Check model compatibility
- **Performance analysis**: Verify statistical methods
- **Clinical interpretation**: Contact domain experts
- **Technical support**: flame-icu-coordination@rush.edu

## Final Deliverables

- Comprehensive ensemble performance report
- Deployment recommendations per site
- Statistical significance analysis
- Clinical interpretability assessment
- Regulatory approval pathway guidance