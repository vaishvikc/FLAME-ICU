# Stage 2 Phase 1: Cross-Site Model Testing

## Overview

This phase evaluates the generalization capability of models trained in Stage 1 Approaches 2-4 by testing them across all participating sites. Each site tests all received models on their local 2023-2024 test data to create a comprehensive performance matrix.

## Who Runs What

### All Sites
- **Script**: `test_all_models.py`
- **Purpose**: Test all models from BOX folder on local test data
- **Input**: All Stage 1 models (approaches 2-4)
- **Output**: Performance matrix for all models on local data

## Prerequisites

- Stage 1 completed (all approaches)
- All models uploaded to BOX folder
- Local 2023-2024 test data prepared
- BOX folder access configured

## Quick Start

### For All Sites
```bash
cd stage2/phase1_cross_testing
python test_all_models.py --site_name "YOUR_SITE_NAME"
# Tests all available models on local 2023-2024 data
```

## Process Details

1. **Model Discovery**: Scan BOX folder for all Stage 1 models
2. **Model Loading**: Load models from approaches 2, 3, and 4
3. **Cross-Site Testing**: Test each model on local test data
4. **Performance Matrix**: Create comprehensive results matrix
5. **Analysis**: Generate generalization performance reports

## Models Tested

### From Stage 1 Approaches
- **Approach 2**: Transfer learning models from all sites
- **Approach 3**: Independent models from all sites  
- **Approach 4**: Round robin collaborative models
- **Baseline**: RUSH centralized model (Approach 1)

### Model Types
- XGBoost models
- Neural Network models (if available)

## Expected Outputs

### Performance Matrix Files
- `cross_site_performance_matrix.csv` - All models × all sites
- `generalization_analysis.json` - Detailed analysis results
- `model_rankings_by_site.csv` - Best performing models per site

### Analysis Reports
- Cross-site generalization patterns
- Site similarity assessments
- Training approach comparisons
- Performance degradation analysis

## Testing Protocol

### For Each Model
1. **Load Model**: Download from BOX folder
2. **Feature Alignment**: Ensure feature compatibility
3. **Scaling**: Apply appropriate feature scaling
4. **Prediction**: Generate predictions on local test data
5. **Metrics**: Calculate standard evaluation metrics

### Metrics Calculated
- AUC-ROC, Precision, Recall, F1-Score
- Specificity, NPV, PPV
- Brier Score, Log Loss
- Calibration metrics

## Configuration

Key parameters in `config.json`:
```json
{
  "cross_site_testing": {
    "test_approaches": ["approach2", "approach3", "approach4"],
    "model_types": ["xgboost", "neural_network"],
    "metrics": ["auc_roc", "f1_score", "precision", "recall"],
    "include_calibration": true
  }
}
```

## Analysis Framework

### Generalization Analysis
- Compare local vs cross-site performance for each model
- Identify performance degradation patterns
- Assess model robustness across sites

### Site Similarity Assessment  
- Cluster sites based on model performance patterns
- Identify sites with similar data characteristics
- Understand cross-site compatibility

### Training Approach Comparison
- Evaluate relative effectiveness of different approaches
- Identify best-performing training strategies
- Compare with baseline centralized model

## Success Criteria

- All available models tested on local data
- Performance matrix generated successfully
- <10% performance degradation for good models
- Successful identification of best approaches

## Expected Results

### Performance Patterns
- Transfer learning models: 0-5% improvement over baseline
- Independent models: Variable performance by site
- Round robin models: Balanced performance across sites

### Generalization Insights
- Models trained on similar populations generalize better
- Some approaches more robust to site differences
- Feature importance varies across sites

## Troubleshooting

**Common Issues:**
- **Missing models**: Check BOX folder access and Stage 1 completion
- **Feature mismatch**: Verify model compatibility
- **Memory errors**: Process models in batches
- **Poor performance**: Check data quality and preprocessing

**Data Issues:**
- **Incompatible features**: Use feature intersection
- **Scaling problems**: Apply consistent preprocessing
- **Missing values**: Handle with model-specific strategies

## Quality Assurance

### Before Running
- Verify all Stage 1 models available in BOX
- Check local test data quality (2023-2024)
- Confirm feature consistency across models
- Test with subset first

### During Execution
- Monitor memory usage and processing time
- Log all errors and warnings
- Track progress across models
- Validate intermediate results

## Timeline

- **Day 1**: Setup and model discovery
- **Day 2-3**: Model testing execution
- **Day 4**: Results compilation and analysis
- **Day 5**: Report generation and sharing

## Output Format

### Performance Matrix Structure
```csv
site_name,model_source,approach,model_type,auc_roc,f1_score,precision,recall
SITE_A,SITE_B,approach2,xgboost,0.78,0.65,0.72,0.59
SITE_A,SITE_C,approach3,xgboost,0.74,0.61,0.68,0.55
...
```

### Analysis Report Structure
- Executive summary
- Detailed performance comparisons
- Visualization charts
- Recommendations for Stage 2 Phase 2

## Support

For issues:
- **Model loading errors**: Check compatibility and format
- **Performance discrepancies**: Verify preprocessing consistency
- **BOX access issues**: Contact IT support
- **Analysis questions**: flame-icu-coordination@rush.edu

## Next Steps

After completion:
- Review cross-site performance patterns
- Identify best-performing models for ensemble
- Prepare for Stage 2 Phase 2 (Ensemble Testing)
- Share results with coordination team