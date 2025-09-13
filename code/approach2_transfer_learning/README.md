# Approach 2: Transfer Learning with Main Model Initialization

## Overview

This approach leverages the RUSH pre-trained model as initialization for site-specific fine-tuning. Each federated site adapts the base model using 50% of their local training data, creating personalized models while benefiting from centralized knowledge.

## Who Runs What

### RUSH Site Only
- Already completed in Approach 1
- Base models available in BOX folder

### All Federated Sites
- **Script**: `fine_tune.py`
- **Purpose**: Fine-tune RUSH model with local data
- **Data Split**: 50% for fine-tuning, 50% reserved
- **Output**: Site-specific fine-tuned models

## Quick Start

### For Federated Sites
```bash
cd stage1/approach2_transfer_learning
python fine_tune.py --site_name "YOUR_SITE_NAME" --model_type "xgboost"
python fine_tune.py --site_name "YOUR_SITE_NAME" --model_type "nn"
# Fine-tuned models uploaded to BOX for Stage 2
```

## Process Details

1. **Download Base Model**: Load RUSH pre-trained model from BOX
2. **Data Split**: Split local 2018-2022 data 50-50
3. **Fine-tune**: Use 50% data to adapt model to local patterns
4. **Test**: Evaluate on local 2023-2024 data
5. **Upload**: Save fine-tuned model to BOX folder

## Expected Outputs

- `{SITE}_transfer_xgboost_model/` - Fine-tuned XGBoost model
- `{SITE}_transfer_nn_model/` - Fine-tuned Neural Network model  
- Performance comparison: Base vs Fine-tuned
- Training logs and metrics

## Fine-tuning Strategy

- **Learning Rate**: 0.1x of original (reduced for stability)
- **Training Duration**: Early stopping based on validation
- **Layer Strategy**: Full model fine-tuning (all layers trainable)
- **Regularization**: Maintain dropout and L2 regularization

## Data Requirements

- **Base Model**: RUSH pre-trained models (from BOX)
- **Fine-tuning Data**: 50% of local 2018-2022 data
- **Testing Data**: Local 2023-2024 data
- **Minimum**: 500+ patients for effective fine-tuning

## Configuration

Key parameters in `config.json`:
```json
{
  "transfer_learning": {
    "learning_rate_multiplier": 0.1,
    "early_stopping_patience": 10,
    "data_split_ratio": 0.5,
    "num_boost_rounds": 50
  }
}
```

## Success Criteria

- Fine-tuned model outperforms base RUSH model locally
- >2% improvement in local AUC
- Successful model upload to BOX folder
- Maintain model compatibility for Stage 2

## Troubleshooting

**Common Issues:**
- **Base model not found**: Check BOX folder access
- **Insufficient data**: Need 500+ patients minimum
- **Poor improvement**: Try different learning rates
- **Upload failure**: Verify BOX write permissions

## Expected Performance

- **Improvement**: 0-5% AUC improvement over base model
- **Training Time**: 10-30 minutes depending on data size
- **Best for**: Sites with different patient populations
- **Resource Requirements**: Moderate computational needs

## Timeline

- **Week 1**: Download base models
- **Week 2-3**: Fine-tuning across all sites
- **Week 4**: Testing and model upload
- **Week 5**: Performance analysis

## Support

For issues:
- Verify base model compatibility
- Check local data quality (50% split)
- Contact: flame-icu-coordination@rush.edu