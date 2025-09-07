# Approach 1: Cross-Site Model Validation

## Overview

This approach tests the generalization capability of a centralized model across diverse healthcare sites without any local training. RUSH trains models on their complete dataset and shares them with all federated sites for evaluation.

## Who Runs What

### RUSH Site Only
- **Script**: `train_rush_model.py`
- **Purpose**: Train models on 2018-2022 RUSH data
- **Output**: Trained XGBoost and Neural Network models

### All Sites (Including RUSH)
- **Script**: `test_rush_model.py` 
- **Purpose**: Test RUSH model on local 2023-2024 data
- **Output**: Performance metrics on local data

## Quick Start

### For RUSH Site
```bash
cd stage1/approach1_cross_site
python train_rush_model.py --site_name "RUSH"
# Models will be saved to BOX folder for sharing
```

### For All Sites (Testing)
```bash
cd stage1/approach1_cross_site
python test_rush_model.py --site_name "YOUR_SITE_NAME"
# Downloads RUSH model from BOX and tests on local data
```

## Expected Outputs

### RUSH Training
- `rush_xgboost_model/` - XGBoost model artifacts
- `rush_nn_model/` - Neural Network model artifacts
- Training performance metrics

### Site Testing
- Local performance metrics (AUC, F1, precision, recall)
- Performance comparison with RUSH results
- Generalization analysis report

## Data Requirements

- **Training Data**: RUSH 2018-2022 data (RUSH only)
- **Testing Data**: Each site's 2023-2024 data (all sites)
- **Format**: CLIF standardized format
- **Features**: Standard vital signs and lab values

## Configuration

Update `config.json` with:
- Site-specific paths
- BOX folder location
- Model hyperparameters

## Success Criteria

- RUSH model achieves >0.70 AUC on RUSH test data
- <10% performance drop across federated sites
- Successful deployment to all 6-7 sites

## Timeline

- **Week 1**: RUSH model training
- **Week 2**: Model distribution via BOX
- **Week 3**: All sites complete testing
- **Week 4**: Results collection and analysis

## Support

For issues:
- Check BOX folder access
- Verify data format (CLIF compliance)
- Contact coordination team: flame-icu-coordination@rush.edu