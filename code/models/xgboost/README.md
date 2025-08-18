# XGBoost Model for ICU Mortality Prediction

This directory contains the XGBoost model implementation for ICU mortality prediction with support for both main site training and other site operations (inference and transfer learning).

## Overview

The XGBoost model uses aggregated patient features (min/max/median values) to predict ICU mortality. It supports three operational modes:

1. **Main Site (RUSH) Training**: Full model training from scratch
2. **Other Sites - Inference Only**: Use the RUSH-trained model for predictions
3. **Other Sites - Transfer Learning**: Fine-tune the RUSH model with local data (50-50 split)

## Files

- `training.py`: Main training script for RUSH site
- `inference.py`: Inference script for all sites
- `transfer_learning.py`: Transfer learning for other sites
- `config.json`: Model configuration file
- `README.md`: This documentation

## Model Architecture

- Input: Aggregated features (min, max, median for each vital sign/lab)
- Model: Gradient Boosted Trees (XGBoost)
  - Max depth: 6
  - Learning rate: 0.1
  - Objective: Binary logistic
- Output: Mortality probability (0-1)

## Run Sequences

### 1. Main Site (RUSH) - Full Training

The main site trains the model from scratch using all available data:

```bash
cd models/xgboost

# Train the XGBoost model
python training.py

# This will:
# 1. Load preprocessed data from ../../output/preprocessing/
# 2. Aggregate features by patient (min/max/median)
# 3. Train XGBoost with early stopping
# 4. Save model artifacts to ../../output/models/xgboost/
# 5. Generate feature importance and calibration plots
```

Expected outputs:
- `xgb_icu_mortality_model.json`: Trained model
- `xgb_feature_scaler.pkl`: Scaling parameters
- `xgb_feature_columns.pkl`: Feature names
- `metrics.json`: Performance metrics
- `plots/`: Feature importance and calibration plots

### 2. Other Sites - Inference Only

Other sites can use the RUSH-trained model without any local training:

```bash
cd models/xgboost

# Run inference using RUSH model
python inference.py

# For site-specific data:
python inference.py --data_path /path/to/site/data.parquet

# This will:
# 1. Load the pre-trained RUSH model
# 2. Aggregate site data features
# 3. Generate mortality predictions
# 4. Display results and metrics (if labels available)
```

Expected outputs:
- Console output with predictions
- Per-patient mortality probabilities
- Aggregate statistics
- Performance metrics (if true labels available)

### 3. Other Sites - Transfer Learning

Sites can boost the RUSH model using 50% of their data:

```bash
cd models/xgboost

# Run transfer learning with default settings
python transfer_learning.py

# With custom parameters
python transfer_learning.py \
    --data_path /path/to/site/data.parquet \
    --output_dir /path/to/output \
    --num_boost_round 50 \
    --learning_rate 0.05

# This will:
# 1. Load the RUSH pre-trained model
# 2. Split site data 50-50 (training/testing)
# 3. Continue boosting with 50% training data
# 4. Evaluate on 50% test data
# 5. Compare with RUSH model performance
# 6. Save the transfer-learned model
```

Expected outputs in `output/models/xgboost/transfer_learning/`:
- `transfer_xgb_model.json`: Fine-tuned model
- `model_comparison.csv`: Performance comparison
- `transfer_metrics.json`: Detailed metrics
- `plots/`: Updated feature importance

## Configuration

Key parameters in `config.json`:

```json
{
  "model_params": {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc"
  },
  "training_config": {
    "num_rounds": 100,
    "early_stopping_rounds": 10
  },
  "data_config": {
    "selected_features": [
      "spo2", "heartrate", "respiration", 
      "sbp", "dbp", "temperature", 
      "glucose", "sodium", "potassium"
    ]
  }
}
```

## Model Sharing Process

### For Main Site (RUSH):

1. Train model using `training.py`
2. Share model artifacts:
   ```bash
   # Package model files
   tar -czf xgboost_rush_model.tar.gz \
       ../../output/models/xgboost/xgb_icu_mortality_model.json \
       ../../output/models/xgboost/xgb_feature_scaler.pkl \
       ../../output/models/xgboost/xgb_feature_columns.pkl
   ```
3. Distribute to other sites via secure transfer

### For Other Sites:

1. Receive RUSH model package
2. Extract to appropriate directory:
   ```bash
   # Extract received model
   tar -xzf xgboost_rush_model.tar.gz -C ../../output/models/xgboost/
   ```
3. Choose operational mode:
   - Inference only: Use `inference.py`
   - Transfer learning: Use `transfer_learning.py`

## Usage Examples

### Example 1: RUSH Training Full Pipeline

```bash
# At RUSH site
cd models/xgboost

# 1. Ensure preprocessing is complete
ls ../../output/preprocessing/by_hourly_wide_df.parquet

# 2. Train model
python training.py

# 3. Check results
cat ../../output/models/xgboost/metrics.json

# 4. View feature importance
open ../../output/models/xgboost/plots/xgb_feature_importance.png
```

### Example 2: Rural Hospital Inference

```bash
# At rural hospital with limited resources
cd models/xgboost

# 1. Ensure RUSH model is available
ls ../../output/models/xgboost/xgb_icu_mortality_model.json

# 2. Run inference on local data
python inference.py --data_path /hospital/icu_data.parquet

# 3. Export predictions
python inference.py --data_path /hospital/icu_data.parquet \
    --output_predictions /hospital/predictions.csv
```

### Example 3: Community Hospital Transfer Learning

```bash
# At community hospital wanting to improve model
cd models/xgboost

# 1. Run transfer learning
python transfer_learning.py \
    --data_path /hospital/icu_cohort.parquet \
    --num_boost_round 30 \
    --learning_rate 0.03

# 2. Review improvement
cat output/models/xgboost/transfer_learning/model_comparison.csv

# 3. If improved, use transfer model
# Update inference to use transfer_xgb_model.json
```

## Performance Expectations

### Main Site (RUSH):
- Training on full dataset
- Expected AUC: 0.70-0.80
- Training time: 10-30 minutes
- Feature importance available

### Transfer Learning Sites:
- 50% data for training, 50% for testing
- Performance improvement: 0-5% over RUSH model
- Training time: 5-15 minutes
- Best for sites with:
  - Different feature distributions
  - Unique patient populations
  - At least 500 patients

### Inference Only Sites:
- No training required
- Performance: Similar to RUSH
- Inference time: <100ms per patient
- Best for sites with:
  - Small datasets
  - Standard populations
  - Limited computational resources

## Feature Engineering

The model uses aggregated features:

```python
# For each patient, calculate:
- feature_min: Minimum value over stay
- feature_max: Maximum value over stay  
- feature_median: Median value over stay

# Example:
- heartrate_min
- heartrate_max
- heartrate_median
```

## Troubleshooting

### Common Issues:

1. **Missing Features**:
   ```python
   # Check available vs required features
   required = ['spo2', 'heartrate', ...]
   available = df.columns.tolist()
   missing = set(required) - set(available)
   ```

2. **Memory Issues**:
   - Use subset of features
   - Reduce max_depth in config
   - Process data in batches

3. **Poor Transfer Learning**:
   - Increase num_boost_round
   - Decrease learning rate
   - Ensure data quality

4. **Calibration Issues**:
   - XGBoost may need calibration
   - Use isotonic regression post-processing

## Advanced Usage

### Feature Selection

Customize features in `config.json`:
```json
{
  "data_config": {
    "selected_features": [
      "custom_feature_1",
      "custom_feature_2"
    ]
  }
}
```

### Hyperparameter Tuning

For site-specific optimization:
```python
# In transfer_learning.py
params = {
    'max_depth': 8,  # Increase for complex data
    'eta': 0.05,     # Decrease for stable learning
    'subsample': 0.7 # Adjust for overfitting
}
```

### Ensemble with Neural Network

Combine XGBoost with Neural Network predictions:
```python
# Simple averaging
final_prob = 0.6 * xgb_prob + 0.4 * nn_prob

# Weighted by performance
xgb_weight = xgb_auc / (xgb_auc + nn_auc)
nn_weight = nn_auc / (xgb_auc + nn_auc)
final_prob = xgb_weight * xgb_prob + nn_weight * nn_prob
```

### Export for Production

Save model for deployment:
```python
# Save in multiple formats
model.save_model('model.json')  # JSON format
model.save_model('model.bin')   # Binary format

# For ONNX deployment
import onnxmltools
onnx_model = onnxmltools.convert_xgboost(model)
onnxmltools.save_model(onnx_model, 'model.onnx')
```

## Model Interpretability

### Feature Importance

Three types available:
1. **Weight**: Number of times feature used
2. **Gain**: Average gain when feature used
3. **Cover**: Average coverage of feature

### SHAP Analysis

For detailed interpretability:
```python
import shap

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)
```

## Prerequisites

- Python 3.8+
- XGBoost 1.5+
- Required packages:
  ```bash
  pip install xgboost pandas numpy scikit-learn matplotlib
  ```

## Performance Monitoring

Track model performance over time:
```bash
# At each site, after inference
python inference.py --save_metrics

# Compare across time
python compare_metrics.py --site_name "Community Hospital"
```