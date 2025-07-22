# LSTM Model for ICU Mortality Prediction

This directory contains the LSTM model implementation for ICU mortality prediction with support for both main site training and other site operations (inference and transfer learning).

## Overview

The LSTM model uses sequential patient data (24-hour windows) to predict ICU mortality. It supports three operational modes:

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

- Input: Variable-length sequences (up to 24 hours of ICU data)
- Architecture: 2-layer LSTM with dropout
  - LSTM Layer 1: 64 hidden units
  - LSTM Layer 2: 32 hidden units
  - Dense layers: 32 → 16 → 1
- Output: Mortality probability (0-1)

## Run Sequences

### 1. Main Site (RUSH) - Full Training

The main site trains the model from scratch using all available data:

```bash
cd models/lstm

# Train the LSTM model
python training.py

# This will:
# 1. Load preprocessed data from ../../output/preprocessing/
# 2. Create 24-hour sequences for each patient
# 3. Train the LSTM model with early stopping
# 4. Save model artifacts to ../../output/models/lstm/
# 5. Generate performance metrics and plots
```

Expected outputs:
- `lstm_icu_mortality_model.pt`: Trained model weights
- `feature_scaler.pkl`: Scaling parameters
- `feature_columns.pkl`: Feature names
- `metrics.json`: Performance metrics
- `plots/`: Training history and calibration plots

### 2. Other Sites - Inference Only

Other sites can use the RUSH-trained model without any local training:

```bash
cd models/lstm

# Run inference using RUSH model
python inference.py

# For site-specific data:
python inference.py --data_path /path/to/site/data.parquet

# This will:
# 1. Load the pre-trained RUSH model
# 2. Process site data into sequences
# 3. Generate mortality predictions
# 4. Display results and metrics (if labels available)
```

Expected outputs:
- Console output with predictions
- Per-patient mortality probabilities
- Aggregate statistics
- Performance metrics (if true labels available)

### 3. Other Sites - Transfer Learning

Sites can fine-tune the RUSH model using 50% of their data:

```bash
cd models/lstm

# Run transfer learning with default settings
python transfer_learning.py

# With custom parameters
python transfer_learning.py \
    --data_path /path/to/site/data.parquet \
    --output_dir /path/to/output \
    --num_epochs 50 \
    --batch_size 8

# This will:
# 1. Load the RUSH pre-trained model
# 2. Split site data 50-50 (training/testing)
# 3. Fine-tune model on 50% training data
# 4. Evaluate on 50% test data
# 5. Compare with RUSH model performance
# 6. Save the transfer-learned model
```

Expected outputs in `output/models/lstm/transfer_learning/`:
- `transfer_lstm_model.pt`: Fine-tuned model
- `model_comparison.csv`: Performance comparison
- `transfer_metrics.json`: Detailed metrics
- `plots/`: Training history and calibration

## Configuration

Key parameters in `config.json`:

```json
{
  "model_params": {
    "hidden_size1": 64,
    "hidden_size2": 32,
    "dropout_rate": 0.2,
    "batch_size": 16
  },
  "training_config": {
    "num_epochs": 100,
    "learning_rate": 0.001,
    "patience": 10,
    "gradient_clip_value": 1.0
  },
  "data_config": {
    "sequence_length": 24,
    "preprocessing_path": "output/preprocessing",
    "feature_file": "by_hourly_wide_df.parquet"
  }
}
```

## Model Sharing Process

### For Main Site (RUSH):

1. Train model using `training.py`
2. Share model artifacts:
   ```bash
   # Package model files
   tar -czf lstm_rush_model.tar.gz \
       ../../output/models/lstm/lstm_icu_mortality_model.pt \
       ../../output/models/lstm/feature_scaler.pkl \
       ../../output/models/lstm/feature_columns.pkl
   ```
3. Distribute to other sites via secure transfer

### For Other Sites:

1. Receive RUSH model package
2. Extract to appropriate directory:
   ```bash
   # Extract received model
   tar -xzf lstm_rush_model.tar.gz -C ../../output/models/lstm/
   ```
3. Choose operational mode:
   - Inference only: Use `inference.py`
   - Transfer learning: Use `transfer_learning.py`

## Usage Examples

### Example 1: RUSH Training Full Pipeline

```bash
# At RUSH site
cd models/lstm

# 1. Ensure preprocessing is complete
ls ../../output/preprocessing/by_hourly_wide_df.parquet

# 2. Train model
python training.py

# 3. Check results
cat ../../output/models/lstm/metrics.json
```

### Example 2: Community Hospital Inference

```bash
# At community hospital
cd models/lstm

# 1. Ensure RUSH model is available
ls ../../output/models/lstm/lstm_icu_mortality_model.pt

# 2. Run inference on local data
python inference.py --data_path /hospital/data/icu_data.parquet

# 3. Review predictions
# Predictions printed to console
```

### Example 3: Academic Hospital Transfer Learning

```bash
# At academic hospital with research goals
cd models/lstm

# 1. Prepare for transfer learning
python transfer_learning.py \
    --data_path /research/data/icu_cohort.parquet \
    --num_epochs 75 \
    --patience 15

# 2. Compare models
cat output/models/lstm/transfer_learning/model_comparison.csv

# 3. Deploy better performing model
# If transfer model is better, use it for inference
# Otherwise, continue with RUSH model
```

## Performance Expectations

### Main Site (RUSH):
- Training on full dataset
- Expected AUC: 0.75-0.85
- Training time: 1-2 hours (depending on data size)

### Transfer Learning Sites:
- 50% data for training, 50% for testing
- Performance improvement: 0-10% over RUSH model
- Training time: 30-60 minutes
- Best for sites with:
  - Unique patient populations
  - Different care protocols
  - Sufficient data (>1000 patients)

### Inference Only Sites:
- No training required
- Performance: Similar to RUSH (may vary by population)
- Inference time: <1 second per patient
- Best for sites with:
  - Limited computational resources
  - Small datasets (<1000 patients)
  - Standard patient populations

## Troubleshooting

### Common Issues:

1. **CUDA/GPU Errors**:
   ```bash
   # Force CPU usage
   export CUDA_VISIBLE_DEVICES=""
   python training.py
   ```

2. **Memory Issues**:
   - Reduce batch size in config.json
   - Reduce sequence length if possible

3. **Missing Features**:
   - The model expects specific feature columns
   - Check feature_columns.pkl for required features
   - Ensure preprocessing matches RUSH pipeline

4. **Poor Transfer Learning**:
   - Increase training epochs
   - Adjust learning rate (try 0.0001)
   - Ensure sufficient training data (>500 patients)

## Advanced Usage

### Custom Sequence Length

Modify `config.json`:
```json
{
  "data_config": {
    "sequence_length": 48  // For 48-hour windows
  }
}
```

### Feature Selection

To use subset of features, modify training.py:
```python
# Define custom feature list
selected_features = ['heart_rate_max', 'bp_systolic_min', ...]
```

### Ensemble with Other Models

Combine LSTM predictions with XGBoost:
```python
# Get predictions from both models
lstm_probs = lstm_model.predict(data)
xgb_probs = xgb_model.predict(data)

# Simple average ensemble
ensemble_probs = (lstm_probs + xgb_probs) / 2
```

## Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Required packages:
  ```bash
  pip install torch pandas numpy scikit-learn matplotlib
  ```