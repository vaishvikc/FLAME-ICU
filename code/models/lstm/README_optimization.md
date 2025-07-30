# LSTM Model Optimization Pipeline

## Overview
This directory contains a streamlined LSTM model pipeline for ICU mortality prediction.

## Main Scripts

### 1. `data_split.py`
- Creates train/test splits for LSTM sequential data
- Saves splits to `../../protected_outputs/intermediate/lstm/`
- Uses split ratios from `config.json`
- Run once to create consistent data splits

### 2. `optimize.py`
- Performs architecture exploration (Basic LSTM, Bidirectional, Attention, Stacked)
- Optimizes hyperparameters using Optuna
- Uses only training data with cross-validation
- Updates `config.json` with best parameters

### 3. `training.py`
- Trains final LSTM model with optimized parameters
- Uses pre-split train/test data
- Saves model and evaluation metrics

### 4. `inference.py`
- Loads trained model for predictions
- Handles new patient data

## Quick Start

Run the complete optimization pipeline:
```bash
./run_optimization.sh
```

Or run steps individually:
```bash
# Step 1: Create data splits
python3 data_split.py

# Step 2: Optimize architecture and hyperparameters
python3 optimize.py

# Step 3: Train final model
python3 training.py
```

## Configuration
All parameters are stored in `config.json`:
- `data_split`: Train/test ratios and paths
- `model_params`: Architecture parameters
- `training_config`: Training hyperparameters
- `data_config`: Data processing settings

## Output Structure
```
../../protected_outputs/
├── intermediate/
│   └── lstm/
│       ├── train_sequences.pkl
│       ├── test_sequences.pkl
│       └── split_metadata.json
└── models/
    └── lstm/
        ├── architecture_exploration/
        ├── optimization_plots/
        └── [trained models and metrics]
```

## Archive
Old scripts have been moved to the `archive/` folder for reference.