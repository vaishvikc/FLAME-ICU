# WUPERR Model - Federated Learning for ICU Mortality Prediction

This directory contains the WUPERR (Weighted Update with Partial Error Reduction and Regularization) federated learning implementation for ICU mortality prediction across multiple hospital sites.

## Overview

WUPERR implements a federated learning approach where 8 different hospital sites sequentially train a shared LSTM model. Each site has unique characteristics and data distributions, simulating real-world healthcare system heterogeneity.

## Hospital Sites

The federated learning system includes 8 sites with varying characteristics:

| Site | Type | Data % | Characteristics |
|------|------|--------|-----------------|
| 1 | Large Academic | 25% | High acuity patients, complex cases |
| 2 | Large Community | 20% | Balanced patient mix |
| 3 | Medium Academic | 15% | Research focused |
| 4 | Medium Community | 15% | Balanced patient mix |
| 5 | Small Community | 10% | Low acuity patients |
| 6 | Small Rural | 8% | Limited resources, missing data |
| 7 | Specialty Hospital | 5% | Specific conditions focus |
| 8 | Critical Access | 2% | Basic care |

## Files

- `simulate_multisite.py`: Orchestrates the entire federated learning process
- `sequential_train.py`: Trains model for a specific site
- `evaluate.py`: Evaluates the federated model performance
- `sequential_wuperr.py`: Core WUPERR algorithm implementation
- `git_model_manager.py`: Manages model versioning and sharing via Git
- `setup_git_lfs.py`: Sets up Git LFS for large model files
- `config_wuperr.json`: Configuration for all parameters
- `SITES.md`: Detailed information about each hospital site

## Run Sequences

### 1. Full Federated Learning Simulation

This runs the complete federated learning process across all 8 sites:

```bash
# Setup and run multi-site simulation
cd models/wuperr

# First time setup: Initialize Git LFS
python setup_git_lfs.py

# Run the complete federated learning simulation
python simulate_multisite.py --num_rounds 3 --num_sites 8

# This will:
# 1. Split the main dataset into 8 site-specific datasets
# 2. Apply site-specific biases (high acuity, low acuity, etc.)
# 3. Save site datasets to data/sites/
# 4. Run sequential training for each site in each round
# 5. Save results and analysis
```

### 2. Individual Site Training

To train a specific site individually:

```bash
# Train Site 1 (Large Academic) for Round 1
python sequential_train.py --site_id 1 --round_num 1

# Train Site 5 (Small Community) for Round 2
python sequential_train.py --site_id 5 --round_num 2

# The script will automatically:
# 1. Pull the latest model from Git repository
# 2. Load site-specific data
# 3. Apply WUPERR algorithm for training
# 4. Save model with site contributions
# 5. Push updated model back to repository
```

### 3. Setup Site Data Only

To prepare site datasets without training:

```bash
python simulate_multisite.py --setup_only

# This creates:
# - data/sites/site_1_data.parquet through site_8_data.parquet
# - data/sites/site_metadata.json
```

### 4. Evaluate Federated Model

After training, evaluate the global model:

```bash
python evaluate.py

# This will:
# 1. Load the final federated model
# 2. Evaluate on each site's test data
# 3. Generate performance metrics and visualizations
# 4. Save results to output/models/wuperr/results/
```

## Configuration

Key configuration parameters in `config_wuperr.json`:

```json
{
  "federated_learning": {
    "num_sites": 8,
    "num_rounds": 3,
    "training_mode": "sequential"
  },
  "wuperr_parameters": {
    "regularization_lambda": 0.01,
    "update_threshold": 0.001,
    "fisher_samples": 200,
    "ewc_lambda": 0.4
  }
}
```

## Training Process

1. **Model Initialization**: Site 1 initializes the model or loads from repository
2. **Sequential Training**: Each site trains in order (1→2→3→...→8)
3. **WUPERR Updates**: Model weights updated using WUPERR algorithm to prevent catastrophic forgetting
4. **Git Synchronization**: After each site, model is committed and pushed
5. **Round Completion**: After Site 8, a new round begins with Site 1

## Outputs

Model artifacts are saved to `../../output/models/wuperr/`:

```
output/models/wuperr/
├── model/              # Git repository with model versions
│   ├── model.pt        # Current model state
│   ├── metadata.json   # Model metadata
│   └── contributions.json  # Site contributions tracking
├── logs/               # Training logs for each site/round
├── graphs/             # Performance visualizations
└── results/            # Evaluation results
```

## Model Sharing Mechanism

The WUPERR system uses Git for model versioning and sharing:

1. Each site pulls the latest model before training
2. After training, the site commits its updated model
3. Commit messages include performance metrics
4. Git history tracks the evolution of the federated model
5. Sites can rollback to previous versions if needed

## Prerequisites

- Python 3.8+
- PyTorch
- Git and Git LFS
- Required packages: pandas, numpy, scikit-learn, matplotlib

## Troubleshooting

1. **Git LFS Issues**: Run `python setup_git_lfs.py` to reinitialize
2. **Missing Site Data**: Ensure `simulate_multisite.py` has been run at least once
3. **Model Convergence**: Adjust learning rate and WUPERR parameters in config
4. **Memory Issues**: Reduce batch size or fisher_samples in configuration

## Advanced Usage

### Custom Site Configuration

Modify site characteristics in `simulate_multisite.py`:

```python
self.site_config = {
    1: {'proportion': 0.25, 'type': 'large_academic', 'bias': 'high_acuity'},
    # Add or modify sites here
}
```

### Resume Training

If training is interrupted:

```bash
# Check last completed site/round
python sequential_train.py --site_id 4 --round_num 2
# Will automatically continue from the last checkpoint
```