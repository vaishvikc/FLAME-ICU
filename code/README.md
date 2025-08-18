# ICU Mortality Prediction - Reorganized Codebase

This codebase has been reorganized to separate shared preprocessing steps from model-specific implementations, improving maintainability and enabling easier addition of new model types.

## Directory Structure

```
code/
├── preprocessing/                     # Shared preprocessing steps
│   ├── 01_cohort.ipynb               # Cohort generation
│   ├── 02_feature_engineering.ipynb  # Feature engineering
│   └── config_demo.json              # Shared configuration
├── models/                           # Model implementations
│   ├── xgboost/                      # XGBoost model
│   │   ├── training.py
│   │   ├── inference.py
│   │   ├── transfer_learning.py
│   │   ├── config.json
│   │   └── README.md
│   └── federated/                    # Federated learning
│       ├── sequential_train.py
│       ├── simulate_multisite.py
│       ├── evaluate.py
│       ├── sequential_federated.py
│       ├── git_model_manager.py
│       ├── setup_git_lfs.py
│       ├── config_federated.json
│       └── README.md
├── shared/                           # Shared utilities and documentation
│   ├── prd.md                        # Product requirements document
│   └── Inference_py.ipynb            # Shared inference notebook
└── output/                           # Output directories
    ├── preprocessing/                # Preprocessing outputs
    │   ├── icu_cohort.csv            # Cohort data
    │   ├── by_event_wide_df.parquet  # Event-level features
    │   └── by_hourly_wide_df.parquet # Hourly features
    └── models/                       # Model-specific outputs
        ├── xgboost/                  # XGBoost artifacts
        └── federated/                # Federated learning artifacts
```

## Workflow

### 1. Preprocessing

Run the preprocessing notebooks in order:

```bash
# Generate cohort
jupyter notebook preprocessing/01_cohort.ipynb

# Engineer features
jupyter notebook preprocessing/02_feature_engineering.ipynb
```

These will create standardized outputs in `output/preprocessing/` that can be used by all models.

### 2. Model Training

Each model can be trained independently:

```bash
# XGBoost
cd models/xgboost
python training.py

# Federated Learning
cd models/federated
python simulate_multisite.py
```

### 3. Model Inference

Run inference using the respective scripts:

```bash
# XGBoost
cd models/xgboost
python inference.py

# Federated Learning
cd models/federated
python evaluate.py
```

## Adding New Models

To add a new model type:

1. Create a new directory under `models/`
2. Add training, inference, and configuration files
3. Update configuration to point to `../output/preprocessing/` for input
4. Set output path to `../output/models/[model_name]/`
5. Create a README documenting the model

## Key Features

- **Shared Preprocessing**: Cohort generation and feature engineering are centralized
- **Model Isolation**: Each model has its own directory and configuration
- **Consistent Outputs**: All models save to their respective output directories
- **Standardized Interface**: Common data formats between preprocessing and models
- **Version Control**: Git-friendly structure with proper separation of concerns