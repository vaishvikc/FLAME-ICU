# claude.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLAME-ICU (Federated Learning Adaptable Mortality Estimator for the ICU) is a machine learning system for predicting ICU mortality using federated learning approaches. The project enables multiple healthcare sites to collaborate on model development while maintaining data privacy.

## Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv flameICU
source flameICU/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## High-Level Architecture

The project follows a federated learning architecture with three operational modes:

1. **Main Site Training**: Full model training from scratch at RUSH with complete datasets
2. **Inference Only**: Other sites use pre-trained models without local training  
3. **Transfer Learning**: Sites fine-tune models with 50% local data

### Data Flow

1. **Raw Data** → **Preprocessing Pipeline** (code/preprocessing/)
   - Creates ICU cohort (24-hour windows, adult patients)
   - Engineers features from hourly/event data
   - Outputs: protected_outputs/preprocessing/

2. **Preprocessed Data** → **Model-Specific Splits** (code/preprocessing/)
   - LSTM: Sequential 24-hour windows → pickle format
   - XGBoost/NN: Aggregated features → parquet format
   - Outputs: protected_outputs/intermediate/

3. **Split Data** → **Model Training** (code/models/)
   - Each model has independent config and training
   - Outputs: protected_outputs/models/

### Key Components

- **Preprocessing Pipeline**: Shared Jupyter notebooks for cohort generation and feature engineering
- **Model Implementations**: Three models (LSTM, XGBoost, NN) with standardized train/inference interfaces
- **Data Loaders**: Model-specific data preparation scripts that handle format conversion
- **Configuration**: Site-specific config in config_demo.json, model configs in respective directories

### Model Architectures

- **LSTM**: 2-layer (64→32 units), dropout 0.216, handles variable-length sequences
- **XGBoost**: Gradient boosting, max_depth 6, learning_rate 0.1, uses aggregated features
- **Neural Network**: Feedforward architecture, uses same data format as XGBoost

## Important Notes

- All data outputs go to `protected_outputs/` directory (gitignored for privacy)
- Configuration file `config_demo.json` must be set up with site-specific paths
- use context7 mcp for getting 
- When using Marimo: reference context7 for additional details
- When working with CLIF data: use context7 pyclif library if the function exists for the assistant
- Marimo dont allow reassigment of varible so be aware of it
- always run files from top folders

## Marimo Notebook Best Practices

### Displaying Charts and Visualizations
- **DON'T assign charts to variables**: In Marimo, directly return the chart expression without variable assignment
- **Correct pattern for Altair charts**:
  ```python
  @app.cell
  def _(alt, data):
          alt.Chart(data).mark_line().encode(
          x='hour:Q',
          y='value:Q'
      )
  ```
- **Incorrect pattern (avoid this)**:
  ```python
  @app.cell
  def _(alt, data):
      chart = alt.Chart(data).mark_line().encode(...)  # Don't do this
      return chart
  ```
- Charts are displayed by returning them as the last expression in a cell
- This keeps the namespace clean and follows Marimo's reactive paradigm