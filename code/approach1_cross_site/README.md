# Approach 1: Cross-Site Model Validation

## What You Downloaded

Pre-trained models from RUSH (trained on 2018-2022 data):
- **XGBoost model**: `stage_1/xgboost/xgb_model.json`
- **Neural Network model**: `stage_1/nn/nn_model.pth`

You will run **inference only** (no training) on your local test data.

---

## Prerequisites

- Python 3.13+
- Your site's CLIF data (parquet format)
- Your test data should be from 2023-2024

---

## Setup Steps

### 1. Install Dependencies

```bash
uv sync
```

Or if not using `uv`:
```bash
pip install -e .
```

### 2. Configure Your Site

Edit `clif_config.json` in the project root:
```json
{
    "site": "your_site_name",
    "data_directory": "/path/to/your/clif/data",
    "filetype": "parquet",
    "timezone": "US/Central"
}
```

Replace:
- `your_site_name` with your site identifier (e.g., "siteA", "northwestern", etc.)
- `/path/to/your/clif/data` with your actual data path

---

## Run Inference

From the project root directory:

```bash
python code/approach1_cross_site/stage_1/inference.py
```

That's it!

---

## What Happens

The script will:
1. Load the pre-trained RUSH models
2. Load your local test data (2023-2024)
3. Run predictions on your test set
4. Calculate performance metrics (AUC, accuracy, precision, recall, etc.)
5. Generate evaluation plots (ROC curves, calibration curves, decision curves)
6. Save all results

---

## Output Location

Results are saved to: `PHASE1_RESULTS_UPLOAD_ME/approach_1_stage_1/`

Files created:
- `inference_metrics_[your_site].json` - Complete metrics and curve data
- `approach_1_stage_1_inference_summary_[your_site].json` - Summary report
- `plots/roc_curves_[your_site].png` - ROC curve visualization
- `plots/calibration_curves_[your_site].png` - Calibration plot
- `plots/decision_curves_[your_site].png` - Decision analysis plot

**Note**: No PHI data is included in outputs - only aggregate metrics and curves.

---

## Troubleshooting

**Error: Models not found**
- Make sure you're running from the project root directory
- Verify `PHASE1_MODELS_UPLOAD_ME/approach_1/stage_1/` contains `xgboost/` and `nn/` folders

**Error: Data file not found**
- Check your `clif_config.json` paths
- Ensure preprocessed data exists at: `PHI_DATA/preprocessing/consolidated_features.parquet`
- Run preprocessing first if needed

**Missing test data**
- Verify your data has a `split_type` column with 'test' values
- Test data should be from 2023-2024

---

## Questions?

Contact the RUSH team or refer to the main project documentation.