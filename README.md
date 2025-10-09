# FLAME-ICU
Federated Learning Adaptable Mortality Estimator for the ICU

## Project Overview

FLAME-ICU implements a multi-site federated learning approach for ICU mortality prediction using CLIF-standardized data. The project coordinates 6-7 institutions with RUSH as the main site, developing both XGBoost and Neural Network models.

### Stage 1: Model Development (3 Approaches)
1. **Cross-Site Validation** - Test RUSH-trained models across sites without local training
2. **Transfer Learning** - Fine-tune RUSH pre-trained models with local site data
3. **Independent Training** - Each site trains models from scratch

### Stage 2: Comprehensive Testing
- **Phase 1**: Cross-site testing of all Stage 1 models
- **Phase 2**: Leave-one-out ensemble construction with accuracy weighting

### Data Split
- **Training**: 2018-2022 admissions
- **Validation**: 2023 admissions
- **Testing**: 2024 admissions

### Deliverables
Models are shared via BOX for cross-site evaluation and ensemble construction, with final deployment recommendations based on performance metrics.

## Setup

### 1. Configure Site
Update `clif_config.json`:
```json
{
    "site": "your_site_name",
    "data_directory": "/path/to/your/clif/data",
    "filetype": "parquet",
    "timezone": "US/Central"
}
```

### 2. Install Dependencies
```bash
uv sync
```

## Required CLIF Tables

| Table | Columns | Categories |
|-------|---------|------------|
| **adt** | All columns | location_category |
| **hospitalization** | All columns | - |
| **patient** | All columns | - |
| **labs** | hospitalization_id, lab_result_dttm, lab_category, lab_value, lab_value_numeric | albumin, alt, ast, bicarbonate, bilirubin_total, bun, chloride, creatinine, inr, lactate, platelet_count, po2_arterial, potassium, pt, ptt, sodium, wbc |
| **vitals** | hospitalization_id, recorded_dttm, vital_category, vital_value | heart_rate, map, sbp, respiratory_rate, spo2, temp_c |
| **patient_assessments** | All columns | gcs_total |
| **medication_admin_continuous** | All columns (including med_dose, med_dose_unit) | norepinephrine, epinephrine, phenylephrine, vasopressin, dopamine, dobutamine, milrinone, isoproterenol |
| **respiratory_support** | All columns | device_category, fio2_set, peep_set |

## Execution Guide

### Prerequisites (All Approaches)
```bash
# 1. Configure site (update clif_config.json)
# 2. Install dependencies
uv sync
# 3. Run preprocessing pipeline
uv run code/preprocessing/00_scan_tables.py
uv run marimo run code/preprocessing/01_cohort.py
uv run marimo run code/preprocessing/02_feature_assmebly.py
uv run marimo run code/preprocessing/03_qc_heatmap.py
```

### Approach 1: Cross-Site Validation

**Federated Sites:**
```bash
# Download RUSH models from BOX
# Visit CLIF BOX and download the model_storage folder, place it in project root
# Only run inference with RUSH models (no training)
uv run python code/approach1_cross_site/stage_1/inference.py
# Upload results to BOX
```

### Approach 2: Transfer Learning

**Federated Sites:**
```bash
# Download RUSH base models from BOX
# Visit CLIF BOX and download the model_storage folder, place it in project root
# After preprocessing, fine-tune RUSH models with local data
uv run python code/approach2_transfer_learning/stage_1/transfer_learning.py
# Upload fine-tuned models to BOX
```

### Approach 3: Independent Training

**Federated Sites:**
```bash
# After preprocessing, train models independently
uv run python code/approach3_independent/stage_1/train_models.py
uv run python code/approach3_independent/stage_1/inference.py
# Upload models to BOX
```
