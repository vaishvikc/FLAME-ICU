#!/bin/bash
echo "Running 01_cohort.py"
uv run code/preprocessing/01_cohort.py
echo "Running 02_feature_assmebly.py"
uv run code/preprocessing/02_feature_assmebly.py
echo "Running 03_hourly_qc.py"
uv run code/preprocessing/03_hourly_qc.py
echo "Running 04_feature_consolidation.py"
uv run code/preprocessing/04_feature_consolidation.py
