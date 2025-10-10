import marimo

__generated_with = "0.16.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ICU Mortality Model - Feature Extraction & 24-Hour Aggregation

    This notebook loads the ICU cohort and creates a 24-hour aggregated feature set (one row per hospitalization).

    ## Objective
    - Load ICU cohort from 01_cohort.py
    - Use clifpy to extract features from CLIF tables
    - Apply outlier handling using built-in clifpy functions
    - Create event-wide dataset for the first 24 hours
    - Aggregate to one row per hospitalization with min/max/median/last values
    - Add derived features (age bins, vasopressor count, device one-hot encoding)
    - Add temporal split column (row_type) based on admission year
    - Save aggregated features for modeling

    ## Feature Sources
    - **Vitals**: Heart rate, MAP, SBP, respiratory rate, SpO2, temperature (min/max/median)
    - **Labs**: Comprehensive lab panel with min/max aggregations (albumin, creatinine, hemoglobin, etc.)
    - **Respiratory Support**: Device one-hot encoding, FiO2, PEEP (max values)
    - **Medications**: Vasoactives and sedatives (vasopressor count)
    - **Patient Assessments**: GCS total (last/most recent value)
    - **Demographics**: Age, sex, race, ethnicity, language

    ## Aggregation Strategy
    - **Max/Worst**: lactate, BUN, creatinine, AST, ALT, INR, PTT, PT, FiO2, PEEP, respiratory rate, heart rate, temp, electrolytes
    - **Min/Worst**: platelets, PaO2, SpO2, SBP, albumin, electrolytes, temp, heart rate
    - **Median**: respiratory rate
    - **Last**: GCS total
    - **One-hot**: Respiratory devices (exclude Room Air and Other)
    - **Derived**: Age bins, vasopressor count, isfemale

    ## Temporal Split
    - **Training data**: 2018-2022 admissions
    - **Testing data**: 2023-2024 admissions
    """
    )
    return


@app.cell
def _():
    import sys
    import os
    sys.path.append('..')
    from config_helper import get_project_root, ensure_dir, get_output_path

    import pandas as pd
    import numpy as np
    from clifpy.clif_orchestrator import ClifOrchestrator
    from clifpy.utils.outlier_handler import apply_outlier_handling
    import json
    import warnings
    warnings.filterwarnings('ignore')

    print("=== ICU Mortality Model - Feature Extraction ===")
    print("Setting up environment...")
    return (
        ClifOrchestrator,
        apply_outlier_handling,
        ensure_dir,
        get_output_path,
        get_project_root,
        json,
        np,
        os,
        pd,
    )


@app.cell
def _(ensure_dir, get_output_path, json):
    # Load configuration from clif_config.json
    with open('clif_config.json', 'r') as config_file:
        config = json.load(config_file)

    print(f"Site: {config['site']}")
    print(f"Data path: {config['data_directory']}")
    print(f"File type: {config['filetype']}")

    # Set up output directory using standardized helper
    output_dir = get_output_path('preprocessing', '')
    ensure_dir(output_dir)
    print(f"Output directory: {output_dir}")
    return (output_dir,)


@app.cell
def _(os, output_dir, pd):
    # Load ICU cohort from 01_cohort.py
    cohort_path = os.path.join(output_dir, 'icu_cohort.parquet')

    if os.path.exists(cohort_path):
        cohort_df = pd.read_parquet(cohort_path)
        print(f"✅ Cohort loaded successfully: {len(cohort_df)} hospitalizations")
        print(f"Mortality rate: {cohort_df['disposition'].mean():.3f}")

        # Convert datetime columns
        datetime_cols = ['start_dttm', 'hour_24_start_dttm', 'hour_24_end_dttm']
        for _col in datetime_cols:
            cohort_df[_col] = pd.to_datetime(cohort_df[_col])

        print(f"Time range: {cohort_df['start_dttm'].min()} to {cohort_df['start_dttm'].max()}")

        # Extract hospitalization IDs for filtering data loads
        cohort_ids = cohort_df['hospitalization_id'].astype(str).unique().tolist()
        print(f"Extracted {len(cohort_ids)} hospitalization IDs for data filtering")

        # Display sample
        print("\nSample cohort records:")
        print(cohort_df.head())
    else:
        raise FileNotFoundError(f"Cohort file not found at {cohort_path}. Please run 01_cohort.py first.")
    return cohort_df, cohort_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Feature Selection Configuration""")
    return


@app.cell
def _(cohort_ids):
    # Define feature selection configuration
    print("Configuring feature selection...")

    print(f"Processing {len(cohort_ids)} hospitalizations from cohort")

    # Define category filters for each table (only features used in aggregation/derivation)
    category_filters = {
        'vitals': [
            'heart_rate', 'map', 'sbp', 'respiratory_rate', 'spo2', 'temp_c'
        ],
        'labs': [
            "albumin", "alt", "ast", "bicarbonate", "bilirubin_total", "bun", "chloride", "creatinine",
            "inr", "lactate", "platelet_count", "po2_arterial", "potassium", "pt", "ptt",
            "sodium", "wbc"
        ],
        'medication_admin_continuous': [  # Vasoactives only (for vasopressor count)
            "norepinephrine", "epinephrine", "phenylephrine", "vasopressin",
            "dopamine", "dobutamine", "milrinone", "isoproterenol"
        ],
        'respiratory_support': [
            'device_category', 'fio2_set', 'peep_set'
        ],
        'patient_assessments': [
            'gcs_total'
        ]
    }

    print("\nFeature selection configuration:")
    for table, categories in category_filters.items():
        print(f"  {table}: {len(categories)} categories")
    return (category_filters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Initialize ClifOrchestrator and Load Tables""")
    return


@app.cell
def _(ClifOrchestrator, category_filters, cohort_ids):
    # Initialize ClifOrchestrator using config file
    clif = ClifOrchestrator(config_path='clif_config.json')

    # Load required tables with optimized filtering (hospitalization_id + category filtering)
    print(f"Loading required tables with filtering for {len(cohort_ids)} cohort hospitalizations...")
    print("Applying category-level filtering to reduce memory usage...")

    # Define columns and category filters for each table
    table_config = {
        'vitals': {
            'columns': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
            'category_filter': ('vital_category', category_filters['vitals'])
        },
        'labs': {
            'columns': ['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value', 'lab_value_numeric'],
            'category_filter': ('lab_category', category_filters['labs'])
        },
        'respiratory_support': {
            'columns': None,  # Load all columns
            'category_filter': None  # No category filtering (uses column-level filtering instead)
        },
        'medication_admin_continuous': {
            'columns': None,  # Load all columns
            'category_filter': ('med_category', category_filters['medication_admin_continuous'])
        },
        'patient_assessments': {
            'columns': None,  # Load all columns
            'category_filter': ('assessment_category', category_filters['patient_assessments'])
        }
    }

    tables_to_load = ['vitals', 'labs', 'respiratory_support', 'medication_admin_continuous', 'patient_assessments']
    for _table_name in tables_to_load:
        tbl_config = table_config.get(_table_name, {})
        table_columns = tbl_config.get('columns')

        # Build filters dict with hospitalization_id and category filtering
        filters_dict = {'hospitalization_id': cohort_ids}

        # Add category filter if defined
        if tbl_config.get('category_filter'):
            category_col, category_values = tbl_config['category_filter']
            filters_dict[category_col] = category_values
            print(f"Loading {_table_name} table: {len(cohort_ids)} hospitalizations × {len(category_values)} categories = filtered rows only...")
        else:
            print(f"Loading {_table_name} table: {len(cohort_ids)} hospitalizations, {len(table_columns) if table_columns else 'all'} columns...")

        clif.load_table(
            _table_name,
            filters=filters_dict,
            columns=table_columns
        )

    # Load hospitalization table to get admission_dttm for temporal split
    print("Loading hospitalization table with cohort ID filters for temporal split...")
    clif.load_table(
        'hospitalization',
        filters={'hospitalization_id': cohort_ids}
    )

    print("✅ ClifOrchestrator initialized and tables loaded successfully with cohort filtering")
    return (clif,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Apply Outlier Handling""")
    return


@app.cell
def _(apply_outlier_handling, clif):
    # Apply outlier handling to all loaded tables using clifpy
    print("Applying outlier handling to loaded tables...")

    tables_for_outlier_handling = ['vitals', 'labs', 'respiratory_support', 'patient_assessments']

    for _table_name in tables_for_outlier_handling:
        table_obj = getattr(clif, _table_name)
        if table_obj is not None:
            print(f"\nProcessing {_table_name} table:")
            apply_outlier_handling(table_obj)
        else:
            print(f"Warning: {_table_name} table not loaded")

    print("\n✅ Outlier handling completed for all tables")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Create Wide Dataset Using clifpy""")
    return


@app.cell
def _(category_filters, clif, cohort_df, os, output_dir):
    # Create wide dataset for cohort hospitalizations using clifpy
    print("Creating wide dataset using clifpy ClifOrchestrator...")

    # Prepare cohort_df with required columns for time filtering
    # This will significantly reduce memory usage by filtering data to only the 24-hour windows
    cohort_time_filter = cohort_df[['hospitalization_id', 'hour_24_start_dttm', 'hour_24_end_dttm']].copy()
    cohort_time_filter.columns = ['hospitalization_id', 'start_time', 'end_time']  # Rename to match expected columns

    print(f"Using cohort_df time filtering for {len(cohort_time_filter)} hospitalizations")
    print(f"This will filter data to 24-hour windows before creating the wide dataset")

    # Create wide dataset using ClifOrchestrator
    clif.create_wide_dataset(
        category_filters=category_filters,
        cohort_df=cohort_time_filter,  # Pass cohort_df for time window filtering
        save_to_data_location=False,
        batch_size=10000,
        memory_limit='6GB',
        threads=4,
        show_progress=True
    )

    wide_df =clif.wide_df.copy()

    print(f"✅ Wide dataset created successfully")
    print(f"Shape: {clif.wide_df.shape}")
    print(f"Hospitalizations: {wide_df['hospitalization_id'].nunique()}")
    print(f"Date range: {wide_df['event_time'].min()} to {wide_df['event_time'].max()}")

    # Save wide dataset for QC analysis
    wide_df_path = os.path.join(output_dir, 'wide_df_24hr.parquet')
    wide_df.to_parquet(wide_df_path)
    print(f"✅ Saved wide dataset for QC analysis: {wide_df_path}")
    print(f"File size: {os.path.getsize(wide_df_path) / 1024**2:.1f} MB")
    return (wide_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Aggregate to 24-Hour Window (One Row Per Hospitalization)""")
    return


@app.cell
def _(wide_df):
    # Create 24-hour aggregated dataset (one row per hospitalization)
    print("Aggregating event-wide data to one row per hospitalization (24-hour window)...")

    # Define aggregation configuration based on clinical requirements
    # Group aggregations by source column (keys = column names, values = list of (output_name, func) tuples)
    from collections import defaultdict
    agg_config = defaultdict(list)

    # Max aggregations (worst values in 24hr)
    max_features = [
        'lactate', 'bun', 'creatinine', 'ast', 'alt', 'bilirubin_total', 'inr', 'ptt', 'pt',
        'fio2_set', 'peep_set', 'respiratory_rate', 'heart_rate', 'temp_c',
        'sodium', 'potassium', 'wbc', 'chloride', 'bicarbonate'
    ]
    for feat in max_features:
        if feat in wide_df.columns:
            agg_config[feat].append((f'{feat}_max', 'max'))

    # Min aggregations (worst values in 24hr)
    min_features = [
        'platelet_count', 'po2_arterial', 'spo2', 'sbp', 'map', 'albumin',
        'sodium', 'potassium', 'wbc', 'temp_c', 'heart_rate'
    ]
    for feat in min_features:
        if feat in wide_df.columns:
            agg_config[feat].append((f'{feat}_min', 'min'))

    # Median aggregations
    median_features = ['respiratory_rate', 'fio2_set', 'bilirubin_total', 'map']
    for feat in median_features:
        if feat in wide_df.columns:
            agg_config[feat].append((f'{feat}_median', 'median'))

    # Last/most recent value
    last_features = ['gcs_total']
    for feat in last_features:
        if feat in wide_df.columns:
            agg_config[feat].append((f'{feat}_last', 'last'))

    # Convert defaultdict to regular dict
    agg_config = dict(agg_config)

    # Perform aggregation
    total_aggs = sum(len(v) for v in agg_config.values())
    print(f"Aggregating {total_aggs} features from {len(agg_config)} source columns...")

    # Sort by event_time to ensure 'last' aggregation works correctly
    wide_df_sorted = wide_df.sort_values(['hospitalization_id', 'event_time']).copy()

    # Group by hospitalization_id and aggregate
    aggregated_df = wide_df_sorted.groupby('hospitalization_id', as_index=False).agg(agg_config)

    # Flatten column names from multi-level index
    # Result has columns like ('hospitalization_id', ''), ('lactate', 'lactate_max'), etc.
    # Use level 1 (output name) when it exists, otherwise use level 0 (groupby key)
    aggregated_df.columns = [col[1] if col[1] else col[0]
                              for col in aggregated_df.columns.values]

    print("✅ Aggregation complete")
    print(f"Event-wide shape: {wide_df.shape}")
    print(f"Aggregated shape: {aggregated_df.shape}")
    print(f"Unique hospitalizations: {aggregated_df['hospitalization_id'].nunique()}")
    return (aggregated_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Add Device One-Hot Encoding and Vasopressor Count""")
    return


@app.cell
def _(aggregated_df, pd, wide_df):
    # Add one-hot encoding for device_category and vasopressor count
    print("Creating device one-hot encoding and vasopressor count from event-level data...")

    aggregated_with_derived_features = aggregated_df.copy()

    # 1. Device one-hot encoding (exclude Room Air and Other)
    if 'device_category' in wide_df.columns:
        # Get device usage per hospitalization (any occurrence in 24hr)
        device_usage = wide_df.groupby('hospitalization_id')['device_category'].apply(
            lambda x: x.dropna().unique().tolist()
        ).reset_index()

        # Define devices to include (exclude Room Air and Other)
        devices_to_include = ['imv', 'nippv', 'cpap', 'high flow nc', 'face mask', 'trach collar', 'nasal cannula']

        # Create binary columns for each device
        for device in devices_to_include:
            device_col = f'device_{device.replace(" ", "_")}'
            device_usage[device_col] = device_usage['device_category'].apply(
                lambda devices: 1 if any(d.lower() == device for d in devices) else 0
            )

        # Drop the list column
        device_usage = device_usage.drop(columns=['device_category'])

        # Merge with aggregated data
        aggregated_with_derived_features = pd.merge(
            aggregated_with_derived_features,
            device_usage,
            on='hospitalization_id',
            how='left'
        )

        # Fill NaN with 0 for device columns
        _device_cols = [col for col in aggregated_with_derived_features.columns if col.startswith('device_')]
        for col in _device_cols:
            aggregated_with_derived_features[col] = aggregated_with_derived_features[col].fillna(0).astype(int)

        print(f"✅ Added {len(_device_cols)} device one-hot encoded columns")
        print(f"Device columns: {_device_cols}")
    else:
        print("⚠️ device_category not found in wide dataset")

    # 2. Vasopressor count (unique vasopressor classes in 24hr)
    print("\nAdding vasopressor count...")

    # Define vasopressor categories
    vasopressor_categories = [
        'norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin',
        'dopamine', 'dobutamine', 'milrinone', 'isoproterenol'
    ]

    # Get medication columns from wide_df
    med_cols = [col for col in wide_df.columns if col in vasopressor_categories]

    if med_cols:
        # Count unique vasopressors used per hospitalization
        vaso_count = wide_df.groupby('hospitalization_id')[med_cols].apply(
            lambda x: sum(x.notna().any())
        ).reset_index(name='vasopressor_count')

        # Merge with aggregated data
        aggregated_with_derived_features = pd.merge(
            aggregated_with_derived_features,
            vaso_count,
            on='hospitalization_id',
            how='left'
        )

        # Fill NaN with 0
        aggregated_with_derived_features['vasopressor_count'] = aggregated_with_derived_features['vasopressor_count'].fillna(0).astype(int)

        print("✅ Vasopressor count added")
        print(f"Vasopressor count distribution: {aggregated_with_derived_features['vasopressor_count'].value_counts().sort_index().to_dict()}")
    else:
        print("⚠️ No vasopressor columns found in wide dataset")
    return (aggregated_with_derived_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Add Demographics from Cohort""")
    return


@app.cell
def _(aggregated_with_derived_features, cohort_df, pd):
    # Add demographics by inner joining with cohort_df
    print("Adding demographics from cohort to aggregated dataset...")

    # Inner join with cohort_df to add demographic, outcome, and SOFA columns
    # This will also filter out any hospitalizations without demographics
    aggregated_with_demographics = pd.merge(
        aggregated_with_derived_features,
        cohort_df[['hospitalization_id', 'hour_24_start_dttm', 'hour_24_end_dttm',
                   'sex_category', 'ethnicity_category',
                   'race_category', 'language_category', 'disposition',
                   'p_f', 'p_f_imputed', 'sofa_cv_97', 'sofa_coag', 'sofa_liver',
                   'sofa_resp', 'sofa_cns', 'sofa_renal', 'sofa_total']],
        on='hospitalization_id',
        how='inner'  # Inner join filters out any missing demographics
    )

    print("✅ Demographics, disposition, and SOFA scores added to aggregated dataset")
    print(f"Shape before merge: {aggregated_with_derived_features.shape}")
    print(f"Shape after merge: {aggregated_with_demographics.shape}")
    print(f"Hospitalizations with demographics: {aggregated_with_demographics['hospitalization_id'].nunique()}")
    print(f"Mortality rate: {aggregated_with_demographics['disposition'].mean():.3f}")

    # Show demographic distribution
    print("\n=== Demographic Distribution ===")
    print("Sex distribution:")
    print(aggregated_with_demographics['sex_category'].value_counts())
    print("\nRace distribution:")
    print(aggregated_with_demographics['race_category'].value_counts())
    print("\nEthnicity distribution:")
    print(aggregated_with_demographics['ethnicity_category'].value_counts())
    print("\nLanguage distribution:")
    print(aggregated_with_demographics['language_category'].value_counts())
    return (aggregated_with_demographics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Add Binary Sex Feature (isfemale)""")
    return


@app.cell
def _(aggregated_with_demographics):
    # Create binary isfemale feature (1 = female, 0 = not female)
    print("Creating binary sex feature (isfemale)...")

    aggregated_with_sex_binary = aggregated_with_demographics.copy()

    # Create isfemale: 1 if female (case-insensitive), 0 otherwise
    aggregated_with_sex_binary['isfemale'] = aggregated_with_sex_binary['sex_category'].str.lower().apply(
        lambda x: 1 if x == 'female' else 0
    )

    print("✅ Binary sex feature created")
    print(f"isfemale distribution: {aggregated_with_sex_binary['isfemale'].value_counts().to_dict()}")
    print(f"  Female (1): {(aggregated_with_sex_binary['isfemale'] == 1).sum()}")
    print(f"  Not Female (0): {(aggregated_with_sex_binary['isfemale'] == 0).sum()}")
    return (aggregated_with_sex_binary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Add Temporal Split Column""")
    return


@app.cell
def _(aggregated_with_sex_binary, clif, pd):
    # Add temporal split column based on admission_dttm from hospitalization table
    print("Adding temporal split column (row_type)...")

    # Get hospitalization data with admission_dttm and age
    hosp_df = clif.hospitalization.df[['hospitalization_id', 'admission_dttm', 'age_at_admission']].copy()
    hosp_df['hospitalization_id'] = hosp_df['hospitalization_id'].astype(str)
    hosp_df['admission_dttm'] = pd.to_datetime(hosp_df['admission_dttm'])

    # Merge with aggregated dataset to get admission year and age
    aggregated_with_admission = pd.merge(
        aggregated_with_sex_binary,
        hosp_df[['hospitalization_id', 'admission_dttm', 'age_at_admission']],
        on='hospitalization_id',
        how='left'
    )

    # Rename age for clarity
    aggregated_with_admission = aggregated_with_admission.rename(columns={'age_at_admission': 'age'})

    # Create temporal split based on admission year
    # Training: 2018-2022, Testing: 2023-2024
    aggregated_with_admission['admission_year'] = aggregated_with_admission['admission_dttm'].dt.year
    aggregated_with_admission['row_type'] = aggregated_with_admission['admission_year'].apply(
        lambda year: 'train' if 2018 <= year <= 2022 else 'test'
    )

    # Create additional split with validation set
    # Training: 2018-2022, Validation: 2023, Testing: 2024
    aggregated_with_admission['split_type'] = aggregated_with_admission['admission_year'].apply(
        lambda year: 'train' if 2018 <= year <= 2022 else
                     'val' if year == 2023 else
                     'test'
    )

    # Display temporal split summary
    print("\nOriginal temporal split (row_type) summary:")
    split_summary = aggregated_with_admission['row_type'].value_counts()
    print(split_summary)

    print("\nNew temporal split (split_type) summary:")
    split_summary_new = aggregated_with_admission['split_type'].value_counts()
    print(split_summary_new)

    year_summary = aggregated_with_admission.groupby(['admission_year', 'row_type', 'split_type']).size().reset_index(name='count')
    print("\nYear breakdown:")
    print(year_summary)

    # Remove intermediate columns but keep both split columns
    aggregated_final = aggregated_with_admission.drop(columns=['admission_dttm', 'admission_year'])

    print("\n✅ Temporal split columns added (row_type and split_type)")
    print(f"Final aggregated dataset shape: {aggregated_final.shape}")
    return (aggregated_final,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Add Age Bins""")
    return


@app.cell
def _(aggregated_final, pd):
    # Create age bins: lt40, 40_64, 65_79, ge80 (XGBoost-compatible names)
    print("Creating age bins...")

    # Create age bins (using XGBoost-compatible names)
    def create_age_bin(age):
        if pd.isna(age):
            return None
        elif age < 40:
            return 'lt40'
        elif age < 65:
            return '40_64'
        elif age < 80:
            return '65_79'
        else:
            return 'ge80'

    aggregated_with_age_bins = aggregated_final.copy()
    aggregated_with_age_bins['age_bin'] = aggregated_with_age_bins['age'].apply(create_age_bin)

    # One-hot encode age bins
    age_dummies = pd.get_dummies(aggregated_with_age_bins['age_bin'], prefix='age')
    aggregated_with_age_bins = pd.concat([aggregated_with_age_bins, age_dummies], axis=1)

    print("✅ Age bins created and one-hot encoded")
    print(f"Age distribution: {aggregated_with_age_bins['age_bin'].value_counts().to_dict()}")
    print(f"Age bin columns: {[col for col in aggregated_with_age_bins.columns if col.startswith('age_')]}")
    return (aggregated_with_age_bins,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Standardize Categorical Columns""")
    return


@app.cell
def _(aggregated_with_age_bins):
    # Standardize all categorical column values to lowercase for consistency
    print("Standardizing categorical columns to lowercase...")

    # Identify categorical columns (object dtype)
    categorical_cols = aggregated_with_age_bins.select_dtypes(include=['object']).columns.tolist()

    # Exclude specific columns that should not be lowercased
    exclude_cols = ['hospitalization_id']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    print(f"Found {len(categorical_cols)} categorical columns to standardize:")
    print(f"  {categorical_cols}")

    # Show before values
    print("\n=== Before Standardization ===")
    for _col in categorical_cols:
        unique_vals = aggregated_with_age_bins[_col].dropna().unique()[:5]
        print(f"{_col}: {unique_vals.tolist()}")

    # Create standardized dataset
    aggregated_standardized = aggregated_with_age_bins.copy()

    # Convert all categorical columns to lowercase
    for _col in categorical_cols:
        if _col in aggregated_standardized.columns:
            aggregated_standardized[_col] = aggregated_standardized[_col].str.lower()

    # Show after values
    print("\n=== After Standardization ===")
    for _col in categorical_cols:
        unique_vals = aggregated_standardized[_col].dropna().unique()[:5]
        print(f"{_col}: {unique_vals.tolist()}")

    print("\n✅ Categorical columns standardized to lowercase")
    print(f"Final dataset shape: {aggregated_standardized.shape}")
    return (aggregated_standardized,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Ensure Correct Data Types for Multi-Site Compatibility""")
    return


@app.cell
def _(aggregated_standardized, np):
    # Ensure all feature columns have compatible data types for PyTorch/XGBoost
    print("Ensuring correct data types for multi-site compatibility...")

    aggregated_typed = aggregated_standardized.copy()
    aggregated_typed = aggregated_typed.replace([np.inf, -np.inf], np.nan)
    # Convert boolean columns to int (important for PyTorch compatibility)
    bool_cols = aggregated_typed.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        print(f"Converting {len(bool_cols)} boolean columns to int: {bool_cols}")
        aggregated_typed[bool_cols] = aggregated_typed[bool_cols].astype(int)

    # Convert nullable Int32/Int64 to regular int64 (important for SOFA scores and PyTorch)
    nullable_int_cols = aggregated_typed.select_dtypes(include=['Int32', 'Int64']).columns.tolist()
    if nullable_int_cols:
        print(f"Converting {len(nullable_int_cols)} nullable integer columns to int64: {nullable_int_cols}")
        # First convert to Int64 to ensure consistent nullable type, then to regular int64
        for nullable_col in nullable_int_cols:
            aggregated_typed[nullable_col] = aggregated_typed[nullable_col].astype('Int64').astype('int64')

    # Verify no object columns remain (except identifiers and excluded columns)
    object_cols = aggregated_typed.select_dtypes(include=['object']).columns.tolist()
    expected_object_cols = ['hospitalization_id', 'sex_category', 'ethnicity_category',
                            'race_category', 'language_category', 'row_type',
                            'split_type', 'age_bin']
    unexpected_object_cols = [col for col in object_cols if col not in expected_object_cols]

    if unexpected_object_cols:
        print(f"⚠️  Warning: Unexpected object columns found: {unexpected_object_cols}")

    # Print dtype summary
    print("\n=== Data Type Summary ===")
    dtype_counts = aggregated_typed.dtypes.value_counts()
    for dtype, dtype_count in dtype_counts.items():
        print(f"  {dtype}: {dtype_count} columns")

    print("\n✅ Data types verified and corrected")
    return (aggregated_typed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Save Final Aggregated Dataset""")
    return


@app.cell
def _(aggregated_typed, get_output_path, os):
    # Save final aggregated dataset (one row per hospitalization)
    print("Saving final aggregated dataset...")

    # Save to PHI_DATA/preprocessing/
    output_path = get_output_path('preprocessing', 'aggregated_features_24hr.parquet')
    aggregated_typed.to_parquet(output_path, index=False)

    print(f"✅ Aggregated dataset saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024**2:.1f} MB")
    print(f"Shape: {aggregated_typed.shape}")

    # Display final summary
    print("\n=== Final Dataset Summary ===")
    print(f"Total records (hospitalizations): {len(aggregated_typed):,}")
    print(f"Unique hospitalizations: {aggregated_typed['hospitalization_id'].nunique():,}")
    print(f"Total features: {aggregated_typed.shape[1]}")

    # Show detailed temporal split
    if 'row_type' in aggregated_typed.columns:
        print("\n=== Temporal Split Details ===")
        temporal_summary = aggregated_typed.groupby('row_type').size().to_dict()

        print("Hospitalizations by temporal split:")
        for row_type, split_count in temporal_summary.items():
            print(f"  {row_type}: {split_count:,} hospitalizations")

        # Calculate and display prevalence percentages by temporal split
        print("\n=== Mortality Prevalence by Temporal Split ===")

        # Calculate prevalence by temporal split (disposition already in dataset)
        prevalence_by_split = aggregated_typed.groupby('row_type').agg({
            'disposition': ['count', 'sum', 'mean']
        }).round(3)
        prevalence_by_split.columns = ['total_patients', 'deaths', 'mortality_prevalence']

        for row_type, prevalence_row in prevalence_by_split.iterrows():
            prevalence_pct = prevalence_row['mortality_prevalence'] * 100
            print(f"  {row_type}: {prevalence_pct:.1f}% mortality prevalence ({int(prevalence_row['deaths'])}/{int(prevalence_row['total_patients'])} patients)")

    # Show sample columns by category
    print("\n=== Dataset Structure ===")

    # Categorize columns
    _agg_cols = [col for col in aggregated_typed.columns if any(col.endswith(suf) for suf in ['_max', '_min', '_median', '_last'])]
    _device_cols = [col for col in aggregated_typed.columns if col.startswith('device_')]
    _age_cols = [col for col in aggregated_typed.columns if col.startswith('age_')]
    _demo_cols = ['sex_category', 'ethnicity_category', 'race_category', 'language_category']
    _meta_cols = ['hospitalization_id', 'row_type', 'split_type', 'hour_24_start_dttm', 'hour_24_end_dttm']

    print(f"Aggregated features ({len(_agg_cols)}): {_agg_cols[:10]}...")
    print(f"Device one-hot ({len(_device_cols)}): {_device_cols}")
    print(f"Age features ({len(_age_cols)}): {_age_cols}")
    print(f"Demographics ({len([c for c in _demo_cols if c in aggregated_typed.columns])}): {[c for c in _demo_cols if c in aggregated_typed.columns]}")
    print(f"Metadata ({len([c for c in _meta_cols if c in aggregated_typed.columns])}): {[c for c in _meta_cols if c in aggregated_typed.columns]}")

    if 'vasopressor_count' in aggregated_typed.columns:
        print("Derived: ['vasopressor_count', 'isfemale']")

    print("\n🎉 Feature extraction and aggregation completed successfully!")
    print("Dataset ready for modeling with one row per hospitalization.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate Research Paper Reporting Tables (Table 1)""")
    return


@app.cell
def _(aggregated_typed, json):
    # Generate comprehensive statistics for research paper Table 1
    print("Generating comprehensive statistics for research paper reporting...")

    # Load site config for reporting
    with open('clif_config.json', 'r') as report_config_file:
        _report_config = json.load(report_config_file)

    # Initialize reporting structure
    reporting_stats = {
        "site": _report_config.get('site', 'unknown'),
        "cohort_info": {},
        "demographics": {},
        "clinical_severity": {},
        "vitals": {},
        "labs": {},
        "respiratory": {},
        "medications": {},
        "derived_features": {},
        "outcomes": {}
    }

    # Helper function for continuous variables
    def get_continuous_stats(series, var_name):
        """Calculate statistics for continuous variables"""
        return {
            "variable": var_name,
            "n": int(series.notna().sum()),
            "n_missing": int(series.isna().sum()),
            "mean": float(series.mean()) if series.notna().sum() > 0 else None,
            "std": float(series.std()) if series.notna().sum() > 0 else None,
            "median": float(series.median()) if series.notna().sum() > 0 else None,
            "q25": float(series.quantile(0.25)) if series.notna().sum() > 0 else None,
            "q75": float(series.quantile(0.75)) if series.notna().sum() > 0 else None,
            "min": float(series.min()) if series.notna().sum() > 0 else None,
            "max": float(series.max()) if series.notna().sum() > 0 else None
        }

    # Helper function for categorical variables
    def get_categorical_stats(series, var_name):
        """Calculate statistics for categorical variables"""
        value_counts = series.value_counts()
        total = len(series)
        return {
            "variable": var_name,
            "n_total": total,
            "n_missing": int(series.isna().sum()),
            "categories": {
                str(cat): {
                    "count": int(count),
                    "percentage": float(count / total * 100)
                }
                for cat, count in value_counts.items()
            }
        }

    # 1. Cohort Information
    reporting_stats["cohort_info"] = {
        "total_patients": int(len(aggregated_typed)),
        "unique_hospitalizations": int(aggregated_typed['hospitalization_id'].nunique()),
        "temporal_splits": {
            str(split): int(count)
            for split, count in aggregated_typed['row_type'].value_counts().items()
        },
        "detailed_splits": {
            str(split): int(count)
            for split, count in aggregated_typed['split_type'].value_counts().items()
        } if 'split_type' in aggregated_typed.columns else {}
    }

    # 2. Demographics
    reporting_stats["demographics"]["categorical"] = {
        "sex_category": get_categorical_stats(aggregated_typed['sex_category'], 'sex_category'),
        "race_category": get_categorical_stats(aggregated_typed['race_category'], 'race_category'),
        "ethnicity_category": get_categorical_stats(aggregated_typed['ethnicity_category'], 'ethnicity_category'),
        "language_category": get_categorical_stats(aggregated_typed['language_category'], 'language_category'),
        "isfemale": get_categorical_stats(aggregated_typed['isfemale'], 'isfemale')
    }

    # Age statistics
    if 'age' in aggregated_typed.columns:
        reporting_stats["demographics"]["continuous"] = {
            "age": get_continuous_stats(aggregated_typed['age'], 'age')
        }
        # Age bins
        reporting_stats["demographics"]["categorical"]["age_bin"] = get_categorical_stats(
            aggregated_typed['age_bin'], 'age_bin'
        )

    # 3. Clinical Severity (SOFA Scores)
    sofa_cols = ['sofa_total', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_renal', 'sofa_cv_97', 'sofa_cns', 'p_f', 'p_f_imputed']
    reporting_stats["clinical_severity"]["sofa_scores"] = {
        col: get_continuous_stats(aggregated_typed[col], col)
        for col in sofa_cols if col in aggregated_typed.columns
    }

    # 4. Vital Signs
    vital_features = [col for col in aggregated_typed.columns
                     if any(col.startswith(v) for v in ['heart_rate', 'map', 'sbp', 'respiratory_rate', 'spo2', 'temp_c'])]
    reporting_stats["vitals"]["continuous"] = {
        col: get_continuous_stats(aggregated_typed[col], col)
        for col in vital_features
    }

    # 5. Laboratory Values
    lab_features = [col for col in aggregated_typed.columns
                   if any(col.startswith(lab) for lab in ['albumin', 'alt', 'ast', 'bicarbonate', 'bun', 'chloride',
                                                           'creatinine', 'inr', 'lactate', 'platelet_count',
                                                           'po2_arterial', 'potassium', 'pt', 'ptt', 'sodium', 'wbc'])]
    reporting_stats["labs"]["continuous"] = {
        col: get_continuous_stats(aggregated_typed[col], col)
        for col in lab_features
    }

    # 6. Respiratory Support
    respiratory_continuous = [col for col in aggregated_typed.columns
                             if any(col.startswith(r) for r in ['fio2_set', 'peep_set'])]
    reporting_stats["respiratory"]["continuous"] = {
        col: get_continuous_stats(aggregated_typed[col], col)
        for col in respiratory_continuous
    }

    # Device one-hot encoding
    device_cols = [col for col in aggregated_typed.columns if col.startswith('device_')]
    reporting_stats["respiratory"]["categorical"] = {
        col: get_categorical_stats(aggregated_typed[col], col)
        for col in device_cols
    }

    # 7. Medications
    if 'vasopressor_count' in aggregated_typed.columns:
        reporting_stats["medications"]["continuous"] = {
            "vasopressor_count": get_continuous_stats(aggregated_typed['vasopressor_count'], 'vasopressor_count')
        }
        # Also as categorical distribution
        reporting_stats["medications"]["categorical"] = {
            "vasopressor_count": get_categorical_stats(aggregated_typed['vasopressor_count'], 'vasopressor_count')
        }

    # 8. Outcomes
    reporting_stats["outcomes"] = {
        "disposition": get_categorical_stats(aggregated_typed['disposition'], 'disposition'),
        "mortality_by_split": {}
    }

    # Mortality by temporal split
    for split_type in aggregated_typed['row_type'].unique():
        split_data = aggregated_typed[aggregated_typed['row_type'] == split_type]
        reporting_stats["outcomes"]["mortality_by_split"][str(split_type)] = {
            "total": int(len(split_data)),
            "deaths": int(split_data['disposition'].sum()),
            "mortality_rate": float(split_data['disposition'].mean()),
            "mortality_percentage": float(split_data['disposition'].mean() * 100)
        }

    print(f"✅ Reporting statistics generated for {reporting_stats['site']}")
    print(f"Total variables: {sum(len(v.get('continuous', {})) + len(v.get('categorical', {})) for v in reporting_stats.values() if isinstance(v, dict))}")
    return (reporting_stats,)


@app.cell
def _(ensure_dir, get_project_root, json, os, reporting_stats):
    # Save reporting statistics to JSON
    print("Saving reporting statistics to JSON...")

    # Create filename with site name
    site_name = reporting_stats['site'].lower().replace(' ', '_')
    report_filename = f'table1_stats_{site_name}.json'

    # Save to PHASE1_RESULTS_UPLOAD_ME directory
    project_root = get_project_root()
    report_path = os.path.join(project_root, 'PHASE1_RESULTS_UPLOAD_ME', report_filename)
    ensure_dir(report_path)

    with open(report_path, 'w') as output_file:
        json.dump(reporting_stats, output_file, indent=2)

    print(f"✅ Reporting statistics saved to: {report_path}")
    print(f"File size: {os.path.getsize(report_path) / 1024:.1f} KB")

    # Display summary
    print("\n=== Reporting Statistics Summary ===")
    print(f"Site: {reporting_stats['site']}")
    print(f"Total patients: {reporting_stats['cohort_info']['total_patients']:,}")
    print(f"Mortality rate: {reporting_stats['outcomes']['disposition']['categories'].get('1', {}).get('percentage', 0):.1f}%")

    print("\nVariable categories:")
    for category, category_data in reporting_stats.items():
        if isinstance(category_data, dict) and category not in ['cohort_info', 'outcomes']:
            n_continuous = len(category_data.get('continuous', {}))
            n_categorical = len(category_data.get('categorical', {}))
            if n_continuous > 0 or n_categorical > 0:
                print(f"  {category}: {n_continuous} continuous, {n_categorical} categorical")

    print("\n✅ Table 1 statistics ready for research paper and multi-site aggregation!")
    return


if __name__ == "__main__":
    app.run()
