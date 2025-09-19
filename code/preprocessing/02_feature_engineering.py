import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # ICU Mortality Model - Feature Extraction

    This notebook loads the ICU cohort and creates an event-wide dataset for the first 24 hours of ICU stay.

    ## Objective
    - Load ICU cohort from 01_cohort.py
    - Use clifpy to extract features from CLIF tables
    - Apply outlier handling using built-in clifpy functions
    - Create event-wide dataset for the first 24 hours
    - Add temporal split column (row_type) based on admission year
    - Save features for modeling

    ## Feature Sources
    - **Vitals**: Heart rate, MAP, respiratory rate, SpO2, temperature, weight, height
    - **Labs**: Comprehensive lab panel (albumin, creatinine, hemoglobin, etc.)
    - **Respiratory Support**: Mode, device category, FiO2
    - **Medications**: Vasoactives and sedatives

    ## Temporal Split
    - **Training data**: 2018-2022 admissions
    - **Testing data**: 2023-2024 admissions
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Setup and Configuration

    **Memory Management Notes:**
    - This notebook processes data only for the cohort hospitalizations within their 24-hour windows
    - clifpy automatically handles large datasets with DuckDB optimization
    - Outlier handling is applied using built-in CLIF standard ranges
    - Time filtering reduces memory usage by processing only relevant data
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
        os,
        pd,
    )


@app.cell
def _(ensure_dir, get_output_path, json):
    # Load configuration from clif_config.json
    with open('clif_config.json', 'r') as f:
        config = json.load(f)

    print(f"Site: {config['site']}")
    print(f"Data path: {config['data_directory']}")
    print(f"File type: {config['filetype']}")

    # Set up output directory using standardized helper
    output_dir = get_output_path('preprocessing', '')
    ensure_dir(output_dir)
    print(f"Output directory: {output_dir}")
    return config, output_dir


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


@app.cell
def _(mo):
    mo.md(r"""## Initialize ClifOrchestrator and Load Tables""")
    return


@app.cell
def _(ClifOrchestrator, cohort_ids):
    # Initialize ClifOrchestrator using config file
    clif = ClifOrchestrator(config_path='clif_config.json')

    # Load required tables for feature engineering with cohort ID filtering
    print(f"Loading required tables with filtering for {len(cohort_ids)} cohort hospitalizations...")

    # Define columns needed for each table to optimize memory usage
    columns_to_load = {
        'vitals': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        'labs': ['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value', 'lab_value_numeric'],
        'respiratory_support': None,  # Load all columns
        'medication_admin_continuous': None  # Load all columns
    }

    tables_to_load = ['vitals', 'labs', 'respiratory_support', 'medication_admin_continuous']
    for _table_name in tables_to_load:
        table_columns = columns_to_load.get(_table_name)
        print(f"Loading {_table_name} table with cohort ID filters and {len(table_columns) if table_columns else 'all'} columns...")
        clif.load_table(
            _table_name,
            filters={'hospitalization_id': cohort_ids},
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


@app.cell
def _(mo):
    mo.md(r"""## Apply Outlier Handling""")
    return


@app.cell
def _(apply_outlier_handling, clif):
    # Apply outlier handling to all loaded tables using clifpy
    print("Applying outlier handling to loaded tables...")

    tables_for_outlier_handling = ['vitals', 'labs', 'respiratory_support']

    for _table_name in tables_for_outlier_handling:
        table_obj = getattr(clif, _table_name)
        if table_obj is not None:
            print(f"\nProcessing {_table_name} table:")
            apply_outlier_handling(table_obj)
        else:
            print(f"Warning: {_table_name} table not loaded")

    print("\n✅ Outlier handling completed for all tables")
    return


@app.cell
def _(mo):
    mo.md(r"""## Feature Selection Configuration""")
    return


@app.cell
def _(cohort_ids):
    # Define feature selection configuration
    print("Configuring feature selection...")

    print(f"Processing {len(cohort_ids)} hospitalizations from cohort")

    # Define category filters for each table
    category_filters = {
        'vitals': [  # Common vital signs
            'heart_rate', 'map', 'respiratory_rate', 'spo2', 'temp_c',
            'weight_kg', 'height_cm'
        ],
        'labs': [  # Common lab values
            "albumin", "alkaline_phosphatase", "alt", "ast", "basophils_percent", "basophils_absolute", 
            "bicarbonate", "bilirubin_total", "bilirubin_conjugated", "bilirubin_unconjugated",
            "bun", "calcium_total", "calcium_ionized", "chloride", "creatinine", "crp", 
            "eosinophils_percent", "eosinophils_absolute", "esr", "ferritin", "glucose_fingerstick", 
            "glucose_serum", "hemoglobin", "phosphate", "inr", "lactate", "ldh",
            "lymphocytes_percent", "lymphocytes_absolute", "magnesium", "monocytes_percent", 
            "monocytes_absolute", "neutrophils_percent", "neutrophils_absolute",
            "pco2_arterial", "po2_arterial", "pco2_venous", "ph_arterial", "ph_venous", 
            "platelet_count", "potassium", "procalcitonin", "pt", "ptt", 
            "so2_arterial", "so2_mixed_venous", "so2_central_venous", "sodium",
            "total_protein", "troponin_i", "troponin_t", "wbc"
        ],
        'medication_admin_continuous': [  # Vasoactives and sedatives
            "norepinephrine", "epinephrine", "phenylephrine", "angiotensin", "vasopressin",
            "dopamine", "dobutamine", "milrinone", "isoproterenol",
            "propofol", "dexmedetomidine", "ketamine", "midazolam", "fentanyl",
            "hydromorphone", "morphine", "remifentanil", "pentobarbital", "lorazepam"
        ],
        'respiratory_support': [  # All respiratory support categories
            'mode_category', 'device_category', 'fio2_set'
        ]
    }

    print("\nFeature selection configuration:")
    for table, categories in category_filters.items():
        print(f"  {table}: {len(categories)} categories")
    return (category_filters,)


@app.cell
def _(mo):
    mo.md(r"""## Create Wide Dataset Using clifpy""")
    return


@app.cell
def _(category_filters, clif, cohort_df):
    # Create wide dataset for cohort hospitalizations using clifpy
    print("Creating wide dataset using clifpy ClifOrchestrator...")

    # Prepare cohort_df with required columns for time filtering
    # This will significantly reduce memory usage by filtering data to only the 24-hour windows
    cohort_time_filter = cohort_df[['hospitalization_id', 'hour_24_start_dttm', 'hour_24_end_dttm']].copy()
    cohort_time_filter.columns = ['hospitalization_id', 'start_time', 'end_time']  # Rename to match expected columns

    print(f"Using cohort_df time filtering for {len(cohort_time_filter)} hospitalizations")
    print(f"This will filter data to 24-hour windows before creating the wide dataset")

    # Create wide dataset using ClifOrchestrator
    wide_df = clif.create_wide_dataset(
        tables_to_load=['vitals', 'labs', 'respiratory_support', 'medication_admin_continuous'],
        category_filters=category_filters,
        cohort_df=cohort_time_filter,  # Pass cohort_df for time window filtering
        save_to_data_location=False,
        batch_size=10000,
        memory_limit='6GB',
        threads=4,
        show_progress=True
    )

    print(f"✅ Wide dataset created successfully")
    print(f"Shape: {wide_df.shape}")
    print(f"Hospitalizations: {wide_df['hospitalization_id'].nunique()}")
    print(f"Date range: {wide_df['event_time'].min()} to {wide_df['event_time'].max()}")
    return (wide_df,)


@app.cell
def _(mo):
    mo.md(r"""## Add Demographics from Cohort""")
    return


@app.cell
def _(cohort_df, pd, wide_df):
    # Add demographics by inner joining with cohort_df
    print("Adding demographics from cohort to wide dataset...")
    
    # Inner join with cohort_df to add demographic and time window columns
    # This will also filter out any hospitalizations without demographics
    wide_df_with_demographics = pd.merge(
        wide_df,
        cohort_df[['hospitalization_id', 'hour_24_start_dttm', 'hour_24_end_dttm',
                   'sex_category', 'ethnicity_category', 
                   'race_category', 'language_category']],
        on='hospitalization_id',
        how='inner'  # Inner join filters out any missing demographics
    )
    
    print(f"✅ Demographics added to wide dataset")
    print(f"Shape before demographics: {wide_df.shape}")
    print(f"Shape after demographics: {wide_df_with_demographics.shape}")
    print(f"Hospitalizations with demographics: {wide_df_with_demographics['hospitalization_id'].nunique()}")
    
    # Show demographic distribution
    print("\n=== Demographic Distribution ===")
    print("Sex distribution:")
    print(wide_df_with_demographics['sex_category'].value_counts())
    print("\nRace distribution:")
    print(wide_df_with_demographics['race_category'].value_counts())
    print("\nEthnicity distribution:")
    print(wide_df_with_demographics['ethnicity_category'].value_counts())
    print("\nLanguage distribution:")
    print(wide_df_with_demographics['language_category'].value_counts())
    
    return (wide_df_with_demographics,)


@app.cell
def _(mo):
    mo.md(r"""## Add Temporal Split Column""")
    return


@app.cell
def _(clif, pd, wide_df_with_demographics):
    # Add temporal split column based on admission_dttm from hospitalization table
    print("Adding temporal split column (row_type)...")

    # Get hospitalization data with admission_dttm
    hosp_df = clif.hospitalization.df[['hospitalization_id', 'admission_dttm']].copy()
    hosp_df['hospitalization_id'] = hosp_df['hospitalization_id'].astype(str)
    hosp_df['admission_dttm'] = pd.to_datetime(hosp_df['admission_dttm'])

    # Merge with wide dataset to get admission year
    wide_df_with_admission = pd.merge(
        wide_df_with_demographics,
        hosp_df[['hospitalization_id', 'admission_dttm']],
        on='hospitalization_id',
        how='left'
    )

    # Create temporal split based on admission year
    # Training: 2018-2022, Testing: 2023-2024
    wide_df_with_admission['admission_year'] = wide_df_with_admission['admission_dttm'].dt.year
    wide_df_with_admission['row_type'] = wide_df_with_admission['admission_year'].apply(
        lambda year: 'train' if 2018 <= year <= 2022 else 'test'
    )

    # Create additional split with validation set
    # Training: 2018-2021, Validation: 2022, Testing: 2023-2024
    wide_df_with_admission['split_type'] = wide_df_with_admission['admission_year'].apply(
        lambda year: 'train' if 2018 <= year <= 2021 else
                     'val' if year == 2022 else
                     'test'
    )

    # Display temporal split summary
    print("\nOriginal temporal split (row_type) summary:")
    split_summary = wide_df_with_admission['row_type'].value_counts()
    print(split_summary)

    print("\nNew temporal split (split_type) summary:")
    split_summary_new = wide_df_with_admission['split_type'].value_counts()
    print(split_summary_new)

    year_summary = wide_df_with_admission.groupby(['admission_year', 'row_type', 'split_type']).size().reset_index(name='count')
    print("\nYear breakdown:")
    print(year_summary)

    # Remove intermediate columns but keep both split columns
    wide_df_final = wide_df_with_admission.drop(columns=['admission_dttm', 'admission_year'])

    print(f"\n✅ Temporal split columns added (row_type and split_type)")
    print(f"Final dataset shape: {wide_df_final.shape}")
    return (wide_df_final,)


@app.cell
def _(mo):
    mo.md(r"""## Save Final Event-Wide Dataset""")
    return


@app.cell
def _(get_output_path, os, pd, wide_df_final):
    # Save final event-wide dataset
    print("Saving final event-wide dataset...")

    # Save to protected_outputs/preprocessing/
    output_path = get_output_path('preprocessing', 'by_event_wide_df.parquet')
    wide_df_final.to_parquet(output_path, index=False)

    print(f"✅ Event-wide dataset saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024**2:.1f} MB")
    print(f"Shape: {wide_df_final.shape}")

    # Display final summary
    print("\n=== Final Dataset Summary ===")
    print(f"Total records: {len(wide_df_final):,}")
    print(f"Unique hospitalizations: {wide_df_final['hospitalization_id'].nunique():,}")
    print(f"Columns: {wide_df_final.shape[1]}")

    # Show detailed temporal split with unique hospitalizations
    if 'row_type' in wide_df_final.columns:
        print(f"\n=== Temporal Split Details ===")
        temporal_summary = wide_df_final.groupby('row_type').agg({
            'hospitalization_id': 'nunique',
            'event_time': 'count'
        }).rename(columns={
            'hospitalization_id': 'unique_hospitalizations', 
            'event_time': 'total_records'
        })

        print("Records by temporal split:")
        for row_type, data in temporal_summary.iterrows():
            print(f"  {row_type}: {data['total_records']:,} records from {data['unique_hospitalizations']:,} unique hospitalizations")

        # Calculate and display prevalence percentages by merging with cohort outcome data
        print(f"\n=== Mortality Prevalence by Temporal Split ===")

        # Get unique hospitalization-outcome pairs from cohort
        hosp_outcomes = wide_df_final[['hospitalization_id', 'row_type']].drop_duplicates()

        # Load cohort data to get disposition
        cohort_path_for_prevalence = get_output_path('preprocessing', 'icu_cohort.parquet')
        cohort_for_prevalence = pd.read_parquet(cohort_path_for_prevalence)

        hosp_outcomes_with_disposition = pd.merge(
            hosp_outcomes,
            cohort_for_prevalence[['hospitalization_id', 'disposition']],
            on='hospitalization_id',
            how='left'
        )

        # Calculate prevalence by temporal split
        prevalence_by_split = hosp_outcomes_with_disposition.groupby('row_type').agg({
            'disposition': ['count', 'sum', 'mean']
        }).round(3)
        prevalence_by_split.columns = ['total_patients', 'deaths', 'mortality_prevalence']

        for row_type, data in prevalence_by_split.iterrows():
            prevalence_pct = data['mortality_prevalence'] * 100
            print(f"  {row_type}: {prevalence_pct:.1f}% mortality prevalence ({int(data['deaths'])}/{int(data['total_patients'])} patients)")

    # Show sample columns
    print(f"\n=== Dataset Structure ===")
    print(f"Sample columns (first 20):")
    print(wide_df_final.columns[:20].tolist())
    if len(wide_df_final.columns) > 20:
        print(f"... and {len(wide_df_final.columns) - 20} more columns")

    # Show time range
    print(f"\nTime range: {wide_df_final['event_time'].min()} to {wide_df_final['event_time'].max()}")

    print("\n🎉 Feature extraction completed successfully!")
    return


if __name__ == "__main__":
    app.run()
