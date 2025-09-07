import marimo

__generated_with = "0.14.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # ICU Mortality Model - Feature Engineering

    This notebook loads the ICU cohort and creates hourly wide dataset for the first 24 hours of ICU stay.

    ## Objective
    - Load ICU cohort from 01_cohort.ipynb
    - Use pyCLIF to extract features from CLIF tables
    - Create hourly wide dataset for the first 24 hours
    - Filter to encounters with complete 24-hour data
    - Save features for modeling

    ## Feature Sources
    - **Vitals**: All vital_category values
    - **Labs**: All lab_category values
    - **Patient Assessments**: GCS_total, RASS
    - **Respiratory Support**: Mode, FiO2, PEEP, ventilator settings (with one-hot encoding)
    - **Medications**: All vasoactives and sedatives
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Setup and Configuration

    **Memory Management Notes:**
    - This notebook processes a large dataset (54K+ hospitalizations) 
    - If you encounter kernel crashes or memory errors:
      1. Set `USE_SAMPLE_DATA=True` in the configuration cell below
      2. Increase `memory_limit` parameter in the hourly aggregation function
      3. Reduce `batch_size` parameters if needed
    - The hourly aggregation function uses DuckDB for optimal performance and automatically handles batching for large datasets
    """
    )
    return


@app.cell
def _():
    import sys
    import os
    sys.path.append('..')
    from config_helper import get_project_root, ensure_dir, get_output_path, load_config

    import pandas as pd
    import numpy as np
    from clifpy import ClifOrchestrator
    from clifpy.utils.outlier_handler import apply_outlier_handling
    import json
    import warnings
    warnings.filterwarnings('ignore')

    print("=== ICU Mortality Model - Feature Engineering ===")
    print("Setting up environment...")
    return ClifOrchestrator, apply_outlier_handling, ensure_dir, get_output_path, json, load_config, os, pd


@app.cell
def _(load_config, get_output_path, ensure_dir):
    # Load configuration using config_helper
    config = load_config()
    print(f"Site: {config['site']}")
    print(f"Data path: {config['clif2_path']}")
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
        
        # Display sample
        print("\nSample cohort records:")
        print(cohort_df.head())
    else:
        raise FileNotFoundError(f"Cohort file not found at {cohort_path}. Please run 01_cohort.py first.")
    
    return (cohort_df,)


@app.cell
def _(mo):
    mo.md(r"""## Initialize ClifOrchestrator and Load Tables""")
    return


@app.cell
def _(ClifOrchestrator, config):
    # Initialize ClifOrchestrator
    clif = ClifOrchestrator(
        data_directory=config['clif2_path'],
        filetype=config['filetype'],
        timezone="US/Eastern"
    )
    
    # Load required tables for feature engineering
    print("Loading required tables...")
    
    tables_to_load = ['vitals', 'labs', 'respiratory_support', 'medication_admin_continuous']
    for table_name in tables_to_load:
        print(f"Loading {table_name} table...")
        clif.load_table(table_name)
    
    # Load hospitalization table to get admission_dttm for temporal split
    print("Loading hospitalization table for temporal split...")
    clif.load_table('hospitalization')
    
    print("✅ ClifOrchestrator initialized and tables loaded successfully")
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
    
    for table_name in tables_for_outlier_handling:
        table_obj = getattr(clif, table_name)
        if table_obj is not None:
            print(f"\nProcessing {table_name} table:")
            apply_outlier_handling(table_obj)
        else:
            print(f"Warning: {table_name} table not loaded")
    
    print("\n✅ Outlier handling completed for all tables")
    return


@app.cell
def _(mo):
    mo.md(r"""## Feature Extraction Configuration""")
    return


@app.cell
def _(cohort_df):
    # Define feature extraction configuration
    print("Configuring feature extraction...")
    
    # Get hospitalization IDs from cohort
    cohort_ids = cohort_df['hospitalization_id'].astype(str).unique().tolist()
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
    
    print("\nFeature extraction configuration:")
    for table, categories in category_filters.items():
        print(f"  {table}: {len(categories)} categories")
    
    return category_filters, cohort_ids


@app.cell
def _(mo):
    mo.md(r"""## Create Wide Dataset Using clifpy""")
    return


@app.cell
def _(category_filters, clif, cohort_df, cohort_ids):
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
        hospitalization_ids=cohort_ids,
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
    mo.md(r"""## Add Temporal Split Column""")
    return


@app.cell
def _(clif, pd, wide_df):
    # Add temporal split column based on admission_dttm from hospitalization table
    print("Adding temporal split column (row_type)...")
    
    # Get hospitalization data with admission_dttm
    hosp_df = clif.hospitalization.df[['hospitalization_id', 'admission_dttm']].copy()
    hosp_df['hospitalization_id'] = hosp_df['hospitalization_id'].astype(str)
    hosp_df['admission_dttm'] = pd.to_datetime(hosp_df['admission_dttm'])
    
    # Merge with wide dataset to get admission year
    wide_df_with_admission = pd.merge(
        wide_df,
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
    
    # Display temporal split summary
    print("\nTemporal split summary:")
    split_summary = wide_df_with_admission['row_type'].value_counts()
    print(split_summary)
    
    year_summary = wide_df_with_admission.groupby(['admission_year', 'row_type']).size().unstack(fill_value=0)
    print("\nYear breakdown:")
    print(year_summary)
    
    # Remove intermediate columns but keep row_type
    wide_df_final = wide_df_with_admission.drop(columns=['admission_dttm', 'admission_year'])
    
    print(f"\n✅ Temporal split column added")
    print(f"Final dataset shape: {wide_df_final.shape}")
    
    return (wide_df_final,)


@app.cell
def _(mo):
    mo.md(r"""## Save Final Event-Wide Dataset""")
    return


@app.cell
def _(get_output_path, os, wide_df_final):
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
    print(f"Hospitalizations: {wide_df_final['hospitalization_id'].nunique():,}")
    print(f"Columns: {wide_df_final.shape[1]}")
    
    # Show temporal split
    if 'row_type' in wide_df_final.columns:
        print(f"\nTemporal split:")
        print(wide_df_final['row_type'].value_counts())
    
    # Show sample columns
    print(f"\nSample columns:")
    print(wide_df_final.columns[:20].tolist())
    if len(wide_df_final.columns) > 20:
        print(f"... and {len(wide_df_final.columns) - 20} more columns")
    
    print("\n🎉 Feature engineering completed successfully!")
    return


@app.cell
def _(category_filters, convert_wide_to_hourly, wide_df):
    # Define aggregation configuration - FIXED to match available columns
    print("Defining aggregation configuration...")

    # Build aggregation config based on what we actually have
    aggregation_config = {
        # Apply multiple aggregations to vital signs and labs that are actually present
        "max": [_col for _col in category_filters['vitals'] + category_filters['labs'] if _col in wide_df.columns],
        "min": [_col for _col in category_filters['vitals'] + category_filters['labs'] if _col in wide_df.columns],
        # "mean": [_col for _col in category_filters['vitals'] + category_filters['labs'] if _col in wide_df.columns],
        "median": [_col for _col in category_filters['vitals'] + category_filters['labs'] if _col in wide_df.columns],
        # Boolean aggregation for medications (1 if present in hour, 0 otherwise)
        "boolean": [_col for _col in category_filters['medication_admin_continuous'] if _col in wide_df.columns],
        # One-hot encode categorical respiratory support columns
        "one_hot_encode": [_col for _col in ["mode_category", "device_category"] if _col in wide_df.columns]
    }

    # Print what will actually be aggregated
    print("Aggregation configuration:")
    for method, cols in aggregation_config.items():
        print(f"  {method}: {len(cols)} columns")
        if len(cols) <= 10:
            print(f"    {cols}")
        else:
            print(f"    {cols[:5]}...{cols[-2:]} (showing first 5 and last 2)")

    # Convert to hourly using optimized DuckDB function
    print(f"\nProcessing {len(wide_df):,} records to hourly aggregation...")

    hourly_df = convert_wide_to_hourly(
        wide_df, 
        aggregation_config, 
        memory_limit='6GB',      # Set memory limit for DuckDB
        batch_size=10000          # Process in batches for large datasets
    )

    print("✅ Hourly aggregation completed!")

    return (hourly_df,)


@app.cell
def _(hourly_df, wide_df):
    # Performance and Results Summary
    print("=== Hourly Aggregation Results ===")
    print(f"✅ Processing complete!")
    print(f"Input wide dataset: {wide_df.shape[0]:,} records")
    print(f"Output hourly dataset: {hourly_df.shape[0]:,} records") 
    print(f"Columns in hourly dataset: {hourly_df.shape[1]}")
    print(f"Compression ratio: {wide_df.shape[0] / hourly_df.shape[0]:.1f}x fewer records")

    # Show hourly distribution
    hourly_stats = hourly_df.groupby('nth_hour').size()
    print(f"\nHourly record distribution:")
    print(f"  Hours covered: 0 to {hourly_stats.index.max()}")
    print(f"  Average records per hour: {hourly_stats.mean():.0f}")
    print(f"  Records in first 24 hours: {hourly_stats[hourly_stats.index < 24].sum():,}")

    # Show sample of output columns
    print(f"\nSample of aggregated columns:")
    agg_columns = [_col for _col in hourly_df.columns if any(_col.endswith(suffix) for suffix in ['_max', '_min', '_mean', '_boolean'])]
    for _col in agg_columns[:10]:
        non_null_count = hourly_df[_col].notna().sum()
        print(f"  {_col}: {non_null_count:,} non-null values")

    return


@app.cell
def _(apply_outlier_clipping, cohort_df, os, outlier_config, output_dir, pd, wide_df):
    # Note: This filtering step is now redundant if cohort_df was used in create_wide_dataset
    # The data is already filtered to the 24-hour windows during the wide dataset creation
    # However, we'll keep this for backward compatibility and verification

    # Filter wide dataset to 24-hour windows
    print("Filtering to 24-hour windows for event wide data...: Shape:", wide_df.shape)
    cohort_df['hospitalization_id'] = cohort_df['hospitalization_id'].astype(str)
    # Merge with cohort to get time windows
    wide_df_filtered = pd.merge(
        wide_df,
        cohort_df[['hospitalization_id', 'hour_24_start_dttm', 'hour_24_end_dttm', 'disposition']],
        on='hospitalization_id',
        how='inner'
    )

    print(f"After merge with cohort: {len(wide_df_filtered)} records")

    print(f"✅ Filtered to 24-hour windows: {len(wide_df_filtered)} records")
    print(f"Hospitalizations with data: {wide_df_filtered['hospitalization_id'].nunique()}")

    # Show time window validation
    print("\nTime window validation:")
    print(f"All events within window: {((wide_df_filtered['event_time'] >= wide_df_filtered['hour_24_start_dttm']) & (wide_df_filtered['event_time'] <= wide_df_filtered['hour_24_end_dttm'])).all()}")
    print(f"Average records per hospitalization: {len(wide_df_filtered) / wide_df_filtered['hospitalization_id'].nunique():.1f}")
    print('Shape: after filtering:', wide_df_filtered.shape)
    
    # Apply outlier clipping before saving
    print("\n📊 Applying outlier clipping to wide dataset...")
    wide_df_clipped = apply_outlier_clipping(wide_df_filtered, outlier_config, is_hourly=False)
    print(f"Shape after clipping: {wide_df_clipped.shape}")

    wide_df_clipped.to_parquet(os.path.join(output_dir, 'by_event_wide_df.parquet'), index=False)

    return (wide_df_clipped,)


@app.cell
def _(apply_outlier_clipping, cohort_df, hourly_df, os, outlier_config, output_dir, pd):
    # Filter hourly dataset to 24-hour windows
    print("\nFiltering hourly dataset to 24-hour windows...| Shape:",hourly_df.shape)
    # Merge with cohort to get time windows
    hourly_df_filtered = pd.merge(
        hourly_df,
        cohort_df[['hospitalization_id', 'hour_24_start_dttm', 'hour_24_end_dttm', 'disposition']],
        on='hospitalization_id',
        how='inner'
    )

    print(f"After merge with cohort: {len(hourly_df_filtered)} records")

    print(f"✅ Filtered hourly dataset to 24-hour windows: {len(hourly_df_filtered)} records")
    print(f"Hospitalizations with data in hourly dataset: {hourly_df_filtered['hospitalization_id'].nunique()}")

    # Show time window validation for hourly dataset
    print("\nTime window validation for hourly dataset:")
    print(f"All events within window: {((hourly_df_filtered['event_time_hour'] >= hourly_df_filtered['hour_24_start_dttm']) & (hourly_df_filtered['event_time_hour'] <= hourly_df_filtered['hour_24_end_dttm'])).all()}")
    print(f"Average records per hospitalization: {len(hourly_df_filtered) / hourly_df_filtered['hospitalization_id'].nunique():.1f}")

    print('Shape:', hourly_df_filtered.shape)
    
    # Apply outlier clipping before saving
    print("\n📊 Applying outlier clipping to hourly dataset...")
    hourly_df_clipped = apply_outlier_clipping(hourly_df_filtered, outlier_config, is_hourly=True)
    print(f"Shape after clipping: {hourly_df_clipped.shape}")
    
    hourly_df_clipped.to_parquet(os.path.join(output_dir, 'by_hourly_wide_df.parquet'), index=False)

    print(f"\n✅ Feature engineering completed successfully!")
    print(f"Event-level data saved to: {os.path.join(output_dir, 'by_event_wide_df.parquet')}")
    print(f"Hourly data saved to: {os.path.join(output_dir, 'by_hourly_wide_df.parquet')}")

    return (hourly_df_clipped,)


@app.cell
def _(hourly_df_clipped):
    print("Hourly dataset columns:")
    print(hourly_df_clipped.columns.tolist())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
