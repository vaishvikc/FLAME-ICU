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
    sys.path.append(os.path.join('..', 'src'))

    import pandas as pd
    import numpy as np
    from pyclif import CLIF
    from pyclif.utils.wide_dataset import convert_wide_to_hourly
    import json
    import warnings
    warnings.filterwarnings('ignore')

    print("=== ICU Mortality Model - Feature Engineering ===")
    print("Setting up environment...")
    return CLIF, convert_wide_to_hourly, json, os, pd


@app.cell
def _(json, os):
    def load_config():
        """Load configuration from config.json or config_demo.json"""
        # Try top-level config_demo.json first (new location)
        config_path = os.path.join("..", "..", "config_demo.json")

        # If running from project root, adjust path
        if not os.path.exists(config_path):
            config_path = "config_demo.json"

        if not os.path.exists(config_path):
            # Try config.json in same location
            config_path = os.path.join("..", "..", "config.json")

        if not os.path.exists(config_path):
            # Fallback to local config_demo.json
            config_path = "config_demo.json"

        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = json.load(file)
            print(f"✅ Loaded configuration from {os.path.basename(config_path)}")
        else:
            raise FileNotFoundError("Configuration file not found. Please create config.json or config_demo.json based on the config_template.")

        return config

    # Load configuration
    config = load_config()
    print(f"Site: {config['site']}")
    print(f"Data path: {config['clif2_path']}")
    print(f"File type: {config['filetype']}")

    # Ensure the directory exists
    # Get the current working directory to determine where we're running from
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    # Check if we're in the code/preprocessing directory or the project root
    if cwd.endswith(('code/preprocessing', 'code\\preprocessing')):
        output_dir = os.path.join('..', '..', 'protected_outputs', 'preprocessing')
    else:
        # Assume we're at project root
        output_dir = os.path.join('protected_outputs', 'preprocessing')

    # Convert to absolute path for consistency
    output_dir = os.path.abspath(output_dir)
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    return config, output_dir


@app.cell
def _(CLIF, config):
    # Initialize pyCLIF
    clif = CLIF(
        data_dir=config['clif2_path'],
        filetype=config['filetype'],
        timezone="US/Eastern"
    )

    print("✅ pyCLIF initialized successfully")
    return (clif,)


@app.cell
def _(mo):
    mo.md(r"""## Load ICU Cohort""")
    return


@app.cell
def _(os, output_dir, pd):
    # Load ICU cohort from 01_cohort.ipynb
    cohort_path = os.path.join(output_dir, 'icu_cohort.parquet')

    # Debug path resolution
    print(f"Looking for cohort file at: {cohort_path}")
    print(f"Output dir exists: {os.path.exists(output_dir)}")
    print(f"Files in output dir: {os.listdir(output_dir) if os.path.exists(output_dir) else 'Directory not found'}")

    if os.path.exists(cohort_path):
        cohort_df = pd.read_parquet(cohort_path)
        print(f"✅ Cohort loaded successfully: {len(cohort_df)} hospitalizations")
        print(f"Mortality rate: {cohort_df['disposition'].mean():.3f}")

        # Convert datetime columns
        datetime_cols = ['start_dttm', 'hour_24_start_dttm', 'hour_24_end_dttm']
        for _col in datetime_cols:
            cohort_df[_col] = pd.to_datetime(cohort_df[_col])

        print(f"Time range: {cohort_df['start_dttm'].min()} to {cohort_df['start_dttm'].max()}")

    else:
        raise FileNotFoundError(f"Cohort file not found at {cohort_path}. Please run 01_cohort.ipynb first.")

    # Display sample
    print("\nSample cohort records:")
    print(cohort_df.head())

    return (cohort_df,)


@app.cell
def _(mo):
    mo.md(r"""## Feature Extraction Configuration""")
    return


@app.cell
def _(cohort_df):
    # Define feature extraction configuration
    print("Configuring feature extraction...")

    # OPTION: Set to True for development/testing with smaller dataset
    USE_SAMPLE_DATA = False  # Set to True to use sample for faster processing
    SAMPLE_SIZE = 100  # Number of hospitalizations to sample

    # Get hospitalization IDs from cohort
    if USE_SAMPLE_DATA:
        print(f"⚠️ Using sample data with {SAMPLE_SIZE} hospitalizations for testing")
        cohort_sample = cohort_df.sample(n=min(SAMPLE_SIZE, len(cohort_df)), random_state=42)
        cohort_ids = cohort_sample['hospitalization_id'].astype(str).unique().tolist()
        print(f"Sampled {len(cohort_ids)} hospitalizations from {len(cohort_df)} total")
    else:
        cohort_ids = cohort_df['hospitalization_id'].astype(str).unique().tolist()
        print(f"Using full dataset: {len(cohort_ids)} hospitalizations")

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

    print(f"\nExtracting features for {len(cohort_ids)} hospitalizations")

    return category_filters, cohort_ids


@app.cell  
def _(json, os, pd):
    # Load outlier configuration for data clipping
    outlier_config_path = os.path.join("..", "..", "outlier_config.json")
    
    # If running from project root, adjust path
    if not os.path.exists(outlier_config_path):
        outlier_config_path = "outlier_config.json"
    
    with open(outlier_config_path, 'r') as f:
        outlier_config = json.load(f)
    
    print("✅ Loaded outlier configuration for data clipping")
    
    def apply_outlier_clipping(df, outlier_config, is_hourly=False):
        """
        Apply outlier clipping to dataframe columns based on outlier_config
        
        Args:
            df: DataFrame to clip
            outlier_config: Dictionary with min/max values for each category
            is_hourly: If True, handles aggregated columns with suffixes (_max, _min, _median)
        
        Returns:
            DataFrame with clipped values
        """
        df_clipped = df.copy()
        total_clipped = 0
        
        # Map config categories to actual column categories
        category_mapping = {
            'vital_category': ['heart_rate', 'map', 'respiratory_rate', 'spo2', 'temp_c', 'weight_kg', 'height_cm', 'sbp', 'dbp'],
            'lab_category': list(outlier_config['lab_category'].keys()),
            'med_category': list(outlier_config['med_category'].keys()),
            'respiratory_support': ['fio2_set', 'peep_set', 'tidal_volume_set', 'resp_rate_set']
        }
        
        for config_category, limits in outlier_config.items():
            for column_name, (min_val, max_val) in limits.items():
                if is_hourly:
                    # For hourly data, check columns with aggregation suffixes
                    matching_cols = [col for col in df_clipped.columns 
                                   if col.startswith(column_name) and 
                                   any(col.endswith(suffix) for suffix in ['_max', '_min', '_median', '_mean'])]
                else:
                    # For wide data, check exact column names
                    matching_cols = [col for col in df_clipped.columns if col == column_name]
                
                for col in matching_cols:
                    if col in df_clipped.columns:
                        # Count values outside range before clipping
                        below_min = (df_clipped[col] < min_val).sum()
                        above_max = (df_clipped[col] > max_val).sum()
                        
                        if below_min > 0 or above_max > 0:
                            # Apply clipping
                            df_clipped[col] = df_clipped[col].clip(lower=min_val, upper=max_val)
                            total_clipped += below_min + above_max
                            print(f"  Clipped {col}: {below_min} below min, {above_max} above max")
        
        print(f"✅ Total values clipped: {total_clipped}")
        return df_clipped
    
    return outlier_config, apply_outlier_clipping


@app.cell
def _(mo):
    mo.md(r"""## Create Wide Dataset Using pyCLIF""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Performance Optimization with Cohort Time Filtering

    The `create_wide_dataset` function now supports an optional `cohort_df` parameter that allows filtering data to specific time windows **before** creating the wide dataset. This significantly improves performance and reduces memory usage when you only need data from specific time periods.

    **Benefits:**
    - Reduces data volume before pivoting operations
    - Significantly lower memory usage
    - Faster processing time
    - Particularly useful for ICU mortality models where we only need the first 24 hours

    **Required columns in cohort_df:**
    - `hospitalization_id`: Unique identifier for each hospitalization
    - `start_time`: Start of the time window (datetime)
    - `end_time`: End of the time window (datetime)
    """
    )
    return


@app.cell
def _(category_filters, clif, cohort_df, cohort_ids):
    # Create wide dataset for cohort hospitalizations
    print("Creating wide dataset using pyCLIF...")

    # Prepare cohort_df with required columns for time filtering
    # This will significantly reduce memory usage by filtering data to only the 24-hour windows
    cohort_time_filter = cohort_df[['hospitalization_id', 'hour_24_start_dttm', 'hour_24_end_dttm']].copy()
    cohort_time_filter.columns = ['hospitalization_id', 'start_time', 'end_time']  # Rename to match expected columns

    print(f"Using cohort_df time filtering for {len(cohort_time_filter)} hospitalizations")
    print(f"This will filter data to 24-hour windows before creating the wide dataset")

    wide_df = clif.create_wide_dataset(
        hospitalization_ids=cohort_ids,
        cohort_df=cohort_time_filter,  # Pass cohort_df for time window filtering
        category_filters=category_filters,  
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
def _(wide_df):
    print("Wide dataset columns:")
    print(wide_df.columns.tolist())
    return


@app.cell
def _(wide_df):
    # Display summary of medication columns
    med_cols = ['angiotensin', 'dexmedetomidine', 'dobutamine', 'dopamine', 'epinephrine', 
                'fentanyl', 'hydromorphone', 'ketamine', 'lorazepam', 'midazolam', 'milrinone', 
                'morphine', 'norepinephrine', 'pentobarbital', 'phenylephrine', 'propofol', 'vasopressin']

    available_med_cols = [_col for _col in med_cols if _col in wide_df.columns]
    if available_med_cols:
        print("Medication columns summary:")
        print(wide_df[available_med_cols].describe())
    return


@app.cell
def _(wide_df):
    # Safely inspect a subset of the data to avoid memory issues
    print("Inspecting data sample...")

    # Check dataset size first
    print(f"Wide dataset shape: {wide_df.shape}")
    print(f"Memory usage: {wide_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Sample a specific hospitalization safely
    sample_hosp_id = wide_df['hospitalization_id'].iloc[0]
    print(f"Examining data for hospitalization: {sample_hosp_id}")

    try:
        # Use query method which is more memory efficient for large datasets
        temp = wide_df.query(f"hospitalization_id == '{sample_hosp_id}'")
        print(f"Records for {sample_hosp_id}: {len(temp)}")

        if len(temp) > 0:
            print("Time range:", temp['event_time'].min(), "to", temp['event_time'].max())
            print("Columns with data:", (temp.notna().sum() > 0).sum())

    except Exception as e:
        print(f"Error inspecting data: {str(e)}")
        print("Dataset might be too large for this operation")

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
