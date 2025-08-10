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
    # ICU Mortality Model - Dataset Statistics

    This notebook loads the event-wide dataset and computes comprehensive statistics for the entire dataset.

    ## Objective
    - Load event-wide dataset from 02_feature_engineering.ipynb
    - Calculate min, max, mean, median, and missing percentage for all numeric features
    - Create a single-row summary DataFrame with all statistics
    - Visualize statistics with interactive charts
    - Save results for reference

    ## Statistics Computed
    - **Min/Max**: Minimum and maximum values for each numeric feature
    - **Mean/Median**: Central tendency measures
    - **Missing %**: Percentage of missing values for each feature
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup and Configuration""")
    return


@app.cell
def _():
    import sys
    import os
    sys.path.append(os.path.join('..', 'src'))

    import pandas as pd
    import numpy as np
    import json
    import warnings
    from tqdm import tqdm
    warnings.filterwarnings('ignore')

    print("=== ICU Mortality Model - Dataset Statistics ===")
    print("Computing comprehensive statistics for preprocessed datasets...")

    return json, np, os, pd, tqdm


@app.cell
def _(os):
    # Get the current working directory to determine where we're running from
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    # Check if we're in the code/preprocessing directory or the project root
    if cwd.endswith(('code/preprocessing', 'code\\preprocessing')):
        data_path = os.path.join('..', '..', 'protected_outputs', 'preprocessing')
        output_dir = os.path.join('..', '..', 'protected_outputs', 'preprocessing')
    else:
        # Assume we're at project root
        data_path = os.path.join('protected_outputs', 'preprocessing')
        output_dir = os.path.join('protected_outputs', 'preprocessing')

    # Convert to absolute paths for consistency
    data_path = os.path.abspath(data_path)
    output_dir = os.path.abspath(output_dir)
    output_path = os.path.join(output_dir, 'dataset_statistics.json')

    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    return data_path, output_path


@app.cell
def _(mo):
    mo.md(r"""## Load Event-Wide Dataset""")
    return


@app.cell
def _(data_path, os, pd):
    # Load event-wide dataset from 02_feature_engineering.ipynb
    event_wide_path = os.path.join(data_path, 'by_event_wide_df.parquet')

    try:
        if not os.path.exists(event_wide_path):
            raise FileNotFoundError(f"Event-wide dataset not found at {event_wide_path}. Please run 02_feature_engineering.ipynb first.")

        event_wide_df = pd.read_parquet(event_wide_path)

        print(f"✅ Loaded event-wide dataset: {event_wide_df.shape}")
        print(f"Hospitalizations: {event_wide_df['hospitalization_id'].nunique()}")
        print(f"Time range: {event_wide_df['event_time'].min()} to {event_wide_df['event_time'].max()}")
        print(f"Memory usage: {event_wide_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # Check for required columns
        required_cols = ['hospitalization_id', 'event_time', 'disposition']
        missing_cols = [col for col in required_cols if col not in event_wide_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        print(f"Please ensure that:")
        print(f"  1. You have run 02_feature_engineering.ipynb successfully")
        print(f"  2. The output file exists at: {event_wide_path}")
        print(f"  3. The file contains the required columns")
        raise

    # Display basic info
    print("\nDataset info:")
    print(f"Total records: {len(event_wide_df):,}")
    print(f"Total columns: {len(event_wide_df.columns)}")
    print(f"Mortality rate: {event_wide_df['disposition'].mean():.3f}")

    # Option to use sample data for faster processing during development
    USE_SAMPLE_DATA = False  # Set to True for testing with smaller dataset
    SAMPLE_FRAC = 0.1  # Use 10% of data when sampling

    if USE_SAMPLE_DATA:
        print(f"\n⚠️ Using sample data ({SAMPLE_FRAC*100:.0f}% of hospitalizations) for faster processing")
        sampled_hosp_ids = event_wide_df['hospitalization_id'].drop_duplicates().sample(frac=SAMPLE_FRAC, random_state=42)
        event_wide_df = event_wide_df[event_wide_df['hospitalization_id'].isin(sampled_hosp_ids)]
        print(f"Sampled dataset: {event_wide_df.shape}")

    return (event_wide_df,)


@app.cell
def _(mo):
    mo.md(r"""## Identify Numeric Columns""")
    return


@app.cell
def _(event_wide_df, np):
    # Identify numeric columns for statistics calculation
    # Exclude non-numeric identifier and datetime columns
    exclude_columns = [
        'hospitalization_id', 'event_time', 'hour_24_start_dttm', 
        'hour_24_end_dttm', 'disposition'
    ]

    # Get numeric columns
    numeric_columns = event_wide_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

    print(f"✅ Identified {len(numeric_columns)} numeric columns for statistics")
    print(f"Total features to analyze: {len(numeric_columns)}")

    # Show sample of columns
    print("\nSample numeric columns:")
    sample_cols_info = [(sample_col, event_wide_df[sample_col].notna().sum()) for sample_col in numeric_columns[:10]]
    for sample_col, non_null_count in sample_cols_info:
        print(f"  {sample_col}: {non_null_count:,} non-null values")
    if len(numeric_columns) > 10:
        print(f"  ... and {len(numeric_columns) - 10} more columns")

    # Categorize columns for better visualization
    feature_categories = {
        'vitals': [],
        'labs': [],
        'medications': [],
        'respiratory': [],
        'other': []
    }

    # Common patterns for categorization
    vitals_keywords = ['heart_rate', 'map', 'respiratory_rate', 'spo2', 'temp_c', 'weight', 'height']
    labs_keywords = ['albumin', 'alkaline', 'alt', 'ast', 'bicarbonate', 'bilirubin', 'bun', 
                     'calcium', 'chloride', 'creatinine', 'glucose', 'hemoglobin', 'lactate',
                     'magnesium', 'platelet', 'potassium', 'sodium', 'troponin', 'wbc', 'ph']
    meds_keywords = ['norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin', 
                     'dopamine', 'dobutamine', 'propofol', 'fentanyl', 'midazolam']
    resp_keywords = ['mode_category', 'device_category', 'fio2', 'peep']

    def categorize_column(col):
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in vitals_keywords):
            return 'vitals'
        elif any(keyword in col_lower for keyword in labs_keywords):
            return 'labs'
        elif any(keyword in col_lower for keyword in meds_keywords):
            return 'medications'
        elif any(keyword in col_lower for keyword in resp_keywords):
            return 'respiratory'
        else:
            return 'other'

    for feat_col in numeric_columns:
        col_category = categorize_column(feat_col)
        feature_categories[col_category].append(feat_col)

    print("\nFeature categories:")
    category_info = [(cat, cols) for cat, cols in feature_categories.items() if cols]
    for cat, cols in category_info:
        print(f"  {cat}: {len(cols)} features")

    return (numeric_columns,)


@app.cell
def _(mo):
    mo.md(r"""## Calculate Comprehensive Statistics""")
    return


@app.cell
def _(event_wide_df, numeric_columns, pd, tqdm):
    # Calculate statistics on whole dataset
    print("Calculating comprehensive statistics for all numeric features...")
    print("Computing dataset-wide statistics...")

    # Initialize dictionary to store all statistics
    all_statistics = {}

    # Get dataset info
    n_hospitalizations = event_wide_df['hospitalization_id'].nunique()

    print(f"Processing {len(numeric_columns)} features on entire dataset...")

    # Process each feature
    for feature in tqdm(numeric_columns, desc="Processing features"):
        # Calculate dataset-wide statistics only
        col_data = event_wide_df[feature]
        
        all_statistics[feature] = {
            'min': float(col_data.min()) if not col_data.isna().all() else None,
            'max': float(col_data.max()) if not col_data.isna().all() else None,
            'mean': float(col_data.mean()) if not col_data.isna().all() else None,
            'median': float(col_data.median()) if not col_data.isna().all() else None,
            'missing_pct': float((col_data.isna().sum() / len(col_data)) * 100) if len(col_data) > 0 else 100.0
        }

    print(f"✅ Calculated statistics for {len(numeric_columns)} features")
    print(f"Statistics computed on entire dataset")

    # Add metadata
    all_statistics['_metadata'] = {
        'total_records': len(event_wide_df),
        'total_hospitalizations': n_hospitalizations,
        'total_features_analyzed': len(numeric_columns),
        'overall_mortality_rate': float(event_wide_df['disposition'].mean()),
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }

    return (all_statistics,)


@app.cell
def _(mo):
    mo.md(r"""## Display Statistics Summary""")
    return


@app.cell
def _(all_statistics):
    # Display summary of calculated statistics
    print("=== Statistics Summary ===")

    # Get metadata
    metadata = all_statistics.get('_metadata', {})

    print(f"Dataset size: {metadata.get('total_records', 'N/A'):,} records")
    print(f"Hospitalizations: {metadata.get('total_hospitalizations', 'N/A'):,}")
    print(f"Features analyzed: {metadata.get('total_features_analyzed', 'N/A')}")
    print(f"Overall mortality rate: {metadata.get('overall_mortality_rate', 0):.3f}")

    # Sample statistics for first few features
    print("\nSample statistics (first 5 features):")
    feature_names = [k for k in all_statistics.keys() if k != '_metadata'][:5]

    for feature_name in feature_names:
        stats = all_statistics[feature_name]
        
        print(f"\n{feature_name}:")
        min_val = stats.get('min')
        max_val = stats.get('max')
        mean_val = stats.get('mean')
        median_val = stats.get('median')
        missing_val = stats.get('missing_pct')
        
        print(f"  min: {min_val:.2f}" if min_val is not None else "  min: N/A")
        print(f"  max: {max_val:.2f}" if max_val is not None else "  max: N/A")
        print(f"  mean: {mean_val:.2f}" if mean_val is not None else "  mean: N/A")
        print(f"  median: {median_val:.2f}" if median_val is not None else "  median: N/A")
        print(f"  missing: {missing_val:.1f}%" if missing_val is not None else "  missing: N/A")

    return


@app.cell
def _(mo):
    mo.md(r"""## Save Results""")
    return


@app.cell
def _(all_statistics, json, np, os, output_path):
    # Save statistics to JSON file
    try:
        # Convert any NaN values to None for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, float):
                if np.isnan(obj):
                    return None
                return obj
            return obj

        # Clean the statistics dictionary
        clean_stats = clean_for_json(all_statistics)

        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(clean_stats, f, indent=2)

        print(f"✅ Saved dataset statistics to: {output_path}")

        # Verify file was saved
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"File size: {file_size:.1f} KB")

        # Print summary
        print("\n=== Analysis Complete ===")
        meta_info = all_statistics.get('_metadata', {})
        print(f"Features analyzed: {meta_info.get('total_features_analyzed', 'N/A')}")
        print(f"Total records: {meta_info.get('total_records', 'N/A'):,}")
        print(f"Total hospitalizations: {meta_info.get('total_hospitalizations', 'N/A'):,}")

    except Exception as e:
        print(f"❌ Error saving statistics: {str(e)}")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    This notebook has successfully:
    1. ✅ Loaded the event-wide dataset
    2. ✅ Computed comprehensive statistics for all numeric features on the entire dataset
    3. ✅ Saved statistics to JSON format for downstream analysis

    The statistics include:
    - Dataset-wide statistics (min, max, mean, median, missing%)
    - Metadata about the dataset (record count, hospitalization count, mortality rate)

    Next steps: Use these statistics to inform feature selection and preprocessing decisions for model training.
    """
    )
    return


if __name__ == "__main__":
    app.run()
