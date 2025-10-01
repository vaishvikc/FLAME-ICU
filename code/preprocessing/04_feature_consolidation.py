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
    # ICU Mortality Model - Feature Consolidation

    This notebook consolidates the 24-hour event-wide dataset into one row per hospitalization by aggregating features based on their clinical meaning.

    ## Objective
    - Load event-wide dataset with multiple events per hospitalization (101 columns)
    - Load feature consolidation configuration mapping each column to aggregation strategy
    - Consolidate into one row per hospitalization using appropriate aggregations:
      - **Constants**: age, demographics, identifiers (take first value)
      - **One-hot encode**: device_category (excluding 'other')
      - **Binary transform**: sex_category → is_female
      - **Aggregations**: vitals, labs, medications (min, max, median, mean)
    - Merge with disposition outcome from cohort
    - Save consolidated dataset for modeling

    ## Expected Output
    - **Input**: 101 columns × multiple events per hospitalization
    - **Output**: ~340 columns × 1 row per hospitalization
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup and Configuration Loading""")
    return


@app.cell
def _():
    import sys
    import os
    sys.path.append('..')
    from config_helper import get_output_path, ensure_dir

    import polars as pl
    import json
    import numpy as np
    import altair as alt
    import pandas as pd
    import pickle
    import warnings
    warnings.filterwarnings('ignore')

    # Enable marimo CSV data transformer
    try:
        alt.data_transformers.enable('marimo_csv')
    except:
        alt.data_transformers.enable('default')

    print("=== ICU Mortality Model - Feature Consolidation ===")
    print("Setting up environment...")
    return get_output_path, json, os, pickle, pl


@app.cell
def _(json, os):
    # Load feature consolidation configuration
    # Get the directory of this script to find config file
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    config_path = os.path.join(script_dir, 'feature_consolidation_config.json')

    with open(config_path, 'r') as _config_file:
        config = json.load(_config_file)

    print(f"✅ Configuration loaded successfully")
    print(f"Total input columns accounted for: {sum(config['metadata']['columns_accounted_for'].values())}")
    print(f"Expected output columns: {config['metadata']['expected_output_columns']['total_estimated']}")

    # Display configuration summary
    print(f"\n=== Configuration Summary ===")
    for _config_category, _config_count in config['metadata']['columns_accounted_for'].items():
        print(f"  {_config_category}: {_config_count} columns")
    return (config,)


@app.cell
def _(mo):
    mo.md(r"""## Load Event-Wide Dataset""")
    return


@app.cell
def _(config, get_output_path, os, pl):
    # Load event-wide dataset
    data_path = get_output_path('preprocessing', 'by_event_wide_df.parquet')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Event-wide dataset not found at {data_path}")

    df = pl.read_parquet(data_path)

    print(f"✅ Event-wide dataset loaded: {df.shape}")
    print(f"Unique hospitalizations: {df['hospitalization_id'].n_unique()}")
    print(f"Total events: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Verify all expected columns are present
    actual_columns = set(df.columns)
    expected_columns = set()

    # Gather all expected columns from config
    expected_columns.update(config['constant_features']['columns'])
    expected_columns.update(config['delete_features']['columns'])
    expected_columns.update(config['one_hot_encode']['columns'])
    expected_columns.add('sex_category')

    for category in config['aggregate_features']:
        if category != 'operations' and category != 'description':
            expected_columns.update(config['aggregate_features'][category]['columns'])

    missing_columns = expected_columns - actual_columns
    unexpected_columns = actual_columns - expected_columns

    if missing_columns:
        print(f"⚠️  Missing expected columns: {sorted(missing_columns)}")

    if unexpected_columns:
        print(f"ℹ️  Unexpected columns (will be ignored): {sorted(unexpected_columns)}")
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""## Load Cohort for Disposition""")
    return


@app.cell
def _(get_output_path, os, pl):
    # Load cohort dataset to get disposition
    cohort_path = get_output_path('preprocessing', 'icu_cohort.parquet')

    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Cohort dataset not found at {cohort_path}")

    cohort_df = pl.read_parquet(cohort_path)

    print(f"✅ Cohort dataset loaded: {cohort_df.shape}")
    print(f"Columns: {list(cohort_df.columns)}")
    print(f"Disposition distribution:")
    print(cohort_df['disposition'].value_counts().sort('disposition'))
    return (cohort_df,)


@app.cell
def _(mo):
    mo.md(r"""## Validate Configuration Against Actual Data""")
    return


@app.cell
def _(config, df):
    # Validate that all configured columns exist in the data
    print("=== Configuration Validation ===")

    all_configured_columns = set()
    validation_results = {}

    # Check constant features
    _val_constant_cols = config['constant_features']['columns']
    missing_constant = [col for col in _val_constant_cols if col not in df.columns]
    validation_results['constant'] = {
        'expected': len(_val_constant_cols),
        'found': len(_val_constant_cols) - len(missing_constant),
        'missing': missing_constant
    }
    all_configured_columns.update(_val_constant_cols)

    # Check delete features
    delete_cols = config['delete_features']['columns']
    missing_delete = [col for col in delete_cols if col not in df.columns]
    validation_results['delete'] = {
        'expected': len(delete_cols),
        'found': len(delete_cols) - len(missing_delete),
        'missing': missing_delete
    }
    all_configured_columns.update(delete_cols)

    # Check one-hot encode features
    _val_onehot_cols = config['one_hot_encode']['columns']
    missing_onehot = [col for col in _val_onehot_cols if col not in df.columns]
    validation_results['one_hot'] = {
        'expected': len(_val_onehot_cols),
        'found': len(_val_onehot_cols) - len(missing_onehot),
        'missing': missing_onehot
    }
    all_configured_columns.update(_val_onehot_cols)

    # Check binary transform
    binary_col = 'sex_category'
    validation_results['binary'] = {
        'expected': 1,
        'found': 1 if binary_col in df.columns else 0,
        'missing': [] if binary_col in df.columns else [binary_col]
    }
    all_configured_columns.add(binary_col)

    # Check aggregate features
    total_agg_cols = []
    for _val_category in config['aggregate_features']:
        if _val_category not in ['operations', 'description']:
            category_cols = config['aggregate_features'][_val_category]['columns']
            total_agg_cols.extend(category_cols)

    missing_agg = [col for col in total_agg_cols if col not in df.columns]
    validation_results['aggregate'] = {
        'expected': len(total_agg_cols),
        'found': len(total_agg_cols) - len(missing_agg),
        'missing': missing_agg
    }
    all_configured_columns.update(total_agg_cols)

    # Print validation summary
    print("Configuration validation results:")
    for _val_category, _val_results in validation_results.items():
        print(f"  {_val_category}: {_val_results['found']}/{_val_results['expected']} columns found")
        if _val_results['missing']:
            print(f"    Missing: {_val_results['missing']}")

    print(f"\nTotal configured columns: {len(all_configured_columns)}")
    print(f"Total columns in dataset: {len(df.columns)}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Feature Consolidation Implementation""")
    return


@app.cell
def _(config, df, pl):
    print("=== Starting Feature Consolidation ===")

    # Prepare aggregation expressions
    agg_expressions = []
    final_column_names = []

    # 1. Constant features - take first value
    # Exclude hospitalization_id from aggregation since it's the grouping column
    _const_cols = [col for col in config['constant_features']['columns'] if col in df.columns and col != 'hospitalization_id']
    if _const_cols:
        agg_expressions.extend([pl.col(col).first() for col in _const_cols])
        final_column_names.extend(_const_cols)
        print(f"✅ Added {len(_const_cols)} constant features")

    # 2. Binary transform: sex_category -> is_female
    if 'sex_category' in df.columns:
        agg_expressions.append(
            (pl.col('sex_category').first() == config['binary_transform']['sex_category']['condition']).cast(pl.Int8).alias('is_female')
        )
        final_column_names.append('is_female')
        print("✅ Added is_female binary feature")

    # 3. One-hot encoding for categorical features
    _onehot_cols = [_col_name for _col_name in config['one_hot_encode']['columns'] if _col_name in df.columns]
    for _col_name in _onehot_cols:
        # Get unique values for this column
        unique_values = df[_col_name].unique().drop_nulls().to_list()

        # Filter out 'other' from device_category one-hot encoding (after lowercasing in preprocessing)
        if _col_name == 'device_category':
            unique_values = [v for v in unique_values if str(v).lower() != 'other']

        print(f"One-hot encoding {_col_name}: {unique_values}")

        for _cat_value in unique_values:
            _new_col_name = f"{_col_name}_{_cat_value}".replace(' ', '_').replace('-', '_').lower()
            agg_expressions.append(
                (pl.col(_col_name) == _cat_value).any().cast(pl.Int8).alias(_new_col_name)
            )
            final_column_names.append(_new_col_name)

    print(f"✅ Added one-hot encoded features for {len(_onehot_cols)} categorical columns")

    # 4. Aggregate numeric features with min, max, median, mean
    operations = config['aggregate_features']['operations']

    for _feat_category in config['aggregate_features']:
        if _feat_category in ['operations', 'description']:
            continue

        _category_cols = [_col_name for _col_name in config['aggregate_features'][_feat_category]['columns'] if _col_name in df.columns]
        print(f"Processing {_feat_category}: {len(_category_cols)} features")

        for _col_name in _category_cols:
            for _agg_op in operations:
                _new_col_name = f"{_col_name}_{_agg_op}"

                if _agg_op == 'min':
                    agg_expressions.append(pl.col(_col_name).min().alias(_new_col_name))
                elif _agg_op == 'max':
                    agg_expressions.append(pl.col(_col_name).max().alias(_new_col_name))
                elif _agg_op == 'median':
                    agg_expressions.append(pl.col(_col_name).median().alias(_new_col_name))
                elif _agg_op == 'mean':
                    agg_expressions.append(pl.col(_col_name).mean().alias(_new_col_name))

                final_column_names.append(_new_col_name)

    print(f"✅ Prepared {len(agg_expressions)} aggregation expressions")
    print(f"Expected final columns: {len(final_column_names)}")
    return (agg_expressions,)


@app.cell
def _(agg_expressions, df):
    # Perform the consolidation
    print("Performing consolidation by hospitalization_id...")

    consolidated_df = df.group_by('hospitalization_id').agg(agg_expressions)

    print(f"✅ Consolidation completed")
    print(f"Consolidated shape: {consolidated_df.shape}")
    print(f"Unique hospitalizations: {consolidated_df['hospitalization_id'].n_unique()}")
    return (consolidated_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Add Disposition Outcome""")
    return


@app.cell
def _(cohort_df, consolidated_df):
    # Merge with cohort to add disposition
    print("Adding disposition outcome from cohort...")

    final_df = consolidated_df.join(
        cohort_df.select(['hospitalization_id', 'disposition']),
        on='hospitalization_id',
        how='left'
    )

    print(f"✅ Disposition added")
    print(f"Final shape: {final_df.shape}")
    print(f"Missing dispositions: {final_df['disposition'].null_count()}")

    # Check disposition distribution
    print("\nDisposition distribution:")
    print(final_df['disposition'].value_counts().sort('disposition'))
    return (final_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Quality Checks and Validation""")
    return


@app.cell
def _(final_df):
    print("=== Quality Checks ===")

    # Check for duplicates
    duplicate_count = final_df['hospitalization_id'].n_unique()
    total_rows = len(final_df)
    print(f"Unique hospitalizations: {duplicate_count}")
    print(f"Total rows: {total_rows}")
    print(f"One row per hospitalization: {duplicate_count == total_rows}")

    # Check for missing values
    print(f"\n=== Missing Value Summary ===")
    missing_summary = []
    for _qc_col in final_df.columns:
        _null_count = final_df[_qc_col].null_count()
        _null_pct = (_null_count / len(final_df)) * 100
        if _null_count > 0:
            missing_summary.append({
                'column': _qc_col,
                'null_count': _null_count,
                'null_pct': round(_null_pct, 2)
            })

    if missing_summary:
        # Sort by null percentage descending
        missing_summary.sort(key=lambda x: x['null_pct'], reverse=True)
        print(f"Columns with missing values: {len(missing_summary)}")
        print("Top 10 columns with most missing values:")
        for _missing_item in missing_summary[:10]:
            print(f"  {_missing_item['column']}: {_missing_item['null_count']} ({_missing_item['null_pct']}%)")
    else:
        print("No missing values found!")

    # Data type summary
    print(f"\n=== Data Type Summary ===")
    dtype_counts = {}
    for _qc_col in final_df.columns:
        _dtype_name = str(final_df[_qc_col].dtype)
        dtype_counts[_dtype_name] = dtype_counts.get(_dtype_name, 0) + 1

    for _dtype_name, _dtype_count in sorted(dtype_counts.items()):
        print(f"  {_dtype_name}: {_dtype_count} columns")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Add Split Information and Apply Scaling""")
    return


@app.cell
def _(df, final_df, pl):
    print("=== Adding Split Information ===")

    # Check if split_type already exists
    if 'split_type' not in final_df.columns:
        # Get split_type from original event-wide data (first occurrence per hospitalization)
        split_info = df.group_by('hospitalization_id').agg([
            pl.col('split_type').first().alias('split_type')
        ])

        # Add split_type to consolidated data
        final_df_with_split = final_df.join(
            split_info,
            on='hospitalization_id',
            how='left'
        )
        print(f"✅ Split information added")
    else:
        # split_type already exists, no need to add
        final_df_with_split = final_df
        print(f"✅ Split information already present in data")

    print(f"Split distribution:")
    print(final_df_with_split['split_type'].value_counts().sort('split_type'))
    return (final_df_with_split,)


@app.cell
def _(final_df_with_split):
    print("=== Applying RobustScaler Scaling ===")

    # Import scaling libraries
    from sklearn.preprocessing import RobustScaler

    # Convert to pandas for sklearn compatibility
    df_pandas = final_df_with_split.to_pandas()

    # Identify feature columns (exclude identifiers, targets, and categorical columns)
    exclude_cols = [
        'hospitalization_id',
        'patient_id',
        'disposition',
        'split_type',
        'row_type',
        'ethnicity_category',
        'race_category',
        'language_category'
    ]
    feature_cols = [col for col in df_pandas.columns if col not in exclude_cols]

    print(f"Features to scale: {len(feature_cols)} columns")

    # Prepare data splits
    train_mask = df_pandas['split_type'] == 'train'
    X_train = df_pandas.loc[train_mask, feature_cols]
    X_all = df_pandas[feature_cols]

    print(f"Training data shape: {X_train.shape}")
    print(f"Total data shape: {X_all.shape}")

    # Fit RobustScaler on training data only
    print("Fitting RobustScaler on training data...")
    transformer = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0)
    )
    transformer.fit(X_train)

    # Transform all data
    print("Transforming all data...")
    X_scaled = transformer.transform(X_all)

    # Create scaled dataframe
    df_scaled = df_pandas.copy()
    df_scaled[feature_cols] = X_scaled

    print("✅ Scaling completed")
    print(f"Scaled features shape: {X_scaled.shape}")
    return df_scaled, feature_cols, transformer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Save Preprocessed Data""")
    return


@app.cell
def _(df_scaled, feature_cols, get_output_path, json, pickle, transformer):
    print("=== Saving Preprocessed Data ===")

    # Save scaled consolidated features
    output_path = get_output_path('preprocessing', 'consolidated_features.parquet')
    df_scaled.to_parquet(output_path, index=False)
    print(f"✅ Scaled consolidated features saved to: {output_path}")

    # Save transformer
    transformer_path = get_output_path('preprocessing', 'preprocessing_transformer.pkl')
    with open(transformer_path, 'wb') as f:
        pickle.dump(transformer, f)
    print(f"✅ RobustScaler saved to: {transformer_path}")

    # Save feature names
    feature_names_path = get_output_path('preprocessing', 'feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"✅ Feature names saved to: {feature_names_path}")

    # Save preprocessing metadata
    preprocessing_info = {
        'scaling_method': 'RobustScaler',
        'scaling_params': {
            'with_centering': True,
            'with_scaling': True,
            'quantile_range': [25.0, 75.0]
        },
        'n_features': len(feature_cols),
        'training_data_shape': df_scaled[df_scaled['split_type'] == 'train'].shape,
        'total_data_shape': df_scaled.shape,
        'feature_names': feature_cols
    }

    metadata_path = get_output_path('preprocessing', 'preprocessing_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"✅ Preprocessing metadata saved to: {metadata_path}")

    print(f"\n=== Final Dataset Summary ===")
    print(f"Total shape: {df_scaled.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Split distribution:")
    print(df_scaled['split_type'].value_counts())
    print(f"Disposition distribution:")
    print(df_scaled['disposition'].value_counts())
    return


if __name__ == "__main__":
    app.run()
