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
    mo.md(r"## Setup and Configuration")
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
    import altair as alt
    warnings.filterwarnings('ignore')

    # Enable marimo CSV data transformer for best performance
    alt.data_transformers.enable('marimo_csv')

    print("=== ICU Mortality Model - Dataset Statistics ===")
    print("Computing comprehensive statistics for preprocessed datasets...")

    return alt, np, os, pd


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
    output_path = os.path.join(output_dir, 'dataset_statistics.parquet')

    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    return data_path, output_dir, output_path


@app.cell
def _(mo):
    mo.md(r"## Load Event-Wide Dataset")
    return


@app.cell
def _(data_path, os, pd):
    # Load event-wide dataset from 02_feature_engineering.ipynb
    event_wide_path = os.path.join(data_path, 'by_event_wide_df.parquet')

    if os.path.exists(event_wide_path):
        event_wide_df = pd.read_parquet(event_wide_path)

        print(f"✅ Loaded event-wide dataset: {event_wide_df.shape}")
        print(f"Hospitalizations: {event_wide_df['hospitalization_id'].nunique()}")
        print(f"Time range: {event_wide_df['event_time'].min()} to {event_wide_df['event_time'].max()}")
        print(f"Memory usage: {event_wide_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    else:
        raise FileNotFoundError(f"Event-wide dataset not found at {event_wide_path}. Please run 02_feature_engineering.ipynb first.")

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
    mo.md(r"## Identify Numeric Columns")
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
    numeric_columns = [_col for _col in numeric_columns if _col not in exclude_columns]

    print(f"✅ Identified {len(numeric_columns)} numeric columns for statistics")
    print(f"Total features to analyze: {len(numeric_columns)}")

    # Show sample of columns
    print("\nSample numeric columns:")
    for _i, _col in enumerate(numeric_columns[:10]):
        non_null_count = event_wide_df[_col].notna().sum()
        print(f"  {_col}: {non_null_count:,} non-null values")
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

    for _col in numeric_columns:
        col_lower = _col.lower()
        if any(keyword in col_lower for keyword in vitals_keywords):
            feature_categories['vitals'].append(_col)
        elif any(keyword in col_lower for keyword in labs_keywords):
            feature_categories['labs'].append(_col)
        elif any(keyword in col_lower for keyword in meds_keywords):
            feature_categories['medications'].append(_col)
        elif any(keyword in col_lower for keyword in resp_keywords):
            feature_categories['respiratory'].append(_col)
        else:
            feature_categories['other'].append(_col)

    print("\nFeature categories:")
    for _category, _cols in feature_categories.items():
        if _cols:
            print(f"  {_category}: {len(_cols)} features")

    return feature_categories, numeric_columns


@app.cell
def _(mo):
    mo.md(r"## Calculate Comprehensive Statistics")
    return


@app.cell
def _(event_wide_df, np, numeric_columns):
    # Calculate statistics for all numeric columns
    print("Calculating comprehensive statistics for all numeric features...")

    # Initialize dictionary to store all statistics
    stats_dict = {}

    # Calculate statistics for each numeric column
    for _col in numeric_columns:
        try:
            # Get the column data
            col_data = event_wide_df[_col]

            # Calculate statistics
            stats_dict[f"{_col}_min"] = col_data.min()
            stats_dict[f"{_col}_max"] = col_data.max()
            stats_dict[f"{_col}_mean"] = col_data.mean()
            stats_dict[f"{_col}_median"] = col_data.median()
            stats_dict[f"{_col}_missing_pct"] = (col_data.isna().sum() / len(col_data)) * 100

        except Exception as e:
            print(f"Warning: Could not calculate statistics for {_col}: {str(e)}")
            # Set NaN values for failed calculations
            stats_dict[f"{_col}_min"] = np.nan
            stats_dict[f"{_col}_max"] = np.nan
            stats_dict[f"{_col}_mean"] = np.nan
            stats_dict[f"{_col}_median"] = np.nan
            stats_dict[f"{_col}_missing_pct"] = np.nan

    print(f"✅ Calculated statistics for {len(numeric_columns)} features")
    print(f"Total statistics computed: {len(stats_dict)}")
    print(f"Statistics per feature: 5 (min, max, mean, median, missing%)")

    return (stats_dict,)


@app.cell
def _(mo):
    mo.md(r"## Create Single-Row Summary DataFrame")
    return


@app.cell
def _(event_wide_df, numeric_columns, pd, stats_dict):
    # Create single-row DataFrame with all statistics
    print("Creating single-row summary DataFrame...")

    # Convert statistics dictionary to single-row DataFrame
    summary_df = pd.DataFrame([stats_dict])

    # Add metadata columns
    summary_df['total_records'] = len(event_wide_df)
    summary_df['total_hospitalizations'] = event_wide_df['hospitalization_id'].nunique()
    summary_df['total_features_analyzed'] = len(numeric_columns)
    summary_df['overall_mortality_rate'] = event_wide_df['disposition'].mean()
    summary_df['analysis_timestamp'] = pd.Timestamp.now()

    print(f"✅ Created summary DataFrame: {summary_df.shape}")
    print(f"Total columns in summary: {len(summary_df.columns)}")

    # Display sample of statistics
    print("\nSample statistics (first few features):")
    sample_cols = [_col for _col in summary_df.columns if any(_col.endswith(suffix) for suffix in ['_min', '_max', '_mean', '_median', '_missing_pct'])][:15]
    if sample_cols:
        print(summary_df[sample_cols].T.to_string())

    return (summary_df,)


@app.cell
def _(mo):
    mo.md(r"## Summary Statistics Overview")
    return


@app.cell
def _(summary_df):
    # Provide overview of the statistics
    print("=== Dataset Statistics Overview ===")

    # Dataset metadata
    print(f"Dataset size: {summary_df['total_records'].iloc[0]:,} records")
    print(f"Hospitalizations: {summary_df['total_hospitalizations'].iloc[0]:,}")
    print(f"Features analyzed: {summary_df['total_features_analyzed'].iloc[0]}")
    print(f"Overall mortality rate: {summary_df['overall_mortality_rate'].iloc[0]:.3f}")

    # Statistics summary
    missing_pct_cols = [_col for _col in summary_df.columns if _col.endswith('_missing_pct')]
    if missing_pct_cols:
        missing_values = summary_df[missing_pct_cols].iloc[0]
        print(f"\nMissing data overview:")
        print(f"  Features with no missing data: {(missing_values == 0).sum()}")
        print(f"  Features with <10% missing: {(missing_values < 10).sum()}")
        print(f"  Features with 10-50% missing: {((missing_values >= 10) & (missing_values < 50)).sum()}")
        print(f"  Features with >50% missing: {(missing_values >= 50).sum()}")
        print(f"  Average missing percentage: {missing_values.mean():.1f}%")

        # Show features with highest and lowest missing percentages
        print(f"\nFeatures with lowest missing data:")
        lowest_missing = missing_values.nsmallest(5)
        for feature, pct in lowest_missing.items():
            feature_name = feature.replace('_missing_pct', '')
            print(f"  {feature_name}: {pct:.1f}% missing")

        print(f"\nFeatures with highest missing data:")
        highest_missing = missing_values.nlargest(5)
        for feature, pct in highest_missing.items():
            feature_name = feature.replace('_missing_pct', '')
            print(f"  {feature_name}: {pct:.1f}% missing")

    return


@app.cell
def _(mo):
    mo.md(r"## Interactive Visualizations")
    return


@app.cell
def _(mo):
    mo.md(r"### 1. Overview Dashboard")
    return


@app.cell
def _(mo, summary_df):
    # Create summary cards for key metrics
    total_records = summary_df['total_records'].iloc[0]
    total_hosp = summary_df['total_hospitalizations'].iloc[0]
    total_features = summary_df['total_features_analyzed'].iloc[0]
    mortality_rate = summary_df['overall_mortality_rate'].iloc[0]

    # Create styled cards
    overview_cards = mo.hstack([
        mo.stat(
            label="Total Records",
            value=f"{total_records:,}",
            caption="Event-level observations"
        ),
        mo.stat(
            label="Hospitalizations",
            value=f"{total_hosp:,}",
            caption="Unique ICU stays"
        ),
        mo.stat(
            label="Features Analyzed",
            value=f"{total_features}",
            caption="Numeric columns"
        ),
        mo.stat(
            label="Mortality Rate",
            value=f"{mortality_rate:.1%}",
            caption="Overall ICU mortality"
        )
    ])

    mo.md("### Dataset Overview").center()
    overview_cards.center()

    return


@app.cell
def _(mo):
    mo.md(r"### 2. Missing Data Analysis")
    return


@app.cell
def _(alt, feature_categories, mo, numeric_columns, pd, summary_df):
    # Prepare data for missing data visualization
    missing_data = []
    for _col in numeric_columns:
        if f"{_col}_missing_pct" in summary_df.columns:
            missing_pct = summary_df[f"{_col}_missing_pct"].iloc[0]

            # Determine category
            _category = 'other'
            for _cat, _cols in feature_categories.items():
                if _col in _cols:
                    _category = _cat
                    break

            missing_data.append({
                'feature': _col,
                'missing_pct': missing_pct,
                'category': _category,
                'completeness': 100 - missing_pct
            })

    missing_df = pd.DataFrame(missing_data)

    # Create histogram of missing percentages
    missing_hist = alt.Chart(missing_df).mark_bar().encode(
        x=alt.X('missing_pct:Q', bin=alt.Bin(step=10), title='Missing Percentage (%)'),
        y=alt.Y('count()', title='Number of Features'),
        color=alt.Color('category:N', title='Feature Category'),
        tooltip=['count()', 'category:N']
    ).properties(
        width=600,
        height=300,
        title='Distribution of Missing Data Across Features'
    ).interactive()

    # Create scatter plot of features sorted by missing percentage
    missing_scatter = alt.Chart(missing_df).mark_circle(size=60).encode(
        x=alt.X('feature:N', sort='-y', title='Feature', axis=alt.Axis(labels=False)),
        y=alt.Y('missing_pct:Q', title='Missing Percentage (%)'),
        color=alt.Color('category:N', title='Category'),
        tooltip=['feature:N', alt.Tooltip('missing_pct:Q', format='.1f'), 'category:N']
    ).properties(
        width=800,
        height=400,
        title='Missing Data by Feature (Hover for details)'
    ).interactive()

    mo.md("### Missing Data Analysis").center()
    mo.vstack([missing_hist, missing_scatter]).center()

    return (missing_df,)


@app.cell
def _(mo):
    mo.md(r"### 3. Feature Statistics Distribution")
    return


@app.cell
def _(alt, mo, numeric_columns, pd, summary_df):
    # Prepare data for statistics visualization
    stats_data = []
    for _col in numeric_columns:
        if all(f"{_col}_{stat}" in summary_df.columns for stat in ['min', 'max', 'mean', 'median']):
            stats_data.append({
                'feature': _col,
                'min': summary_df[f"{_col}_min"].iloc[0],
                'max': summary_df[f"{_col}_max"].iloc[0],
                'mean': summary_df[f"{_col}_mean"].iloc[0],
                'median': summary_df[f"{_col}_median"].iloc[0],
                'missing_pct': summary_df[f"{_col}_missing_pct"].iloc[0] if f"{_col}_missing_pct" in summary_df.columns else 0
            })

    stats_viz_df = pd.DataFrame(stats_data)

    # Filter out features with all NaN values for better visualization
    stats_viz_df = stats_viz_df[stats_viz_df[['min', 'max', 'mean', 'median']].notna().any(axis=1)]

    # Create scatter plot comparing mean vs missing percentage
    mean_vs_missing = alt.Chart(stats_viz_df).mark_circle(size=100).encode(
        x=alt.X('mean:Q', title='Mean Value', scale=alt.Scale(zero=False)),
        y=alt.Y('missing_pct:Q', title='Missing Percentage (%)'),
        color=alt.Color('missing_pct:Q', scale=alt.Scale(scheme='viridis'), title='Missing %'),
        tooltip=['feature:N', alt.Tooltip('mean:Q', format='.2f'), 
                 alt.Tooltip('missing_pct:Q', format='.1f')]
    ).properties(
        width=600,
        height=400,
        title='Feature Mean Values vs Missing Data'
    ).interactive()

    # Create box plot style visualization for min/max/mean/median
    # Sample top 20 features with lowest missing data for clarity
    top_features = stats_viz_df.nsmallest(20, 'missing_pct')

    # Melt the dataframe for box plot visualization
    melted_stats = pd.melt(
        top_features[['feature', 'min', 'max', 'mean', 'median']], 
        id_vars=['feature'], 
        var_name='statistic', 
        value_name='value'
    )

    stats_comparison = alt.Chart(melted_stats).mark_point(filled=True).encode(
        x=alt.X('feature:N', title='Feature', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('value:Q', title='Value', scale=alt.Scale(zero=False)),
        color=alt.Color('statistic:N', title='Statistic'),
        shape=alt.Shape('statistic:N'),
        tooltip=['feature:N', 'statistic:N', alt.Tooltip('value:Q', format='.2f')]
    ).properties(
        width=800,
        height=400,
        title='Feature Statistics Comparison (Top 20 Complete Features)'
    ).interactive()

    mo.md("### Feature Statistics Analysis").center()
    mo.vstack([mean_vs_missing, stats_comparison]).center()

    return


@app.cell
def _(mo):
    mo.md(r"### 4. Category-wise Analysis")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Category-wise Feature Analysis").center()
        mo.hstack([category_bar, category_pie]).center()
    
        # Display category details
        mo.md("#### Category Statistics
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"### 5. Top Features Analysis")
    return


@app.cell
def _(mo):
    # Create interactive selection for top/bottom features
    n_features_slider = mo.ui.slider(
        start=5, 
        stop=20, 
        value=10, 
        step=5,
        label="Number of features to show"
    )

    mo.md("### Top and Bottom Features by Missing Data").center()
    n_features_slider.center()

    return (n_features_slider,)


@app.cell
def _(alt, missing_df, mo, n_features_slider):
    # Get top and bottom features based on slider value
    n_show = n_features_slider.value

    top_complete = missing_df.nsmallest(n_show, 'missing_pct')
    top_missing = missing_df.nlargest(n_show, 'missing_pct')

    # Create horizontal bar charts
    complete_chart = alt.Chart(top_complete).mark_bar(color='#2a7e3b').encode(
        x=alt.X('completeness:Q', title='Data Completeness (%)'),
        y=alt.Y('feature:N', sort='-x', title='Feature'),
        tooltip=['feature:N', alt.Tooltip('completeness:Q', format='.1f', title='Completeness %'),
                 alt.Tooltip('missing_pct:Q', format='.1f', title='Missing %')]
    ).properties(
        width=400,
        height=300,
        title=f'Top {n_show} Most Complete Features'
    )

    missing_chart = alt.Chart(top_missing).mark_bar(color='#e74c3c').encode(
        x=alt.X('missing_pct:Q', title='Missing Percentage (%)'),
        y=alt.Y('feature:N', sort='-x', title='Feature'),
        tooltip=['feature:N', alt.Tooltip('missing_pct:Q', format='.1f', title='Missing %'),
                 alt.Tooltip('completeness:Q', format='.1f', title='Completeness %')]
    ).properties(
        width=400,
        height=300,
        title=f'Top {n_show} Most Missing Features'
    )

    mo.hstack([complete_chart, missing_chart]).center()

    return


@app.cell
def _(mo):
    mo.md(r"## Save Results")
    return


@app.cell
def _(os, output_dir, output_path, summary_df):
    # Save summary statistics to parquet file
    try:
        summary_df.to_parquet(output_path, index=False)
        print(f"✅ Saved dataset statistics to: {output_path}")

        # Verify file was saved
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"File size: {file_size:.1f} KB")

    except Exception as e:
        print(f"❌ Error saving statistics: {str(e)}")
        # Fallback to CSV if parquet fails
        csv_path = os.path.join(output_dir, 'dataset_statistics.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"✅ Saved as CSV instead: {csv_path}")

    print("\n=== Analysis Complete ===") 
    print(f"Summary DataFrame shape: {summary_df.shape}")
    print(f"Features analyzed: {summary_df['total_features_analyzed'].iloc[0]}")
    print(f"Total statistics generated: {len([_col for _col in summary_df.columns if any(_col.endswith(suffix) for suffix in ['_min', '_max', '_mean', '_median', '_missing_pct'])])}")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary
    
        This notebook has successfully:
        1. ✅ Loaded the event-wide dataset
        2. ✅ Computed comprehensive statistics for all numeric features
        3. ✅ Created interactive visualizations for data exploration
        4. ✅ Saved statistics for downstream analysis
    
        The interactive visualizations provide insights into:
        - Missing data patterns across features
        - Feature value distributions
        - Category-wise analysis
        - Top complete and incomplete features
    
        Next steps: Use these statistics to inform feature selection and preprocessing decisions for model training.
        """
    )
    return


if __name__ == "__main__":
    app.run()
