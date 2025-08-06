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

    # Enable marimo CSV data transformer for best performance in marimo UI
    # Use default transformer when running as script
    try:
        alt.data_transformers.enable('marimo_csv')
    except:
        # Fallback to default transformer if marimo_csv is not available
        alt.data_transformers.enable('default')

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
    def calculate_column_stats(col):
        try:
            # Get the column data
            col_data = event_wide_df[col]

            return {
                f"{col}_min": col_data.min(),
                f"{col}_max": col_data.max(),
                f"{col}_mean": col_data.mean(),
                f"{col}_median": col_data.median(),
                f"{col}_missing_pct": (col_data.isna().sum() / len(col_data)) * 100
            }
        except Exception as e:
            print(f"Warning: Could not calculate statistics for {col}: {str(e)}")
            # Return NaN values for failed calculations
            return {
                f"{col}_min": np.nan,
                f"{col}_max": np.nan,
                f"{col}_mean": np.nan,
                f"{col}_median": np.nan,
                f"{col}_missing_pct": np.nan
            }

    # Calculate all statistics
    for stats_col in numeric_columns:
        col_stats = calculate_column_stats(stats_col)
        stats_dict.update(col_stats)

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
    suffixes = ['_min', '_max', '_mean', '_median', '_missing_pct']
    sample_cols = [col for col in summary_df.columns if any(col.endswith(suffix) for suffix in suffixes)][:15]
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
    missing_pct_cols = [col for col in summary_df.columns if col.endswith('_missing_pct')]
    if missing_pct_cols:
        missing_pct_values = summary_df[missing_pct_cols].iloc[0]
        print(f"\nMissing data overview:")
        print(f"  Features with no missing data: {(missing_pct_values == 0).sum()}")
        print(f"  Features with <10% missing: {(missing_pct_values < 10).sum()}")
        print(f"  Features with 10-50% missing: {((missing_pct_values >= 10) & (missing_pct_values < 50)).sum()}")
        print(f"  Features with >50% missing: {(missing_pct_values >= 50).sum()}")
        print(f"  Average missing percentage: {missing_pct_values.mean():.1f}%")

        # Show features with highest and lowest missing percentages
        print(f"\nFeatures with lowest missing data:")
        lowest_missing = missing_pct_values.nsmallest(5)
        for feat_name, pct in lowest_missing.items():
            feature_display_name = feat_name.replace('_missing_pct', '')
            print(f"  {feature_display_name}: {pct:.1f}% missing")

        print(f"\nFeatures with highest missing data:")
        highest_missing = missing_pct_values.nlargest(5)
        for feat_name, pct in highest_missing.items():
            feature_display_name = feat_name.replace('_missing_pct', '')
            print(f"  {feature_display_name}: {pct:.1f}% missing")

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

    def get_feature_category(col, feature_categories):
        for cat, cols in feature_categories.items():
            if col in cols:
                return cat
        return 'other'

    for miss_col in numeric_columns:
        if f"{miss_col}_missing_pct" in summary_df.columns:
            miss_pct_value = summary_df[f"{miss_col}_missing_pct"].iloc[0]
            miss_col_category = get_feature_category(miss_col, feature_categories)

            missing_data.append({
                'feature': miss_col,
                'missing_pct': miss_pct_value,
                'category': miss_col_category,
                'completeness': 100 - miss_pct_value
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

    def get_feature_stats(col, summary_df):
        return {
            'feature': col,
            'min': summary_df[f"{col}_min"].iloc[0],
            'max': summary_df[f"{col}_max"].iloc[0],
            'mean': summary_df[f"{col}_mean"].iloc[0],
            'median': summary_df[f"{col}_median"].iloc[0],
            'missing_pct': summary_df[f"{col}_missing_pct"].iloc[0] if f"{col}_missing_pct" in summary_df.columns else 0
        }

    for stat_col in numeric_columns:
        if all(f"{stat_col}_{stat}" in summary_df.columns for stat in ['min', 'max', 'mean', 'median']):
            stats_data.append(get_feature_stats(stat_col, summary_df))

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
def _(alt, feature_categories, missing_df, mo, pd):
    mo.md(
        r"""
        ### Category-wise Feature Analysis
        """
    ).center()
    
    # Create category-wise summary
    category_summary = []
    for cat_name, cat_features in feature_categories.items():
        if cat_features:
            # Get missing data for features in this category
            category_data = missing_df[missing_df['feature'].isin(cat_features)]
            if not category_data.empty:
                category_summary.append({
                    'category': cat_name,
                    'feature_count': len(cat_features),
                    'avg_missing_pct': category_data['missing_pct'].mean(),
                    'min_missing_pct': category_data['missing_pct'].min(),
                    'max_missing_pct': category_data['missing_pct'].max()
                })
    
    category_summary_df = pd.DataFrame(category_summary)
    
    # Create bar chart for feature counts by category
    category_bar = alt.Chart(category_summary_df).mark_bar().encode(
        x=alt.X('category:N', title='Category'),
        y=alt.Y('feature_count:Q', title='Number of Features'),
        color=alt.Color('category:N', legend=None),
        tooltip=['category:N', 'feature_count:Q']
    ).properties(
        width=400,
        height=300,
        title='Feature Count by Category'
    )
    
    # Create pie chart for feature distribution
    category_pie = alt.Chart(category_summary_df).mark_arc().encode(
        theta=alt.Theta('feature_count:Q'),
        color=alt.Color('category:N', title='Category'),
        tooltip=['category:N', 'feature_count:Q', 
                 alt.Tooltip('avg_missing_pct:Q', format='.1f', title='Avg Missing %')]
    ).properties(
        width=400,
        height=300,
        title='Feature Distribution by Category'
    )
    
    mo.hstack([category_bar, category_pie]).center()
    
    # Display category details
    mo.md("#### Category Statistics").center()
    print("\nCategory Summary:")
    for _, row in category_summary_df.iterrows():
        print(f"  {row['category']}: {row['feature_count']} features, "
              f"avg missing: {row['avg_missing_pct']:.1f}%")
    
    return


@app.cell
def _(mo):
    mo.md(r"### 5. Top Features Analysis")
    return


@app.cell
def _(mo):
    # Create interactive selection for top/bottom features
    # Use fixed value when running as script
    n_features_value = 10

    mo.md("### Top and Bottom Features by Missing Data").center()
    mo.md(f"Showing top {n_features_value} features").center()

    return (n_features_value,)


@app.cell
def _(alt, missing_df, mo, n_features_value):
    # Get top and bottom features based on value
    n_show = n_features_value

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
    mo.md(
        r"""
        ## 24-Hour Pattern Analysis
        
        **Note**: The 24-hour temporal pattern heatmaps have been moved to a separate notebook `04_heatmap.py` for better performance and modularity.
        
        To view the heatmaps showing:
        - Missing data patterns across 24 hours
        - Maximum and minimum value patterns
        
        Please run: `python code/preprocessing/04_heatmap.py`
        """
    )
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
    stat_suffixes = ['_min', '_max', '_mean', '_median', '_missing_pct']
    stat_cols = [col for col in summary_df.columns if any(col.endswith(suffix) for suffix in stat_suffixes)]
    print(f"Total statistics generated: {len(stat_cols)}")

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
    
        **Note**: For 24-hour temporal pattern analysis, please run the separate notebook `04_heatmap.py`.
    
        Next steps: Use these statistics to inform feature selection and preprocessing decisions for model training.
        """
    )
    return


if __name__ == "__main__":
    app.run()
