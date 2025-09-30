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
    # ICU Mortality Model - Hourly Quality Control Analysis

    This notebook performs hourly quality control analysis of all features across 24-hour ICU stays.

    ## Objective
    - Load event-wide dataset with 24-hour windows per hospitalization
    - Bucket events into hourly bins (0-23) within each hospitalization's window
    - Calculate coverage statistics: % of hospitalizations with data by hour
    - Create heatmaps grouped by feature category (vitals, labs, medications, respiratory, other)
    - Generate Table 1 summary statistics (missing%, min, max, median, Q1, Q3)
    - Save results to share_to_box/qc folder
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setup and Configuration""")
    return


@app.cell
def _():
    import sys
    import os
    sys.path.append(os.path.join('..', 'src'))

    import polars as pl
    import altair as alt
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    # Enable marimo CSV data transformer
    try:
        alt.data_transformers.enable('marimo_csv')
    except:
        alt.data_transformers.enable('default')

    print("=== ICU Hourly Quality Control Analysis ===")
    return alt, os, pl


@app.cell
def _(os):
    # Set up paths
    cwd = os.getcwd()
    if cwd.endswith(('code/preprocessing', 'code\\preprocessing')):
        data_path = os.path.join('..', '..', 'protected_outputs', 'preprocessing')
        output_path = os.path.join('..', '..', 'share_to_box')
    else:
        data_path = os.path.join('protected_outputs', 'preprocessing')
        output_path = os.path.join('share_to_box')

    data_path = os.path.abspath(data_path)
    output_path = os.path.abspath(output_path)
    qc_path = os.path.join(output_path, 'qc')
    graphs_path = os.path.join(qc_path, 'graphs')

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(qc_path, exist_ok=True)
    os.makedirs(graphs_path, exist_ok=True)

    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print(f"QC path: {qc_path}")
    print(f"Graphs path: {graphs_path}")
    return data_path, graphs_path, output_path, qc_path


@app.cell
def _(mo):
    mo.md(r"""## Load Event-Wide Dataset""")
    return


@app.cell
def _(data_path, os, pl):
    # Load event-wide dataset using Polars
    event_wide_path = os.path.join(data_path, 'by_event_wide_df.parquet')

    if not os.path.exists(event_wide_path):
        raise FileNotFoundError(f"Event-wide dataset not found at {event_wide_path}")

    event_wide_df = pl.read_parquet(event_wide_path)

    print(f"✅ Loaded dataset: {event_wide_df.shape}")
    print(f"Hospitalizations: {event_wide_df['hospitalization_id'].n_unique()}")
    print(f"Columns: {len(event_wide_df.columns)}")
    return (event_wide_df,)


@app.cell
def _(mo):
    mo.md(r"""## Calculate Hourly Buckets Per Hospitalization""")
    return


@app.cell
def _(event_wide_df, pl):
    # Calculate hour buckets within each hospitalization's 24-hour window
    df_with_hours = event_wide_df.with_columns([
        # Drop timezone info to avoid timezone mismatch
        pl.col('event_time').dt.replace_time_zone(None).alias('event_time_no_tz'),
        pl.col('hour_24_start_dttm').dt.replace_time_zone(None).alias('hour_24_start_dttm_no_tz')
    ]).with_columns([
        # Calculate hours since start of 24-hour window for this hospitalization
        ((pl.col('event_time_no_tz') - pl.col('hour_24_start_dttm_no_tz')).dt.total_seconds() / 3600).alias('hours_from_start'),
    ]).with_columns([
        # Create hour bucket (0-23)
        pl.col('hours_from_start').floor().cast(pl.Int32).alias('hour_bucket')
    ]).filter(
        # Keep only events within the 24-hour window
        (pl.col('hour_bucket') >= 0) & (pl.col('hour_bucket') <= 23)
    )

    print(f"✅ Added hour buckets: {df_with_hours.shape}")
    print(f"Hour buckets range: {df_with_hours['hour_bucket'].min()} to {df_with_hours['hour_bucket'].max()}")
    print(f"Records within 24 hours: {len(df_with_hours):,}")
    return (df_with_hours,)


@app.cell
def _(mo):
    mo.md(r"""## Categorize Features""")
    return


@app.cell
def _(df_with_hours, pl):
    # Identify numeric columns for analysis
    exclude_columns = [
        'hospitalization_id', 'event_time', 'hour_24_start_dttm', 
        'hour_24_end_dttm', 'disposition', 'hours_from_start', 'hour_bucket',
        'sex_category', 'ethnicity_category', 'race_category', 'language_category'
    ]

    # Get only truly numeric columns (exclude strings, categoricals, etc.)
    numeric_columns = [
        col for col in df_with_hours.columns 
        if col not in exclude_columns and df_with_hours[col].dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
    ]

    # Track categorical respiratory columns separately
    categorical_columns = []
    respiratory_categorical = ['mode_category', 'device_category']
    for col in respiratory_categorical:
        if col in df_with_hours.columns and col not in numeric_columns:
            categorical_columns.append(col)
            print(f"Added categorical respiratory column: {col}")

    # Combine for coverage analysis
    all_analysis_columns = numeric_columns + categorical_columns

    # Define feature categories
    feature_categories = {
        'vitals': [],
        'labs': [],
        'medications': [],
        'respiratory': [],
        'other': []
    }

    # Keywords for categorization
    vitals_keywords = ['heart_rate', 'map', 'respiratory_rate', 'spo2', 'temp_c', 'weight', 'height']
    labs_keywords = ['albumin', 'alkaline', 'alt', 'ast', 'bicarbonate', 'bilirubin', 'bun', 
                     'calcium', 'chloride', 'creatinine', 'glucose', 'hemoglobin', 'lactate',
                     'magnesium', 'platelet', 'potassium', 'sodium', 'troponin', 'wbc', 'ph',
                     'pco2', 'po2', 'so2', 'lymphocytes', 'neutrophils', 'eosinophils',
                     'basophils', 'monocytes', 'inr', 'pt', 'ptt', 'ferritin', 'ldh',
                     'procalcitonin', 'crp', 'esr', 'phosphate', 'total_protein']
    meds_keywords = ['norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin', 
                     'dopamine', 'dobutamine', 'propofol', 'fentanyl', 'midazolam',
                     'angiotensin', 'milrinone', 'isoproterenol', 'dexmedetomidine',
                     'ketamine', 'hydromorphone', 'morphine', 'remifentanil', 
                     'pentobarbital', 'lorazepam']
    resp_keywords = ['mode_category', 'device_category', 'fio2_set']

    # Categorize columns
    for column_name in all_analysis_columns:
        col_lower = column_name.lower()
        if any(keyword in col_lower for keyword in vitals_keywords):
            feature_categories['vitals'].append(column_name)
        elif any(keyword in col_lower for keyword in labs_keywords):
            feature_categories['labs'].append(column_name)
        elif any(keyword in col_lower for keyword in meds_keywords):
            feature_categories['medications'].append(column_name)
        elif any(keyword in col_lower for keyword in resp_keywords):
            feature_categories['respiratory'].append(column_name)
        else:
            feature_categories['other'].append(column_name)

    # Print summary
    print(f"Total numeric features: {len(numeric_columns)}")
    print(f"Total categorical features: {len(categorical_columns)}")
    print(f"Total features for analysis: {len(all_analysis_columns)}")
    for cat_name, feat_list in feature_categories.items():
        if feat_list:
            print(f"  {cat_name}: {len(feat_list)} features")
    return feature_categories, numeric_columns


@app.cell
def _(mo):
    mo.md(r"""## Calculate Coverage Statistics""")
    return


@app.cell
def _(df_with_hours, feature_categories, pl):
    # Calculate coverage for each feature by hour
    # For each hospitalization-hour-feature combination: does at least one non-null value exist?

    coverage_results = []

    for cat_key, features in feature_categories.items():
        if not features:
            continue

        print(f"Processing {cat_key} ({len(features)} features)...")

        # For each feature in this category
        for feature in features:
            if feature not in df_with_hours.columns:
                continue

            # Group by hospitalization and hour, check if feature has any non-null values
            hosp_hour_coverage = df_with_hours.group_by(['hospitalization_id', 'hour_bucket']).agg([
                pl.col(feature).is_not_null().any().alias('has_data')
            ])

            # For each hour, calculate what % of hospitalizations have data
            hour_coverage = hosp_hour_coverage.group_by('hour_bucket').agg([
                pl.col('has_data').sum().alias('hospitalizations_with_data'),
                pl.len().alias('total_hospitalizations')
            ]).with_columns([
                (pl.col('hospitalizations_with_data') / pl.col('total_hospitalizations') * 100).alias('coverage_pct'),
                pl.lit(feature).alias('feature'),
                pl.lit(cat_key).alias('category')
            ]).select(['hour_bucket', 'feature', 'category', 'coverage_pct', 'hospitalizations_with_data', 'total_hospitalizations'])

            coverage_results.append(hour_coverage)

    # Combine all coverage results
    all_coverage = pl.concat(coverage_results)

    print(f"✅ Calculated coverage for {all_coverage['feature'].n_unique()} features across 24 hours")
    print(f"Total coverage records: {len(all_coverage):,}")
    return (all_coverage,)


@app.cell
def _(mo):
    mo.md(r"""## Create Heatmaps by Feature Category""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Vitals""")
    return


@app.cell
def _(all_coverage, alt, graphs_path, os, pl):
    # Create heatmap for vitals
    vitals_data = all_coverage.filter(pl.col('category') == 'vitals').to_pandas()

    # Ensure hour_bucket is sorted
    if len(vitals_data) > 0:
        vitals_data['hour_bucket'] = vitals_data['hour_bucket'].astype(int)
        vitals_data = vitals_data.sort_values('hour_bucket')

    # Sort features by average coverage
    vitals_order = vitals_data.groupby('feature')['coverage_pct'].mean().sort_values().index.tolist() if len(vitals_data) > 0 else []

    # Create base chart
    vitals_base = alt.Chart(vitals_data).encode(
        x=alt.X('hour_bucket:O', 
                title='Hour from ICU Admission',
                sort=list(range(24))),
        y=alt.Y('feature:N', 
                title='Vital Signs',
                sort=vitals_order)
    )

    # Heatmap layer
    vitals_heatmap = vitals_base.mark_rect().encode(
        color=alt.Color('coverage_pct:Q',
                       title='% Coverage',
                       scale=alt.Scale(scheme='viridis', domain=[0, 100])),
        tooltip=[
            alt.Tooltip('hour_bucket:O', title='Hour'),
            alt.Tooltip('feature:N', title='Feature'),
            alt.Tooltip('coverage_pct:Q', title='% Coverage', format='.1f'),
            alt.Tooltip('hospitalizations_with_data:Q', title='N with Data'),
            alt.Tooltip('total_hospitalizations:Q', title='Total N')
        ]
    )

    # Text annotation layer
    vitals_text = vitals_base.mark_text(baseline='middle', fontSize=10, fontWeight='bold').encode(
        text=alt.Text('coverage_pct:Q', format='.0f'),
        color=alt.condition(
            alt.datum.coverage_pct > 50,
            alt.value('white'),
            alt.value('black')
        )
    )

    # Combine layers
    vitals_chart = (vitals_heatmap + vitals_text).properties(
        width=1200,
        height=max(200, len(vitals_order) * 30) if len(vitals_order) > 0 else 200,
        title='Vitals: % of Hospitalizations with Data by Hour'
    ).configure_view(strokeWidth=0).configure_axis(domain=False, labelFontSize=12, titleFontSize=14)

    # Save chart
    if len(vitals_data) > 0:
        vitals_chart.save(os.path.join(graphs_path, 'vitals_hourly_coverage.html'))

    vitals_chart
    return


@app.cell
def _(mo):
    mo.md(r"""### Labs""")
    return


@app.cell
def _(all_coverage, alt, graphs_path, os, pl):
    # Create heatmap for labs
    labs_data = all_coverage.filter(pl.col('category') == 'labs').to_pandas()

    # Ensure hour_bucket is sorted
    if len(labs_data) > 0:
        labs_data['hour_bucket'] = labs_data['hour_bucket'].astype(int)
        labs_data = labs_data.sort_values('hour_bucket')

    # Sort features by average coverage
    labs_order = labs_data.groupby('feature')['coverage_pct'].mean().sort_values().index.tolist() if len(labs_data) > 0 else []

    # Create base chart
    labs_base = alt.Chart(labs_data).encode(
        x=alt.X('hour_bucket:O', 
                title='Hour from ICU Admission',
                sort=list(range(24))),
        y=alt.Y('feature:N', 
                title='Laboratory Values',
                sort=labs_order)
    )

    # Heatmap layer
    labs_heatmap = labs_base.mark_rect().encode(
        color=alt.Color('coverage_pct:Q',
                       title='% Coverage',
                       scale=alt.Scale(scheme='viridis', domain=[0, 100])),
        tooltip=[
            alt.Tooltip('hour_bucket:O', title='Hour'),
            alt.Tooltip('feature:N', title='Feature'),
            alt.Tooltip('coverage_pct:Q', title='% Coverage', format='.1f'),
            alt.Tooltip('hospitalizations_with_data:Q', title='N with Data'),
            alt.Tooltip('total_hospitalizations:Q', title='Total N')
        ]
    )

    # Text annotation layer
    labs_text = labs_base.mark_text(baseline='middle', fontSize=10, fontWeight='bold').encode(
        text=alt.Text('coverage_pct:Q', format='.0f'),
        color=alt.condition(
            alt.datum.coverage_pct > 50,
            alt.value('white'),
            alt.value('black')
        )
    )

    # Combine layers
    labs_chart = (labs_heatmap + labs_text).properties(
        width=1200,
        height=max(400, len(labs_order) * 20) if len(labs_order) > 0 else 400,
        title='Labs: % of Hospitalizations with Data by Hour'
    ).configure_view(strokeWidth=0).configure_axis(domain=False, labelFontSize=12, titleFontSize=14)

    # Save chart
    if len(labs_data) > 0:
        labs_chart.save(os.path.join(graphs_path, 'labs_hourly_coverage.html'))

    labs_chart
    return


@app.cell
def _(mo):
    mo.md(r"""### Medications""")
    return


@app.cell
def _(all_coverage, alt, graphs_path, os, pl):
    # Create heatmap for medications
    meds_data = all_coverage.filter(pl.col('category') == 'medications').to_pandas()

    # Ensure hour_bucket is sorted
    if len(meds_data) > 0:
        meds_data['hour_bucket'] = meds_data['hour_bucket'].astype(int)
        meds_data = meds_data.sort_values('hour_bucket')

    # Sort features by average coverage
    meds_order = meds_data.groupby('feature')['coverage_pct'].mean().sort_values().index.tolist() if len(meds_data) > 0 else []

    # Create base chart
    meds_base = alt.Chart(meds_data).encode(
        x=alt.X('hour_bucket:O', 
                title='Hour from ICU Admission',
                sort=list(range(24))),
        y=alt.Y('feature:N', 
                title='Medications',
                sort=meds_order)
    )

    # Heatmap layer
    meds_heatmap = meds_base.mark_rect().encode(
        color=alt.Color('coverage_pct:Q',
                       title='% Coverage',
                       scale=alt.Scale(scheme='viridis', domain=[0, 100])),
        tooltip=[
            alt.Tooltip('hour_bucket:O', title='Hour'),
            alt.Tooltip('feature:N', title='Feature'),
            alt.Tooltip('coverage_pct:Q', title='% Coverage', format='.1f'),
            alt.Tooltip('hospitalizations_with_data:Q', title='N with Data'),
            alt.Tooltip('total_hospitalizations:Q', title='Total N')
        ]
    )

    # Text annotation layer
    meds_text = meds_base.mark_text(baseline='middle', fontSize=10, fontWeight='bold').encode(
        text=alt.Text('coverage_pct:Q', format='.0f'),
        color=alt.condition(
            alt.datum.coverage_pct > 50,
            alt.value('white'),
            alt.value('black')
        )
    )

    # Combine layers
    meds_chart = (meds_heatmap + meds_text).properties(
        width=1200,
        height=max(300, len(meds_order) * 25) if len(meds_order) > 0 else 300,
        title='Medications: % of Hospitalizations with Data by Hour'
    ).configure_view(strokeWidth=0).configure_axis(domain=False, labelFontSize=12, titleFontSize=14)

    # Save chart
    if len(meds_data) > 0:
        meds_chart.save(os.path.join(graphs_path, 'medications_hourly_coverage.html'))

    meds_chart
    return


@app.cell
def _(mo):
    mo.md(r"""### Respiratory""")
    return


@app.cell
def _(all_coverage, alt, graphs_path, os, pl):
    # Create heatmap for respiratory
    resp_data = all_coverage.filter(pl.col('category') == 'respiratory').to_pandas()

    # Ensure hour_bucket is sorted
    if len(resp_data) > 0:
        resp_data['hour_bucket'] = resp_data['hour_bucket'].astype(int)
        resp_data = resp_data.sort_values('hour_bucket')

    # Sort features by average coverage
    resp_order = resp_data.groupby('feature')['coverage_pct'].mean().sort_values().index.tolist() if len(resp_data) > 0 else []

    # Create base chart
    resp_base = alt.Chart(resp_data).encode(
        x=alt.X('hour_bucket:O', 
                title='Hour from ICU Admission',
                sort=list(range(24))),
        y=alt.Y('feature:N', 
                title='Respiratory Features',
                sort=resp_order)
    )

    # Heatmap layer
    resp_heatmap = resp_base.mark_rect().encode(
        color=alt.Color('coverage_pct:Q',
                       title='% Coverage',
                       scale=alt.Scale(scheme='viridis', domain=[0, 100])),
        tooltip=[
            alt.Tooltip('hour_bucket:O', title='Hour'),
            alt.Tooltip('feature:N', title='Feature'),
            alt.Tooltip('coverage_pct:Q', title='% Coverage', format='.1f'),
            alt.Tooltip('hospitalizations_with_data:Q', title='N with Data'),
            alt.Tooltip('total_hospitalizations:Q', title='Total N')
        ]
    )

    # Text annotation layer
    resp_text = resp_base.mark_text(baseline='middle', fontSize=10, fontWeight='bold').encode(
        text=alt.Text('coverage_pct:Q', format='.0f'),
        color=alt.condition(
            alt.datum.coverage_pct > 50,
            alt.value('white'),
            alt.value('black')
        )
    )

    # Combine layers
    resp_chart = (resp_heatmap + resp_text).properties(
        width=1200,
        height=max(200, len(resp_order) * 35) if len(resp_order) > 0 else 200,
        title='Respiratory: % of Hospitalizations with Data by Hour'
    ).configure_view(strokeWidth=0).configure_axis(domain=False, labelFontSize=12, titleFontSize=14)

    # Save chart
    if len(resp_data) > 0:
        resp_chart.save(os.path.join(graphs_path, 'respiratory_hourly_coverage.html'))

    resp_chart
    return


@app.cell
def _(mo):
    mo.md(r"""### Other Features""")
    return


@app.cell
def _(all_coverage, alt, graphs_path, os, pl):
    # Create heatmap for other features
    other_data = all_coverage.filter(pl.col('category') == 'other').to_pandas()

    # Ensure hour_bucket is sorted
    if len(other_data) > 0:
        other_data['hour_bucket'] = other_data['hour_bucket'].astype(int)
        other_data = other_data.sort_values('hour_bucket')

    # Sort features by average coverage
    other_order = other_data.groupby('feature')['coverage_pct'].mean().sort_values().index.tolist() if len(other_data) > 0 else []

    # Create base chart
    other_base = alt.Chart(other_data).encode(
        x=alt.X('hour_bucket:O', 
                title='Hour from ICU Admission',
                sort=list(range(24))),
        y=alt.Y('feature:N', 
                title='Other Features',
                sort=other_order)
    )

    # Heatmap layer
    other_heatmap = other_base.mark_rect().encode(
        color=alt.Color('coverage_pct:Q',
                       title='% Coverage',
                       scale=alt.Scale(scheme='viridis', domain=[0, 100])),
        tooltip=[
            alt.Tooltip('hour_bucket:O', title='Hour'),
            alt.Tooltip('feature:N', title='Feature'),
            alt.Tooltip('coverage_pct:Q', title='% Coverage', format='.1f'),
            alt.Tooltip('hospitalizations_with_data:Q', title='N with Data'),
            alt.Tooltip('total_hospitalizations:Q', title='Total N')
        ]
    )

    # Text annotation layer
    other_text = other_base.mark_text(baseline='middle', fontSize=10, fontWeight='bold').encode(
        text=alt.Text('coverage_pct:Q', format='.0f'),
        color=alt.condition(
            alt.datum.coverage_pct > 50,
            alt.value('white'),
            alt.value('black')
        )
    )

    # Combine layers
    other_chart = (other_heatmap + other_text).properties(
        width=1200,
        height=max(300, len(other_order) * 20) if len(other_order) > 0 else 300,
        title='Other Features: % of Hospitalizations with Data by Hour'
    ).configure_view(strokeWidth=0).configure_axis(domain=False, labelFontSize=12, titleFontSize=14)

    # Save chart
    if len(other_data) > 0:
        other_chart.save(os.path.join(graphs_path, 'other_features_hourly_coverage.html'))

    other_chart
    return


@app.cell
def _(mo):
    mo.md(r"""## Generate Table 1 Summary Statistics""")
    return


@app.cell
def _(df_with_hours, numeric_columns, pl):
    # Calculate comprehensive statistics for each feature
    table_one_stats = []

    for feat_name in numeric_columns:
        if feat_name not in df_with_hours.columns:
            continue

        # Skip categorical columns for numeric statistics
        if df_with_hours[feat_name].dtype == pl.Utf8:
            print(f"Skipping categorical column in Table 1: {feat_name}")
            continue

        # Calculate statistics using Polars
        feature_stats = df_with_hours.select([
            pl.lit(feat_name).alias('feature'),
            (pl.col(feat_name).is_null().sum() / pl.len() * 100).alias('missing_pct'),
            pl.col(feat_name).min().cast(pl.Float64).alias('min_value'),
            pl.col(feat_name).max().cast(pl.Float64).alias('max_value'),
            pl.col(feat_name).median().cast(pl.Float64).alias('median'),
            pl.col(feat_name).quantile(0.25).cast(pl.Float64).alias('q1'),
            pl.col(feat_name).quantile(0.75).cast(pl.Float64).alias('q3'),
            pl.col(feat_name).is_not_null().sum().cast(pl.Int64).alias('n_observations')
        ])

        table_one_stats.append(feature_stats)

    # Combine all statistics
    table_one = pl.concat(table_one_stats)

    # Round numeric columns for readability
    table_one_rounded = table_one.with_columns([
        pl.col('missing_pct').round(1),
        pl.col('min_value').round(3),
        pl.col('max_value').round(3),
        pl.col('median').round(3),
        pl.col('q1').round(3),
        pl.col('q3').round(3)
    ])

    print(f"✅ Generated Table 1 statistics for {len(table_one_rounded)} features")
    print("\nSample statistics:")
    print(table_one_rounded.head(5).to_pandas())
    return (table_one_rounded,)


@app.cell
def _(mo):
    mo.md(r"""## Save Results""")
    return


@app.cell
def _(all_coverage, os, qc_path, table_one_rounded):
    # Save all results to CSV files
    try:
        # Save coverage data
        coverage_path = os.path.join(qc_path, 'hourly_coverage_by_feature.csv')
        all_coverage.to_pandas().to_csv(coverage_path, index=False)
        print(f"✅ Saved coverage data: {coverage_path}")

        # Save Table 1 statistics
        table_one_path = os.path.join(qc_path, 'table_one.csv')
        table_one_rounded.to_pandas().to_csv(table_one_path, index=False)
        print(f"✅ Saved Table 1 statistics: {table_one_path}")

        # Display file sizes
        coverage_size = os.path.getsize(coverage_path) / 1024
        table_size = os.path.getsize(table_one_path) / 1024

        print(f"\nFile sizes:")
        print(f"  Coverage data: {coverage_size:.1f} KB")
        print(f"  Table 1: {table_size:.1f} KB")

    except Exception as e:
        print(f"❌ Error saving files: {str(e)}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    This notebook has successfully completed hourly quality control analysis:

    1. ✅ **Data Processing**: Loaded event-wide dataset and bucketed into 24-hour bins per hospitalization
    2. ✅ **Feature Categorization**: Grouped features into vitals, labs, medications, respiratory, and other categories  
    3. ✅ **Coverage Analysis**: Calculated % of hospitalizations with data for each feature by hour
    4. ✅ **Visualizations**: Created heatmaps for each feature category using Altair
    5. ✅ **Table 1**: Generated comprehensive statistics (missing%, min, max, median, Q1, Q3)
    6. ✅ **Output**: Saved results to share_to_box/qc folder

    ### Key Insights:
    - **Hour-level bucketing**: Each hospitalization's events are bucketed into 0-23 hours from their ICU admission
    - **Coverage patterns**: Visualizations show temporal patterns in data availability
    - **Feature quality**: Table 1 provides comprehensive quality metrics for model development

    ### Output Files:
    - `share_to_box/qc/hourly_coverage_by_feature.csv`: Coverage statistics by hour and feature
    - `share_to_box/qc/table_one.csv`: Summary statistics for all features
    - `share_to_box/qc/graphs/*.html`: Interactive heatmaps for each feature category
    """
    )
    return


if __name__ == "__main__":
    app.run()
