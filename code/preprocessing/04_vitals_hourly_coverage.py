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
    # ICU Mortality Model - Hourly Coverage Analysis for All Features

    This notebook analyzes the coverage of all features across 24-hour blocks for ICU patients.

    ## Objective
    - Load event-wide dataset from 02_feature_engineering.ipynb
    - Categorize features into conceptual groups (vitals, labs, medications, respiratory, other)
    - Calculate missing data percentage for each feature by hour (0-23)
    - Create heatmaps showing missing data patterns for each feature category
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
    import altair as alt
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings('ignore')

    # Enable marimo CSV data transformer
    try:
        alt.data_transformers.enable('marimo_csv')
    except:
        alt.data_transformers.enable('default')

    print("=== Feature Hourly Coverage Analysis ===")
    return alt, np, os, pd, tqdm


@app.cell
def _(os):
    # Set up paths
    cwd = os.getcwd()
    if cwd.endswith(('code/preprocessing', 'code\\preprocessing')):
        data_path = os.path.join('..', '..', 'protected_outputs', 'preprocessing')
    else:
        data_path = os.path.join('protected_outputs', 'preprocessing')

    data_path = os.path.abspath(data_path)
    print(f"Data path: {data_path}")

    return (data_path,)


@app.cell
def _(data_path, os, pd):
    # Load event-wide dataset
    event_wide_path = os.path.join(data_path, 'by_event_wide_df.parquet')

    if not os.path.exists(event_wide_path):
        raise FileNotFoundError(f"Event-wide dataset not found at {event_wide_path}")

    event_wide_df = pd.read_parquet(event_wide_path)
    print(f"✅ Loaded dataset: {event_wide_df.shape}")
    print(f"Hospitalizations: {event_wide_df['hospitalization_id'].nunique()}")

    return (event_wide_df,)


@app.cell
def _(event_wide_df, np):
    # Categorize all numeric features
    exclude_columns = [
        'hospitalization_id', 'event_time', 'hour_24_start_dttm', 
        'hour_24_end_dttm', 'disposition'
    ]

    # Get numeric columns
    numeric_columns = event_wide_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

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
    resp_keywords = ['mode_category', 'device_category', 'fio2', 'peep']

    # Categorize columns
    for col in numeric_columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in vitals_keywords):
            feature_categories['vitals'].append(col)
        elif any(keyword in col_lower for keyword in labs_keywords):
            feature_categories['labs'].append(col)
        elif any(keyword in col_lower for keyword in meds_keywords):
            feature_categories['medications'].append(col)
        elif any(keyword in col_lower for keyword in resp_keywords):
            feature_categories['respiratory'].append(col)
        else:
            feature_categories['other'].append(col)

    # Print summary
    print(f"Total features: {len(numeric_columns)}")
    for category, feature_list in feature_categories.items():
        if feature_list:
            print(f"  {category}: {len(feature_list)} features")

    return (feature_categories,)


@app.cell
def _(event_wide_df, pd):
    # Calculate hour blocks
    df_with_hour = event_wide_df.copy()

    # Ensure datetime columns
    datetime_cols = ['event_time', 'hour_24_start_dttm']
    for _col in datetime_cols:
        if _col in df_with_hour.columns:
            df_with_hour[_col] = pd.to_datetime(df_with_hour[_col])

    # Calculate hours from admission
    df_with_hour['hours_from_admission'] = (
        df_with_hour['event_time'] - df_with_hour['hour_24_start_dttm']
    ).dt.total_seconds() / 3600

    df_with_hour['hour_block'] = df_with_hour['hours_from_admission'].astype(int)

    # Filter to first 24 hours
    df_24h = df_with_hour[
        (df_with_hour['hour_block'] >= 0) & 
        (df_with_hour['hour_block'] <= 23)
    ].copy()

    print(f"Filtered to 24 hours: {len(df_24h):,} records")
    return (df_24h,)


@app.cell
def _(df_24h, feature_categories, pd, tqdm):
    # Calculate missing percentage for all features by hour
    # Check if at least one value exists per hospitalization per hour
    all_coverage_data = {}
    
    # Calculate total number of feature-hour combinations for progress bar
    total_features = sum(len(feat_list) for feat_list in feature_categories.values())
    
    print(f"Processing {total_features} features across 24 hours...")

    for cat, feat_list in feature_categories.items():
        if not feat_list:
            continue

        coverage_data = []
        
        # Create progress bar for this category
        cat_desc = f"Processing {cat} ({len(feat_list)} features)"
        
        # Process each hour with a progress bar
        for hour in tqdm(sorted(range(24)), desc=cat_desc, leave=False):
            hour_data = df_24h[df_24h['hour_block'] == hour]
            
            # Get unique hospitalizations in this hour
            unique_hospitalizations = hour_data['hospitalization_id'].unique()
            total_hospitalizations = len(unique_hospitalizations)

            for feature in feat_list:
                if feature in hour_data.columns:
                    # Group by hospitalization and check if ANY value exists
                    hosp_has_data = hour_data.groupby('hospitalization_id')[feature].apply(
                        lambda x: x.notna().any()
                    )
                    
                    # Count hospitalizations with no data for this feature in this hour
                    hospitalizations_with_data = hosp_has_data.sum()
                    hospitalizations_missing = total_hospitalizations - hospitalizations_with_data
                    
                    # Calculate percentage of hospitalizations missing data
                    missing_pct = (hospitalizations_missing / total_hospitalizations * 100) if total_hospitalizations > 0 else 100

                    coverage_data.append({
                        'hour': hour,
                        'feature': feature,
                        'missing_pct': missing_pct,
                        'hospitalizations_missing': hospitalizations_missing,
                        'total_hospitalizations': total_hospitalizations
                    })

        if coverage_data:
            coverage_df = pd.DataFrame(coverage_data)
            # Calculate average missing percentage for sorting
            avg_missing = coverage_df.groupby('feature')['missing_pct'].mean().to_dict()
            coverage_df['avg_missing'] = coverage_df['feature'].map(avg_missing)
            all_coverage_data[cat] = coverage_df

    print("\n✅ Calculated coverage for all features by category (per hospitalization)")
    for cat, df in all_coverage_data.items():
        print(f"  {cat}: {df['feature'].nunique()} features")
        avg_missing_overall = df.groupby('feature')['missing_pct'].mean().mean()
        print(f"    Average % hospitalizations missing data: {avg_missing_overall:.1f}%")

    return (all_coverage_data,)


@app.cell
def _(mo):
    mo.md("""## Missing Data Heatmaps by Category""")
    return


@app.cell
def _(mo):
    mo.md("""### Vitals""")
    return


@app.cell
def _(all_coverage_data, alt):
    # Create heatmap for vitals
    vitals_df = all_coverage_data['vitals']
    # Sort by hour to ensure proper ordering
    vitals_df = vitals_df.sort_values('hour')
    # Sort features by average missing percentage
    vitals_order = vitals_df.groupby('feature')['missing_pct'].mean().sort_values().index.tolist()

    alt.Chart(vitals_df).mark_rect().encode(
        x=alt.X('hour:O', 
                title='Hour from ICU Admission',
                axis=alt.Axis(labelAngle=0),
                sort=list(range(24))),
        y=alt.Y('feature:N', 
                title='Vital Sign',
                sort=vitals_order),
        color=alt.Color('missing_pct:Q',
                       title='% Hospitalizations Missing',
                       scale=alt.Scale(scheme='reds', domain=[0, 100])),
        tooltip=[
            alt.Tooltip('hour:O', title='Hour'),
            alt.Tooltip('feature:N', title='Feature'),
            alt.Tooltip('missing_pct:Q', title='% Hospitalizations Missing', format='.1f'),
            alt.Tooltip('hospitalizations_missing:Q', title='Hospitalizations Missing'),
            alt.Tooltip('total_hospitalizations:Q', title='Total Hospitalizations')
        ]
    ).properties(
        width=800,
        height=max(150, len(vitals_order) * 20),
        title='Vitals: % of Hospitalizations with No Data by Hour'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domain=False
    )
    return


@app.cell
def _(mo):
    mo.md("""### Labs""")
    return


@app.cell
def _(all_coverage_data, alt):
    # Create heatmap for labs
    labs_df = all_coverage_data['labs']
    # Sort by hour to ensure proper ordering
    labs_df = labs_df.sort_values('hour')
    # Sort features by average missing percentage
    labs_order = labs_df.groupby('feature')['missing_pct'].mean().sort_values().index.tolist()

    alt.Chart(labs_df).mark_rect().encode(
        x=alt.X('hour:O', 
                title='Hour from ICU Admission',
                axis=alt.Axis(labelAngle=0),
                sort=list(range(24))),
        y=alt.Y('feature:N', 
                title='Lab',
                sort=labs_order),
            color=alt.Color('missing_pct:Q',
                           title='% Hospitalizations Missing',
                           scale=alt.Scale(scheme='reds', domain=[0, 100])),
            tooltip=[
                alt.Tooltip('hour:O', title='Hour'),
                alt.Tooltip('feature:N', title='Feature'),
                alt.Tooltip('missing_pct:Q', title='% Hospitalizations Missing', format='.1f'),
                alt.Tooltip('hospitalizations_missing:Q', title='Hospitalizations Missing'),
                alt.Tooltip('total_hospitalizations:Q', title='Total Hospitalizations')
        ]
    ).properties(
        width=800,
        height=max(300, len(labs_order) * 15),
        title='Labs: % of Hospitalizations with No Data by Hour'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domain=False
    )
    return


@app.cell
def _(all_coverage_data, alt):
    # Create heatmap for medications
    meds_df = all_coverage_data['medications']
    # Sort by hour to ensure proper ordering
    meds_df = meds_df.sort_values('hour')
    # Sort features by average missing percentage
    meds_order = meds_df.groupby('feature')['missing_pct'].mean().sort_values().index.tolist()

    alt.Chart(meds_df).mark_rect().encode(
        x=alt.X('hour:O', 
                title='Hour from ICU Admission',
                axis=alt.Axis(labelAngle=0),
                sort=list(range(24))),
        y=alt.Y('feature:N', 
                title='Medication',
                sort=meds_order),
            color=alt.Color('missing_pct:Q',
                           title='% Hospitalizations Missing',
                           scale=alt.Scale(scheme='reds', domain=[0, 100])),
            tooltip=[
                alt.Tooltip('hour:O', title='Hour'),
                alt.Tooltip('feature:N', title='Feature'),
                alt.Tooltip('missing_pct:Q', title='% Hospitalizations Missing', format='.1f'),
                alt.Tooltip('hospitalizations_missing:Q', title='Hospitalizations Missing'),
                alt.Tooltip('total_hospitalizations:Q', title='Total Hospitalizations')
        ]
    ).properties(
        width=800,
        height=max(200, len(meds_order) * 20),
        title='Medications: % of Hospitalizations with No Data by Hour'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domain=False
    )
    return


@app.cell
def _(mo):
    mo.md("""### Respiratory""")
    return


@app.cell
def _(all_coverage_data, alt):
    # Create heatmap for respiratory
    resp_df = all_coverage_data['respiratory']
    # Sort by hour to ensure proper ordering
    resp_df = resp_df.sort_values('hour')
    # Sort features by average missing percentage
    resp_order = resp_df.groupby('feature')['missing_pct'].mean().sort_values().index.tolist()

    alt.Chart(resp_df).mark_rect().encode(
        x=alt.X('hour:O', 
                title='Hour from ICU Admission',
                axis=alt.Axis(labelAngle=0),
                sort=list(range(24))),
        y=alt.Y('feature:N', 
                title='Respiratory Feature',
                sort=resp_order),
            color=alt.Color('missing_pct:Q',
                           title='% Hospitalizations Missing',
                           scale=alt.Scale(scheme='reds', domain=[0, 100])),
            tooltip=[
                alt.Tooltip('hour:O', title='Hour'),
                alt.Tooltip('feature:N', title='Feature'),
                alt.Tooltip('missing_pct:Q', title='% Hospitalizations Missing', format='.1f'),
                alt.Tooltip('hospitalizations_missing:Q', title='Hospitalizations Missing'),
                alt.Tooltip('total_hospitalizations:Q', title='Total Hospitalizations')
        ]
    ).properties(
        width=800,
        height=max(100, len(resp_order) * 25),
        title='Respiratory: % of Hospitalizations with No Data by Hour'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domain=False
    )
    return


@app.cell
def _(mo):
    mo.md("""### Other Features""")
    return


@app.cell
def _(all_coverage_data, alt):
    # Create heatmap for other features
    other_df = all_coverage_data['other']
    # Sort by hour to ensure proper ordering
    other_df = other_df.sort_values('hour')
    # Sort features by average missing percentage
    other_order = other_df.groupby('feature')['missing_pct'].mean().sort_values().index.tolist()

    alt.Chart(other_df).mark_rect().encode(
        x=alt.X('hour:O', 
                title='Hour from ICU Admission',
                axis=alt.Axis(labelAngle=0),
                sort=list(range(24))),
        y=alt.Y('feature:N', 
                title='Other Features',
                sort=other_order),
            color=alt.Color('missing_pct:Q',
                           title='% Hospitalizations Missing',
                           scale=alt.Scale(scheme='reds', domain=[0, 100])),
            tooltip=[
                alt.Tooltip('hour:O', title='Hour'),
                alt.Tooltip('feature:N', title='Feature'),
                alt.Tooltip('missing_pct:Q', title='% Hospitalizations Missing', format='.1f'),
                alt.Tooltip('hospitalizations_missing:Q', title='Hospitalizations Missing'),
                alt.Tooltip('total_hospitalizations:Q', title='Total Hospitalizations')
        ]
    ).properties(
        width=800,
        height=max(200, len(other_order) * 15),
        title='Other Features: % of Hospitalizations with No Data by Hour'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domain=False
    )
    return


@app.cell
def _(all_coverage_data, data_path, os, pd):
    # Save results
    output_path = os.path.join(data_path, 'all_features_hourly_coverage.parquet')

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Combine all dataframes
        combined_df = pd.concat([df.assign(category=cat_name) for cat_name, df in all_coverage_data.items()], ignore_index=True)
        combined_df.to_parquet(output_path, index=False)
        print(f"✅ Saved coverage analysis to: {output_path}")
    except Exception as e:
        print(f"Could not save file: {e}")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    This notebook has analyzed all features coverage across the first 24 hours of ICU stay:

    - Categorized features into conceptual groups (vitals, labs, medications, respiratory, other)
    - Created heatmaps showing the **percentage of hospitalizations with no data** for each feature by hour
    - For each hospitalization-hour combination, checked if at least one value exists for each feature
    - Sorted features by average percentage of hospitalizations missing data within each category

    The visualizations help identify temporal patterns in data collection and can inform:
    - Data quality assessment (which features are commonly missing across hospitalizations)
    - Feature engineering decisions (features with consistent availability)
    - Imputation strategies (understanding missingness patterns at the hospitalization level)
    """
    )
    return


if __name__ == "__main__":
    app.run()
